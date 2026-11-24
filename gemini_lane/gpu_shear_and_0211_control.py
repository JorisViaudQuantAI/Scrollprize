# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T16-GPU: Grid Shear Stress Harness (PHerc. 0306B) & PHerc. 0211 Winner Validation.

Executes two strictly VRAM-capped (`--vram-fraction 0.35`, `--studio-batch 1`),
thermally guarded (`--max-gpu-temp-c 74`, `--batch-pause-ms 350`), zero-disk-bloat
RTX 4060 GPU controls using single RAM-streamed surface-volume stacks:

1. Arm A (`shear` — Public `PHerc. 0306B w074`, `uv_crop=95:120,190:250`, `8.64 um`):
   * Addresses `collab/claude-023` section 1 (*"can your stretch harness take a shear
     arm (15 deg and 30 deg, both signs) on the same public tiles? That is the missing
     number for section 1."*).
   * Applies area-preserving in-plane affine grid shear (`x' = x + tan(theta) * (y - cy)`,
     bilinear `order=1` on expanded unclipped canvas with exact positive-weight valid
     masking) at angles `0, +15, -15, +30, -30` degrees, runs `Studio fp16 stride=48`
     (`desc` and `asc`), inverts outputs back to canonical coordinates, and measures pixel
     agreement (`IoU@0.5`, `IoU@0.7`, `hit_retention@0.7`, `pearson_r`, `Q34` TFCE) both
     on all common valid pixels and on a 32 px boundary-eroded interior mask.

2. Arm B (`pherc0211` — Public `PHerc. 0211` First Letters Winner Mesh):
   * Addresses `collab/claude-020` item 2 & `collab/claude-022`: streams the 3-row letter
     crop (`uv_crop=12:76,184:214`, `1261 x 581` px, `0.642 cm^2`, 100% valid mesh) of
     `PHerc0211/segments/20261028201525-crop1711-lines123-v9/mesh/20261028201525-on-20250821151803-9.362um.tifxyz`
     from the public S3 CT Zarr through `screen_scroll_rtx4060.py` (`Studio fp16 stride=48`
     in both `desc` and `asc` reading orders, optional A-hn) + `Q34` opposite-face TFCE.

Strictly zero persistent CT on disk; private visual montages go strictly outside git
(`ensure_outside_git_repo`) while a path-sanitized public JSON receipt is written to
`gemini_lane/evidence/gpu_shear_and_0211_control.json`.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpu_transverse_interventions as GTI  # noqa: E402
import screen_scroll_rtx4060 as SSRX  # noqa: E402
import stream_snap_render as SSR  # noqa: E402

SHEAR_ANGLES_DEG: tuple[float, ...] = (0.0, 15.0, -15.0, 30.0, -30.0)
INTERIOR_EROSION_PX: int = 32
DEFAULT_PUBLIC_RECEIPT = HERE / "evidence" / "gpu_shear_and_0211_control.json"

# Pinned public S3 sources for Arm B (PHerc. 0211 First Letters Winner Mesh)
PHERC0211_SEGMENT = "20261028201525-crop1711-lines123-v9"
PHERC0211_MESH_TAG = "20261028201525-on-20250821151803-9.362um.tifxyz"
PHERC0211_MESH_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    f"PHerc0211/segments/{PHERC0211_SEGMENT}/mesh/{PHERC0211_MESH_TAG}"
)
PHERC0211_CT_ZARR_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr"
)
PHERC0211_DEFAULT_UV_CROP: tuple[int, int, int, int] = (12, 76, 184, 214)
PHERC0211_VOXEL_UM: float = 9.362


def shear_variant_key(angle_deg: float) -> str:
    """Return canonical dict key for a shear angle in degrees (e.g. ``shear_p15``)."""
    val = int(round(float(angle_deg)))
    if not math.isclose(float(angle_deg), float(val), abs_tol=1e-6):
        raise ValueError(f"Expected integer shear angle in degrees, got {angle_deg!r}")
    if val == 0:
        return "shear_0"
    return f"shear_p{val}" if val > 0 else f"shear_m{abs(val)}"


def compute_shear_geometry(shape_hw: tuple[int, int], angle_deg: float) -> dict[str, Any]:
    """Compute expanded-canvas area-preserving shear parameters for `(H, W)` arrays."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h < 1 or w < 1:
        raise ValueError(f"shape_hw must have positive dimensions, got {shape_hw!r}")
    if not (-60.0 <= float(angle_deg) <= 60.0):
        raise ValueError(f"angle_deg must lie in [-60, 60], got {angle_deg!r}")
    k = math.tan(math.radians(float(angle_deg)))
    cy = 0.5 * float(h - 1)
    pad_x = 0 if abs(k) < 1e-12 else int(math.ceil(abs(k) * cy)) + 4
    out_w = w + 2 * pad_x
    return {
        "angle_deg": float(angle_deg),
        "k": k,
        "h": h,
        "w": w,
        "cy": cy,
        "pad_x": pad_x,
        "out_w": out_w,
    }


def apply_shear_transform(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply area-preserving horizontal shear `x' = x + tan(angle)*(y - cy) + pad_x`.

    Uses bilinear interpolation (`order=1`) with exact positive-weight stencil support
    (`warped_valid > 0.999`) so exterior triangle padding stays zero and never bleeds
    into ``valid_sheared``.
    """
    a = np.asarray(arr)
    v = np.asarray(valid_mask, dtype=bool)
    if a.ndim not in (2, 3) or a.shape[-2:] != v.shape:
        raise ValueError(f"Shape mismatch: arr {a.shape} vs valid_mask {v.shape}")

    geom = compute_shear_geometry(v.shape, angle_deg)
    if abs(geom["k"]) < 1e-12:
        out_arr = np.where(v if a.ndim == 2 else v[None, :, :], a, 0)
        return np.ascontiguousarray(out_arr.copy()), np.ascontiguousarray(v.copy())

    h, cy, pad_x, out_w, k = geom["h"], geom["cy"], geom["pad_x"], geom["out_w"], geom["k"]
    inv_mat = np.array([[1.0, 0.0], [-k, 1.0]], dtype=np.float64)
    inv_off = np.array([0.0, k * cy - float(pad_x)], dtype=np.float64)

    # Zero-masked float input preserves interior support; valid stencil requires all
    # contributing bilinear neighbors to be inside valid_mask.
    warped_valid = ndi.affine_transform(
        v.astype(np.float32),
        inv_mat,
        offset=inv_off,
        output_shape=(h, out_w),
        order=1,
        mode="constant",
        cval=0.0,
    )
    sheared_valid = warped_valid > 0.999

    def _warp_2d(plane: np.ndarray) -> np.ndarray:
        masked = np.where(v, plane.astype(np.float32), 0.0)
        warped = ndi.affine_transform(
            masked,
            inv_mat,
            offset=inv_off,
            output_shape=(h, out_w),
            order=1,
            mode="constant",
            cval=0.0,
        )
        return warped

    if a.ndim == 2:
        warped = _warp_2d(a)
        if np.issubdtype(a.dtype, np.integer):
            q = np.clip(np.round(warped), 0.0, 255.0).astype(a.dtype)
            out_arr = np.where(sheared_valid, np.maximum(q, 1), 0).astype(a.dtype)
        else:
            out_arr = np.where(sheared_valid, warped, 0.0).astype(a.dtype)
        return np.ascontiguousarray(out_arr), np.ascontiguousarray(sheared_valid)

    # 3D stack (L, H, W)
    out_stack = np.zeros((a.shape[0], h, out_w), dtype=a.dtype)
    is_int = np.issubdtype(a.dtype, np.integer)
    for li in range(a.shape[0]):
        warped = _warp_2d(a[li])
        if is_int:
            # Keep sheared CT positive voxels >= 1 when sheared_valid so Studio empty/valid checks align.
            q = np.clip(np.round(warped), 0.0, 255.0).astype(a.dtype)
            pos_src = ndi.affine_transform(
                (v & (a[li] > 0)).astype(np.float32),
                inv_mat,
                offset=inv_off,
                output_shape=(h, out_w),
                order=1,
                mode="constant",
                cval=0.0,
            ) > 0.999
            q = np.where(sheared_valid & pos_src, np.maximum(q, 1), q)
            out_stack[li] = np.where(sheared_valid, q, 0)
        else:
            out_stack[li] = np.where(sheared_valid, warped, 0.0)
    return np.ascontiguousarray(out_stack), np.ascontiguousarray(sheared_valid)


def invert_shear_map(
    prob_sheared: np.ndarray,
    valid_sheared: np.ndarray,
    orig_shape_hw: tuple[int, int],
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert sheared 2D probability and validity maps back to canonical ``orig_shape_hw``."""
    p = np.asarray(prob_sheared, dtype=np.float32)
    vs = np.asarray(valid_sheared, dtype=bool)
    if p.ndim != 2 or vs.shape != p.shape:
        raise ValueError(f"Expected matching 2D sheared prob/valid, got {p.shape} vs {vs.shape}")

    geom = compute_shear_geometry(orig_shape_hw, angle_deg)
    if p.shape != (geom["h"], geom["out_w"]):
        raise ValueError(
            f"Sheared map shape {p.shape} does not match expected {(geom['h'], geom['out_w'])} for {angle_deg} deg"
        )
    if abs(geom["k"]) < 1e-12:
        out_p = np.where(vs, p, 0.0).astype(np.float32)
        return np.ascontiguousarray(out_p.copy()), np.ascontiguousarray(vs.copy())

    h, w, cy, pad_x, k = geom["h"], geom["w"], geom["cy"], geom["pad_x"], geom["k"]
    fwd_mat = np.array([[1.0, 0.0], [k, 1.0]], dtype=np.float64)
    fwd_off = np.array([0.0, -k * cy + float(pad_x)], dtype=np.float64)

    back_valid = (
        ndi.affine_transform(
            vs.astype(np.float32),
            fwd_mat,
            offset=fwd_off,
            output_shape=(h, w),
            order=1,
            mode="constant",
            cval=0.0,
        )
        > 0.999
    )
    back_prob = ndi.affine_transform(
        np.where(vs, p, 0.0),
        fwd_mat,
        offset=fwd_off,
        output_shape=(h, w),
        order=1,
        mode="constant",
        cval=0.0,
    )
    back_prob = np.where(back_valid, np.clip(back_prob, 0.0, 1.0), 0.0).astype(np.float32)
    return np.ascontiguousarray(back_prob), np.ascontiguousarray(back_valid)


def erode_interior_mask(valid: np.ndarray, margin_px: int = INTERIOR_EROSION_PX) -> np.ndarray:
    """Return interior mask at Euclidean distance ``> margin_px`` from invalid/canvas borders."""
    v = np.asarray(valid, dtype=bool)
    if margin_px <= 0 or not v.any():
        return v.copy()
    padded = np.pad(v, 1, mode="constant", constant_values=False)
    dist = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    eroded = v & (dist > float(margin_px))
    return eroded if eroded.any() else v.copy()


def _run_studio_sheared_canonical(
    stack_lhw: np.ndarray,
    valid_ct: np.ndarray,
    mesh_index: int,
    staged: dict[str, Path],
    *,
    angle_deg: float,
    stride: int,
    batch_size: int,
    batch_pause_ms: int,
    max_gpu_temp_c: int,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]],
    guardrail_fn: Callable[..., dict[str, int] | None] | None,
) -> dict[str, dict[str, Any]]:
    """Shear CT stack by ``angle_deg``, run Studio, and un-shear maps to canonical frame."""
    if guardrail_fn is not None:
        guardrail_fn(max_temp_c=max_gpu_temp_c)
    orig_hw = (stack_lhw.shape[1], stack_lhw.shape[2])
    is_zero = abs(float(angle_deg)) < 1e-12
    t_stack, t_valid = (
        (stack_lhw, valid_ct)
        if is_zero
        else apply_shear_transform(stack_lhw, valid_ct, angle_deg)
    )
    try:
        raw_res = run_studio_fn(
            t_stack,
            mesh_index,
            staged,
            stride=stride,
            batch_size=batch_size,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
        )
        canonical: dict[str, dict[str, Any]] = {}
        for order in ("desc", "asc"):
            r = raw_res[order]
            r_valid = np.asarray(r["valid"], dtype=bool) & t_valid
            if is_zero:
                prob = np.where(r_valid, r["prob"], 0.0).astype(np.float32)
                val = r_valid.copy()
            else:
                prob, val = invert_shear_map(r["prob"], r_valid, orig_hw, angle_deg)
                val &= valid_ct
                prob = np.where(val, prob, 0.0).astype(np.float32)
            canonical[order] = {
                "prob": prob,
                "valid": val,
                "seconds": r.get("seconds", 0.0),
                "peak_allocated_mib": r.get("peak_allocated_mib", 0),
                "peak_reserved_mib": r.get("peak_reserved_mib", 0),
            }
        return canonical
    finally:
        if not is_zero:
            del t_stack, t_valid
        GTI._cleanup_ram_and_cuda()
        if guardrail_fn is not None:
            guardrail_fn(max_temp_c=max_gpu_temp_c)


def run_arm_a_shear_stress(
    *,
    out_private_dir: Path,
    work_root: Path,
    staged: dict[str, Path],
    mesh_source: str = GTI.PHERC0306B_MESH_URL,
    ct_zarr_url: str = GTI.PHERC0306B_CT_ZARR_URL,
    uv_crop: tuple[int, int, int, int] | None = GTI.PHERC0306B_DEFAULT_UV_CROP,
    voxel_um: float = GTI.PHERC0306B_VOXEL_UM,
    shear_angles_deg: Sequence[float] = SHEAR_ANGLES_DEG,
    num_slices: int = 31,
    studio_stride: int = 48,
    studio_batch: int = GTI.DEFAULT_STUDIO_BATCH,
    batch_pause_ms: int = GTI.DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = GTI.DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = GTI.DEFAULT_CACHE_MB,
    vram_fraction: float = GTI.DEFAULT_VRAM_FRACTION,
    interior_erosion_px: int = INTERIOR_EROSION_PX,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    guardrail_fn: Callable[..., dict[str, int] | None] | None = SSRX.enforce_gpu_guardrail,
) -> dict[str, Any]:
    """Execute Arm A grid-shear stress sweep (`0, +/-15, +/-30 deg`) on `PHerc. 0306B w074`."""
    out_dir = SSRX.ensure_outside_git_repo(out_private_dir) / "gpu_shear_0211"
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_runner = run_studio_fn or SSRX.run_studio_both_orders

    t0 = time.perf_counter()
    angles = sorted(float(a) for a in shear_angles_deg)
    if 0.0 not in angles:
        raise ValueError("shear_angles_deg must include 0.0 baseline")
    ordered_angles = [0.0] + [a for a in shear_angles_deg if abs(float(a)) > 1e-12]

    with tempfile.TemporaryDirectory(prefix="arma-shear-", dir=str(work_root), ignore_cleanup_errors=True) as mtmp:
        rendered = SSRX.render_mesh_ct_stack(
            mesh_source,
            Path(mtmp),
            ct_zarr_url=ct_zarr_url,
            m7_zarr_url=None,
            snap_passes=0,
            num_slices=num_slices,
            uv_crop=uv_crop,
            cache_mb=cache_mb,
        )
        stack_lhw = rendered.pop("stack_lhw")
        valid_ct = rendered.pop("valid_ct")
        mesh_idx = (num_slices - 1) // 2
        interior_mask = erode_interior_mask(valid_ct, margin_px=interior_erosion_px)

        variants_out: dict[str, dict[str, Any]] = {}
        montage_panels: list[tuple[str, np.ndarray, np.ndarray]] = []
        ref_prob: np.ndarray | None = None
        ref_valid: np.ndarray | None = None
        ref_face = "asc"

        for angle in ordered_angles:
            vkey = shear_variant_key(angle)
            res_s = _run_studio_sheared_canonical(
                stack_lhw,
                valid_ct,
                mesh_idx,
                staged,
                angle_deg=angle,
                stride=studio_stride,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                run_studio_fn=studio_runner,
                guardrail_fn=guardrail_fn,
            )
            if abs(angle) < 1e-12:
                metrics = GTI.compute_variant_metrics(res_s, valid_ct, voxel_um)
                ref_face = metrics["tfce_q34"]["dominant_face"]
                ref_prob = res_s[ref_face]["prob"].copy()
                ref_valid = (np.asarray(res_s[ref_face]["valid"], dtype=bool) & valid_ct).copy()
                metrics["agreement_vs_ref"] = GTI.compute_map_agreement(ref_prob, ref_valid, ref_prob, ref_valid)
                metrics["agreement_interior_vs_ref"] = GTI.compute_map_agreement(
                    ref_prob, ref_valid & interior_mask, ref_prob, ref_valid & interior_mask
                )
            else:
                metrics = GTI.compute_variant_metrics(
                    res_s, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
                )
                dom_p = res_s[ref_face]["prob"]
                dom_v = np.asarray(res_s[ref_face]["valid"], dtype=bool) & valid_ct
                metrics["agreement_interior_vs_ref"] = GTI.compute_map_agreement(
                    dom_p,
                    dom_v & interior_mask,
                    ref_prob,
                    ref_valid & interior_mask,
                )

            metrics["angle_deg"] = float(angle)
            metrics["shear_k_tan"] = round(math.tan(math.radians(float(angle))), 6)
            variants_out[vkey] = metrics

            st_dom = metrics["detectors"][f"studio_{ref_face}"]
            agr_int = metrics["agreement_interior_vs_ref"]
            montage_panels.append(
                (
                    f"{vkey} ({angle:+.0f}deg {ref_face} gt07={st_dom['frac_gt_07']:.4f} "
                    f"ret07={agr_int['ref_hit_retention_ge_07']:.3f} IoU07={agr_int['iou_ge_07']:.3f})",
                    res_s[ref_face]["prob"],
                    np.asarray(res_s[ref_face]["valid"], dtype=bool) & valid_ct,
                )
            )
            del res_s

        ct_mid = stack_lhw[mesh_idx].copy()
        del stack_lhw
        GTI._cleanup_ram_and_cuda()

    GTI.write_comparison_montage(
        out_dir / "arm_a_shear_montage.png",
        "Exp T16-GPU Arm A: PHerc. 0306B w074 Grid Shear Stress (0, +/-15, +/-30 deg, Studio fp16 s48)",
        ct_mid,
        valid_ct,
        montage_panels,
        voxel_um,
    )

    non_zero_keys = [shear_variant_key(a) for a in ordered_angles if abs(a) > 1e-12]
    post_gpu = SSRX.query_gpu_telemetry()
    orig_gt07 = max(float(variants_out["shear_0"]["detectors"][f"studio_{ref_face}"]["frac_gt_07"]), 1e-9)

    shear_summary: dict[str, Any] = {
        "schema_version": "gpu_shear_stress_arm_v1",
        "scroll": "PHerc0306B",
        "mesh_tag": "PHerc0306B_20250521133212_w074own_touchup_V5Rc",
        "uv_crop": list(uv_crop) if uv_crop is not None else None,
        "voxel_um": float(voxel_um),
        "num_slices": int(num_slices),
        "studio_stride": int(studio_stride),
        "interpolation": "bilinear_order_1_area_preserving_expanded_canvas",
        "interior_erosion_px": int(interior_erosion_px),
        "valid_ct_pixels": int(valid_ct.sum()),
        "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
        "interior_valid_pixels": int((valid_ct & interior_mask).sum()),
        "ct_chunks_fetched": int(rendered["ct_chunks"]),
        "ct_mib_fetched": round(float(rendered["ct_bytes"]) / (1024 ** 2), 2),
        "reference_reading_face": ref_face,
        "variants": variants_out,
        "findings_summary": {
            "per_angle_interior_summary": {
                k: {
                    "angle_deg": variants_out[k]["angle_deg"],
                    "frac_gt_07": variants_out[k]["detectors"][f"studio_{ref_face}"]["frac_gt_07"],
                    "frac_gt_07_ratio_vs_0": round(
                        float(variants_out[k]["detectors"][f"studio_{ref_face}"]["frac_gt_07"]) / orig_gt07,
                        6,
                    ),
                    "interior_pearson_r": variants_out[k]["agreement_interior_vs_ref"]["pearson_r"],
                    "interior_iou_ge_05": variants_out[k]["agreement_interior_vs_ref"]["iou_ge_05"],
                    "interior_iou_ge_07": variants_out[k]["agreement_interior_vs_ref"]["iou_ge_07"],
                    "interior_hit_retention_ge_05": variants_out[k]["agreement_interior_vs_ref"][
                        "ref_hit_retention_ge_05"
                    ],
                    "interior_hit_retention_ge_07": variants_out[k]["agreement_interior_vs_ref"][
                        "ref_hit_retention_ge_07"
                    ],
                    "width_median_mm": variants_out[k]["tfce_q34"]["width_median_mm"],
                    "q34_score": variants_out[k]["tfce_q34"]["q34_score"],
                }
                for k in variants_out
            },
            "min_15deg_interior_ret_ge_07": min(
                float(variants_out[k]["agreement_interior_vs_ref"]["ref_hit_retention_ge_07"])
                for k in ("shear_p15", "shear_m15")
                if k in variants_out
            )
            if any(k in variants_out for k in ("shear_p15", "shear_m15"))
            else None,
            "min_30deg_interior_ret_ge_07": min(
                float(variants_out[k]["agreement_interior_vs_ref"]["ref_hit_retention_ge_07"])
                for k in ("shear_p30", "shear_m30")
                if k in variants_out
            )
            if any(k in variants_out for k in ("shear_p30", "shear_m30"))
            else None,
            "min_overall_interior_pearson_r": min(
                float(variants_out[k]["agreement_interior_vs_ref"]["pearson_r"])
                for k in non_zero_keys
            )
            if non_zero_keys
            else 1.0,
        },
        "safety_profile": {
            "vram_fraction": float(vram_fraction),
            "studio_batch": int(studio_batch),
            "batch_pause_ms": int(batch_pause_ms),
            "max_gpu_temp_c": int(max_gpu_temp_c),
            "post_run_temp_c": post_gpu["temp_c"] if post_gpu is not None else None,
            "total_seconds": round(time.perf_counter() - t0, 2),
        },
    }
    return shear_summary


def run_arm_b_pherc0211_winner(
    *,
    out_private_dir: Path,
    work_root: Path,
    staged: dict[str, Path],
    mesh_source: str = PHERC0211_MESH_URL,
    ct_zarr_url: str = PHERC0211_CT_ZARR_URL,
    uv_crop: tuple[int, int, int, int] | None = PHERC0211_DEFAULT_UV_CROP,
    voxel_um: float = PHERC0211_VOXEL_UM,
    num_slices: int = 31,
    studio_stride: int = 48,
    studio_batch: int = GTI.DEFAULT_STUDIO_BATCH,
    ahn_batch: int = 1,
    include_ahn: bool = False,
    batch_pause_ms: int = GTI.DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = GTI.DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = GTI.DEFAULT_CACHE_MB,
    vram_fraction: float = GTI.DEFAULT_VRAM_FRACTION,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    run_ahn_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    guardrail_fn: Callable[..., dict[str, int] | None] | None = SSRX.enforce_gpu_guardrail,
) -> dict[str, Any]:
    """Execute Arm B validation on the public `PHerc. 0211` winner mesh (3-line region)."""
    base_out = SSRX.ensure_outside_git_repo(out_private_dir)
    out_dir = base_out / "gpu_shear_0211"
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_runner = run_studio_fn or SSRX.run_studio_both_orders
    ahn_runner = run_ahn_fn or SSRX.run_ahn_both_orders

    t0 = time.perf_counter()
    if guardrail_fn is not None:
        guardrail_fn(max_temp_c=max_gpu_temp_c)

    with tempfile.TemporaryDirectory(prefix="armb-0211-", dir=str(work_root), ignore_cleanup_errors=True) as mtmp:
        rendered = SSRX.render_mesh_ct_stack(
            mesh_source,
            Path(mtmp),
            ct_zarr_url=ct_zarr_url,
            m7_zarr_url=None,
            snap_passes=0,
            num_slices=num_slices,
            uv_crop=uv_crop,
            cache_mb=cache_mb,
        )
        stack_lhw = rendered.pop("stack_lhw")
        valid_ct = rendered.pop("valid_ct")
        mesh_idx = (num_slices - 1) // 2

        try:
            import onsheet_score  # type: ignore[import-not-found]

            coherence = onsheet_score.score_stack(stack_lhw, mid=mesh_idx, dk=2)
        except ModuleNotFoundError:
            fb = SSR.layer_coherence_score(
                stack_lhw, mesh_index=mesh_idx, delta=min(2, mesh_idx), valid_mask=valid_ct
            )
            coherence = {
                "f_parallel": fb["sheet_parallel_fraction"],
                "n_tiles": fb["patch_count"],
                "mean_zero_shift_r": fb["mean_zero_shift_r"],
                "mean_best_shifted_r": fb["mean_best_shifted_r"],
                "backend": "stream_snap_render_fallback",
            }

        res_studio = studio_runner(
            stack_lhw,
            mesh_idx,
            staged,
            stride=studio_stride,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
        )

        res_ahn = None
        if include_ahn:
            if guardrail_fn is not None:
                guardrail_fn(max_temp_c=max_gpu_temp_c)
            res_ahn = ahn_runner(
                stack_lhw,
                mesh_idx,
                staged,
                batch_size=ahn_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
            )

        metrics = GTI.compute_variant_metrics(res_studio, valid_ct, voxel_um)
        fp_diag = SSRX.compute_fp_diagnostics(stack_lhw[mesh_idx], valid_ct, res_studio, res_ahn, voxel_um)

        if res_ahn is not None:
            for order in ("asc", "desc"):
                ar = res_ahn[order]
                astats = SSRX.summarize_map(ar["prob"], ar["valid"], voxel_um)
                astats.update(
                    tiles=ar.get("tiles", 0),
                    seconds=ar.get("seconds", 0.0),
                    peak_allocated_mib=ar.get("peak_allocated_mib", 0),
                    peak_reserved_mib=ar.get("peak_reserved_mib", 0),
                )
                metrics["detectors"][f"ahn_s0_{order}"] = astats

        dom_face = metrics["tfce_q34"]["dominant_face"]
        opp_face = metrics["tfce_q34"]["opposite_face"]
        panels: list[tuple[str, np.ndarray, np.ndarray]] = [
            (
                f"Studio fp16 {dom_face} (dom, gt07={metrics['detectors'][f'studio_{dom_face}']['frac_gt_07']:.4f})",
                res_studio[dom_face]["prob"],
                np.asarray(res_studio[dom_face]["valid"], dtype=bool) & valid_ct,
            ),
            (
                f"Studio fp16 {opp_face} (opp, gt07={metrics['detectors'][f'studio_{opp_face}']['frac_gt_07']:.4f})",
                res_studio[opp_face]["prob"],
                np.asarray(res_studio[opp_face]["valid"], dtype=bool) & valid_ct,
            ),
        ]
        if res_ahn is not None:
            for order in ("asc", "desc"):
                panels.append(
                    (
                        f"A-hn fp16 {order} (gt07={metrics['detectors'][f'ahn_s0_{order}']['frac_gt_07']:.4f})",
                        res_ahn[order]["prob"],
                        np.asarray(res_ahn[order]["valid"], dtype=bool) & valid_ct,
                    )
                )

        ct_mid = stack_lhw[mesh_idx].copy()
        del stack_lhw, res_studio, res_ahn
        GTI._cleanup_ram_and_cuda()

    GTI.write_comparison_montage(
        out_dir / "arm_b_pherc0211_winner_montage.png",
        "Exp T16-GPU Arm B: PHerc. 0211 Winner Mesh 3-Line Control (Studio fp16 s48)",
        ct_mid,
        valid_ct,
        panels,
        voxel_um,
    )

    post_gpu = SSRX.query_gpu_telemetry()
    dom_st = metrics["detectors"][f"studio_{dom_face}"]
    opp_st = metrics["detectors"][f"studio_{opp_face}"]
    dom_over_opp_ratio = round(
        float(dom_st["frac_gt_07"]) / max(float(opp_st["frac_gt_07"]), 1e-6), 4
    )

    return {
        "schema_version": "gpu_pherc0211_winner_control_v1",
        "scroll": "PHerc0211",
        "segment": PHERC0211_SEGMENT,
        "mesh_tag": PHERC0211_MESH_TAG,
        "uv_crop": list(uv_crop) if uv_crop is not None else None,
        "voxel_um": float(voxel_um),
        "num_slices": int(num_slices),
        "studio_stride": int(studio_stride),
        "include_ahn": bool(include_ahn),
        "render_seconds": rendered["render_sec"],
        "valid_ct_pixels": int(valid_ct.sum()),
        "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
        "ct_chunks_fetched": int(rendered["ct_chunks"]),
        "ct_mib_fetched": round(float(rendered["ct_bytes"]) / (1024 ** 2), 2),
        "onsheet_coherence": coherence,
        "detectors": metrics["detectors"],
        "tfce_q34": metrics["tfce_q34"],
        "fp_diagnostics": fp_diag,
        "validation_summary": {
            "dominant_face": dom_face,
            "opposite_face": opp_face,
            "dom_frac_gt_07": dom_st["frac_gt_07"],
            "opp_frac_gt_07": opp_st["frac_gt_07"],
            "dom_over_opp_gt_07_ratio": dom_over_opp_ratio,
            "passes_q34_bar": metrics["tfce_q34"]["passes_q34_bar"],
            "regime_label": metrics["tfce_q34"]["regime_label"],
            "width_median_mm": metrics["tfce_q34"]["width_median_mm"],
            "max_b1": metrics["tfce_q34"]["max_b1"],
            "q34_score": metrics["tfce_q34"]["q34_score"],
            "winner_letters_recovered": bool(
                dom_over_opp_ratio >= 3.0
                and dom_st["frac_gt_07"] >= 0.05
                and metrics["tfce_q34"]["diff_tfce_p99"] >= 4.0
                and metrics["tfce_q34"]["passes_stroke_morphology"]
            ),
        },
        "safety_profile": {
            "vram_fraction": float(vram_fraction),
            "studio_batch": int(studio_batch),
            "batch_pause_ms": int(batch_pause_ms),
            "max_gpu_temp_c": int(max_gpu_temp_c),
            "post_run_temp_c": post_gpu["temp_c"] if post_gpu is not None else None,
            "total_seconds": round(time.perf_counter() - t0, 2),
        },
    }


def run_exp_t16_gpu(
    *,
    arm: str = "both",
    out_private_dir: Path,
    public_receipt_path: Path | None = DEFAULT_PUBLIC_RECEIPT,
    vram_fraction: float = GTI.DEFAULT_VRAM_FRACTION,
    studio_batch: int = GTI.DEFAULT_STUDIO_BATCH,
    ahn_batch: int = 1,
    include_ahn: bool = False,
    batch_pause_ms: int = GTI.DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = GTI.DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = GTI.DEFAULT_CACHE_MB,
    shear_angles_deg: Sequence[float] = SHEAR_ANGLES_DEG,
) -> dict[str, Any]:
    """Run Arm A and/or Arm B sequentially under strict RTX 4060 caps and write public receipt."""
    if not (0.05 <= float(vram_fraction) <= GTI.MAX_SAFE_VRAM_FRACTION):
        raise ValueError(
            f"--vram-fraction must be within [0.05, {GTI.MAX_SAFE_VRAM_FRACTION}], got {vram_fraction}"
        )
    out_dir = SSRX.ensure_outside_git_repo(out_private_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SSRX.enforce_gpu_guardrail(max_temp_c=int(max_gpu_temp_c))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_shear_and_0211_control.py")
    torch.cuda.set_per_process_memory_fraction(float(vram_fraction), 0)

    detectors = {"studio", "ahn"} if (include_ahn and arm in ("pherc0211", "both")) else {"studio"}
    t_total = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="rtx4060-t16-work-", ignore_cleanup_errors=True
    ) as tmp:
        work_root = Path(tmp)
        staged = SSRX.stage_models(work_root, detectors)

        arm_a_res = None
        if arm in ("shear", "both"):
            arm_a_res = run_arm_a_shear_stress(
                out_private_dir=out_dir,
                work_root=work_root,
                staged=staged,
                shear_angles_deg=shear_angles_deg,
                studio_batch=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                cache_mb=cache_mb,
                vram_fraction=vram_fraction,
            )

        arm_b_res = None
        if arm in ("pherc0211", "both"):
            arm_b_res = run_arm_b_pherc0211_winner(
                out_private_dir=out_dir,
                work_root=work_root,
                staged=staged,
                studio_batch=studio_batch,
                ahn_batch=ahn_batch,
                include_ahn=include_ahn,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                cache_mb=cache_mb,
                vram_fraction=vram_fraction,
            )

    post_gpu = SSRX.query_gpu_telemetry()
    payload: dict[str, Any] = {
        "schema_version": "gpu_shear_and_0211_control_v1",
        "experiment": "Exp T16-GPU: PHerc. 0306B Grid Shear Stress Harness & PHerc. 0211 Winner Validation",
        "arm_executed": arm,
        "hardware_safety": {
            "vram_fraction": float(vram_fraction),
            "studio_batch": int(studio_batch),
            "ahn_batch": int(ahn_batch),
            "batch_pause_ms": int(batch_pause_ms),
            "max_gpu_temp_c": int(max_gpu_temp_c),
            "cache_mb": int(cache_mb),
            "post_run_temp_c": post_gpu["temp_c"] if post_gpu is not None else None,
            "total_seconds": round(time.perf_counter() - t_total, 2),
            "persistent_ct_on_disk": False,
        },
        "arm_a_shear_stress": arm_a_res,
        "arm_b_pherc0211_winner": arm_b_res,
    }
    GTI.assert_no_local_paths_in_public_json(payload)
    if public_receipt_path is not None:
        public_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        public_receipt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser with strict RTX 4060 Laptop VRAM and thermal guardrails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("shear", "pherc0211", "both"), default="both")
    ap.add_argument("--out-private-dir", type=Path, required=True)
    ap.add_argument("--public-receipt", type=Path, default=DEFAULT_PUBLIC_RECEIPT)
    ap.add_argument("--vram-fraction", type=float, default=GTI.DEFAULT_VRAM_FRACTION)
    ap.add_argument("--studio-batch", type=int, default=GTI.DEFAULT_STUDIO_BATCH)
    ap.add_argument("--ahn-batch", type=int, default=1)
    ap.add_argument("--include-ahn", action="store_true")
    ap.add_argument("--batch-pause-ms", type=int, default=GTI.DEFAULT_BATCH_PAUSE_MS)
    ap.add_argument("--max-gpu-temp-c", type=int, default=GTI.DEFAULT_MAX_GPU_TEMP_C)
    ap.add_argument("--cache-mb", type=int, default=GTI.DEFAULT_CACHE_MB)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_exp_t16_gpu(
        arm=args.arm,
        out_private_dir=args.out_private_dir,
        public_receipt_path=args.public_receipt,
        vram_fraction=args.vram_fraction,
        studio_batch=args.studio_batch,
        ahn_batch=args.ahn_batch,
        include_ahn=args.include_ahn,
        batch_pause_ms=args.batch_pause_ms,
        max_gpu_temp_c=args.max_gpu_temp_c,
        cache_mb=args.cache_mb,
    )
    print(f"Wrote {args.public_receipt} in {payload['hardware_safety']['total_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
