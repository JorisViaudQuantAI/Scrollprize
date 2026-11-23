# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Safe RTX 4060 GPU transverse interventions (Exp T10-GPU).

Executes two strictly VRAM-capped, thermally guarded, zero-persistent-CT GPU experiments
using a single streamed surface-volume stack per arm in RAM:

1. Arm 1 (`control` — Public `PHerc. 0306B w074`, `uv_crop=95:120,190:250`, `8.64 um`):
   * Tests `villa #1909` mirror/orientation invariance (`orig`, `flip_lr`, `flip_ud`, `rot180`)
     under `Studio fp16 stride=48` (Hann blend), un-flipping probability maps back to canonical
     coordinates for pixel-exact IoU / Pearson correlation and `Q34` Stroke-Bounded TFCE checks.
   * Tests `Q22` radiometric shift sensitivity and Nyul-Udupa rescue (`nyul_udupa` no-harm check,
     `shift_plus24` simulating `PHerc. 0358`'s `+24` count papyrus-wall shift, and
     `shift_plus24_nyul` standardized back to the pooled reference control landmarks).
   * Writes a sanitized public-ready JSON receipt (`arm1_pherc0306b_control_public.json`,
     strictly verified to contain zero local filesystem paths) plus a labeled comparison montage.

2. Arm 2 (`adversarial` — Private `PHerc0125/z8736_w035`, `uv_crop=10:56,178:255`, `9.362 um`):
   * Streams one `43`-layer `2x m7`-snapped stack (`base_mesh_index=21`) in RAM.
   * Sweeps depth offsets `dz in (-6, -3, 0, +3, +6)` (`31`-layer windows, `stride=96`) and
     fine `stride=48` orientation states (`dz0_orig_s48`, `dz0_flip_lr_s48`, `dz0_flip_ud_s48`,
     `dz0_rot180_s48`) combined with `tfce_opposite_face_fwer` (`width_median_mm`, `q34_score`,
     `passes_q34_bar`, `stroke_bounded_tfce` retention) to settle whether the `1.5-2.5 mm`
     seam/edge blobs resolve into crisp `~0.35 mm` letter strokes or close as non-stroke artifacts.
   * Writes all private outputs strictly outside the git repository (`ensure_outside_git_repo`).
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
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import q22_radiometric_shift_audit as Q22  # noqa: E402
import screen_scroll_rtx4060 as SSRX  # noqa: E402
import stream_snap_render as SSR  # noqa: E402
import tfce_stroke_calibration as TFCE  # noqa: E402

VALID_SPATIAL_MODES: tuple[str, ...] = ("orig", "flip_lr", "flip_ud", "rot180")
MAX_SAFE_VRAM_FRACTION: float = 0.50
DEFAULT_VRAM_FRACTION: float = 0.35
DEFAULT_STUDIO_BATCH: int = 1
DEFAULT_BATCH_PAUSE_MS: int = 350
DEFAULT_MAX_GPU_TEMP_C: int = 74
DEFAULT_CACHE_MB: int = 256

# Pinned public URLs for Arm 1 (PHerc0306B w074 positive control)
PHERC0306B_MESH_URL = (
    "https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/"
    "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/"
    "PHerc0306B_20250521133212_w074own_touchup_V5Rc"
)
PHERC0306B_CT_ZARR_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr"
)
PHERC0306B_DEFAULT_UV_CROP: tuple[int, int, int, int] = (95, 120, 190, 250)
PHERC0306B_VOXEL_UM: float = 8.64

# Pinned public URLs for Arm 2 (PHerc0125 z8736_w035 private adversarial verification)
PHERC0125_Z8736_MESH_URL = (
    "https://raw.githubusercontent.com/pscamillo/vesuvius-eligible-meshes/"
    "66000516c97e043b696ce874f145e29638109c91/meshes/PHerc0125/z8736_w035"
)
PHERC0125_CT_ZARR_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr"
)
PHERC0125_M7_ZARR_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0125/representations/predictions/surfaces/"
    "20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
PHERC0125_Z8736_DEFAULT_UV_CROP: tuple[int, int, int, int] = (10, 56, 178, 255)
PHERC0125_VOXEL_UM: float = 9.362

DEFAULT_Q22_AUDIT_JSON = HERE / "evidence" / "q22_radiometric_shift_audit.json"
FALLBACK_NYUL_PERCENTILES: tuple[float, ...] = (1.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.0)
FALLBACK_NYUL_REF_LANDMARKS: tuple[float, ...] = (23.0, 39.0, 56.0, 100.0, 127.0, 146.0, 182.0)


def load_q22_reference_landmarks(
    audit_json_path: Path | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load pooled control Nyul-Udupa reference percentiles and landmarks from Exp T4 JSON."""
    path = audit_json_path if audit_json_path is not None else DEFAULT_Q22_AUDIT_JSON
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        base = data.get("reference_control_baseline", {})
        pcts = tuple(float(v) for v in base.get("nyul_udupa_default_percentiles", FALLBACK_NYUL_PERCENTILES))
        refs = tuple(float(v) for v in base.get("nyul_udupa_ref_landmarks", FALLBACK_NYUL_REF_LANDMARKS))
        if len(pcts) >= 2 and len(pcts) == len(refs):
            return pcts, refs
    return FALLBACK_NYUL_PERCENTILES, FALLBACK_NYUL_REF_LANDMARKS


def apply_spatial_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    """Apply an orientation transform (`orig`, `flip_lr`, `flip_ud`, `rot180`) over trailing `(H, W)` axes."""
    if mode not in VALID_SPATIAL_MODES:
        raise ValueError(f"Unsupported spatial transform mode {mode!r}; expected one of {VALID_SPATIAL_MODES}")
    a = np.asarray(arr)
    if a.ndim < 2:
        raise ValueError(f"Expected at least 2D array with trailing (H, W) axes, got shape {a.shape}")
    if mode == "orig":
        return np.ascontiguousarray(a.copy())
    if mode == "flip_lr":
        return np.ascontiguousarray(a[..., :, ::-1])
    if mode == "flip_ud":
        return np.ascontiguousarray(a[..., ::-1, :])
    return np.ascontiguousarray(a[..., ::-1, ::-1])


def invert_spatial_map(arr_hw: np.ndarray, mode: str) -> np.ndarray:
    """Invert an orientation transform on a 2D map (all 4 spatial modes are involutions)."""
    a = np.asarray(arr_hw)
    if a.ndim != 2:
        raise ValueError(f"invert_spatial_map expects a 2D array, got shape {a.shape}")
    return apply_spatial_transform(a, mode)


def apply_radiometric_shift(
    stack_lhw: np.ndarray,
    valid_mask: np.ndarray | None,
    delta_counts: float,
) -> np.ndarray:
    """Shift positive valid CT voxels by ``delta_counts`` into ``[1, 255]`` while keeping ``0`` strictly ``0``."""
    arr = np.asarray(stack_lhw)
    if arr.ndim < 2:
        raise ValueError(f"Expected stack with at least 2 dimensions, got shape {arr.shape}")
    if not math.isfinite(delta_counts):
        raise ValueError(f"delta_counts must be finite, got {delta_counts!r}")

    out = np.zeros(arr.shape, dtype=np.uint8)
    fg = np.isfinite(arr) & (arr > 0)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape == arr.shape:
            fg &= vm
        elif arr.ndim == 3 and vm.ndim == 2 and vm.shape == arr.shape[1:]:
            fg &= vm[None, :, :]
        else:
            fg &= np.broadcast_to(vm, arr.shape)

    if fg.any():
        shifted = np.clip(np.round(arr[fg].astype(np.float64) + float(delta_counts)), 1.0, 255.0)
        out[fg] = shifted.astype(np.uint8)
    return out


def apply_nyul_standardization_u8(
    stack_lhw: np.ndarray,
    valid_mask: np.ndarray | None,
    ref_landmarks: Sequence[float] = FALLBACK_NYUL_REF_LANDMARKS,
    percentiles: Sequence[float] = FALLBACK_NYUL_PERCENTILES,
) -> np.ndarray:
    """Apply piecewise-linear Nyul-Udupa standardization and quantize foreground back to ``uint8`` ``[1..255]``."""
    std_f32 = Q22.nyul_udupa_standardize_stack(
        stack_lhw,
        valid_mask,
        ref_landmarks=ref_landmarks,
        percentiles=percentiles,
        out_range=(1.0, 255.0),
    )
    out = np.zeros(std_f32.shape, dtype=np.uint8)
    fg = std_f32 > 0.0
    if fg.any():
        out[fg] = np.clip(np.round(std_f32[fg]), 1.0, 255.0).astype(np.uint8)
    return out


def summarize_stack_radiometry(
    stack_lhw: np.ndarray,
    valid_mask: np.ndarray | None,
    percentiles: Sequence[float] = FALLBACK_NYUL_PERCENTILES,
) -> dict[str, Any]:
    """Compute foreground voxel percentiles and clipping fraction (>=200) on a CT stack."""
    arr = np.asarray(stack_lhw)
    fg = np.isfinite(arr) & (arr > 0)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if arr.ndim == 3 and vm.ndim == 2 and vm.shape == arr.shape[1:]:
            fg &= vm[None, :, :]
        else:
            fg &= np.broadcast_to(vm, arr.shape)
    if not fg.any():
        return {
            "nonzero_voxels": 0,
            "mean": 0.0,
            "frac_ge_200": 0.0,
            "landmarks": {f"p{int(p):02d}": 0.0 for p in percentiles},
        }
    vals = arr[fg].astype(np.float64)
    pct_vals = np.percentile(vals, percentiles)
    return {
        "nonzero_voxels": int(vals.size),
        "mean": round(float(vals.mean()), 4),
        "frac_ge_200": round(float((vals >= 200.0).mean()), 6),
        "landmarks": {f"p{int(p):02d}": round(float(v), 3) for p, v in zip(percentiles, pct_vals)},
    }


def extract_depth_window(
    stack_lhw: np.ndarray,
    base_mesh_index: int,
    dz: int,
    window_slices: int = 31,
) -> tuple[np.ndarray, int]:
    """Extract a ``window_slices`` sub-stack centered at ``base_mesh_index + dz``."""
    arr = np.asarray(stack_lhw)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack (L, H, W), got shape {arr.shape}")
    if window_slices < 17 or window_slices % 2 == 0:
        raise ValueError(f"window_slices must be an odd integer >= 17, got {window_slices}")
    half = (window_slices - 1) // 2
    center = int(base_mesh_index) + int(dz)
    start = center - half
    end = center + half + 1
    if start < 0 or end > arr.shape[0]:
        raise ValueError(
            f"Depth shift dz={dz} around base_mesh_index={base_mesh_index} requires slices "
            f"[{start}:{end}], outside stack depth {arr.shape[0]}"
        )
    return np.ascontiguousarray(arr[start:end]), half


def _extract_stroke_width_metrics(tfce_res: dict[str, Any]) -> dict[str, Any]:
    """Extract stroke-width diagnostics from ``tfce_opposite_face_fwer`` output."""
    topo = tfce_res.get("transverse_stroke_topology", {})
    sw = topo.get("stroke_width", {})
    hi = sw.get("hi", sw.get("hi_ge_07", {})) if isinstance(sw, dict) else {}
    lo = sw.get("lo", sw.get("lo_ge_05", {})) if isinstance(sw, dict) else {}
    width_med_hi = hi.get("width_median_mm")
    width_med_lo = lo.get("width_median_mm")
    width_med = width_med_hi if width_med_hi is not None else width_med_lo
    return {
        "width_median_mm": width_med,
        "width_median_hi_mm": width_med_hi,
        "width_median_lo_mm": width_med_lo,
        "width_cv_hi": hi.get("width_cv"),
        "narrow_stroke_frac_hi": hi.get("narrow_stroke_frac", 0.0),
        "wide_blob_frac_hi": hi.get("wide_blob_frac", 0.0),
        "skeleton_length_hi_mm": hi.get("skeleton_length_mm", 0.0),
    }


def compute_map_agreement(
    prob: np.ndarray,
    valid: np.ndarray,
    ref_prob: np.ndarray,
    ref_valid: np.ndarray,
) -> dict[str, float]:
    """Compute pixel-aligned IoU (@0.5, @0.7), Pearson r, MAE, and recall retention vs reference map."""
    common = np.asarray(valid, dtype=bool) & np.asarray(ref_valid, dtype=bool)
    n_common = int(common.sum())
    if n_common == 0:
        return {
            "common_valid_px": 0,
            "pearson_r": 0.0,
            "mae": 0.0,
            "iou_ge_05": 0.0,
            "iou_ge_07": 0.0,
            "ref_hit_retention_ge_05": 0.0,
            "ref_hit_retention_ge_07": 0.0,
        }
    p = np.asarray(prob, dtype=np.float64)[common]
    r = np.asarray(ref_prob, dtype=np.float64)[common]
    mae = float(np.mean(np.abs(p - r)))
    p_std = float(np.std(p))
    r_std = float(np.std(r))
    if p_std > 1e-12 and r_std > 1e-12:
        pearson_r = float(np.corrcoef(p, r)[0, 1])
    else:
        pearson_r = 1.0 if mae < 1e-9 else 0.0

    out: dict[str, float] = {
        "common_valid_px": n_common,
        "pearson_r": round(pearson_r, 6),
        "mae": round(mae, 6),
    }
    for tag, thr in (("05", 0.5), ("07", 0.7)):
        pm = p >= thr
        rm = r >= thr
        inter = int(np.sum(pm & rm))
        union = int(np.sum(pm | rm))
        ref_cnt = int(np.sum(rm))
        out[f"iou_ge_{tag}"] = round(float(inter / union) if union > 0 else 1.0, 6)
        out[f"ref_hit_retention_ge_{tag}"] = round(float(inter / ref_cnt) if ref_cnt > 0 else 1.0, 6)
    return out


def compute_variant_metrics(
    res_studio: dict[str, dict[str, Any]],
    valid_ct: np.ndarray,
    voxel_um: float,
    *,
    ref_prob: np.ndarray | None = None,
    ref_valid: np.ndarray | None = None,
    ref_face: str | None = None,
) -> dict[str, Any]:
    """Summarize ``Studio`` ``asc``/``desc`` outputs, ``Q34`` TFCE metrics, and optional reference agreement."""
    v_ct = np.asarray(valid_ct, dtype=bool)
    detector_stats: dict[str, dict[str, Any]] = {}
    for order in ("desc", "asc"):
        r = res_studio[order]
        v_ord = np.asarray(r["valid"], dtype=bool) & v_ct
        stats = SSRX.summarize_map(r["prob"], v_ord, voxel_um)
        stats.update(
            seconds=r.get("seconds", 0.0),
            peak_allocated_mib=r.get("peak_allocated_mib", 0),
            peak_reserved_mib=r.get("peak_reserved_mib", 0),
        )
        detector_stats[f"studio_{order}"] = stats

    common_valid = (
        v_ct
        & np.asarray(res_studio["asc"]["valid"], dtype=bool)
        & np.asarray(res_studio["desc"]["valid"], dtype=bool)
    )
    tfce_full = TFCE.tfce_opposite_face_fwer(
        res_studio["asc"]["prob"],
        res_studio["desc"]["prob"],
        common_valid,
        voxel_um=voxel_um,
    )
    sw = _extract_stroke_width_metrics(tfce_full)
    q34 = tfce_full["q34_calibrated"]
    sb = tfce_full["stroke_bounded_tfce"]
    topo_comp = tfce_full["transverse_stroke_topology"]["composite"]
    betti_sec = tfce_full["transverse_stroke_topology"]["betti"]
    betti_dom = betti_sec.get("dominant", betti_sec.get(tfce_full["dominant_face"], {}))

    tfce_compact = {
        "dominant_face": tfce_full["dominant_face"],
        "opposite_face": tfce_full["opposite_face"],
        "max_tfce_ratio": tfce_full["max_tfce_ratio"],
        "p995_tfce_ratio": tfce_full["p995_tfce_ratio"],
        "diff_tfce_p99": tfce_full["diff_tfce_p99"],
        "sb_dom_max": sb["dom_max"],
        "sb_opp_max": sb["opp_max"],
        "sb_retention_max": sb["retention_max"],
        "sb_retention_p995": sb["retention_p995"],
        "empirical_fwer_p": tfce_full["empirical_null_fwer"]["empirical_fwer_p"],
        "width_median_mm": sw["width_median_mm"],
        "width_median_hi_mm": sw["width_median_hi_mm"],
        "width_median_lo_mm": sw["width_median_lo_mm"],
        "width_cv_hi": sw["width_cv_hi"],
        "narrow_stroke_frac_hi": sw["narrow_stroke_frac_hi"],
        "wide_blob_frac_hi": sw["wide_blob_frac_hi"],
        "max_b1": betti_dom.get("max_b1", 0),
        "single_face_score": topo_comp.get("single_face_score", 0.0),
        "stroke_consistency_score": topo_comp.get("stroke_consistency_score", 0.0),
        "topology_score": topo_comp.get("topology_score", 0.0),
        "q34_score": q34["calibrated_score"],
        "passes_tfce_asymmetry": q34["passes_tfce_asymmetry"],
        "passes_width_retention": q34["passes_width_retention"],
        "passes_stroke_morphology": q34["passes_stroke_morphology"],
        "passes_q34_bar": q34["passes_q34_bar"],
        "regime_label": q34["regime_label"],
    }

    agreement = None
    if ref_prob is not None and ref_valid is not None:
        target_face = ref_face if ref_face in ("asc", "desc") else tfce_full["dominant_face"]
        agreement = compute_map_agreement(
            res_studio[target_face]["prob"],
            np.asarray(res_studio[target_face]["valid"], dtype=bool) & v_ct,
            ref_prob,
            ref_valid,
        )

    return {
        "detectors": detector_stats,
        "tfce_q34": tfce_compact,
        "agreement_vs_ref": agreement,
    }


def write_comparison_montage(
    out_path: Path,
    title: str,
    ct_mid: np.ndarray,
    valid_ct: np.ndarray,
    variant_panels: list[tuple[str, np.ndarray, np.ndarray]],
    voxel_um: float,
) -> Path:
    """Write a labeled comparison montage PNG with a 1 cm scale bar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = ct_mid.shape
    target_w = min(max(w, 320), 1600)
    scale = target_w / max(w, 1)
    target_h = max(48, int(round(h * scale)))

    all_panels: list[tuple[str, np.ndarray]] = [
        ("CT mid (sheet plane)", SSRX._to_u8_stretch(ct_mid, valid_ct))
    ]
    for label, prob, val in variant_panels:
        u8 = np.where(val, np.clip(np.round(np.asarray(prob, dtype=np.float32) * 255.0), 0, 255), 0).astype(
            np.uint8
        )
        all_panels.append((label, u8))

    pad = 28
    canvas_h = len(all_panels) * (target_h + pad) + 32
    canvas = Image.new("L", (target_w, canvas_h), color=20)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=245)

    cm_px_native = 10000.0 / float(voxel_um)
    cm_px_scaled = int(round(cm_px_native * scale))
    for idx, (label, img_u8) in enumerate(all_panels):
        y0 = 24 + idx * (target_h + pad)
        draw.text((8, y0), label, fill=235)
        tile = Image.fromarray(img_u8, mode="L").resize((target_w, target_h), Image.Resampling.BILINEAR)
        canvas.paste(tile, (0, y0 + 20))
        if idx == 0 and 10 <= cm_px_scaled <= target_w - 20:
            by = y0 + 20 + target_h - 10
            draw.line([(16, by), (16 + cm_px_scaled, by)], fill=255, width=3)
            draw.text((16, by - 14), "1 cm", fill=255)

    canvas.save(out_path, format="PNG")
    return out_path


def assert_no_local_paths_in_public_json(payload: dict[str, Any]) -> None:
    """Verify a public-bound JSON payload contains zero local filesystem paths or tokens."""
    serialized = json.dumps(payload, indent=2)
    for pattern in Q22.FORBIDDEN_PATH_PATTERNS:
        m = pattern.search(serialized)
        if m:
            raise ValueError(f"Forbidden local path or token matched ({m.group(0)!r}) in public JSON payload")
    if "Scrollprize_private" in serialized:
        raise ValueError("Forbidden private directory name found in public JSON payload")


def _cleanup_ram_and_cuda() -> None:
    """Force Python GC and clear PyTorch CUDA allocator cache if loaded."""
    gc.collect()
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def _run_studio_variant_canonical(
    stack_lhw: np.ndarray,
    mesh_index: int,
    staged: dict[str, Path],
    *,
    spatial_mode: str,
    stride: int,
    batch_size: int,
    batch_pause_ms: int,
    max_gpu_temp_c: int,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]],
    guardrail_fn: Callable[..., dict[str, int] | None] | None,
) -> dict[str, dict[str, Any]]:
    """Run Studio on an optionally flipped stack and invert maps back to canonical coordinates."""
    if guardrail_fn is not None:
        guardrail_fn(max_temp_c=max_gpu_temp_c)
    t_stack = apply_spatial_transform(stack_lhw, spatial_mode) if spatial_mode != "orig" else stack_lhw
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
        canonical_res: dict[str, dict[str, Any]] = {}
        for order in ("desc", "asc"):
            r = raw_res[order]
            prob = invert_spatial_map(r["prob"], spatial_mode) if spatial_mode != "orig" else r["prob"]
            val = invert_spatial_map(r["valid"], spatial_mode) if spatial_mode != "orig" else r["valid"]
            canonical_res[order] = {
                "prob": prob,
                "valid": val,
                "seconds": r.get("seconds", 0.0),
                "peak_allocated_mib": r.get("peak_allocated_mib", 0),
                "peak_reserved_mib": r.get("peak_reserved_mib", 0),
            }
        return canonical_res
    finally:
        if spatial_mode != "orig":
            del t_stack
        _cleanup_ram_and_cuda()
        if guardrail_fn is not None:
            guardrail_fn(max_temp_c=max_gpu_temp_c)


def run_arm1_control(
    *,
    out_private_dir: Path,
    work_root: Path,
    staged: dict[str, Path],
    mesh_source: str = PHERC0306B_MESH_URL,
    ct_zarr_url: str = PHERC0306B_CT_ZARR_URL,
    uv_crop: tuple[int, int, int, int] | None = PHERC0306B_DEFAULT_UV_CROP,
    voxel_um: float = PHERC0306B_VOXEL_UM,
    num_slices: int = 31,
    studio_stride: int = 48,
    studio_batch: int = DEFAULT_STUDIO_BATCH,
    batch_pause_ms: int = DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = DEFAULT_CACHE_MB,
    vram_fraction: float = DEFAULT_VRAM_FRACTION,
    q22_audit_json: Path | None = None,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    guardrail_fn: Callable[..., dict[str, int] | None] | None = SSRX.enforce_gpu_guardrail,
) -> dict[str, Any]:
    """Execute Arm 1 on public PHerc. 0306B w074: orientation invariance + Q22 Nyul-Udupa rescue."""
    out_dir = SSRX.ensure_outside_git_repo(out_private_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_runner = run_studio_fn or SSRX.run_studio_both_orders
    pcts, ref_landmarks = load_q22_reference_landmarks(q22_audit_json)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="arm1-0306b-", dir=str(work_root), ignore_cleanup_errors=True) as mtmp:
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

        radiometry: dict[str, Any] = {
            "nyul_percentiles": list(pcts),
            "nyul_ref_landmarks": list(ref_landmarks),
            "orig": summarize_stack_radiometry(stack_lhw, valid_ct, pcts),
        }

        variants_out: dict[str, dict[str, Any]] = {}
        montage_panels: list[tuple[str, np.ndarray, np.ndarray]] = []

        # 1. Baseline orig
        res_orig = _run_studio_variant_canonical(
            stack_lhw,
            mesh_idx,
            staged,
            spatial_mode="orig",
            stride=studio_stride,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
            run_studio_fn=studio_runner,
            guardrail_fn=guardrail_fn,
        )
        m_orig = compute_variant_metrics(res_orig, valid_ct, voxel_um)
        ref_face = m_orig["tfce_q34"]["dominant_face"]
        ref_prob = res_orig[ref_face]["prob"].copy()
        ref_valid = (np.asarray(res_orig[ref_face]["valid"], dtype=bool) & valid_ct).copy()
        m_orig["agreement_vs_ref"] = compute_map_agreement(ref_prob, ref_valid, ref_prob, ref_valid)
        variants_out["orig"] = m_orig
        montage_panels.append(
            (
                f"orig ({ref_face} gt07={m_orig['detectors'][f'studio_{ref_face}']['frac_gt_07']:.4f} "
                f"q34={m_orig['tfce_q34']['q34_score']:.3f})",
                ref_prob,
                ref_valid,
            )
        )
        del res_orig

        # 2-4. Spatial orientations (villa #1909 mirror/rotation test)
        for smode in ("flip_lr", "flip_ud", "rot180"):
            res_s = _run_studio_variant_canonical(
                stack_lhw,
                mesh_idx,
                staged,
                spatial_mode=smode,
                stride=studio_stride,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                run_studio_fn=studio_runner,
                guardrail_fn=guardrail_fn,
            )
            m_s = compute_variant_metrics(
                res_s, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
            )
            variants_out[smode] = m_s
            montage_panels.append(
                (
                    f"{smode} ({ref_face} gt07={m_s['detectors'][f'studio_{ref_face}']['frac_gt_07']:.4f} "
                    f"IoU05={m_s['agreement_vs_ref']['iou_ge_05']:.3f})",
                    res_s[ref_face]["prob"],
                    np.asarray(res_s[ref_face]["valid"], dtype=bool) & valid_ct,
                )
            )
            del res_s

        # 5. Direct Nyul-Udupa no-harm check
        stack_nyul = apply_nyul_standardization_u8(stack_lhw, valid_ct, ref_landmarks, pcts)
        radiometry["nyul_udupa"] = summarize_stack_radiometry(stack_nyul, valid_ct, pcts)
        res_nyul = _run_studio_variant_canonical(
            stack_nyul,
            mesh_idx,
            staged,
            spatial_mode="orig",
            stride=studio_stride,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
            run_studio_fn=studio_runner,
            guardrail_fn=guardrail_fn,
        )
        m_nyul = compute_variant_metrics(
            res_nyul, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
        )
        variants_out["nyul_udupa"] = m_nyul
        montage_panels.append(
            (
                f"nyul_udupa ({ref_face} gt07={m_nyul['detectors'][f'studio_{ref_face}']['frac_gt_07']:.4f} "
                f"IoU05={m_nyul['agreement_vs_ref']['iou_ge_05']:.3f})",
                res_nyul[ref_face]["prob"],
                np.asarray(res_nyul[ref_face]["valid"], dtype=bool) & valid_ct,
            )
        )
        del stack_nyul, res_nyul
        _cleanup_ram_and_cuda()

        # 6. Synthetic +24 count shift (simulating PHerc. 0358 papyrus-wall shift)
        stack_p24 = apply_radiometric_shift(stack_lhw, valid_ct, delta_counts=24.0)
        radiometry["shift_plus24"] = summarize_stack_radiometry(stack_p24, valid_ct, pcts)
        res_p24 = _run_studio_variant_canonical(
            stack_p24,
            mesh_idx,
            staged,
            spatial_mode="orig",
            stride=studio_stride,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
            run_studio_fn=studio_runner,
            guardrail_fn=guardrail_fn,
        )
        m_p24 = compute_variant_metrics(
            res_p24, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
        )
        variants_out["shift_plus24"] = m_p24
        montage_panels.append(
            (
                f"shift_plus24 ({ref_face} gt07={m_p24['detectors'][f'studio_{ref_face}']['frac_gt_07']:.4f} "
                f"IoU05={m_p24['agreement_vs_ref']['iou_ge_05']:.3f})",
                res_p24[ref_face]["prob"],
                np.asarray(res_p24[ref_face]["valid"], dtype=bool) & valid_ct,
            )
        )
        del res_p24

        # 7. Synthetic +24 count shift rescued by Nyul-Udupa standardization
        stack_p24_nyul = apply_nyul_standardization_u8(stack_p24, valid_ct, ref_landmarks, pcts)
        del stack_p24
        radiometry["shift_plus24_nyul"] = summarize_stack_radiometry(stack_p24_nyul, valid_ct, pcts)
        res_p24_nyul = _run_studio_variant_canonical(
            stack_p24_nyul,
            mesh_idx,
            staged,
            spatial_mode="orig",
            stride=studio_stride,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
            run_studio_fn=studio_runner,
            guardrail_fn=guardrail_fn,
        )
        m_p24_nyul = compute_variant_metrics(
            res_p24_nyul, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
        )
        variants_out["shift_plus24_nyul"] = m_p24_nyul
        montage_panels.append(
            (
                f"shift_plus24_nyul ({ref_face} gt07={m_p24_nyul['detectors'][f'studio_{ref_face}']['frac_gt_07']:.4f} "
                f"IoU05={m_p24_nyul['agreement_vs_ref']['iou_ge_05']:.3f})",
                res_p24_nyul[ref_face]["prob"],
                np.asarray(res_p24_nyul[ref_face]["valid"], dtype=bool) & valid_ct,
            )
        )
        del stack_p24_nyul, res_p24_nyul

        ct_mid = stack_lhw[mesh_idx].copy()
        del stack_lhw
        _cleanup_ram_and_cuda()

    write_comparison_montage(
        out_dir / "arm1_pherc0306b_montage.png",
        "Exp T10-GPU Arm 1: PHerc0306B w074 Orientation (villa #1909) & Q22 Nyul-Udupa Control",
        ct_mid,
        valid_ct,
        montage_panels,
        voxel_um,
    )

    post_gpu = SSRX.query_gpu_telemetry()
    orig_gt07 = max(float(variants_out["orig"]["detectors"][f"studio_{ref_face}"]["frac_gt_07"]), 1e-9)
    p24_gt07 = float(variants_out["shift_plus24"]["detectors"][f"studio_{ref_face}"]["frac_gt_07"])
    p24_nyul_gt07 = float(variants_out["shift_plus24_nyul"]["detectors"][f"studio_{ref_face}"]["frac_gt_07"])

    payload: dict[str, Any] = {
        "schema_version": "gpu_orientation_and_nyul_control_v1",
        "experiment": "Exp T10-GPU Arm 1: PHerc0306B w074 Orientation Invariance (villa #1909) & Q22 Nyul-Udupa Control",
        "scroll": "PHerc0306B",
        "mesh_tag": "PHerc0306B_20250521133212_w074own_touchup_V5Rc",
        "uv_crop": list(uv_crop) if uv_crop is not None else None,
        "voxel_um": float(voxel_um),
        "num_slices": int(num_slices),
        "studio_stride": int(studio_stride),
        "valid_ct_pixels": int(valid_ct.sum()),
        "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
        "ct_chunks_fetched": int(rendered["ct_chunks"]),
        "ct_mib_fetched": round(float(rendered["ct_bytes"]) / (1024 ** 2), 2),
        "reference_reading_face": ref_face,
        "radiometry": radiometry,
        "variants": variants_out,
        "findings_summary": {
            "min_orientation_iou_ge_05": round(
                min(
                    float(variants_out[m]["agreement_vs_ref"]["iou_ge_05"])
                    for m in ("flip_lr", "flip_ud", "rot180")
                ),
                6,
            ),
            "min_orientation_pearson_r": round(
                min(
                    float(variants_out[m]["agreement_vs_ref"]["pearson_r"])
                    for m in ("flip_lr", "flip_ud", "rot180")
                ),
                6,
            ),
            "all_orientations_pass_q34_bar": bool(
                all(variants_out[m]["tfce_q34"]["passes_q34_bar"] for m in VALID_SPATIAL_MODES)
            ),
            "nyul_no_harm_iou_ge_05": variants_out["nyul_udupa"]["agreement_vs_ref"]["iou_ge_05"],
            "shift_plus24_frac_gt_07_ratio_vs_orig": round(p24_gt07 / orig_gt07, 6),
            "shift_plus24_nyul_frac_gt_07_ratio_vs_orig": round(p24_nyul_gt07 / orig_gt07, 6),
            "shift_plus24_nyul_iou_ge_05_vs_orig": variants_out["shift_plus24_nyul"]["agreement_vs_ref"]["iou_ge_05"],
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
    assert_no_local_paths_in_public_json(payload)
    json_path = out_dir / "arm1_pherc0306b_control_public.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_arm2_adversarial(
    *,
    out_private_dir: Path,
    work_root: Path,
    staged: dict[str, Path],
    mesh_source: str = PHERC0125_Z8736_MESH_URL,
    ct_zarr_url: str = PHERC0125_CT_ZARR_URL,
    m7_zarr_url: str | None = PHERC0125_M7_ZARR_URL,
    uv_crop: tuple[int, int, int, int] | None = PHERC0125_Z8736_DEFAULT_UV_CROP,
    voxel_um: float = PHERC0125_VOXEL_UM,
    snap_passes: int = 2,
    num_slices: int = 43,
    window_slices: int = 31,
    depth_shifts: Sequence[int] = (-6, -3, 0, 3, 6),
    depth_sweep_stride: int = 96,
    fine_stride: int = 48,
    studio_batch: int = DEFAULT_STUDIO_BATCH,
    batch_pause_ms: int = DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = DEFAULT_CACHE_MB,
    vram_fraction: float = DEFAULT_VRAM_FRACTION,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    guardrail_fn: Callable[..., dict[str, int] | None] | None = SSRX.enforce_gpu_guardrail,
) -> dict[str, Any]:
    """Execute Arm 2 private adversarial depth/orientation/TFCE check on PHerc0125/z8736_w035."""
    base_out = SSRX.ensure_outside_git_repo(out_private_dir)
    adv_dir = base_out / "pherc0125_z8736_adversarial_check"
    adv_dir.mkdir(parents=True, exist_ok=True)
    studio_runner = run_studio_fn or SSRX.run_studio_both_orders

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="arm2-z8736-", dir=str(work_root), ignore_cleanup_errors=True) as mtmp:
        rendered = SSRX.render_mesh_ct_stack(
            mesh_source,
            Path(mtmp),
            ct_zarr_url=ct_zarr_url,
            m7_zarr_url=m7_zarr_url,
            snap_passes=snap_passes,
            num_slices=num_slices,
            uv_crop=uv_crop,
            cache_mb=cache_mb,
        )
        stack_43 = rendered.pop("stack_lhw")
        valid_ct = rendered.pop("valid_ct")
        base_mesh_idx = (num_slices - 1) // 2

        variants_out: dict[str, dict[str, Any]] = {}
        montage_panels: list[tuple[str, np.ndarray, np.ndarray]] = []

        # Part A: Fine stride=48 orientation sweep at dz=0 first so dz0_orig_s48 serves as reference
        sub_dz0, win_mid = extract_depth_window(stack_43, base_mesh_idx, dz=0, window_slices=window_slices)
        ref_prob = None
        ref_valid = None
        ref_face = "desc"

        for smode in VALID_SPATIAL_MODES:
            vkey = f"dz0_{smode}_s{fine_stride}"
            res_s = _run_studio_variant_canonical(
                sub_dz0,
                win_mid,
                staged,
                spatial_mode=smode,
                stride=fine_stride,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                run_studio_fn=studio_runner,
                guardrail_fn=guardrail_fn,
            )
            if smode == "orig":
                m_s = compute_variant_metrics(res_s, valid_ct, voxel_um)
                ref_face = m_s["tfce_q34"]["dominant_face"]
                ref_prob = res_s[ref_face]["prob"].copy()
                ref_valid = (np.asarray(res_s[ref_face]["valid"], dtype=bool) & valid_ct).copy()
                m_s["agreement_vs_ref"] = compute_map_agreement(ref_prob, ref_valid, ref_prob, ref_valid)
            else:
                m_s = compute_variant_metrics(
                    res_s, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
                )
            m_s["dz"] = 0
            m_s["spatial_mode"] = smode
            m_s["studio_stride"] = int(fine_stride)
            variants_out[vkey] = m_s
            dom_f = m_s["tfce_q34"]["dominant_face"]
            w_mm = m_s["tfce_q34"]["width_median_mm"]
            w_str = f"{w_mm:.3f}mm" if isinstance(w_mm, (int, float)) else "n/a"
            montage_panels.append(
                (
                    f"{vkey} ({dom_f} gt07={m_s['detectors'][f'studio_{dom_f}']['frac_gt_07']:.4f} "
                    f"w={w_str} q34={m_s['tfce_q34']['q34_score']:.3f})",
                    res_s[dom_f]["prob"],
                    np.asarray(res_s[dom_f]["valid"], dtype=bool) & valid_ct,
                )
            )
            del res_s
        del sub_dz0

        # Part B: Depth sweep dz in depth_shifts at depth_sweep_stride
        for dz in depth_shifts:
            sign_tag = f"p{dz}" if dz > 0 else (f"m{abs(dz)}" if dz < 0 else "0")
            vkey = f"dz_{sign_tag}_s{depth_sweep_stride}"
            sub_dz, sub_mid = extract_depth_window(stack_43, base_mesh_idx, dz=int(dz), window_slices=window_slices)
            res_d = _run_studio_variant_canonical(
                sub_dz,
                sub_mid,
                staged,
                spatial_mode="orig",
                stride=depth_sweep_stride,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                run_studio_fn=studio_runner,
                guardrail_fn=guardrail_fn,
            )
            m_d = compute_variant_metrics(
                res_d, valid_ct, voxel_um, ref_prob=ref_prob, ref_valid=ref_valid, ref_face=ref_face
            )
            m_d["dz"] = int(dz)
            m_d["spatial_mode"] = "orig"
            m_d["studio_stride"] = int(depth_sweep_stride)
            variants_out[vkey] = m_d
            dom_f = m_d["tfce_q34"]["dominant_face"]
            w_mm = m_d["tfce_q34"]["width_median_mm"]
            w_str = f"{w_mm:.3f}mm" if isinstance(w_mm, (int, float)) else "n/a"
            montage_panels.append(
                (
                    f"{vkey} ({dom_f} gt07={m_d['detectors'][f'studio_{dom_f}']['frac_gt_07']:.4f} "
                    f"w={w_str} q34={m_d['tfce_q34']['q34_score']:.3f})",
                    res_d[dom_f]["prob"],
                    np.asarray(res_d[dom_f]["valid"], dtype=bool) & valid_ct,
                )
            )
            del sub_dz, res_d

        ct_mid = stack_43[base_mesh_idx].copy()
        del stack_43
        _cleanup_ram_and_cuda()

    write_comparison_montage(
        adv_dir / "pherc0125_z8736_adversarial_montage.png",
        "Exp T10-GPU Arm 2: PHerc0125/z8736_w035 Depth Sweep & Orientation Adversarial Verification",
        ct_mid,
        valid_ct,
        montage_panels,
        voxel_um,
    )

    stroke_resolved_variants = [
        k
        for k, v in variants_out.items()
        if v["tfce_q34"]["passes_q34_bar"]
        and isinstance(v["tfce_q34"]["width_median_mm"], (int, float))
        and 0.20 <= float(v["tfce_q34"]["width_median_mm"]) <= 0.75
    ]
    best_q34_key = max(variants_out, key=lambda k: float(variants_out[k]["tfce_q34"]["q34_score"]))
    post_gpu = SSRX.query_gpu_telemetry()

    summary: dict[str, Any] = {
        "schema_version": "gpu_adversarial_verification_v1",
        "experiment": "Exp T10-GPU Arm 2: PHerc0125/z8736_w035 Adversarial Depth Sweep, Orientation & Q34 TFCE Check",
        "scroll": "PHerc0125",
        "mesh_tag": "z8736_w035",
        "uv_crop": list(uv_crop) if uv_crop is not None else None,
        "voxel_um": float(voxel_um),
        "num_slices_streamed": int(num_slices),
        "window_slices": int(window_slices),
        "snap_passes": int(snap_passes),
        "geometry_snap_metrics": rendered["geom_metrics"],
        "valid_ct_pixels": int(valid_ct.sum()),
        "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
        "m7_chunks_fetched": int(rendered["m7_chunks"]),
        "m7_mib_fetched": round(float(rendered["m7_bytes"]) / (1024 ** 2), 2),
        "ct_chunks_fetched": int(rendered["ct_chunks"]),
        "ct_mib_fetched": round(float(rendered["ct_bytes"]) / (1024 ** 2), 2),
        "variants": variants_out,
        "adversarial_assessment": {
            "best_q34_variant": best_q34_key,
            "best_q34_score": variants_out[best_q34_key]["tfce_q34"]["q34_score"],
            "best_q34_width_median_mm": variants_out[best_q34_key]["tfce_q34"]["width_median_mm"],
            "best_q34_regime_label": variants_out[best_q34_key]["tfce_q34"]["regime_label"],
            "stroke_resolved_variants": stroke_resolved_variants,
            "adversarial_verdict": (
                "CANDIDATE_STROKE_RESOLVED"
                if stroke_resolved_variants
                else "ARTIFACT_OR_WIDE_BLOB_CONFIRMED"
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
    (adv_dir / "pherc0125_z8736_adversarial_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser with strict RTX 4060 Laptop VRAM and thermal guardrails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("control", "adversarial", "both"), default="both")
    ap.add_argument(
        "--out-private-dir",
        type=Path,
        required=True,
        help="Private output directory strictly outside the public git repository",
    )
    ap.add_argument(
        "--vram-fraction",
        type=float,
        default=DEFAULT_VRAM_FRACTION,
        help=f"Per-process CUDA VRAM cap fraction (default {DEFAULT_VRAM_FRACTION}, max {MAX_SAFE_VRAM_FRACTION})",
    )
    ap.add_argument("--studio-batch", type=int, default=DEFAULT_STUDIO_BATCH)
    ap.add_argument("--batch-pause-ms", type=int, default=DEFAULT_BATCH_PAUSE_MS)
    ap.add_argument("--max-gpu-temp-c", type=int, default=DEFAULT_MAX_GPU_TEMP_C)
    ap.add_argument("--cache-mb", type=int, default=DEFAULT_CACHE_MB)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (0.05 <= float(args.vram_fraction) <= MAX_SAFE_VRAM_FRACTION):
        raise ValueError(
            f"--vram-fraction must be within [0.05, {MAX_SAFE_VRAM_FRACTION}], got {args.vram_fraction}"
        )
    out_dir = SSRX.ensure_outside_git_repo(args.out_private_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SSRX.enforce_gpu_guardrail(max_temp_c=int(args.max_gpu_temp_c))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_transverse_interventions.py")
    torch.cuda.set_per_process_memory_fraction(float(args.vram_fraction), 0)

    with tempfile.TemporaryDirectory(
        prefix="rtx4060-interventions-work-", ignore_cleanup_errors=True
    ) as tmp:
        work_root = Path(tmp)
        staged = SSRX.stage_models(work_root, {"studio"})

        if args.arm in ("control", "both"):
            arm1 = run_arm1_control(
                out_private_dir=out_dir,
                work_root=work_root,
                staged=staged,
                studio_batch=args.studio_batch,
                batch_pause_ms=args.batch_pause_ms,
                max_gpu_temp_c=args.max_gpu_temp_c,
                cache_mb=args.cache_mb,
                vram_fraction=args.vram_fraction,
            )
            print(
                f"COMPLETED Arm 1 (PHerc0306B control) in {arm1['safety_profile']['total_seconds']}s "
                f"(min_orientation_IoU05={arm1['findings_summary']['min_orientation_iou_ge_05']})",
                flush=True,
            )

        if args.arm in ("adversarial", "both"):
            arm2 = run_arm2_adversarial(
                out_private_dir=out_dir,
                work_root=work_root,
                staged=staged,
                studio_batch=args.studio_batch,
                batch_pause_ms=args.batch_pause_ms,
                max_gpu_temp_c=args.max_gpu_temp_c,
                cache_mb=args.cache_mb,
                vram_fraction=args.vram_fraction,
            )
            print(
                f"COMPLETED Arm 2 (PHerc0125/z8736_w035 adversarial) in {arm2['safety_profile']['total_seconds']}s "
                f"-> verdict={arm2['adversarial_assessment']['adversarial_verdict']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
