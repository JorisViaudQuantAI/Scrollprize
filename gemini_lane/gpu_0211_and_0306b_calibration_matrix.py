# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T17-GPU: PHerc. 0211 `snap x stride x family x order` Matrix & PHerc. 0306B Layer-Order Card.

Addresses the two open calibration gates from ``collab/claude-020.md`` and ``claude-022.md``
under strict RTX 4060 Laptop VRAM/thermal guardrails and zero persistent CT storage:

1. Arm 1 (`pherc0211` — ``collab/claude-020.md`` item 2):
   Runs the exact screening vs gate matrix on the public ``PHerc. 0211`` First Letters
   winner crop (``uv_crop=12:76,184:214``, ``0.6421 cm^2``, 3-line region):
   ``snap in {0, 2}`` (with pinned S3 ``m7-L0``) x ``Studio fp16 stride in {96, 48}``
   + ``A-hn hn4_s0`` x ``order in {asc, desc}``, evaluated with both legacy ``Exp T6`` Q34
   and upgraded ``Exp T13`` Area-Matched Rim-Eroded Cloud-Masked Q34 + ``screening_fp_filter``.

2. Arm 2 (`pherc0306b` — ``collab/claude-020.md`` item 5):
   Runs ``Studio fp16 (stride 48 & 96)`` and ``A-hn hn4_s0`` in both ``asc`` and ``desc``
   orders on the public ``PHerc. 0306B w074`` gate crop (``uv_crop=95:120,190:250``,
   ``8.64 um``), exporting an exact **Layer-Order Orientation & Per-Layer Profile Card**
   so cross-lane renderers can verify layer-index parity without exchanging raw CT volumes.

All visual montages go strictly outside git (``ensure_outside_git_repo``), and only the
path-sanitized public JSON receipt is written inside ``gemini_lane/evidence/``.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpu_shear_and_0211_control as G16  # noqa: E402
import gpu_transverse_interventions as GTI  # noqa: E402
import screen_scroll_rtx4060 as SSRX  # noqa: E402
import stream_snap_render as SSR  # noqa: E402
import tfce_stroke_calibration as TFCE  # noqa: E402
import tiled_q34_cloud_masked_ranker as T13  # noqa: E402

PHERC0211_M7_ZARR_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0211/representations/predictions/surfaces/"
    "20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
DEFAULT_PUBLIC_RECEIPT = HERE / "evidence" / "gpu_0211_and_0306b_calibration_matrix.json"
DEFAULT_VRAM_FRACTION: float = 0.50
DEFAULT_STUDIO_BATCH: int = 2
DEFAULT_AHN_BATCH: int = 2
DEFAULT_BATCH_PAUSE_MS: int = 300
DEFAULT_MAX_GPU_TEMP_C: int = 74
DEFAULT_CACHE_MB: int = 320


def _score_onsheet_coherence(
    stack_lhw: np.ndarray,
    mesh_idx: int,
    valid_ct: np.ndarray,
) -> dict[str, Any]:
    """Compute ``onsheet_score`` stack coherence with ``stream_snap_render`` fallback."""
    try:
        import onsheet_score  # type: ignore[import-not-found]

        return onsheet_score.score_stack(stack_lhw, mid=mesh_idx, dk=2)
    except ModuleNotFoundError:
        fb = SSR.layer_coherence_score(
            stack_lhw,
            mesh_index=mesh_idx,
            delta=min(2, mesh_idx),
            valid_mask=valid_ct,
        )
        return {
            "f_parallel": fb["sheet_parallel_fraction"],
            "n_tiles": fb["patch_count"],
            "mean_zero_shift_r": fb["mean_zero_shift_r"],
            "mean_best_shifted_r": fb["mean_best_shifted_r"],
            "backend": "stream_snap_render_fallback",
        }


def compute_layer_order_profile_card(
    stack_lhw: np.ndarray,
    valid_ct: np.ndarray,
    *,
    mesh_idx: int = 15,
    slice_step_vx: float = 1.0,
) -> dict[str, Any]:
    """Export exact detector slice indices and per-layer CT profile statistics for parity checks."""
    arr = np.asarray(stack_lhw)
    valid = np.asarray(valid_ct, dtype=bool)
    if arr.ndim != 3 or valid.shape != arr.shape[1:]:
        raise ValueError(f"Expected stack (L,H,W) and valid (H,W), got {arr.shape} vs {valid.shape}")
    n_layers = int(arr.shape[0])
    if not (0 <= int(mesh_idx) < n_layers):
        raise ValueError(f"mesh_idx {mesh_idx} out of range 0..{n_layers - 1}")

    center = arr[int(mesh_idx)][valid].astype(np.float64) if valid.any() else np.zeros(0, dtype=np.float64)
    center_std = float(np.std(center)) if center.size > 0 else 0.0

    def _corr(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
        if a_vec.size == 0 or b_vec.size == 0:
            return 0.0
        sa = float(np.std(a_vec))
        sb = float(np.std(b_vec))
        if sa <= 1e-12 or sb <= 1e-12:
            mae = float(np.mean(np.abs(a_vec - b_vec)))
            return 1.0 if mae < 1e-9 else 0.0
        return float(np.corrcoef(a_vec, b_vec)[0, 1])

    per_layer: list[dict[str, Any]] = []
    for k in range(n_layers):
        vals = arr[k][valid].astype(np.float64) if valid.any() else np.zeros(0, dtype=np.float64)
        rev_k = n_layers - 1 - k
        rev_vals = arr[rev_k][valid].astype(np.float64) if valid.any() else np.zeros(0, dtype=np.float64)
        per_layer.append(
            {
                "layer_index": int(k),
                "normal_offset_vx": round((k - int(mesh_idx)) * float(slice_step_vx), 3),
                "valid_mean": round(float(vals.mean()), 4) if vals.size else 0.0,
                "valid_std": round(float(vals.std()), 4) if vals.size else 0.0,
                "valid_p50": round(float(np.percentile(vals, 50.0)), 3) if vals.size else 0.0,
                "pearson_r_vs_center_layer": round(_corr(vals, center), 6) if center_std > 0 else 1.0,
                "pearson_r_vs_reversed_layer": round(_corr(vals, rev_vals), 6),
            }
        )

    pos8_idx = min(n_layers - 1, int(mesh_idx) + 8)
    neg8_idx = max(0, int(mesh_idx) - 8)
    pos8_r = per_layer[pos8_idx]["pearson_r_vs_center_layer"]
    neg8_r = per_layer[neg8_idx]["pearson_r_vs_center_layer"]

    return {
        "num_layers": n_layers,
        "mesh_center_index": int(mesh_idx),
        "slice_step_vx": float(slice_step_vx),
        "detector_slice_conventions": {
            "stack_axis0_meaning": (
                "stack_lhw[k] samples (surface_xyz + (k - mesh_idx) * unit_normal) via "
                "vertex-aligned grid p * scale (no +0.5/20 offset)."
            ),
            "studio_desc_indices": list(range(int(mesh_idx) + 8, int(mesh_idx) - 8, -1))
            if n_layers >= 24 and int(mesh_idx) >= 8
            else [],
            "studio_asc_indices": list(range(int(mesh_idx) - 8, int(mesh_idx) + 8))
            if n_layers >= 24 and int(mesh_idx) >= 8
            else [],
            "ahn_s0_asc_indices": list(range(max(0, int(mesh_idx) - 12), min(n_layers, int(mesh_idx) + 12))),
            "ahn_s0_desc_indices": list(
                reversed(range(max(0, int(mesh_idx) - 12), min(n_layers, int(mesh_idx) + 12)))
            ),
        },
        "center_symmetry_summary": {
            "plus8_layer": int(pos8_idx),
            "minus8_layer": int(neg8_idx),
            "plus8_r_vs_center": pos8_r,
            "minus8_r_vs_center": neg8_r,
            "plus8_minus8_cross_r": per_layer[pos8_idx]["pearson_r_vs_reversed_layer"]
            if pos8_idx == n_layers - 1 - neg8_idx
            else None,
            "mean_diff_plus8_minus_minus8": round(
                per_layer[pos8_idx]["valid_mean"] - per_layer[neg8_idx]["valid_mean"], 4
            ),
        },
        "per_layer_profiles": per_layer,
    }


def _compact_legacy_q34(tfce_full: dict[str, Any]) -> dict[str, Any]:
    """Extract compact summary fields from ``tfce_stroke_calibration.tfce_opposite_face_fwer``."""
    sw = GTI._extract_stroke_width_metrics(tfce_full)
    q34 = tfce_full["q34_calibrated"]
    sb = tfce_full["stroke_bounded_tfce"]
    topo_comp = tfce_full["transverse_stroke_topology"]["composite"]
    betti_sec = tfce_full["transverse_stroke_topology"]["betti"]
    betti_dom = betti_sec.get("dominant", betti_sec.get(tfce_full["dominant_face"], {}))
    return {
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
        "max_b1": betti_dom.get("max_b1", 0),
        "single_face_score": topo_comp.get("single_face_score", 0.0),
        "q34_score": q34["calibrated_score"],
        "passes_tfce_asymmetry": q34["passes_tfce_asymmetry"],
        "passes_width_retention": q34["passes_width_retention"],
        "passes_stroke_morphology": q34["passes_stroke_morphology"],
        "passes_q34_bar": q34["passes_q34_bar"],
        "regime_label": q34["regime_label"],
    }


def _compact_upgraded_q34(upg: dict[str, Any]) -> dict[str, Any]:
    """Extract compact summary fields from ``tiled_q34_cloud_masked_ranker``."""
    return {
        "dominant_face": upg["dominant_face"],
        "opposite_face": upg["opposite_face"],
        "area_matched_retention_max": upg["area_matched_retention_max"],
        "area_matched_retention_p995": upg["area_matched_retention_p995"],
        "legacy_retention_max": upg.get("legacy_retention_max"),
        "cloud_masked_max_tfce_ratio": upg["cloud_masked_max_tfce_ratio"],
        "cloud_masked_p995_tfce_ratio": upg["cloud_masked_p995_tfce_ratio"],
        "diff_tfce_p99": upg["diff_tfce_p99"],
        "diff_tfce_p995": upg["diff_tfce_p995"],
        "diff_tfce_asymmetry_ratio": upg["diff_tfce_asymmetry_ratio"],
        "top_tile_q34_score": upg["top_tile_q34_score"],
        "top_tile_passes_q34_bar": upg["top_tile_passes_q34_bar"],
        "global_upgraded_passes_q34_bar": upg["global_upgraded_passes_q34_bar"],
        "upgraded_q34_score": upg["upgraded_q34_score"],
        "upgraded_regime_label": upg["upgraded_regime_label"],
        "upgraded_q34_gates": upg.get("upgraded_q34_gates"),
        "rim_erosion_diagnostic": upg.get("rim_erosion_diagnostic"),
        "pipeline_b_cloud_masked_mad": upg.get("pipeline_b_cloud_masked_mad"),
    }


def score_detector_pair(
    res_pair: dict[str, dict[str, Any]],
    *,
    family: str,
    stack_mid: np.ndarray,
    valid_ct: np.ndarray,
    voxel_um: float,
    paired_ahn_res: dict[str, dict[str, Any]] | None = None,
    ref_prob: np.ndarray | None = None,
    ref_valid: np.ndarray | None = None,
    ref_face: str | None = None,
) -> dict[str, Any]:
    """Evaluate a two-order `(asc, desc)` map pair on legacy Q34, upgraded Exp T13 Q34, and FP filters."""
    v_ct = np.asarray(valid_ct, dtype=bool)
    prefix = "studio" if family == "studio" else "ahn_s0"
    orders: dict[str, dict[str, Any]] = {}
    for order in ("desc", "asc"):
        r = res_pair[order]
        v_ord = np.asarray(r["valid"], dtype=bool) & v_ct
        st = SSRX.summarize_map(r["prob"], v_ord, voxel_um)
        st.update(
            seconds=r.get("seconds", 0.0),
            peak_allocated_mib=r.get("peak_allocated_mib", 0),
            peak_reserved_mib=r.get("peak_reserved_mib", 0),
        )
        if "tiles" in r:
            st["tiles"] = r["tiles"]
        orders[f"{prefix}_{order}"] = st

    common_valid = (
        v_ct
        & np.asarray(res_pair["asc"]["valid"], dtype=bool)
        & np.asarray(res_pair["desc"]["valid"], dtype=bool)
    )
    asc_prob = np.asarray(res_pair["asc"]["prob"], dtype=np.float32)
    desc_prob = np.asarray(res_pair["desc"]["prob"], dtype=np.float32)

    leg_full = TFCE.tfce_opposite_face_fwer(asc_prob, desc_prob, common_valid, voxel_um=voxel_um)
    leg_q34 = _compact_legacy_q34(leg_full)
    upg_full = T13.evaluate_upgraded_q34_and_q31_ranker(
        asc_prob,
        desc_prob,
        common_valid,
        voxel_um=voxel_um,
    )
    upg_q34 = _compact_upgraded_q34(upg_full)
    fp_diag = SSRX.compute_fp_diagnostics(stack_mid, v_ct, res_pair, paired_ahn_res, voxel_um)

    dom_face = upg_q34["dominant_face"]
    opp_face = upg_q34["opposite_face"]
    dom_stats = orders[f"{prefix}_{dom_face}"]
    opp_stats = orders[f"{prefix}_{opp_face}"]
    dom_over_opp = round(float(dom_stats["frac_gt_07"]) / max(float(opp_stats["frac_gt_07"]), 1e-6), 4)

    agreement = None
    if ref_prob is not None and ref_valid is not None:
        cmp_face = ref_face if ref_face in ("asc", "desc") else dom_face
        agreement = GTI.compute_map_agreement(
            res_pair[cmp_face]["prob"],
            np.asarray(res_pair[cmp_face]["valid"], dtype=bool) & v_ct,
            ref_prob,
            ref_valid,
        )

    strokeness_mean = float((fp_diag.get("strokeness") or {}).get("weighted_mean", 0.0) or 0.0)
    recovers_signal = bool(
        dom_over_opp >= 3.0
        and float(dom_stats["frac_gt_07"]) >= 0.03
        and (
            bool(upg_q34["global_upgraded_passes_q34_bar"])
            or bool(upg_q34["top_tile_passes_q34_bar"])
            or bool(fp_diag["ink_like_candidate"])
            or (strokeness_mean >= 1.5 and float(upg_q34["diff_tfce_p99"]) >= 4.0)
        )
    )

    return {
        "family": family,
        "orders": orders,
        "dominant_face": dom_face,
        "opposite_face": opp_face,
        "dom_frac_gt_07": dom_stats["frac_gt_07"],
        "opp_frac_gt_07": opp_stats["frac_gt_07"],
        "dom_over_opp_gt_07_ratio": dom_over_opp,
        "legacy_exp_t6_q34": leg_q34,
        "upgraded_exp_t13_q34": upg_q34,
        "fp_diagnostics": fp_diag,
        "agreement_vs_canonical": agreement,
        "recovers_signal": recovers_signal,
    }


def run_arm1_pherc0211_matrix(
    *,
    out_private_dir: Path,
    work_root: Path,
    staged: dict[str, Path],
    mesh_source: str = G16.PHERC0211_MESH_URL,
    ct_zarr_url: str = G16.PHERC0211_CT_ZARR_URL,
    m7_zarr_url: str = PHERC0211_M7_ZARR_URL,
    uv_crop: tuple[int, int, int, int] | None = G16.PHERC0211_DEFAULT_UV_CROP,
    voxel_um: float = G16.PHERC0211_VOXEL_UM,
    num_slices: int = 31,
    studio_batch: int = DEFAULT_STUDIO_BATCH,
    ahn_batch: int = DEFAULT_AHN_BATCH,
    batch_pause_ms: int = DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = DEFAULT_CACHE_MB,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    run_ahn_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    guardrail_fn: Callable[..., dict[str, int] | None] | None = SSRX.enforce_gpu_guardrail,
) -> dict[str, Any]:
    """Run ``snap in {0, 2}`` x ``Studio {s96, s48} + A-hn`` x ``{asc, desc}`` on ``PHerc. 0211``."""
    base_out = SSRX.ensure_outside_git_repo(out_private_dir)
    out_dir = base_out / "gpu_0211_0306b_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_runner = run_studio_fn or SSRX.run_studio_both_orders
    ahn_runner = run_ahn_fn or SSRX.run_ahn_both_orders

    t0 = time.perf_counter()
    mesh_idx = (num_slices - 1) // 2

    snap0_stack_ref: np.ndarray | None = None
    snap0_valid_ref: np.ndarray | None = None
    canon_prob: np.ndarray | None = None
    canon_valid: np.ndarray | None = None
    canon_face: str | None = None
    canon_ct_mid: np.ndarray | None = None
    canon_ct_valid: np.ndarray | None = None

    montage_panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    snap_runs: dict[str, dict[str, Any]] = {}
    total_ct_bytes = 0
    total_m7_bytes = 0

    for snap_passes in (0, 2):
        snap_key = f"snap{snap_passes}"
        if guardrail_fn is not None:
            guardrail_fn(max_temp_c=max_gpu_temp_c)

        with tempfile.TemporaryDirectory(prefix=f"arm1-0211-{snap_key}-", dir=str(work_root), ignore_cleanup_errors=True) as mtmp:
            rendered = SSRX.render_mesh_ct_stack(
                mesh_source,
                Path(mtmp),
                ct_zarr_url=ct_zarr_url,
                m7_zarr_url=m7_zarr_url if snap_passes > 0 else None,
                snap_passes=snap_passes,
                num_slices=num_slices,
                uv_crop=uv_crop,
                cache_mb=cache_mb,
            )
            stack_lhw = rendered.pop("stack_lhw")
            valid_ct = rendered.pop("valid_ct")
            total_ct_bytes += int(rendered["ct_bytes"])
            total_m7_bytes += int(rendered["m7_bytes"])
            coherence = _score_onsheet_coherence(stack_lhw, mesh_idx, valid_ct)

            ct_vs_snap0 = None
            if snap_passes == 0:
                snap0_stack_ref = stack_lhw[mesh_idx].copy()
                snap0_valid_ref = valid_ct.copy()
                canon_ct_mid = stack_lhw[mesh_idx].copy()
                canon_ct_valid = valid_ct.copy()
            elif snap0_stack_ref is not None and snap0_valid_ref is not None:
                common_ct = valid_ct & snap0_valid_ref
                if common_ct.any():
                    a = stack_lhw[mesh_idx][common_ct].astype(np.float64)
                    b = snap0_stack_ref[common_ct].astype(np.float64)
                    ct_vs_snap0 = {
                        "common_valid_px": int(common_ct.sum()),
                        "mid_plane_pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 6)
                        if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12
                        else 1.0,
                        "mid_plane_mae_counts": round(float(np.mean(np.abs(a - b))), 4),
                    }

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

            if guardrail_fn is not None:
                guardrail_fn(max_temp_c=max_gpu_temp_c)
            res_s48 = studio_runner(
                stack_lhw,
                mesh_idx,
                staged,
                stride=48,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
            )

            if guardrail_fn is not None:
                guardrail_fn(max_temp_c=max_gpu_temp_c)
            res_s96 = studio_runner(
                stack_lhw,
                mesh_idx,
                staged,
                stride=96,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
            )

            # Set canonical reference from snap0 + studio s48 dominant face
            if snap_passes == 0:
                s48_entry = score_detector_pair(
                    res_s48,
                    family="studio",
                    stack_mid=stack_lhw[mesh_idx],
                    valid_ct=valid_ct,
                    voxel_um=voxel_um,
                    paired_ahn_res=res_ahn,
                )
                canon_face = s48_entry["dominant_face"]
                canon_prob = np.asarray(res_s48[canon_face]["prob"], dtype=np.float32).copy()
                canon_valid = (np.asarray(res_s48[canon_face]["valid"], dtype=bool) & valid_ct).copy()
                s48_entry["agreement_vs_canonical"] = GTI.compute_map_agreement(
                    canon_prob, canon_valid, canon_prob, canon_valid
                )
            else:
                s48_entry = score_detector_pair(
                    res_s48,
                    family="studio",
                    stack_mid=stack_lhw[mesh_idx],
                    valid_ct=valid_ct,
                    voxel_um=voxel_um,
                    paired_ahn_res=res_ahn,
                    ref_prob=canon_prob,
                    ref_valid=canon_valid,
                    ref_face=canon_face,
                )

            s96_entry = score_detector_pair(
                res_s96,
                family="studio",
                stack_mid=stack_lhw[mesh_idx],
                valid_ct=valid_ct,
                voxel_um=voxel_um,
                paired_ahn_res=res_ahn,
                ref_prob=canon_prob,
                ref_valid=canon_valid,
                ref_face=canon_face,
            )
            ahn_entry = score_detector_pair(
                res_ahn,
                family="ahn",
                stack_mid=stack_lhw[mesh_idx],
                valid_ct=valid_ct,
                voxel_um=voxel_um,
                ref_prob=canon_prob,
                ref_valid=canon_valid,
                ref_face=canon_face,
            )

            for tag, entry, res_map in (
                ("studio_s48", s48_entry, res_s48),
                ("studio_s96", s96_entry, res_s96),
                ("ahn_s0", ahn_entry, res_ahn),
            ):
                df = entry["dominant_face"]
                montage_panels.append(
                    (
                        f"{snap_key} {tag} {df} (gt07={entry['dom_frac_gt_07']:.4f} "
                        f"ratio={entry['dom_over_opp_gt_07_ratio']:.2f}x upg={entry['upgraded_exp_t13_q34']['upgraded_q34_score']:.3f})",
                        res_map[df]["prob"],
                        np.asarray(res_map[df]["valid"], dtype=bool) & valid_ct,
                    )
                )

            snap_runs[snap_key] = {
                "snap_passes": int(snap_passes),
                "geometry_snap_metrics": rendered["geom_metrics"],
                "onsheet_coherence": coherence,
                "valid_ct_pixels": int(valid_ct.sum()),
                "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
                "ct_chunks_fetched": int(rendered["ct_chunks"]),
                "ct_mib_fetched": round(float(rendered["ct_bytes"]) / (1024 ** 2), 4),
                "m7_chunks_fetched": int(rendered["m7_chunks"]),
                "m7_mib_fetched": round(float(rendered["m7_bytes"]) / (1024 ** 2), 4),
                "render_seconds": rendered["render_sec"],
                "ct_mid_vs_snap0": ct_vs_snap0,
                "studio_s48": s48_entry,
                "studio_s96": s96_entry,
                "ahn_s0": ahn_entry,
            }
            del stack_lhw, res_ahn, res_s48, res_s96
            GTI._cleanup_ram_and_cuda()

    del snap0_stack_ref, snap0_valid_ref
    if canon_ct_mid is not None and canon_ct_valid is not None:
        GTI.write_comparison_montage(
            out_dir / "arm1_pherc0211_calibration_montage.png",
            "Exp T17-GPU Arm 1: PHerc. 0211 Winner snap{0,2} x Studio{s48,s96} + A-hn Matrix",
            canon_ct_mid,
            canon_ct_valid,
            montage_panels,
            voxel_um,
        )

    s0_48 = snap_runs["snap0"]["studio_s48"]
    s0_96 = snap_runs["snap0"]["studio_s96"]
    s2_48 = snap_runs["snap2"]["studio_s48"]
    s2_96 = snap_runs["snap2"]["studio_s96"]
    ahn0 = snap_runs["snap0"]["ahn_s0"]
    ahn2 = snap_runs["snap2"]["ahn_s0"]
    s2_96_agr = s2_96["agreement_vs_canonical"] or {}

    verdict = {
        "canonical_reading_face": canon_face,
        "snap0_s48_recovers_winner": bool(s0_48["recovers_signal"]),
        "snap0_s96_recovers_winner": bool(s0_96["recovers_signal"]),
        "snap2_s48_recovers_winner": bool(s2_48["recovers_signal"]),
        "snap2_s96_recovers_winner": bool(s2_96["recovers_signal"]),
        "ahn_snap0_recovers_winner": bool(ahn0["recovers_signal"]),
        "ahn_snap2_recovers_winner": bool(ahn2["recovers_signal"]),
        "upgraded_exp_t13_passes_snap0_s48": bool(
            s0_48["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
        ),
        "upgraded_exp_t13_passes_snap2_s48": bool(
            s2_48["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
        ),
        "stride96_snap2_ret_ge_07_vs_canon": float(s2_96_agr.get("ref_hit_retention_ge_07", 0.0)),
        "stride96_snap2_screening_gate_pass": bool(
            s2_96["recovers_signal"]
            and float(s2_96_agr.get("ref_hit_retention_ge_07", 0.0)) >= 0.70
        ),
    }
    post_gpu = SSRX.query_gpu_telemetry()
    return {
        "schema_version": "gpu_0211_calibration_matrix_v1",
        "scroll": "PHerc0211",
        "segment": G16.PHERC0211_SEGMENT,
        "mesh_tag": G16.PHERC0211_MESH_TAG,
        "uv_crop": list(uv_crop) if uv_crop is not None else None,
        "voxel_um": float(voxel_um),
        "num_slices": int(num_slices),
        "total_ct_bytes": total_ct_bytes,
        "total_m7_bytes": total_m7_bytes,
        "snap_modes": snap_runs,
        "verdict_summary": verdict,
        "safety_profile": {
            "post_run_temp_c": post_gpu["temp_c"] if post_gpu is not None else None,
            "total_seconds": round(time.perf_counter() - t0, 2),
        },
    }


def run_arm2_pherc0306b_layer_card(
    *,
    out_private_dir: Path,
    work_root: Path,
    staged: dict[str, Path],
    mesh_source: str = GTI.PHERC0306B_MESH_URL,
    ct_zarr_url: str = GTI.PHERC0306B_CT_ZARR_URL,
    uv_crop: tuple[int, int, int, int] | None = GTI.PHERC0306B_DEFAULT_UV_CROP,
    voxel_um: float = GTI.PHERC0306B_VOXEL_UM,
    num_slices: int = 31,
    studio_batch: int = DEFAULT_STUDIO_BATCH,
    ahn_batch: int = DEFAULT_AHN_BATCH,
    batch_pause_ms: int = DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = DEFAULT_CACHE_MB,
    run_studio_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    run_ahn_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
    guardrail_fn: Callable[..., dict[str, int] | None] | None = SSRX.enforce_gpu_guardrail,
) -> dict[str, Any]:
    """Run Studio (s48 & s96) + A-hn (`asc` & `desc`) + export layer-order card on `PHerc. 0306B w074`."""
    base_out = SSRX.ensure_outside_git_repo(out_private_dir)
    out_dir = base_out / "gpu_0211_0306b_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    studio_runner = run_studio_fn or SSRX.run_studio_both_orders
    ahn_runner = run_ahn_fn or SSRX.run_ahn_both_orders

    t0 = time.perf_counter()
    mesh_idx = (num_slices - 1) // 2
    if guardrail_fn is not None:
        guardrail_fn(max_temp_c=max_gpu_temp_c)

    with tempfile.TemporaryDirectory(prefix="arm2-0306b-", dir=str(work_root), ignore_cleanup_errors=True) as mtmp:
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
        coherence = _score_onsheet_coherence(stack_lhw, mesh_idx, valid_ct)
        layer_card = compute_layer_order_profile_card(stack_lhw, valid_ct, mesh_idx=mesh_idx)

        res_ahn = ahn_runner(
            stack_lhw,
            mesh_idx,
            staged,
            batch_size=ahn_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
        )
        if guardrail_fn is not None:
            guardrail_fn(max_temp_c=max_gpu_temp_c)

        res_s48 = studio_runner(
            stack_lhw,
            mesh_idx,
            staged,
            stride=48,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
        )
        if guardrail_fn is not None:
            guardrail_fn(max_temp_c=max_gpu_temp_c)

        res_s96 = studio_runner(
            stack_lhw,
            mesh_idx,
            staged,
            stride=96,
            batch_size=studio_batch,
            batch_pause_ms=batch_pause_ms,
            max_gpu_temp_c=max_gpu_temp_c,
        )

        s48_entry = score_detector_pair(
            res_s48,
            family="studio",
            stack_mid=stack_lhw[mesh_idx],
            valid_ct=valid_ct,
            voxel_um=voxel_um,
            paired_ahn_res=res_ahn,
        )
        canon_face = s48_entry["dominant_face"]
        canon_prob = np.asarray(res_s48[canon_face]["prob"], dtype=np.float32).copy()
        canon_valid = (np.asarray(res_s48[canon_face]["valid"], dtype=bool) & valid_ct).copy()
        s48_entry["agreement_vs_canonical"] = GTI.compute_map_agreement(
            canon_prob, canon_valid, canon_prob, canon_valid
        )
        s96_entry = score_detector_pair(
            res_s96,
            family="studio",
            stack_mid=stack_lhw[mesh_idx],
            valid_ct=valid_ct,
            voxel_um=voxel_um,
            paired_ahn_res=res_ahn,
            ref_prob=canon_prob,
            ref_valid=canon_valid,
            ref_face=canon_face,
        )
        ahn_entry = score_detector_pair(
            res_ahn,
            family="ahn",
            stack_mid=stack_lhw[mesh_idx],
            valid_ct=valid_ct,
            voxel_um=voxel_um,
            ref_prob=canon_prob,
            ref_valid=canon_valid,
            ref_face=canon_face,
        )

        ct_mid = stack_lhw[mesh_idx].copy()
        panels: list[tuple[str, np.ndarray, np.ndarray]] = [
            (
                f"Studio s48 {s48_entry['dominant_face']} (gt07={s48_entry['dom_frac_gt_07']:.4f})",
                res_s48[s48_entry["dominant_face"]]["prob"],
                np.asarray(res_s48[s48_entry["dominant_face"]]["valid"], dtype=bool) & valid_ct,
            ),
            (
                f"Studio s96 {s96_entry['dominant_face']} (gt07={s96_entry['dom_frac_gt_07']:.4f})",
                res_s96[s96_entry["dominant_face"]]["prob"],
                np.asarray(res_s96[s96_entry["dominant_face"]]["valid"], dtype=bool) & valid_ct,
            ),
            (
                f"A-hn hn4_s0 {ahn_entry['dominant_face']} (gt07={ahn_entry['dom_frac_gt_07']:.4f})",
                res_ahn[ahn_entry["dominant_face"]]["prob"],
                np.asarray(res_ahn[ahn_entry["dominant_face"]]["valid"], dtype=bool) & valid_ct,
            ),
            (
                f"A-hn hn4_s0 {ahn_entry['opposite_face']} (opp gt07={ahn_entry['opp_frac_gt_07']:.4f})",
                res_ahn[ahn_entry["opposite_face"]]["prob"],
                np.asarray(res_ahn[ahn_entry["opposite_face"]]["valid"], dtype=bool) & valid_ct,
            ),
        ]
        del stack_lhw, res_ahn, res_s48, res_s96
        GTI._cleanup_ram_and_cuda()

    GTI.write_comparison_montage(
        out_dir / "arm2_pherc0306b_layer_card_montage.png",
        "Exp T17-GPU Arm 2: PHerc. 0306B w074 Studio{s48,s96} + A-hn Layer-Order Control",
        ct_mid,
        valid_ct,
        panels,
        voxel_um,
    )
    post_gpu = SSRX.query_gpu_telemetry()
    return {
        "schema_version": "gpu_0306b_layer_order_card_v1",
        "scroll": "PHerc0306B",
        "mesh_tag": "PHerc0306B_20250521133212_w074own_touchup_V5Rc",
        "uv_crop": list(uv_crop) if uv_crop is not None else None,
        "voxel_um": float(voxel_um),
        "num_slices": int(num_slices),
        "valid_ct_pixels": int(valid_ct.sum()),
        "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
        "ct_bytes": int(rendered["ct_bytes"]),
        "ct_chunks_fetched": int(rendered["ct_chunks"]),
        "ct_mib_fetched": round(float(rendered["ct_bytes"]) / (1024 ** 2), 4),
        "onsheet_coherence": coherence,
        "layer_order_profile_card": layer_card,
        "studio_s48": s48_entry,
        "studio_s96": s96_entry,
        "ahn_s0": ahn_entry,
        "reading_order_alignment": {
            "studio_dominant_face": s48_entry["dominant_face"],
            "ahn_dominant_face": ahn_entry["dominant_face"],
            "same_face_direction_between_families": bool(
                s48_entry["dominant_face"] == ahn_entry["dominant_face"]
            ),
        },
        "safety_profile": {
            "post_run_temp_c": post_gpu["temp_c"] if post_gpu is not None else None,
            "total_seconds": round(time.perf_counter() - t0, 2),
        },
    }


def run_exp_t17_gpu(
    *,
    arm: str = "both",
    out_private_dir: Path,
    public_receipt_path: Path | None = DEFAULT_PUBLIC_RECEIPT,
    vram_fraction: float = DEFAULT_VRAM_FRACTION,
    studio_batch: int = DEFAULT_STUDIO_BATCH,
    ahn_batch: int = DEFAULT_AHN_BATCH,
    batch_pause_ms: int = DEFAULT_BATCH_PAUSE_MS,
    max_gpu_temp_c: int = DEFAULT_MAX_GPU_TEMP_C,
    cache_mb: int = DEFAULT_CACHE_MB,
) -> dict[str, Any]:
    """Run Exp T17-GPU calibration arms sequentially on RTX 4060 and save sanitized receipt."""
    if not (0.05 <= float(vram_fraction) <= GTI.MAX_SAFE_VRAM_FRACTION):
        raise ValueError(
            f"--vram-fraction must be within [0.05, {GTI.MAX_SAFE_VRAM_FRACTION}], got {vram_fraction}"
        )
    out_dir = SSRX.ensure_outside_git_repo(out_private_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SSRX.enforce_gpu_guardrail(max_temp_c=int(max_gpu_temp_c))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_0211_and_0306b_calibration_matrix.py")
    torch.cuda.set_per_process_memory_fraction(float(vram_fraction), 0)

    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rtx4060-t17-work-", ignore_cleanup_errors=True) as tmp:
        work_root = Path(tmp)
        staged = SSRX.stage_models(work_root, {"studio", "ahn"})

        arm1 = (
            run_arm1_pherc0211_matrix(
                out_private_dir=out_dir,
                work_root=work_root,
                staged=staged,
                studio_batch=studio_batch,
                ahn_batch=ahn_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                cache_mb=cache_mb,
            )
            if arm in ("pherc0211", "both")
            else None
        )
        arm2 = (
            run_arm2_pherc0306b_layer_card(
                out_private_dir=out_dir,
                work_root=work_root,
                staged=staged,
                studio_batch=studio_batch,
                ahn_batch=ahn_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
                cache_mb=cache_mb,
            )
            if arm in ("pherc0306b", "both")
            else None
        )

    post_gpu = SSRX.query_gpu_telemetry()
    streamed_bytes = (
        (int(arm1["total_ct_bytes"]) + int(arm1["total_m7_bytes"]) if arm1 else 0)
        + (int(arm2["ct_bytes"]) if arm2 else 0)
    )
    payload: dict[str, Any] = {
        "schema_version": "gpu_0211_and_0306b_calibration_matrix_v1",
        "experiment": (
            "Exp T17-GPU: PHerc. 0211 Full Screening Calibration Matrix (claude-020 Item 2) & "
            "PHerc. 0306B A-hn + Layer-Order Profile Card (claude-020 Item 5)"
        ),
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
            "streamed_zarr_bytes": streamed_bytes,
            "persistent_ct_on_disk": False,
        },
        "arm1_pherc0211_matrix": arm1,
        "arm2_pherc0306b_layer_card": arm2,
    }
    GTI.assert_no_local_paths_in_public_json(payload)
    if public_receipt_path is not None:
        public_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        public_receipt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser with RTX 4060 Laptop VRAM and thermal guardrails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("pherc0211", "pherc0306b", "both"), default="both")
    ap.add_argument("--out-private-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_PUBLIC_RECEIPT)
    ap.add_argument("--vram-fraction", type=float, default=DEFAULT_VRAM_FRACTION)
    ap.add_argument("--studio-batch", type=int, default=DEFAULT_STUDIO_BATCH)
    ap.add_argument("--ahn-batch", type=int, default=DEFAULT_AHN_BATCH)
    ap.add_argument("--batch-pause-ms", type=int, default=DEFAULT_BATCH_PAUSE_MS)
    ap.add_argument("--max-gpu-temp-c", type=int, default=DEFAULT_MAX_GPU_TEMP_C)
    ap.add_argument("--cache-mb", type=int, default=DEFAULT_CACHE_MB)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_exp_t17_gpu(
        arm=args.arm,
        out_private_dir=args.out_private_dir,
        public_receipt_path=args.out_json,
        vram_fraction=args.vram_fraction,
        studio_batch=args.studio_batch,
        ahn_batch=args.ahn_batch,
        batch_pause_ms=args.batch_pause_ms,
        max_gpu_temp_c=args.max_gpu_temp_c,
        cache_mb=args.cache_mb,
    )
    print(f"Wrote {args.out_json} in {payload['hardware_safety']['total_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
