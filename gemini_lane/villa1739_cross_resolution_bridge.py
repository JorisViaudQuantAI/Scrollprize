# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T12-Villa1739: Cross-Resolution 2.403 um -> 9.366 um Co-Registered Human-Label Bridge
& 42D Depth-Spatial Kernel Transfer Benchmark.

Strictly CPU-only ($env:CUDA_VISIBLE_DEVICES="-1", zero torch/CUDA imports),
zero persistent CT on disk (RAM only), and hard-capped global HTTP byte budget
(<= 100 MiB = 104,857,600 B).

Scientific Design:
1. Co-Registering PHerc. 0841 w00 Native 2.403 um Human Ground Truth onto the 9.366 um
   Surface Volume (Vesuvius Challenge rule `villa #1739` & `collab/chatgpt-031.md`):
   - Loads human Zarr v3 ground truth (`inklabels.zarr` + `supervision.zarr`, 16460 x 18560)
     and the 9.366 um `argus-v3` reference TIFF (4220 x 4760, decoded in RAM via PIL.Image.open).
   - Downsamples the 2.403 um human maps to the 9.366 um canvas (exact scale ratio ~3.90047x
     via exact separable block-area integration) and solves the fine 2D translational shift
     (dy, dx) in [-24, +24]^2 pixels maximizing overlap with the 9.366 um coordinate frame.
2. Multi-Feature 3D Depth-Spatial Representation (D = 42 features per pixel (y, x)):
   - Extracted from any 3D surface-volume slab (either on-sheet peak-aligned z_peak(y, x)
     or off-sheet shifted by +/-8 slices for the adversarial kill-test):
     * 31D peak-centered depth waveform decomposed into 16 laterally-detrended symmetric
       depth profile channels (P(k) - mu_lateral(k)) + 15 antisymmetric depth-difference
       channels (P(+k) - P(-k), k = 1..15) capturing one-sided ink vs symmetric papyrus wall.
     * 3 multi-scale sheet-plane DoG contrast channels (sigma in {1.0, 2.0, 4.0} - sigma_bg=8.0)
       on surface CT + 3 multi-scale DoG channels on near-peak antisymmetric depth contrast.
     * 2 local depth-gradient asymmetry channels around the sheet peak.
     * 1 `oct_hilbert_env_peak` channel + 2 2D wavelet-scattering channels (`wst_s1_isotropic`,
       `wst_s1_scale_slope`).
   - Per-ROI/fold robust median/IQR standardization (zero label leakage, Q22 gain/offset invariant).
3. Supervised Ridge Distillation (lambda = 100.0) on Spatial Fold A (top half, 32 px buffer
   excluded with strict filter-footprint isolation) under two modes:
   - Mode 1 (`human_2p4um_bridged`): trained purely on co-registered 2.403 um human labels.
   - Mode 2 (`teacher_2p4um_distilled`): trained on soft `argus-v3` targets inside human supervision.
   Evaluates frozen weights w (locked polarity `+`) on:
   - Held-out Spatial Fold B of PHerc. 0841 w00 (9.366 um) vs BOTH human labels and argus-v3.
   - Zero-shot transfer to PHerc. 0343 w047-w048 (8.64 um).
   - Zero-shot transfer to PHerc. 0306B w074 (8.64 um, First Letters single-line claim PNG).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal as ssig

try:
    from .pherc0139_transverse_physics import (
        BLANK_LOW_U8_THRESHOLD,
        INK_HIGH_U8_THRESHOLD,
        MIB,
        BudgetExceededError,
        BudgetedRemoteZarrV2,
        HttpByteBudget,
        _fill_nearest_2d,
        _normalized_gaussian,
        assert_no_local_paths,
        budgeted_http_get,
        build_16_rolled_shifts,
        cohens_d,
        exact_roc_auc,
        extract_subvoxel_height_field,
        load_tifxyz_in_ram,
    )
    from .stream_snap_render import _sanitize_source_label
    from .transverse_cross_scroll_replication import (
        DEFAULT_VOLUME_SPECS as T7_VOLUME_SPECS,
        HUMAN_LABEL_MAX_SHARD_BYTES,
        MultiVolumeHttpByteBudget,
        _sample_slab_at_depth,
        extract_shifted_subslab,
        list_s3_prefix_budgeted,
        load_zarr3_sharded_2d_u8,
        select_chunk_aligned_roi,
    )
    from .transverse_wavelet_oct_physics import (
         DEFAULT_URLS as T5_DEFAULT_URLS,
        OCT_WAVEFORM_DIM,
        PHERC0306B_BLANK_D_THRESHOLD,
        PHERC0306B_INK_D_THRESHOLD,
        PHERC0306B_NUM_SLICES,
        PHERC0306B_ROI_H,
        PHERC0306B_ROI_W,
        PHERC0306B_VOXEL_UM,
        build_0306b_rays_for_roi,
        compute_wavelet_scattering_features,
        decode_claim_png_tone,
        extract_peak_aligned_depth_profiles,
        load_json_in_ram,
        load_png_rgb_in_ram,
        select_0306b_line1_roi,
        verify_0306b_alignment,
    )
except ImportError:  # pragma: no cover - direct script execution
    from pherc0139_transverse_physics import (
        BLANK_LOW_U8_THRESHOLD,
        INK_HIGH_U8_THRESHOLD,
        MIB,
        BudgetExceededError,
        BudgetedRemoteZarrV2,
        HttpByteBudget,
        _fill_nearest_2d,
        _normalized_gaussian,
        assert_no_local_paths,
        budgeted_http_get,
        build_16_rolled_shifts,
        cohens_d,
        exact_roc_auc,
        extract_subvoxel_height_field,
        load_tifxyz_in_ram,
    )
    from stream_snap_render import _sanitize_source_label
    from transverse_cross_scroll_replication import (
        DEFAULT_VOLUME_SPECS as T7_VOLUME_SPECS,
        HUMAN_LABEL_MAX_SHARD_BYTES,
        MultiVolumeHttpByteBudget,
        _sample_slab_at_depth,
        extract_shifted_subslab,
        list_s3_prefix_budgeted,
        load_zarr3_sharded_2d_u8,
        select_chunk_aligned_roi,
    )
    from transverse_wavelet_oct_physics import (
        DEFAULT_URLS as T5_DEFAULT_URLS,
        OCT_WAVEFORM_DIM,
        PHERC0306B_BLANK_D_THRESHOLD,
        PHERC0306B_INK_D_THRESHOLD,
        PHERC0306B_NUM_SLICES,
        PHERC0306B_ROI_H,
        PHERC0306B_ROI_W,
        PHERC0306B_VOXEL_UM,
        build_0306b_rays_for_roi,
        compute_wavelet_scattering_features,
        decode_claim_png_tone,
        extract_peak_aligned_depth_profiles,
        load_json_in_ram,
        load_png_rgb_in_ram,
        select_0306b_line1_roi,
        verify_0306b_alignment,
    )


# ---------------------------------------------------------------------------
# Predeclared Experiment Constants & Public S3 / GitHub Endpoints
# ---------------------------------------------------------------------------

S3_BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

GLOBAL_HTTP_BUDGET_MIB = 100.0
GLOBAL_HTTP_BUDGET_BYTES = int(round(GLOBAL_HTTP_BUDGET_MIB * MIB))  # 104,857,600 B

DEFAULT_PART_BUDGETS_MIB: dict[str, float] = {
    "pherc0841_9p366um": 28.0,
    "pherc0343_8p64um": 24.0,
    "pherc0306b_8p64um": 48.0,
}

FEATURE_DIM = 42
RIDGE_LAMBDA = 100.0
SPATIAL_BUFFER_PX = 32
REGISTRATION_MAX_SHIFT_PX = 24
OFF_SHEET_SHIFT_SLICES = 8
OFF_SHEET_SUBSLAB_DEPTH = 5

DOG_SIGMAS_PX: tuple[float, float, float] = (1.0, 2.0, 4.0)
DOG_BG_SIGMA_PX: float = 8.0
LATERAL_BG_SIGMA_PX: float = 8.0

# Predeclared Evaluation Gate (locked polarity '+', NEVER flipped post-hoc)
GATE_MIN_ON_SHEET_AUC = 0.70
GATE_MIN_DELTA_NULL = 0.10
GATE_MIN_OFF_SHEET_DROP = 0.08

SUPERVISION_MODES: tuple[str, str] = (
    "human_2p4um_bridged",
    "teacher_2p4um_distilled",
)

FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"depth_sym_detrend_m{m:02d}" for m in range(16))
    + tuple(f"depth_antisym_diff_m{m:02d}" for m in range(1, 16))
    + tuple(f"surface_dog_s{int(s)}_bg8" for s in DOG_SIGMAS_PX)
    + tuple(f"antisym_dog_s{int(s)}_bg8" for s in DOG_SIGMAS_PX)
    + (
        "depth_grad_front_back_diff",
        "depth_grad_front_back_logratio",
        "oct_hilbert_env_peak",
        "wst_s1_isotropic",
        "wst_s1_scale_slope",
    )
)
assert len(FEATURE_NAMES) == FEATURE_DIM


# ---------------------------------------------------------------------------
# 1. Cross-Resolution 2.403 um -> 9.366 um Downsampling & 2D Registration
# ---------------------------------------------------------------------------

def _resample_1d_box_area_axis0(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Exact 1D area-weighted box integration along axis 0 via cumulative sum interpolation."""
    src_len = int(arr.shape[0])
    dst_len = int(target_len)
    if src_len == dst_len:
        return arr.astype(np.float32, copy=True)
    if src_len <= 0 or dst_len <= 0:
        raise ValueError(f"Invalid lengths for box-area resample: {src_len} -> {dst_len}")

    # Cumulative integral F(u) at integer grid points u = 0, 1, ..., src_len
    csum = np.zeros((src_len + 1,) + arr.shape[1:], dtype=np.float64)
    np.cumsum(arr, axis=0, out=csum[1:])

    edges = np.linspace(0.0, float(src_len), dst_len + 1, dtype=np.float64)
    idx0 = np.clip(np.floor(edges).astype(np.int64), 0, src_len - 1)
    frac = edges - idx0.astype(np.float64)
    shape_exp = (-1,) + (1,) * (arr.ndim - 1)
    f_edges = (1.0 - frac.reshape(shape_exp)) * csum[idx0] + frac.reshape(shape_exp) * csum[idx0 + 1]
    widths = np.maximum(edges[1:] - edges[:-1], 1e-12).reshape(shape_exp)
    return ((f_edges[1:] - f_edges[:-1]) / widths).astype(np.float32)


def downsample_block_area_2d(
    arr_hi: np.ndarray,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Downsample a 2D map from high-res canvas (e.g. 16460x18560) to low-res (4220x4760) via exact block-area average."""
    src = np.asarray(arr_hi, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"Expected 2D array for block-area downsampling, got {src.shape}")
    h_hi, w_hi = int(src.shape[0]), int(src.shape[1])
    h_lo, w_lo = int(target_hw[0]), int(target_hw[1])
    if h_lo <= 0 or w_lo <= 0:
        raise ValueError(f"Invalid target_hw: {target_hw}")

    tmp = _resample_1d_box_area_axis0(src, h_lo)
    out = _resample_1d_box_area_axis0(tmp.T, w_lo).T
    ry = float(h_hi) / float(h_lo)
    rx = float(w_hi) / float(w_lo)
    info = {
        "source_shape": [h_hi, w_hi],
        "target_shape": [h_lo, w_lo],
        "scale_ratio_y": round(ry, 5),
        "scale_ratio_x": round(rx, 5),
        "scale_ratio_mean": round(0.5 * (ry + rx), 5),
        "method": "exact_separable_block_area_integration",
    }
    return np.ascontiguousarray(out, dtype=np.float32), info


def shift_2d_zero_pad(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2D array by (dy, dx) pixels with zero-padding (no toroidal wrap-around)."""
    h, w = arr.shape
    out = np.zeros_like(arr)
    y_src0 = max(0, -int(dy))
    y_src1 = min(h, h - int(dy))
    x_src0 = max(0, -int(dx))
    x_src1 = min(w, w - int(dx))
    if y_src1 <= y_src0 or x_src1 <= x_src0:
        return out
    y_dst0 = y_src0 + int(dy)
    y_dst1 = y_src1 + int(dy)
    x_dst0 = x_src0 + int(dx)
    x_dst1 = x_src1 + int(dx)
    out[y_dst0:y_dst1, x_dst0:x_dst1] = arr[y_src0:y_src1, x_src0:x_src1]
    return out


def _binary_iou_and_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    inter = int(np.count_nonzero(mask_a & mask_b))
    sum_a = int(np.count_nonzero(mask_a))
    sum_b = int(np.count_nonzero(mask_b))
    union = sum_a + sum_b - inter
    iou = float(inter) / float(max(union, 1))
    dice = float(2 * inter) / float(max(sum_a + sum_b, 1))
    return iou, dice


def register_2p4um_human_labels_to_9p366um(
    human_ink_hi_u8: np.ndarray,
    human_sup_hi_u8: np.ndarray,
    ref_9p366um_u8: np.ndarray,
    *,
    max_shift_px: int = REGISTRATION_MAX_SHIFT_PX,
    ref_ink_threshold_u8: int = 128,
) -> dict[str, Any]:
    """Downsample 2.403 um human labels to 9.366 um canvas and solve 2D translational shift (dy, dx) in [-24, +24]^2."""
    ref_u8 = np.asarray(ref_9p366um_u8, dtype=np.uint8)
    target_hw = (int(ref_u8.shape[0]), int(ref_u8.shape[1]))

    ink_prob_hi = (np.asarray(human_ink_hi_u8, dtype=np.float32) >= 128.0).astype(np.float32)
    sup_prob_hi = (np.asarray(human_sup_hi_u8, dtype=np.float32) >= 128.0).astype(np.float32)

    ink_prob_lo, ds_info = downsample_block_area_2d(ink_prob_hi, target_hw)
    sup_prob_lo, _ = downsample_block_area_2d(sup_prob_hi, target_hw)

    raw_human_sup = sup_prob_lo >= 0.5
    raw_human_ink = (ink_prob_lo >= 0.5) & raw_human_sup
    ref_bin_full = ref_u8 >= int(ref_ink_threshold_u8)

    # Bound the 2D search to the tight bounding box of human supervision (+/- max_shift_px)
    # so searching [-24, +24]^2 (49 x 49 = 2401 shifts) runs in ~50 ms on CPU!
    ys, xs = np.nonzero(raw_human_sup)
    if ys.size == 0:
        raise ValueError("Human supervision map has zero supervised pixels after downsampling.")
    pad = int(max_shift_px) + 2
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(target_hw[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(target_hw[1], int(xs.max()) + 1 + pad)

    sub_h_ink = raw_human_ink[y0:y1, x0:x1]
    sub_h_sup = raw_human_sup[y0:y1, x0:x1]
    sub_ref_bin = ref_bin_full[y0:y1, x0:x1]

    pre_iou, pre_dice = _binary_iou_and_dice(sub_h_ink, sub_ref_bin & sub_h_sup)

    best_dy = 0
    best_dx = 0
    best_iou = -1.0
    best_dice = -1.0
    best_key: tuple[float, float, int] = (-1.0, -1.0, 0)

    ms = int(max_shift_px)
    for dy in range(-ms, ms + 1):
        for dx in range(-ms, ms + 1):
            sh_ink = shift_2d_zero_pad(sub_h_ink, dy, dx)
            sh_sup = shift_2d_zero_pad(sub_h_sup, dy, dx)
            ref_in_sup = sub_ref_bin & sh_sup
            iou, dice = _binary_iou_and_dice(sh_ink, ref_in_sup)
            # Tie-break toward smaller Euclidean shift norm
            key = (round(iou, 8), round(dice, 8), -(dy * dy + dx * dx))
            if key > best_key:
                best_key = key
                best_iou = iou
                best_dice = dice
                best_dy = int(dy)
                best_dx = int(dx)

    aligned_ink_prob = shift_2d_zero_pad(ink_prob_lo, best_dy, best_dx)
    aligned_sup_prob = shift_2d_zero_pad(sup_prob_lo, best_dy, best_dx)
    aligned_sup_mask = aligned_sup_prob >= 0.5
    aligned_ink_mask = (aligned_ink_prob >= 0.5) & aligned_sup_mask
    # Conservative blank mask: strictly zero ink density inside full supervision support
    aligned_blank_mask = (aligned_ink_prob <= 0.05) & (aligned_sup_prob >= 0.95)

    return {
        "aligned_ink_mask": aligned_ink_mask,
        "aligned_blank_mask": aligned_blank_mask,
        "aligned_sup_mask": aligned_sup_mask,
        "aligned_ink_prob": aligned_ink_prob,
        "aligned_sup_prob": aligned_sup_prob,
        "registration_info": {
            **ds_info,
            "search_range_px": [-ms, ms],
            "best_shift_dy_dx": [int(best_dy), int(best_dx)],
            "shift_euclidean_px": round(math.hypot(best_dy, best_dx), 4),
            "shift_physical_um": [
                round(best_dy * 9.366, 3),
                round(best_dx * 9.366, 3),
            ],
            "pre_alignment_iou": round(pre_iou, 5),
            "post_alignment_iou": round(best_iou, 5),
            "iou_gain": round(best_iou - pre_iou, 5),
            "pre_alignment_dice": round(pre_dice, 5),
            "post_alignment_dice": round(best_dice, 5),
            "dice_gain": round(best_dice - pre_dice, 5),
            "supervised_pixels_9p366um": int(aligned_sup_mask.sum()),
            "human_ink_pixels_9p366um": int(aligned_ink_mask.sum()),
            "human_blank_pixels_9p366um": int(aligned_blank_mask.sum()),
        },
    }


def load_pherc0841_human_2p4um_and_argus_9p366um(
    spec_0841: dict[str, Any],
    sv_9p366_hw: tuple[int, int],
    *,
    budget: HttpByteBudget,
    part: str = "pherc0841_9p366um",
    s3_base: str = S3_BASE,
) -> dict[str, Any]:
    """Load PHerc. 0841 w00 2.403 um human Zarr v3 labels + 9.366 um argus-v3 TIFF and co-register."""
    explicit_ink = spec_0841.get("human_2p4um_inklabels_zarr")
    explicit_sup = spec_0841.get("human_2p4um_supervision_zarr")
    s3_entries_count = 0

    if explicit_ink:
        ink_hi, ink_meta = load_zarr3_sharded_2d_u8(
            explicit_ink, budget=budget, part=part, label=f"{part}:human_2p4um_ink"
        )
        if explicit_sup:
            sup_hi, sup_meta = load_zarr3_sharded_2d_u8(
                explicit_sup, budget=budget, part=part, label=f"{part}:human_2p4um_sup"
            )
        else:
            sup_hi = np.full_like(ink_hi, 255)
            sup_meta = {"shape": list(ink_hi.shape), "shard_bytes": 0}
        ink_src_label = _sanitize_source_label(explicit_ink)
        sup_src_label = _sanitize_source_label(explicit_sup) if explicit_sup else "full_canvas"
    else:
        prefix = str(spec_0841.get("ink_labels_s3_prefix", "PHerc0841/segments/20260220213127-w00/ink-labels/"))
        entries = list_s3_prefix_budgeted(
            s3_base, prefix, budget, part, label=f"{part}:s3_ink_labels_list"
        )
        s3_entries_count = len(entries)
        by_key = dict(entries)
        matched_dirs: list[tuple[str, int]] = []
        for key, sz in entries:
            if key.endswith("/inklabels.zarr/0/c/0/0") and "2.403um" in key:
                matched_dirs.append((key[: -len("/inklabels.zarr/0/c/0/0")], int(sz)))
        matched_dirs.sort(key=lambda item: item[0], reverse=True)
        if not matched_dirs:
            raise RuntimeError("No 2.403 um human inklabels.zarr found under PHerc0841 S3 prefix.")
        zdir, ink_sz = matched_dirs[0]
        sup_key = f"{zdir}/supervision.zarr/0/c/0/0"
        sup_sz = int(by_key.get(sup_key, 0))
        if ink_sz + sup_sz > HUMAN_LABEL_MAX_SHARD_BYTES:
            raise BudgetExceededError(f"Human label shards too large: {ink_sz + sup_sz} B")
        ink_url = f"{s3_base.rstrip('/')}/{zdir}/inklabels.zarr"
        sup_url = f"{s3_base.rstrip('/')}/{zdir}/supervision.zarr"
        ink_hi, ink_meta = load_zarr3_sharded_2d_u8(
            ink_url, budget=budget, part=part, label=f"{part}:human_2p4um_ink"
        )
        sup_hi, sup_meta = load_zarr3_sharded_2d_u8(
            sup_url, budget=budget, part=part, label=f"{part}:human_2p4um_sup"
        )
        ink_src_label = _sanitize_source_label(ink_url)
        sup_src_label = _sanitize_source_label(sup_url)

    # Load 9.366 um argus-v3 reference TIFF in RAM via PIL.Image.open
    tif_url = str(spec_0841["argus_ink_tif_url"])
    if tif_url.lower().startswith(("http://", "https://")):
        raw_tif = budgeted_http_get(tif_url, budget, part, label=f"{part}:argus_v3_tif")
        assert raw_tif is not None
        with Image.open(io.BytesIO(raw_tif)) as im:
            arr_tif = np.array(im, copy=True)
        del raw_tif
    else:
        with Image.open(tif_url) as im:
            arr_tif = np.array(im, copy=True)
    if arr_tif.ndim == 3:
        arr_tif = arr_tif[..., 0]
    argus_u8 = np.ascontiguousarray(arr_tif.astype(np.uint8, copy=False))
    if argus_u8.shape != tuple(sv_9p366_hw):
        raise ValueError(f"Argus TIFF shape {argus_u8.shape} != 9.366 um SV shape {sv_9p366_hw}")

    reg = register_2p4um_human_labels_to_9p366um(
        ink_hi,
        sup_hi,
        argus_u8,
        max_shift_px=int(spec_0841.get("registration_max_shift_px", REGISTRATION_MAX_SHIFT_PX)),
    )
    del ink_hi, sup_hi

    argus_ink_mask = argus_u8 >= INK_HIGH_U8_THRESHOLD
    argus_blank_mask = (argus_u8 > 0) & (argus_u8 <= BLANK_LOW_U8_THRESHOLD)

    return {
        **reg,
        "argus_u8": argus_u8,
        "argus_ink_mask": argus_ink_mask,
        "argus_blank_mask": argus_blank_mask,
        "provenance": {
            "human_2p4um_inklabels_source": ink_src_label,
            "human_2p4um_supervision_source": sup_src_label,
            "argus_9p366um_tif_source": _sanitize_source_label(tif_url),
            "s3_ink_labels_entries_found": int(s3_entries_count),
            "human_inklabels_meta": ink_meta,
            "human_supervision_meta": sup_meta,
            "villa_rule_reference": "villa #1739 (2.4 um -> 9.366 um cross-resolution label/model bridge permitted)",
        },
    }


def select_bridged_roi_0841(
    human_ink_mask: np.ndarray,
    human_blank_mask: np.ndarray,
    human_sup_mask: np.ndarray,
    argus_ink_mask: np.ndarray,
    argus_blank_mask: np.ndarray,
    sv_chunks_hw: tuple[int, int],
    *,
    roi_h: int = 512,
    roi_w: int = 640,
    buffer_px: int = SPATIAL_BUFFER_PX,
) -> dict[str, Any]:
    """Deterministically select a chunk-aligned 9.366 um ROI where BOTH Fold A and Fold B have human & argus ink/blank."""
    h_sv, w_sv = human_ink_mask.shape
    cy_sz, cx_sz = int(sv_chunks_hw[0]), int(sv_chunks_hw[1])
    if roi_h % cy_sz != 0 or roi_w % cx_sz != 0:
        raise ValueError(f"ROI {(roi_h, roi_w)} must align to chunk grid {(cy_sz, cx_sz)}")

    n_cy = h_sv // cy_sz
    n_cx = w_sv // cx_sz
    ny_win = roi_h // cy_sz
    nx_win = roi_w // cx_sz

    half_h = roi_h // 2
    buf_half = int(buffer_px) // 2
    fold_a_end = half_h - buf_half
    fold_b_start = half_h + (int(buffer_px) - buf_half)

    best_cand: dict[str, Any] | None = None
    best_score = -1.0
    screened = 0

    for ty0 in range(0, n_cy - ny_win + 1):
        r0 = ty0 * cy_sz
        r1 = r0 + roi_h
        for tx0 in range(0, n_cx - nx_win + 1):
            c0 = tx0 * cx_sz
            c1 = c0 + roi_w
            sup_patch = human_sup_mask[r0:r1, c0:c1]
            sup_frac = float(sup_patch.mean())
            if sup_frac < 0.25:
                continue
            screened += 1

            h_ink = human_ink_mask[r0:r1, c0:c1]
            h_blk = human_blank_mask[r0:r1, c0:c1]
            a_ink = argus_ink_mask[r0:r1, c0:c1]
            a_blk = argus_blank_mask[r0:r1, c0:c1]

            # Both Fold A (top) and Fold B (bottom) must contain substantial human ink & blank
            hi_a = int(h_ink[:fold_a_end, :].sum())
            hb_a = int(h_blk[:fold_a_end, :].sum())
            hi_b = int(h_ink[fold_b_start:, :].sum())
            hb_b = int(h_blk[fold_b_start:, :].sum())

            ai_b = int(a_ink[fold_b_start:, :].sum())
            ab_b = int(a_blk[fold_b_start:, :].sum())

            min_fold_count = min(hi_a, hb_a, hi_b, hb_b, ai_b, ab_b)
            if min_fold_count < 50:
                continue

            fold_area = float(max(fold_a_end * roi_w, 1))
            balance_a = min(hi_a, hb_a) / fold_area
            balance_b = min(hi_b, hb_b) / fold_area
            score = math.sqrt(max(balance_a * balance_b, 0.0)) * sup_frac
            if score > best_score:
                best_score = score
                best_cand = {
                    "roi_bounds_yx": [int(r0), int(r1), int(c0), int(c1)],
                    "roi_shape_yx": [int(roi_h), int(roi_w)],
                    "human_supervised_fraction": round(sup_frac, 4),
                    "fold_a_rows": [0, int(fold_a_end)],
                    "buffer_zone_rows": [int(fold_a_end), int(fold_b_start)],
                    "fold_b_rows": [int(fold_b_start), int(roi_h)],
                    "fold_a_human_ink": hi_a,
                    "fold_a_human_blank": hb_a,
                    "fold_b_human_ink": hi_b,
                    "fold_b_human_blank": hb_b,
                    "fold_b_argus_ink": ai_b,
                    "fold_b_argus_blank": ab_b,
                    "candidates_screened": screened,
                    "selection_score": round(score, 5),
                    "selection_mode": "deterministic_chunk_aligned_dual_fold_human_bridge_zero_ct_peek",
                }

    if best_cand is None:
        raise RuntimeError("No chunk-aligned ROI on PHerc. 0841 9.366 um has human labels in both Fold A and Fold B.")
    best_cand["candidates_screened"] = int(screened)
    return best_cand


# ---------------------------------------------------------------------------
# 2. Multi-Feature 3D Depth-Spatial Extractor (D = 42) & Robust Standardization
# ---------------------------------------------------------------------------

def extract_42d_depth_spatial_features(
    slab: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    oct_half_span_vx: float = 7.5,
    search_half_slices: int = 6,
    z_smooth_sigma: float = 0.8,
    fixed_center_idx: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract the 42-dimensional depth-spatial feature tensor (D=42, H, W) from a 3D slab.

    Channels (D = 42):
    - [0..15]  (16 ch): Peak-centered symmetric depth profile P_even(m) - mu_lateral(m) (m = 0..15)
    - [16..30] (15 ch): Antisymmetric depth difference P(+m) - P(-m) (m = 1..15)
                        (Together [0..30] span the exact 31D peak-centered depth profile!)
    - [31..33]  (3 ch): Multi-scale sheet-plane DoG contrast (sigma in {1, 2, 4} - sigma_bg=8) on surface CT
    - [34..36]  (3 ch): Multi-scale sheet-plane DoG contrast (sigma in {1, 2, 4} - sigma_bg=8) on near-peak antisym
    - [37..38]  (2 ch): Local depth-gradient asymmetry around the sheet peak (signed diff & log-ratio)
    - [39]      (1 ch): `oct_hilbert_env_peak`
    - [40..41]  (2 ch): 2D wavelet scattering (`wst_s1_isotropic`, `wst_s1_scale_slope`)
    """
    vol = np.asarray(slab, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D slab (L, H, W), got {vol.shape}")
    depth, h, w = vol.shape
    vm = np.asarray(valid_mask, dtype=bool)
    if vm.shape != (h, w):
        raise ValueError(f"valid_mask shape {vm.shape} != {(h, w)}")

    if fixed_center_idx is None:
        hf = extract_subvoxel_height_field(
            vol,
            voxel_um=voxel_um,
            valid_mask=vm,
            search_half_slices=min(int(search_half_slices), (depth - 1) // 2),
            z_smooth_sigma=z_smooth_sigma,
            detrend_sigma_um=200.0,
            band_half_width=1,
        )
        sheet_valid = hf["sheet_valid"]
        surface_ct = hf["surface_ct"]
        n_peak = hf["n_peak"]
        mode_str = "on_sheet_peak_aligned"
    else:
        c_idx = float(np.clip(float(fixed_center_idx), 1.0, float(depth - 2)))
        n_peak = np.full((h, w), c_idx, dtype=np.float32)
        sheet_valid = vm.copy()
        s_m1 = _sample_slab_at_depth(vol, n_peak - 1.0)
        s_0 = _sample_slab_at_depth(vol, n_peak)
        s_p1 = _sample_slab_at_depth(vol, n_peak + 1.0)
        surface_ct = ((s_m1 + s_0 + s_p1) / 3.0).astype(np.float32)
        mode_str = "off_sheet_fixed_depth"

    eff_half_span = min(float(oct_half_span_vx), 0.45 * float(depth - 1))
    profiles, offsets, in_bounds = extract_peak_aligned_depth_profiles(
        vol, n_peak, half_span_vx=max(eff_half_span, 1.0), waveform_dim=OCT_WAVEFORM_DIM
    )
    p64 = profiles.astype(np.float64)
    c = OCT_WAVEFORM_DIM // 2  # 15

    feats = np.zeros((FEATURE_DIM, h, w), dtype=np.float32)

    # 1) 16 laterally-detrended symmetric depth channels (m = 0..15)
    #    and 15 antisymmetric depth-difference channels P(+m) - P(-m) (m = 1..15)
    mu_lat0 = _normalized_gaussian(p64[c], sheet_valid, sigma=LATERAL_BG_SIGMA_PX)
    feats[0] = np.where(sheet_valid, p64[c] - mu_lat0, 0.0).astype(np.float32)
    for m in range(1, c + 1):
        p_pos = p64[c + m]
        p_neg = p64[c - m]
        p_sym = 0.5 * (p_pos + p_neg)
        mu_lat_m = _normalized_gaussian(p_sym, sheet_valid, sigma=LATERAL_BG_SIGMA_PX)
        feats[m] = np.where(sheet_valid, p_sym - mu_lat_m, 0.0).astype(np.float32)
        feats[15 + m] = np.where(sheet_valid, p_pos - p_neg, 0.0).astype(np.float32)

    # 2) Multi-scale sheet-plane DoG contrast (sigma in {1.0, 2.0, 4.0} - sigma_bg=8.0)
    #    on surface CT [31..33] and on near-peak antisymmetric layer (m=1..4 mean) [34..36]
    surf_bg = _normalized_gaussian(surface_ct, sheet_valid, sigma=DOG_BG_SIGMA_PX)
    near_antisym = np.mean(feats[16:20], axis=0).astype(np.float64)
    anti_bg = _normalized_gaussian(near_antisym, sheet_valid, sigma=DOG_BG_SIGMA_PX)
    for idx_s, sig in enumerate(DOG_SIGMAS_PX):
        s_fg = _normalized_gaussian(surface_ct, sheet_valid, sigma=float(sig))
        a_fg = _normalized_gaussian(near_antisym, sheet_valid, sigma=float(sig))
        feats[31 + idx_s] = np.where(sheet_valid, s_fg - surf_bg, 0.0).astype(np.float32)
        feats[34 + idx_s] = np.where(sheet_valid, a_fg - anti_bg, 0.0).astype(np.float32)

    # 3) Local depth-gradient asymmetry around the sheet peak [37..38]
    k_slope = 5
    dp_pos = np.diff(p64[c : c + k_slope + 1], axis=0)
    dp_neg = np.diff(p64[c - k_slope : c + 1], axis=0)
    slope_pos = np.mean(np.abs(dp_pos), axis=0)
    slope_neg = np.mean(np.abs(dp_neg), axis=0)
    signed_grad_asym = np.mean(dp_pos, axis=0) + np.mean(dp_neg, axis=0)
    log_slope_ratio = np.log((slope_pos + 1e-3) / (slope_neg + 1e-3))
    feats[37] = np.where(sheet_valid, signed_grad_asym, 0.0).astype(np.float32)
    feats[38] = np.where(sheet_valid, log_slope_ratio, 0.0).astype(np.float32)

    # 4) oct_hilbert_env_peak [39]
    p_zero = p64 - p64.mean(axis=0, keepdims=True)
    z_shifted = ssig.hilbert(np.fft.ifftshift(p_zero, axes=0), axis=0)
    z_analytic = np.fft.fftshift(z_shifted, axes=0)
    env_peak = np.abs(z_analytic)[c]
    feats[39] = np.where(sheet_valid, env_peak, 0.0).astype(np.float32)

    # 5) 2D Wavelet Scattering channels [40..41] (`wst_s1_isotropic`, `wst_s1_scale_slope`)
    wst_feats, _ = compute_wavelet_scattering_features(
        surface_ct, sheet_valid, voxel_um=voxel_um
    )
    feats[40] = wst_feats["wst_s1_isotropic"]
    feats[41] = wst_feats["wst_s1_scale_slope"]

    diag = {
        "mode": mode_str,
        "feature_dim": FEATURE_DIM,
        "oct_half_span_vx": round(float(eff_half_span), 4),
        "step_vx": round(float(offsets[1] - offsets[0]), 4),
        "fully_in_bounds_fraction": round(float(in_bounds[sheet_valid].mean()) if sheet_valid.any() else 0.0, 4),
        "sheet_valid_fraction": round(float(sheet_valid.mean()), 4),
    }
    return feats, sheet_valid, n_peak, diag


def standardize_features_robust_per_roi(
    feature_tensor: np.ndarray,
    valid_mask: np.ndarray,
    *,
    clip_z: float = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardize each feature channel using per-ROI median and IQR (zero label leakage, Q22 invariant)."""
    feats = np.asarray(feature_tensor, dtype=np.float64)
    if feats.ndim != 3 or feats.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected feature_tensor (42, H, W), got {feats.shape}")
    vm = np.asarray(valid_mask, dtype=bool)
    d_dim, h, w = feats.shape
    out = np.zeros((d_dim, h, w), dtype=np.float32)
    if not vm.any():
        return out, {"n_valid": 0}

    pts = feats[:, vm]  # (D, N_valid)
    med = np.median(pts, axis=1)
    q75 = np.percentile(pts, 75.0, axis=1)
    q25 = np.percentile(pts, 25.0, axis=1)
    iqr = (q75 - q25) / 1.3489795  # normal-consistent IQR scale
    std_fallback = np.std(pts, axis=1)
    scale = np.where(iqr > 1e-6, iqr, np.where(std_fallback > 1e-6, std_fallback, 1.0))

    z = (feats - med[:, None, None]) / scale[:, None, None]
    if clip_z > 0.0:
        z = np.clip(z, -float(clip_z), float(clip_z))
    out = np.where(vm[None, :, :], z, 0.0).astype(np.float32)
    return out, {
        "n_valid": int(vm.sum()),
        "median_l2_norm": round(float(np.linalg.norm(med)), 4),
        "mean_iqr_scale": round(float( np.mean(scale)), 5),
    }


def extract_and_standardize_42d_for_slab(
    slab: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    oct_half_span_vx: float = 7.5,
    search_half_slices: int = 6,
    z_smooth_sigma: float = 0.8,
    fixed_center_idx: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract 42D depth-spatial features and standardize per-ROI via robust median/IQR."""
    raw_feats, sheet_valid, n_peak, diag = extract_42d_depth_spatial_features(
        slab,
        valid_mask,
        voxel_um=voxel_um,
        oct_half_span_vx=oct_half_span_vx,
        search_half_slices=search_half_slices,
        z_smooth_sigma=z_smooth_sigma,
        fixed_center_idx=fixed_center_idx,
    )
    std_feats, std_diag = standardize_features_robust_per_roi(raw_feats, sheet_valid)
    diag["standardization"] = std_diag
    return std_feats, sheet_valid, n_peak, diag


# ---------------------------------------------------------------------------
# 3. Spatial Fold Split (32 px Buffer) & Ridge Classifier Distillation
# ---------------------------------------------------------------------------

def split_spatial_folds_with_buffer(
    height: int,
    *,
    buffer_px: int = SPATIAL_BUFFER_PX,
) -> tuple[slice, slice, dict[str, Any]]:
    """Return strict disjoint row slices (slice_fold_a, slice_fold_b) separated by buffer_px rows."""
    h = int(height)
    buf = int(buffer_px)
    if h < buf + 16:
        raise ValueError(f"Height {h} too small for buffer_px={buf}")
    mid = h // 2
    half_buf = buf // 2
    r_a_end = mid - half_buf
    r_b_start = r_a_end + buf
    return (
        slice(0, r_a_end),
        slice(r_b_start, h),
        {
            "total_height_px": h,
            "buffer_px": buf,
            "fold_a_row_range": [0, int(r_a_end)],
            "buffer_row_range": [int(r_a_end), int(r_b_start)],
            "fold_b_row_range": [int(r_b_start), h],
            "fold_a_height_px": int(r_a_end),
            "fold_b_height_px": int(h - r_b_start),
        },
    )


def fit_ridge_linear_classifier(
    std_features_d_h_w: np.ndarray,
    target_y_h_w: np.ndarray,
    train_mask_h_w: np.ndarray,
    *,
    l2_lambda: float = RIDGE_LAMBDA,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit L2-regularized Ridge classifier w = (X^T X + lambda I)^{-1} X^T (y - y_bar) on train_mask."""
    feats = np.asarray(std_features_d_h_w, dtype=np.float64)
    y_full = np.asarray(target_y_h_w, dtype=np.float64)
    tm = np.asarray(train_mask_h_w, dtype=bool)
    d_dim = feats.shape[0]
    n_train = int(tm.sum())
    if n_train < d_dim + 4:
        raise ValueError(f"Insufficient training pixels ({n_train}) for D={d_dim} Ridge fit.")

    X = feats[:, tm].T  # (N, D)
    y = y_full[tm]      # (N,)
    # Center y and X on the training mask so intercept b_0 is decoupled from L2 penalty
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean[None, :]
    yc = y - y_mean

    # Scale-normalized effective regularization: lambda * I (locked lambda = 100.0)
    lam = float(l2_lambda)
    gram = Xc.T @ Xc + lam * np.eye(d_dim, dtype=np.float64)
    rhs = Xc.T @ yc
    w = np.linalg.solve(gram, rhs)
    bias = float(y_mean - np.dot(x_mean, w))

    scores_train = X @ w + bias
    # Group weight norms by physical family for interpretability
    w_sym_norm = float(np.linalg.norm(w[0:16]))
    w_antisym_norm = float(np.linalg.norm(w[16:31]))
    w_dog_norm = float(np.linalg.norm(w[31:37]))
    w_grad_oct_wst_norm = float(np.linalg.norm(w[37:42]))
    top_idx = int(np.argmax(np.abs(w)))

    info = {
        "n_train_pixels": n_train,
        "feature_dim": int(d_dim),
        "l2_lambda": lam,
        "target_mean": round(y_mean, 5),
        "target_std": round(float(y.std()), 5),
        "weight_l2_norm": round(float(np.linalg.norm(w)), 6),
        "intercept": round(bias, 6),
        "family_weight_l2_norms": {
            "depth_sym_profile_16d": round(w_sym_norm, 6),
            "depth_antisym_diff_15d": round(w_antisym_norm, 6),
            "multiscale_dog_6d": round(w_dog_norm, 6),
            "grad_oct_wst_5d": round(w_grad_oct_wst_norm, 6),
        },
        "top_weighted_feature": {
            "index": top_idx,
            "name": FEATURE_NAMES[top_idx],
            "weight": round(float(w[top_idx]), 6),
        },
        "train_score_std": round(float(np.std(scores_train)), 6),
    }
    return w.astype(np.float64), bias, info


def apply_frozen_ridge_weights(
    std_features_d_h_w: np.ndarray,
    valid_mask_h_w: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
) -> np.ndarray:
    """Project standardized 42D features onto frozen Ridge weights w (locked polarity '+')."""
    feats = np.asarray(std_features_d_h_w, dtype=np.float64)
    vm = np.asarray(valid_mask_h_w, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    score = np.tensordot(w, feats, axes=(0, 0)) + float(bias)
    return np.where(vm, score, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Held-Out Evaluation with 16-Shift Rolled Null & +/-8-Slice Off-Sheet Kill-Test
# ---------------------------------------------------------------------------

def evaluate_score_map_locked_plus(
    score_map: np.ndarray,
    ink_mask: np.ndarray,
    blank_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    compute_rolled_null: bool = True,
) -> dict[str, Any]:
    """Evaluate a 2D score map under strictly locked polarity '+' (no post-hoc sign flipping)."""
    feat = np.asarray(score_map, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool) & np.isfinite(feat)
    ink = np.asarray(ink_mask, dtype=bool) & vm
    blank = np.asarray(blank_mask, dtype=bool) & vm & (~ink)

    n_ink = int(ink.sum())
    n_blank = int(blank.sum())
    if n_ink < 10 or n_blank < 10:
        return {
            "n_ink": n_ink,
            "n_blank": n_blank,
            "auc_locked_plus": 0.5,
            "unsigned_diagnostic_auc": 0.5,
            "observed_polarity": "+",
            "cohens_d_locked_plus": 0.0,
            "rolled_null_two_sided_mean": 0.5,
            "rolled_null_two_sided_std": 0.0,
            "rolled_null_two_sided_max": 0.5,
            "rolled_null_signed_mean": 0.5,
            "delta_null_locked_plus": 0.0,
        }

    eval_m = ink | blank
    auc_plus = exact_roc_auc(feat[eval_m], ink[eval_m])
    d_plus = cohens_d(feat[ink], feat[blank])

    null_two_mean = 0.5
    null_two_std = 0.0
    null_two_max = 0.5
    null_signed_mean = 0.5
    if compute_rolled_null:
        ink_orig = np.asarray(ink_mask, dtype=bool)
        blank_orig = np.asarray(blank_mask, dtype=bool) & (~ink_orig)
        shifts = build_16_rolled_shifts(feat.shape)
        null_signed: list[float] = []
        null_two: list[float] = []
        for dy, dx in shifts:
            ri = np.roll(ink_orig, shift=(dy, dx), axis=(0, 1)) & vm
            rb = np.roll(blank_orig, shift=(dy, dx), axis=(0, 1)) & vm & (~ri)
            if int(ri.sum()) < 10 or int(rb.sum()) < 10:
                null_signed.append(0.5)
                null_two.append(0.5)
                continue
            em = ri | rb
            a_s = exact_roc_auc(feat[em], ri[em])
            null_signed.append(a_s)
            null_two.append(max(a_s, 1.0 - a_s))
        nt_arr = np.asarray(null_two, dtype=np.float64)
        ns_arr = np.asarray(null_signed, dtype=np.float64)
        null_two_mean = float(nt_arr.mean())
        null_two_std = float(nt_arr.std())
        null_two_max = float(nt_arr.max())
        null_signed_mean = float(ns_arr.mean())

    return {
        "n_ink": n_ink,
        "n_blank": n_blank,
        "auc_locked_plus": round(float(auc_plus), 5),
        "unsigned_diagnostic_auc": round(float(max(auc_plus, 1.0 - auc_plus)), 5),
        "observed_polarity": "+" if auc_plus >= 0.5 else "-",
        "cohens_d_locked_plus": round(float(d_plus), 5),
        "rolled_null_two_sided_mean": round(null_two_mean, 5),
        "rolled_null_two_sided_std": round(null_two_std, 5),
        "rolled_null_two_sided_max": round(null_two_max, 5),
        "rolled_null_signed_mean": round(null_signed_mean, 5),
        "delta_null_locked_plus": round(float(auc_plus - null_two_mean), 5),
    }


def evaluate_frozen_weights_on_and_off_sheet(
    slab: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    oct_half_span_vx: float = 7.5,
    search_half_slices: int = 6,
    z_smooth_sigma: float = 0.8,
    border_guard_px: int = 16,
    off_sheet_shift_slices: int = OFF_SHEET_SHIFT_SLICES,
    off_sheet_subslab_depth: int = OFF_SHEET_SUBSLAB_DEPTH,
    models: dict[str, tuple[np.ndarray, float]],
    label_targets: dict[str, tuple[np.ndarray, np.ndarray]],
    precomputed_on_sheet: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen Ridge weight vectors on-sheet and on +/-8-slice off-sheet sub-slabs."""
    depth = int(slab.shape[0])
    if precomputed_on_sheet is not None:
        on_std_feats, sheet_valid, n_peak, on_diag = precomputed_on_sheet
    else:
        on_std_feats, sheet_valid, n_peak, on_diag = extract_and_standardize_42d_for_slab(
            slab,
            valid_mask,
            voxel_um=voxel_um,
            oct_half_span_vx=oct_half_span_vx,
            search_half_slices=search_half_slices,
            z_smooth_sigma=z_smooth_sigma,
            fixed_center_idx=None,
        )

    eval_valid = sheet_valid.copy()
    guard = min(int(border_guard_px), min(eval_valid.shape) // 8)
    if guard > 0:
        eval_valid[:guard, :] = False
        eval_valid[-guard:, :] = False
        eval_valid[:, :guard] = False
        eval_valid[:, -guard:] = False

    d_sub = min(int(off_sheet_subslab_depth), depth)
    if d_sub % 2 == 0:
        d_sub = max(3, d_sub - 1)
    sub_half = float(d_sub // 2)

    sub_m8, c_m8, info_m8 = extract_shifted_subslab(
        slab, n_peak, -float(off_sheet_shift_slices), subslab_depth=d_sub
    )
    sub_p8, c_p8, info_p8 = extract_shifted_subslab(
        slab, n_peak, +float(off_sheet_shift_slices), subslab_depth=d_sub
    )

    std_m8, _, _, _ = extract_and_standardize_42d_for_slab(
        sub_m8,
        sheet_valid,
        voxel_um=voxel_um,
        oct_half_span_vx=sub_half,
        search_half_slices=1,
        z_smooth_sigma=z_smooth_sigma,
        fixed_center_idx=c_m8,
    )
    std_p8, _, _, _ = extract_and_standardize_42d_for_slab(
        sub_p8,
        sheet_valid,
        voxel_um=voxel_um,
        oct_half_span_vx=sub_half,
        search_half_slices=1,
        z_smooth_sigma=z_smooth_sigma,
        fixed_center_idx=c_p8,
    )

    evaluations: dict[str, dict[str, Any]] = {}
    for m_name, (w_vec, b_val) in models.items():
        score_on = apply_frozen_ridge_weights(on_std_feats, sheet_valid, w_vec, b_val)
        score_m8 = apply_frozen_ridge_weights(std_m8, sheet_valid, w_vec, b_val)
        score_p8 = apply_frozen_ridge_weights(std_p8, sheet_valid, w_vec, b_val)

        per_target: dict[str, Any] = {}
        for t_name, (ink_m, blank_m) in label_targets.items():
            res_on = evaluate_score_map_locked_plus(
                score_on, ink_m, blank_m, eval_valid, compute_rolled_null=True
            )
            res_m8 = evaluate_score_map_locked_plus(
                score_m8, ink_m, blank_m, eval_valid, compute_rolled_null=False
            )
            res_p8 = evaluate_score_map_locked_plus(
                score_p8, ink_m, blank_m, eval_valid, compute_rolled_null=False
            )

            on_auc = float(res_on["auc_locked_plus"])
            d_null = float(res_on["delta_null_locked_plus"])
            m8_auc = float(res_m8["auc_locked_plus"])
            p8_auc = float(res_p8["auc_locked_plus"])
            pm8_mean = 0.5 * (m8_auc + p8_auc)
            pm8_max = max(m8_auc, p8_auc)
            drop_pm8 = on_auc - pm8_mean
            drop_pm8_max = on_auc - pm8_max

            pass_auc = on_auc >= GATE_MIN_ON_SHEET_AUC
            pass_null = d_null >= GATE_MIN_DELTA_NULL
            pass_drop = drop_pm8 >= GATE_MIN_OFF_SHEET_DROP
            full_pass = bool(pass_auc and pass_null and pass_drop)
            if full_pass:
                t_verdict = "PASS"
            elif (pass_auc and pass_null) or (pass_auc and pass_drop):
                t_verdict = "PARTIAL"
            else:
                t_verdict = "CLOSED NO-GO"

            per_target[t_name] = {
                "n_ink": res_on["n_ink"],
                "n_blank": res_on["n_blank"],
                "locked_polarity": "+",
                "observed_polarity": res_on["observed_polarity"],
                "on_sheet_auc": round(on_auc, 5),
                "cohens_d": res_on["cohens_d_locked_plus"],
                "rolled_null_mean": res_on["rolled_null_two_sided_mean"],
                "rolled_null_std": res_on["rolled_null_two_sided_std"],
                "rolled_null_max": res_on["rolled_null_two_sided_max"],
                "rolled_null_signed_mean": res_on["rolled_null_signed_mean"],
                "delta_null": round(d_null, 5),
                "off_sheet_minus8_auc": round(m8_auc, 5),
                "off_sheet_plus8_auc": round(p8_auc, 5),
                "off_sheet_pm8_mean_auc": round(pm8_mean, 5),
                "off_sheet_pm8_max_auc": round(pm8_max, 5),
                "drop_vs_pm8": round(drop_pm8, 5),
                "drop_vs_pm8_max": round(drop_pm8_max, 5),
                "unsigned_diagnostic_on_sheet_auc": res_on["unsigned_diagnostic_auc"],
                "passes_on_sheet_auc": bool(pass_auc),
                "passes_delta_null": bool(pass_null),
                "passes_drop_vs_pm8": bool(pass_drop),
                "passes_all_gates": full_pass,
                "verdict": t_verdict,
            }
        evaluations[m_name] = per_target

    return {
        "diagnostics": on_diag,
        "border_guard_px": int(guard),
        "eval_valid_fraction": round(float(eval_valid.mean()), 4),
        "off_sheet_probes": {
            "shift_minus8": info_m8,
            "shift_plus8": info_p8,
        },
        "models": evaluations,
    }


# ---------------------------------------------------------------------------
# 5. End-to-End Live / Offline Pipeline (<= 100 MiB Global Cap)
# ---------------------------------------------------------------------------

def get_default_experiment_specs() -> dict[str, dict[str, Any]]:
    """Return default public S3 & GitHub specs for PHerc. 0841, PHerc. 0343, and PHerc. 0306B."""
    return {
        "pherc0841_9p366um": {
            "scroll": "PHerc0841",
            "segment": "20260220213127-w00",
            "voxel_um": 9.366,
            "sv_zarr_url": T7_VOLUME_SPECS["pherc0841_9p366um"]["sv_zarr_url"],
            "ink_labels_s3_prefix": "PHerc0841/segments/20260220213127-w00/ink-labels/",
            "argus_ink_tif_url": T7_VOLUME_SPECS["pherc0841_9p366um"]["fallback_ink_tif_url"],
            "border_guard_px": 16,
            "z_smooth_sigma": 0.8,
            "registration_max_shift_px": REGISTRATION_MAX_SHIFT_PX,
        },
        "pherc0343_8p64um": dict(T7_VOLUME_SPECS["pherc0343_8p64um"]),
        "pherc0306b_8p64um": {
            "scroll": "PHerc0306B",
            "segment": "PHerc0306B_20250521133212_w074own_touchup_V5Rc",
            "voxel_um": PHERC0306B_VOXEL_UM,
            "ct_zarr_url": T5_DEFAULT_URLS["pherc0306b_ct_zarr"],
            "mesh_url": T5_DEFAULT_URLS["pherc0306b_mesh"],
            "claim_png_url": T5_DEFAULT_URLS["pherc0306b_claim_png"],
            "claim_json_url": T5_DEFAULT_URLS["pherc0306b_claim_json"],
            "num_slices": PHERC0306B_NUM_SLICES,
            "roi_h": PHERC0306B_ROI_H,
            "roi_w": PHERC0306B_ROI_W,
            "border_guard_px": 16,
            "z_smooth_sigma": 0.8,
        },
    }


def run_villa1739_cross_resolution_bridge(
    *,
    specs: dict[str, dict[str, Any]] | None = None,
    part_budgets_mib: dict[str, float] | None = None,
    global_budget_mib: float = GLOBAL_HTTP_BUDGET_MIB,
    roi_h: int = 512,
    roi_w: int = 640,
    spatial_buffer_px: int = SPATIAL_BUFFER_PX,
    ridge_lambda: float = RIDGE_LAMBDA,
    off_sheet_shift_slices: int = OFF_SHEET_SHIFT_SLICES,
) -> dict[str, Any]:
    """Run Exp T12-Villa1739 end-to-end strictly in RAM on CPU under <= 100 MiB global HTTP cap."""
    t0 = time.monotonic()
    cfg = get_default_experiment_specs()
    if specs:
        for k, v in specs.items():
            cfg[k] = {**cfg.get(k, {}), **v}

    p_mib = dict(DEFAULT_PART_BUDGETS_MIB)
    if part_budgets_mib:
        p_mib.update(part_budgets_mib)

    budget = MultiVolumeHttpByteBudget(
        {k: int(round(v * MIB)) for k, v in p_mib.items()},
        global_limit_bytes=int(round(global_budget_mib * MIB)),
    )

    with ExitStack() as stack:
        # ======================================================================
        # STEP 1: PHerc. 0841 w00 (2.403 um Human Ground Truth -> 9.366 um Bridge)
        # ======================================================================
        spec841 = cfg["pherc0841_9p366um"]
        vox841 = float(spec841["voxel_um"])
        reader841 = stack.enter_context(
            BudgetedRemoteZarrV2(
                spec841["sv_zarr_url"],
                level="0",
                budget=budget,
                part="pherc0841_9p366um",
                label="pherc0841_9p366um:sv_zarr",
                cache_mb=48.0,
                workers=8,
            )
        )
        d841, h841, w841 = [int(s) for s in reader841.shape]
        bridged841 = load_pherc0841_human_2p4um_and_argus_9p366um(
            spec841, (h841, w841), budget=budget, part="pherc0841_9p366um"
        )

        eff_h841 = min(int(roi_h), (h841 // int(reader841.chunks[1])) * int(reader841.chunks[1]))
        eff_w841 = min(int(roi_w), (w841 // int(reader841.chunks[2])) * int(reader841.chunks[2]))
        if "roi_override" in spec841 and spec841["roi_override"] is not None:
            r0, r1, c0, c1 = [int(v) for v in spec841["roi_override"]]
            roi841 = {
                "roi_bounds_yx": [r0, r1, c0, c1],
                "roi_shape_yx": [r1 - r0, c1 - c0],
                "selection_mode": "override",
            }
        else:
            roi841 = select_bridged_roi_0841(
                bridged841["aligned_ink_mask"],
                bridged841["aligned_blank_mask"],
                bridged841["aligned_sup_mask"],
                bridged841["argus_ink_mask"],
                bridged841["argus_blank_mask"],
                (int(reader841.chunks[1]), int(reader841.chunks[2])),
                roi_h=eff_h841,
                roi_w=eff_w841,
                buffer_px=spatial_buffer_px,
            )
        r0_841, r1_841, c0_841, c1_841 = [int(v) for v in roi841["roi_bounds_yx"]]
        n_ch841, planned_b841 = reader841.plan_box_chunks((0, r0_841, c0_841), (d841, r1_841, c1_841))
        budget.check_planned("pherc0841_9p366um", planned_b841, label="pherc0841_9p366um:sv_roi")
        roi841["planned_sv_chunks"] = int(n_ch841)
        roi841["planned_sv_mib"] = round(planned_b841 / MIB, 4)

        # Crop ROI labels and immediately release full-canvas label arrays
        h_ink_roi = bridged841["aligned_ink_mask"][r0_841:r1_841, c0_841:c1_841].copy()
        h_blk_roi = bridged841["aligned_blank_mask"][r0_841:r1_841, c0_841:c1_841].copy()
        h_sup_roi = bridged841["aligned_sup_mask"][r0_841:r1_841, c0_841:c1_841].copy()
        a_u8_roi = bridged841["argus_u8"][r0_841:r1_841, c0_841:c1_841].copy()
        a_ink_roi = bridged841["argus_ink_mask"][r0_841:r1_841, c0_841:c1_841].copy()
        a_blk_roi = bridged841["argus_blank_mask"][r0_841:r1_841, c0_841:c1_841].copy()
        reg_info_841 = bridged841["registration_info"]
        prov_841 = bridged841["provenance"]
        del bridged841

        slab841, _, supp841 = reader841.read_box(
            (0, r0_841, c0_841), (d841, r1_841, c1_841), return_support=True
        )
        valid841 = supp841.all(axis=0) & (slab841.max(axis=0) > 0)

        # Strict Spatial Split into Fold A (train) and Fold B (held-out test) with 32 px buffer zone!
        # Feature extraction & standardization run SEPARATELY on Fold A sub-slab and Fold B sub-slab
        # so zero filter footprint or normalization statistic can ever cross the 32 px buffer!
        sl_a, sl_b, split_info = split_spatial_folds_with_buffer(
            r1_841 - r0_841, buffer_px=spatial_buffer_px
        )

        slab841_a = slab841[:, sl_a, :]
        valid841_a = valid841[sl_a, :]
        std_feats_a, sheet_val_a, _, diag_a = extract_and_standardize_42d_for_slab(
            slab841_a,
            valid841_a,
            voxel_um=vox841,
            oct_half_span_vx=7.5,
            search_half_slices=6,
            z_smooth_sigma=float(spec841.get("z_smooth_sigma", 0.8)),
        )

        # Exclude outer border guard on Fold A before fitting Ridge weights
        train_valid_a = sheet_val_a.copy()
        g_a = min(int(spec841.get("border_guard_px", 16)), min(train_valid_a.shape) // 8)
        if g_a > 0:
            train_valid_a[:g_a, :] = False
            train_valid_a[-g_a:, :] = False
            train_valid_a[:, :g_a] = False
            train_valid_a[:, -g_a:] = False

        # Mode 1: `human_2p4um_bridged` (binary co-registered 2.403 um human ink vs blank labels)
        h_ink_a = h_ink_roi[sl_a, :] & train_valid_a
        h_blk_a = h_blk_roi[sl_a, :] & train_valid_a & (~h_ink_a)
        train_mask_human = h_ink_a | h_blk_a
        target_human_a = h_ink_a.astype(np.float64)
        w_human, b_human, fit_info_human = fit_ridge_linear_classifier(
            std_feats_a, target_human_a, train_mask_human, l2_lambda=ridge_lambda
        )

        # Mode 2: `teacher_2p4um_distilled` (soft argus-v3 probability target inside human supervision)
        train_mask_teacher = h_sup_roi[sl_a, :] & train_valid_a & (a_u8_roi[sl_a, :] > 0)
        target_teacher_a = a_u8_roi[sl_a, :].astype(np.float64) / 255.0
        w_teacher, b_teacher, fit_info_teacher = fit_ridge_linear_classifier(
            std_feats_a, target_teacher_a, train_mask_teacher, l2_lambda=ridge_lambda
        )

        # Diagnostics on Fold A training set
        score_a_human = apply_frozen_ridge_weights(std_feats_a, sheet_val_a, w_human, b_human)
        score_a_teacher = apply_frozen_ridge_weights(std_feats_a, sheet_val_a, w_teacher, b_teacher)
        fit_info_human["train_fold_a_human_auc"] = round(
            float(exact_roc_auc(score_a_human[train_mask_human], h_ink_a[train_mask_human])), 5
        )
        fit_info_teacher["train_fold_a_human_auc"] = round(
            float(exact_roc_auc(score_a_teacher[train_mask_human], h_ink_a[train_mask_human])), 5
        )

        frozen_models: dict[str, tuple[np.ndarray, float]] = {
            "human_2p4um_bridged": (w_human, b_human),
            "teacher_2p4um_distilled": (w_teacher, b_teacher),
        }

        # Evaluate frozen weights on held-out Spatial Fold B of PHerc. 0841 w00 (9.366 um)
        # against BOTH co-registered 2.403 um human labels and 9.366 um argus-v3 reference!
        slab841_b = slab841[:, sl_b, :]
        valid841_b = valid841[sl_b, :]
        eval_841_fold_b = evaluate_frozen_weights_on_and_off_sheet(
            slab841_b,
            valid841_b,
            voxel_um=vox841,
            oct_half_span_vx=7.5,
            search_half_slices=6,
            z_smooth_sigma=float(spec841.get("z_smooth_sigma", 0.8)),
            border_guard_px=int(spec841.get("border_guard_px", 16)),
            off_sheet_shift_slices=off_sheet_shift_slices,
            models=frozen_models,
            label_targets={
                "human_2p4um_coregistered": (h_ink_roi[sl_b, :], h_blk_roi[sl_b, :]),
                "argus_v3_reference": (a_ink_roi[sl_b, :], a_blk_roi[sl_b, :]),
            },
        )
        del slab841, supp841, slab841_a, slab841_b, std_feats_a

        # ======================================================================
        # STEP 2: Cross-Scroll Zero-Shot Transfer to PHerc. 0343 w047-w048 (8.64 um)
        # ======================================================================
        spec343 = cfg["pherc0343_8p64um"]
        vox343 = float(spec343["voxel_um"])
        reader343 = stack.enter_context(
            BudgetedRemoteZarrV2(
                spec343["sv_zarr_url"],
                level="0",
                budget=budget,
                part="pherc0343_8p64um",
                label="pherc0343_8p64um:sv_zarr",
                cache_mb=48.0,
                workers=8,
            )
        )
        d343, h343, w343 = [int(s) for s in reader343.shape]
        tif343_url = str(spec343["fallback_ink_tif_url"])
        if tif343_url.lower().startswith(("http://", "https://")):
            raw_tif343 = budgeted_http_get(
                tif343_url, budget, "pherc0343_8p64um", label="pherc0343_8p64um:argus_v3_tif"
            )
            assert raw_tif343 is not None
            with Image.open(io.BytesIO(raw_tif343)) as im343:
                arr343 = np.array(im343, copy=True)
            del raw_tif343
        else:
            with Image.open(tif343_url) as im343:
                arr343 = np.array(im343, copy=True)
        if arr343.ndim == 3:
            arr343 = arr343[..., 0]
        ink343_u8 = np.ascontiguousarray(arr343.astype(np.uint8, copy=False))
        ink343_mask = ink343_u8 >= INK_HIGH_U8_THRESHOLD
        blk343_mask = (ink343_u8 > 0) & (ink343_u8 <= BLANK_LOW_U8_THRESHOLD)

        if "roi_override" in spec343 and spec343["roi_override"] is not None:
            r0_343, r1_343, c0_343, c1_343 = [int(v) for v in spec343["roi_override"]]
            roi343 = {
                "roi_bounds_yx": [r0_343, r1_343, c0_343, c1_343],
                "roi_shape_yx": [r1_343 - r0_343, c1_343 - c0_343],
                "selection_mode": "override",
            }
        else:
            eff_h343 = min(int(roi_h), (h343 // int(reader343.chunks[1])) * int(reader343.chunks[1]))
            eff_w343 = min(int(roi_w), (w343 // int(reader343.chunks[2])) * int(reader343.chunks[2]))
            roi343 = select_chunk_aligned_roi(
                ink343_mask,
                blk343_mask,
                ink343_u8,
                (int(reader343.chunks[1]), int(reader343.chunks[2])),
                roi_h=eff_h343,
                roi_w=eff_w343,
            )
            r0_343, r1_343, c0_343, c1_343 = [int(v) for v in roi343["roi_bounds_yx"]]

        n_ch343, planned_b343 = reader343.plan_box_chunks((0, r0_343, c0_343), (d343, r1_343, c1_343))
        budget.check_planned("pherc0343_8p64um", planned_b343, label="pherc0343_8p64um:sv_roi")
        roi343["planned_sv_chunks"] = int(n_ch343)
        roi343["planned_sv_mib"] = round(planned_b343 / MIB, 4)

        ink343_roi = ink343_mask[r0_343:r1_343, c0_343:c1_343].copy()
        blk343_roi = blk343_mask[r0_343:r1_343, c0_343:c1_343].copy()
        del ink343_u8, ink343_mask, blk343_mask

        slab343, _, supp343 = reader343.read_box(
            (0, r0_343, c0_343), (d343, r1_343, c1_343), return_support=True
        )
        valid343 = supp343.all(axis=0) & (slab343.max(axis=0) > 0)
        eval_343 = evaluate_frozen_weights_on_and_off_sheet(
            slab343,
            valid343,
            voxel_um=vox343,
            oct_half_span_vx=7.5,
            search_half_slices=6,
            z_smooth_sigma=float(spec343.get("z_smooth_sigma", 0.8)),
            border_guard_px=int(spec343.get("border_guard_px", 16)),
            off_sheet_shift_slices=off_sheet_shift_slices,
            models=frozen_models,
            label_targets={
                "argus_v3_reference": (ink343_roi, blk343_roi),
            },
        )
        del slab343, supp343

        # ======================================================================
        # STEP 3: Cross-Scroll Zero-Shot Transfer to PHerc. 0306B w074 (8.64 um)
        # ======================================================================
        spec306 = cfg["pherc0306b_8p64um"]
        vox306 = float(spec306["voxel_um"])
        num_slices_306 = int(spec306.get("num_slices", PHERC0306B_NUM_SLICES))

        claim_meta_306 = load_json_in_ram(
            spec306["claim_json_url"], budget=budget, part="pherc0306b_8p64um", label="0306b:claim_json"
        )
        mesh_306 = load_tifxyz_in_ram(
            spec306["mesh_url"], budget=budget, part="pherc0306b_8p64um", label="0306b:mesh"
        )
        cx306, cy306, cz306, val306, mesh_meta_306 = mesh_306
        png_306 = load_png_rgb_in_ram(
            spec306["claim_png_url"], budget=budget, part="pherc0306b_8p64um", label="0306b:claim_png"
        )
        align_306 = verify_0306b_alignment(png_306, claim_meta_306, val306, mesh_meta_306)

        reader306 = stack.enter_context(
            BudgetedRemoteZarrV2(
                spec306["ct_zarr_url"],
                level="0",
                budget=budget,
                part="pherc0306b_8p64um",
                label="0306b:ct_zarr",
                cache_mb=64.0,
                workers=8,
            )
        )

        rem_306 = max(0, budget.remaining("pherc0306b_8p64um") - 1 * MIB)
        if "roi_override" in spec306 and spec306["roi_override"] is not None:
            y0_306, y1_306, x0_306, x1_306 = [int(v) for v in spec306["roi_override"]]
            rays_306, m_use_306, coord_306 = build_0306b_rays_for_roi(
                cx306, cy306, cz306, val306,
                claim_meta=claim_meta_306,
                mesh_meta=mesh_meta_306,
                roi_bounds_img_px=(y0_306, y1_306, x0_306, x1_306),
                num_slices=num_slices_306,
                slice_step=1.0,
            )
            n_ch306, planned_b306 = reader306.plan_trilinear_chunks(
                rays_306[:, m_use_306, :], max_direct_box_voxels=1
            )
            roi306 = {
                "roi_bounds_image_px": [y0_306, y1_306, x0_306, x1_306],
                "roi_shape_px": [y1_306 - y0_306, x1_306 - x0_306],
                "planned_zarr_chunks": int(n_ch306),
                "planned_zarr_bytes": int(planned_b306),
                "planned_zarr_mib": round(planned_b306 / MIB, 4),
                "selection_mode": "override",
                **coord_306,
            }
        else:
            roi306 = select_0306b_line1_roi(
                png_306,
                claim_meta_306,
                mesh_306,
                reader306,
                roi_h=int(spec306.get("roi_h", PHERC0306B_ROI_H)),
                roi_w=int(spec306.get("roi_w", PHERC0306B_ROI_W)),
                num_slices=num_slices_306,
                max_planned_bytes=rem_306,
            )
            y0_306, y1_306, x0_306, x1_306 = [int(v) for v in roi306["roi_bounds_image_px"]]
            rays_306, m_use_306, _ = build_0306b_rays_for_roi(
                cx306, cy306, cz306, val306,
                claim_meta=claim_meta_306,
                mesh_meta=mesh_meta_306,
                roi_bounds_img_px=(y0_306, y1_306, x0_306, x1_306),
                num_slices=num_slices_306,
                slice_step=1.0,
            )
            _, planned_b306 = reader306.plan_trilinear_chunks(
                rays_306[:, m_use_306, :], max_direct_box_voxels=1
            )

        budget.check_planned("pherc0306b_8p64um", planned_b306, label="0306b:ct_zarr_roi")
        h306, w306 = y1_306 - y0_306, x1_306 - x0_306
        slab306 = np.zeros((num_slices_306, h306, w306), dtype=np.float32)
        supp306 = np.zeros((num_slices_306, h306, w306), dtype=bool)
        if m_use_306.any():
            vals306, s306 = reader306.sample_trilinear(
                rays_306[:, m_use_306, :], max_direct_box_voxels=1, return_support=True
            )
            slab306[:, m_use_306] = vals306
            supp306[:, m_use_306] = s306

        d_tone_306, on_sheet_png_306, _ = decode_claim_png_tone(
            png_306[y0_306:y1_306, x0_306:x1_306], claim_meta_306
        )
        del png_306
        valid306 = m_use_306 & supp306.all(axis=0) & (slab306.max(axis=0) > 0) & on_sheet_png_306
        ink306_roi = valid306 & (d_tone_306 >= PHERC0306B_INK_D_THRESHOLD)
        blk306_roi = valid306 & (d_tone_306 <= PHERC0306B_BLANK_D_THRESHOLD)

        eval_306 = evaluate_frozen_weights_on_and_off_sheet(
            slab306,
            valid306,
            voxel_um=vox306,
            oct_half_span_vx=7.2,
            search_half_slices=6,
            z_smooth_sigma=float(spec306.get("z_smooth_sigma", 0.8)),
            border_guard_px=int(spec306.get("border_guard_px", 16)),
            off_sheet_shift_slices=off_sheet_shift_slices,
            models=frozen_models,
            label_targets={
                "first_letters_claim_line1": (ink306_roi, blk306_roi),
            },
        )
        del slab306, supp306

    # Summarize per-target verdicts across both supervision modes
    target_verdicts: dict[str, dict[str, str]] = {}
    for mode in SUPERVISION_MODES:
        target_verdicts[mode] = {
            "pherc0841_fold_b_vs_human_2p4um": eval_841_fold_b["models"][mode]["human_2p4um_coregistered"]["verdict"],
            "pherc0841_fold_b_vs_argus_v3": eval_841_fold_b["models"][mode]["argus_v3_reference"]["verdict"],
            "pherc0343_zero_shot_vs_argus_v3": eval_343["models"][mode]["argus_v3_reference"]["verdict"],
            "pherc0306b_zero_shot_vs_claim_png": eval_306["models"][mode]["first_letters_claim_line1"]["verdict"],
        }

    all_verdicts = [
        v for m_dict in target_verdicts.values() for v in m_dict.values()
    ]
    if all(v == "PASS" for v in all_verdicts):
        overall_verdict = "PASS"
    elif any(v in ("PASS", "PARTIAL") for v in all_verdicts):
        overall_verdict = "PARTIAL"
    else:
        overall_verdict = "CLOSED NO-GO"

    w_cos_sim = float(
        np.dot(w_human, w_teacher)
        / max(float(np.linalg.norm(w_human) * np.linalg.norm(w_teacher)), 1e-12)
    )

    receipt: dict[str, Any] = {
        "experiment_id": "exp_t12_villa1739_cross_resolution_bridge_42d_transfer",
        "compute_mode": "CPU_ONLY",
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "villa_rule_context": {
            "rule_id": "villa #1739",
            "rule_summary": "Training or adapting models/labels on 2.4 um scans and running inference on eligible ~9 um scans is permitted.",
            "addresses_question": "collab/chatgpt-031.md open question on bridging human-annotated 2.403 um Zarr v3 ground truth onto 9.366 um surface volumes.",
        },
        "predeclared_gate": {
            "locked_polarity": "+",
            "min_on_sheet_auc": GATE_MIN_ON_SHEET_AUC,
            "min_delta_null": GATE_MIN_DELTA_NULL,
            "min_drop_vs_pm8": GATE_MIN_OFF_SHEET_DROP,
            "off_sheet_shift_slices": int(off_sheet_shift_slices),
            "rolled_null_shifts": 16,
            "spatial_buffer_px": int(spatial_buffer_px),
            "ridge_l2_lambda": float(ridge_lambda),
            "feature_dim": FEATURE_DIM,
        },
        "http_byte_budget": budget.summary(),
        "feature_representation": {
            "feature_dim": FEATURE_DIM,
            "feature_names": list(FEATURE_NAMES),
            "standardization": "per_roi_robust_median_iqr_zero_label_leakage",
        },
        "pherc0841_cross_resolution_bridge": {
            "scroll": "PHerc0841",
            "segment": "20260220213127-w00",
            "source_human_voxel_um": 2.403,
            "target_sv_voxel_um": vox841,
            "surface_volume_source": _sanitize_source_label(spec841["sv_zarr_url"]),
            "surface_volume_shape": [d841, h841, w841],
            "provenance": prov_841,
            "registration_2p4um_to_9p366um": reg_info_841,
            "roi": roi841,
            "spatial_fold_split": split_info,
            "fold_a_training": {
                "diagnostics": diag_a,
                "human_2p4um_bridged": {
                    **fit_info_human,
                    "weights": [round(float(x), 6) for x in w_human],
                },
                "teacher_2p4um_distilled": {
                    **fit_info_teacher,
                    "weights": [round(float(x), 6) for x in w_teacher],
                },
                "human_vs_teacher_weight_cosine_similarity": round(w_cos_sim, 5),
            },
            "fold_b_held_out_evaluation": eval_841_fold_b,
        },
        "cross_scroll_zero_shot_transfer": {
            "pherc0343_8p64um": {
                "scroll": "PHerc0343",
                "segment": str(spec343["segment"]),
                "voxel_um": vox343,
                "surface_volume_source": _sanitize_source_label(spec343["sv_zarr_url"]),
                "reference_label_source": _sanitize_source_label(tif343_url),
                "surface_volume_shape": [d343, h343, w343],
                "roi": roi343,
                "evaluation": eval_343,
            },
            "pherc0306b_8p64um": {
                "scroll": "PHerc0306B",
                "segment": str(spec306["segment"]),
                "voxel_um": vox306,
                "ct_volume_source": _sanitize_source_label(spec306["ct_zarr_url"]),
                "mesh_source": _sanitize_source_label(spec306["mesh_url"]),
                "claim_png_source": _sanitize_source_label(spec306["claim_png_url"]),
                "alignment_verification": align_306,
                "roi": roi306,
                "evaluation": eval_306,
            },
        },
        "verdict": {
            "overall": overall_verdict,
            "per_mode_and_target": target_verdicts,
        },
    }

    assert_no_local_paths(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exp T12-Villa1739: Cross-Resolution 2.403 um -> 9.366 um Human-Label Bridge & 42D Transfer (CPU-only)."
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence" / "villa1739_cross_resolution_bridge.json",
        help="Destination path for sanitized JSON receipt.",
    )
    parser.add_argument("--global-budget-mib", type=float, default=GLOBAL_HTTP_BUDGET_MIB)
    parser.add_argument("--roi-h", type=int, default=512)
    parser.add_argument("--roi-w", type=int, default=640)
    args = parser.parse_args(argv)

    receipt = run_villa1739_cross_resolution_bridge(
        global_budget_mib=args.global_budget_mib,
        roi_h=args.roi_h,
        roi_w=args.roi_w,
    )
    assert_no_local_paths(receipt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    b = receipt["http_byte_budget"]
    reg = receipt["pherc0841_cross_resolution_bridge"]["registration_2p4um_to_9p366um"]
    print(
        f"[Exp T12-Villa1739] verdict={receipt['verdict']['overall']} | "
        f"shift={reg['best_shift_dy_dx']} (IoU {reg['pre_alignment_iou']:.4f} -> {reg['post_alignment_iou']:.4f}) | "
        f"HTTP={b['global_mib']:.2f} / {b['global_limit_mib']:.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
