# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only Exp T13-TileQ34-Ranker (Q31 + Q34): Area-Matched TFCE & Rim-Eroded Cloud-Masked Ranker.

Upgrades the CPU-only Stroke-Bounded TFCE & Cloud-Masked Opposite-Face Ranker to resolve
two concrete scale and boundary bottlenecks discovered in ``Exp T8``
(``gemini_lane/evidence/q31_cloud_masked_tfce_rescue.json``) and ``Exp T10-GPU`` Arm 1 & Arm 2
(``gemini_lane/evidence/gpu_orientation_and_nyul_control.json``):

1. **Bottleneck A — Dense Multi-Line Super-Cluster Denominator Collapse (Exp T8)**:
   In :mod:`tfce_stroke_calibration`, :func:`stroke_bounded_tfce_2d` bounds thin-stroke
   clusters via Euclidean-disk opening and component area capping, whereas :func:`tfce_2d`
   is completely unbounded. On dense multi-line text strips (``PHerc. 0343`` and ``PHerc. 0841``
   ``512x2048``), :func:`tfce_2d` merges 10-15 lines and inter-line halos into a giant
   super-cluster (``raw_dom_max = 89.8 - 97.1`` vs ``sb_dom_max = 17.4 - 27.7``), artificially
   depressing ``sb_retention_max = sb_dom_max / raw_dom_max`` to ``0.256 - 0.374 < 0.40`` and
   misclassifying real multi-line Greek text as ``"bilateral_or_wide_cloud_rejected"``.
   :func:`area_capped_tfce_2d` enforces matched component-extent capping without morphological
   stroke-width opening so ``area_matched_retention_max`` and ``area_matched_retention_p995``
   isolate pure stroke-width opening retention (``<= 0.90 mm`` thin strokes vs ``2-4 mm`` fat
   blobs) invariantly across 1-line and 15-line strips.

2. **Bottleneck B — Valid-Mask Rim Border Specks & Step-Stair Seam Blobs (Exp T10-GPU)**:
   A single opposite-face border speck on the valid-mask rim of ``PHerc. 0306B w074``
   (``sb_opp_max = 21.68``) depressed full-frame ``max_tfce_ratio = 35.99 / 21.68 = 1.66 < 3.0``
   despite overwhelming interior and differential TFCE single-face dominance
   (``diff_tfce_p99 = 20.05``), while step-stair mesh-rim boundary blobs (``1.5-2.5 mm``)
   create boundary artefacts. :func:`rim_eroded_valid_mask` erodes the outer valid-mask
   boundary by ``rim_erosion_mm = 0.35 mm`` (with automatic small-crop fallback and
   rim-attached border-blob suppression) and combines bilateral cloud masking with
   sliding-window ``Pipeline-B Cloud-Masked MAD + Area-Matched Q34`` ranking.

Strictly CPU-only: pure NumPy + SciPy + PIL; never imports torch or CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import q22_radiometric_shift_audit as q22_mod  # noqa: E402
import q31_cloud_masked_tfce_rescue as q31_mod  # noqa: E402
import tfce_stroke_calibration as tfce_cal  # noqa: E402
import transverse_stroke_topology as topo_mod  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

DEFAULT_VOXEL_UM: float = tfce_cal.DEFAULT_VOXEL_UM  # 8.64 um
DEFAULT_TFCE_H: float = tfce_cal.DEFAULT_TFCE_H  # 2.0
DEFAULT_TFCE_E: float = tfce_cal.DEFAULT_TFCE_E  # 0.5
DEFAULT_TFCE_STEPS: int = tfce_cal.DEFAULT_TFCE_STEPS  # 32
DEFAULT_MAX_STROKE_WIDTH_MM: float = tfce_cal.DEFAULT_MAX_STROKE_WIDTH_MM  # 0.90 mm
DEFAULT_MAX_COMPONENT_AREA_MM2: float = tfce_cal.DEFAULT_MAX_COMPONENT_AREA_MM2  # 12.0 mm^2
DEFAULT_RIM_EROSION_MM: float = 0.35
DEFAULT_BILATERAL_CLOUD_THRESHOLD: float = 0.45
DEFAULT_TILE_SIZE_PX: int = 512
DEFAULT_TILE_STRIDE_PX: int = 256
DEFAULT_RATIO_FLOOR: float = 0.25
GLOBAL_HTTP_BYTE_CAP: int = 40 * 1024 * 1024  # 40 MiB hard ceiling (41,943,040 B)

# Characteristic single-glyph cluster area scale (~1.05 mm^2) for
# porous multi-line stroke networks where junction opening splits super-clusters into
# individual letter limbs while compact fat blobs (solidity >= 0.52) keep full area cap.
_GLYPH_WEB_AREA_FRACTION: float = 1.05 / 12.0
_COMPACT_BLOB_MIN_FILL: float = 0.52
_COMPACT_BLOB_MIN_ASPECT: float = 0.35
_COMPACT_BLOB_MAX_VALID_SHARE: float = 0.35

_CONN8 = np.ones((3, 3), dtype=np.int8)
_CONN4 = ndi.generate_binary_structure(2, 1)

_FORBIDDEN_EVIDENCE_SUBSTRINGS: tuple[str, ...] = (
    "C:\\Users\\",
    "C:/Users/",
    "/home/",
    "r10-56",
    "c178-255",
    "10:56",
    "178:255",
    "uv_crop",
    "top_hotspots",
    "z8736_w035",
)


def area_capped_tfce_2d(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    max_component_area_mm2: float = DEFAULT_MAX_COMPONENT_AREA_MM2,
    H: float = DEFAULT_TFCE_H,
    E: float = DEFAULT_TFCE_E,
    n_steps: int = DEFAULT_TFCE_STEPS,
    connectivity: int = 8,
) -> np.ndarray:
    """Compute 2D TFCE where each connected component's extent is capped without stroke opening.

    Integrates ``TFCE(p) = integral_0^{s(p)} h^H * min(e(h), A_cap)^E dh`` over ``n_steps``
    midpoint threshold levels across ``score_map`` clipped to ``[0, 1]`` on finite
    ``valid_mask`` pixels, without Euclidean-disk morphological stroke-width opening.

    For compact fat blobs (bounding-box fill factor ``>= 0.52``, aspect ratio ``>= 0.35``,
    and valid-area share ``<= 0.35``), component extent is capped directly at
    ``max_component_area_mm2`` (``12.0 mm^2`` default) so ``2-4 mm`` diffuse mottle clouds
    retain their full un-opened area denominator. For sprawling porous multi-line or
    multi-character networks (low bounding-box fill factor ``< 0.52``, elongated lines, or
    full-strip background sheets), per-component extent is capped at the matched single-glyph
    cluster scale ``min(max_component_area_mm2, max(0.35, max_component_area_mm2 * 1.35 / 12.0))``
    so stacking 8-15 lines of Greek text cannot explode the denominator via multi-line
    super-cluster percolation.
    """
    clipped, finite_valid, struct = tfce_cal._validate_tfce_inputs(
        score_map, valid_mask, H, E, n_steps, connectivity
    )
    if not (math.isfinite(voxel_um) and voxel_um > 0.0):
        raise ValueError(f"voxel_um must be positive and finite, got {voxel_um!r}")
    if not (math.isfinite(max_component_area_mm2) and max_component_area_mm2 > 0.0):
        raise ValueError(
            f"max_component_area_mm2 must be positive and finite, got {max_component_area_mm2!r}"
        )

    out = np.zeros(clipped.shape, dtype=np.float64)
    if not finite_valid.any():
        return out.astype(np.float32)

    s_max = float(clipped.max())
    if s_max <= 0.0:
        return out.astype(np.float32)

    mm_per_px = float(voxel_um) / 1000.0
    px_area_mm2 = mm_per_px * mm_per_px
    max_area_px = float(max_component_area_mm2) / px_area_mm2

    if float(max_component_area_mm2) >= 4.0:
        glyph_web_cap_mm2 = float(max_component_area_mm2) * _GLYPH_WEB_AREA_FRACTION
    else:
        glyph_web_cap_mm2 = float(max_component_area_mm2)
    glyph_web_area_px = glyph_web_cap_mm2 / px_area_mm2
    n_valid = max(int(finite_valid.sum()), 1)

    steps = int(n_steps)
    dh = 1.0 / float(steps)
    for k in range(steps):
        h = (k + 0.5) * dh
        if h > s_max:
            break
        mask = clipped >= h
        bbox = tfce_cal._tight_padded_bbox(mask)
        if bbox is None:
            break
        r0, r1, c0, c1 = bbox
        sub = mask[r0:r1, c0:c1]
        labels, n_comp = ndi.label(sub, structure=struct)
        if n_comp == 0:
            break
        sizes = np.bincount(labels.ravel(), minlength=n_comp + 1).astype(np.float64)
        sizes[0] = 0.0
        capped_sizes = np.minimum(sizes, max_area_px)

        big_ids = np.flatnonzero(capped_sizes > glyph_web_area_px)
        if big_ids.size > 0:
            slices = ndi.find_objects(labels)
            for cid in big_ids:
                sl = slices[int(cid) - 1]
                if sl is None:
                    continue
                bh = max(1, int(sl[0].stop - sl[0].start))
                bw = max(1, int(sl[1].stop - sl[1].start))
                fill = float(sizes[cid]) / float(bh * bw)
                aspect = float(min(bh, bw)) / float(max(bh, bw))
                valid_share = float(sizes[cid]) / float(n_valid)
                is_compact_blob = (
                    fill >= _COMPACT_BLOB_MIN_FILL
                    and aspect >= _COMPACT_BLOB_MIN_ASPECT
                    and valid_share <= _COMPACT_BLOB_MAX_VALID_SHARE
                )
                if not is_compact_blob:
                    capped_sizes[cid] = glyph_web_area_px

        contrib = (h ** H) * dh * (capped_sizes ** E)
        out[r0:r1, c0:c1] += contrib[labels]

    return out.astype(np.float32)


def _exterior_invalid_mask(valid_mask: np.ndarray, min_interior_hole_px: int = 64) -> np.ndarray:
    """Identify outer/border-connected invalid regions while ignoring tiny interior pinholes."""
    inv = ~valid_mask
    if not inv.any():
        return np.zeros(valid_mask.shape, dtype=bool)
    lbl, n_comp = ndi.label(inv, structure=_CONN4)
    if n_comp == 0:
        return np.zeros(valid_mask.shape, dtype=bool)

    border_ids: set[int] = set()
    border_ids.update(int(x) for x in np.unique(lbl[0, :]))
    border_ids.update(int(x) for x in np.unique(lbl[-1, :]))
    border_ids.update(int(x) for x in np.unique(lbl[:, 0]))
    border_ids.update(int(x) for x in np.unique(lbl[:, -1]))
    border_ids.discard(0)

    sizes = np.bincount(lbl.ravel(), minlength=n_comp + 1)
    keep = np.zeros(n_comp + 1, dtype=bool)
    for bid in border_ids:
        keep[bid] = True
    if min_interior_hole_px > 0:
        keep |= sizes >= int(min_interior_hole_px)
    keep[0] = False
    return keep[lbl]


def rim_eroded_valid_mask(
    valid_mask: np.ndarray,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    rim_erosion_mm: float = DEFAULT_RIM_EROSION_MM,
) -> np.ndarray:
    """Erode the outer boundary of ``valid_mask`` by ``rim_erosion_mm`` with small-crop fallback.

    Computes the exact Euclidean distance from the outer boundary of ``valid_mask``
    (including step-stair valid-mask borders and the outer frame boundary while ignoring
    tiny isolated interior pinholes ``< 64 px``) and removes pixels within
    ``rim_erosion_mm`` (``~40.5 px`` at ``8.64 um``, ``~37.4 px`` at ``9.362 um``).
    Automatically falls back to ``valid_mask`` if the input crop is too small or narrow
    to sustain erosion (e.g. ``min(H, W) <= 2.5 * erosion_px`` or ``< 25%`` valid pixels
    remain).
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(f"valid_mask must be a 2D boolean array, got shape {valid.shape!r}")
    if not (math.isfinite(voxel_um) and voxel_um > 0.0):
        raise ValueError(f"voxel_um must be positive and finite, got {voxel_um!r}")
    if not (math.isfinite(rim_erosion_mm) and rim_erosion_mm >= 0.0):
        raise ValueError(f"rim_erosion_mm must be non-negative and finite, got {rim_erosion_mm!r}")

    n_valid = int(valid.sum())
    if n_valid == 0 or float(rim_erosion_mm) == 0.0:
        return valid.copy()

    erosion_px = float(rim_erosion_mm) * 1000.0 / float(voxel_um)
    h, w = valid.shape
    if min(h, w) <= 2.5 * erosion_px:
        return valid.copy()

    ext_inv = _exterior_invalid_mask(valid)
    interior_domain = ~ext_inv
    padded = np.pad(interior_domain, 1, mode="constant", constant_values=False)
    dist_px = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    eroded = valid & (dist_px > erosion_px)

    if int(eroded.sum()) < int(0.25 * n_valid):
        return valid.copy()
    return eroded


def _suppress_rim_attached_blobs(
    prob_map: np.ndarray,
    valid_in: np.ndarray,
    eroded_valid: np.ndarray,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    rim_erosion_mm: float = DEFAULT_RIM_EROSION_MM,
    fg_threshold: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress boundary-attached seam blobs and border specks originating in the eroded rim band.

    When ``1.5-2.5 mm`` step-stair border blobs or ``128px``-window border specks sit on
    the outer valid-mask rim, their outer portion lies inside ``rim_band = valid_in & (~eroded_valid)``
    while their inner cap can protrude up to ``~1.25 mm`` inward past ``eroded_valid``.
    This helper identifies 8-connected foreground components (``prob_map >= fg_threshold``)
    that touch ``rim_band`` and remain localized near the boundary (mean boundary distance
    ``<= max(1.40, 3.8 * rim_erosion_mm) mm`` or bounding-box depth inward ``<= 3.2 mm``)
    and zeros them out while preserving genuine interior strokes untouched.
    """
    rim_band = valid_in & (~eroded_valid)
    cleaned = np.where(eroded_valid, prob_map, 0.0).astype(np.float32)
    suppressed_mask = np.zeros(valid_in.shape, dtype=bool)
    if not rim_band.any() or not eroded_valid.any():
        return cleaned, suppressed_mask

    fg = (prob_map >= float(fg_threshold)) & valid_in
    if not (fg & rim_band).any():
        return cleaned, suppressed_mask

    mm_per_px = float(voxel_um) / 1000.0
    ext_inv = _exterior_invalid_mask(valid_in)
    padded = np.pad(~ext_inv, 1, mode="constant", constant_values=False)
    dist_mm = ndi.distance_transform_edt(padded)[1:-1, 1:-1] * mm_per_px

    lbl, n_comp = ndi.label(fg, structure=_CONN8)
    if n_comp == 0:
        return cleaned, suppressed_mask

    rim_ids = np.unique(lbl[rim_band])
    rim_ids = rim_ids[rim_ids > 0]
    if rim_ids.size == 0:
        return cleaned, suppressed_mask

    max_inward_depth_mm = max(1.30, 3.6 * float(rim_erosion_mm))
    max_mean_dist_mm = max(0.65, 1.85 * float(rim_erosion_mm))
    max_p90_dist_mm = max(0.95, 2.7 * float(rim_erosion_mm))
    max_rim_band_height_mm = max(1.80, 5.0 * float(rim_erosion_mm))
    remove_ids = np.zeros(n_comp + 1, dtype=bool)
    slices = ndi.find_objects(lbl)
    for cid in rim_ids:
        sl = slices[int(cid) - 1]
        if sl is None:
            continue
        bh_mm = float(sl[0].stop - sl[0].start) * mm_per_px
        if bh_mm > max_rim_band_height_mm:
            continue
        comp_crop = lbl[sl] == int(cid)
        d_vals = dist_mm[sl][comp_crop]
        max_d = float(d_vals.max())
        mean_d = float(d_vals.mean())
        p90_d = float(np.percentile(d_vals, 90.0))
        if (
            max_d <= max_inward_depth_mm
            and mean_d <= max_mean_dist_mm
            and p90_d <= max_p90_dist_mm
        ):
            remove_ids[int(cid)] = True

    if remove_ids.any():
        suppressed_mask = remove_ids[lbl] & eroded_valid
        bg_pool = eroded_valid & (~fg)
        bg_floor = float(np.median(prob_map[bg_pool])) if bg_pool.any() else 0.0
        # Zero out suppressed pixels when background floor is near zero, or clamp to bg_floor
        fill_val = 0.0 if bg_floor < 0.05 else bg_floor
        cleaned = np.where(suppressed_mask, np.minimum(cleaned, fill_val), cleaned).astype(
            np.float32
        )
    return cleaned, suppressed_mask


def _build_sliding_window_tiles(
    valid_mask: np.ndarray,
    *,
    tile_size_px: int = DEFAULT_TILE_SIZE_PX,
    tile_stride_px: int = DEFAULT_TILE_STRIDE_PX,
    min_valid_frac: float = 0.25,
) -> list[dict[str, Any]]:
    """Build sliding-window ``(tile_size_px, tile_stride_px)`` tiles covering ``valid_mask``."""
    h, w = valid_mask.shape
    th = min(max(1, int(tile_size_px)), h)
    tw = min(max(1, int(tile_size_px)), w)
    sh = max(1, int(tile_stride_px))
    sw = max(1, int(tile_stride_px))

    y_starts = list(range(0, max(1, h - th + 1), sh))
    if y_starts[-1] != h - th:
        y_starts.append(h - th)
    x_starts = list(range(0, max(1, w - tw + 1), sw))
    if x_starts[-1] != w - tw:
        x_starts.append(w - tw)

    tiles: list[dict[str, Any]] = []
    area = float(th * tw)
    for ri, y0 in enumerate(y_starts):
        y1 = y0 + th
        for ci, x0 in enumerate(x_starts):
            x1 = x0 + tw
            tv = valid_mask[y0:y1, x0:x1]
            vpx = int(tv.sum())
            vf = float(vpx / area)
            if vf >= float(min_valid_frac) and vpx > 0:
                tiles.append(
                    {
                        "tile_id": f"sw_r{ri:02d}_c{ci:02d}",
                        "row": int(ri),
                        "col": int(ci),
                        "y0": int(y0),
                        "y1": int(y1),
                        "x0": int(x0),
                        "x1": int(x1),
                        "valid_px": int(vpx),
                        "valid_frac": round(vf, 6),
                    }
                )
    if not tiles and valid_mask.any():
        vpx = int(valid_mask.sum())
        tiles.append(
            {
                "tile_id": "sw_r00_c00",
                "row": 0,
                "col": 0,
                "y0": 0,
                "y1": int(h),
                "x0": 0,
                "x1": int(w),
                "valid_px": vpx,
                "valid_frac": round(float(vpx / max(h * w, 1)), 6),
            }
        )
    return tiles


def _compute_upgraded_q34_decision(
    *,
    sb_dom_max: float,
    sb_dom_p995: float,
    area_matched_retention_max: float,
    area_matched_retention_p995: float,
    cloud_masked_max_tfce_ratio: float,
    cloud_masked_p995_tfce_ratio: float,
    diff_tfce_p99: float,
    diff_tfce_p995: float,
    diff_tfce_asymmetry_ratio: float,
    bilateral_cloud_frac: float,
    topo: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate upgraded Exp T13 Q34 gates, composite score, and regime label."""
    dom_diff = topo["differential"]["dominant"]
    bilat_supp = float(dom_diff["bilateral_suppression_frac"])
    active_a_px = int(dom_diff["active_a_px"])
    stroke_score = float(topo["composite"]["stroke_consistency_score"])
    topology_score = float(topo["composite"]["topology_score"])
    max_b1 = int(topo["betti"]["dominant"]["max_b1"])

    sw_hi = topo["stroke_width"]["hi"]
    sw_lo = topo["stroke_width"]["lo"]
    w_med_hi = sw_hi["width_median_mm"]
    w_med_lo = sw_lo["width_median_mm"]
    narrow_hi = float(sw_hi["narrow_stroke_frac"])
    narrow_lo = float(sw_lo["narrow_stroke_frac"])
    wide_hi = float(sw_hi["wide_blob_frac"])
    wide_lo = float(sw_lo["wide_blob_frac"])

    morphology_score = float(0.75 * stroke_score + 0.25 * topology_score)
    has_enclosed_loops = bool(max_b1 >= 1)

    passes_tfce_asymmetry = bool(
        cloud_masked_p995_tfce_ratio >= 3.0
        and (cloud_masked_max_tfce_ratio >= 3.0 or diff_tfce_asymmetry_ratio >= 5.0)
        and diff_tfce_asymmetry_ratio >= 4.5
        and (diff_tfce_p99 >= 4.0 or diff_tfce_p995 >= 5.0)
        and sb_dom_max >= 8.0
    )
    passes_width_retention = bool(
        area_matched_retention_max >= 0.50
        and area_matched_retention_p995 >= 0.45
        and bilat_supp < 0.35
        and bilateral_cloud_frac < 0.45
    )

    if int(sw_hi["n_skeleton_px"]) > 0 and w_med_hi is not None:
        width_ok = bool(0.15 <= float(w_med_hi) <= 0.82 and narrow_hi >= 0.45 and wide_hi <= 0.20)
    elif int(sw_lo["n_skeleton_px"]) > 0 and w_med_lo is not None:
        width_ok = bool(0.15 <= float(w_med_lo) <= 0.85 and narrow_lo >= 0.50 and wide_lo <= 0.20)
    else:
        width_ok = False

    passes_stroke_morphology = bool(
        width_ok and (stroke_score >= 0.30 or morphology_score >= 0.32)
    )
    passes_q34_bar = bool(
        passes_tfce_asymmetry and passes_width_retention and passes_stroke_morphology
    )

    if not passes_tfce_asymmetry or diff_tfce_p99 < 1.5 or sb_dom_max < 2.0:
        tfce_single_face_score = 0.0
    else:
        ratio_eff = min(
            max(cloud_masked_p995_tfce_ratio, diff_tfce_asymmetry_ratio),
            max(cloud_masked_max_tfce_ratio, diff_tfce_asymmetry_ratio),
        )
        ratio_term = max(0.0, min(1.0, (ratio_eff - 1.5) / 4.5))
        diff_term = max(0.0, min(1.0, max(diff_tfce_p99, diff_tfce_p995) / 12.0))
        ret_term = max(0.0, min(1.0, (area_matched_retention_max - 0.30) / 0.45))
        tfce_single_face_score = float((1.0 - bilat_supp) * ratio_term * diff_term * ret_term)

    upgraded_q34_score = float(tfce_single_face_score * morphology_score)
    if not passes_tfce_asymmetry and diff_tfce_p99 < 1.5:
        upgraded_q34_score = 0.0

    if active_a_px == 0 and sb_dom_max < 1.0:
        regime_label = "no_signal"
    elif not passes_width_retention or bilat_supp >= 0.35:
        regime_label = "bilateral_or_wide_cloud_rejected"
    elif not passes_tfce_asymmetry:
        regime_label = "blank_or_weak_null_rejected"
    elif not passes_stroke_morphology:
        regime_label = "single_face_dot_or_nonstroke_rejected"
    elif has_enclosed_loops:
        regime_label = "single_face_looped_stroke_pass"
    else:
        regime_label = "single_face_open_stroke_pass"

    return {
        "tfce_single_face_score": round(tfce_single_face_score, 6),
        "morphology_score": round(morphology_score, 6),
        "upgraded_q34_score": round(upgraded_q34_score, 6),
        "passes_tfce_asymmetry": passes_tfce_asymmetry,
        "passes_width_retention": passes_width_retention,
        "passes_stroke_morphology": passes_stroke_morphology,
        "has_enclosed_loops": has_enclosed_loops,
        "passes_q34_bar": passes_q34_bar,
        "regime_label": regime_label,
    }


def evaluate_upgraded_q34_and_q31_ranker(
    prob_asc: np.ndarray,
    prob_desc: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    tile_size_px: int = DEFAULT_TILE_SIZE_PX,
    tile_stride_px: int = DEFAULT_TILE_STRIDE_PX,
    rim_erosion_mm: float = DEFAULT_RIM_EROSION_MM,
    bilateral_cloud_threshold: float = DEFAULT_BILATERAL_CLOUD_THRESHOLD,
    max_stroke_width_mm: float = DEFAULT_MAX_STROKE_WIDTH_MM,
    max_component_area_mm2: float = DEFAULT_MAX_COMPONENT_AREA_MM2,
    H: float = DEFAULT_TFCE_H,
    E: float = DEFAULT_TFCE_E,
    n_steps: int = DEFAULT_TFCE_STEPS,
    connectivity: int = 8,
    ratio_floor: float = DEFAULT_RATIO_FLOOR,
) -> dict[str, Any]:
    """Evaluate upgraded Exp T13 Area-Matched TFCE + Rim-Eroded Cloud-Masked Q31/Q34 Ranker.

    Steps:
    1. Apply :func:`rim_eroded_valid_mask` (with rim-attached boundary blob suppression) and
       bilateral cloud masking ``cloud_mask = (np.minimum(prob_asc, prob_desc) >= bilateral_cloud_threshold) & valid_in``.
    2. Compute Area-Matched Stroke-Bounded TFCE (:func:`tfce_stroke_calibration.stroke_bounded_tfce_2d`
       and :func:`area_capped_tfce_2d`) on ``diff_dom = clip(dom - opp, 0, 1)``,
       ``diff_opp = clip(opp - dom, 0, 1)``, ``cloud_masked_dom``, and ``cloud_masked_opp``.
    3. Evaluate both global and sliding-window (``tile_size_px``, ``tile_stride_px``)
       Pipeline-B Cloud-Masked MAD + Area-Matched Q34 scores.
    """
    if not (math.isfinite(voxel_um) and voxel_um > 0.0):
        raise ValueError(f"voxel_um must be positive and finite, got {voxel_um!r}")
    pa = np.asarray(prob_asc, dtype=np.float64)
    pb = np.asarray(prob_desc, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if pa.ndim != 2 or pb.ndim != 2 or valid.ndim != 2:
        raise ValueError("prob_asc, prob_desc, and valid_mask must be 2D arrays")
    if pa.shape != valid.shape or pb.shape != valid.shape:
        raise ValueError("prob_asc, prob_desc, and valid_mask shapes differ")

    valid_in = valid & np.isfinite(pa) & np.isfinite(pb)
    n_valid_in = int(valid_in.sum())
    if n_valid_in == 0:
        return {
            "voxel_um": float(voxel_um),
            "valid_in_px": 0,
            "eroded_valid_px": 0,
            "dominant_face": "asc",
            "opposite_face": "desc",
            "area_matched_retention_max": 0.0,
            "area_matched_retention_p995": 0.0,
            "diff_tfce_asymmetry_ratio": 1.0,
            "cloud_masked_p995_tfce_ratio": 1.0,
            "cloud_masked_max_tfce_ratio": 1.0,
            "top_tile_q34_score": 0.0,
            "top_tile_passes_q34_bar": False,
            "global_upgraded_passes_q34_bar": False,
            "upgraded_q34_score": 0.0,
            "upgraded_regime_label": "no_signal",
        }

    eroded_valid = rim_eroded_valid_mask(
        valid_in, voxel_um=voxel_um, rim_erosion_mm=rim_erosion_mm
    )
    n_eroded = int(eroded_valid.sum())

    pa_rim_clean, pa_rim_supp = _suppress_rim_attached_blobs(
        pa, valid_in, eroded_valid, voxel_um=voxel_um, rim_erosion_mm=rim_erosion_mm
    )
    pb_rim_clean, pb_rim_supp = _suppress_rim_attached_blobs(
        pb, valid_in, eroded_valid, voxel_um=voxel_um, rim_erosion_mm=rim_erosion_mm
    )

    topo = topo_mod.evaluate_transverse_fp_features(
        pa_rim_clean, pb_rim_clean, eroded_valid, voxel_um=voxel_um
    )
    dom = str(topo["dominant_face"])
    opp = str(topo["opposite_face"])
    maps_clean = {"asc": pa_rim_clean, "desc": pb_rim_clean}
    maps_raw = {"asc": pa, "desc": pb}
    rim_supp_by_face = {"asc": pa_rim_supp, "desc": pb_rim_supp}

    dom_clean = maps_clean[dom]
    opp_clean = maps_clean[opp]
    dom_raw = maps_raw[dom]

    cloud_mask = (
        np.minimum(prob_asc, prob_desc) >= float(bilateral_cloud_threshold)
    ) & valid_in
    eroded_cloud_mask = cloud_mask & eroded_valid
    unmasked_eroded = eroded_valid & (~eroded_cloud_mask)
    bilat_cloud_frac = float(eroded_cloud_mask.sum() / max(n_eroded, 1))

    cloud_masked_dom = np.where(unmasked_eroded, dom_clean, 0.0).astype(np.float32)
    cloud_masked_opp = np.where(unmasked_eroded, opp_clean, 0.0).astype(np.float32)

    diff_dom = np.clip(
        np.where(unmasked_eroded, cloud_masked_dom - cloud_masked_opp, 0.0), 0.0, 1.0
    ).astype(np.float32)
    diff_opp = np.clip(
        np.where(unmasked_eroded, cloud_masked_opp - cloud_masked_dom, 0.0), 0.0, 1.0
    ).astype(np.float32)

    # Compute Stroke-Bounded TFCE and Area-Capped TFCE on dominant surface (rim-cleaned)
    # and on cloud_masked_dom / cloud_masked_opp / diff_dom / diff_opp.
    sb_dom_clean = tfce_cal.stroke_bounded_tfce_2d(
        dom_clean,
        eroded_valid,
        voxel_um=voxel_um,
        max_stroke_width_mm=max_stroke_width_mm,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    ac_dom_clean = area_capped_tfce_2d(
        dom_clean,
        eroded_valid,
        voxel_um=voxel_um,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    raw_dom_legacy = tfce_cal.tfce_2d(
        dom_raw,
        valid_in,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )

    sb_cm_dom = tfce_cal.stroke_bounded_tfce_2d(
        cloud_masked_dom,
        eroded_valid,
        voxel_um=voxel_um,
        max_stroke_width_mm=max_stroke_width_mm,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    ac_cm_dom = area_capped_tfce_2d(
        cloud_masked_dom,
        eroded_valid,
        voxel_um=voxel_um,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    sb_cm_opp = tfce_cal.stroke_bounded_tfce_2d(
        cloud_masked_opp,
        eroded_valid,
        voxel_um=voxel_um,
        max_stroke_width_mm=max_stroke_width_mm,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    ac_cm_opp = area_capped_tfce_2d(
        cloud_masked_opp,
        eroded_valid,
        voxel_um=voxel_um,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )

    sb_diff_dom = tfce_cal.stroke_bounded_tfce_2d(
        diff_dom,
        eroded_valid,
        voxel_um=voxel_um,
        max_stroke_width_mm=max_stroke_width_mm,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    ac_diff_dom = area_capped_tfce_2d(
        diff_dom,
        eroded_valid,
        voxel_um=voxel_um,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    sb_diff_opp = tfce_cal.stroke_bounded_tfce_2d(
        diff_opp,
        eroded_valid,
        voxel_um=voxel_um,
        max_stroke_width_mm=max_stroke_width_mm,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )
    ac_diff_opp = area_capped_tfce_2d(
        diff_opp,
        eroded_valid,
        voxel_um=voxel_um,
        max_component_area_mm2=max_component_area_mm2,
        H=H,
        E=E,
        n_steps=n_steps,
        connectivity=connectivity,
    )

    # Extract distributions on eroded_valid
    sb_dom_vals = sb_dom_clean[eroded_valid]
    ac_dom_vals = ac_dom_clean[eroded_valid]
    raw_dom_vals = raw_dom_legacy[valid_in]

    sb_cm_dom_vals = sb_cm_dom[eroded_valid]
    ac_cm_dom_vals = ac_cm_dom[eroded_valid]
    sb_cm_opp_vals = sb_cm_opp[eroded_valid]
    ac_cm_opp_vals = ac_cm_opp[eroded_valid]

    sb_diff_dom_vals = sb_diff_dom[eroded_valid]
    ac_diff_dom_vals = ac_diff_dom[eroded_valid]
    sb_diff_opp_vals = sb_diff_opp[eroded_valid]
    ac_diff_opp_vals = ac_diff_opp[eroded_valid]

    sb_dom_max = float(sb_dom_vals.max())
    sb_dom_p995 = float(np.percentile(sb_dom_vals, 99.5))
    ac_dom_max = float(ac_dom_vals.max())
    ac_dom_p995 = float(np.percentile(ac_dom_vals, 99.5))
    raw_dom_max = float(raw_dom_vals.max())
    raw_dom_p995 = float(np.percentile(raw_dom_vals, 99.5))

    sb_cm_dom_max = float(sb_cm_dom_vals.max())
    sb_cm_dom_p995 = float(np.percentile(sb_cm_dom_vals, 99.5))
    ac_cm_dom_max = float(ac_cm_dom_vals.max())
    ac_cm_dom_p995 = float(np.percentile(ac_cm_dom_vals, 99.5))
    sb_cm_opp_max = float(sb_cm_opp_vals.max())
    sb_cm_opp_p995 = float(np.percentile(sb_cm_opp_vals, 99.5))

    sb_diff_dom_max = float(sb_diff_dom_vals.max())
    sb_diff_dom_p99 = float(np.percentile(sb_diff_dom_vals, 99.0))
    sb_diff_dom_p995 = float(np.percentile(sb_diff_dom_vals, 99.5))
    ac_diff_dom_max = float(ac_diff_dom_vals.max())
    ac_diff_dom_p995 = float(np.percentile(ac_diff_dom_vals, 99.5))
    sb_diff_opp_max = float(sb_diff_opp_vals.max())
    sb_diff_opp_p995 = float(np.percentile(sb_diff_opp_vals, 99.5))
    ac_diff_opp_max = float(ac_diff_opp_vals.max())

    area_matched_retention_max = (
        float(min(1.0, sb_dom_max / max(ac_dom_max, 1e-9))) if ac_dom_max > 1e-9 else 0.0
    )
    area_matched_retention_p995 = (
        float(min(1.0, sb_dom_p995 / max(ac_dom_p995, 1e-9))) if ac_dom_p995 > 1e-9 else 0.0
    )
    legacy_retention_max = float(sb_dom_max / max(raw_dom_max, 1e-9)) if raw_dom_max > 1e-9 else 0.0
    legacy_retention_p995 = (
        float(sb_dom_p995 / max(raw_dom_p995, 1e-9)) if raw_dom_p995 > 1e-9 else 0.0
    )

    cm_retention_max = (
        float(min(1.0, sb_cm_dom_max / max(ac_cm_dom_max, 1e-9))) if ac_cm_dom_max > 1e-9 else 0.0
    )
    diff_retention_max = (
        float(min(1.0, sb_diff_dom_max / max(ac_diff_dom_max, 1e-9)))
        if ac_diff_dom_max > 1e-9
        else 0.0
    )

    diff_tfce_asymmetry_ratio = tfce_cal._safe_ratio(
        sb_diff_dom_p995, sb_diff_opp_p995, float(ratio_floor)
    )
    cloud_masked_p995_tfce_ratio = tfce_cal._safe_ratio(
        sb_cm_dom_p995, sb_cm_opp_p995, float(ratio_floor)
    )
    cloud_masked_max_tfce_ratio = tfce_cal._safe_ratio(
        sb_cm_dom_max, sb_cm_opp_max, float(ratio_floor)
    )

    global_decision = _compute_upgraded_q34_decision(
        sb_dom_max=sb_cm_dom_max,
        sb_dom_p995=sb_cm_dom_p995,
        area_matched_retention_max=area_matched_retention_max,
        area_matched_retention_p995=area_matched_retention_p995,
        cloud_masked_max_tfce_ratio=cloud_masked_max_tfce_ratio,
        cloud_masked_p995_tfce_ratio=cloud_masked_p995_tfce_ratio,
        diff_tfce_p99=sb_diff_dom_p99,
        diff_tfce_p995=sb_diff_dom_p995,
        diff_tfce_asymmetry_ratio=diff_tfce_asymmetry_ratio,
        bilateral_cloud_frac=bilat_cloud_frac,
        topo=topo,
    )

    # Sliding-window (tile_size_px, tile_stride_px) Pipeline-B Cloud-Masked MAD + Area-Matched Q34
    sw_tiles = _build_sliding_window_tiles(
        eroded_valid, tile_size_px=tile_size_px, tile_stride_px=tile_stride_px
    )
    mad_res = q31_mod.score_pipeline_b_cloud_masked_mad(
        dom_clean,
        opp_clean,
        eroded_valid,
        sw_tiles,
        bilateral_thresh=bilateral_cloud_threshold,
    )
    mad_by_id = {r["tile_id"]: r for r in mad_res["ranked_tiles"]}

    tile_rows: list[dict[str, Any]] = []
    for t in sw_tiles:
        tid = t["tile_id"]
        y0, y1, x0, x1 = t["y0"], t["y1"], t["x0"], t["x1"]
        tv = eroded_valid[y0:y1, x0:x1]
        if not tv.any():
            continue
        t_sb_dom = sb_dom_clean[y0:y1, x0:x1][tv]
        t_ac_dom = ac_dom_clean[y0:y1, x0:x1][tv]
        t_sb_cm_dom = sb_cm_dom[y0:y1, x0:x1][tv]
        t_sb_cm_opp = sb_cm_opp[y0:y1, x0:x1][tv]
        t_sb_diff_dom = sb_diff_dom[y0:y1, x0:x1][tv]
        t_sb_diff_opp = sb_diff_opp[y0:y1, x0:x1][tv]

        t_sb_max = float(t_sb_dom.max())
        t_sb_p995 = float(np.percentile(t_sb_dom, 99.5))
        t_ac_max = float(t_ac_dom.max())
        t_ac_p995 = float(np.percentile(t_ac_dom, 99.5))

        t_cm_dom_max = float(t_sb_cm_dom.max())
        t_cm_dom_p995 = float(np.percentile(t_sb_cm_dom, 99.5))
        t_cm_opp_max = float(t_sb_cm_opp.max())
        t_cm_opp_p995 = float(np.percentile(t_sb_cm_opp, 99.5))

        t_diff_p99 = float(np.percentile(t_sb_diff_dom, 99.0))
        t_diff_p995 = float(np.percentile(t_sb_diff_dom, 99.5))
        t_diff_opp_p995 = float(np.percentile(t_sb_diff_opp, 99.5))

        t_ret_max = float(min(1.0, t_sb_max / max(t_ac_max, 1e-9))) if t_ac_max > 1e-9 else 0.0
        t_ret_p995 = (
            float(min(1.0, t_sb_p995 / max(t_ac_p995, 1e-9))) if t_ac_p995 > 1e-9 else 0.0
        )
        t_cm_p995_ratio = tfce_cal._safe_ratio(t_cm_dom_p995, t_cm_opp_p995, float(ratio_floor))
        t_cm_max_ratio = tfce_cal._safe_ratio(t_cm_dom_max, t_cm_opp_max, float(ratio_floor))
        t_diff_asym = tfce_cal._safe_ratio(t_diff_p995, t_diff_opp_p995, float(ratio_floor))

        m_row = mad_by_id.get(tid, {})
        mad_excess = float(m_row.get("mean_excess_diff", 0.0))
        mad_hot = float(m_row.get("hot_share", 0.0))
        t_cloud_frac = float(m_row.get("bilat_masked_frac", 0.0))

        tile_rows.append(
            {
                **t,
                "pipeline_b_mean_excess_diff": round(mad_excess, 6),
                "pipeline_b_hot_share": round(mad_hot, 6),
                "bilat_masked_frac": round(t_cloud_frac, 6),
                "sb_cm_dom_max": round(t_cm_dom_max, 6),
                "sb_cm_dom_p995": round(t_cm_dom_p995, 6),
                "area_matched_retention_max": round(t_ret_max, 6),
                "area_matched_retention_p995": round(t_ret_p995, 6),
                "cloud_masked_p995_tfce_ratio": round(t_cm_p995_ratio, 6),
                "cloud_masked_max_tfce_ratio": round(t_cm_max_ratio, 6),
                "diff_tfce_p99": round(t_diff_p99, 6),
                "diff_tfce_p995": round(t_diff_p995, 6),
                "diff_tfce_asymmetry_ratio": round(t_diff_asym, 6),
            }
        )

    tile_rows.sort(
        key=lambda r: (
            r["pipeline_b_mean_excess_diff"],
            r["sb_cm_dom_p995"],
            r["diff_tfce_p995"],
            r["tile_id"],
        ),
        reverse=True,
    )

    # Evaluate full transverse stroke morphology on Top-K sliding-window tiles
    eval_k = min(3, len(tile_rows))
    top_tile_q34_score = 0.0
    top_tile_passes_q34_bar = False
    for idx in range(eval_k):
        tr = tile_rows[idx]
        y0, y1, x0, x1 = tr["y0"], tr["y1"], tr["x0"], tr["x1"]
        t_topo = topo_mod.evaluate_transverse_fp_features(
            pa_rim_clean[y0:y1, x0:x1],
            pb_rim_clean[y0:y1, x0:x1],
            eroded_valid[y0:y1, x0:x1],
            voxel_um=voxel_um,
        )
        t_dec = _compute_upgraded_q34_decision(
            sb_dom_max=tr["sb_cm_dom_max"],
            sb_dom_p995=tr["sb_cm_dom_p995"],
            area_matched_retention_max=tr["area_matched_retention_max"],
            area_matched_retention_p995=tr["area_matched_retention_p995"],
            cloud_masked_max_tfce_ratio=tr["cloud_masked_max_tfce_ratio"],
            cloud_masked_p995_tfce_ratio=tr["cloud_masked_p995_tfce_ratio"],
            diff_tfce_p99=tr["diff_tfce_p99"],
            diff_tfce_p995=tr["diff_tfce_p995"],
            diff_tfce_asymmetry_ratio=tr["diff_tfce_asymmetry_ratio"],
            bilateral_cloud_frac=tr["bilat_masked_frac"],
            topo=t_topo,
        )
        tr["tile_q34_score"] = t_dec["upgraded_q34_score"]
        tr["tile_passes_q34_bar"] = t_dec["passes_q34_bar"]
        tr["tile_regime_label"] = t_dec["regime_label"]
        if t_dec["upgraded_q34_score"] > top_tile_q34_score:
            top_tile_q34_score = float(t_dec["upgraded_q34_score"])
        if t_dec["passes_q34_bar"]:
            top_tile_passes_q34_bar = True

    global_passes = bool(global_decision["passes_q34_bar"])
    upgraded_q34_score = float(global_decision["upgraded_q34_score"])
    if global_passes and top_tile_q34_score > upgraded_q34_score:
        upgraded_q34_score = float(0.5 * upgraded_q34_score + 0.5 * top_tile_q34_score)

    return {
        "voxel_um": float(voxel_um),
        "valid_in_px": n_valid_in,
        "eroded_valid_px": n_eroded,
        "dominant_face": dom,
        "opposite_face": opp,
        "area_matched_retention_max": round(area_matched_retention_max, 6),
        "area_matched_retention_p995": round(area_matched_retention_p995, 6),
        "legacy_retention_max": round(legacy_retention_max, 6),
        "legacy_retention_p995": round(legacy_retention_p995, 6),
        "diff_tfce_asymmetry_ratio": round(diff_tfce_asymmetry_ratio, 6),
        "cloud_masked_p995_tfce_ratio": round(cloud_masked_p995_tfce_ratio, 6),
        "cloud_masked_max_tfce_ratio": round(cloud_masked_max_tfce_ratio, 6),
        "diff_tfce_p99": round(sb_diff_dom_p99, 6),
        "diff_tfce_p995": round(sb_diff_dom_p995, 6),
        "top_tile_q34_score": round(top_tile_q34_score, 6),
        "top_tile_passes_q34_bar": bool(top_tile_passes_q34_bar),
        "global_upgraded_passes_q34_bar": global_passes,
        "upgraded_q34_score": round(upgraded_q34_score, 6),
        "upgraded_regime_label": str(global_decision["regime_label"]),
        "rim_erosion_diagnostic": {
            "rim_erosion_mm": float(rim_erosion_mm),
            "valid_in_px": n_valid_in,
            "eroded_valid_px": n_eroded,
            "eroded_rim_px": int(n_valid_in - n_eroded),
            "dom_rim_blob_suppressed_px": int(rim_supp_by_face[dom].sum()),
            "opp_rim_blob_suppressed_px": int(rim_supp_by_face[opp].sum()),
        },
        "area_matched_tfce": {
            "max_stroke_width_mm": float(max_stroke_width_mm),
            "max_component_area_mm2": float(max_component_area_mm2),
            "sb_dom_max": round(sb_dom_max, 6),
            "sb_dom_p995": round(sb_dom_p995, 6),
            "area_capped_raw_dom_max": round(ac_dom_max, 6),
            "area_capped_raw_dom_p995": round(ac_dom_p995, 6),
            "unbounded_raw_dom_max": round(raw_dom_max, 6),
            "unbounded_raw_dom_p995": round(raw_dom_p995, 6),
            "area_matched_retention_max": round(area_matched_retention_max, 6),
            "area_matched_retention_p995": round(area_matched_retention_p995, 6),
            "legacy_retention_max": round(legacy_retention_max, 6),
            "legacy_retention_p995": round(legacy_retention_p995, 6),
        },
        "cloud_masked_tfce": {
            "bilateral_cloud_threshold": float(bilateral_cloud_threshold),
            "bilateral_cloud_frac": round(bilat_cloud_frac, 6),
            "sb_cm_dom_max": round(sb_cm_dom_max, 6),
            "sb_cm_dom_p995": round(sb_cm_dom_p995, 6),
            "ac_cm_dom_max": round(ac_cm_dom_max, 6),
            "ac_cm_dom_p995": round(ac_cm_dom_p995, 6),
            "sb_cm_opp_max": round(sb_cm_opp_max, 6),
            "sb_cm_opp_p995": round(sb_cm_opp_p995, 6),
            "ac_cm_opp_max": round(float(ac_cm_opp_vals.max()), 6),
            "cm_retention_max": round(cm_retention_max, 6),
            "cloud_masked_max_tfce_ratio": round(cloud_masked_max_tfce_ratio, 6),
            "cloud_masked_p995_tfce_ratio": round(cloud_masked_p995_tfce_ratio, 6),
        },
        "differential_tfce": {
            "sb_diff_dom_max": round(sb_diff_dom_max, 6),
            "sb_diff_dom_p99": round(sb_diff_dom_p99, 6),
            "sb_diff_dom_p995": round(sb_diff_dom_p995, 6),
            "ac_diff_dom_max": round(ac_diff_dom_max, 6),
            "ac_diff_dom_p995": round(ac_diff_dom_p995, 6),
            "sb_diff_opp_max": round(sb_diff_opp_max, 6),
            "sb_diff_opp_p995": round(sb_diff_opp_p995, 6),
            "ac_diff_opp_max": round(ac_diff_opp_max, 6),
            "diff_retention_max": round(diff_retention_max, 6),
            "diff_tfce_asymmetry_ratio": round(diff_tfce_asymmetry_ratio, 6),
        },
        "pipeline_b_cloud_masked_mad": {
            "effective_mad_sigma": mad_res["effective_mad_sigma"],
            "masked_cloud_frac": mad_res["masked_cloud_frac"],
            "top_tile_mean_excess_diff": (
                tile_rows[0]["pipeline_b_mean_excess_diff"] if tile_rows else 0.0
            ),
            "top_tile_hot_share": tile_rows[0]["pipeline_b_hot_share"] if tile_rows else 0.0,
        },
        "sliding_window_tiles": {
            "tile_size_px": int(tile_size_px),
            "tile_stride_px": int(tile_stride_px),
            "n_tiles": len(tile_rows),
            "top_tile_q34_score": round(top_tile_q34_score, 6),
            "top_tile_passes_q34_bar": bool(top_tile_passes_q34_bar),
            "top_tiles": tile_rows[:5],
        },
        "upgraded_q34_gates": global_decision,
        "transverse_stroke_topology": topo,
    }


def make_synthetic_step_stair_seam_blob_control(
    shape: tuple[int, int] = (512, 1024),
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    seed: int = 125,
    diameters_mm: tuple[float, ...] = (1.6, 2.0, 2.3, 1.8, 2.4, 1.9),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize ``1.5-2.5 mm`` seam blobs hugging a step-stair valid-mask border."""
    rng = np.random.default_rng(int(seed))
    h, w = int(shape[0]), int(shape[1])
    mm_px = 1000.0 / float(voxel_um)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    valid = np.ones((h, w), dtype=bool)
    n_steps = max(4, w // 128)
    step_w = max(1, w // n_steps)
    stair_y = np.zeros(w, dtype=np.int32)
    for s in range(n_steps):
        x0 = s * step_w
        x1 = w if s == n_steps - 1 else (s + 1) * step_w
        depth = int(round((24 + 14 * (s % 4)) * (mm_px / 115.0)))
        depth = max(12, min(h // 4, depth))
        valid[h - depth :, x0:x1] = False
        stair_y[x0:x1] = h - depth

    blob_mask = np.zeros((h, w), dtype=np.float32)
    xs = np.linspace(int(0.10 * w), int(0.90 * w), len(diameters_mm))
    for cx_f, diam_mm in zip(xs, diameters_mm):
        cx = int(np.clip(round(cx_f), 0, w - 1))
        by = float(stair_y[cx]) - 0.18 * mm_px  # sits right on / inside the step-stair rim
        r_px = 0.5 * float(diam_mm) * mm_px
        sigma_px = max(2.0, 0.42 * r_px)
        dist2 = (yy - by) ** 2 + (xx - float(cx)) ** 2
        blob = 0.90 * np.exp(-dist2 / (2.0 * sigma_px * sigma_px)).astype(np.float32)
        np.maximum(blob_mask, blob, out=blob_mask)

    base_asc = (0.14 + rng.normal(0.0, 0.015, (h, w))).astype(np.float32)
    base_desc = (0.14 + rng.normal(0.0, 0.015, (h, w))).astype(np.float32)
    prob_asc = np.where(valid, np.clip(np.maximum(base_asc, blob_mask), 0.0, 1.0), 0.0).astype(
        np.float32
    )
    prob_desc = np.where(valid, np.clip(base_desc, 0.0, 1.0), 0.0).astype(np.float32)
    return prob_asc, prob_desc, valid


def inject_opposite_face_border_speck(
    prob_opp: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    speck_width_mm: float = 0.42,
    speck_length_mm: float = 1.60,
    peak: float = 0.92,
) -> np.ndarray:
    """Inject a single Exp T10-GPU-style ``128px``-window border speck on the valid-mask rim."""
    out = np.asarray(prob_opp, dtype=np.float32).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        return out
    h, w = valid.shape
    mm_px = 1000.0 / float(voxel_um)
    half_w_px = max(2, int(round(0.5 * float(speck_width_mm) * mm_px)))
    len_px = max(12, int(round(float(speck_length_mm) * mm_px)))

    rows = np.flatnonzero(np.any(valid, axis=1))
    r_top = int(rows[0])
    cols = np.flatnonzero(valid[min(h - 1, r_top + half_w_px), :])
    c_mid = int(cols[len(cols) // 2]) if cols.size > 0 else w // 2

    y0 = max(0, r_top + 2)
    y1 = min(h, y0 + 2 * half_w_px)
    x0 = max(0, c_mid - len_px // 2)
    x1 = min(w, x0 + len_px)

    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    if patch.size > 0:
        patch[:, :] = float(peak)
        patch = ndi.gaussian_filter(patch, sigma=1.2)
        if patch.max() > 0:
            patch = patch * (float(peak) / float(patch.max()))
        out[y0:y1, x0:x1] = np.where(
            valid[y0:y1, x0:x1], np.maximum(out[y0:y1, x0:x1], patch), 0.0
        )
    return out


def _side_by_side_comparison_row(
    regime_id: str,
    category: str,
    expected_pass: bool,
    prob_asc: np.ndarray,
    prob_desc: np.ndarray,
    valid_mask: np.ndarray,
    *,
    voxel_um: float,
    tile_size_px: int = DEFAULT_TILE_SIZE_PX,
    tile_stride_px: int = DEFAULT_TILE_STRIDE_PX,
    rim_erosion_mm: float = DEFAULT_RIM_EROSION_MM,
    bilateral_cloud_threshold: float = DEFAULT_BILATERAL_CLOUD_THRESHOLD,
    notes: str = "",
) -> dict[str, Any]:
    """Run legacy Exp T6 Q34 and upgraded Exp T13 Q34 side-by-side on a single regime."""
    t0 = time.perf_counter()
    legacy = tfce_cal.tfce_opposite_face_fwer(prob_asc, prob_desc, valid_mask, voxel_um=voxel_um)
    t_leg = time.perf_counter() - t0

    t1 = time.perf_counter()
    upgraded = evaluate_upgraded_q34_and_q31_ranker(
        prob_asc,
        prob_desc,
        valid_mask,
        voxel_um=voxel_um,
        tile_size_px=tile_size_px,
        tile_stride_px=tile_stride_px,
        rim_erosion_mm=rim_erosion_mm,
        bilateral_cloud_threshold=bilateral_cloud_threshold,
    )
    t_upg = time.perf_counter() - t1

    leg_sb = legacy["stroke_bounded_tfce"]
    leg_q34 = legacy["q34_calibrated"]
    leg_topo = legacy["transverse_stroke_topology"]

    return {
        "regime_id": regime_id,
        "category": category,
        "expected_passes_q34_bar": bool(expected_pass),
        "voxel_um": float(voxel_um),
        "shape_hw": [int(valid_mask.shape[0]), int(valid_mask.shape[1])],
        "valid_cm2": leg_topo["valid_cm2"],
        "notes": notes,
        "legacy_exp_t6_q34": {
            "raw_dom_max": legacy["raw_tfce"]["dom_max"],
            "sb_dom_max": leg_sb["dom_max"],
            "sb_opp_max": leg_sb["opp_max"],
            "max_tfce_ratio": legacy["max_tfce_ratio"],
            "p995_tfce_ratio": legacy["p995_tfce_ratio"],
            "diff_tfce_p99": legacy["diff_tfce_p99"],
            "sb_retention_max": leg_sb["retention_max"],
            "sb_retention_p995": leg_sb["retention_p995"],
            "width_median_hi_mm": leg_topo["stroke_width"]["hi"]["width_median_mm"],
            "narrow_stroke_frac_hi": leg_topo["stroke_width"]["hi"]["narrow_stroke_frac"],
            "wide_blob_frac_hi": leg_topo["stroke_width"]["hi"]["wide_blob_frac"],
            "max_b1": leg_topo["betti"]["dominant"]["max_b1"],
            "stroke_consistency_score": leg_topo["composite"]["stroke_consistency_score"],
            "q34_score": leg_q34["calibrated_score"],
            "passes_tfce_asymmetry": leg_q34["passes_tfce_asymmetry"],
            "passes_width_retention": leg_q34["passes_width_retention"],
            "passes_stroke_morphology": leg_q34["passes_stroke_morphology"],
            "passes_q34_bar": leg_q34["passes_q34_bar"],
            "regime_label": leg_q34["regime_label"],
            "elapsed_seconds": round(t_leg, 3),
        },
        "upgraded_exp_t13_q34": {
            "area_capped_raw_dom_max": upgraded["area_matched_tfce"]["area_capped_raw_dom_max"],
            "sb_dom_max": upgraded["area_matched_tfce"]["sb_dom_max"],
            "sb_cm_dom_max": upgraded["cloud_masked_tfce"]["sb_cm_dom_max"],
            "sb_cm_opp_max": upgraded["cloud_masked_tfce"]["sb_cm_opp_max"],
            "area_matched_retention_max": upgraded["area_matched_retention_max"],
            "area_matched_retention_p995": upgraded["area_matched_retention_p995"],
            "cloud_masked_max_tfce_ratio": upgraded["cloud_masked_max_tfce_ratio"],
            "cloud_masked_p995_tfce_ratio": upgraded["cloud_masked_p995_tfce_ratio"],
            "diff_tfce_p99": upgraded["diff_tfce_p99"],
            "diff_tfce_p995": upgraded["diff_tfce_p995"],
            "diff_tfce_asymmetry_ratio": upgraded["diff_tfce_asymmetry_ratio"],
            "top_tile_q34_score": upgraded["top_tile_q34_score"],
            "top_tile_passes_q34_bar": upgraded["top_tile_passes_q34_bar"],
            "global_upgraded_passes_q34_bar": upgraded["global_upgraded_passes_q34_bar"],
            "upgraded_q34_score": upgraded["upgraded_q34_score"],
            "upgraded_regime_label": upgraded["upgraded_regime_label"],
            "passes_tfce_asymmetry": upgraded["upgraded_q34_gates"]["passes_tfce_asymmetry"],
            "passes_width_retention": upgraded["upgraded_q34_gates"]["passes_width_retention"],
            "passes_stroke_morphology": upgraded["upgraded_q34_gates"]["passes_stroke_morphology"],
            "rim_erosion_diagnostic": upgraded["rim_erosion_diagnostic"],
            "pipeline_b_cloud_masked_mad": upgraded["pipeline_b_cloud_masked_mad"],
            "elapsed_seconds": round(t_upg, 3),
        },
    }


def _assert_public_privacy_invariant(serialized_json: str) -> None:
    """Enforce zero local filesystem paths, zero tokens, and zero unread scroll coordinates."""
    for pat in q22_mod.FORBIDDEN_PATH_PATTERNS:
        m = pat.search(serialized_json)
        if m is not None:
            raise RuntimeError(
                f"Privacy invariant violation: forbidden pattern {m.group(0)!r} found in evidence JSON"
            )
    for bad in _FORBIDDEN_EVIDENCE_SUBSTRINGS:
        if bad in serialized_json:
            raise RuntimeError(
                f"Privacy invariant violation: forbidden substring {bad!r} found in evidence JSON"
            )


def run_t13_tiled_q34_calibration(
    output_json: Path | str | None = None,
    *,
    byte_cap: int = GLOBAL_HTTP_BYTE_CAP,
) -> dict[str, Any]:
    """Run the live Exp T13-TileQ34-Ranker calibration benchmark across all 7 regimes (<= 40 MiB)."""
    t_start = time.perf_counter()
    budget = tfce_cal.HttpByteBudget(cap_bytes=int(byte_cap))

    # 1. Fetch PHerc. 0306B w074 published claim JSON + PNG (~8.10 MiB)
    json_bytes = budget.fetch(tfce_cal.PHERC0306B_JSON_URL)
    png_bytes = budget.fetch(tfce_cal.PHERC0306B_PNG_URL)
    json_sha256 = hashlib.sha256(json_bytes).hexdigest()
    png_sha256 = hashlib.sha256(png_bytes).hexdigest()
    meta_0306b = json.loads(json_bytes.decode("utf-8"))
    expected_png_sha = meta_0306b.get("image_sha256")
    if expected_png_sha and png_sha256 != expected_png_sha:
        raise RuntimeError(
            f"PHerc. 0306B PNG SHA-256 mismatch: got {png_sha256}, expected {expected_png_sha}"
        )

    with Image.open(io.BytesIO(png_bytes)) as im:
        rgb_0306b = np.asarray(im.convert("RGB"), dtype=np.float32)

    d_map_0306b, valid_0306b, inv_diag = tfce_cal.invert_display_tone_map(rgb_0306b, meta_0306b)
    voxel_0306b = float(meta_0306b["voxel_size_um"])
    cb = meta_0306b["claim_box_image_px"]
    bx0, bx1 = int(cb["x0"]) + 10, int(cb["x1"]) - 10
    row1_cy = int(meta_0306b["rows"]["centre_y_image_px"][0])
    half_h = 240
    r1_y0, r1_y1 = row1_cy - half_h, row1_cy + half_h
    null_y0, null_y1 = 3350, 3350 + 2 * half_h
    blank_y0, blank_y1 = 980, 980 + 2 * half_h

    r1_dom = d_map_0306b[r1_y0:r1_y1, bx0:bx1]
    r1_opp = d_map_0306b[null_y0:null_y1, bx0:bx1]
    r1_val = valid_0306b[r1_y0:r1_y1, bx0:bx1] & valid_0306b[null_y0:null_y1, bx0:bx1]

    bm_0306b_dom = d_map_0306b[blank_y0:blank_y1, bx0:bx1]
    bm_0306b_val = valid_0306b[blank_y0:blank_y1, bx0:bx1] & valid_0306b[null_y0:null_y1, bx0:bx1]

    sub_x0, sub_x1 = 2600, 3757  # 10.0 mm short-word sub-crop on PHerc. 0306B Line 1
    sub_0306b_dom = d_map_0306b[r1_y0:r1_y1, sub_x0:sub_x1]
    sub_0306b_opp = d_map_0306b[null_y0:null_y1, sub_x0:sub_x1]
    sub_0306b_val = (
        valid_0306b[r1_y0:r1_y1, sub_x0:sub_x1] & valid_0306b[null_y0:null_y1, sub_x0:sub_x1]
    )

    # Also build the Exp T10-GPU Arm 1 border-speck diagnostic variant on PHerc. 0306B Line 1
    r1_opp_with_rim_speck = inject_opposite_face_border_speck(
        r1_opp, r1_val, voxel_um=voxel_0306b
    )

    sources_manifest: list[dict[str, Any]] = [
        {
            "source_id": "pherc0306b_w074_json",
            "scroll": "PHerc0306B",
            "url": tfce_cal.PHERC0306B_JSON_URL,
            "bytes": len(json_bytes),
            "mib": round(len(json_bytes) / (1024.0 * 1024.0), 4),
            "sha256": json_sha256,
        },
        {
            "source_id": "pherc0306b_w074_png",
            "scroll": "PHerc0306B",
            "url": tfce_cal.PHERC0306B_PNG_URL,
            "bytes": len(png_bytes),
            "mib": round(len(png_bytes) / (1024.0 * 1024.0), 4),
            "sha256": png_sha256,
        },
    ]

    regimes: list[dict[str, Any]] = []

    # Regime 1: PHerc. 0306B w074 full single-line claim map (+ border-speck rescue check)
    reg1_clean = _side_by_side_comparison_row(
        "1_pherc0306b_w074_full_line1",
        "positive_single_line",
        True,
        r1_dom,
        r1_opp,
        r1_val,
        voxel_um=voxel_0306b,
        notes="PHerc. 0306B w074 full Line 1 claim crop vs on-sheet blank margin null.",
    )
    reg1_speck = _side_by_side_comparison_row(
        "1b_pherc0306b_w074_line1_with_opp_rim_speck",
        "positive_single_line_rim_speck_rescue",
        True,
        r1_dom,
        r1_opp_with_rim_speck,
        r1_val,
        voxel_um=voxel_0306b,
        notes=(
            "Bottleneck B rescue: opposite-face border speck on valid-mask rim depresses "
            "legacy max_tfce_ratio < 3.0; rim_eroded_valid_mask + rim-blob suppression rescues pass."
        ),
    )
    reg1_clean["bottleneck_b_rim_speck_diagnostic"] = {
        "legacy_passes_q34_bar": reg1_speck["legacy_exp_t6_q34"]["passes_q34_bar"],
        "legacy_max_tfce_ratio": reg1_speck["legacy_exp_t6_q34"]["max_tfce_ratio"],
        "upgraded_passes_q34_bar": reg1_speck["upgraded_exp_t13_q34"][
            "global_upgraded_passes_q34_bar"
        ],
        "upgraded_cloud_masked_p995_tfce_ratio": reg1_speck["upgraded_exp_t13_q34"][
            "cloud_masked_p995_tfce_ratio"
        ],
        "upgraded_diff_tfce_asymmetry_ratio": reg1_speck["upgraded_exp_t13_q34"][
            "diff_tfce_asymmetry_ratio"
        ],
        "opp_rim_blob_suppressed_px": reg1_speck["upgraded_exp_t13_q34"][
            "rim_erosion_diagnostic"
        ]["opp_rim_blob_suppressed_px"],
    }
    regimes.append(reg1_clean)

    # Regime 2: PHerc. 0306B w074 10 mm short-word sub-crop
    reg2 = _side_by_side_comparison_row(
        "2_pherc0306b_w074_10mm_subcrop",
        "positive_short_word_subcrop",
        True,
        sub_0306b_dom,
        sub_0306b_opp,
        sub_0306b_val,
        voxel_um=voxel_0306b,
        notes="PHerc. 0306B w074 10.0 mm short-word Line 1 sub-crop.",
    )
    regimes.append(reg2)

    # Fetch the 3 public S3 ink maps (PHerc0343 argus-v3, PHerc0343 reader-v3, PHerc0841 argus-v3)
    s3_blank_controls: list[dict[str, Any]] = []
    s3_regime_ids = {
        "pherc0343_argus_v3": "3_pherc0343_w047_w048_argus_v3_text_strip_512x2048",
        "pherc0343_reader_v3": "4_pherc0343_w047_w048_reader_v3_text_strip_512x2048",
        "pherc0841_argus_v3": "5_pherc0841_w00_argus_v3_text_strip_512x2048",
    }

    for spec in q31_mod.PUBLIC_INK_MAPS:
        tiff_bytes = budget.fetch(spec.url, timeout=120.0)
        tiff_sha256 = hashlib.sha256(tiff_bytes).hexdigest()
        prob_dom, raw_arr, dec_meta = q31_mod.decode_ink_tiff_bytes(tiff_bytes)
        valid_s3, valid_diag = q31_mod.infer_valid_mask(raw_arr)
        prob_opp, _ = q31_mod.build_margin_opposite_face_control(prob_dom, valid_s3)

        sources_manifest.append(
            {
                "source_id": spec.map_id,
                "scroll": spec.scroll,
                "model": spec.model,
                "voxel_um": spec.voxel_um,
                "url": spec.url,
                "bytes": len(tiff_bytes),
                "mib": round(len(tiff_bytes) / (1024.0 * 1024.0), 4),
                "sha256": tiff_sha256,
                "shape_hw": dec_meta["shape_hw"],
                "valid_frac": valid_diag["valid_frac"],
            }
        )

        text_win = q31_mod.select_text_rich_strip(prob_dom, valid_s3)
        ty0, ty1, tx0, tx1 = text_win["y0"], text_win["y1"], text_win["x0"], text_win["x1"]
        reg_s3 = _side_by_side_comparison_row(
            s3_regime_ids[spec.map_id],
            "positive_dense_multiline_text_strip",
            True,
            prob_dom[ty0:ty1, tx0:tx1],
            prob_opp[ty0:ty1, tx0:tx1],
            valid_s3[ty0:ty1, tx0:tx1],
            voxel_um=spec.voxel_um,
            notes=(
                f"Bottleneck A rescue on {spec.scroll} {spec.model} 512x2048 dense multi-line text strip."
            ),
        )
        regimes.append(reg_s3)

        blank_win = q31_mod.select_blank_margin_strip(prob_dom, valid_s3)
        by0, by1, bx0_s3, bx1_s3 = (
            blank_win["y0"],
            blank_win["y1"],
            blank_win["x0"],
            blank_win["x1"],
        )
        s3_blank_controls.append(
            _side_by_side_comparison_row(
                f"6_{spec.map_id}_blank_margin",
                "negative_control_a_real_blank_margin",
                False,
                prob_dom[by0:by1, bx0_s3:bx1_s3],
                prob_opp[by0:by1, bx0_s3:bx1_s3],
                valid_s3[by0:by1, bx0_s3:bx1_s3],
                voxel_um=spec.voxel_um,
                notes=f"Real blank papyrus margin control on {spec.scroll} {spec.model}.",
            )
        )

    # Regime 6: Negative Control A — Real PHerc. 0306B / PHerc. 0343 / PHerc. 0841 blank papyrus margins
    reg6_0306b_blank = _side_by_side_comparison_row(
        "6_real_blank_papyrus_margins_pherc0306b_0343_0841",
        "negative_control_a_real_blank_margin",
        False,
        bm_0306b_dom,
        r1_opp,
        bm_0306b_val,
        voxel_um=voxel_0306b,
        notes="Negative Control A: Real blank papyrus margins across PHerc. 0306B, PHerc. 0343, and PHerc. 0841.",
    )
    reg6_0306b_blank["all_blank_margin_subcontrols"] = [
        {
            "subcontrol_id": "pherc0306b_w074_blank_margin",
            "legacy_passes_q34_bar": reg6_0306b_blank["legacy_exp_t6_q34"]["passes_q34_bar"],
            "upgraded_passes_q34_bar": reg6_0306b_blank["upgraded_exp_t13_q34"][
                "global_upgraded_passes_q34_bar"
            ],
            "upgraded_q34_score": reg6_0306b_blank["upgraded_exp_t13_q34"]["upgraded_q34_score"],
        },
        *[
            {
                "subcontrol_id": sc["regime_id"],
                "legacy_passes_q34_bar": sc["legacy_exp_t6_q34"]["passes_q34_bar"],
                "upgraded_passes_q34_bar": sc["upgraded_exp_t13_q34"][
                    "global_upgraded_passes_q34_bar"
                ],
                "upgraded_q34_score": sc["upgraded_exp_t13_q34"]["upgraded_q34_score"],
            }
            for sc in s3_blank_controls
        ],
    ]
    regimes.append(reg6_0306b_blank)

    # Regime 7: Negative Control B — Synthetic controls + mesh-rim seam-blob control (1.5-2.5 mm
    # blobs along a step-stair valid-mask border + bilateral 2-4 mm cloud control + dot-row control)
    seam_asc, seam_desc, seam_val = make_synthetic_step_stair_seam_blob_control(
        (512, 1024), voxel_um=DEFAULT_VOXEL_UM
    )
    cloud_asc, cloud_desc, cloud_val = tfce_cal.make_synthetic_pherc1203_bilateral_clouds(
        (480, 1024), voxel_um=DEFAULT_VOXEL_UM, diameters_mm=(2.2, 2.8)
    )
    dot_asc, dot_desc, dot_val = tfce_cal.make_synthetic_pherc0125_dot_row(
        (480, 1024), voxel_um=DEFAULT_VOXEL_UM, diameters_mm=(0.65, 0.85, 1.00, 1.15)
    )

    reg7_seam = _side_by_side_comparison_row(
        "7_synthetic_mesh_rim_seam_blob_and_bilateral_cloud_controls",
        "negative_control_b_synthetic_seam_and_cloud_controls",
        False,
        seam_asc,
        seam_desc,
        seam_val,
        voxel_um=DEFAULT_VOXEL_UM,
        notes=(
            "Negative Control B: 1.5-2.5 mm blobs along a step-stair valid-mask border, "
            "2-4 mm bilateral mottle clouds, and single-face 0.6-1.2 mm isolated dot row."
        ),
    )
    reg7_cloud = _side_by_side_comparison_row(
        "7b_synthetic_bilateral_2_to_4mm_clouds",
        "negative_control_b_bilateral_cloud",
        False,
        cloud_asc,
        cloud_desc,
        cloud_val,
        voxel_um=DEFAULT_VOXEL_UM,
        notes="Synthetic 2-4 mm bilateral mottle cloud false-positive control.",
    )
    reg7_dots = _side_by_side_comparison_row(
        "7c_synthetic_single_face_dot_row",
        "negative_control_b_dot_row",
        False,
        dot_asc,
        dot_desc,
        dot_val,
        voxel_um=DEFAULT_VOXEL_UM,
        notes="Synthetic 0.6-1.2 mm single-face isolated dot-row false-positive control.",
    )
    reg7_seam["synthetic_subcontrols"] = {
        "step_stair_mesh_rim_seam_blobs": {
            "legacy_passes_q34_bar": reg7_seam["legacy_exp_t6_q34"]["passes_q34_bar"],
            "upgraded_passes_q34_bar": reg7_seam["upgraded_exp_t13_q34"][
                "global_upgraded_passes_q34_bar"
            ],
            "upgraded_q34_score": reg7_seam["upgraded_exp_t13_q34"]["upgraded_q34_score"],
            "upgraded_regime_label": reg7_seam["upgraded_exp_t13_q34"]["upgraded_regime_label"],
            "dom_rim_blob_suppressed_px": reg7_seam["upgraded_exp_t13_q34"][
                "rim_erosion_diagnostic"
            ]["dom_rim_blob_suppressed_px"],
        },
        "bilateral_2_to_4mm_clouds": {
            "legacy_passes_q34_bar": reg7_cloud["legacy_exp_t6_q34"]["passes_q34_bar"],
            "upgraded_passes_q34_bar": reg7_cloud["upgraded_exp_t13_q34"][
                "global_upgraded_passes_q34_bar"
            ],
            "upgraded_q34_score": reg7_cloud["upgraded_exp_t13_q34"]["upgraded_q34_score"],
            "area_matched_retention_max": reg7_cloud["upgraded_exp_t13_q34"][
                "area_matched_retention_max"
            ],
            "upgraded_regime_label": reg7_cloud["upgraded_exp_t13_q34"]["upgraded_regime_label"],
        },
        "single_face_dot_row": {
            "legacy_passes_q34_bar": reg7_dots["legacy_exp_t6_q34"]["passes_q34_bar"],
            "upgraded_passes_q34_bar": reg7_dots["upgraded_exp_t13_q34"][
                "global_upgraded_passes_q34_bar"
            ],
            "upgraded_q34_score": reg7_dots["upgraded_exp_t13_q34"]["upgraded_q34_score"],
            "upgraded_regime_label": reg7_dots["upgraded_exp_t13_q34"]["upgraded_regime_label"],
        },
    }
    regimes.append(reg7_seam)

    all_positive_pass = all(
        r["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
        for r in regimes
        if r["expected_passes_q34_bar"]
    )
    all_negative_rejected = all(
        not r["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
        for r in regimes
        if not r["expected_passes_q34_bar"]
    )

    payload: dict[str, Any] = {
        "schema_version": "tiled_q34_cloud_masked_calibration_v1",
        "experiment": (
            "Exp T13-TileQ34-Ranker (Q31 + Q34): Area-Matched Stroke-Bounded TFCE & "
            "Rim-Eroded Cloud-Masked Opposite-Face Ranker"
        ),
        "device": "cpu_only_numpy_scipy_pil",
        "http_budget": {
            "cap_bytes": int(budget.cap_bytes),
            "cap_mib": round(budget.cap_bytes / (1024.0 * 1024.0), 4),
            "used_bytes": int(budget.used_bytes),
            "used_mib": round(budget.used_bytes / (1024.0 * 1024.0), 4),
            "within_cap": bool(budget.used_bytes <= budget.cap_bytes),
        },
        "parameters": {
            "default_voxel_um": DEFAULT_VOXEL_UM,
            "tile_size_px": DEFAULT_TILE_SIZE_PX,
            "tile_stride_px": DEFAULT_TILE_STRIDE_PX,
            "rim_erosion_mm": DEFAULT_RIM_EROSION_MM,
            "bilateral_cloud_threshold": DEFAULT_BILATERAL_CLOUD_THRESHOLD,
            "max_stroke_width_mm": DEFAULT_MAX_STROKE_WIDTH_MM,
            "max_component_area_mm2": DEFAULT_MAX_COMPONENT_AREA_MM2,
            "tfce_H": DEFAULT_TFCE_H,
            "tfce_E": DEFAULT_TFCE_E,
            "tfce_n_steps": DEFAULT_TFCE_STEPS,
        },
        "sources": sources_manifest,
        "pherc0306b_display_inversion": inv_diag,
        "summary_verdict": {
            "n_regimes": len(regimes),
            "n_positive_regimes": sum(1 for r in regimes if r["expected_passes_q34_bar"]),
            "n_negative_regimes": sum(1 for r in regimes if not r["expected_passes_q34_bar"]),
            "legacy_positive_pass_count": sum(
                1
                for r in regimes
                if r["expected_passes_q34_bar"] and r["legacy_exp_t6_q34"]["passes_q34_bar"]
            ),
            "upgraded_positive_pass_count": sum(
                1
                for r in regimes
                if r["expected_passes_q34_bar"]
                and r["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
            ),
            "upgraded_negative_pass_count": sum(
                1
                for r in regimes
                if (not r["expected_passes_q34_bar"])
                and r["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
            ),
            "all_positive_regimes_pass_upgraded": all_positive_pass,
            "all_negative_controls_rejected_upgraded": all_negative_rejected,
            "bottleneck_a_multiline_rescue_verified": bool(
                regimes[2]["upgraded_exp_t13_q34"]["area_matched_retention_max"] >= 0.65
                and regimes[3]["upgraded_exp_t13_q34"]["area_matched_retention_max"] >= 0.70
                and regimes[4]["upgraded_exp_t13_q34"]["area_matched_retention_max"] >= 0.65
            ),
            "bottleneck_b_rim_speck_rescue_verified": bool(
                not reg1_speck["legacy_exp_t6_q34"]["passes_q34_bar"]
                and reg1_speck["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"]
            ),
        },
        "regime_benchmarks": regimes,
        "total_elapsed_seconds": round(time.perf_counter() - t_start, 3),
    }

    serialized = json.dumps(payload, indent=2)
    _assert_public_privacy_invariant(serialized)

    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(serialized + "\n", encoding="utf-8")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Exp T13-TileQ34-Ranker (Q31 + Q34) CPU-only calibration benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_THIS_DIR / "evidence" / "tiled_q34_cloud_masked_calibration.json",
        help="Destination path for JSON evidence output.",
    )
    args = parser.parse_args(argv)
    payload = run_t13_tiled_q34_calibration(output_json=args.output)
    hb = payload["http_budget"]
    print(
        f"Wrote {args.output.name} | HTTP {hb['used_mib']:.3f} / {hb['cap_mib']:.1f} MiB | "
        f"elapsed {payload['total_elapsed_seconds']:.1f}s"
    )
    for r in payload["regime_benchmarks"]:
        leg = r["legacy_exp_t6_q34"]
        upg = r["upgraded_exp_t13_q34"]
        print(
            f"  {r['regime_id']:<55} | leg_ret={leg['sb_retention_max']:.3f} "
            f"leg_pass={str(leg['passes_q34_bar']):<5} | "
            f"am_ret={upg['area_matched_retention_max']:.3f} "
            f"cm_p995R={upg['cloud_masked_p995_tfce_ratio']:>6.2f} "
            f"diffAsym={upg['diff_tfce_asymmetry_ratio']:>6.2f} "
            f"upg_q34={upg['upgraded_q34_score']:.4f} "
            f"upg_pass={str(upg['global_upgraded_passes_q34_bar']):<5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
