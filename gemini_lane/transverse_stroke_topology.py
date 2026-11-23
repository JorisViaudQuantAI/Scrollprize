# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only transverse stroke topology and opposite-face empirical-null diagnostics.

Addresses two recurring First Letters screening false-positive regimes observed in
``collab/claude-013``, ``collab/gemini-006``, and ``collab/claude-017``:

* **Single-face dot rows** (``~0.11/cm^2`` on ``PHerc0125``) that exhibit genuine
  global depth-order asymmetry yet lack continuous narrow letter strokes and enclosed
  topological loops (``b1 == 0``);
* **Bilateral 2-4 mm adjacent-wrap mottle clouds** that fire on both depth orders
  (``asc`` and ``desc``) with wide diffuse medial radii and featureless topology.

Implements three complementary, strictly CPU-only (NumPy + SciPy) families of
non-gating diagnostics inspired by TENT/RX empirical-null calibration, Stroke Width
Transform medial-axis geometry, and 2D Betti persistent filtration (8-connected
foreground, 4-connected background):

1. :func:`opposite_face_differential_map` — clipped signed log-odds differential
   ``logit(prob_a) - logit(prob_b)`` calibrated against the opposite-face empirical
   null distribution.
2. :func:`stroke_width_consistency` — Euclidean distance transform sampled along a
   LUT-accelerated Zhang-Suen medial skeleton in physical millimetres.
3. :func:`betti_filtration_signature` — multi-threshold 2D Betti numbers ``(b0, b1)``
   and loop persistence summary across probability filtrations.
4. :func:`evaluate_transverse_fp_features` — unified JSON-serializable summary
   combining both depth-order directions and a non-gating discriminability score.

Pure NumPy + SciPy; never imports torch or CUDA.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

DEFAULT_VOXEL_UM = 9.362
FACES = ("asc", "desc")
DEFAULT_BETTI_THRESHOLDS = (0.45, 0.55, 0.65, 0.75, 0.85)
DEFAULT_NARROW_STROKE_BAND_MM = (0.15, 0.80)
DEFAULT_WIDE_BLOB_MM = 1.20
DEFAULT_DIFF_SIGMA_FLOOR = 0.10

_CONN8 = np.ones((3, 3), dtype=np.int8)
_CONN4 = ndi.generate_binary_structure(2, 1)


def _build_zhang_suen_luts() -> tuple[np.ndarray, np.ndarray]:
    """Precompute 256-entry lookup tables for the two Zhang-Suen thinning sub-iterations."""
    lut0 = np.zeros(256, dtype=bool)
    lut1 = np.zeros(256, dtype=bool)
    for code in range(256):
        # Clockwise 8-neighbor ring starting North: p2, p3, p4, p5, p6, p7, p8, p9
        bits = [(code >> k) & 1 for k in range(8)]
        p2, _, p4, _, p6, _, p8, _ = bits
        b = sum(bits)
        a = sum(1 for i in range(8) if bits[i] == 0 and bits[(i + 1) & 7] == 1)
        if 2 <= b <= 6 and a == 1:
            if (p2 & p4 & p6) == 0 and (p4 & p6 & p8) == 0:
                lut0[code] = True
            if (p2 & p4 & p8) == 0 and (p2 & p6 & p8) == 0:
                lut1[code] = True
    return lut0, lut1


_ZS_LUT0, _ZS_LUT1 = _build_zhang_suen_luts()


def _zhang_suen_thin_crop(mask: np.ndarray) -> np.ndarray:
    """LUT-accelerated vectorized Zhang-Suen thinning of a cropped boolean mask."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for lut in (_ZS_LUT0, _ZS_LUT1):
            c = img[1:-1, 1:-1]
            code = (
                img[:-2, 1:-1]
                | (img[:-2, 2:] << 1)
                | (img[1:-1, 2:] << 2)
                | (img[2:, 2:] << 3)
                | (img[2:, 1:-1] << 4)
                | (img[2:, :-2] << 5)
                | (img[1:-1, :-2] << 6)
                | (img[:-2, :-2] << 7)
            )
            rm = (c == 1) & lut[code]
            if rm.any():
                c[rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _medial_skeleton(mask: np.ndarray, edt: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, list[tuple[int, int]]]:
    """Thin each 8-connected component on its tight bounding box.

    Guarantees that every non-empty component yields at least one medial pixel
    (falling back to its maximum-EDT pixel if a degenerate 2x2 block vanishes under
    standard Zhang-Suen thinning). Returns ``(skeleton, labels, n_components,
    component_area_skel_pairs)``.
    """
    labels, n = ndi.label(mask, structure=_CONN8)
    skeleton = np.zeros_like(mask, dtype=bool)
    comp_stats: list[tuple[int, int]] = []
    if n == 0:
        return skeleton, labels, 0, comp_stats
    slices = ndi.find_objects(labels)
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp = labels[sl] == idx
        area = int(comp.sum())
        if area == 0:
            continue
        if area <= 2:
            skel_crop = comp
        else:
            skel_crop = _zhang_suen_thin_crop(comp)
            if not skel_crop.any():
                edt_crop = np.where(comp, edt[sl], -1.0)
                peak = np.unravel_index(int(np.argmax(edt_crop)), comp.shape)
                skel_crop = np.zeros_like(comp, dtype=bool)
                skel_crop[peak] = True
        skeleton[sl] |= skel_crop
        comp_stats.append((area, int(skel_crop.sum())))
    return skeleton, labels, int(n), comp_stats


def _sanitize_prob_pair(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate 2D shapes and intersect ``valid_mask`` with finite probability pixels."""
    pa = np.asarray(prob_a, dtype=np.float64)
    pb = np.asarray(prob_b, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if pa.ndim != 2 or pb.ndim != 2 or valid.ndim != 2:
        raise ValueError("prob_a, prob_b, and valid_mask must be 2D arrays")
    if pa.shape != valid.shape or pb.shape != valid.shape:
        raise ValueError("prob_a, prob_b, and valid_mask shapes differ")
    finite_valid = valid & np.isfinite(pa) & np.isfinite(pb)
    return pa, pb, finite_valid


def opposite_face_differential_map(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-4,
) -> dict:
    """Compute opposite-face log-odds differential and empirical-null z-score map.

    Evaluates ``delta = logit(clip(prob_a, eps, 1 - eps)) - logit(clip(prob_b, eps, 1 - eps))``
    on finite valid pixels, then calibrates an empirical-null z-score against the
    opposite-face (``prob_b``) distribution via median/MAD and negative-tail spread.
    Handles empty, tiny, constant, and NaN/Inf-contaminated inputs safely.
    """
    if not (math.isfinite(eps) and 0.0 < eps < 0.5):
        raise ValueError(f"eps must lie in (0, 0.5), got {eps!r}")
    pa, pb, valid = _sanitize_prob_pair(prob_a, prob_b, valid_mask)
    delta_map = np.zeros(valid.shape, dtype=np.float32)
    z_map = np.zeros(valid.shape, dtype=np.float32)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "delta_map": delta_map,
            "z_map": z_map,
            "valid_px": 0,
            "active_a_px": 0,
            "active_b_px": 0,
            "null_center": 0.0,
            "null_scale": 0.0,
            "bilateral_suppression_frac": 0.0,
            "diff_p99": 0.0,
            "diff_p995": 0.0,
            "opp_diff_p99": 0.0,
            "opp_diff_p995": 0.0,
            "diff_excess_ratio": 0.0,
            "z_p99": 0.0,
            "z_p995": 0.0,
            "z_pos_frac": 0.0,
            "z_neg_frac": 0.0,
            "tail_energy_asymmetry": 0.5,
        }

    pa_c = np.clip(pa[valid], eps, 1.0 - eps)
    pb_c = np.clip(pb[valid], eps, 1.0 - eps)
    logit_a = np.log(pa_c / (1.0 - pa_c))
    logit_b = np.log(pb_c / (1.0 - pb_c))
    d_vals = logit_a - logit_b
    delta_map[valid] = d_vals.astype(np.float32)

    null_center = float(np.median(d_vals))
    mad_all = 1.4826 * float(np.median(np.abs(d_vals - null_center)))
    neg_dev = null_center - d_vals[d_vals <= null_center]
    mad_neg = 1.4826 * float(np.median(neg_dev)) if neg_dev.size > 0 else 0.0
    lb_med = float(np.median(logit_b))
    mad_b = 1.4826 * float(np.median(np.abs(logit_b - lb_med)))
    null_scale = float(max(mad_neg, mad_all, mad_b, DEFAULT_DIFF_SIGMA_FLOOR))

    z_vals = (d_vals - null_center) / null_scale
    z_map[valid] = z_vals.astype(np.float32)

    p005, p01, p99, p995 = [float(x) for x in np.percentile(d_vals, [0.5, 1.0, 99.0, 99.5])]
    pos_excess_p995 = max(p995 - null_center, 0.0)
    neg_excess_p995 = max(null_center - p005, 0.0)
    diff_excess_ratio = float(pos_excess_p995 / max(neg_excess_p995, null_scale))

    z_p99, z_p995 = [float(x) for x in np.percentile(z_vals, [99.0, 99.5])]
    z_pos_frac = float(np.mean(z_vals > 3.0))
    z_neg_frac = float(np.mean(z_vals < -3.0))

    pos_energy = float(np.sum(np.maximum(z_vals - 3.0, 0.0)))
    neg_energy = float(np.sum(np.maximum(-z_vals - 3.0, 0.0)))
    tot_energy = pos_energy + neg_energy
    tail_energy_asymmetry = float(pos_energy / tot_energy) if tot_energy > 0.0 else 0.5

    active_a = pa_c >= 0.5
    active_b = pb_c >= 0.5
    n_active_a = int(active_a.sum())
    n_active_b = int(active_b.sum())
    if n_active_a > 0:
        suppressed = active_a & ((pb_c >= 0.5) | (d_vals <= 1.0))
        bilateral_suppression_frac = float(suppressed.sum() / n_active_a)
    else:
        bilateral_suppression_frac = 0.0

    return {
        "delta_map": delta_map,
        "z_map": z_map,
        "valid_px": n_valid,
        "active_a_px": n_active_a,
        "active_b_px": n_active_b,
        "null_center": round(null_center, 6),
        "null_scale": round(null_scale, 6),
        "bilateral_suppression_frac": round(bilateral_suppression_frac, 6),
        "diff_p99": round(p99, 6),
        "diff_p995": round(p995, 6),
        "opp_diff_p99": round(-p01, 6),
        "opp_diff_p995": round(-p005, 6),
        "diff_excess_ratio": round(diff_excess_ratio, 6),
        "z_p99": round(z_p99, 6),
        "z_p995": round(z_p995, 6),
        "z_pos_frac": round(z_pos_frac, 6),
        "z_neg_frac": round(z_neg_frac, 6),
        "tail_energy_asymmetry": round(tail_energy_asymmetry, 6),
    }


def stroke_width_consistency(
    binary_mask: np.ndarray,
    voxel_um: float = DEFAULT_VOXEL_UM,
    *,
    narrow_band_mm: tuple[float, float] = DEFAULT_NARROW_STROKE_BAND_MM,
    wide_blob_mm: float = DEFAULT_WIDE_BLOB_MM,
) -> dict:
    """Estimate physical stroke widths (mm) via EDT sampled along the medial skeleton.

    Full local stroke width is ``2 * edt * voxel_um / 1000.0`` (mm) at each skeleton
    pixel. Narrow Greek ink strokes (``~0.25-0.65 mm``, low CV, long per-component
    skeleton) separate cleanly from wide ``1.5-4 mm`` mottle clouds and isolated dots.
    """
    if not (math.isfinite(voxel_um) and voxel_um > 0.0):
        raise ValueError(f"voxel_um must be positive and finite, got {voxel_um!r}")
    lo_mm, hi_mm = float(narrow_band_mm[0]), float(narrow_band_mm[1])
    if not (math.isfinite(lo_mm) and math.isfinite(hi_mm) and 0.0 <= lo_mm < hi_mm):
        raise ValueError(f"narrow_band_mm must satisfy 0 <= lo < hi, got {narrow_band_mm!r}")

    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array")

    fg_px = int(mask.sum())
    if fg_px == 0:
        return {
            "voxel_um": float(voxel_um),
            "foreground_px": 0,
            "n_components": 0,
            "n_skeleton_px": 0,
            "skeleton_length_mm": 0.0,
            "weighted_component_skeleton_mm": 0.0,
            "width_median_mm": None,
            "width_mean_mm": None,
            "width_std_mm": None,
            "width_iqr_mm": None,
            "width_cv": None,
            "narrow_stroke_frac": 0.0,
            "wide_blob_frac": 0.0,
            "narrow_band_mm": [lo_mm, hi_mm],
        }

    mm_per_px = float(voxel_um) / 1000.0
    edt = ndi.distance_transform_edt(mask)
    skeleton, _, n_comp, comp_stats = _medial_skeleton(mask, edt)
    radii = edt[skeleton]
    n_skel = int(radii.size)
    widths_mm = 2.0 * radii * mm_per_px

    q25, med, q75 = [float(x) for x in np.percentile(widths_mm, [25.0, 50.0, 75.0])]
    mean_w = float(np.mean(widths_mm))
    std_w = float(np.std(widths_mm))
    iqr_w = float(q75 - q25)
    cv_w = float(std_w / max(mean_w, 1e-12))
    narrow_frac = float(np.mean((widths_mm >= lo_mm) & (widths_mm <= hi_mm)))
    wide_frac = float(np.mean(widths_mm >= float(wide_blob_mm)))

    total_area = sum(a for a, _ in comp_stats)
    weighted_comp_skel_mm = (
        sum(a * (s * mm_per_px) for a, s in comp_stats) / max(total_area, 1)
        if comp_stats
        else 0.0
    )

    return {
        "voxel_um": float(voxel_um),
        "foreground_px": fg_px,
        "n_components": int(n_comp),
        "n_skeleton_px": n_skel,
        "skeleton_length_mm": round(n_skel * mm_per_px, 6),
        "weighted_component_skeleton_mm": round(float(weighted_comp_skel_mm), 6),
        "width_median_mm": round(med, 6),
        "width_mean_mm": round(mean_w, 6),
        "width_std_mm": round(std_w, 6),
        "width_iqr_mm": round(iqr_w, 6),
        "width_cv": round(cv_w, 6),
        "narrow_stroke_frac": round(narrow_frac, 6),
        "wide_blob_frac": round(wide_frac, 6),
        "narrow_band_mm": [lo_mm, hi_mm],
    }


def betti_filtration_signature(
    prob_map: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_BETTI_THRESHOLDS,
    min_area_px: int = 25,
    *,
    min_hole_px: int | None = None,
) -> dict:
    """Compute 2D Betti filtration ``(b0, b1)`` and loop persistence across thresholds.

    At each threshold ``t``, thresholds ``(prob_map >= t) & valid_mask``, removes
    8-connected foreground components smaller than ``min_area_px``, and counts:

    * ``b0``: surviving 8-connected foreground components;
    * ``b1``: 4-connected interior background holes not touching the image border or
      any invalid pixel (``~valid_mask``) with area at least ``min_hole_px``.

    Persistent ``b1 > 0`` separates enclosed Greek letterforms (``O, Theta, Alpha,
    Delta, Rho``) from isolated dot rows and featureless mottle clouds.
    """
    prob = np.asarray(prob_map, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if prob.ndim != 2 or valid.ndim != 2 or prob.shape != valid.shape:
        raise ValueError("prob_map and valid_mask must be 2D arrays with identical shape")
    ts = tuple(sorted(float(t) for t in thresholds))
    if not ts or not all(math.isfinite(t) for t in ts):
        raise ValueError("thresholds must be a non-empty sequence of finite floats")
    if min_area_px < 1:
        raise ValueError(f"min_area_px must be >= 1, got {min_area_px!r}")

    hole_floor = (
        min(9, max(1, int(min_area_px) // 3))
        if min_hole_px is None
        else max(1, int(min_hole_px))
    )
    finite_valid = valid & np.isfinite(prob)
    has_valid = bool(finite_valid.any())
    all_valid = bool(finite_valid.all())

    levels = []
    b0s: list[int] = []
    b1s: list[int] = []

    for t in ts:
        if not has_valid:
            b0, b1, fg_px, hole_px = 0, 0, 0, 0
        else:
            fg = (prob >= t) & finite_valid
            fg_labels, n_fg = ndi.label(fg, structure=_CONN8)
            if n_fg == 0:
                b0, b1, fg_px, hole_px = 0, 0, 0, 0
            else:
                sizes = np.bincount(fg_labels.ravel(), minlength=n_fg + 1)
                keep = sizes >= min_area_px
                keep[0] = False
                b0 = int(keep.sum())
                if b0 == 0:
                    b1, fg_px, hole_px = 0, 0, 0
                else:
                    fg_clean = keep[fg_labels]
                    fg_px = int(fg_clean.sum())
                    bg_labels, n_bg = ndi.label(~fg_clean, structure=_CONN4)
                    if n_bg == 0:
                        b1, hole_px = 0, 0
                    else:
                        is_ext = np.zeros(n_bg + 1, dtype=bool)
                        is_ext[0] = True
                        is_ext[bg_labels[0, :]] = True
                        is_ext[bg_labels[-1, :]] = True
                        is_ext[bg_labels[:, 0]] = True
                        is_ext[bg_labels[:, -1]] = True
                        if not all_valid:
                            is_ext[bg_labels[~finite_valid]] = True
                        bg_sizes = np.bincount(bg_labels.ravel(), minlength=n_bg + 1)
                        is_hole = (~is_ext) & (bg_sizes >= hole_floor)
                        b1 = int(is_hole.sum())
                        hole_px = int(bg_sizes[is_hole].sum())
        b0s.append(b0)
        b1s.append(b1)
        levels.append(
            {
                "threshold": round(t, 6),
                "b0": b0,
                "b1": b1,
                "euler_characteristic": int(b0 - b1),
                "fg_px": fg_px,
                "hole_px": hole_px,
            }
        )

    t_arr = np.asarray(ts, dtype=np.float64)
    b0_arr = np.asarray(b0s, dtype=np.float64)
    b1_arr = np.asarray(b1s, dtype=np.float64)
    max_b0 = int(max(b0s))
    max_b1 = int(max(b1s))

    if len(ts) > 1 and t_arr[-1] > t_arr[0]:
        integrated_b1 = float(np.trapezoid(b1_arr, t_arr))
        span = float(t_arr[-1] - t_arr[0])
    else:
        integrated_b1 = float(b1_arr[0])
        span = 1.0
    norm_integrated_b1 = integrated_b1 / span

    b1_active = [k for k, v in enumerate(b1s) if v > 0]
    b1_persistence_frac = float(len(b1_active) / len(ts))
    if max_b1 == 0:
        loop_persistence_score = 0.0
    else:
        mean_b1_per_b0 = float(np.mean([b1s[k] / max(b0s[k], 1) for k in b1_active]))
        loop_persistence_score = float(
            b1_persistence_frac
            * (1.0 - math.exp(-2.0 * norm_integrated_b1))
            * min(1.0, max(0.0, mean_b1_per_b0 / 0.25))
        )

    return {
        "thresholds": [round(t, 6) for t in ts],
        "min_area_px": int(min_area_px),
        "min_hole_px": int(hole_floor),
        "levels": levels,
        "max_b0": max_b0,
        "max_b1": max_b1,
        "mean_b0": round(float(np.mean(b0_arr)), 6),
        "mean_b1": round(float(np.mean(b1_arr)), 6),
        "integrated_b1": round(integrated_b1, 6),
        "b1_persistence_frac": round(b1_persistence_frac, 6),
        "loop_persistence_score": round(loop_persistence_score, 6),
    }


def _strip_diff_arrays(diff_res: dict) -> dict:
    """Return a JSON-serializable copy of :func:`opposite_face_differential_map` output."""
    return {k: v for k, v in diff_res.items() if k not in ("delta_map", "z_map")}


def _stroke_level_score(sw: dict) -> float:
    """Non-gating [0, 1] stroke quality score for a single threshold mask."""
    if sw["n_skeleton_px"] == 0 or sw["width_median_mm"] is None or sw["width_cv"] is None:
        return 0.0
    narrow = float(sw["narrow_stroke_frac"])
    cv_term = max(0.0, min(1.0, 1.0 - 0.5 * float(sw["width_cv"])))
    length_term = min(1.0, float(sw["weighted_component_skeleton_mm"]) / 1.5)
    return float(narrow * cv_term * length_term)


def _regime_hint(
    active_px: int,
    single_face_score: float,
    stroke_score: float,
    topology_score: float,
) -> str:
    """Advisory (non-gating) morphological regime label."""
    if active_px == 0:
        return "no_signal"
    if single_face_score < 0.35:
        return "bilateral_cloud_like"
    if stroke_score < 0.30 and topology_score < 0.20:
        return "single_face_dot_or_wide_like"
    if topology_score >= 0.35:
        return "single_face_looped_stroke_like"
    return "single_face_stroke_like"


def evaluate_transverse_fp_features(
    prob_asc: np.ndarray,
    prob_desc: np.ndarray,
    valid_mask: np.ndarray,
    voxel_um: float = DEFAULT_VOXEL_UM,
    *,
    eps: float = 1e-4,
    hi_threshold: float = 0.7,
    lo_threshold: float = 0.5,
    betti_thresholds: Sequence[float] = DEFAULT_BETTI_THRESHOLDS,
    min_area_px: int = 25,
) -> dict:
    """Evaluate opposite-face differential, stroke-width, and Betti topology features.

    Evaluates both ``asc`` and ``desc`` face directions and returns a single
    JSON-serializable summary dict combining:

    * opposite-face log-odds differential self-calibration for both faces;
    * dominant-face stroke-width consistency at ``hi >= 0.7`` and ``lo >= 0.5``;
    * 2D Betti filtration signatures for both faces;
    * a non-gating composite discriminability summary separating narrow single-face
      Greek letterforms from bilateral ``2-4 mm`` mottle clouds and isolated single-face
      dot rows.
    """
    if not (math.isfinite(voxel_um) and voxel_um > 0.0):
        raise ValueError(f"voxel_um must be positive and finite, got {voxel_um!r}")
    pa, pb, valid = _sanitize_prob_pair(prob_asc, prob_desc, valid_mask)
    maps = {"asc": pa, "desc": pb}

    diff_by_face = {
        "asc": _strip_diff_arrays(opposite_face_differential_map(pa, pb, valid, eps=eps)),
        "desc": _strip_diff_arrays(opposite_face_differential_map(pb, pa, valid, eps=eps)),
    }

    hi_counts = {f: int(((maps[f] >= hi_threshold) & valid).sum()) for f in FACES}
    lo_counts = {f: int(((maps[f] >= lo_threshold) & valid).sum()) for f in FACES}
    dom = max(
        FACES,
        key=lambda f: (hi_counts[f], lo_counts[f], diff_by_face[f]["diff_p995"]),
    )
    opp = "desc" if dom == "asc" else "asc"

    hi_mask = (maps[dom] >= hi_threshold) & valid
    lo_mask = (maps[dom] >= lo_threshold) & valid
    stroke_hi = stroke_width_consistency(hi_mask, voxel_um=voxel_um)
    stroke_lo = stroke_width_consistency(lo_mask, voxel_um=voxel_um)

    betti_by_face = {
        f: betti_filtration_signature(
            maps[f],
            valid,
            thresholds=betti_thresholds,
            min_area_px=min_area_px,
        )
        for f in FACES
    }
    betti_dom = betti_by_face[dom]
    dom_diff = diff_by_face[dom]

    if dom_diff["active_a_px"] == 0:
        single_face_score = 0.0
    else:
        excess_term = max(0.0, min(1.0, (dom_diff["diff_excess_ratio"] - 1.0) / 3.0))
        tail_term = max(0.0, min(1.0, (dom_diff["tail_energy_asymmetry"] - 0.5) / 0.45))
        single_face_score = float((1.0 - dom_diff["bilateral_suppression_frac"]) * max(excess_term, tail_term))

    stroke_hi_score = _stroke_level_score(stroke_hi)
    stroke_lo_score = _stroke_level_score(stroke_lo)
    if stroke_hi["n_skeleton_px"] > 0:
        stroke_consistency_score = float(0.6 * stroke_hi_score + 0.4 * stroke_lo_score)
    else:
        stroke_consistency_score = float(0.4 * stroke_lo_score)

    topology_score = float(betti_dom["loop_persistence_score"])
    discriminability_score = float(
        single_face_score * (0.55 * stroke_consistency_score + 0.45 * topology_score)
    )

    n_valid = int(valid.sum())
    px_cm2 = (float(voxel_um) / 10000.0) ** 2

    return {
        "voxel_um": float(voxel_um),
        "valid_px": n_valid,
        "valid_cm2": round(n_valid * px_cm2, 6),
        "dominant_face": dom,
        "opposite_face": opp,
        "hit_counts": {
            "hi_threshold": float(hi_threshold),
            "lo_threshold": float(lo_threshold),
            "hi": hi_counts,
            "lo": lo_counts,
        },
        "differential": {
            "dominant": dom_diff,
            "by_face": diff_by_face,
        },
        "stroke_width": {
            "face": dom,
            "hi": stroke_hi,
            "lo": stroke_lo,
        },
        "betti": {
            "dominant": betti_dom,
            "by_face": betti_by_face,
        },
        "composite": {
            "gating": False,
            "single_face_score": round(single_face_score, 6),
            "stroke_consistency_score": round(stroke_consistency_score, 6),
            "topology_score": round(topology_score, 6),
            "discriminability_score": round(discriminability_score, 6),
            "regime_hint": _regime_hint(
                dom_diff["active_a_px"],
                single_face_score,
                stroke_consistency_score,
                topology_score,
            ),
        },
    }
