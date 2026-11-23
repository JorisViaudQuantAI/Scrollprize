# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CPU-only (< 250 MB) 1.129 um papyrus-peak surface recentering & crust/crackle scanner for PHerc. Paris 4.

Builds on ``gemini_lane/paris4_dual_energy.py`` (TPS r2 + local 3D NCC lock) to:
(a) Recenter each 1.129 um native slab onto the local papyrus sheet peak along the
    surface normal (``argmax_n mu_1129`` with sub-voxel parabolic refinement, bounded
    strictly inside ``|n| <= 36 um`` so it cannot jump to an adjacent wrap).
(b) Score recentered surface crust/crackle/HF-roughness/top-hat features:
    - Recentered depth top-hat (``|n'| <= 4 um`` minus ``|n'| in [10, 22] um``)
    - Local HF micro-roughness RMS (2D Laplacian local RMS on recentered surface)
    - Surface-minus-bulk HF roughness excess (cancels radial CT noise gradients)
    - Crackle DoG micro-texture RMS (2-8 um band-pass local RMS)
    - Isotropic crackle structure-tensor minor eigenvalue ``sqrt(lambda_2)``
      (suppresses 1D papyrus fibres, isolates 2D polygonal ink crackle/crust)
    - Control-B standardized equal-weight composite crust z-score
    plus wrap-behind ghosting exclusion (``90-260 um``).
(c) Validate quantitatively on Control A (known carbon ink column) vs Control B
    (confirmed blank agraphon) at 1.129 um, reporting ROC-AUC, Cohen's d, blank
    noise floor, and empirical 3-sigma / 4-sigma FPR.
(d) ONLY IF Control A vs Control B achieves ``ROC AUC >= 0.70`` (positive direction,
    NCC-locked), execute a targeted 1.129 um crust scan across non-overlapping
    agraphon / title-band tiles strictly within the ``< 250 MB`` streaming budget.

Privacy & resource guardrails:
- 100% CPU-only (zero CUDA / PyTorch dependency).
- Streaming OME-Zarr v2 + HTTP Range ``tifxyz`` in RAM (``cache_mb <= 256.0``,
  total streamed ``< 250 MB``); zero persistent CT files on disk.
- Public JSON receipt contains ZERO local filesystem paths and ZERO unread
  candidate coordinates/images; all private maps/PNGs/coordinates are written
  strictly outside any git repository.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None

requests = _requests

try:
    from . import paris4_dual_energy as p4
    from .stream_snap_render import RemoteZarrV2, _sanitize_source_label
except ImportError:  # pragma: no cover - script execution
    import paris4_dual_energy as p4
    from stream_snap_render import RemoteZarrV2, _sanitize_source_label


# ---------------------------------------------------------------------------
# Default control & non-overlapping agraphon tile definitions on w010-027
# (inside 1.129 um axial FOV z_78 = 28,080..56,356 -> v ~ 900..2400)
# ---------------------------------------------------------------------------

DEFAULT_CONTROL_PAIRS: tuple[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]], ...] = (
    # Pair 0 reproduces the exact 918fbb4 1.129 um preflight location (z_78 ~ 52,750).
    ("pair_0_v1100", (1100, 1103, 4092, 4096), (1100, 1103, 5122, 5126)),
    # Pair 1 (+2.4 mm along column, z_78 ~ 51,860) probes across the 6.75 mm line pitch.
    ("pair_1_v1150", (1150, 1153, 4092, 4096), (1150, 1153, 5122, 5126)),
    # Pair 2 (Step 0 band z_78 ~ 46,900) provides a third independent height on Control A vs B.
    ("pair_2_v1430", (1430, 1433, 4092, 4096), (1430, 1433, 5122, 5126)),
)

# Non-overlapping agraphon / title-band tiles (strictly distinct from Control A/B wraps).
DEFAULT_AGRAPH_SCAN_TILES: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("agraph_w09_v1100", (1100, 1103, 5600, 5604)),
    ("agraph_w10_v1100", (1100, 1103, 6040, 6044)),
    ("agraph_w11_v1100", (1100, 1103, 6450, 6454)),
    ("agraph_w12_v1430", (1430, 1433, 6825, 6829)),
    ("agraph_w13_v1430", (1430, 1433, 7175, 7179)),
    ("agraph_w14_v1750", (1750, 1753, 7495, 7499)),
    ("agraph_w15_v1750", (1750, 1753, 7795, 7799)),
    ("agraph_w16_v2000", (2000, 2003, 8075, 8079)),
)

GATED_CRUST_METRICS: tuple[str, ...] = (
    "tophat_recentered",
    "hf_roughness_local",
    "hf_roughness_excess",
    "crackle_dog_rms",
    "crackle_isotropic",
    "crust_composite_z",
)

CRUST_FORBIDDEN_UNREAD_KEYS = p4.FORBIDDEN_UNREAD_COORD_KEYS | frozenset({"tiles", "tile_id"})


# ---------------------------------------------------------------------------
# 1. Mask-normalized 2D/3D filtering helpers
# ---------------------------------------------------------------------------

def _norm_gauss_2d(arr: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Mask-normalized 2D Gaussian smoothing (zero edge bleed across invalid masks)."""
    v = np.asarray(arr, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if sigma_px <= 1e-6 or not np.any(m > 0.5):
        return np.where(m > 0.5, v, 0.0)
    num = ndi.gaussian_filter(np.where(m > 0.5, v, 0.0), sigma=float(sigma_px), mode="constant", cval=0.0)
    den = ndi.gaussian_filter(m, sigma=float(sigma_px), mode="constant", cval=0.0)
    return np.where(den > 1e-6, num / np.where(den > 1e-6, den, 1.0), 0.0)


def _norm_gauss_3d(
    arr: np.ndarray,
    mask: np.ndarray,
    sigmas_nvu: tuple[float, float, float],
) -> np.ndarray:
    """Mask-normalized 3D Gaussian smoothing."""
    v = np.asarray(arr, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if not np.any(m > 0.5) or all(s <= 1e-6 for s in sigmas_nvu):
        return np.where(m > 0.5, v, 0.0)
    num = ndi.gaussian_filter(np.where(m > 0.5, v, 0.0), sigma=sigmas_nvu, mode="constant", cval=0.0)
    den = ndi.gaussian_filter(m, sigma=sigmas_nvu, mode="constant", cval=0.0)
    return np.where(den > 1e-6, num / np.where(den > 1e-6, den, 1.0), 0.0)


# ---------------------------------------------------------------------------
# 2. Local papyrus-peak surface recentering along the normal (argmax_n mu_1129)
# ---------------------------------------------------------------------------

def recenter_papyrus_peak_slab(
    mu_slab: np.ndarray,
    valid_3d: np.ndarray,
    n_offsets_um: Sequence[float],
    *,
    uv_spacing_um: float,
    search_half_width_um: float = 36.0,
    local_track_half_width_um: float = 10.0,
    smooth_n_um: float = 4.0,
    smooth_uv_um: float = 12.0,
    smooth_peak_uv_um: float = 8.0,
    out_n_offsets_um: Sequence[float] | None = None,
    min_valid_depths: int = 5,
) -> dict[str, Any]:
    """Two-stage continuous papyrus-peak recentering along normal ``n`` (``argmax_n mu_1129``).

    Stage 1 finds the tile's dominant sheet peak ``n_tile_anchor_um`` within
    ``|n| <= search_half_width_um`` (preferring the local peak nearest ``n = 0``
    among peaks >= 85% of max, so adjacent wraps cannot hijack the sheet).
    Stage 2 refines per-pixel ``n_peak(v, u)`` via sub-voxel parabolic ``argmax_n``
    strictly within ``|n - n_tile_anchor_um| <= local_track_half_width_um`` and
    smooths ``n_peak(v, u)`` laterally, preventing discontinuous bimodal layer-jump
    cliffs that would otherwise create spurious HF Laplacian artifacts.
    """
    mu = np.asarray(mu_slab, dtype=np.float32)
    val = np.asarray(valid_3d, dtype=bool)
    n_in = np.asarray(n_offsets_um, dtype=np.float64)
    if mu.ndim != 3 or val.shape != mu.shape or len(n_in) != mu.shape[0]:
        raise ValueError(f"Mismatched slab/mask/offset shapes: {mu.shape}, {val.shape}, {len(n_in)}")
    if len(n_in) < 3:
        raise ValueError("n_offsets_um must have at least 3 depth samples for peak recentering.")

    diffs = np.diff(n_in)
    dn_um = float(np.median(diffs))
    if dn_um <= 0.0 or not np.allclose(diffs, dn_um, rtol=1e-3, atol=1e-4):
        raise ValueError("n_offsets_um must be strictly increasing and uniformly spaced.")

    n_out = (
        np.asarray(out_n_offsets_um, dtype=np.float64)
        if out_n_offsets_um is not None
        else np.arange(-12, 13, dtype=np.float64) * dn_um
    )

    search_idx = np.flatnonzero(np.abs(n_in) <= float(search_half_width_um) + 1e-6)
    if len(search_idx) < 3:
        raise ValueError(
            f"search_half_width_um={search_half_width_um} selects fewer than 3 planes in n_offsets_um."
        )

    # Stage 1: Tile-level 1D mean attenuation profile within |n| <= search_half_width_um.
    prof = np.full(len(search_idx), -np.inf, dtype=np.float64)
    prof_ok = np.zeros(len(search_idx), dtype=bool)
    for j, k in enumerate(search_idx):
        mk = val[k]
        if mk.sum() >= 8:
            prof[j] = float(mu[k][mk].mean())
            prof_ok[j] = True

    n_tile_anchor_um = 0.0
    if prof_ok.any():
        p_fill = np.where(prof_ok, prof, float(np.min(prof[prof_ok])))
        p_smooth = ndi.gaussian_filter1d(p_fill, sigma=max(0.5, 3.0 / dn_um), mode="nearest")
        p_smooth = np.where(prof_ok, p_smooth, -np.inf)
        p_max = float(np.max(p_smooth))
        p_min = float(np.min(p_smooth[prof_ok]))
        p_rng = max(p_max - p_min, 1e-9)
        # Identify local maxima >= 85% of peak prominence; pick closest to n = 0 to stay on meshed sheet.
        cand_j: list[int] = []
        for j in range(len(search_idx)):
            if not prof_ok[j]:
                continue
            left_ok = (j == 0) or (p_smooth[j] >= p_smooth[j - 1])
            right_ok = (j == len(search_idx) - 1) or (p_smooth[j] >= p_smooth[j + 1])
            if left_ok and right_ok and (p_smooth[j] - p_min) >= 0.85 * p_rng:
                cand_j.append(j)
        if cand_j:
            best_j = min(cand_j, key=lambda j: abs(float(n_in[search_idx[j]])))
        else:
            best_j = int(np.argmax(p_smooth))
        n_tile_anchor_um = float(n_in[search_idx[best_j]])

    # Stage 2: Local corridor around n_tile_anchor_um for smooth continuous per-pixel peak tracking.
    corridor_idx = np.flatnonzero(
        (np.abs(n_in) <= float(search_half_width_um) + 1e-6)
        & (np.abs(n_in - n_tile_anchor_um) <= float(local_track_half_width_um) + 1e-6)
    )
    if len(corridor_idx) < 3:
        corridor_idx = search_idx

    sig_n = float(smooth_n_um) / dn_um
    sig_uv = float(smooth_uv_um) / max(float(uv_spacing_um), 1e-6)
    mu_smooth = _norm_gauss_3d(mu, val, (sig_n, sig_uv, sig_uv))

    sub_val = val[corridor_idx]
    valid_count = sub_val.sum(axis=0)
    peak_valid = valid_count >= min(int(min_valid_depths), len(corridor_idx))

    h, w = mu.shape[1], mu.shape[2]
    n_peak_um = np.zeros((h, w), dtype=np.float32)
    k_peak = np.zeros((h, w), dtype=np.float64)

    if np.any(peak_valid):
        sub_mu = np.where(sub_val, mu_smooth[corridor_idx], -np.inf)
        local_argmax = np.argmax(sub_mu, axis=0)
        k_max = corridor_idx[local_argmax]

        # Parabolic sub-voxel refinement along n around k_max.
        vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        k_lo = np.maximum(k_max - 1, int(corridor_idx[0]))
        k_hi = np.minimum(k_max + 1, int(corridor_idx[-1]))
        can_refine = (
            peak_valid
            & (k_max > int(corridor_idx[0]))
            & (k_max < int(corridor_idx[-1]))
            & val[k_lo, vv, uu]
            & val[k_max, vv, uu]
            & val[k_hi, vv, uu]
        )
        yl = mu_smooth[k_lo, vv, uu]
        yc = mu_smooth[k_max, vv, uu]
        yr = mu_smooth[k_hi, vv, uu]
        d2 = yl - 2.0 * yc + yr
        good_curv = can_refine & (d2 < -1e-9) & (yc >= yl) & (yc >= yr)
        delta_k = np.where(good_curv, np.clip(0.5 * (yl - yr) / np.where(good_curv, d2, -1.0), -0.5, 0.5), 0.0)
        k_peak_raw = k_max.astype(np.float64) + delta_k
        n_peak_raw = np.where(peak_valid, n_in[0] + k_peak_raw * dn_um, 0.0)
        if float(smooth_peak_uv_um) > 0.0:
            sig_pk = float(smooth_peak_uv_um) / max(float(uv_spacing_um), 1e-6)
            n_peak_s = _norm_gauss_2d(n_peak_raw, peak_valid, sig_pk)
        else:
            n_peak_s = n_peak_raw
        n_peak_um = np.where(peak_valid, n_peak_s, 0.0).astype(np.float32)
        k_peak = (n_peak_um.astype(np.float64) - n_in[0]) / dn_um

    # Resample raw native mu_slab along normal axis onto n' = n - n_peak_um(v, u).
    n_out_len = len(n_out)
    mu_rec = np.zeros((n_out_len, h, w), dtype=np.float32)
    val_rec = np.zeros((n_out_len, h, w), dtype=bool)
    if np.any(peak_valid):
        vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        k_src = k_peak[None, :, :] + (n_out[:, None, None] / dn_um)
        in_b = (k_src >= 0.0) & (k_src <= float(mu.shape[0] - 1)) & peak_valid[None, :, :]
        k_clamped = np.clip(k_src, 0.0, float(mu.shape[0] - 1))
        k0 = np.floor(k_clamped).astype(np.int64)
        k1 = np.minimum(k0 + 1, mu.shape[0] - 1)
        frac = (k_clamped - k0).astype(np.float32)

        v0 = val[k0, vv[None, :, :], uu[None, :, :]]
        v1 = val[k1, vv[None, :, :], uu[None, :, :]]
        ok = in_b & v0 & v1
        m0 = mu[k0, vv[None, :, :], uu[None, :, :]]
        m1 = mu[k1, vv[None, :, :], uu[None, :, :]]
        mu_rec = np.where(ok, (1.0 - frac) * m0 + frac * m1, np.float32(0.0)).astype(np.float32)
        val_rec = ok

    # Pre- vs post-recenter surface mean diagnostics (|n| <= 4 um).
    pre_sel = np.abs(n_in) <= 4.0 + 1e-6
    post_sel = np.abs(n_out) <= 4.0 + 1e-6
    pre_v = val[pre_sel]
    post_v = val_rec[post_sel]
    pre_mean = float(mu[pre_sel][pre_v].mean()) if pre_v.any() else 0.0
    post_mean = float(mu_rec[post_sel][post_v].mean()) if post_v.any() else 0.0

    pk_vals = n_peak_um[peak_valid].astype(np.float64)
    stats: dict[str, float | int] = {
        "search_half_width_um": float(search_half_width_um),
        "local_track_half_width_um": float(local_track_half_width_um),
        "n_tile_anchor_um": round(n_tile_anchor_um, 3),
        "peak_valid_pixels": int(peak_valid.sum()),
        "peak_valid_fraction": round(float(peak_valid.mean()), 4),
        "n_peak_mean_um": round(float(pk_vals.mean()), 3) if pk_vals.size else 0.0,
        "n_peak_median_um": round(float(np.median(pk_vals)), 3) if pk_vals.size else 0.0,
        "n_peak_std_um": round(float(pk_vals.std()), 3) if pk_vals.size else 0.0,
        "n_peak_p05_um": round(float(np.percentile(pk_vals, 5)), 3) if pk_vals.size else 0.0,
        "n_peak_p95_um": round(float(np.percentile(pk_vals, 95)), 3) if pk_vals.size else 0.0,
        "n_peak_abs_max_um": round(float(np.abs(pk_vals).max()), 3) if pk_vals.size else 0.0,
        "pre_recenter_surface_mu_mean": round(pre_mean, 6),
        "post_recenter_surface_mu_mean": round(post_mean, 6),
        "surface_mu_gain": round(post_mean - pre_mean, 6),
    }
    return {
        "mu_recentered": mu_rec,
        "valid_recentered": val_rec,
        "n_offsets_um": n_out,
        "n_peak_um": n_peak_um,
        "peak_valid": peak_valid,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# 3. Surface crust / crackle / HF-roughness / top-hat scoring & ghosting
# ---------------------------------------------------------------------------

def compute_crust_crackle_maps(
    mu_recentered: np.ndarray,
    valid_recentered: np.ndarray,
    n_offsets_um: Sequence[float],
    *,
    uv_spacing_um: float,
    surface_max_um: float = 4.0,
    bulk_min_um: float = 10.0,
    bulk_max_um: float = 22.0,
    dog_fine_um: float = 2.0,
    dog_coarse_um: float = 8.0,
    local_window_um: float = 12.0,
    grad_sigma_um: float = 2.0,
) -> dict[str, Any]:
    """Compute recentered 1.129 um crust/crackle/HF-roughness/top-hat maps and scalar metrics."""
    th, surf, bulk, v2d = p4.compute_depth_tophat(
        mu_recentered,
        valid_recentered,
        n_offsets_um,
        surface_max_um=surface_max_um,
        bulk_min_um=bulk_min_um,
        bulk_max_um=bulk_max_um,
    )
    interior = ndi.binary_erosion(v2d, iterations=1) if v2d.any() else v2d
    uv_um = max(float(uv_spacing_um), 1e-6)
    win_px = float(local_window_um) / uv_um

    # 1. Surface and bulk HF micro-roughness (2D discrete Laplacian local RMS).
    surf_f64 = np.where(v2d, surf.astype(np.float64), 0.0)
    bulk_f64 = np.where(v2d, bulk.astype(np.float64), 0.0)
    lap_surf = np.where(interior, ndi.laplace(surf_f64), 0.0)
    lap_bulk = np.where(interior, ndi.laplace(bulk_f64), 0.0)

    hf_surf_local = np.sqrt(np.maximum(0.0, _norm_gauss_2d(lap_surf ** 2, interior, win_px)))
    hf_bulk_local = np.sqrt(np.maximum(0.0, _norm_gauss_2d(lap_bulk ** 2, interior, win_px)))
    hf_excess = np.where(interior, hf_surf_local - hf_bulk_local, 0.0)

    # 2. Band-pass crackle micro-texture DoG RMS (2-8 um scale).
    dog_lo = _norm_gauss_2d(surf_f64, v2d, float(dog_fine_um) / uv_um)
    dog_hi = _norm_gauss_2d(surf_f64, v2d, float(dog_coarse_um) / uv_um)
    dog_bp = np.where(v2d, dog_lo - dog_hi, 0.0)
    crackle_dog = np.sqrt(np.maximum(0.0, _norm_gauss_2d(dog_bp ** 2, interior, win_px)))

    # 3. Isotropic crackle structure-tensor minor eigenvalue sqrt(lambda_2).
    # Suppresses unidirectional 1D papyrus fibers (lambda_2 ~ 0), peaks on 2D polygonal crackle.
    surf_g = _norm_gauss_2d(surf_f64, v2d, float(grad_sigma_um) / uv_um)
    gv, gu = np.gradient(surf_g, uv_um)
    gv = np.where(interior, gv, 0.0)
    gu = np.where(interior, gu, 0.0)
    j_vv = _norm_gauss_2d(gv * gv, interior, win_px)
    j_uu = _norm_gauss_2d(gu * gu, interior, win_px)
    j_vu = _norm_gauss_2d(gv * gu, interior, win_px)
    tr = j_vv + j_uu
    disc = np.sqrt(np.maximum(0.0, (j_vv - j_uu) ** 2 + 4.0 * (j_vu ** 2)))
    lam2 = np.maximum(0.0, 0.5 * (tr - disc))
    crackle_iso = np.where(interior, np.sqrt(lam2), 0.0)

    lap_vals = lap_surf[interior]
    hf_rms_global = float(np.sqrt(np.mean(lap_vals ** 2))) if lap_vals.size else 0.0
    lap_b_vals = lap_bulk[interior]
    hf_bulk_rms_global = float(np.sqrt(np.mean(lap_b_vals ** 2))) if lap_b_vals.size else 0.0

    def _mean_std(arr: np.ndarray, m: np.ndarray) -> tuple[float, float]:
        vals = arr[m]
        if vals.size == 0:
            return 0.0, 0.0
        return round(float(vals.mean()), 6), round(float(vals.std()), 6)

    surf_m, surf_s = _mean_std(surf, v2d)
    bulk_m, _ = _mean_std(bulk, v2d)
    th_m, th_s = _mean_std(th, v2d)
    hfl_m, hfl_s = _mean_std(hf_surf_local, interior)
    hfe_m, hfe_s = _mean_std(hf_excess, interior)
    dog_m, dog_s = _mean_std(crackle_dog, interior)
    iso_m, iso_s = _mean_std(crackle_iso, interior)

    summary: dict[str, float | int] = {
        "valid_pixels": int(v2d.sum()),
        "interior_pixels": int(interior.sum()),
        "surface_mu_mean": surf_m,
        "surface_mu_std": surf_s,
        "bulk_mu_mean": bulk_m,
        "tophat_mean": th_m,
        "tophat_std": th_s,
        "surface_hf_roughness_rms": round(hf_rms_global, 6),
        "bulk_hf_roughness_rms": round(hf_bulk_rms_global, 6),
        "hf_roughness_local_mean": hfl_m,
        "hf_roughness_local_std": hfl_s,
        "hf_roughness_excess_mean": hfe_m,
        "hf_roughness_excess_std": hfe_s,
        "crackle_dog_rms_mean": dog_m,
        "crackle_dog_rms_std": dog_s,
        "crackle_isotropic_mean": iso_m,
        "crackle_isotropic_std": iso_s,
    }
    return {
        "valid_2d": v2d,
        "interior_2d": interior,
        "maps": {
            "surface_mu_recentered": surf.astype(np.float32),
            "bulk_mu_recentered": bulk.astype(np.float32),
            "tophat_recentered": th.astype(np.float32),
            "hf_roughness_local": np.where(interior, hf_surf_local, 0.0).astype(np.float32),
            "hf_roughness_excess": hf_excess.astype(np.float32),
            "crackle_dog_rms": np.where(interior, crackle_dog, 0.0).astype(np.float32),
            "crackle_isotropic": crackle_iso.astype(np.float32),
        },
        "summary": summary,
    }


def evaluate_crust_ghosting_exclusion(
    mu_ghost_slab: np.ndarray,
    valid_ghost_3d: np.ndarray,
    ghost_offsets_um: Sequence[float],
    *,
    uv_spacing_um: float,
    candidate_mask_2d: np.ndarray | None = None,
    surface_max_um: float = 12.0,
    ghost_min_um: float = 90.0,
    ghost_max_um: float = 260.0,
    local_window_um: float = 12.0,
) -> dict[str, Any]:
    """Check whether crust/HF-roughness peaks in the 90-260 um wrap-behind zone.

    Evaluates 2D local Laplacian HF roughness energy at each depth plane in
    ``ghost_offsets_um`` (e.g. ``[-150, -110, 0, +110, +150] um`` relative to the
    recentered sheet peak). Flags ``is_ghost_dominated`` if a wrap-behind plane
    exceeds the surface by > 15% with spatial correlation ``r > 0.25``, OR if
    wrap-behind HF energy exceeds surface HF energy by > 35% inside the ROI.
    """
    arr = np.asarray(mu_ghost_slab, dtype=np.float64)
    val = np.asarray(valid_ghost_3d, dtype=bool)
    offs = np.asarray(ghost_offsets_um, dtype=np.float64)
    abs_offs = np.abs(offs)

    surf_idx = np.flatnonzero(abs_offs <= float(surface_max_um) + 1e-6)
    ghost_idx = np.flatnonzero((abs_offs >= float(ghost_min_um) - 1e-6) & (abs_offs <= float(ghost_max_um) + 1e-6))
    if len(surf_idx) == 0 or len(ghost_idx) == 0:
        return {
            "checked": False,
            "reason": "missing_surface_or_ghost_depths",
            "is_ghost_dominated": False,
        }

    win_px = float(local_window_um) / max(float(uv_spacing_um), 1e-6)
    hf_planes = np.zeros_like(arr, dtype=np.float64)
    int_planes = np.zeros_like(val, dtype=bool)
    for k in range(len(offs)):
        mk = val[k]
        ik = ndi.binary_erosion(mk, iterations=1) if mk.any() else mk
        lap_k = np.where(ik, ndi.laplace(np.where(mk, arr[k], 0.0)), 0.0)
        hf_planes[k] = np.sqrt(np.maximum(0.0, _norm_gauss_2d(lap_k ** 2, ik, win_px)))
        int_planes[k] = ik

    if candidate_mask_2d is not None and np.any(candidate_mask_2d):
        roi = np.asarray(candidate_mask_2d, dtype=bool)
    else:
        roi = int_planes[surf_idx].any(axis=0) & int_planes[ghost_idx].any(axis=0)

    if not np.any(roi):
        return {
            "checked": False,
            "reason": "empty_roi",
            "is_ghost_dominated": False,
        }

    depth_p95: list[float] = []
    for k in range(len(offs)):
        mk = int_planes[k] & roi
        if mk.sum() >= 4:
            depth_p95.append(float(np.percentile(hf_planes[k][mk], 95)))
        else:
            depth_p95.append(0.0)
    depth_p95_arr = np.asarray(depth_p95, dtype=np.float64)

    surf_peak = float(depth_p95_arr[surf_idx].max())
    ghost_best_rel = int(np.argmax(depth_p95_arr[ghost_idx]))
    ghost_best_k = int(ghost_idx[ghost_best_rel])
    ghost_peak = float(depth_p95_arr[ghost_best_k])
    peak_k = int(np.argmax(depth_p95_arr))
    peak_depth_um = float(offs[peak_k])
    ratio = float(ghost_peak / max(surf_peak, 1e-9))

    surf_map = np.mean(hf_planes[surf_idx], axis=0)
    ghost_map = hf_planes[ghost_best_k]
    joint = roi & int_planes[surf_idx].all(axis=0) & int_planes[ghost_best_k]
    corr = p4._masked_pearson_r(surf_map, ghost_map, joint, min_voxels=8) if joint.sum() >= 8 else 0.0

    is_ghost = bool((ratio > 1.35) or (ratio > 1.15 and corr > 0.25))
    return {
        "checked": True,
        "roi_pixels": int(roi.sum()),
        "surface_hf_p95": round(surf_peak, 6),
        "ghost_hf_p95": round(ghost_peak, 6),
        "ghost_to_surface_hf_ratio": round(ratio, 4),
        "peak_hf_depth_um": round(peak_depth_um, 2),
        "surface_vs_ghost_hf_r": round(float(corr), 4),
        "is_ghost_dominated": is_ghost,
    }


def build_composite_crust_z(
    maps: dict[str, np.ndarray],
    interior_mask: np.ndarray,
    blank_ref: dict[str, tuple[float, float]],
    *,
    components: Sequence[str] = ("hf_roughness_local", "crackle_dog_rms", "crackle_isotropic"),
) -> np.ndarray:
    """Compute equal-weight composite crust z-score standardized strictly on Control B blank stats."""
    acc = np.zeros_like(next(iter(maps.values())), dtype=np.float64)
    used = 0
    for key in components:
        if key not in maps or key not in blank_ref:
            continue
        mu_b, sig_b = blank_ref[key]
        if sig_b <= 1e-12:
            continue
        acc += (maps[key].astype(np.float64) - float(mu_b)) / float(sig_b)
        used += 1
    if used == 0:
        return np.zeros_like(acc, dtype=np.float32)
    z = acc / math.sqrt(float(used))
    return np.where(interior_mask, z, 0.0).astype(np.float32)


def assert_no_unread_crust_coordinates(payload: Any) -> None:
    """Block any local paths or unread candidate coordinates in public crust preflight JSON."""
    p4.assert_no_local_paths(payload)
    p4.assert_no_unread_coordinates(
        payload,
        unread_section_names=("targeted_agraphon_scan", "unread_scans"),
    )

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in CRUST_FORBIDDEN_UNREAD_KEYS:
                    raise ValueError(f"Forbidden unread key {k!r} exposed at {path}.{k}")
                _walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _walk(item, f"{path}[{idx}]")

    if isinstance(payload, dict):
        for sec in ("targeted_agraphon_scan", "unread_scans"):
            if sec in payload:
                _walk(payload[sec], f"$.{sec}")


# ---------------------------------------------------------------------------
# 4. Paired 1.129 um tile evaluator (TPS r2 + 3D NCC lock + peak recentering)
# ---------------------------------------------------------------------------

def evaluate_1129_crust_tile(
    mesh_reader: p4.TifxyzPatchReader,
    r78: RemoteZarrV2,
    r1129: RemoteZarrV2,
    m_78_to_1129: np.ndarray,
    *,
    v0: int,
    v1: int,
    u0: int,
    u1: int,
    ref_level_78: int = 1,
    tgt_level_1129: int = 0,
    w78: tuple[float, float] = p4.DEFAULT_WINDOW_F32["78keV"],
    w1129: tuple[float, float] = p4.DEFAULT_WINDOW_F32["1129_78keV"],
    uv_spacing_um: float = 1.5,
    ncc_uv_downsample: int = 2,
    n_offsets_um: Sequence[float] | None = None,
    out_n_offsets_um: Sequence[float] | None = None,
    ghost_offsets_um: Sequence[float] | None = (-150.0, -110.0, 0.0, 110.0, 150.0),
    tps_on_ref: dict[str, Any] | None = None,
    search_half_width_um: float = 36.0,
    max_shift_um: float = 10.0,
    null_offset_um: float = 20.0,
    ghost_use_ref_78: bool = True,
) -> dict[str, Any]:
    """Stream one 1.129 um tile with TPS r2, 3D NCC lock, papyrus-peak recentering, and crust scoring."""
    ref_vox_um = p4.DEFAULT_VOXEL_UM["78keV"]
    tgt_vox_um = p4.DEFAULT_VOXEL_UM["1129_78keV"]
    ref_fwhm = ref_vox_um * (2 ** int(ref_level_78))
    tgt_fwhm = tgt_vox_um * (2 ** int(tgt_level_1129))
    psf_um = max(ref_fwhm, tgt_fwhm)

    n_arr = (
        np.asarray(n_offsets_um, dtype=np.float64)
        if n_offsets_um is not None
        else np.arange(-24, 25, dtype=np.float64) * 2.0  # -48..+48 um in 2.0 um steps
    )
    dn_um = float(np.median(np.diff(n_arr)))
    out_n_arr = (
        np.asarray(out_n_offsets_um, dtype=np.float64)
        if out_n_offsets_um is not None
        else np.arange(-12, 13, dtype=np.float64) * dn_um  # -24..+24 um recentered
    )

    scale_mesh = mesh_reader.meta.get("scale", [0.05, 0.05])
    mesh_du_um = (1.0 / float(scale_mesh[0])) * ref_vox_um
    mesh_dv_um = (1.0 / float(scale_mesh[1])) * ref_vox_um

    x, y, z, valid = mesh_reader.read_patch(v0, v1, u0, u1, v_stride=1)
    pts_ref, e_u_ref, e_v_ref, e_n_ref, uv_valid = p4.build_surface_frame(
        x,
        y,
        z,
        valid,
        grid_spacing_u_um=mesh_du_um,
        grid_spacing_v_um=mesh_dv_um,
        target_spacing_um=float(uv_spacing_um),
    )
    if not uv_valid.any():
        raise ValueError(f"Empty valid mesh patch for v=[{v0},{v1}), u=[{u0},{u1})")

    pts_tgt = p4.map_points_affine_and_tps(pts_ref, m_78_to_1129, tps_on_input=tps_on_ref)
    e_n_tgt = p4.apply_affine_vectors(e_n_ref, m_78_to_1129, normalize=True)

    # Sample reference 78 keV and native 1.129 um slabs on the shared physical frame.
    mu_ref, _, val_ref = p4.sample_surface_slab(
        r78,
        pts_ref,
        e_n_ref,
        uv_valid,
        n_arr,
        native_voxel_um=ref_vox_um,
        level=ref_level_78,
        f32_min=w78[0],
        f32_max=w78[1],
    )
    mu_tgt_raw, _, val_tgt_raw = p4.sample_surface_slab(
        r1129,
        pts_tgt,
        e_n_tgt,
        uv_valid,
        n_arr,
        native_voxel_um=tgt_vox_um,
        level=tgt_level_1129,
        f32_min=w1129[0],
        f32_max=w1129[1],
    )

    # PSF-match 1.129 um to 78 keV L1 (4.8 um) on a 2x-decimated UV grid for fast, exact 3D NCC lock.
    ds = max(1, int(ncc_uv_downsample))
    if mu_ref.shape[1] // ds < 12 or mu_ref.shape[2] // ds < 12:
        ds = 1
    mu_ref_ncc = mu_ref[:, ::ds, ::ds]
    val_ref_ncc = val_ref[:, ::ds, ::ds]
    mu_tgt_ncc = mu_tgt_raw[:, ::ds, ::ds]
    val_tgt_ncc = val_tgt_raw[:, ::ds, ::ds]
    uv_ncc_um = float(uv_spacing_um) * float(ds)

    if psf_um > tgt_fwhm + 1e-6:
        mu_tgt_ncc_psf = p4.psf_match_volume(
            mu_tgt_ncc,
            val_tgt_ncc,
            src_fwhm_um=tgt_fwhm,
            dst_fwhm_um=psf_um,
            spacing_nvu_um=(dn_um, uv_ncc_um, uv_ncc_um),
        )
    else:
        mu_tgt_ncc_psf = mu_tgt_ncc

    lock = p4.local_ncc_3d_lock(
        mu_ref_ncc,
        val_ref_ncc,
        mu_tgt_ncc_psf,
        val_tgt_ncc,
        spacing_nvu_um=(dn_um, uv_ncc_um, uv_ncc_um),
        max_shift_um=max_shift_um,
        null_offset_um=null_offset_um,
        min_ncc=0.10,
    )

    # Apply the locked physical shift to the FULL native-resolution 1.129 um slab.
    dn_vx = lock.shift_nvu_um[0] / dn_um
    dv_vx = lock.shift_nvu_um[1] / float(uv_spacing_um)
    du_vx = lock.shift_nvu_um[2] / float(uv_spacing_um)
    mu_tgt_aligned, val_tgt_aligned = p4.shift_moving_slab(mu_tgt_raw, val_tgt_raw, (dn_vx, dv_vx, du_vx))
    joint_valid = val_ref & val_tgt_aligned

    # Local papyrus-peak recentering along the normal (argmax_n mu_1129).
    recenter = recenter_papyrus_peak_slab(
        mu_tgt_aligned,
        joint_valid,
        n_arr,
        uv_spacing_um=float(uv_spacing_um),
        search_half_width_um=float(search_half_width_um),
        out_n_offsets_um=out_n_arr,
    )
    crust = compute_crust_crackle_maps(
        recenter["mu_recentered"],
        recenter["valid_recentered"],
        recenter["n_offsets_um"],
        uv_spacing_um=float(uv_spacing_um),
    )

    # Wrap-behind ghosting exclusion around the recentered sheet peak.
    # Default uses the already-cached 78 keV L1 (4.8 um, 614 um chunk) volume (0 extra CT bytes)
    # so the < 250 MB budget goes to native 1.129 um L0 surface coverage; set ghost_use_ref_78=False
    # to sample +/-150 um directly from 1.129 um L0.
    ghost_info: dict[str, Any] = {"checked": False, "is_ghost_dominated": False}
    if ghost_offsets_um is not None and len(ghost_offsets_um) > 0:
        g_offs = np.asarray(ghost_offsets_um, dtype=np.float64)
        peak_n_um = recenter["n_peak_um"][uv_valid].astype(np.float64)
        if ghost_use_ref_78:
            shift_ref_l0 = (peak_n_um / ref_vox_um)[:, None] * e_n_ref[uv_valid]
            mu_g, _, val_g = p4.sample_surface_slab(
                r78,
                pts_ref,
                e_n_ref,
                uv_valid,
                g_offs,
                native_voxel_um=ref_vox_um,
                level=ref_level_78,
                f32_min=w78[0],
                f32_max=w78[1],
                shift_xyz_l0=shift_ref_l0,
            )
            ghost_info = evaluate_crust_ghosting_exclusion(
                mu_g[:, ::ds, ::ds],
                val_g[:, ::ds, ::ds],
                g_offs,
                uv_spacing_um=uv_ncc_um,
            )
            ghost_info["ghost_source"] = "vol_78_l1_cached"
        else:
            e_u_tgt = p4.apply_affine_vectors(e_u_ref, m_78_to_1129, normalize=True)
            e_v_tgt = p4.apply_affine_vectors(e_v_ref, m_78_to_1129, normalize=True)
            dn_s_um, dv_s_um, du_s_um = lock.shift_nvu_um
            shift_tgt_l0 = (
                (du_s_um / tgt_vox_um) * e_u_tgt[uv_valid]
                + (dv_s_um / tgt_vox_um) * e_v_tgt[uv_valid]
                + ((dn_s_um + peak_n_um) / tgt_vox_um)[:, None] * e_n_tgt[uv_valid]
            )
            mu_g, _, val_g = p4.sample_surface_slab(
                r1129,
                pts_tgt,
                e_n_tgt,
                uv_valid,
                g_offs,
                native_voxel_um=tgt_vox_um,
                level=tgt_level_1129,
                f32_min=w1129[0],
                f32_max=w1129[1],
                shift_xyz_l0=shift_tgt_l0,
            )
            ghost_info = evaluate_crust_ghosting_exclusion(
                mu_g,
                val_g,
                g_offs,
                uv_spacing_um=float(uv_spacing_um),
            )
            ghost_info["ghost_source"] = "vol_1129_l0"

    z_ref_vals = pts_ref[uv_valid, 2]
    return {
        "tile_bounds_vu": [int(v0), int(v1), int(u0), int(u1)],
        "z78_range": [round(float(z_ref_vals.min()), 1), round(float(z_ref_vals.max()), 1)],
        "slab_shape_nvu": [int(s) for s in mu_tgt_aligned.shape],
        "recentered_shape_nvu": [int(s) for s in recenter["mu_recentered"].shape],
        "uv_spacing_um": float(uv_spacing_um),
        "lock": lock,
        "recenter": recenter,
        "crust": crust,
        "ghosting_check": ghost_info,
    }


# ---------------------------------------------------------------------------
# 5. End-to-end 1.129 um crust preflight + gated agraphon scan runner
# ---------------------------------------------------------------------------

def _summarize_unread_crust_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Produce strictly non-identifying aggregate public metrics from unread agraphon tiles."""
    evaluated = [r for r in records if r.get("evaluated")]
    locked = [r for r in records if r.get("ncc_lock", {}).get("passed")]
    skipped = [r for r in records if r.get("skipped_budget")]
    ncc_peaks = [float(r["ncc_lock"]["ncc_peak"]) for r in records if "ncc_lock" in r]
    null_margins = [float(r["ncc_lock"]["null_margin"]) for r in records if "ncc_lock" in r]
    ghost_dom = sum(1 for r in evaluated if r.get("ghosting_check", {}).get("is_ghost_dominated"))
    return {
        "tiles_planned": len(records),
        "tiles_evaluated": len(evaluated),
        "tiles_skipped_budget": len(skipped),
        "tiles_ncc_locked": len(locked),
        "ncc_peak_min": round(min(ncc_peaks), 6) if ncc_peaks else None,
        "ncc_peak_median": round(float(np.median(ncc_peaks)), 6) if ncc_peaks else None,
        "ncc_peak_max": round(max(ncc_peaks), 6) if ncc_peaks else None,
        "null_margin_min": round(min(null_margins), 6) if null_margins else None,
        "max_z_p99_9_pos": round(max((float(r["z_p99_9_pos"]) for r in evaluated), default=0.0), 4),
        "max_z_pos": round(max((float(r["z_max_pos"]) for r in evaluated), default=0.0), 4),
        "max_frac_ge_3sigma_pos": round(max((float(r["frac_ge_3sigma_pos"]) for r in evaluated), default=0.0), 6),
        "max_frac_ge_4sigma_pos": round(max((float(r["frac_ge_4sigma_pos"]) for r in evaluated), default=0.0), 6),
        "max_4sigma_component_span_mm": round(
            max((float(r["max_4sigma_component_span_mm"]) for r in evaluated), default=0.0), 3
        ),
        "ghost_dominated_count": int(ghost_dom),
        "any_stopping_rule_trigger": any(bool(r.get("exceeds_stopping_rule")) for r in evaluated),
    }


def run_crust_1129_preflight_and_scan(
    *,
    vol_78_url: str = p4.DEFAULT_URLS["vol_78"],
    vol_1129_url: str = p4.DEFAULT_URLS["vol_1129"],
    transform_1129_url: str = p4.DEFAULT_URLS["transform_1129_to_78_r2"],
    mesh_78_url: str = p4.DEFAULT_URLS["mesh_78"],
    ref_level_78: int = 1,
    tgt_level_1129: int = 0,
    uv_spacing_um: float = 1.5,
    cache_mb: float = 256.0,
    workers: int = 8,
    max_total_mb: float = 230.0,
    roc_auc_gate: float = 0.70,
    control_pairs: Sequence[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]] = DEFAULT_CONTROL_PAIRS,
    agraphon_tiles: Sequence[tuple[str, tuple[int, int, int, int]]] = DEFAULT_AGRAPH_SCAN_TILES,
    n_offsets_um: Sequence[float] | None = None,
    out_n_offsets_um: Sequence[float] | None = None,
    ghost_offsets_um: Sequence[float] | None = (-150.0, -110.0, 0.0, 110.0, 150.0),
    out_json: str | os.PathLike[str] | None = None,
    private_out_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run 1.129 um papyrus-peak recentering preflight on Control A vs B + gated agraphon scan."""
    if float(cache_mb) > 256.0:
        raise ValueError(f"cache_mb ({cache_mb}) exceeds strict 256.0 MiB RAM cap.")
    if float(max_total_mb) > 236.0:
        raise ValueError(f"max_total_mb ({max_total_mb} MiB) exceeds strict < 250 MB ceiling.")

    if private_out_dir is not None:
        priv_path = Path(private_out_dir)
        p4._assert_outside_git_repo(priv_path)
        priv_path.mkdir(parents=True, exist_ok=True)
    else:
        priv_path = None

    sess = requests.Session() if requests is not None else None
    try:
        w78_lo, w78_hi, _ = p4.load_volume_window_f32(
            vol_78_url, fallback=p4.DEFAULT_WINDOW_F32["78keV"], session=sess
        )
        w1129_lo, w1129_hi, _ = p4.load_volume_window_f32(
            vol_1129_url, fallback=p4.DEFAULT_WINDOW_F32["1129_78keV"], session=sess
        )
        t_bundle = p4.load_transform_bundle(transform_1129_url, session=sess)
        m_78_to_1129 = np.linalg.inv(t_bundle["matrix"])
        tps_inv = t_bundle.get("inverse_deformation")

        with (
            p4.TifxyzPatchReader(mesh_78_url) as mesh_78,
            RemoteZarrV2(vol_78_url, level=str(ref_level_78), cache_mb=cache_mb, workers=workers) as r78,
            RemoteZarrV2(vol_1129_url, level=str(tgt_level_1129), cache_mb=cache_mb, workers=workers) as r1129,
        ):
            def _bytes_used() -> int:
                return int(r78.bytes_fetched + r1129.bytes_fetched + mesh_78.bytes_fetched)

            def _remaining_mb() -> float:
                return float(max_total_mb) - (_bytes_used() / (1024.0 * 1024.0))

            # ------------------------------------------------------------------
            # Stage 1: Evaluate Control A & Control B pairs
            # ------------------------------------------------------------------
            pair_results: list[dict[str, Any]] = []
            tiles_a_raw: list[dict[str, Any]] = []
            tiles_b_raw: list[dict[str, Any]] = []

            for pair_name, box_a, box_b in control_pairs:
                res_b = evaluate_1129_crust_tile(
                    mesh_78,
                    r78,
                    r1129,
                    m_78_to_1129,
                    v0=box_b[0],
                    v1=box_b[1],
                    u0=box_b[2],
                    u1=box_b[3],
                    ref_level_78=ref_level_78,
                    tgt_level_1129=tgt_level_1129,
                    w78=(w78_lo, w78_hi),
                    w1129=(w1129_lo, w1129_hi),
                    uv_spacing_um=uv_spacing_um,
                    n_offsets_um=n_offsets_um,
                    out_n_offsets_um=out_n_offsets_um,
                    ghost_offsets_um=ghost_offsets_um,
                    tps_on_ref=tps_inv,
                )
                res_a = evaluate_1129_crust_tile(
                    mesh_78,
                    r78,
                    r1129,
                    m_78_to_1129,
                    v0=box_a[0],
                    v1=box_a[1],
                    u0=box_a[2],
                    u1=box_a[3],
                    ref_level_78=ref_level_78,
                    tgt_level_1129=tgt_level_1129,
                    w78=(w78_lo, w78_hi),
                    w1129=(w1129_lo, w1129_hi),
                    uv_spacing_um=uv_spacing_um,
                    n_offsets_um=n_offsets_um,
                    out_n_offsets_um=out_n_offsets_um,
                    ghost_offsets_um=ghost_offsets_um,
                    tps_on_ref=tps_inv,
                )
                tiles_a_raw.append(res_a)
                tiles_b_raw.append(res_b)
                pair_results.append({"pair_name": pair_name, "tile_a": res_a, "tile_b": res_b})

            ctrl_bytes = _bytes_used()
            ctrl_chunks_78 = int(r78.chunks_fetched)
            ctrl_chunks_1129 = int(r1129.chunks_fetched)

            # Fit blank reference mean/std strictly on pooled Control B pixels.
            base_metrics = (
                "surface_mu_recentered",
                "tophat_recentered",
                "hf_roughness_local",
                "hf_roughness_excess",
                "crackle_dog_rms",
                "crackle_isotropic",
            )
            pooled_b: dict[str, np.ndarray] = {}
            pooled_a: dict[str, np.ndarray] = {}
            blank_ref_params: dict[str, tuple[float, float]] = {}

            for mkey in base_metrics:
                mask_key = "valid_2d" if mkey in ("surface_mu_recentered", "tophat_recentered") else "interior_2d"
                vals_b = np.concatenate([t["crust"]["maps"][mkey][t["crust"][mask_key]] for t in tiles_b_raw])
                vals_a = np.concatenate([t["crust"]["maps"][mkey][t["crust"][mask_key]] for t in tiles_a_raw])
                pooled_b[mkey] = vals_b
                pooled_a[mkey] = vals_a
                if vals_b.size >= 2:
                    blank_ref_params[mkey] = (float(vals_b.mean()), float(vals_b.std(ddof=1)))
                else:
                    blank_ref_params[mkey] = (0.0, 0.0)

            # Build Control-B standardized composite crust z-score map on every control tile.
            comp_vals_a: list[np.ndarray] = []
            comp_vals_b: list[np.ndarray] = []
            for t in tiles_b_raw:
                cz = build_composite_crust_z(t["crust"]["maps"], t["crust"]["interior_2d"], blank_ref_params)
                t["crust"]["maps"]["crust_composite_z"] = cz
                comp_vals_b.append(cz[t["crust"]["interior_2d"]])
            for t in tiles_a_raw:
                cz = build_composite_crust_z(t["crust"]["maps"], t["crust"]["interior_2d"], blank_ref_params)
                t["crust"]["maps"]["crust_composite_z"] = cz
                comp_vals_a.append(cz[t["crust"]["interior_2d"]])
            pooled_a["crust_composite_z"] = np.concatenate(comp_vals_a)
            pooled_b["crust_composite_z"] = np.concatenate(comp_vals_b)

            # Compute pooled Control A vs Control B ROC-AUC, Cohen's d, blank noise floor & 3s/4s FPR.
            pooled_separation: dict[str, dict[str, float]] = {}
            blank_noise_stats: dict[str, dict[str, float | int]] = {}
            for mkey in (*base_metrics, "crust_composite_z"):
                pooled_separation[mkey] = p4.compute_control_separation(pooled_a[mkey], pooled_b[mkey])
                b_arr = pooled_b[mkey]
                blank_noise_stats[mkey] = p4.compute_blank_fpr_stats(b_arr, np.ones_like(b_arr, dtype=bool))

            # Per-pair diagnostics + private control maps.
            public_control_pairs: list[dict[str, Any]] = []
            for entry in pair_results:
                pname = entry["pair_name"]
                ta = entry["tile_a"]
                tb = entry["tile_b"]
                pair_sep: dict[str, dict[str, float]] = {}
                for mkey in GATED_CRUST_METRICS:
                    mk = "valid_2d" if mkey == "tophat_recentered" else "interior_2d"
                    pair_sep[mkey] = p4.compute_control_separation(
                        ta["crust"]["maps"][mkey][ta["crust"][mk]],
                        tb["crust"]["maps"][mkey][tb["crust"][mk]],
                    )

                if priv_path is not None:
                    for tag, t in (("ctrl_a", ta), ("ctrl_b", tb)):
                        p4._save_private_heatmap_png(
                            t["recenter"]["n_peak_um"],
                            t["recenter"]["peak_valid"],
                            priv_path / f"{pname}_{tag}_n_peak_um.png",
                            -36.0,
                            36.0,
                        )
                        p4._save_private_heatmap_png(
                            t["crust"]["maps"]["crust_composite_z"],
                            t["crust"]["interior_2d"],
                            priv_path / f"{pname}_{tag}_crust_composite_z.png",
                            -4.0,
                            4.0,
                        )
                        np.savez_compressed(
                            priv_path / f"{pname}_{tag}.npz",
                            n_peak_um=t["recenter"]["n_peak_um"],
                            peak_valid=t["recenter"]["peak_valid"],
                            valid_2d=t["crust"]["valid_2d"],
                            interior_2d=t["crust"]["interior_2d"],
                            **t["crust"]["maps"],
                        )

                public_control_pairs.append(
                    {
                        "pair_name": pname,
                        "control_a": {
                            "tile_bounds_vu": ta["tile_bounds_vu"],
                            "z78_range": ta["z78_range"],
                            "slab_shape_nvu": ta["slab_shape_nvu"],
                            "ncc_lock": ta["lock"].to_dict(),
                            "recentering": ta["recenter"]["stats"],
                            "crust_summary": ta["crust"]["summary"],
                            "ghosting_check": ta["ghosting_check"],
                        },
                        "control_b": {
                            "tile_bounds_vu": tb["tile_bounds_vu"],
                            "z78_range": tb["z78_range"],
                            "slab_shape_nvu": tb["slab_shape_nvu"],
                            "ncc_lock": tb["lock"].to_dict(),
                            "recentering": tb["recenter"]["stats"],
                            "crust_summary": tb["crust"]["summary"],
                            "ghosting_check": tb["ghosting_check"],
                        },
                        "pair_separation_a_vs_b": pair_sep,
                    }
                )

            ncc_passed_all = all(bool(t["lock"].passed) for t in (*tiles_a_raw, *tiles_b_raw))
            best_metric = max(GATED_CRUST_METRICS, key=lambda k: float(pooled_separation[k]["roc_auc"]))
            best_auc = float(pooled_separation[best_metric]["roc_auc"])
            best_d = float(pooled_separation[best_metric]["cohens_d"])
            gate_passed = bool(ncc_passed_all and (best_auc >= float(roc_auc_gate)) and (best_d > 0.0))

            # ------------------------------------------------------------------
            # Stage 2: Gated targeted 1.129 um crust scan across agraphon tiles
            # ------------------------------------------------------------------
            scan_records: list[dict[str, Any]] = []
            scan_bytes_start = _bytes_used()
            scan_chunks_78_start = int(r78.chunks_fetched)
            scan_chunks_1129_start = int(r1129.chunks_fetched)

            if gate_passed and len(agraphon_tiles) > 0:
                score_key = best_metric
                score_mask_key = "valid_2d" if score_key == "tophat_recentered" else "interior_2d"
                b_stat = blank_noise_stats[score_key]
                mu_b = float(b_stat["mean_blank"])
                sig_b = max(float(b_stat["sigma_blank"]), 1e-6)
                p999_b = float(b_stat.get("p99_9_pos_z", 3.5))

                for tile_id, (v0, v1, u0, u1) in agraphon_tiles:
                    est_mb = p4._estimate_extra_chunks_mb(
                        mesh_78,
                        [
                            (r78, ref_level_78, None),
                            (
                                r1129,
                                tgt_level_1129,
                                lambda p: p4.map_points_affine_and_tps(p, m_78_to_1129, tps_on_input=tps_inv),
                            ),
                        ],
                        v0,
                        v1,
                        u0,
                        u1,
                        v_stride=1,
                        pad_vx_lvl=6,
                    )
                    # Reserve conservative headroom so total stays strictly < 250 MB.
                    if est_mb + 6.0 > _remaining_mb():
                        scan_records.append(
                            {
                                "tile_id": tile_id,
                                "skipped_budget": True,
                                "evaluated": False,
                                "est_extra_mb": round(est_mb, 2),
                            }
                        )
                        continue

                    ts = evaluate_1129_crust_tile(
                        mesh_78,
                        r78,
                        r1129,
                        m_78_to_1129,
                        v0=v0,
                        v1=v1,
                        u0=u0,
                        u1=u1,
                        ref_level_78=ref_level_78,
                        tgt_level_1129=tgt_level_1129,
                        w78=(w78_lo, w78_hi),
                        w1129=(w1129_lo, w1129_hi),
                        uv_spacing_um=uv_spacing_um,
                        n_offsets_um=n_offsets_um,
                        out_n_offsets_um=out_n_offsets_um,
                        ghost_offsets_um=ghost_offsets_um,
                        tps_on_ref=tps_inv,
                    )
                    lock_s: p4.NccLockResult = ts["lock"]
                    if not lock_s.passed:
                        scan_records.append(
                            {
                                "tile_id": tile_id,
                                "tile_bounds_vu": ts["tile_bounds_vu"],
                                "z78_range": ts["z78_range"],
                                "ncc_lock": lock_s.to_dict(),
                                "evaluated": False,
                            }
                        )
                        continue

                    cz_s = build_composite_crust_z(ts["crust"]["maps"], ts["crust"]["interior_2d"], blank_ref_params)
                    ts["crust"]["maps"]["crust_composite_z"] = cz_s

                    raw_score = ts["crust"]["maps"][score_key]
                    vmask = ts["crust"][score_mask_key]
                    z_map = np.where(vmask, (raw_score.astype(np.float64) - mu_b) / sig_b, 0.0).astype(np.float32)
                    vz = z_map[vmask]
                    hi_mask = vmask & (z_map >= max(4.0, p999_b))
                    lbl, n_comp = ndi.label(hi_mask)
                    max_comp_px = 0
                    max_comp_span_mm = 0.0
                    if n_comp > 0:
                        sizes = np.bincount(lbl.ravel())[1:]
                        top_cid = int(np.argmax(sizes)) + 1
                        max_comp_px = int(sizes[top_cid - 1])
                        ys, xs = np.nonzero(lbl == top_cid)
                        max_comp_span_mm = float(
                            math.hypot(
                                (ys.max() - ys.min() + 1) * uv_spacing_um,
                                (xs.max() - xs.min() + 1) * uv_spacing_um,
                            )
                            / 1000.0
                        )

                    if priv_path is not None:
                        np.savez_compressed(
                            priv_path / f"scan_{tile_id}.npz",
                            z_map=z_map,
                            n_peak_um=ts["recenter"]["n_peak_um"],
                            peak_valid=ts["recenter"]["peak_valid"],
                            valid_2d=ts["crust"]["valid_2d"],
                            interior_2d=ts["crust"]["interior_2d"],
                            **ts["crust"]["maps"],
                        )
                        p4._save_private_heatmap_png(z_map, vmask, priv_path / f"scan_{tile_id}_zmap.png", -4.5, 4.5)

                    scan_records.append(
                        {
                            "tile_id": tile_id,
                            "tile_bounds_vu": ts["tile_bounds_vu"],
                            "z78_range": ts["z78_range"],
                            "ncc_lock": lock_s.to_dict(),
                            "recentering": ts["recenter"]["stats"],
                            "crust_summary": ts["crust"]["summary"],
                            "ghosting_check": ts["ghosting_check"],
                            "evaluated": True,
                            "scored_metric": score_key,
                            "valid_pixels": int(vmask.sum()),
                            "z_mean": round(float(vz.mean()), 4) if vz.size else 0.0,
                            "z_p99_9_pos": round(float(np.percentile(vz, 99.9)), 4) if vz.size else 0.0,
                            "z_max_pos": round(float(vz.max()), 4) if vz.size else 0.0,
                            "frac_ge_3sigma_pos": round(float(np.mean(vz >= 3.0)), 6) if vz.size else 0.0,
                            "frac_ge_4sigma_pos": round(float(np.mean(vz >= 4.0)), 6) if vz.size else 0.0,
                            "max_4sigma_component_px": max_comp_px,
                            "max_4sigma_component_span_mm": round(max_comp_span_mm, 3),
                            "exceeds_stopping_rule": bool(
                                (max_comp_span_mm >= 0.08)
                                and (float(np.mean(vz >= 4.0)) >= 0.01 if vz.size else False)
                                and (not ts["ghosting_check"].get("is_ghost_dominated", False))
                            ),
                        }
                    )

            total_bytes = _bytes_used()
            scan_bytes = total_bytes - scan_bytes_start
            total_chunks_78 = int(r78.chunks_fetched)
            total_chunks_1129 = int(r1129.chunks_fetched)
            total_chunks = total_chunks_78 + total_chunks_1129

        if gate_passed:
            targeted_scan_public: dict[str, Any] = {
                "executed": True,
                "scored_metric": best_metric,
                "gate_threshold_roc_auc": float(roc_auc_gate),
                **_summarize_unread_crust_records(scan_records),
            }
        else:
            targeted_scan_public = {
                "executed": False,
                "skipped_reason": (
                    "control_ncc_failed" if not ncc_passed_all else "control_gate_below_roc_auc_threshold"
                ),
                "gate_threshold_roc_auc": float(roc_auc_gate),
                "best_control_metric": best_metric,
                "best_control_roc_auc": round(best_auc, 6),
                "best_control_cohens_d": round(best_d, 6),
                "tiles_planned": len(agraphon_tiles),
                "tiles_evaluated": 0,
                "any_stopping_rule_trigger": False,
            }

        public_result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "pipeline": "gemini_lane/paris4_crust_1129.py",
            "phase": "paris4_1129_recentered_crust_preflight",
            "mode": "cpu_only_streaming",
            "sources": {
                "vol_78": _sanitize_source_label(vol_78_url),
                "vol_1129": _sanitize_source_label(vol_1129_url),
                "transform_1129_to_78_r2": _sanitize_source_label(transform_1129_url),
                "mesh_78": _sanitize_source_label(mesh_78_url),
            },
            "radiometry": {
                "window_f32_78keV": [w78_lo, w78_hi],
                "window_f32_1129_78keV": [w1129_lo, w1129_hi],
                "ref_level_78": int(ref_level_78),
                "tgt_level_1129": int(tgt_level_1129),
                "native_voxel_um": round(p4.DEFAULT_VOXEL_UM["1129_78keV"] * (2 ** int(tgt_level_1129)), 3),
                "ref_78_voxel_um": round(p4.DEFAULT_VOXEL_UM["78keV"] * (2 ** int(ref_level_78)), 3),
                "uv_spacing_um": round(float(uv_spacing_um), 3),
                "tps_deformation_applied": bool(tps_inv is not None),
            },
            "ncc_passed_all_controls": ncc_passed_all,
            "gate_threshold_roc_auc": float(roc_auc_gate),
            "best_crust_metric": best_metric,
            "best_crust_roc_auc": round(best_auc, 6),
            "best_crust_cohens_d": round(best_d, 6),
            "control_gate_passed": gate_passed,
            "pooled_control_a_vs_b": pooled_separation,
            "blank_noise_floor_control_b": blank_noise_stats,
            "control_pairs": public_control_pairs,
            "targeted_agraphon_scan": targeted_scan_public,
            "bytes_fetched": int(total_bytes),
            "chunks_fetched": int(total_chunks),
            "total_mib": round(total_bytes / (1024.0 * 1024.0), 3),
            "total_mb_decimal": round(total_bytes / 1_000_000.0, 3),
            "within_250mb_budget": bool(total_bytes < 250_000_000),
            "cache_mb": float(cache_mb),
            "streaming_breakdown": {
                "controls_bytes": int(ctrl_bytes),
                "controls_mib": round(ctrl_bytes / (1024.0 * 1024.0), 3),
                "controls_chunks_78": ctrl_chunks_78,
                "controls_chunks_1129": ctrl_chunks_1129,
                "scan_bytes": int(scan_bytes),
                "scan_mib": round(scan_bytes / (1024.0 * 1024.0), 3),
                "scan_chunks_78": total_chunks_78 - scan_chunks_78_start,
                "scan_chunks_1129": total_chunks_1129 - scan_chunks_1129_start,
            },
            "any_stopping_rule_trigger": bool(targeted_scan_public.get("any_stopping_rule_trigger", False)),
        }

        assert_no_unread_crust_coordinates(public_result)
        if out_json is not None:
            out_p = Path(out_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(public_result, indent=2) + "\n", encoding="utf-8")

        if priv_path is not None:
            priv_summary = {
                "schema_version": "1.0.0-private",
                "control_gate_passed": gate_passed,
                "best_crust_metric": best_metric,
                "best_crust_roc_auc": round(best_auc, 6),
                "best_crust_cohens_d": round(best_d, 6),
                "pooled_control_a_vs_b": pooled_separation,
                "blank_noise_floor_control_b": blank_noise_stats,
                "control_pairs": public_control_pairs,
                "targeted_agraphon_scan_records": scan_records,
                "streaming_breakdown": public_result["streaming_breakdown"],
                "total_mib": public_result["total_mib"],
                "total_mb_decimal": public_result["total_mb_decimal"],
            }
            (priv_path / "crust_1129_private_summary.json").write_text(
                json.dumps(priv_summary, indent=2) + "\n", encoding="utf-8"
            )

        return public_result
    finally:
        if sess is not None:
            sess.close()


# ---------------------------------------------------------------------------
# 6. CLI entrypoint
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CPU-only (<250 MB) PHerc. Paris 4 1.129 um papyrus-peak recentered crust preflight & gated scan."
    )
    ap.add_argument("--vol-78", default=p4.DEFAULT_URLS["vol_78"], help="78 keV OME-Zarr v2 URL or local dir")
    ap.add_argument("--vol-1129", default=p4.DEFAULT_URLS["vol_1129"], help="1.129 um 78 keV OME-Zarr v2 URL or local dir")
    ap.add_argument(
        "--transform-1129",
        default=p4.DEFAULT_URLS["transform_1129_to_78_r2"],
        help="1.129->78 r2 transform.json",
    )
    ap.add_argument("--mesh-78", default=p4.DEFAULT_URLS["mesh_78"], help="w010-027 tifxyz URL or local dir")
    ap.add_argument("--uv-spacing-um", type=float, default=1.5, help="In-plane UV sampling spacing in um (default: 1.5)")
    ap.add_argument("--cache-mb", type=float, default=256.0, help="In-memory LRU cache cap in MiB (<= 256)")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent chunk-fetch workers (default: 8)")
    ap.add_argument("--max-total-mb", type=float, default=230.0, help="Hard streaming cap in MiB (<= 236, < 250 MB)")
    ap.add_argument("--roc-auc-gate", type=float, default=0.70, help="Minimum Control A vs B ROC-AUC to unlock scan")
    ap.add_argument("--out-json", required=True, help="Output path for sanitized public JSON receipt")
    ap.add_argument(
        "--private-out-dir",
        default=None,
        help="Private output directory OUTSIDE git for maps/PNGs/coordinates",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_crust_1129_preflight_and_scan(
        vol_78_url=args.vol_78,
        vol_1129_url=args.vol_1129,
        transform_1129_url=args.transform_1129,
        mesh_78_url=args.mesh_78,
        uv_spacing_um=args.uv_spacing_um,
        cache_mb=args.cache_mb,
        workers=args.workers,
        max_total_mb=args.max_total_mb,
        roc_auc_gate=args.roc_auc_gate,
        out_json=args.out_json,
        private_out_dir=args.private_out_dir,
    )
    print(
        json.dumps(
            {
                "ncc_passed_all_controls": res["ncc_passed_all_controls"],
                "best_crust_metric": res["best_crust_metric"],
                "best_crust_roc_auc": res["best_crust_roc_auc"],
                "best_crust_cohens_d": res["best_crust_cohens_d"],
                "control_gate_passed": res["control_gate_passed"],
                "scan_executed": res["targeted_agraphon_scan"]["executed"],
                "scan_tiles_evaluated": res["targeted_agraphon_scan"]["tiles_evaluated"],
                "total_mib": res["total_mib"],
                "total_mb_decimal": res["total_mb_decimal"],
                "within_250mb_budget": res["within_250mb_budget"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
