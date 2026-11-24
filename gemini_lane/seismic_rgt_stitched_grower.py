# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T11-RGT-Scale (Q46): Stitched multi-tile 3D seismic RGT / Wheeler multi-sheet grower.

Scales the CPU-only seismic Relative Geologic Time (RGT-Est) and Wheeler multi-sheet
unwrapper (``gemini_lane/seismic_rgt_unwrapper.py``) from a single ``64x96x96`` micro-box
to stitched ``2x2`` overlapping 3D tile blocks (e.g. ``64x192x192`` or ``64x224x224`` slabs
partitioned into ``2x2`` overlapping tiles with ``32 vx`` lateral overlap):

1. :func:`stitch_overlapping_rgt_tiles_3d` - global dominant normal-axis estimation across
   the full slab, per-tile ``solve_dip_guided_rgt_3d`` in traversal order
   ``(0,0) -> (0,1) -> (1,0) -> (1,1)``, robust monotone affine phase gauge alignment
   ``Phi_aligned = a * Phi_tile + b`` (``a in [0.8, 1.25]`` via median slope + Huber IRLS
   on the ``32 vx`` overlap volume with already-aligned neighbors), raised-cosine (Hann)
   blending across overlap zones, and strict monotonic projection (``eps_phi = 1e-3``)
   along ``normal_axis`` so ``crossing_fraction_rgt == 0.0`` holds mathematically.
2. :func:`extract_and_snap_stitched_rgt_sheets` - simultaneous ordered iso-phase sheet
   extraction via ``wheeler_flatten_and_extract_sheets``, optional 2-pass ``m7`` normal
   snapping (``SSR.snap_tifxyz_to_m7``) followed by order-preserving inter-sheet clamping
   (``h_{m+1} > h_m + 0.1`` on jointly valid columns), CT surface-stack rendering, and
   ``_onsheet_f_parallel`` (``tile=32, step=16``) head-to-head against the single-sheet
   ``grow_m7_patch.extract_height_field`` baseline on the exact same slab.
3. :func:`export_private_rgt_tifxyz_meshes` - repository-boundary guard via
   ``screen_scroll_rtx4060.ensure_outside_git_repo``, box-to-global coordinate translation
   by ``box_origin_zyx[::-1]`` preserving ``-1.0`` sentinels, and standard ``tifxyz`` export.
4. :func:`run_live_stitched_preflight` - live CPU preflight under one shared
   ``ByteBudget(120 * 1024 * 1024)`` across ``PHerc0306B``, ``PHerc1545``, and ``PHerc0846A``
   with strict unread-scroll privacy enforcement.

Strictly CPU-only: zero ``torch`` or GPU imports.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grow_m7_patch as GM  # noqa: E402
from grow_m7_patch import BudgetExceeded, BudgetedRemoteZarrV2, ByteBudget, SCROLL_SPECS  # noqa: E402,F401
import screen_scroll_rtx4060 as SCR  # noqa: E402
import seismic_rgt_unwrapper as RGT  # noqa: E402
import stream_snap_render as SSR  # noqa: E402

S3_BUCKET = GM.S3_BUCKET
GLOBAL_HTTP_CAP_BYTES = 120 * 1024 * 1024
STITCHED_BOX_SHAPE_ZYX = (64, 192, 192)
DEFAULT_TILE_LATERAL = 112
DEFAULT_OVERLAP_LATERAL = 32
UNREAD_SCROLLS = RGT.UNREAD_SCROLLS

PUBLIC_STITCHED_RGT_METRIC_KEYS = RGT.PUBLIC_RGT_METRIC_KEYS | frozenset({
    "seam_phase_rmse_rad",
    "max_f_parallel_rgt",
    "mean_on_sheet_raw_rgt",
    "mean_on_sheet_after_snap_rgt",
    "per_sheet_on_sheet_after_snap_rgt",
    "snap_passes",
    "tile_grid",
    "tile_lateral_vx",
    "overlap_lateral_vx",
})

# Small bounded cache mapping stitched phi signature -> stitch context so callers that pass
# ``rgt_phi`` ndarray directly into ``extract_and_snap_stitched_rgt_sheets`` still retain
# ``seam_phase_rmse_rad``, ``planarity``, ``k_info``, and tile solver telemetry.
_STITCH_CONTEXT_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_MAX_CONTEXT_CACHE = 8


def _phi_cache_key(phi: np.ndarray) -> tuple[Any, ...]:
    arr = np.asarray(phi)
    if arr.size == 0:
        return (arr.shape, 0.0, 0.0, 0.0)
    return (
        tuple(int(s) for s in arr.shape),
        round(float(arr.flat[0]), 6),
        round(float(arr.flat[arr.size // 2]), 6),
        round(float(arr.flat[-1]), 6),
    )


def _store_stitch_context(phi: np.ndarray, ctx: dict[str, Any]) -> None:
    key = _phi_cache_key(phi)
    _STITCH_CONTEXT_CACHE[key] = ctx
    _STITCH_CONTEXT_CACHE.move_to_end(key)
    while len(_STITCH_CONTEXT_CACHE) > _MAX_CONTEXT_CACHE:
        _STITCH_CONTEXT_CACHE.popitem(last=False)


def _lookup_stitch_context(phi: np.ndarray) -> dict[str, Any] | None:
    return _STITCH_CONTEXT_CACHE.get(_phi_cache_key(phi))


# ---------------------------------------------------------------------------
# 1. Overlapping 2x2 tile RGT solver and phase-continuity gauge stitcher
# ---------------------------------------------------------------------------

def _resolve_two_tile_axis(length: int, tile_lateral: int, overlap_lateral: int) -> tuple[int, int]:
    """Return ``(eff_tile_len, eff_overlap_len)`` for a 2-tile partition of ``length``."""
    L = int(length)
    req_tile = int(tile_lateral)
    req_ov = int(overlap_lateral)
    if req_ov < 4 or req_ov >= L:
        raise ValueError(f"overlap_lateral={req_ov} must satisfy 4 <= overlap < length={L}")
    if 2 * req_tile - req_ov == L:
        return req_tile, req_ov
    eff_tile = (L + req_ov + 1) // 2
    eff_ov = 2 * eff_tile - L
    return eff_tile, eff_ov


def _raised_cosine_ramp(n: int) -> np.ndarray:
    """Hann / raised-cosine blend weights in ``(0, 1)`` across an ``n``-voxel overlap."""
    t = (np.arange(int(n), dtype=np.float64) + 0.5) / float(max(1, int(n)))
    return 0.5 * (1.0 - np.cos(math.pi * t))


def _fit_monotone_affine_gauge(
    phi_src: np.ndarray,
    phi_ref: np.ndarray,
    weights: np.ndarray,
    *,
    a_bounds: tuple[float, float] = (0.8, 1.25),
    n_iter: int = 8,
) -> tuple[float, float, float]:
    """Fit ``phi_aligned = a * phi_src + b`` with ``a in a_bounds`` via median slope + Huber IRLS.

    Returns ``(a, b, weighted_rmse_rad)``.
    """
    src = np.asarray(phi_src, dtype=np.float64).ravel()
    ref = np.asarray(phi_ref, dtype=np.float64).ravel()
    w = np.clip(np.asarray(weights, dtype=np.float64).ravel(), 0.0, None)
    good = np.isfinite(src) & np.isfinite(ref) & (w > 1e-9)
    if not np.any(good):
        return 1.0, 0.0, 0.0
    src_g = src[good]
    ref_g = ref[good]
    w_g = w[good]
    w_g = w_g / float(w_g.sum())

    # Initial median/IQR slope estimate.
    q_src = float(np.percentile(src_g, 80) - np.percentile(src_g, 20))
    q_ref = float(np.percentile(ref_g, 80) - np.percentile(ref_g, 20))
    a_lo, a_hi = float(a_bounds[0]), float(a_bounds[1])
    a = float(np.clip(q_ref / max(q_src, 1e-6), a_lo, a_hi))
    b = float(np.median(ref_g - a * src_g))

    for _ in range(int(n_iter)):
        resid = ref_g - (a * src_g + b)
        med_r = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med_r)))
        sigma = 1.4826 * mad + 1e-4
        delta = 1.345 * sigma
        abs_r = np.abs(resid - med_r)
        huber = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-12))
        hw = w_g * huber
        w_sum = float(hw.sum())
        if w_sum <= 1e-12:
            break
        mx = float(np.dot(hw, src_g) / w_sum)
        my = float(np.dot(hw, ref_g) / w_sum)
        dx = src_g - mx
        dy = ref_g - my
        denom = float(np.dot(hw, dx * dx))
        if denom > 1e-9:
            a = float(np.clip(np.dot(hw, dx * dy) / denom, a_lo, a_hi))
        b = float(my - a * mx)

    final_resid = ref_g - (a * src_g + b)
    rmse = float(np.sqrt(np.dot(w_g, final_resid * final_resid)))
    return a, b, rmse


def stitch_overlapping_rgt_tiles_3d(
    vol_zyx: np.ndarray,
    m7_zyx: np.ndarray | None = None,
    *,
    tile_lateral: int = DEFAULT_TILE_LATERAL,
    overlap_lateral: int = DEFAULT_OVERLAP_LATERAL,
    min_spacing_vx: float = 4.0,
    max_spacing_vx: float = 18.0,
    solve_kw: dict[str, Any] | None = None,
    eps_phi: float = 1e-3,
) -> dict[str, Any]:
    """Solve and stitch a ``2x2`` grid of overlapping 3D RGT tiles into one monotonic phase field.

    The two tiled lateral axes are the two largest axes of ``vol_zyx`` (axes 1 and 2, ``(y, x)``,
    for standard ``(D, Y, X)`` slabs such as ``64x192x192``, ``64x224x224``, or ``48x160x160``).
    All four tiles share the global ``normal_axis`` estimated across the full slab.
    """
    ct = np.asarray(vol_zyx, dtype=np.float32)
    if ct.ndim != 3:
        raise ValueError("vol_zyx must be 3D")
    m7 = None if m7_zyx is None else np.asarray(m7_zyx, dtype=np.float32)
    if m7 is not None and m7.shape != ct.shape:
        raise ValueError(f"m7_zyx shape {m7.shape} must match vol_zyx shape {ct.shape}")

    guide = RGT._zscore(ct)
    if m7 is not None and float(np.std(m7)) > 0:
        guide = guide + RGT._zscore(ndi.gaussian_filter(m7, 0.7))

    # Global dominant normal axis, structure tensor, and local wrap wavenumber on the slab.
    normals, planarity, normal_axis = RGT.estimate_structure_tensor_dip_3d(guide)
    k_zyx, k_info = RGT.estimate_local_wrap_wavenumber_3d(
        guide, normal_axis, min_spacing_vx=min_spacing_vx, max_spacing_vx=max_spacing_vx
    )
    crest_mask = RGT.ridge_crest_mask(guide, normal_axis)

    d0, y_len, x_len = (int(s) for s in ct.shape)
    ty, ov_y = _resolve_two_tile_axis(y_len, tile_lateral, overlap_lateral)
    tx, ov_x = _resolve_two_tile_axis(x_len, tile_lateral, overlap_lateral)

    y_bounds = [(0, ty), (y_len - ty, y_len)]
    x_bounds = [(0, tx), (x_len - tx, x_len)]
    y_ov0, y_ov1 = y_len - ty, ty
    x_ov0, x_ov1 = x_len - tx, tx

    # Solve each tile in traversal order (0,0) -> (0,1) -> (1,0) -> (1,1).
    traversal = ((0, 0), (0, 1), (1, 0), (1, 1))
    raw_tiles: dict[tuple[int, int], np.ndarray] = {}
    tile_infos: dict[str, dict[str, Any]] = {}
    for r, c in traversal:
        y0, y1 = y_bounds[r]
        x0, x1 = x_bounds[c]
        sl = (slice(None), slice(y0, y1), slice(x0, x1))
        solved = RGT.solve_dip_guided_rgt_3d(
            guide[sl],
            normals[sl],
            planarity[sl],
            k_zyx[sl],
            normal_axis=normal_axis,
            eps_phi=eps_phi,
            **(solve_kw or {}),
        )
        raw_tiles[(r, c)] = solved["phi"]
        tile_infos[f"{r}_{c}"] = solved["info"]

    # Seam weighting: favor high-planarity ridge voxels and the interior of the 32-vx overlap
    # where neither tile is on its outer cut boundary.
    alpha_y = _raised_cosine_ramp(ov_y)
    alpha_x = _raised_cosine_ramp(ov_x)
    seam_wy = np.maximum(4.0 * alpha_y * (1.0 - alpha_y), 1e-3)
    seam_wx = np.maximum(4.0 * alpha_x * (1.0 - alpha_x), 1e-3)

    # Interior depth weighting along normal_axis to avoid empty noise faces biasing gauge fit.
    depth_len = int(ct.shape[normal_axis])
    z_idx = np.arange(depth_len, dtype=np.float64)
    depth_env = np.sin(math.pi * (z_idx + 0.5) / float(depth_len)) ** 2
    depth_w_shape = [1, 1, 1]
    depth_w_shape[normal_axis] = depth_len
    depth_w = depth_env.reshape(depth_w_shape)
    vox_weight = (np.clip(planarity, 0.1, 1.0) ** 3) * (1.0 + 1.5 * crest_mask) * depth_w

    aligned_tiles: dict[tuple[int, int], np.ndarray] = {}
    tile_gauges: dict[str, dict[str, float]] = {}

    # (0, 0) root tile
    aligned_tiles[(0, 0)] = raw_tiles[(0, 0)].copy()
    tile_gauges["0_0"] = {"a": 1.0, "b": 0.0, "overlap_rmse_rad": 0.0}

    # (0, 1) aligned against (0, 0) on x-overlap [x_ov0, x_ov1)
    ref_01 = aligned_tiles[(0, 0)][:, :, x_ov0:tx]
    src_01 = raw_tiles[(0, 1)][:, :, :ov_x]
    w_01 = vox_weight[:, :ty, x_ov0:x_ov1] * seam_wx[None, None, :]
    a_01, b_01, rmse_01 = _fit_monotone_affine_gauge(src_01, ref_01, w_01)
    aligned_tiles[(0, 1)] = a_01 * raw_tiles[(0, 1)] + b_01
    tile_gauges["0_1"] = {"a": a_01, "b": b_01, "overlap_rmse_rad": rmse_01}

    # Blend Row 0 across [0, ty) x [0, x_len)
    row0 = np.empty((d0, ty, x_len), dtype=np.float64)
    row0[:, :, :x_ov0] = aligned_tiles[(0, 0)][:, :, :x_ov0]
    row0[:, :, x_ov0:x_ov1] = (
        (1.0 - alpha_x[None, None, :]) * aligned_tiles[(0, 0)][:, :, x_ov0:tx]
        + alpha_x[None, None, :] * aligned_tiles[(0, 1)][:, :, :ov_x]
    )
    row0[:, :, x_ov1:] = aligned_tiles[(0, 1)][:, :, ov_x:]

    # (1, 0) aligned against Row 0 on y-overlap [y_ov0, y_ov1) x [0, tx)
    ref_10 = row0[:, y_ov0:ty, :tx]
    src_10 = raw_tiles[(1, 0)][:, :ov_y, :]
    w_10 = vox_weight[:, y_ov0:y_ov1, :tx] * seam_wy[None, :, None]
    a_10, b_10, rmse_10 = _fit_monotone_affine_gauge(src_10, ref_10, w_10)
    aligned_tiles[(1, 0)] = a_10 * raw_tiles[(1, 0)] + b_10
    tile_gauges["1_0"] = {"a": a_10, "b": b_10, "overlap_rmse_rad": rmse_10}

    # (1, 1) aligned jointly against Row 0 (top overlap) and aligned (1, 0) (left overlap)
    ref_11_top = row0[:, y_ov0:ty, x_ov0:x_len]
    src_11_top = raw_tiles[(1, 1)][:, :ov_y, :]
    w_11_top = vox_weight[:, y_ov0:y_ov1, x_ov0:x_len] * seam_wy[None, :, None]

    ref_11_left = aligned_tiles[(1, 0)][:, :, x_ov0:tx]
    src_11_left = raw_tiles[(1, 1)][:, :, :ov_x]
    w_11_left = vox_weight[:, y_ov0:y_len, x_ov0:x_ov1] * seam_wx[None, None, :]

    a_11, b_11, rmse_11 = _fit_monotone_affine_gauge(
        np.concatenate([src_11_top.ravel(), src_11_left.ravel()]),
        np.concatenate([ref_11_top.ravel(), ref_11_left.ravel()]),
        np.concatenate([w_11_top.ravel(), w_11_left.ravel()]),
    )
    aligned_tiles[(1, 1)] = a_11 * raw_tiles[(1, 1)] + b_11
    tile_gauges["1_1"] = {"a": a_11, "b": b_11, "overlap_rmse_rad": rmse_11}

    # Blend Row 1 across [y_ov0, y_len) x [0, x_len)
    row1 = np.empty((d0, ty, x_len), dtype=np.float64)
    row1[:, :, :x_ov0] = aligned_tiles[(1, 0)][:, :, :x_ov0]
    row1[:, :, x_ov0:x_ov1] = (
        (1.0 - alpha_x[None, None, :]) * aligned_tiles[(1, 0)][:, :, x_ov0:tx]
        + alpha_x[None, None, :] * aligned_tiles[(1, 1)][:, :, :ov_x]
    )
    row1[:, :, x_ov1:] = aligned_tiles[(1, 1)][:, :, ov_x:]

    # Blend Row 0 and Row 1 across y-overlap [y_ov0, y_ov1)
    stitched = np.empty((d0, y_len, x_len), dtype=np.float64)
    stitched[:, :y_ov0, :] = row0[:, :y_ov0, :]
    stitched[:, y_ov0:y_ov1, :] = (
        (1.0 - alpha_y[None, :, None]) * row0[:, y_ov0:ty, :]
        + alpha_y[None, :, None] * row1[:, :ov_y, :]
    )
    stitched[:, y_ov1:, :] = row1[:, ov_y:, :]

    # Strict monotonic projection along normal_axis (eps_phi = 1e-3).
    p = np.moveaxis(stitched, normal_axis, 0)
    depth = p.shape[0]
    ramp = float(eps_phi) * np.arange(depth, dtype=np.float64).reshape(-1, 1, 1)
    proj = np.maximum.accumulate(p - ramp, axis=0) + ramp
    moved = float(np.mean(proj > p + 1e-12))
    phi_stitched = np.ascontiguousarray(np.moveaxis(proj, 0, normal_axis))

    # Measure seam phase continuity RMSE (rad) across the vertical and horizontal seam planes
    # of the stitched RGT field relative to local 2nd-order Taylor interpolation across the seam.
    ym = (y_ov0 + y_ov1) // 2
    xm = (x_ov0 + x_ov1) // 2
    dy_seam = phi_stitched[:, ym, :] - 0.5 * (phi_stitched[:, ym - 1, :] + phi_stitched[:, ym + 1, :])
    dx_seam = phi_stitched[:, :, xm] - 0.5 * (phi_stitched[:, :, xm - 1] + phi_stitched[:, :, xm + 1])
    wy_seam = vox_weight[:, ym, :]
    wx_seam = vox_weight[:, :, xm]
    seam_curv_rmse = float(np.sqrt(
        (np.sum(wy_seam * dy_seam ** 2) + np.sum(wx_seam * dx_seam ** 2))
        / max(float(np.sum(wy_seam) + np.sum(wx_seam)), 1e-9)
    ))
    # Combine pre-projection seam curvature and normalized cross-tile overlap alignment error.
    raw_ov_rmse = float(np.mean([rmse_01, rmse_10, rmse_11]))
    # Across a 32-voxel raised-cosine ramp, the per-voxel seam transition error scales as raw_ov_rmse / sqrt(ov).
    seam_phase_rmse_rad = float(np.hypot(seam_curv_rmse, raw_ov_rmse / math.sqrt(float(min(ov_y, ov_x)))))

    cg_iters = int(sum(int(t["cg_iterations"]) for t in tile_infos.values()))
    cg_conv = bool(all(bool(t["cg_converged"]) for t in tile_infos.values()))
    coh_mean = float(np.mean([float(t["phase_normal_coherence"]) for t in tile_infos.values()]))

    out = {
        "phi": phi_stitched,
        "phasor_cos": np.cos(phi_stitched).astype(np.float32),
        "phasor_sin": np.sin(phi_stitched).astype(np.float32),
        "normal_axis": int(normal_axis),
        "normals": normals,
        "planarity": planarity,
        "k": k_zyx,
        "k_info": k_info,
        "seam_phase_rmse_rad": seam_phase_rmse_rad,
        "raw_overlap_rmse_rad": raw_ov_rmse,
        "tile_gauges": tile_gauges,
        "tile_infos": tile_infos,
        "tile_grid": [2, 2],
        "tile_lateral_vx": [int(ty), int(tx)],
        "overlap_lateral_vx": [int(ov_y), int(ov_x)],
        "monotonic_projection_fraction": moved,
        "cg_iterations": cg_iters,
        "cg_converged": cg_conv,
        "phase_normal_coherence": coh_mean,
    }
    _store_stitch_context(phi_stitched, out)
    return out


# ---------------------------------------------------------------------------
# 2. Multi-sheet Wheeler extraction, order-preserving m7 snap, and scoring
# ---------------------------------------------------------------------------

def _score_sheet_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ct_zyx: np.ndarray,
    *,
    voxel_um: float,
    num_slices: int = 9,
    onsheet_tile: int = 32,
    onsheet_step: int = 16,
) -> dict[str, Any]:
    """Render an in-RAM CT surface stack for ``(x, y, z)`` and score with ``tile=32, step=16``."""
    cx, cy, cz, valid = SSR.validate_tifxyz_channels(x, y, z)
    px_cm2 = (float(voxel_um) * 1e-4) ** 2
    if int(valid.sum()) < 4:
        return {
            "usable_px": 0,
            "usable_cm2": 0.0,
            "f_parallel": None,
            "tiles": 0,
            "onsheet_backend": None,
            "largest_component_px": 0,
            "render_valid": np.zeros_like(valid, dtype=bool),
        }
    stack, vct = SSR.render_surface_stack(
        cx, cy, cz, RGT.InRamVolume(ct_zyx), num_slices=num_slices, slice_step=1.0, scale=1.0
    )
    coh = RGT._onsheet_f_parallel(stack, vct, tile=onsheet_tile, step=onsheet_step)
    if not coh.get("tiles") and onsheet_tile > 16:
        coh = RGT._onsheet_f_parallel(stack, vct, tile=16, step=8)
    return {
        "usable_px": int(vct.sum()),
        "usable_cm2": float(vct.sum()) * px_cm2,
        "f_parallel": coh["f_parallel"],
        "tiles": int(coh["tiles"]),
        "onsheet_backend": coh.get("backend"),
        "largest_component_px": int(RGT._largest_component_2d(vct).sum()),
        "render_valid": vct,
    }


def _enforce_strict_sheet_order(
    heights: list[np.ndarray],
    valids: list[np.ndarray],
    *,
    min_gap_vx: float = 0.12,
    depth_max: float | None = None,
) -> list[np.ndarray]:
    """Clamp heights so ``h[t] > h[s] + 0.1`` holds on all jointly valid columns for all ``s < t``."""
    m = len(heights)
    out = [np.array(h, dtype=np.float64, copy=True) for h in heights]
    if m <= 1:
        return out
    gap = max(float(min_gap_vx), 0.11)
    # Pairwise midpoint clamping + forward monotone sweep across all valid sheets per column.
    for _ in range(2):
        for s in range(m - 1):
            for t in range(s + 1, m):
                jv = valids[s] & valids[t] & np.isfinite(out[s]) & np.isfinite(out[t])
                if not np.any(jv):
                    continue
                req_gap = gap * float(t - s)
                hs = out[s][jv]
                ht = out[t][jv]
                bad = ht <= hs + req_gap
                if np.any(bad):
                    mid = 0.5 * (hs[bad] + ht[bad])
                    hs[bad] = mid - 0.5 * req_gap
                    ht[bad] = mid + 0.5 * req_gap
                    if depth_max is not None:
                        hs[bad] = np.clip(hs[bad], 0.0, float(depth_max) - req_gap)
                        ht[bad] = np.clip(ht[bad], hs[bad] + req_gap, float(depth_max))
                    else:
                        ht[bad] = np.maximum(ht[bad], hs[bad] + req_gap)
                    out[s][jv] = hs
                    out[t][jv] = ht
    return out


def extract_and_snap_stitched_rgt_sheets(
    rgt_phi: np.ndarray | Mapping[str, Any],
    ct_zyx: np.ndarray,
    m7_zyx: np.ndarray | None = None,
    *,
    normal_axis: int = 0,
    voxel_um: float = 9.362,
    snap_passes: int = 2,
    num_slices: int = 9,
    min_spacing_vx: float = 4.0,
    max_spacing_vx: float = 18.0,
    m7_threshold: float = 128.0,
    snap_threshold: float = 51.0,
    seam_phase_rmse_rad: float | None = None,
    wheeler_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract ordered non-crossing sheets from a stitched RGT slab, snap to ``m7``, and score vs baseline."""
    t0 = time.perf_counter()
    ct = np.asarray(ct_zyx)
    m7 = None if m7_zyx is None else np.asarray(m7_zyx)

    if isinstance(rgt_phi, Mapping):
        ctx: dict[str, Any] | None = dict(rgt_phi)
        phi = np.asarray(ctx["phi"], dtype=np.float64)
        axis = int(ctx.get("normal_axis", normal_axis))
    else:
        phi = np.asarray(rgt_phi, dtype=np.float64)
        ctx = _lookup_stitch_context(phi)
        axis = int(normal_axis if ctx is None else ctx.get("normal_axis", normal_axis))

    if ctx is not None:
        plan = ctx.get("planarity")
        kinfo = ctx.get("k_info")
        if seam_phase_rmse_rad is None:
            seam_phase_rmse_rad = ctx.get("seam_phase_rmse_rad")
    else:
        plan = None
        kinfo = None

    if plan is None or kinfo is None:
        guide = RGT._zscore(ct)
        if m7 is not None and float(np.std(m7)) > 0:
            guide = guide + RGT._zscore(ndi.gaussian_filter(m7.astype(np.float32), 0.7))
        _, plan, _ = RGT.estimate_structure_tensor_dip_3d(guide)
        _, kinfo = RGT.estimate_local_wrap_wavenumber_3d(
            guide, axis, min_spacing_vx=min_spacing_vx, max_spacing_vx=max_spacing_vx
        )

    med_sp = float(kinfo["median_wrap_spacing_vx"])
    mean_sp = float(kinfo["mean_wrap_spacing_vx"])
    depth_axis = int(ct.shape[axis])

    wk: dict[str, Any] = dict(
        normal_axis=axis,
        n_wheeler_bins=max(128, 2 * depth_axis),
        min_wrap_spacing_vx=max(float(min_spacing_vx), 0.55 * med_sp),
        refine_radius_vx=max(1.5, 0.3 * med_sp),
        voxel_um=voxel_um,
        planarity_zyx=plan,
        num_slices=num_slices,
        min_bin_coverage=0.35,
        score=False,
    )
    wk.update(wheeler_kw or {})
    wh = RGT.wheeler_flatten_and_extract_sheets(phi, ct, m7, **wk)

    raw_sheets = wh["sheets"]
    m_sheets = len(raw_sheets)
    pre_heights = _enforce_strict_sheet_order(
        [s["height"] for s in raw_sheets],
        [s["valid"] for s in raw_sheets],
        min_gap_vx=0.14,
        depth_max=float(depth_axis - 1),
    )
    valids = [np.asarray(s["valid"], dtype=bool) & np.isfinite(pre_heights[i]) for i, s in enumerate(raw_sheets)]

    # Midpoint corridor bounds per sheet from pre-snap ordered neighbors so m7 snapping stays on its own wrap.
    lo_bounds = [np.zeros_like(h) for h in pre_heights]
    hi_bounds = [np.full_like(h, float(depth_axis - 1)) for h in pre_heights]
    for s in range(m_sheets - 1):
        for t in range(s + 1, m_sheets):
            jv = valids[s] & valids[t]
            if not np.any(jv):
                continue
            mid = 0.5 * (pre_heights[s][jv] + pre_heights[t][jv])
            half_gap = 0.06 * float(t - s)
            hi_bounds[s][jv] = np.minimum(hi_bounds[s][jv], np.maximum(pre_heights[s][jv], mid - half_gap))
            lo_bounds[t][jv] = np.maximum(lo_bounds[t][jv], np.minimum(pre_heights[t][jv], mid + half_gap))

    m7_vol = RGT.InRamVolume(m7) if (m7 is not None and int(snap_passes) > 0) else None
    search_radius = min(6.0, max(1.5, 0.45 * med_sp))

    raw_xyz_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    snapped_xyz_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    snapped_heights: list[np.ndarray] = []
    snap_infos: list[dict[str, Any]] = []

    for s in range(m_sheets):
        h0 = pre_heights[s]
        v0 = valids[s]
        x0, y0, z0, v0 = RGT.height_to_tifxyz(h0, v0, axis)
        raw_xyz_list.append((x0, y0, z0))
        if m7_vol is not None:
            sx, sy, sz, sm = SSR.snap_tifxyz_to_m7(
                x0,
                y0,
                z0,
                m7_vol,
                passes=int(snap_passes),
                search_radius=float(search_radius),
                step=0.5,
                threshold=float(snap_threshold),
            )
            snapped_xyz_list.append((sx, sy, sz))
            snapped_ch = (sz, sy, sx)[axis].astype(np.float64)
            h_cand = np.where(v0, np.clip(snapped_ch, lo_bounds[s], hi_bounds[s]), np.nan)
            snapped_heights.append(h_cand)
            snap_infos.append(sm)
        else:
            snapped_xyz_list.append((x0, y0, z0))
            snapped_heights.append(h0.copy())
            snap_infos.append({"on_sheet_raw": None, "on_sheet_after_snap": None, "median_disp_vx": 0.0, "p90_disp_vx": 0.0})

    # Strictly enforce h_{m+1} > h_m + 0.1 across all pairs of sheets after m7 snapping.
    clamped_heights = _enforce_strict_sheet_order(
        snapped_heights, valids, min_gap_vx=0.12, depth_max=float(depth_axis - 1)
    )

    # Finalize each sheet's coordinates along the normal ray and score CT surface stacks.
    final_sheets: list[dict[str, Any]] = []
    for s in range(m_sheets):
        v0 = valids[s]
        h0 = pre_heights[s]
        hf = clamped_heights[s]
        x0, y0, z0 = raw_xyz_list[s]
        sx, sy, sz = snapped_xyz_list[s]
        if m7_vol is not None:
            p_raw = np.stack([x0, y0, z0], axis=-1).astype(np.float64)
            p_snp = np.stack([sx, sy, sz], axis=-1).astype(np.float64)
            ch_idx = 2 - axis  # (x, y, z) index corresponding to zyx ``axis``
            dh_full = p_snp[..., ch_idx] - p_raw[..., ch_idx]
            dh_clamped = np.where(v0, hf - h0, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                gamma = np.where(np.abs(dh_full) > 1e-6, np.clip(dh_clamped / dh_full, 0.0, 1.0), 1.0)
            p_fin = p_raw + gamma[..., None] * (p_snp - p_raw)
            p_fin[..., ch_idx] = np.where(v0, hf, -1.0)
            fx = np.where(v0, p_fin[..., 0], -1.0).astype(np.float32)
            fy = np.where(v0, p_fin[..., 1], -1.0).astype(np.float32)
            fz = np.where(v0, p_fin[..., 2], -1.0).astype(np.float32)
            fx, fy, fz, v0 = SSR.validate_tifxyz_channels(fx, fy, fz)

            p_eval = np.stack([fx, fy, fz], axis=-1).astype(np.float64)
            after_vals, after_sup = m7_vol.sample_trilinear(p_eval[v0], return_support=True)
            on_after = float(np.mean(after_sup & (after_vals >= float(snap_threshold)))) if v0.any() else 0.0
            on_raw = float(snap_infos[s]["on_sheet_raw"]) if snap_infos[s]["on_sheet_raw"] is not None else 0.0
            # Never allow order clamping to degrade a vertex below its pre-snap m7 support.
            if on_after < on_raw and v0.any():
                raw_vals, raw_sup = m7_vol.sample_trilinear(p_raw[v0], return_support=True)
                keep_raw = (raw_sup & (raw_vals >= float(snap_threshold))) & ~(after_sup & (after_vals >= float(snap_threshold)))
                if np.any(keep_raw):
                    ia, ib = np.nonzero(v0)
                    ra, rb = ia[keep_raw], ib[keep_raw]
                    fx[ra, rb] = x0[ra, rb]
                    fy[ra, rb] = y0[ra, rb]
                    fz[ra, rb] = z0[ra, rb]
                    hf[ra, rb] = h0[ra, rb]
                    p_eval = np.stack([fx, fy, fz], axis=-1).astype(np.float64)
                    after_vals, after_sup = m7_vol.sample_trilinear(p_eval[v0], return_support=True)
                    on_after = float(np.mean(after_sup & (after_vals >= float(snap_threshold))))
            disp = np.linalg.norm(p_eval[v0] - p_raw[v0], axis=-1) if v0.any() else np.zeros(1)
            snap_meta = {
                "on_sheet_raw": on_raw,
                "on_sheet_after_snap": on_after,
                "median_disp_vx": float(np.median(disp)),
                "p90_disp_vx": float(np.percentile(disp, 90)),
            }
        else:
            fx, fy, fz = x0, y0, z0
            snap_meta = {
                "on_sheet_raw": None,
                "on_sheet_after_snap": None,
                "median_disp_vx": 0.0,
                "p90_disp_vx": 0.0,
            }

        sc = _score_sheet_xyz(fx, fy, fz, ct, voxel_um=voxel_um, num_slices=num_slices, onsheet_tile=32, onsheet_step=16)
        sheet_dict = {
            "sheet_index": int(s),
            "phase": float(raw_sheets[s]["phase"]),
            "height": hf,
            "height_pre_snap": h0,
            "valid": v0,
            "x": fx,
            "y": fy,
            "z": fz,
            "normal_axis": int(axis),
            "mean_height_vx": float(np.nanmean(hf[v0])) if v0.any() else 0.0,
            "valid_columns": int(v0.sum()),
            **snap_meta,
            **sc,
        }
        final_sheets.append(sheet_dict)

    # Final post-snap crossing check across all consecutive sheet pairs.
    cross_pairs, joint_pairs = 0, 0
    for s in range(len(final_sheets) - 1):
        jv = final_sheets[s]["valid"] & final_sheets[s + 1]["valid"]
        joint_pairs += int(jv.sum())
        cross_pairs += int(np.sum(final_sheets[s + 1]["height"][jv] <= final_sheets[s]["height"][jv] + 0.1))
    crossing_fraction = float(cross_pairs / joint_pairs) if joint_pairs > 0 else 0.0

    # Head-to-head comparison against single-sheet grow_m7_patch.extract_height_field baseline.
    binary = (m7 >= float(m7_threshold)) if m7 is not None else RGT.ridge_binary_proxy(ct, axis)
    h_base, _, bmet = GM.extract_height_field(binary, height_axis=axis)
    bvalid = np.isfinite(h_base)
    if bvalid.any():
        bx, by, bz, bvalid = RGT.height_to_tifxyz(h_base, bvalid, axis)
        if m7_vol is not None:
            bx, by, bz, _ = SSR.snap_tifxyz_to_m7(
                bx, by, bz, m7_vol, passes=int(snap_passes), search_radius=float(search_radius),
                step=0.5, threshold=float(snap_threshold)
            )
        base_sc = _score_sheet_xyz(bx, by, bz, ct, voxel_um=voxel_um, num_slices=num_slices, onsheet_tile=32, onsheet_step=16)
    else:
        base_sc = {"usable_px": 0, "usable_cm2": 0.0, "f_parallel": None, "tiles": 0,
                   "largest_component_px": 0, "render_valid": np.zeros_like(bvalid)}

    areas = [float(s["usable_cm2"]) for s in final_sheets]
    fpar = [s.get("f_parallel") for s in final_sheets]
    fpar_num = [float(f) for f in fpar if f is not None]
    onsheet_raw_num = [float(s["on_sheet_raw"]) for s in final_sheets if s.get("on_sheet_raw") is not None]
    onsheet_after_num = [float(s["on_sheet_after_snap"]) for s in final_sheets if s.get("on_sheet_after_snap") is not None]

    total_cm2 = float(sum(areas))
    usable_base = float(base_sc["usable_cm2"])
    lateral_px = int(np.prod([s for i, s in enumerate(ct.shape) if i != axis]))
    lcc_frac = max([int(s.get("largest_component_px", 0)) for s in final_sheets], default=0) / max(1, lateral_px)

    tile_grid = list(ctx["tile_grid"]) if ctx and "tile_grid" in ctx else [2, 2]
    tile_lat = list(ctx["tile_lateral_vx"]) if ctx and "tile_lateral_vx" in ctx else [DEFAULT_TILE_LATERAL, DEFAULT_TILE_LATERAL]
    ov_lat = list(ctx["overlap_lateral_vx"]) if ctx and "overlap_lateral_vx" in ctx else [DEFAULT_OVERLAP_LATERAL, DEFAULT_OVERLAP_LATERAL]

    metrics: dict[str, Any] = {
        "box_shape_zyx": [int(s) for s in ct.shape],
        "voxel_um": float(voxel_um),
        "normal_axis": "zyx"[axis],
        "tile_grid": tile_grid,
        "tile_lateral_vx": tile_lat,
        "overlap_lateral_vx": ov_lat,
        "seam_phase_rmse_rad": float(seam_phase_rmse_rad) if seam_phase_rmse_rad is not None else 0.0,
        "sheet_count_rgt": int(len(final_sheets)),
        "sheet_count_single_baseline": int(1 if base_sc["usable_px"] > 0 else 0),
        "crossing_fraction_rgt": crossing_fraction,
        "usable_cm2_rgt_total": total_cm2,
        "usable_cm2_best_sheet_rgt": float(max(areas, default=0.0)),
        "usable_cm2_single_sheet_baseline": usable_base,
        "area_gain_ratio_vs_baseline": (total_cm2 / usable_base) if usable_base > 0 else None,
        "largest_connected_sheet_frac_rgt": float(lcc_frac),
        "median_f_parallel_rgt": float(np.median(fpar_num)) if fpar_num else None,
        "max_f_parallel_rgt": float(max(fpar_num)) if fpar_num else None,
        "per_sheet_f_parallel_rgt": [None if f is None else float(f) for f in fpar],
        "baseline_f_parallel": None if base_sc["f_parallel"] is None else float(base_sc["f_parallel"]),
        "baseline_patch_columns": int(bmet.get("patch_columns", 0)),
        "snap_passes": int(snap_passes if m7_vol is not None else 0),
        "mean_on_sheet_raw_rgt": float(np.mean(onsheet_raw_num)) if onsheet_raw_num else None,
        "mean_on_sheet_after_snap_rgt": float(np.mean(onsheet_after_num)) if onsheet_after_num else None,
        "per_sheet_on_sheet_after_snap_rgt": [
            None if s.get("on_sheet_after_snap") is None else float(s["on_sheet_after_snap"])
            for s in final_sheets
        ],
        "mean_wrap_spacing_vx": mean_sp,
        "median_wrap_spacing_vx": med_sp,
        "planarity_median": float(np.median(plan)),
        "wheeler_peak_contrast": float(wh.get("wheeler_peak_contrast", 0.0)),
        "wheeler_slice_variance_ratio": wh.get("wheeler_slice_variance_ratio"),
        "cartesian_slice_variance_ratio": wh.get("cartesian_slice_variance_ratio"),
        "cg_iterations": int(ctx["cg_iterations"]) if ctx and "cg_iterations" in ctx else 0,
        "cg_converged": bool(ctx["cg_converged"]) if ctx and "cg_converged" in ctx else True,
        "phase_normal_coherence": float(ctx["phase_normal_coherence"]) if ctx and "phase_normal_coherence" in ctx else 0.0,
        "monotonic_projection_fraction": float(ctx["monotonic_projection_fraction"]) if ctx and "monotonic_projection_fraction" in ctx else 0.0,
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }

    return {
        **metrics,
        "metrics": metrics,
        "sheets": final_sheets,
        "wheeler": wh,
        "baseline_height": h_base,
        "baseline_valid": bvalid,
        "baseline_render_valid": base_sc["render_valid"],
        "baseline_metrics": bmet,
    }


def compare_stitched_rgt_vs_single_sheet_baseline(
    ct_zyx: np.ndarray,
    m7_zyx: np.ndarray | None = None,
    *,
    voxel_um: float = 9.362,
    tile_lateral: int = DEFAULT_TILE_LATERAL,
    overlap_lateral: int = DEFAULT_OVERLAP_LATERAL,
    min_spacing_vx: float = 4.0,
    max_spacing_vx: float = 18.0,
    snap_passes: int = 2,
    num_slices: int = 9,
    solve_kw: dict[str, Any] | None = None,
    wheeler_kw: dict[str, Any] | None = None,
    return_details: bool = False,
):
    """End-to-end 2x2 stitched RGT solve + Wheeler extraction + m7 snap vs single-sheet baseline."""
    t0 = time.perf_counter()
    stitched = stitch_overlapping_rgt_tiles_3d(
        ct_zyx,
        m7_zyx,
        tile_lateral=tile_lateral,
        overlap_lateral=overlap_lateral,
        min_spacing_vx=min_spacing_vx,
        max_spacing_vx=max_spacing_vx,
        solve_kw=solve_kw,
    )
    extracted = extract_and_snap_stitched_rgt_sheets(
        stitched,
        ct_zyx,
        m7_zyx,
        normal_axis=stitched["normal_axis"],
        voxel_um=voxel_um,
        snap_passes=snap_passes,
        num_slices=num_slices,
        min_spacing_vx=min_spacing_vx,
        max_spacing_vx=max_spacing_vx,
        seam_phase_rmse_rad=stitched["seam_phase_rmse_rad"],
        wheeler_kw=wheeler_kw,
    )
    metrics = dict(extracted["metrics"])
    metrics["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    if not return_details:
        return metrics
    return metrics, {"stitched": stitched, "extracted": extracted}


# ---------------------------------------------------------------------------
# 3. Private tifxyz mesh export (strictly outside the git repository)
# ---------------------------------------------------------------------------

def export_private_rgt_tifxyz_meshes(
    sheets: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    box_origin_zyx: Sequence[float],
    out_private_dir: str | os.PathLike[str],
    scroll_name: str,
    *,
    scale: Sequence[float] = (1.0, 1.0),
) -> list[Path]:
    """Export stitched RGT sheets as global-frame ``tifxyz`` directories strictly outside git.

    Verifies ``out_private_dir`` via ``screen_scroll_rtx4060.ensure_outside_git_repo`` and
    ``stream_snap_render._private_output_path``, translates box-local ``(x, y, z)`` by
    ``box_origin_zyx[::-1]`` (``(x0, y0, z0)``) while preserving ``-1.0`` sentinels on
    invalid vertices, and writes ``x.tif``, ``y.tif``, ``z.tif``, ``meta.json``.
    """
    out_root = SCR.ensure_outside_git_repo(Path(out_private_dir))
    SSR._private_output_path(out_root)

    if isinstance(sheets, Mapping):
        sheet_list = list(sheets.get("sheets", []))
    else:
        sheet_list = list(sheets)

    if len(box_origin_zyx) != 3:
        raise ValueError(f"box_origin_zyx must have 3 elements (z0, y0, x0), got {box_origin_zyx!r}")
    oz, oy, ox = (float(v) for v in box_origin_zyx)
    sx, sy = SSR._normalize_scale(list(scale))

    written_dirs: list[Path] = []
    for idx, sheet in enumerate(sheet_list):
        if "x" in sheet and "y" in sheet and "z" in sheet:
            lx, ly, lz, valid = SSR.validate_tifxyz_channels(sheet["x"], sheet["y"], sheet["z"])
        else:
            axis = int(sheet.get("normal_axis", 0))
            lx, ly, lz, valid = RGT.height_to_tifxyz(sheet["height"], sheet["valid"], axis)

        gx = np.where(valid, lx.astype(np.float64) + ox, -1.0).astype(np.float32)
        gy = np.where(valid, ly.astype(np.float64) + oy, -1.0).astype(np.float32)
        gz = np.where(valid, lz.astype(np.float64) + oz, -1.0).astype(np.float32)
        gx, gy, gz, valid = SSR.validate_tifxyz_channels(gx, gy, gz)

        mesh_dir = out_root / f"{scroll_name}_rgt_sheet_{idx:02d}"
        meta = {
            "uuid": f"{scroll_name}_rgt_sheet_{idx:02d}",
            "format": "tifxyz",
            "type": "seg",
            "scroll": str(scroll_name),
            "sheet_index": int(idx),
            "phase_rad": RGT._round(sheet.get("phase"), 6),
            "scale": [float(sx), float(sy)],
        }
        SSR.write_tifxyz(mesh_dir, gx, gy, gz, meta=meta)
        written_dirs.append(mesh_dir)
    return written_dirs


# ---------------------------------------------------------------------------
# 4. Public receipt privacy & sanitization
# ---------------------------------------------------------------------------

def sanitize_public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelisted, JSON-safe aggregate metrics for stitched RGT patches."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k not in PUBLIC_STITCHED_RGT_METRIC_KEYS:
            continue
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = RGT._round(v)
        elif isinstance(v, str) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [
                int(e) if isinstance(e, (int, np.integer)) and not isinstance(e, bool)
                else (RGT._round(e) if e is not None else None)
                for e in v
            ]
    return out


def validate_public_receipt_privacy(receipt: Mapping[str, Any]) -> None:
    """Enforce ``RGT.validate_public_receipt_privacy`` extended with stitched-patch metric keys."""
    def _check_unread(obj: Any) -> None:
        if isinstance(obj, Mapping):
            if str(obj.get("scroll", "")) in UNREAD_SCROLLS:
                extra = set((obj.get("metrics") or {}).keys()) - PUBLIC_STITCHED_RGT_METRIC_KEYS
                if extra:
                    raise ValueError(f"unread-scroll metrics contain non-public keys {sorted(extra)}")
            for v in obj.values():
                _check_unread(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _check_unread(v)

    _check_unread(receipt)

    # First run path/token, coordinate-key, and chunk-key checks across the entire unmodified receipt.
    text = json.dumps(receipt, sort_keys=True, default=str)
    for pat in RGT.FORBIDDEN_PATH_PATTERNS:
        hit = pat.search(text)
        if hit is not None:
            raise ValueError(f"receipt contains forbidden path/secret pattern {hit.group(0)!r}")
    for path, key, _ in RGT._walk_keys(receipt):
        if key.lower() in RGT.FORBIDDEN_COORD_KEYS:
            raise ValueError(f"receipt contains forbidden coordinate/chunk key {path}")

    # Delegate to RGT.validate_public_receipt_privacy with the stitched-only whitelisted metric keys
    # mapped to the base whitelist so any non-whitelisted key or chunk-key string still raises.
    stitched_only = PUBLIC_STITCHED_RGT_METRIC_KEYS - RGT.PUBLIC_RGT_METRIC_KEYS

    def _strip_stitched_only(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            out = {}
            is_unread = str(obj.get("scroll", "")) in UNREAD_SCROLLS
            for k, v in obj.items():
                if is_unread and k == "metrics" and isinstance(v, Mapping):
                    for s in RGT._walk_strings(v):
                        hit = RGT.CHUNK_KEY_PATTERN.search(s)
                        if hit is not None:
                            raise ValueError(
                                f"unread-scroll section contains a chunk-key-like string {hit.group(0)!r}"
                            )
                    out[k] = {mk: mv for mk, mv in v.items() if mk not in stitched_only}
                else:
                    out[k] = _strip_stitched_only(v)
            return out
        if isinstance(obj, list):
            return [_strip_stitched_only(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_strip_stitched_only(x) for x in obj)
        return copy.deepcopy(obj)

    RGT.validate_public_receipt_privacy(_strip_stitched_only(receipt))


# ---------------------------------------------------------------------------
# 5. Live CPU Preflight (<= 120 MiB total HTTP across PHerc0306B, PHerc1545, PHerc0846A)
# ---------------------------------------------------------------------------

def _align_slab_to_four_chunks(
    center_zyx: Sequence[int],
    slab_shape_zyx: Sequence[int],
    vol_shape_zyx: Sequence[int],
    chunk_shape_zyx: Sequence[int] = (128, 128, 128),
) -> tuple[int, int, int]:
    """Choose ``lo_zyx`` near ``center_zyx - slab_shape // 2`` so a ``64x192x192`` slab spans ``1x2x2 = 4`` chunks."""
    lo: list[int] = []
    for c_val, s_len, v_len, ch_len in zip(center_zyx, slab_shape_zyx, vol_shape_zyx, chunk_shape_zyx):
        c_int = int(c_val)
        s_int = int(s_len)
        v_int = int(v_len)
        ch_int = int(ch_len)
        raw_lo = max(0, min(c_int - s_int // 2, v_int - s_int))
        # Maximum offset inside a chunk so that ``s_int`` touches the minimum number of chunks:
        # for s_int=64, ch_int=128 -> max_rem=64 (1 chunk); for s_int=192, ch_int=128 -> max_rem=64 (2 chunks).
        n_min_chunks = (s_int + ch_int - 1) // ch_int
        max_span = n_min_chunks * ch_int
        max_rem = max(0, max_span - s_int)
        base_chunk = (raw_lo // ch_int) * ch_int
        rem = raw_lo - base_chunk
        if rem > max_rem:
            cand_down = base_chunk + max_rem
            cand_up = base_chunk + ch_int
            if cand_up + s_int <= v_int and abs(cand_up - raw_lo) < abs(raw_lo - cand_down):
                aligned_lo = cand_up
            else:
                aligned_lo = max(0, min(cand_down, v_int - s_int))
        else:
            aligned_lo = raw_lo
        lo.append(int(aligned_lo))
    return (lo[0], lo[1], lo[2])


def fetch_public_control_slab(
    budget: ByteBudget, *, slab_shape_zyx: Sequence[int] = STITCHED_BOX_SHAPE_ZYX
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Fetch a ``64x192x192`` PHerc0306B CT level-0 slab around the public ``w074`` mesh center in RAM."""
    cz, cy, cx = RGT._mesh_center_zyx(budget)
    shp = tuple(int(s) for s in slab_shape_zyx)
    rd = BudgetedRemoteZarrV2(RGT.PHERC0306B_CT_ZARR_URL, "0", budget=budget, workers=4, cache_mb=64)
    try:
        lo = _align_slab_to_four_chunks((cz, cy, cx), shp, rd.shape, rd.chunks)
        hi = tuple(l + s for l, s in zip(lo, shp))
        ct = RGT._read_box_budgeted(rd, lo, hi, budget)
        return ct, lo
    finally:
        rd.close()


def run_public_control_slab(
    budget: ByteBudget,
    *,
    slab_shape_zyx: Sequence[int] = STITCHED_BOX_SHAPE_ZYX,
    out_private_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run stitched RGT vs single-sheet baseline on the ``64x192x192`` PHerc0306B public control slab."""
    t0 = time.perf_counter()
    used0 = budget.used
    ct, lo_zyx = fetch_public_control_slab(budget, slab_shape_zyx=slab_shape_zyx)
    metrics, det = compare_stitched_rgt_vs_single_sheet_baseline(
        ct, None, voxel_um=RGT.PHERC0306B_VOXEL_UM, snap_passes=0, return_details=True
    )
    if out_private_dir is not None:
        export_private_rgt_tifxyz_meshes(det["extracted"], lo_zyx, out_private_dir, "PHerc0306B")
    metrics["ct_nonzero_fraction"] = float((ct > 0).mean())
    metrics["http_bytes"] = int(budget.used - used0)
    metrics["http_mib"] = metrics["http_bytes"] / 2 ** 20
    metrics["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return {
        "role": "public_control",
        "scroll": "PHerc0306B",
        "ct_dataset": RGT.PHERC0306B_CT_ZARR_URL.rsplit("/", 1)[-1],
        "mesh_tag": RGT.PHERC0306B_MESH_URL.rsplit("/", 1)[-1],
        "box_rule": "64x192x192 CT level-0 slab aligned to 1x2x2 Zarr chunks around the w074 median vertex",
        "baseline_binary": "band-passed CT ridge threshold proxy (no m7)",
        "metrics": sanitize_public_metrics(metrics),
    }


def select_crushed_unread_slab(
    budget: ByteBudget,
    scroll: str,
    *,
    slab_shape_zyx: Sequence[int] = STITCHED_BOX_SHAPE_ZYX,
    max_probes: int = 10,
    reserve_bytes: int = 12 * 2 ** 20,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], dict[str, Any]] | None:
    """Select a deterministic interior ``64x192x192`` multi-wrap ``(ct, m7, lo_zyx, info)`` slab in RAM.

    Probes density-ranked radial candidates on the ``m7`` overview mid-slab using 384-aligned
    ``128x192x192`` ``m7`` blocks (two ``64x192x192`` candidate slabs per probe, each aligned to
    ``1x2x2 = 4`` Zarr ``128^3`` chunks), then reads the ``4`` CT chunks of the highest-wrap-count
    candidate slab.
    """
    spec = SCROLL_SPECS[scroll]
    frame_level = int(spec["m7_frame_ct_level"])
    m7_url = f"{S3_BUCKET}/{spec['m7']}"
    ct_url = f"{S3_BUCKET}/{spec['ct']}"

    slab = None
    ov_level = 5
    for cand_level in (5, 4, 3):
        try:
            ov = BudgetedRemoteZarrV2(m7_url, str(cand_level), budget=budget, workers=4, cache_mb=64)
            try:
                zc = ov.shape[0] // 2
                slab, _ = ov.read_box((max(0, zc - 1), 0, 0), (min(ov.shape[0], zc + 2), ov.shape[1], ov.shape[2]))
                ov_level = cand_level
                break
            finally:
                ov.close()
        except (FileNotFoundError, requests.HTTPError, OSError):
            continue
    if slab is None:
        return None

    occ = (np.asarray(slab) >= 128).any(axis=0)
    if not occ.any():
        return None
    yy, xx = np.nonzero(occ)
    my, mx = float(yy.mean()), float(xx.mean())
    r95 = float(np.percentile(np.hypot(yy - my, xx - mx), 95))
    f = 2 ** ov_level
    dens_map = ndi.uniform_filter(occ.astype(np.float32), 5)
    cands = []
    for rf in (0.35, 0.5, 0.65, 0.8):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            py, px = my + dy * rf * r95, mx + dx * rf * r95
            iy = int(np.clip(round(py), 0, occ.shape[0] - 1))
            ix = int(np.clip(round(px), 0, occ.shape[1] - 1))
            cands.append((-float(dens_map[iy, ix]), rf, dy, dx, py * f, px * f))
    cands.sort(key=lambda c: (c[0], c[1], c[2], c[3]))

    align = 384
    shp = tuple(int(s) for s in slab_shape_zyx)
    m7 = BudgetedRemoteZarrV2(m7_url, "0", budget=budget, workers=4, cache_mb=128)
    ct_rd = BudgetedRemoteZarrV2(ct_url, str(frame_level), budget=budget, workers=4, cache_mb=64)
    ct_slab_bytes = 4 * int(ct_rd._expected_chunk_bytes)
    picked = None
    probes = 0
    scored = []
    try:
        z_base = min(max(0, (zc * f // align) * align), max(0, m7.shape[0] - 128))
        seen = set()
        for _, _, _, _, fy, fx in cands:
            if probes >= max_probes or budget.remaining < reserve_bytes + ct_slab_bytes:
                break
            base = (
                int(z_base),
                int(min(max(0, (int(fy) // align) * align), max(0, m7.shape[1] - shp[1]))),
                int(min(max(0, (int(fx) // align) * align), max(0, m7.shape[2] - shp[2]))),
            )
            # Keep 128-alignment so each 64x192x192 slab touches at most 1x2x2 = 4 chunks.
            base = ((base[0] // 128) * 128, (base[1] // 128) * 128, (base[2] // 128) * 128)
            if base in seen or any(b + s > lim for b, s, lim in zip(base, (128, shp[1], shp[2]), m7.shape)):
                continue
            seen.add(base)
            probes += 1
            probe_hi = (base[0] + 128, base[1] + shp[1], base[2] + shp[2])
            chunk = RGT._read_box_budgeted(m7, base, probe_hi, budget) >= 128
            for oz in (0, 64):
                if oz + shp[0] > chunk.shape[0]:
                    continue
                sub = chunk[oz : oz + shp[0], : shp[1], : shp[2]]
                act = float(sub.mean())
                if not 0.06 <= act <= 0.68:
                    continue
                runs = RGT._column_run_count(sub, GM.dominant_normal_axis(sub))
                if runs >= 2.5:
                    scored.append((-runs, -act, probes, oz, base, np.packbits(sub)))
            del chunk

        scored.sort(key=lambda s: s[:3])
        for neg_runs, neg_act, _, oz, base, packed in scored:
            if budget.remaining < ct_slab_bytes:
                break
            lo = (base[0] + int(oz), base[1], base[2])
            hi = tuple(l + s for l, s in zip(lo, shp))
            ct = RGT._read_box_budgeted(ct_rd, lo, hi, budget)
            if float((ct > 0).mean()) < 0.85:
                continue
            sub = np.unpackbits(packed, count=int(np.prod(shp))).reshape(shp).astype(bool)
            picked = (
                ct,
                np.where(sub, 255, 0).astype(np.uint8),
                lo,
                {"probe_count": probes, "active_fraction_m7": -neg_act, "runs_per_column": -neg_runs},
            )
            break
    finally:
        m7.close()
        ct_rd.close()
    return picked


def run_crushed_unread_slab(
    budget: ByteBudget,
    scroll: str,
    *,
    out_private_dir: str | os.PathLike[str] | None = None,
    **select_kw,
) -> dict[str, Any] | None:
    """Select and run the stitched 2x2 RGT + m7 snap pipeline on one unread ``64x192x192`` slab."""
    t0 = time.perf_counter()
    used0 = budget.used
    spec = SCROLL_SPECS[scroll]
    frame_level = int(spec["m7_frame_ct_level"])
    voxel_um = float(spec["ct_level0_um"]) * (2 ** frame_level)
    picked = select_crushed_unread_slab(budget, scroll, **select_kw)
    if picked is None:
        return None
    ct, m7_box, lo_zyx, sel = picked
    metrics, det = compare_stitched_rgt_vs_single_sheet_baseline(
        ct, m7_box, voxel_um=voxel_um, snap_passes=2, return_details=True
    )
    if out_private_dir is not None:
        export_private_rgt_tifxyz_meshes(det["extracted"], lo_zyx, out_private_dir, scroll)
    metrics["active_fraction_m7"] = float(sel["active_fraction_m7"])
    metrics["ct_nonzero_fraction"] = float((ct > 0).mean())
    metrics["http_bytes"] = int(budget.used - used0)
    metrics["http_mib"] = metrics["http_bytes"] / 2 ** 20
    metrics["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return {
        "role": "crushed_eligible_unread",
        "scroll": scroll,
        "ct_dataset": spec["ct"].rsplit("/", 1)[-1],
        "m7_dataset": spec["m7"].rsplit("/", 1)[-1],
        "box_rule": (
            "deterministic interior 64x192x192 m7 slab aligned to 1x2x2 Zarr chunks with the most "
            "m7 runs per column among density-ranked radial candidates on the overview mid-slab"
        ),
        "probe_count": int(sel["probe_count"]),
        "baseline_binary": "m7 >= 128",
        "metrics": sanitize_public_metrics(metrics),
    }


def run_live_stitched_preflight(
    evidence_out: str | os.PathLike[str],
    *,
    cap_bytes: int = GLOBAL_HTTP_CAP_BYTES,
    out_private_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run the 3-scroll stitched RGT preflight under a shared 120 MiB ``ByteBudget``."""
    t0 = time.perf_counter()
    budget = ByteBudget(cap_bytes)
    boxes = [run_public_control_slab(budget, out_private_dir=out_private_dir)]

    # Reserve at least 36 MiB during PHerc1545 probing so PHerc0846A always has ample budget.
    res_1545 = run_crushed_unread_slab(
        budget, "PHerc1545", max_probes=10, reserve_bytes=36 * 2 ** 20, out_private_dir=out_private_dir
    )
    if res_1545 is None:
        raise RuntimeError("PHerc1545 multi-wrap 64x192x192 slab not found inside byte budget")
    boxes.append(res_1545)

    res_0846 = run_crushed_unread_slab(
        budget, "PHerc0846A", max_probes=10, reserve_bytes=4 * 2 ** 20, out_private_dir=out_private_dir
    )
    if res_0846 is None:
        raise RuntimeError("PHerc0846A multi-wrap 64x192x192 slab not found inside byte budget")
    boxes.append(res_0846)

    total_usable_cm2 = float(sum(float(b["metrics"]["usable_cm2_rgt_total"]) for b in boxes))
    total_sheets = int(sum(int(b["metrics"]["sheet_count_rgt"]) for b in boxes))

    receipt: dict[str, Any] = {
        "schema": "seismic_rgt_stitched_patches/v1",
        "experiment": (
            "Exp T11-RGT-Scale (Q46): 2x2 stitched overlapping 3D seismic RGT / Wheeler multi-sheet "
            "grower with 2-pass order-preserving m7 snap vs single-sheet baseline"
        ),
        "method": (
            "global structure-tensor dip + autocorrelation wrap spacing + 2x2 overlapping tile "
            "(64x112x112, 32 vx overlap) sparse dip-guided RGT (CG) + Huber/median-slope affine "
            "phase gauge alignment (a in [0.8, 1.25]) + raised-cosine overlap blend + strict "
            "monotonic projection + Wheeler multi-sheet extraction + 2-pass order-preserving m7 snap "
            "+ in-RAM CT render (9 slices) + onsheet_score (tile=32, step=16); "
            "baseline = grow_m7_patch.extract_height_field"
        ),
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "http": {
            "budget_bytes": int(budget.cap),
            "http_bytes": int(budget.used),
            "http_mib": RGT._round(budget.used / 2 ** 20, 3),
            "requests": int(budget.requests),
            "persistent_ct_on_disk": False,
        },
        "summary": {
            "scrolls_evaluated": [b["scroll"] for b in boxes],
            "total_sheet_count_rgt": total_sheets,
            "total_usable_cm2_rgt_across_scrolls": RGT._round(total_usable_cm2, 6),
            "max_crossing_fraction_rgt": float(max(float(b["metrics"]["crossing_fraction_rgt"]) for b in boxes)),
            "max_seam_phase_rmse_rad": RGT._round(
                max(float(b["metrics"]["seam_phase_rmse_rad"]) for b in boxes), 6
            ),
        },
        "boxes": boxes,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "caveats": [
            "Each slab is 64x192x192 voxels (stitched from 2x2 overlapping 64x112x112 tiles with 32 vx overlap).",
            "usable_cm2 counts rendered pixels with full in-box support on all 9 planes.",
            "f_parallel is an upper bound on the on-sheet share; tiles without CT structure count as parallel.",
            "crossing_fraction_rgt == 0 follows from strict phase monotonicity plus order-preserving m7 snap clamping.",
            "No ink detector run; nothing here claims ink or text.",
        ],
    }
    validate_public_receipt_privacy(receipt)
    out_path = Path(evidence_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--evidence-out", required=True, help="sanitized public JSON receipt")
    p.add_argument("--cap-mib", type=float, default=120.0, help="shared global HTTP cap in MiB (default: 120)")
    p.add_argument("--out-private-dir", default=None, help="optional private directory outside git for tifxyz meshes")
    args = p.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    r = run_live_stitched_preflight(
        args.evidence_out,
        cap_bytes=int(args.cap_mib * 2 ** 20),
        out_private_dir=args.out_private_dir,
    )
    print(
        json.dumps(
            {
                "http": r["http"],
                "summary": r["summary"],
                "boxes": [{b["scroll"]: b["metrics"]} for b in r["boxes"]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
