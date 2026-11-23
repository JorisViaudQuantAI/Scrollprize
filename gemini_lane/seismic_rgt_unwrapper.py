# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Exp T9-Unwrap (Q46): CPU-only seismic RGT / Wheeler multi-sheet unwrapper for crushed scroll zones.

Seismic horizon extraction (Stark 2004; Lomask et al. 2006; Wu & Hale 2015; Dou et al. RGT-Est)
unwraps tightly folded strata without horizon crossing by solving for one monotonic Relative
Geologic Time field ``Phi(z, y, x)`` whose iso-surfaces follow the local dip. Here the strata are
papyrus wraps 4-10 voxels apart, where single-sheet growers fragment or jump between wraps:

1. :func:`estimate_structure_tensor_dip_3d` - Gaussian-derivative structure tensor, dominant
   normal ``n`` (largest eigenvalue), planarity ``(l1 - l2) / l1`` and the dominant normal axis.
2. :func:`estimate_local_wrap_wavenumber_3d` - local wrap spacing ``Lambda`` from a windowed,
   normalized autocorrelation of the band-passed signal along the normal axis; ``k = 2 pi / Lambda``.
3. :func:`solve_dip_guided_rgt_3d` - sparse weighted least squares on the voxel grid,
   ``sum w ||grad Phi - k n||^2 + lambda sum ||grad_perp Phi||^2`` with the in-sheet (tangential)
   part weighted by ``w + lambda`` and the along-normal scale part by ``scale_weight * w``, solved
   with Jacobi-preconditioned CG from the ramp ``Phi_0 = mean(k) * x_axis``, then projected onto
   strictly increasing columns along the normal axis (minimum step ``eps_phi``).
4. :func:`wheeler_flatten_and_extract_sheets` - Wheeler volume ``V(phi_b, u, v)`` by inverse
   interpolation of each monotonic column, sheet phases from peaks of the lateral-mean ridge
   profile, iso-phase heights refined to the nearest ridge crest inside half the iso-phase gap to
   each neighbour (so order, hence zero crossing, is preserved), render + ``onsheet_score``.
5. :func:`compare_rgt_vs_single_sheet_baseline` - RGT versus ``grow_m7_patch.extract_height_field``
   (single-crossing columns + local-median smoothness) on the same in-RAM box.

``python seismic_rgt_unwrapper.py --evidence-out ...`` runs the live preflight: one public control
box (PHerc0306B around the public w074 mesh) and one crushed eligible-unread box (PHerc1545,
fallback PHerc0846A), all HTTP bytes under one 80 MiB :class:`ByteBudget`, chunks only in RAM.
The public receipt for unread scrolls carries aggregate metrics only (no coordinates, chunk keys,
seeds or local paths), enforced by :func:`validate_public_receipt_privacy`.

No GPU library is imported. Geometry/coherence preflight only: no ink detector is run.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import requests
from scipy import ndimage as ndi
from scipy import signal as sps
from scipy import sparse
from scipy.sparse import linalg as spla
import tifffile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stream_snap_render as SSR  # noqa: E402
import grow_m7_patch as GM  # noqa: E402
from grow_m7_patch import BudgetExceeded, BudgetedRemoteZarrV2, ByteBudget, SCROLL_SPECS  # noqa: E402,F401

S3_BUCKET = GM.S3_BUCKET
GLOBAL_HTTP_CAP_BYTES = 80 * 1024 * 1024
TWO_PI = 2.0 * math.pi

PHERC0306B_CT_ZARR_URL = (
    f"{S3_BUCKET}/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr"
)
PHERC0306B_MESH_URL = (
    "https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/"
    "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/"
    "PHerc0306B_20250521133212_w074own_touchup_V5Rc"
)
PHERC0306B_VOXEL_UM = 8.64
BOX_SHAPE_ZYX = (64, 96, 96)
UNREAD_SCROLLS = frozenset({"PHerc0846A", "PHerc1545"})

FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])[A-Za-z]:\\"),
    re.compile(r"(?<![A-Za-z])[A-Za-z]:/"),
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"\\Users\\[^\\]+\\", re.I),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/gpfs/", re.I),
    re.compile(r"/data/visitor/", re.I),
    re.compile(r"file://", re.I),
    re.compile(r"Scrollprize_private", re.I),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

FORBIDDEN_COORD_KEYS = frozenset({
    "chunk_indices", "chunk_index", "chunk_key", "chunk_keys", "chunks_touched", "probed_indices",
    "probed_chunks", "anchor_indices", "coords", "coordinates", "voxel_coords", "voxel_bounds",
    "bbox", "bbox_zyx", "box_lo", "box_hi", "box_lo_zyx", "box_hi_zyx", "lo_zyx", "hi_zyx",
    "origin", "origin_zyx", "offset_zyx", "center", "centre", "center_zyx", "centre_zyx",
    "seed", "seeds", "seed_zyx", "seed_xyz", "position", "z0", "y0", "x0", "z1", "y1", "x1",
    "cz", "cy", "cx", "z_idx", "y_idx", "x_idx", "direction", "radius_fraction", "z_fraction",
})

# Chunk keys such as ``12.3.4`` or ``0/12/3/4``. Applied to unread-scroll sections only.
CHUNK_KEY_PATTERN = re.compile(r"(?<![\w.])\d+[./]\d+[./]\d+(?![\w.])")

PUBLIC_RGT_METRIC_KEYS = frozenset({
    "box_shape_zyx", "voxel_um", "normal_axis", "sheet_count_rgt", "sheet_count_single_baseline",
    "crossing_fraction_rgt", "usable_cm2_rgt_total", "usable_cm2_best_sheet_rgt",
    "usable_cm2_single_sheet_baseline", "area_gain_ratio_vs_baseline",
    "largest_connected_sheet_frac_rgt", "median_f_parallel_rgt", "per_sheet_f_parallel_rgt",
    "baseline_f_parallel", "mean_wrap_spacing_vx", "median_wrap_spacing_vx", "planarity_median",
    "wheeler_peak_contrast", "wheeler_slice_variance_ratio", "cartesian_slice_variance_ratio",
    "cg_iterations", "cg_converged", "phase_normal_coherence", "monotonic_projection_fraction",
    "baseline_patch_columns", "active_fraction_m7", "ct_nonzero_fraction",
    "runtime_seconds", "http_bytes", "http_mib",
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return (v - float(v.mean())) / (float(v.std()) + 1e-6)


def _bandpass_along(v: np.ndarray, axis: int, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    """Difference of Gaussians along ``axis`` only (sheet ridges positive)."""
    return (ndi.gaussian_filter1d(v, sigma_lo, axis=axis, mode="nearest")
            - ndi.gaussian_filter1d(v, sigma_hi, axis=axis, mode="nearest"))


def _largest_component_2d(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _drop_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= int(min_px)
    keep[0] = False
    return keep[lab]


def _round(v: Any, nd: int = 6) -> Any:
    if v is None:
        return None
    f = float(v)
    return round(f, nd) if math.isfinite(f) else None


class InRamVolume:
    """Read-only in-RAM CT box with the ``sample_trilinear`` contract of ``RemoteZarrV2``.

    A sample is supported iff it is finite and inside the box (all trilinear corners exist).
    """

    def __init__(self, vol_zyx: np.ndarray) -> None:
        src = np.asarray(vol_zyx)
        self.dtype = np.dtype(np.uint8) if src.dtype == np.uint8 else np.dtype(np.float32)
        self.vol = np.ascontiguousarray(src, dtype=np.float32)
        self.shape = tuple(int(s) for s in self.vol.shape)
        self.fill_value = 0

    def sample_trilinear(self, xyz: np.ndarray, *, return_support: bool = False, **_ignored):
        pts = np.asarray(xyz, dtype=np.float64)
        flat = pts.reshape(-1, 3)
        zyx = flat[:, ::-1]
        sz, sy, sx = self.shape
        with np.errstate(invalid="ignore"):
            ok = (np.isfinite(zyx).all(axis=1)
                  & (zyx[:, 0] >= 0) & (zyx[:, 0] <= sz - 1)
                  & (zyx[:, 1] >= 0) & (zyx[:, 1] <= sy - 1)
                  & (zyx[:, 2] >= 0) & (zyx[:, 2] <= sx - 1))
        vals = np.zeros(flat.shape[0], dtype=np.float32)
        if ok.any():
            vals[ok] = ndi.map_coordinates(self.vol, zyx[ok].T, order=1, mode="nearest")
        vals = vals.reshape(pts.shape[:-1])
        return (vals, ok.reshape(pts.shape[:-1])) if return_support else vals


# ---------------------------------------------------------------------------
# 1. Structure-tensor dip field
# ---------------------------------------------------------------------------

def estimate_structure_tensor_dip_3d(
    vol_zyx: np.ndarray, *, sigma_grad: float = 1.0, sigma_tensor: float = 2.5
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(normals_zyx3, planarity_zyx, normal_axis)``.

    ``normals_zyx3[..., a]`` is the zyx component ``a`` of the unit eigenvector of the largest
    eigenvalue of ``J = G_sigma_tensor * (grad grad^T)``; it is sign-aligned so the component on
    ``normal_axis`` (the axis with maximal mean ``|n_a|``) is non-negative everywhere.
    """
    v = _zscore(vol_zyx)
    if v.ndim != 3:
        raise ValueError("vol_zyx must be 3D")
    g = [ndi.gaussian_filter(v, sigma_grad, order=o, mode="nearest")
         for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    J = np.empty(v.shape + (3, 3), dtype=np.float32)
    for a in range(3):
        for b in range(a, 3):
            J[..., a, b] = ndi.gaussian_filter(g[a] * g[b], sigma_tensor, mode="nearest")
            J[..., b, a] = J[..., a, b]
    del g
    w, vec = np.linalg.eigh(J)
    del J
    l1 = np.maximum(w[..., 2], 0.0)
    l2 = np.maximum(w[..., 1], 0.0)
    planarity = np.clip((l1 - l2) / (l1 + 1e-6), 0.0, 1.0).astype(np.float32)
    normals = np.ascontiguousarray(vec[..., :, 2], dtype=np.float32)
    axis = int(np.argmax(np.abs(normals).reshape(-1, 3).mean(axis=0)))
    sign = np.where(normals[..., axis] < 0.0, -1.0, 1.0).astype(np.float32)
    normals *= sign[..., None]
    return normals, planarity, axis


# ---------------------------------------------------------------------------
# 2. Local wrap wavenumber
# ---------------------------------------------------------------------------

def estimate_local_wrap_wavenumber_3d(
    vol_zyx: np.ndarray,
    normal_axis: int,
    *,
    min_spacing_vx: float = 4.0,
    max_spacing_vx: float = 20.0,
    lateral_sigma: float = 6.0,
    normal_sigma: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Local wavenumber ``k = 2 pi / Lambda`` (rad / voxel) and spacing statistics.

    ``Lambda`` is the first strong peak (>= 0.75 x the best) of the windowed normalized
    autocorrelation ``R(L)`` of the band-passed signal along ``normal_axis`` for integer lags in
    ``[min_spacing_vx, max_spacing_vx]``, refined by a parabola. Weak-correlation voxels take
    the confidence-weighted global median before a light smoothing.
    """
    if not 0 < min_spacing_vx < max_spacing_vx:
        raise ValueError("need 0 < min_spacing_vx < max_spacing_vx")
    v = np.moveaxis(_zscore(vol_zyx), int(normal_axis), 0)
    bp = _bandpass_along(v, 0, 0.7, max(2.0, 0.5 * float(max_spacing_vx)))
    bp = ndi.gaussian_filter(bp, (0.0, 1.0, 1.0), mode="nearest")
    depth = bp.shape[0]
    lags = np.arange(int(math.floor(min_spacing_vx)), int(math.ceil(max_spacing_vx)) + 1)
    lags = lags[(lags >= 1) & (lags < depth - 1)]
    if lags.size < 3:
        raise ValueError("box too thin along the normal axis for the requested spacing range")
    ns = float(normal_sigma) if normal_sigma else max(2.0 * float(min_spacing_vx), 0.6 * float(max_spacing_vx))
    win = (ns, float(lateral_sigma), float(lateral_sigma))
    R = np.empty((lags.size,) + bp.shape, dtype=np.float32)
    sq = bp * bp
    for j, L in enumerate(lags):
        prod = np.zeros_like(bp)
        en = np.zeros_like(bp)
        c = int(L) // 2
        prod[c:depth - L + c] = bp[:-L] * bp[L:]
        en[c:depth - L + c] = 0.5 * (sq[:-L] + sq[L:])
        R[j] = ndi.gaussian_filter(prod, win, mode="constant") / (
            ndi.gaussian_filter(en, win, mode="constant") + 1e-6)
    rmax = R.max(axis=0)
    peak = np.zeros_like(R, dtype=bool)
    peak[1:-1] = (R[1:-1] >= R[:-2]) & (R[1:-1] >= R[2:])
    cand = peak & (R >= 0.75 * rmax[None]) & (R > 0.05)
    has = cand.any(axis=0)
    j = np.where(has, np.argmax(cand, axis=0), np.argmax(R, axis=0))
    jc = np.clip(j, 1, lags.size - 2)
    r0 = np.take_along_axis(R, (jc - 1)[None], 0)[0]
    r1 = np.take_along_axis(R, jc[None], 0)[0]
    r2 = np.take_along_axis(R, (jc + 1)[None], 0)[0]
    den = r0 - 2.0 * r1 + r2
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(den < -1e-9, 0.5 * (r0 - r2) / den, 0.0)
    delta = np.where(j == jc, np.clip(delta, -0.5, 0.5), 0.0)
    lam = np.clip(lags[j].astype(np.float32) + delta, min_spacing_vx, max_spacing_vx)
    conf = np.clip(rmax, 0.0, 1.0) * has
    good = conf > 0.2
    if good.any():
        wts = conf[good].astype(np.float64)
        vals = lam[good].astype(np.float64)
        order = np.argsort(vals)
        cw = np.cumsum(wts[order])
        med = float(vals[order][np.searchsorted(cw, 0.5 * cw[-1])])
        mean = float(np.average(vals, weights=wts))
    else:
        med = mean = 0.5 * (float(min_spacing_vx) + float(max_spacing_vx))
    lam = np.where(good, lam, med)
    wsm = ndi.gaussian_filter(conf + 1e-3, 3.0, mode="nearest")
    lam = ndi.gaussian_filter(lam * (conf + 1e-3), 3.0, mode="nearest") / wsm
    lam = np.clip(lam, min_spacing_vx, max_spacing_vx).astype(np.float32)
    k = np.moveaxis((TWO_PI / lam).astype(np.float32), 0, int(normal_axis))
    info = {
        "mean_wrap_spacing_vx": mean,
        "median_wrap_spacing_vx": med,
        "confident_fraction": float(good.mean()),
        "lags_vx": [int(lags[0]), int(lags[-1])],
    }
    return np.ascontiguousarray(k), info


# ---------------------------------------------------------------------------
# 3. Dip-guided RGT solve
# ---------------------------------------------------------------------------

def _forward_diff_ops(shape: Sequence[int]) -> tuple[list[sparse.csr_matrix], np.ndarray]:
    """Forward differences per axis (empty rows on each far face) and the all-interior mask."""
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.int64).reshape(shape)
    ops = []
    interior = np.ones(shape, dtype=bool)
    for a in range(3):
        sl = [slice(None)] * 3
        sl[a] = slice(0, shape[a] - 1)
        rows = idx[tuple(sl)].ravel()
        stride = int(np.prod(shape[a + 1:]))
        r = np.concatenate([rows, rows])
        c = np.concatenate([rows, rows + stride])
        d = np.concatenate([-np.ones(rows.size), np.ones(rows.size)])
        ops.append(sparse.csr_matrix((d, (r, c)), shape=(n, n)))
        last = [slice(None)] * 3
        last[a] = shape[a] - 1
        interior[tuple(last)] = False
    return ops, interior


def ridge_crest_mask(vol_zyx: np.ndarray, normal_axis: int, *, sigma_lo: float = 0.8, sigma_hi: float = 3.0) -> np.ndarray:
    """Voxels that are positive local maxima of the band-passed signal along ``normal_axis``."""
    bp = np.moveaxis(_bandpass_along(_zscore(vol_zyx), int(normal_axis), sigma_lo, sigma_hi), int(normal_axis), 0)
    crest = np.zeros(bp.shape, dtype=bool)
    thr = 0.5 * float(bp.std())
    crest[1:-1] = (bp[1:-1] >= bp[:-2]) & (bp[1:-1] > bp[2:]) & (bp[1:-1] > thr)
    return np.moveaxis(crest, 0, int(normal_axis))


def solve_dip_guided_rgt_3d(
    vol_zyx: np.ndarray,
    normals_zyx3: np.ndarray,
    planarity_zyx: np.ndarray,
    k_zyx: np.ndarray,
    *,
    normal_axis: int = 0,
    lambda_smooth: float = 0.15,
    ridge_weight: float = 0.25,
    max_iter: int = 250,
    tol: float = 1e-5,
    scale_weight: float = 0.2,
    eps_phi: float = 1e-3,
) -> dict[str, Any]:
    """Solve for a strictly monotonic RGT phase ``Phi`` (radians) aligned with the dip field.

    Returns ``phi`` (float64, box shape), ``phasor_cos`` / ``phasor_sin`` (float32) and ``info``.
    """
    shape = tuple(int(s) for s in np.shape(vol_zyx))
    if np.shape(normals_zyx3) != shape + (3,) or np.shape(planarity_zyx) != shape or np.shape(k_zyx) != shape:
        raise ValueError("normals, planarity and k must match the volume shape")
    axis = int(normal_axis)
    n_vox = int(np.prod(shape))
    nrm = np.asarray(normals_zyx3, dtype=np.float64).reshape(n_vox, 3)
    k = np.asarray(k_zyx, dtype=np.float64).ravel()
    w = np.clip(np.asarray(planarity_zyx, dtype=np.float64) ** 2, 0.05, 1.0)
    if ridge_weight > 0:
        w = w * (1.0 + float(ridge_weight) * ridge_crest_mask(vol_zyx, axis))
    ops, interior = _forward_diff_ops(shape)
    inter = interior.ravel().astype(np.float64)
    w = w.ravel() * inter
    wt = (w + float(lambda_smooth) * inter)
    wn = float(scale_weight) * w

    rn = sum(sparse.diags(nrm[:, a]) @ ops[a] for a in range(3)).tocsr()
    M = (rn.T @ sparse.diags(wn) @ rn).tocsr()
    for a in range(3):
        t = (ops[a] - sparse.diags(nrm[:, a]) @ rn).tocsr()
        M = M + (t.T @ sparse.diags(wt) @ t)
    M = M.tocsr()
    k_mean = float(np.average(k, weights=w + 1e-9))
    coord = np.indices(shape, dtype=np.float64)[axis].ravel()
    phi0 = k_mean * coord
    diag = M.diagonal()
    gauge = 1e-4 * float(diag.mean())
    M = M + sparse.identity(n_vox, format="csr") * gauge
    rhs = rn.T @ (wn * k) + gauge * phi0
    dinv = 1.0 / (diag + gauge)
    pre = spla.LinearOperator(M.shape, matvec=lambda x: dinv * x, dtype=np.float64)
    iters = [0]

    def _cb(_xk):
        iters[0] += 1

    phi, cg_info = spla.cg(M, rhs, x0=phi0, rtol=float(tol), maxiter=int(max_iter), M=pre, callback=_cb)
    del M, rn, ops

    p = np.moveaxis(phi.reshape(shape), axis, 0)
    depth = p.shape[0]
    ramp = float(eps_phi) * np.arange(depth, dtype=np.float64).reshape(-1, 1, 1)
    proj = np.maximum.accumulate(p - ramp, axis=0) + ramp
    moved = float(np.mean(proj > p + 1e-12))
    phi = np.ascontiguousarray(np.moveaxis(proj, 0, axis))

    grad = np.stack(np.gradient(phi), axis=-1).reshape(n_vox, 3)
    gnorm = np.linalg.norm(grad, axis=1)
    coh = np.abs((grad * nrm).sum(axis=1)) / (gnorm + 1e-9)
    resid = np.linalg.norm(grad - k[:, None] * nrm, axis=1) / (k + 1e-9)
    wsel = w > 0
    info = {
        "cg_iterations": int(iters[0]),
        "cg_converged": bool(cg_info == 0),
        "k_mean": k_mean,
        "phase_normal_coherence": float(np.average(coh[wsel], weights=w[wsel])) if wsel.any() else 0.0,
        "phase_gradient_residual_median": float(np.median(resid[wsel])) if wsel.any() else 0.0,
        "monotonic_projection_fraction": moved,
        "eps_phi": float(eps_phi),
    }
    return {
        "phi": phi,
        "phasor_cos": np.cos(phi).astype(np.float32),
        "phasor_sin": np.sin(phi).astype(np.float32),
        "info": info,
    }


# ---------------------------------------------------------------------------
# 4. Wheeler flattening and multi-sheet extraction
# ---------------------------------------------------------------------------

def _heights_at_phase(p_dab: np.ndarray, level) -> np.ndarray:
    """Sub-voxel depth where each strictly increasing column of ``p_dab`` reaches ``level``."""
    depth = p_dab.shape[0]
    lvl = np.broadcast_to(np.asarray(level, dtype=np.float64), p_dab.shape[1:])
    cnt = (p_dab < lvl[None]).sum(axis=0)
    ok = (cnt >= 1) & (cnt <= depth - 1)
    i = np.clip(cnt, 1, depth - 1)
    p0 = np.take_along_axis(p_dab, (i - 1)[None], 0)[0]
    p1 = np.take_along_axis(p_dab, i[None], 0)[0]
    f = (lvl - p0) / np.maximum(p1 - p0, 1e-12)
    z = (i - 1) + np.clip(f, 0.0, 1.0)
    return np.where(ok, z, np.nan)


def _sample_along(v_dab: np.ndarray, z_ab: np.ndarray) -> np.ndarray:
    """Linear sample of ``v_dab`` at depth ``z_ab`` per column (NaN where ``z_ab`` is NaN)."""
    depth = v_dab.shape[0]
    fin = np.isfinite(z_ab)
    zz = np.where(fin, z_ab, 0.0)
    i0 = np.clip(np.floor(zz).astype(np.int64), 0, depth - 2)
    f = np.clip(zz - i0, 0.0, 1.0)
    a = np.take_along_axis(v_dab, i0[None], 0)[0]
    b = np.take_along_axis(v_dab, (i0 + 1)[None], 0)[0]
    return np.where(fin, a * (1.0 - f) + b * f, np.nan)


def height_to_tifxyz(height_ab: np.ndarray, valid_ab: np.ndarray, axis: int):
    """Box-frame ``(x, y, z)`` grid (one vertex per lateral voxel, ``-1`` sentinel when invalid)."""
    ha, hb = height_ab.shape
    ga, gb = np.meshgrid(np.arange(ha, dtype=np.float64), np.arange(hb, dtype=np.float64), indexing="ij")
    coords = np.zeros((ha, hb, 3), dtype=np.float64)
    others = [a for a in range(3) if a != axis]
    coords[..., axis] = np.where(valid_ab, height_ab, 0.0)
    coords[..., others[0]] = ga
    coords[..., others[1]] = gb
    x = np.where(valid_ab, coords[..., 2], -1.0)
    y = np.where(valid_ab, coords[..., 1], -1.0)
    z = np.where(valid_ab, coords[..., 0], -1.0)
    cx, cy, cz, v = SSR.validate_tifxyz_channels(x, y, z)
    return cx, cy, cz, v


def _onsheet_f_parallel(stack: np.ndarray, valid: np.ndarray, *, tile: int = 16, step: int = 8, dk: int = 2) -> dict[str, Any]:
    """``onsheet_score`` with tiles sized for <= 96-pixel renders (fallback: layer diagnostic)."""
    masked = np.array(stack, copy=True)
    masked[:, ~np.asarray(valid, dtype=bool)] = 0
    mid = (masked.shape[0] - 1) // 2
    mod = GM._load_onsheet_module()
    if mod is not None:
        res = dict(mod.score_stack(masked, mid=mid, dk=dk, tile=tile, step=step, hp_sigma=2.0))
        return {"f_parallel": res.get("f_parallel") if res.get("tiles") else None,
                "tiles": int(res.get("tiles") or 0), "backend": "onsheet_score"}
    fb = SSR.layer_coherence_score(masked, mesh_index=mid, delta=dk, valid_mask=valid)
    return {"f_parallel": fb["zero_shift_dominant_fraction"] if fb["patch_count"] else None,
            "tiles": int(fb["patch_count"]), "backend": "stream_snap_render_fallback"}


def score_height_field(
    height_ab: np.ndarray,
    valid_ab: np.ndarray,
    ct_zyx: np.ndarray,
    axis: int,
    *,
    voxel_um: float,
    num_slices: int = 9,
) -> dict[str, Any]:
    """Render an in-RAM CT surface stack for one sheet and report usable area and ``f_parallel``."""
    valid_ab = np.asarray(valid_ab, dtype=bool) & np.isfinite(height_ab)
    px_cm2 = (float(voxel_um) * 1e-4) ** 2
    if valid_ab.sum() < 4:
        return {"usable_px": 0, "usable_cm2": 0.0, "f_parallel": None, "tiles": 0,
                "largest_component_px": 0, "render_valid": np.zeros_like(valid_ab)}
    x, y, z, _ = height_to_tifxyz(height_ab, valid_ab, axis)
    stack, vct = SSR.render_surface_stack(x, y, z, InRamVolume(ct_zyx), num_slices=num_slices,
                                          slice_step=1.0, scale=1.0)
    coh = _onsheet_f_parallel(stack, vct)
    return {
        "usable_px": int(vct.sum()),
        "usable_cm2": float(vct.sum()) * px_cm2,
        "f_parallel": coh["f_parallel"],
        "tiles": coh["tiles"],
        "onsheet_backend": coh.get("backend"),
        "largest_component_px": int(_largest_component_2d(vct).sum()),
        "render_valid": vct,
    }


def wheeler_flatten_and_extract_sheets(
    rgt_phi: np.ndarray,
    ct_zyx: np.ndarray,
    m7_zyx: np.ndarray | None = None,
    *,
    normal_axis: int = 0,
    n_wheeler_bins: int = 128,
    min_peak_prominence: float = 0.08,
    min_wrap_spacing_vx: float = 4.0,
    voxel_um: float = 9.362,
    planarity_zyx: np.ndarray | None = None,
    min_planarity: float = 0.15,
    num_slices: int = 9,
    refine_radius_vx: float | None = None,
    min_component_px: int = 16,
    min_bin_coverage: float = 0.5,
    min_sheet_fraction: float = 0.1,
    face_margin_vx: float = 2.0,
    score: bool = True,
) -> dict[str, Any]:
    """Flatten into the Wheeler domain, pick sheet phases, extract ordered non-crossing sheets.

    Peaks whose mean depth is within ``face_margin_vx`` of a box face (band-pass edge artefacts)
    are ignored; sheets with fewer than ``min_sheet_fraction`` valid columns are not reported
    (the crossing check still covers every candidate sheet).
    """
    axis = int(normal_axis)
    p = np.moveaxis(np.asarray(rgt_phi, dtype=np.float64), axis, 0)
    if np.any(np.diff(p, axis=0) <= 0):
        raise ValueError("rgt_phi must be strictly increasing along normal_axis")
    ct = np.moveaxis(np.asarray(ct_zyx, dtype=np.float32), axis, 0)
    depth, a_dim, b_dim = p.shape
    ridge = _bandpass_along(_zscore(ct), 0, 0.8, 3.0)
    if m7_zyx is not None:
        m7 = np.moveaxis(np.asarray(m7_zyx, dtype=np.float32), axis, 0)
        if m7.std() > 0:
            ridge = ridge / (ridge.std() + 1e-6) + _zscore(ndi.gaussian_filter(m7, 0.7))
    ridge = ridge.astype(np.float64)

    levels = np.linspace(float(p.min()), float(p.max()), int(n_wheeler_bins))
    z_lv = np.stack([_heights_at_phase(p, lv) for lv in levels])
    wheeler_ct = np.stack([_sample_along(ct.astype(np.float64), z) for z in z_lv]).astype(np.float32)
    w_ridge = np.stack([_sample_along(ridge, z) for z in z_lv])
    coverage = np.isfinite(z_lv).mean(axis=(1, 2))
    good_bin = coverage >= float(min_bin_coverage)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.where(good_bin, np.nanmean(w_ridge.reshape(len(levels), -1), axis=1), np.nan)
        zbar = np.nanmean(z_lv.reshape(len(levels), -1), axis=1)
        ct_prof = np.nanmean(wheeler_ct.reshape(len(levels), -1), axis=1)

    out: dict[str, Any] = {"phase_levels": levels, "wheeler_profile": prof, "wheeler_ct": wheeler_ct,
                           "wheeler_coverage": coverage, "sheets": []}
    finite = np.isfinite(prof)
    if finite.sum() < 3:
        out.update(sheet_count=0, crossing_fraction=0.0, wheeler_peak_contrast=0.0, wheeler_peak_phases=[])
        return out
    fi = np.flatnonzero(finite)
    pf = np.interp(np.arange(prof.size), fi, prof[fi])
    span = float(np.nanmax(prof) - np.nanmin(prof))
    pk, props = sps.find_peaks(pf, prominence=float(min_peak_prominence) * max(span, 1e-9))
    keep = finite[pk] & (pf[pk] > 0)
    pk, prom = pk[keep], props["prominences"][keep]
    inside = (zbar[pk] >= float(face_margin_vx)) & (zbar[pk] <= depth - 1 - float(face_margin_vx))
    pk, prom = pk[inside], prom[inside]
    chosen: list[int] = []
    for i in pk[np.argsort(-prom, kind="stable")]:
        if all(abs(zbar[i] - zbar[j]) >= float(min_wrap_spacing_vx) for j in chosen):
            chosen.append(int(i))
    chosen.sort()
    peak_phases = [float(levels[i]) for i in chosen]

    # Michelson contrast of the CT Wheeler profile between chosen peaks and the troughs between them.
    contrast = 0.0
    if len(chosen) >= 2:
        troughs = [int(chosen[i] + np.nanargmin(ct_prof[chosen[i]:chosen[i + 1] + 1])) for i in range(len(chosen) - 1)]
        pk_v = float(np.nanmean(ct_prof[chosen]))
        tr_v = float(np.nanmean(ct_prof[troughs]))
        contrast = (pk_v - tr_v) / (abs(pk_v) + abs(tr_v) + 1e-9)
    out["wheeler_peak_contrast"] = float(contrast)
    out["wheeler_peak_phases"] = peak_phases

    # Wheeler-vs-Cartesian lateral variance of horizontal slices (normalized by total variance).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full_bins = coverage >= 0.95
        wv = np.nanvar(wheeler_ct[full_bins].reshape(int(full_bins.sum()), -1), axis=1) if full_bins.any() else np.array([np.nan])
        out["wheeler_slice_variance_ratio"] = float(np.nanmean(wv) / (np.nanvar(wheeler_ct[full_bins]) + 1e-9)) if full_bins.any() else None
        cv = ct.reshape(depth, -1).var(axis=1)
        out["cartesian_slice_variance_ratio"] = float(cv.mean() / (ct.var() + 1e-9))

    if not chosen:
        out.update(sheet_count=0, crossing_fraction=0.0)
        return out

    h_iso = np.stack([_heights_at_phase(p, lv) for lv in peak_phases])
    m = len(peak_phases)
    inf = np.full((1, a_dim, b_dim), np.inf)
    gaps = np.nan_to_num(np.diff(h_iso, axis=0), nan=np.inf)
    gap_below = np.concatenate([inf, gaps])
    gap_above = np.concatenate([gaps, inf])
    r = float(refine_radius_vx) if refine_radius_vx else max(1.5, 0.4 * float(min_wrap_spacing_vx))
    delta = 0.05
    lo_shift = np.minimum(r, np.maximum(0.0, 0.5 * gap_below - delta))
    hi_shift = np.minimum(r, np.maximum(0.0, 0.5 * gap_above - delta))

    coeffs = ndi.spline_filter(ridge, order=3, mode="nearest")
    offs = np.arange(-r, r + 1e-9, 0.25)
    aa, bb = np.meshgrid(np.arange(a_dim), np.arange(b_dim), indexing="ij")
    plan = np.moveaxis(np.asarray(planarity_zyx, dtype=np.float32), axis, 0) if planarity_zyx is not None else None
    heights, valids = [], []
    for s in range(m):
        h0 = h_iso[s]
        fin = np.isfinite(h0)
        zq = np.where(fin, h0, 0.0)[None] + offs[:, None, None]
        allowed = (offs[:, None, None] >= -lo_shift[s][None]) & (offs[:, None, None] <= hi_shift[s][None])
        allowed &= (zq >= 0) & (zq <= depth - 1)
        sig = ndi.map_coordinates(
            coeffs, [np.clip(zq, 0, depth - 1).ravel(), np.broadcast_to(aa, zq.shape).ravel(),
                     np.broadcast_to(bb, zq.shape).ravel()], order=3, prefilter=False, mode="nearest",
        ).reshape(zq.shape)
        sig = np.where(allowed, sig, -np.inf)
        j = np.argmax(sig, axis=0)
        s1 = np.take_along_axis(sig, j[None], 0)[0]
        s0 = np.take_along_axis(sig, np.clip(j - 1, 0, len(offs) - 1)[None], 0)[0]
        s2 = np.take_along_axis(sig, np.clip(j + 1, 0, len(offs) - 1)[None], 0)[0]
        interior_peak = np.isfinite(s0) & np.isfinite(s2) & (j > 0) & (j < len(offs) - 1) & (s1 >= s0) & (s1 >= s2)
        den = s0 - 2.0 * s1 + s2
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(interior_peak & (den < -1e-12), 0.5 * (s0 - s2) / den, 0.0)
        shift = offs[j] + np.clip(np.nan_to_num(frac), -0.5, 0.5) * 0.25
        shift = np.clip(shift, -lo_shift[s], hi_shift[s])
        h = np.where(fin, h0 + shift, np.nan)
        strength = np.where(fin, _sample_along(ridge, h), np.nan)
        ok = fin & interior_peak & (strength > 0)
        if ok.any():
            ok &= strength >= 0.3 * float(np.median(strength[ok]))
        if plan is not None and ok.any():
            ok &= _sample_along(plan.astype(np.float64), h) >= float(min_planarity)
        ok = _drop_small_components(ok, min_component_px)
        heights.append(h)
        valids.append(ok)

    cross, joint = 0, 0
    for s in range(m - 1):
        jv = valids[s] & valids[s + 1]
        joint += int(jv.sum())
        cross += int((heights[s + 1][jv] <= heights[s][jv]).sum())
    out["crossing_fraction"] = float(cross / joint) if joint else 0.0
    out["crossing_pairs_checked"] = int(joint)
    for s in range(m):
        if valids[s].mean() < float(min_sheet_fraction):
            continue
        sheet = {"phase": peak_phases[s], "height": heights[s], "valid": valids[s],
                 "mean_height_vx": float(np.nanmean(heights[s][valids[s]])),
                 "valid_columns": int(valids[s].sum())}
        if score:
            sc = score_height_field(heights[s], valids[s], ct_zyx, axis, voxel_um=voxel_um, num_slices=num_slices)
            sheet.update(sc)
        out["sheets"].append(sheet)
    out["sheet_count"] = len(out["sheets"])
    return out


# ---------------------------------------------------------------------------
# 5. Head-to-head comparison against the single-sheet grower
# ---------------------------------------------------------------------------

def ridge_binary_proxy(ct_zyx: np.ndarray, normal_axis: int, *, frac: float = 0.35) -> np.ndarray:
    """Thresholded band-passed CT ridges: a binary m7 stand-in for CT-only boxes."""
    bp = _bandpass_along(_zscore(ct_zyx), int(normal_axis), 0.8, 3.0)
    return bp > float(frac) * float(np.percentile(bp, 99))


def compare_rgt_vs_single_sheet_baseline(
    ct_zyx: np.ndarray,
    m7_zyx: np.ndarray | None = None,
    *,
    voxel_um: float = 9.362,
    min_spacing_vx: float = 4.0,
    max_spacing_vx: float = 18.0,
    m7_threshold: float = 128.0,
    num_slices: int = 9,
    solve_kw: dict[str, Any] | None = None,
    wheeler_kw: dict[str, Any] | None = None,
    return_details: bool = False,
):
    """RGT multi-sheet unwrapper vs ``grow_m7_patch.extract_height_field`` on one in-RAM box."""
    t0 = time.perf_counter()
    ct = np.asarray(ct_zyx)
    m7 = None if m7_zyx is None else np.asarray(m7_zyx)
    guide = _zscore(ct)
    if m7 is not None and float(np.std(m7)) > 0:
        guide = guide + _zscore(ndi.gaussian_filter(m7.astype(np.float32), 0.7))
    normals, plan, axis = estimate_structure_tensor_dip_3d(guide)
    k, kinfo = estimate_local_wrap_wavenumber_3d(guide, axis, min_spacing_vx=min_spacing_vx, max_spacing_vx=max_spacing_vx)
    rgt = solve_dip_guided_rgt_3d(guide, normals, plan, k, normal_axis=axis, **(solve_kw or {}))
    med_sp = float(kinfo["median_wrap_spacing_vx"])
    wk = dict(normal_axis=axis, min_wrap_spacing_vx=max(float(min_spacing_vx), 0.6 * med_sp),
              refine_radius_vx=max(1.5, 0.3 * med_sp), voxel_um=voxel_um,
              planarity_zyx=plan, num_slices=num_slices)
    wk.update(wheeler_kw or {})
    wh = wheeler_flatten_and_extract_sheets(rgt["phi"], ct, m7, **wk)

    binary = (m7 >= m7_threshold) if m7 is not None else ridge_binary_proxy(ct, axis)
    h_base, _, bmet = GM.extract_height_field(binary, height_axis=axis)
    bvalid = np.isfinite(h_base)
    base = score_height_field(h_base, bvalid, ct, axis, voxel_um=voxel_um, num_slices=num_slices)

    sheets = wh["sheets"]
    areas = [float(s.get("usable_cm2", 0.0)) for s in sheets]
    fpar = [s.get("f_parallel") for s in sheets]
    fpar_num = [float(f) for f in fpar if f is not None]
    total = float(sum(areas))
    lateral = int(np.prod([s for i, s in enumerate(ct.shape) if i != axis]))
    lcc = max([s.get("largest_component_px", 0) for s in sheets], default=0) / max(1, lateral)
    usable_base = float(base["usable_cm2"])
    metrics = {
        "box_shape_zyx": [int(s) for s in ct.shape],
        "voxel_um": float(voxel_um),
        "normal_axis": "zyx"[axis],
        "sheet_count_rgt": int(len(sheets)),
        "sheet_count_single_baseline": int(1 if base["usable_px"] > 0 else 0),
        "crossing_fraction_rgt": float(wh.get("crossing_fraction", 0.0)),
        "usable_cm2_rgt_total": total,
        "usable_cm2_best_sheet_rgt": float(max(areas, default=0.0)),
        "usable_cm2_single_sheet_baseline": usable_base,
        "area_gain_ratio_vs_baseline": (total / usable_base) if usable_base > 0 else None,
        "largest_connected_sheet_frac_rgt": float(lcc),
        "median_f_parallel_rgt": float(np.median(fpar_num)) if fpar_num else None,
        "per_sheet_f_parallel_rgt": [None if f is None else float(f) for f in fpar],
        "baseline_f_parallel": None if base["f_parallel"] is None else float(base["f_parallel"]),
        "baseline_patch_columns": int(bmet.get("patch_columns", 0)),
        "mean_wrap_spacing_vx": float(kinfo["mean_wrap_spacing_vx"]),
        "median_wrap_spacing_vx": float(kinfo["median_wrap_spacing_vx"]),
        "planarity_median": float(np.median(plan)),
        "wheeler_peak_contrast": float(wh.get("wheeler_peak_contrast", 0.0)),
        "wheeler_slice_variance_ratio": wh.get("wheeler_slice_variance_ratio"),
        "cartesian_slice_variance_ratio": wh.get("cartesian_slice_variance_ratio"),
        "cg_iterations": int(rgt["info"]["cg_iterations"]),
        "cg_converged": bool(rgt["info"]["cg_converged"]),
        "phase_normal_coherence": float(rgt["info"]["phase_normal_coherence"]),
        "monotonic_projection_fraction": float(rgt["info"]["monotonic_projection_fraction"]),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    if not return_details:
        return metrics
    details = {"axis": axis, "normals": normals, "planarity": plan, "k": k, "k_info": kinfo, "rgt": rgt,
               "wheeler": wh, "baseline_height": h_base, "baseline_valid": bvalid,
               "baseline_render_valid": base["render_valid"], "baseline_metrics": bmet, "binary": binary}
    return metrics, details


# ---------------------------------------------------------------------------
# Public receipt privacy
# ---------------------------------------------------------------------------

def _walk_keys(obj: Any, path: str = ""):
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield f"{path}.{k}" if path else str(k), str(k), v
            yield from _walk_keys(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from _walk_keys(item, f"{path}[{i}]")


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_strings(item)


def validate_public_receipt_privacy(receipt: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` on local paths, tokens, coordinate/chunk keys or unread chunk keys.

    Coordinate-like keys are refused anywhere in the receipt. Inside any mapping whose
    ``scroll`` is an eligible unread scroll, strings must not look like chunk keys and the
    ``metrics`` mapping may only hold :data:`PUBLIC_RGT_METRIC_KEYS`.
    """
    text = json.dumps(receipt, sort_keys=True, default=str)
    for pat in FORBIDDEN_PATH_PATTERNS:
        hit = pat.search(text)
        if hit is not None:
            raise ValueError(f"receipt contains forbidden path/secret pattern {hit.group(0)!r}")
    for path, key, _ in _walk_keys(receipt):
        if key.lower() in FORBIDDEN_COORD_KEYS:
            raise ValueError(f"receipt contains forbidden coordinate/chunk key {path}")

    def unread_sections(obj: Any):
        if isinstance(obj, Mapping):
            if str(obj.get("scroll", "")) in UNREAD_SCROLLS:
                yield obj
            for v in obj.values():
                yield from unread_sections(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from unread_sections(v)

    for sec in unread_sections(receipt):
        for s in _walk_strings(sec):
            hit = CHUNK_KEY_PATTERN.search(s)
            if hit is not None:
                raise ValueError(f"unread-scroll section contains a chunk-key-like string {hit.group(0)!r}")
        extra = set((sec.get("metrics") or {}).keys()) - PUBLIC_RGT_METRIC_KEYS
        if extra:
            raise ValueError(f"unread-scroll metrics contain non-public keys {sorted(extra)}")


def sanitize_public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelisted, JSON-safe aggregate metrics."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k not in PUBLIC_RGT_METRIC_KEYS:
            continue
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = _round(v)
        elif isinstance(v, str) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [int(e) if isinstance(e, (int, np.integer)) and not isinstance(e, bool)
                      else (_round(e) if e is not None else None) for e in v]
    return out


# ---------------------------------------------------------------------------
# Live preflight (bounded HTTP, chunks in RAM only)
# ---------------------------------------------------------------------------

def _budgeted_http_get(url: str, budget: ByteBudget, *, timeout: float = 60.0) -> bytes:
    budget.precheck()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    raw = resp.content
    budget.charge(len(raw))
    return raw


def _read_box_budgeted(reader: BudgetedRemoteZarrV2, lo: Sequence[int], hi: Sequence[int], budget: ByteBudget) -> np.ndarray:
    """Read ``[lo, hi)``; uncompressed chunks are pre-checked against the remaining budget."""
    n_chunks = 1
    for l, h, c in zip(lo, hi, reader.chunks):
        n_chunks *= (int(h) - 1) // int(c) - int(l) // int(c) + 1
    if reader._decode is None:
        budget.precheck(n_chunks * reader._expected_chunk_bytes)
    block, _ = reader.read_box(lo, hi)
    return np.asarray(block)


def _mesh_center_zyx(budget: ByteBudget) -> tuple[int, int, int]:
    ch = {}
    for name in ("x", "y", "z"):
        ch[name] = np.asarray(tifffile.imread(io.BytesIO(_budgeted_http_get(f"{PHERC0306B_MESH_URL}/{name}.tif", budget))),
                              dtype=np.float64)
    valid = (ch["x"] >= 0) & (ch["y"] >= 0) & (ch["z"] >= 0) & np.isfinite(ch["x"] + ch["y"] + ch["z"])
    pts = np.stack([ch["z"][valid], ch["y"][valid], ch["x"][valid]], axis=1)
    med = np.median(pts, axis=0)
    best = pts[int(np.argmin(np.linalg.norm(pts - med, axis=1)))]
    return tuple(int(round(v)) for v in best)


def fetch_public_control_box(budget: ByteBudget) -> np.ndarray:
    """PHerc0306B CT level-0 box (``BOX_SHAPE_ZYX``) centred on the public w074 mesh, in RAM."""
    cz, cy, cx = _mesh_center_zyx(budget)
    shp = BOX_SHAPE_ZYX
    lo = (cz - shp[0] // 2, cy - shp[1] // 2, cx - shp[2] // 2)
    hi = tuple(l + s for l, s in zip(lo, shp))
    rd = BudgetedRemoteZarrV2(PHERC0306B_CT_ZARR_URL, "0", budget=budget, workers=4, cache_mb=64)
    try:
        return _read_box_budgeted(rd, lo, hi, budget)
    finally:
        rd.close()


def run_public_control_box(budget: ByteBudget) -> dict[str, Any]:
    t0 = time.perf_counter()
    used0 = budget.used
    ct = fetch_public_control_box(budget)
    metrics = compare_rgt_vs_single_sheet_baseline(ct, None, voxel_um=PHERC0306B_VOXEL_UM)
    metrics["ct_nonzero_fraction"] = float((ct > 0).mean())
    metrics["http_bytes"] = int(budget.used - used0)
    metrics["http_mib"] = metrics["http_bytes"] / 2 ** 20
    metrics["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return {
        "role": "public_control",
        "scroll": "PHerc0306B",
        "ct_dataset": PHERC0306B_CT_ZARR_URL.rsplit("/", 1)[-1],
        "mesh_tag": PHERC0306B_MESH_URL.rsplit("/", 1)[-1],
        "box_rule": "64x96x96 CT level-0 box centred on the w074 vertex nearest the mesh median",
        "baseline_binary": "band-passed CT ridge threshold proxy (no m7)",
        "metrics": sanitize_public_metrics(metrics),
    }


def _column_run_count(binary_zyx: np.ndarray, axis: int) -> float:
    m = np.moveaxis(binary_zyx, axis, 0)
    starts = m[1:] & ~m[:-1]
    n = starts.sum(axis=0) + m[0]
    occ = m.any(axis=0)
    return float(np.median(n[occ])) if occ.any() else 0.0


def select_crushed_unread_box(
    budget: ByteBudget, scroll: str, *, max_probes: int = 16, reserve_bytes: int = 3 * 2 ** 20
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Deterministic interior multi-wrap ``(ct, m7, info)`` box in RAM, or ``None``.

    Radial candidates on the m7 overview mid-slab (density-ranked) are probed as one 384-aligned
    m7-L0 chunk each, so every sub-box lies in one m7 and one CT chunk. Among sub-boxes with
    10-60 % active voxels and >= 3 runs per column, the one with the most runs per column (the
    tightest wraps) is read from CT; a mostly-masked CT box falls through to the next best.
    """
    spec = SCROLL_SPECS[scroll]
    frame_level = int(spec["m7_frame_ct_level"])
    m7_url = f"{S3_BUCKET}/{spec['m7']}"
    ct_url = f"{S3_BUCKET}/{spec['ct']}"
    ov_level = 5
    ov = BudgetedRemoteZarrV2(m7_url, str(ov_level), budget=budget, workers=4, cache_mb=64)
    try:
        zc = ov.shape[0] // 2
        slab, _ = ov.read_box((zc - 1, 0, 0), (zc + 2, ov.shape[1], ov.shape[2]))
    finally:
        ov.close()
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
            iy, ix = int(np.clip(round(py), 0, occ.shape[0] - 1)), int(np.clip(round(px), 0, occ.shape[1] - 1))
            cands.append((-float(dens_map[iy, ix]), rf, dy, dx, py * f, px * f))
    cands.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    align = 384
    m7 = BudgetedRemoteZarrV2(m7_url, "0", budget=budget, workers=4, cache_mb=128)
    ct_rd = BudgetedRemoteZarrV2(ct_url, str(frame_level), budget=budget, workers=4, cache_mb=64)
    shp = BOX_SHAPE_ZYX
    picked = None
    probes = 0
    scored = []
    try:
        z_base = (zc * f // align) * align
        seen = set()
        for _, _, _, _, fy, fx in cands:
            if probes >= max_probes or budget.remaining < reserve_bytes + 2 * ct_rd._expected_chunk_bytes:
                break
            base = (z_base, int(fy // align) * align, int(fx // align) * align)
            if base in seen or any(b + 128 > s for b, s in zip(base, m7.shape)):
                continue
            seen.add(base)
            probes += 1
            chunk = _read_box_budgeted(m7, base, tuple(b + 192 for b in base), budget) >= 128
            for oz in (0, 64):
                for oy in (0, 32):
                    for ox in (0, 32):
                        sub = chunk[oz:oz + shp[0], oy:oy + shp[1], ox:ox + shp[2]]
                        act = float(sub.mean())
                        if not 0.10 <= act <= 0.6:
                            continue
                        runs = _column_run_count(sub, GM.dominant_normal_axis(sub))
                        if runs >= 3:
                            scored.append((-runs, -act, probes, (oz, oy, ox), base, np.packbits(sub)))
            del chunk
        scored.sort(key=lambda s: s[:3])
        for neg_runs, neg_act, _, off, base, packed in scored:
            if budget.remaining < ct_rd._expected_chunk_bytes:
                break
            lo = tuple(b + o for b, o in zip(base, off))
            ct = _read_box_budgeted(ct_rd, lo, tuple(l + s for l, s in zip(lo, shp)), budget)
            if float((ct > 0).mean()) < 0.9:
                continue
            sub = np.unpackbits(packed, count=int(np.prod(shp))).reshape(shp).astype(bool)
            picked = (ct, np.where(sub, 255, 0).astype(np.uint8),
                      {"probe_count": probes, "active_fraction_m7": -neg_act, "runs_per_column": -neg_runs})
            break
    finally:
        m7.close()
        ct_rd.close()
    return picked


def run_crushed_unread_box(budget: ByteBudget, scroll: str, **select_kw) -> dict[str, Any] | None:
    """Select (:func:`select_crushed_unread_box`) and analyse one unread box; ``None`` if none fits."""
    t0 = time.perf_counter()
    used0 = budget.used
    spec = SCROLL_SPECS[scroll]
    voxel_um = float(spec["ct_level0_um"]) * 2 ** int(spec["m7_frame_ct_level"])
    picked = select_crushed_unread_box(budget, scroll, **select_kw)
    if picked is None:
        return None
    ct, m7_box, sel = picked
    probes = sel["probe_count"]
    metrics = compare_rgt_vs_single_sheet_baseline(ct, m7_box, voxel_um=voxel_um)
    metrics["active_fraction_m7"] = sel["active_fraction_m7"]
    metrics["ct_nonzero_fraction"] = float((ct > 0).mean())
    metrics["http_bytes"] = int(budget.used - used0)
    metrics["http_mib"] = metrics["http_bytes"] / 2 ** 20
    metrics["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return {
        "role": "crushed_eligible_unread",
        "scroll": scroll,
        "ct_dataset": spec["ct"].rsplit("/", 1)[-1],
        "m7_dataset": spec["m7"].rsplit("/", 1)[-1],
        "box_rule": ("deterministic interior m7-L0 box with the most m7 runs per column (tightest wraps) "
                     "among density-ranked radial candidates on the m7 overview mid-slab, 10-60% active voxels"),
        "probe_count": int(probes),
        "baseline_binary": "m7 >= 128",
        "metrics": sanitize_public_metrics(metrics),
    }


def run_live_preflight(evidence_out: str | os.PathLike[str], *, cap_bytes: int = GLOBAL_HTTP_CAP_BYTES) -> dict[str, Any]:
    t0 = time.perf_counter()
    budget = ByteBudget(cap_bytes)
    boxes = [run_public_control_box(budget)]
    unread = None
    for scroll in ("PHerc1545", "PHerc0846A"):
        unread = run_crushed_unread_box(budget, scroll)
        if unread is not None:
            break
    if unread is None:
        raise RuntimeError("no multi-wrap unread box found inside the byte budget")
    boxes.append(unread)
    receipt = {
        "schema": "seismic_rgt_crushed_preflight/v1",
        "experiment": "Exp T9-Unwrap (Q46): seismic RGT / Wheeler multi-sheet unwrapper vs single-sheet grower",
        "method": ("structure-tensor dip + autocorrelation wrap spacing + sparse dip-guided RGT (CG) with "
                   "monotonic projection + Wheeler peak picking + ridge-refined ordered sheets + "
                   "in-RAM render (9 slices) + onsheet_score tile 32; baseline = grow_m7_patch.extract_height_field"),
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "http": {"budget_bytes": int(budget.cap), "http_bytes": int(budget.used),
                 "http_mib": _round(budget.used / 2 ** 20, 3), "requests": int(budget.requests),
                 "persistent_ct_on_disk": False},
        "boxes": boxes,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "caveats": [
            "Box areas are tiny (one 64x96x96 box); usable_cm2 counts rendered pixels with full in-box support on all 9 planes.",
            "f_parallel is an upper bound on the on-sheet share; tiles without CT structure count as parallel.",
            "crossing_fraction_rgt == 0 follows from strict phase monotonicity plus order-preserving refinement; it does not prove each sheet is one physical wrap.",
            "No ink detector run; nothing here claims ink or text.",
        ],
    }
    validate_public_receipt_privacy(receipt)
    Path(evidence_out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--evidence-out", required=True, help="sanitized public JSON receipt")
    p.add_argument("--cap-mib", type=float, default=80.0)
    args = p.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    r = run_live_preflight(args.evidence_out, cap_bytes=int(args.cap_mib * 2 ** 20))
    print(json.dumps({"http": r["http"], "boxes": [{b["scroll"]: b["metrics"]} for b in r["boxes"]]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
