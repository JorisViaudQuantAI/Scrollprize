# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Isometric resampling + exact geometric QA for spiral-fit tifxyz windings (CPU-only).

Spiral-fit windings (e.g. the PHerc. 1203 / PHerc. 0175A fits and the incoming
PHerc. 0846A handoff) are sampled at *equal angular steps* around crushed wraps.
Along a row the 3D column step ``du`` therefore swings between roughly 10 and
60 voxels while the ``scale`` metadata claims a uniform 20 vx cell.  Detector
windows are stretched or squeezed 0.6-3x and vertex-count areas are wrong
(``valid_vertices * cell^2`` can under-report the true triangulated area by
~30%).  This module:

1. Loads tifxyz (``x.tif``, ``y.tif``, ``z.tif``, ``meta.json``, optional
   ``mask.tif``) with three-channel sentinel validation.  A vertex is invalid
   when any channel is non-finite or ``<= 0`` (``-1`` is the canonical
   sentinel; exact zeros are treated as holes too, which is stricter than the
   ``< 0`` rule in ``stream_snap_render.py`` because a few writers emit
   ``(0, 0, 0)`` for holes).
2. Computes exact geometric QA: triangulated physical area in cm^2 vs the
   nominal vertex-count area, ``du``/``dv`` step quantiles and stretch factors
   vs ``target_step_vx``, and local grid shear (deviation from 90 degrees
   between column and row tangents).
3. Resamples the grid:

   * ``common_column``: one common column arc-length map per winding (median
     edge length over rows), so every row is sampled at the same fractional
     source columns: zero added inter-row shear, median spacing
     ``target_step_vx``.
   * ``regularized_arclength``: each row's own arc-length map, coupled across
     rows by a Tikhonov penalty ``lam * sum (delta[i+1,k] - delta[i,k])^2``
     on the column offsets relative to the common map.  ``lam`` is chosen by
     log-bisection as the weakest coupling whose geometric shear stays within
     ``max_shear_deg`` (or the common-column baseline, if that is already
     larger); it falls back to ``common_column`` if no candidate improves the
     local scale error.

   Rows are first resampled with a common row arc-length map (both modes) unless
   ``resample_rows=False``.  Every interpolation is 1D linear between two
   adjacent *valid* source vertices; nothing is ever interpolated across a
   sentinel gap, and invalid outputs are written as ``-1`` in all channels.

The report is a diagnostic of parameterization quality, not an on-sheet or ink
certificate.  No GPU, no network.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

import numpy as np
from scipy.linalg import solve_banded

SENTINEL = -1.0
DEFAULT_TARGET_STEP_VX = 20.0
DEFAULT_MAX_SHEAR_DEG = 15.0
DEFAULT_VOXEL_UM = 7.91
DEFAULT_SCALE = 0.05
MODES = ("common_column", "regularized_arclength")
_DEGENERATE_VX = 1e-6
_SNAP = 1e-9
END_SNAP_FRAC = 0.1  # row-end samples closer than this * step to a source end are snapped onto it
_VOXEL_KEYS = ("voxel_um", "voxel_size_um", "voxelsize_um", "voxel_size")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_tiff_2d(path: Path) -> np.ndarray:
    try:
        import tifffile  # type: ignore[import-not-found]

        return np.asarray(tifffile.imread(str(path)))
    except ImportError:  # pragma: no cover - tifffile is the normal path
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img)


def _write_tiff_float32(path: Path, arr: np.ndarray) -> None:
    data = np.ascontiguousarray(arr, dtype=np.float32)
    try:
        import tifffile  # type: ignore[import-not-found]

        tifffile.imwrite(str(path), data)
    except ImportError:  # pragma: no cover
        from PIL import Image

        Image.fromarray(data, mode="F").save(str(path), format="TIFF")


def validate_channels(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, zero_is_sentinel: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Stack x/y/z into ``(H, W, 3)`` float64 and return ``(points, valid)``.

    A vertex is valid only if all three channels are finite and ``> 0``
    (``>= 0`` when ``zero_is_sentinel=False``).  Invalid vertices become
    ``SENTINEL`` in every channel.
    """
    xs = [np.asarray(c, dtype=np.float64) for c in (x, y, z)]
    if xs[0].ndim != 2 or any(c.shape != xs[0].shape for c in xs):
        raise ValueError(f"Expected matching 2D x/y/z arrays, got {[c.shape for c in xs]}")
    pts = np.stack(xs, axis=-1)
    valid = np.isfinite(pts).all(axis=-1)
    with np.errstate(invalid="ignore"):
        if zero_is_sentinel:
            valid &= (pts > 0.0).all(axis=-1)
        else:
            valid &= (pts >= 0.0).all(axis=-1)
    pts = np.where(valid[..., None], pts, SENTINEL)
    return pts, valid


@dataclass
class TifxyzGrid:
    points: np.ndarray  # (H, W, 3) float64, SENTINEL where invalid
    valid: np.ndarray  # (H, W) bool
    meta: dict[str, Any] = field(default_factory=dict)


def read_tifxyz(mesh_dir: str | os.PathLike[str], *, zero_is_sentinel: bool = True) -> TifxyzGrid:
    """Load a tifxyz directory, honoring sentinels and an optional ``mask.tif``."""
    root = Path(mesh_dir)
    chans = [_read_tiff_2d(root / f"{c}.tif") for c in ("x", "y", "z")]
    pts, valid = validate_channels(*chans, zero_is_sentinel=zero_is_sentinel)
    meta_path = root / "meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    mask_path = root / "mask.tif"
    if mask_path.is_file():
        m = _read_tiff_2d(mask_path)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape == valid.shape:
            valid &= m >= 255
            pts = np.where(valid[..., None], pts, SENTINEL)
    return TifxyzGrid(pts, valid, meta)


def write_tifxyz(
    mesh_dir: str | os.PathLike[str], points: np.ndarray, valid: np.ndarray, meta: dict[str, Any]
) -> None:
    """Write a sentinel-clean float32 tifxyz directory with refreshed ``bbox``."""
    root = Path(mesh_dir)
    root.mkdir(parents=True, exist_ok=True)
    pts = np.where(valid[..., None], points, SENTINEL)
    for ci, name in enumerate(("x", "y", "z")):
        _write_tiff_float32(root / f"{name}.tif", pts[..., ci])
    out = dict(meta)
    out.setdefault("format", "tifxyz")
    out.setdefault("type", "seg")
    if valid.any():
        vp = points[valid]
        out["bbox"] = [[float(v) for v in vp.min(axis=0)], [float(v) for v in vp.max(axis=0)]]
    else:
        out["bbox"] = None
    (root / "meta.json").write_text(json.dumps(_json_safe(out), indent=2) + "\n", encoding="utf-8")


def normalize_scale(scale: Any) -> tuple[float, float]:
    """Return ``(sx, sy)`` from a scalar or 2-element ``scale`` entry."""
    if isinstance(scale, (list, tuple, np.ndarray)):
        if len(scale) != 2:
            raise ValueError(f"Expected 2 elements for scale, got {scale!r}")
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = sy = float(scale)
    if not (math.isfinite(sx) and math.isfinite(sy)) or sx <= 0 or sy <= 0:
        raise ValueError(f"scale must be positive and finite, got {(sx, sy)!r}")
    return sx, sy


def meta_scale(meta: dict[str, Any]) -> tuple[tuple[float, float], str]:
    if "scale" in meta and meta["scale"] is not None:
        return normalize_scale(meta["scale"]), "meta"
    return (DEFAULT_SCALE, DEFAULT_SCALE), "default"


def resolve_voxel_um(meta: dict[str, Any], override: float | None = None) -> tuple[float, str]:
    if override is not None:
        v, src = float(override), "cli"
    else:
        v, src = DEFAULT_VOXEL_UM, "default"
        for key in _VOXEL_KEYS:
            if isinstance(meta.get(key), (int, float)):
                v, src = float(meta[key]), f"meta:{key}"
                break
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"voxel_um must be positive, got {v!r}")
    return v, src


# ---------------------------------------------------------------------------
# Geometric QA
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("...k,...k->...", v, v))


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 0.5 * _norm(np.cross(b - a, c - a))


def triangulated_area_vx2(points: np.ndarray, valid: np.ndarray) -> float:
    """Exact area of the valid triangulation in voxel^2.

    Full quads are split along the (01, 10) diagonal; quads with exactly three
    valid corners contribute the single triangle formed by those corners.
    """
    if points.shape[0] < 2 or points.shape[1] < 2:
        return 0.0
    p00, p01 = points[:-1, :-1], points[:-1, 1:]
    p10, p11 = points[1:, :-1], points[1:, 1:]
    v00, v01 = valid[:-1, :-1], valid[:-1, 1:]
    v10, v11 = valid[1:, :-1], valid[1:, 1:]
    n = v00.astype(np.int8) + v01 + v10 + v11
    ta = _tri_area(p00, p01, p10)
    tb = _tri_area(p01, p11, p10)
    tc = _tri_area(p00, p01, p11)
    td = _tri_area(p00, p11, p10)
    area = np.where(n == 4, ta + tb, 0.0)
    three = n == 3
    area = area + np.where(three & ~v00, tb, 0.0)
    area = area + np.where(three & ~v11, ta, 0.0)
    area = area + np.where(three & ~v01, td, 0.0)
    area = area + np.where(three & ~v10, tc, 0.0)
    return float(area.sum())


def edge_lengths(points: np.ndarray, valid: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Edge lengths along ``axis`` (1 = column step du, 0 = row step dv) and validity."""
    if axis == 1:
        d = points[:, 1:] - points[:, :-1]
        ok = valid[:, 1:] & valid[:, :-1]
    else:
        d = points[1:] - points[:-1]
        ok = valid[1:] & valid[:-1]
    return _norm(d), ok


def shear_deviation_deg(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-corner |90 - angle(e_u, e_v)| in degrees over valid non-degenerate corners."""
    if points.shape[0] < 2 or points.shape[1] < 2:
        return np.zeros(0)
    eu = points[:-1, 1:] - points[:-1, :-1]
    ev = points[1:, :-1] - points[:-1, :-1]
    ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
    nu, nv = _norm(eu), _norm(ev)
    ok &= (nu > _DEGENERATE_VX) & (nv > _DEGENERATE_VX)
    if not ok.any():
        return np.zeros(0)
    c = np.abs(np.einsum("...k,...k->...", eu[ok], ev[ok])) / (nu[ok] * nv[ok])
    return np.degrees(np.arcsin(np.clip(c, 0.0, 1.0)))


def _stat(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _step_stats(lengths: np.ndarray, target: float) -> dict[str, Any]:
    if lengths.size == 0:
        return {"n": 0}
    p05, p50, p95 = (float(v) for v in np.percentile(lengths, [5, 50, 95]))
    rel = lengths / target
    return {
        "n": int(lengths.size),
        "p05_vx": p05,
        "p50_vx": p50,
        "p95_vx": p95,
        "min_vx": float(lengths.min()),
        "max_vx": float(lengths.max()),
        "stretch_p05": p05 / target,
        "stretch_p50": p50 / target,
        "stretch_p95": p95 / target,
        "p95_over_p05": p95 / p05 if p05 > 0 else float("inf"),
        "log_err_mean": float(np.mean(np.abs(np.log(rel)))),
        "rel_err_p50": float(np.percentile(np.abs(rel - 1.0), 50)),
        "rel_err_p95": float(np.percentile(np.abs(rel - 1.0), 95)),
    }


def du_log_err_mean(points: np.ndarray, valid: np.ndarray, target: float) -> float:
    L, ok = edge_lengths(points, valid, 1)
    L = L[ok & (L > _DEGENERATE_VX)]
    return float(np.mean(np.abs(np.log(L / target)))) if L.size else float("inf")


def grid_qa(
    points: np.ndarray,
    valid: np.ndarray,
    *,
    target_step_vx: float = DEFAULT_TARGET_STEP_VX,
    voxel_um: float = DEFAULT_VOXEL_UM,
    cell_vx: tuple[float, float] = (1.0 / DEFAULT_SCALE, 1.0 / DEFAULT_SCALE),
) -> dict[str, Any]:
    """Exact geometric QA of a tifxyz grid (``cell_vx`` = nominal (u, v) cell size)."""
    cm_per_vx = voxel_um / 1e4
    nvalid = int(valid.sum())
    tri_vx2 = triangulated_area_vx2(points, valid)
    true_cm2 = tri_vx2 * cm_per_vx**2
    nominal_cm2 = nvalid * cell_vx[0] * cell_vx[1] * cm_per_vx**2
    out: dict[str, Any] = {
        "shape": [int(points.shape[0]), int(points.shape[1])],
        "valid_vertices": nvalid,
        "valid_fraction": nvalid / max(1, valid.size),
        "cell_vx": [float(cell_vx[0]), float(cell_vx[1])],
        "target_step_vx": float(target_step_vx),
        "voxel_um": float(voxel_um),
        "area": {
            "triangulated_vx2": tri_vx2,
            "triangulated_cm2": true_cm2,
            "nominal_cm2": nominal_cm2,
            "nominal_over_true": nominal_cm2 / true_cm2 if true_cm2 > 0 else float("nan"),
        },
    }
    for name, axis in (("du", 1), ("dv", 0)):
        L, ok = edge_lengths(points, valid, axis)
        L = L[ok & (L > _DEGENERATE_VX)]
        out[name] = _step_stats(L, target_step_vx)
    sh = shear_deviation_deg(points, valid)
    out["shear_deg"] = {
        "n": int(sh.size),
        "p05": _stat(sh, 5),
        "p50": _stat(sh, 50),
        "p95": _stat(sh, 95),
        "p99": _stat(sh, 99),
        "max": float(sh.max()) if sh.size else float("nan"),
        "mean": float(sh.mean()) if sh.size else float("nan"),
    }
    return out


# ---------------------------------------------------------------------------
# Sentinel-safe 1D sampling
# ---------------------------------------------------------------------------

def _sample_axis1(
    points: np.ndarray, valid: np.ndarray, q: np.ndarray, qvalid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample each row ``i`` at fractional columns ``q[i, k]`` (linear, never across gaps)."""
    H, W = valid.shape
    q = np.clip(q, 0.0, W - 1.0)
    j0 = np.floor(q).astype(np.int64)
    f = q - j0
    hi = f >= 1.0 - _SNAP
    j0 = np.where(hi, np.minimum(j0 + 1, W - 1), j0)
    f = np.where(hi | (f <= _SNAP), 0.0, f)
    j1 = np.minimum(j0 + 1, W - 1)
    rows = np.arange(H)[:, None]
    v0 = valid[rows, j0]
    v1 = valid[rows, j1]
    ok = qvalid & v0 & ((f == 0.0) | v1)
    p = (1.0 - f)[..., None] * points[rows, j0] + f[..., None] * points[rows, j1]
    p = np.where(ok[..., None], p, SENTINEL)
    return p, ok


def _sample_axis0(points: np.ndarray, valid: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample all columns at common fractional rows ``q[k]``."""
    pt = np.swapaxes(points, 0, 1)
    vt = valid.T
    qq = np.broadcast_to(q[None, :], (vt.shape[0], q.size))
    p, ok = _sample_axis1(pt, vt, qq, np.ones(qq.shape, dtype=bool))
    return np.swapaxes(p, 0, 1), ok.T


def _arclength_samples(t0: float, t1: float, step: float) -> np.ndarray:
    """Samples ``t0, t0+step, ...`` ending exactly at ``t1`` (short tail merged if < step/4)."""
    span = t1 - t0
    if span <= 0:
        return np.array([t0])
    n = int(math.floor(span / step + 1e-9))
    t = t0 + np.arange(n + 1) * step
    rem = t1 - t[-1]
    if rem > 0.25 * step:
        t = np.append(t, t1)
    elif rem > 1e-9 * step:
        if t.size > 1:
            t[-1] = t1
        else:
            t = np.append(t, t1)
    return t


def _common_edge_profile(points: np.ndarray, valid: np.ndarray, axis: int, fill: float) -> np.ndarray:
    """Median edge length along ``axis`` over the orthogonal axis (valid edges only)."""
    L, ok = edge_lengths(points, valid, axis)
    L = np.where(ok, L, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmedian(L, axis=0 if axis == 1 else 1)
    prof = np.where(np.isfinite(prof) & (prof > _DEGENERATE_VX), prof, fill)
    return prof


def _cumulative(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.maximum(lengths, 1e-9), axis=-1)])


def resample_rows_common(
    points: np.ndarray, valid: np.ndarray, step: float, fill: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample rows to ``step`` spacing with one common row arc-length map."""
    H = valid.shape[0]
    if H < 2:
        return points.copy(), valid.copy()
    S = _cumulative(_common_edge_profile(points, valid, 0, fill))
    t = _arclength_samples(0.0, float(S[-1]), step)
    q = np.interp(t, S, np.arange(H, dtype=np.float64))
    return _sample_axis0(points, valid, q)


def common_column_map(points: np.ndarray, valid: np.ndarray, step: float, fill: float) -> np.ndarray:
    """Fractional source column per output column (identical for every row)."""
    W = valid.shape[1]
    S = _cumulative(_common_edge_profile(points, valid, 1, fill))
    t = _arclength_samples(0.0, float(S[-1]), step)
    return np.interp(t, S, np.arange(W, dtype=np.float64))


def _interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """``np.interp`` with linear extrapolation from the end segments."""
    y = np.interp(x, xp, fp)
    if xp.size >= 2:
        sl = (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-12)
        sr = (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-12)
        y = np.where(x < xp[0], fp[0] + (x - xp[0]) * sl, y)
        y = np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * sr, y)
    return y


def _own_column_maps(
    points: np.ndarray, valid: np.ndarray, step: float, fill: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row arc-length maps aligned to the common map at the middle column.

    Returns ``(j_common (K,), j_own (H, K), weight (H, K), row_arclength (H, W))`` on a shared output
    column axis covering the union of all rows' arc-length ranges.  Both maps
    are linearly extrapolated past their ends (so offsets stay smooth); the
    caller clips to the source column range.
    """
    H, W = valid.shape
    idx = np.arange(W, dtype=np.float64)
    C = _common_edge_profile(points, valid, 1, fill)
    S = _cumulative(C)
    L, ok = edge_lengths(points, valid, 1)
    L = np.where(ok & (L > _DEGENERATE_VX), L, C[None, :])
    s = np.concatenate([np.zeros((H, 1)), np.cumsum(np.maximum(L, 1e-9), axis=1)], axis=1)
    a = W // 2
    off = S[a] - s[:, a]  # row-i own arc-length origin in common coordinates
    ends = off + s[:, -1]
    t = _arclength_samples(min(0.0, float(off.min())), max(float(S[-1]), float(ends.max())), step)
    j_common = _interp_extrap(t, S, idx)
    j_own = np.empty((H, t.size))
    for i in range(H):
        j_own[i] = _interp_extrap(t - off[i], s[i], idx)
    in_range = (j_own >= 0.0) & (j_own <= W - 1.0)
    j0 = np.clip(np.floor(j_own).astype(np.int64), 0, max(W - 2, 0))
    edge_ok = ok[np.arange(H)[:, None], j0] if W > 1 else np.ones_like(in_range)
    w = np.where(in_range & edge_ok, 1.0, 1e-3)
    return j_common, j_own, w, s


def tikhonov_row_coupled(
    delta_own: np.ndarray, weight: np.ndarray, lam: float, anchor: float = 1e-2
) -> np.ndarray:
    """argmin_d sum w (d - d_own)^2 + lam * sum_k [sum_i (d[i+1,k] - d[i,k])^2 + anchor * sum_i d[i,k]^2].

    ``d`` is the column offset from the common map.  The row-coupling term
    suppresses inter-row shear; the small ``anchor`` term makes the
    ``lam -> inf`` limit exactly the common column map (``d = 0``).  Each output
    column is an independent tridiagonal system; they are solved together as
    one banded system ordered column-major.
    """
    H, K = delta_own.shape
    if lam <= 0:
        return delta_own.copy()
    wf = weight.T.reshape(-1)
    rhs = (weight * delta_own).T.reshape(-1)
    i = np.tile(np.arange(H), K)
    deg = (i > 0).astype(np.float64) + (i < H - 1)
    main = wf + lam * (deg + anchor)
    off = np.where(i[:-1] < H - 1, -lam, 0.0)
    ab = np.zeros((3, main.size))
    ab[0, 1:] = off
    ab[1] = main
    ab[2, :-1] = off
    d = solve_banded((1, 1), ab, rhs, check_finite=False)
    return d.reshape(K, H).T


def _finalize_column_maps(
    j: np.ndarray, row_arclength: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clip to the source column range, enforce monotone rows, and tidy row ends.

    Samples clipped onto the first/last source column collapse onto one
    boundary sample.  If the next interior sample lies closer than
    ``END_SNAP_FRAC * step`` (in that row's arc length) to the boundary, it is snapped onto
    the boundary instead, so no near-degenerate end edge is emitted and
    every row still reaches its source ends (no area loss).
    """
    H, K = j.shape
    W = row_arclength.shape[1]
    idx = np.arange(W, dtype=np.float64)
    q = np.maximum.accumulate(np.clip(j, 0.0, W - 1.0), axis=1)
    qvalid = np.ones(q.shape, dtype=bool)
    if W < 2:
        return q, qvalid
    for i in range(H):
        s_i = row_arclength[i]
        n0 = int(np.count_nonzero(q[i] <= _SNAP))
        if 1 <= n0 < K:
            f = n0
            if np.interp(q[i, f], idx, s_i) - s_i[0] < END_SNAP_FRAC * step:
                q[i, f] = 0.0
                qvalid[i, :f] = False
            else:
                qvalid[i, : f - 1] = False
        n1 = int(np.count_nonzero(q[i] >= (W - 1.0) - _SNAP))
        if 1 <= n1 < K:
            last = K - n1 - 1
            if last >= 0 and qvalid[i, last]:
                if s_i[-1] - np.interp(q[i, last], idx, s_i) < END_SNAP_FRAC * step:
                    q[i, last] = W - 1.0
                    qvalid[i, last + 1 :] = False
                else:
                    qvalid[i, last + 2 :] = False
    return q, qvalid


# ---------------------------------------------------------------------------
# Resampling driver
# ---------------------------------------------------------------------------

@dataclass
class ResampleResult:
    points: np.ndarray
    valid: np.ndarray
    meta: dict[str, Any]
    report: dict[str, Any]


def _shear_stat(points: np.ndarray, valid: np.ndarray, quantile: float) -> float:
    sh = shear_deviation_deg(points, valid)
    if sh.size == 0:
        return 0.0
    return float(sh.max()) if quantile >= 100.0 else float(np.percentile(sh, quantile))


def resample_spiral_winding(
    points: np.ndarray,
    valid: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    mode: str = "regularized_arclength",
    target_step_vx: float = DEFAULT_TARGET_STEP_VX,
    max_shear_deg: float = DEFAULT_MAX_SHEAR_DEG,
    voxel_um: float | None = None,
    resample_rows: bool = True,
    shear_quantile: float = 100.0,
    lambda_bounds: tuple[float, float] = (1e-4, 1e6),
    max_iter: int = 16,
    n_scan: int = 12,
) -> ResampleResult:
    """Resample a spiral-fit winding to near-isometric ``target_step_vx`` spacing."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not (target_step_vx > 0 and math.isfinite(target_step_vx)):
        raise ValueError("target_step_vx must be positive")
    if not (max_shear_deg > 0 and max_shear_deg < 90):
        raise ValueError("max_shear_deg must be in (0, 90)")
    if not 0 < shear_quantile <= 100:
        raise ValueError("shear_quantile must be in (0, 100]")
    meta = dict(meta or {})
    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[:2] != valid.shape:
        raise ValueError("points must be (H, W, 3) matching valid (H, W)")
    if valid.sum() < 4:
        raise ValueError("need at least 4 valid vertices")
    (sx, sy), scale_src = meta_scale(meta)
    vox, vox_src = resolve_voxel_um(meta, voxel_um)
    cell_in = (1.0 / sx, 1.0 / sy)
    qa_kw = dict(target_step_vx=target_step_vx, voxel_um=vox)
    before = grid_qa(points, valid, cell_vx=cell_in, **qa_kw)

    if resample_rows:
        p1, v1 = resample_rows_common(points, valid, target_step_vx, fill=cell_in[1])
        cell_v_out = target_step_vx
    else:
        p1, v1 = points, valid
        cell_v_out = cell_in[1]
    H1, W1 = v1.shape
    fill_u = cell_in[0]

    qc = common_column_map(p1, v1, target_step_vx, fill_u)
    qc2 = np.broadcast_to(qc[None, :], (H1, qc.size))
    pc, vc = _sample_axis1(p1, v1, qc2, np.ones(qc2.shape, dtype=bool))
    reg_info: dict[str, Any] = {}

    if mode == "common_column":
        out_p, out_v = pc, vc
    else:
        base_shear = _shear_stat(pc, vc, shear_quantile)
        base_err = du_log_err_mean(pc, vc, target_step_vx)
        cap_eff = max(float(max_shear_deg), base_shear)
        j_common, j_own, w, s_rows = _own_column_maps(p1, v1, target_step_vx, fill_u)
        d_own = j_own - j_common[None, :]
        evals: list[dict[str, float]] = []

        def evaluate(lam: float) -> dict[str, Any]:
            d = tikhonov_row_coupled(d_own, w, lam)
            q, qv = _finalize_column_maps(j_common[None, :] + d, s_rows, target_step_vx)
            p, v = _sample_axis1(p1, v1, q, qv)
            r = {
                "lam": float(lam),
                "shear": _shear_stat(p, v, shear_quantile),
                "err": du_log_err_mean(p, v, target_step_vx),
                "points": p,
                "valid": v,
            }
            evals.append({"lambda": r["lam"], "shear_stat_deg": r["shear"], "du_log_err_mean": r["err"]})
            return r

        best: dict[str, Any] | None = None

        def consider(r: dict[str, Any]) -> bool:
            nonlocal best
            feasible = r["shear"] <= cap_eff + 1e-9
            if feasible and (best is None or r["err"] < best["err"]):
                best = r
            return feasible

        # Shear is only piecewise-monotone in lam (row-end snapping is discrete),
        # so scan a log grid first, then bisect the first infeasible->feasible step.
        if not consider(evaluate(0.0)):
            lo_b, hi_b = lambda_bounds
            hi_b = max(hi_b, 10.0 * H1 * H1)
            grid = np.exp(np.linspace(math.log(lo_b), math.log(hi_b), max(2, n_scan)))
            prev_bad = None
            for lam in grid:
                if consider(evaluate(float(lam))):
                    break
                prev_bad = float(lam)
            else:
                lam = None
            if lam is not None and prev_bad is not None:
                lo, hi = math.log(prev_bad), math.log(float(lam))
                for _ in range(max(0, max_iter)):
                    if hi - lo < 1e-2:
                        break
                    mid = 0.5 * (lo + hi)
                    if consider(evaluate(math.exp(mid))):
                        hi = mid
                    else:
                        lo = mid
        fallback = None
        if best is None:
            fallback = "no lambda met the shear cap"
        elif not best["err"] < base_err:
            fallback = "no scale-error improvement over common_column"
        if fallback is not None:
            out_p, out_v, lam_sel, shear_sel, err_sel = pc, vc, float("inf"), base_shear, base_err
        else:
            assert best is not None
            out_p, out_v = best["points"], best["valid"]
            lam_sel, shear_sel, err_sel = best["lam"], best["shear"], best["err"]
        reg_info = {
            "lambda": lam_sel,
            "shear_cap_deg": float(max_shear_deg),
            "shear_cap_effective_deg": cap_eff,
            "shear_quantile": float(shear_quantile),
            "shear_stat_deg": shear_sel,
            "du_log_err_mean": err_sel,
            "common_column_shear_stat_deg": base_shear,
            "common_column_du_log_err_mean": base_err,
            "fallback_to_common_column": fallback,
            "evaluations": evals,
        }

    cell_out = (float(target_step_vx), float(cell_v_out))
    after = grid_qa(out_p, out_v, cell_vx=cell_out, **qa_kw)
    out_meta = dict(meta)
    out_meta.pop("bbox", None)
    out_meta["scale"] = [1.0 / cell_out[0], 1.0 / cell_out[1]]
    summary = {
        "mode": mode,
        "target_step_vx": float(target_step_vx),
        "max_shear_deg": float(max_shear_deg),
        "resample_rows": bool(resample_rows),
        "voxel_um": vox,
        "source_shape": before["shape"],
        "source_scale": [sx, sy],
        "triangulated_cm2_before": before["area"]["triangulated_cm2"],
        "triangulated_cm2_after": after["area"]["triangulated_cm2"],
    }
    out_meta["spiral_winding_resample"] = summary
    tb, ta = before["area"]["triangulated_cm2"], after["area"]["triangulated_cm2"]
    report = {
        "mode": mode,
        "params": {
            "target_step_vx": float(target_step_vx),
            "max_shear_deg": float(max_shear_deg),
            "shear_quantile": float(shear_quantile),
            "resample_rows": bool(resample_rows),
            "voxel_um": vox,
            "voxel_um_source": vox_src,
            "scale": [sx, sy],
            "scale_source": scale_src,
        },
        "before": before,
        "after": after,
        "area_change_frac": (ta - tb) / tb if tb > 0 else float("nan"),
        "du_p95_over_p05_before": before["du"].get("p95_over_p05"),
        "du_p95_over_p05_after": after["du"].get("p95_over_p05"),
    }
    if reg_info:
        report["regularization"] = reg_info
    return ResampleResult(out_p, out_v, out_meta, report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _brief(report: dict[str, Any]) -> dict[str, Any]:
    b, a = report["before"], report["after"]
    out = {
        "mode": report["mode"],
        "shape": [b["shape"], a["shape"]],
        "triangulated_cm2": [b["area"]["triangulated_cm2"], a["area"]["triangulated_cm2"]],
        "nominal_cm2": [b["area"]["nominal_cm2"], a["area"]["nominal_cm2"]],
        "area_change_frac": report["area_change_frac"],
        "du_p50_vx": [b["du"].get("p50_vx"), a["du"].get("p50_vx")],
        "du_p95_over_p05": [b["du"].get("p95_over_p05"), a["du"].get("p95_over_p05")],
        "du_log_err_mean": [b["du"].get("log_err_mean"), a["du"].get("log_err_mean")],
        "shear_deg_p95": [b["shear_deg"]["p95"], a["shear_deg"]["p95"]],
        "shear_deg_max": [b["shear_deg"]["max"], a["shear_deg"]["max"]],
    }
    if "regularization" in report:
        r = report["regularization"]
        out["lambda"] = r["lambda"]
        out["fallback_to_common_column"] = r["fallback_to_common_column"]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mesh", required=True, help="Input tifxyz directory (x.tif, y.tif, z.tif, meta.json)")
    ap.add_argument("--out-dir", default=None, help="Output tifxyz directory (required unless --dry-run)")
    ap.add_argument("--mode", choices=MODES, default="regularized_arclength")
    ap.add_argument("--target-step-vx", type=float, default=DEFAULT_TARGET_STEP_VX)
    ap.add_argument("--max-shear-deg", type=float, default=DEFAULT_MAX_SHEAR_DEG)
    ap.add_argument("--shear-quantile", type=float, default=100.0,
                    help="Shear statistic bounded by --max-shear-deg (100 = max over corners)")
    ap.add_argument("--voxel-um", type=float, default=None,
                    help=f"Voxel size in um (default: meta.json or {DEFAULT_VOXEL_UM})")
    ap.add_argument("--keep-rows", action="store_true", help="Do not resample rows (column map only)")
    ap.add_argument("--dry-run", action="store_true", help="Compute QA/report only; write no tifxyz")
    ap.add_argument("--json-out", default=None, help="Write the full before/after QA report here")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = build_arg_parser().parse_args(argv)
    if not args.dry_run and not args.out_dir:
        print("error: --out-dir is required unless --dry-run", file=sys.stderr)
        return 2
    grid = read_tifxyz(args.mesh)
    res = resample_spiral_winding(
        grid.points,
        grid.valid,
        grid.meta,
        mode=args.mode,
        target_step_vx=args.target_step_vx,
        max_shear_deg=args.max_shear_deg,
        voxel_um=args.voxel_um,
        resample_rows=not args.keep_rows,
        shear_quantile=args.shear_quantile,
    )
    report = dict(res.report)
    report["mesh"] = Path(args.mesh).name  # basename only: no private paths in reports
    report["dry_run"] = bool(args.dry_run)
    if not args.dry_run:
        out_dir = Path(args.out_dir)
        meta = dict(res.meta)
        if "uuid" in meta:
            meta["uuid"] = out_dir.name
        write_tifxyz(out_dir, res.points, res.valid, meta)
        report["out_dir"] = out_dir.name
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(_brief(report)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
