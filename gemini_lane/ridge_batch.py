"""Batch nearest-ridge offset; preserve scalar fallback on plateau/other dtypes."""
import numpy as np


def pick_nearest_ridge_offsets(profiles, offsets, threshold, step, reference):
    """One offset per ray; identical first-run tie policy and float64 refinement.

    Fast path is the finite-or-minus-infinity float32 SSR profile contract.
    Plateau or near-tied maxima use the original scalar (reduction order matters).
    No changed threshold, clipping, ray support or acceptance rule.
    """
    p = np.asarray(profiles)
    o = np.asarray(offsets)
    if p.ndim != 2 or o.ndim != 1 or p.shape[1] != len(o) or not len(o):
        raise ValueError('expected MxK profiles and K offsets')
    if (p.dtype != np.float32 or o.dtype != np.float64 or not np.isfinite(o).all()
            or not np.isfinite([threshold, step]).all() or step <= 0
            or np.isnan(p).any() or np.isposinf(p).any()):
        return np.asarray([reference(row, o, threshold, step) for row in p], dtype=np.float64)
    n, k = p.shape
    result = np.zeros(n, np.float64)
    above = p >= threshold
    prev = np.pad(above[:, :-1], ((0, 0), (1, 0)))
    nxt = np.pad(above[:, 1:], ((0, 0), (0, 1)))
    rows, starts = np.nonzero(above & ~prev)
    erows, ends = np.nonzero(above & ~nxt)
    if not np.array_equal(rows, erows):
        raise RuntimeError('inconsistent runs')
    if not len(rows):
        return result
    j = np.arange(k)[None, :]
    member = (j >= starts[:, None]) & (j <= ends[:, None])
    runs = np.where(member, p[rows], -np.inf)
    maxima = runs.max(axis=1)
    # NumPy float32 array compared with a Python scalar casts it to float32.
    near = member & (runs >= (maxima.astype(np.float64)-1e-6).astype(np.float32)[:, None])
    count = near.sum(axis=1)
    # All components in any ambiguous ray go through the exact legacy routine.
    fallback = np.zeros(n, bool)
    fallback[rows[count != 1]] = True
    keep = ~fallback[rows]
    rows, runs = rows[keep], runs[keep]
    if len(rows):
        peak = runs.argmax(axis=1)
        cand = o[peak].copy()
        li = np.maximum(peak-1, 0)
        ri = np.minimum(peak+1, k-1)
        yl, yc, yr = (p[rows, ix].astype(np.float64) for ix in (li, peak, ri))
        with np.errstate(invalid='ignore'):
            denom = yl - 2.0 * yc + yr
        good = ((peak > 0) & (peak < k-1) & np.isfinite(yl) & np.isfinite(yc)
                & np.isfinite(yr) & (denom < -1e-9) & (yc > yl) & (yc > yr))
        cand[good] += np.clip(0.5 * (yl[good]-yr[good])/denom[good], -.5, .5)*step
        dist = np.abs(cand)
        best = np.full(n, np.inf)
        np.minimum.at(best, rows, dist)
        ids = np.arange(len(rows))
        win = np.full(n, len(rows), dtype=np.int64)
        isbest = dist == best[rows]
        np.minimum.at(win, rows[isbest], ids[isbest])
        selected = win < len(rows)
        result[selected] = cand[win[selected]]
    for i in np.flatnonzero(fallback):
        result[i] = reference(p[i], o, threshold, step)
    return result
