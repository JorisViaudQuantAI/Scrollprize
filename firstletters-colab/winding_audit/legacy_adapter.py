"""Callback adapter for the archived, defective winding-chain diagnostic.

The numerical loop is mechanically copied from legacy_snapshot.py at repository
revision f7fc619ec11f9dc95caa0c90fb2148cbf7c86ebd. It deliberately retains the
storage-order defect, first-encountered tie behavior, default NumPy argsort and
absence of the corrected implementation's validation/cycle guards.

Only input lookup plumbing, parameter exposure and output instrumentation change.
The CLI's fixed seed 0 and hop limit 60 become same-default API parameters; exact
archived-CLI parity is asserted for those historical settings. Transverse IDs
must already select the requested slab. They are sorted and deduplicated because
the archived CLI produced them with np.nonzero. No longitudinal coordinates are
requested. The sole added summary key is completed_start_ids, and empty results
are normalized from historical shape (0,) to (0, 3).

This file must remain beside the exact legacy_snapshot.py. SHA verification
prevents a silently substituted baseline. It does not assert the legacy physics
or numerical behavior is valid.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

LEGACY_SHA256 = "65fa04d337067e8ee1e1fee2c768c167c29e7d84e954b93d6a1bff22bc42caf4"


def verify_legacy_snapshot() -> str:
    """Fail if the adjacent provenance baseline is absent or has changed."""
    path = Path(__file__).with_name("legacy_snapshot.py")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != LEGACY_SHA256:
        raise ValueError("Adjacent legacy_snapshot.py does not match the pinned baseline")
    return observed


def trace_legacy(
    transverse_ids: Sequence[int],
    coordinates_for: Callable[[int], np.ndarray],
    crossings_for: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    family_for: Callable[[int], int],
    length_for: Callable[[int], int],
    control_points: Sequence[Mapping[str, float]],
    *,
    z0: int,
    dz: float = 40,
    nstart: int = 3000,
    rtol: float = 6.0,
    front: float = 0.3,
    jmax: int = 200,
    zstep: float = 1e9,
    rmax: float = 1e9,
    mirror: int = 1,
    seed: int = 0,
    max_hops: int = 60,
) -> tuple[dict, np.ndarray]:
    """Replay the historical algorithm over callbacks, returning summary/rows."""
    verify_legacy_snapshot()
    DZ, NSTART, RTOL, FRONT, JMAX, ZSTEP = dz, nstart, rtol, front, jmax, zstep
    cp = sorted(control_points, key=lambda p: p['z'])
    uz = np.array([p['z'] for p in cp], float); uy = np.array([p['y'] for p in cp], float); ux = np.array([p['x'] for p in cp], float)
    H = np.asarray(sorted(set(int(t) for t in transverse_ids)), dtype=np.int64)
    Hset = set(H.tolist())
    geo = {}
    def g(h):
        if h not in geo:
            a = np.asarray(coordinates_for(int(h))).astype(float)
            z = a[0, 0]; cy, cx = np.interp(z, uz, uy), np.interp(z, uz, ux)
            th = np.unwrap(np.arctan2(a[:, 1] - cy, MIR * (a[:, 2] - cx))); r = np.hypot(a[:, 1] - cy, a[:, 2] - cx)
            geo[h] = (z, th, r, 1.0 if th[-1] >= th[0] else -1.0)
        return geo[h]
    def crossings(t):
        return crossings_for(int(t))
    rng = np.random.default_rng(seed)
    lens = np.asarray([length_for(int(h)) for h in H], dtype=np.int64)
    cand = H[lens >= 40]
    RMAX = rmax
    MIR = mirror
    cand = np.array([h for h in cand if np.median(g(h)[2]) < RMAX]) if RMAX < 1e8 else cand
    starts = rng.choice(cand, min(NSTART, len(cand)), replace=False)
    res = []; brk = []; completed_ids = []
    for h0 in starts:
        z, th, r, sg = g(h0)
        i0 = 0 if sg > 0 else len(th) - 1
        start = (th[i0], r[i0], z); acc = 0.0; h = h0; ientry = i0; ok = False; zs = [z]; hops = 0
        while hops < max_hops:
            z, th, r, sg = g(h)
            front = len(th) - 1 if sg > 0 else 0
            span = (th[front] - th[ientry]) * sg
            if acc + span >= 2 * np.pi:  # finish inside this track
                need = 2 * np.pi - acc
                tt = (th - th[ientry]) * sg
                idx = np.arange(len(th))
                sel = idx >= ientry if sg > 0 else idx <= ientry
                o = np.argsort(tt[sel]); r1 = np.interp(need, tt[sel][o], r[sel][o])
                res.append((r1 - start[1], z - start[2], hops)); completed_ids.append(int(h0)); ok = True; break
            # hop: crossings of h in the front 25% of its theta range (beyond entry)
            P, SL, PL = crossings(h)
            tt = (th - th[ientry]) * sg
            best = None
            for v, il, jv in zip(P, SL, PL):
                if family_for(int(v)) != 1 or tt[il] < FRONT * span: continue
                P2, SL2, PL2 = crossings(v)
                for h2, jv2, i2 in zip(P2, SL2, PL2):
                    if h2 == h or h2 not in Hset: continue
                    if abs(int(jv2) - int(jv)) > JMAX: continue
                    z2, th2, r2, sg2 = g(h2)
                    if abs(z2 - z) > ZSTEP: continue
                    dth = np.angle(np.exp(1j * (th2[i2] - th[il])))
                    if abs(dth) > 0.05 or abs(r2[i2] - r[il]) > RTOL: continue
                    front2 = len(th2) - 1 if sg2 > 0 else 0
                    adv = (th2[front2] - th2[i2]) * sg2
                    gain = tt[il] + dth + adv - span
                    cost = abs(r2[i2] - r[il]) + 0.1 * abs(z2 - z0)
                    if gain > 0.02 and (best is None or cost < best[0]):
                        best = (cost, h2, i2, tt[il] + dth)
            if best is None: brk.append((hops, acc)); break
            acc += best[3]; h, ientry = best[1], best[2]; hops += 1; zs.append(g(h)[0])
    dr = np.array([x[0] for x in res]); dzz = np.array([x[1] for x in res])
    out = dict(z0=z0, DZ=DZ, transverse_tracks_in_slab=int(len(H)), chains_started=int(len(starts)), chains_full_turn=int(len(res)),
               dr_positive=int((dr > 0).sum()), dr_negative=int((dr < 0).sum()),
               median_dr=float(np.median(dr)) if len(dr) else None, median_abs_dr=float(np.median(np.abs(dr))) if len(dr) else None,
               n_abs_dr_5_60_pos=int(((dr > 5) & (dr < 60)).sum()), n_abs_dr_5_60_neg=int(((dr < -5) & (dr > -60)).sum()),
               median_abs_dz=float(np.median(np.abs(dzz))) if len(dzz) else None, median_hops=float(np.median([x[2] for x in res])) if res else None)
    out['broken']=len(brk); out['broken_median_acc_deg']=float(np.degrees(np.median([b[1] for b in brk]))) if brk else None; out['broken_max_acc_deg']=float(np.degrees(max(b[1] for b in brk))) if brk else None
    out['completed_start_ids'] = completed_ids
    return out, np.asarray(res, dtype=float).reshape(-1, 3)
