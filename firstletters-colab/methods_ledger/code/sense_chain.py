"""Winding-sense diagnostic from transverse tracks linked through crossings.

Ledger S5 / Q12. Coordinates are ZYX, with image x right and y down. Travel is
towards increasing atan2(y-cy, x-cx). Positive radial drift is the fitter's CW.

The original script applied storage direction to angular progress after already
selecting the low-angle endpoint. Reversing point storage could change a completed
chain to failure. Here storage direction selects indices only; angles are unwrapped
in canonical travel order, including across the atan2 branch cut.

Other explicit changes: crossings must lie ahead of the entry index; exact-cost
ties use a deterministic track/index key; duplicate routes are neutral; cycles
and hop-limit exits count as unsuccessful chains; family-0 tracks must have constant
z. The thresholds, seed, cost, forward-gain rule and 5 < abs(dr) < 60 band remain.
Final interpolation retains the original sorted-angle convention. It does not
certify that a nonmonotone track follows one anatomical sheet.

NO AUTOMATIC WINDING-SENSE VERDICT: chains share tracks, and axial taper confounds
radial drift when a chain changes z. On r=R+pitch*theta/(2*pi)+k*z, dr=pitch+k*dz.
Mirroring can reverse that confounded drift too. Exact-same-z closure statistics
are reported separately; they still do not establish anatomical sheet identity.

Legacy CLI:
 sense_chain.py TRACK_STORE_DIR CROSSINGS.npz UMBILICUS.json OUT.npy z0
     [DZ NSTART RTOL FRONT JMAX ZSTEP RMAX MIR]

The callback API requires transverse coordinates only. Longitudinal tracks
contribute crossing indices and lengths, not their coordinate arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Callable, Mapping, Sequence

import numpy as np


OUTCOME_SCHEMA = "sense-chain-terminal-outcomes-v1"


def trace_winding_sense(
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
    """Return descriptive counts and rows (dr, dz, hops) for full turns.

    transverse_ids must already select the desired z slab; original IDs are
    retained. crossings_for(t) returns partner IDs, local indices on t, then
    local indices on each partner. Bounds are checked. Crossing reciprocity and
    shared XYZ remain assumptions: no longitudinal coordinates are loaded.

    The compact outcome SHA-256 covers every start in seeded selection order,
    including failures. It begins with OUTCOME_SCHEMA encoded as ASCII and NUL.
    A completion record is b'C' + struct.pack('<qddq', start_id, dr, dz, hops).
    A failure is b'B' + struct.pack('<qqdI', start_id, hops, accumulated_angle,
    reason_byte_length) + UTF-8 reason. These are terminal outcomes, not a hash
    of every intermediate hop or of source geometry. Float bits are preserved.
    """
    if mirror not in (-1, 1):
        raise ValueError("mirror must be +1 or -1")
    if nstart < 0 or jmax < 0 or max_hops < 1:
        raise ValueError("nstart/jmax must be nonnegative; max_hops positive")
    if not all(np.isfinite(v) for v in (z0, dz, rtol, front, zstep, rmax)):
        raise ValueError("parameters must be finite")
    if dz < 0 or rtol < 0 or zstep < 0 or rmax <= 0 or not 0 <= front <= 1:
        raise ValueError("invalid slab, tolerance, radius or front parameter")
    cp = sorted(control_points, key=lambda p: p["z"])
    if not cp:
        raise ValueError("at least one umbilicus control point is required")
    uz, uy, ux = (np.asarray([p[k] for p in cp], dtype=float) for k in ("z", "y", "x"))
    if not all(np.isfinite(a).all() for a in (uz, uy, ux)) or np.any(np.diff(uz) <= 0):
        raise ValueError("umbilicus coordinates must be finite with distinct z")

    hids = np.asarray(sorted(set(int(t) for t in transverse_ids)), dtype=np.int64)
    if np.any(hids < 0):
        raise ValueError("track IDs must be nonnegative")
    if any(int(family_for(int(t))) != 0 for t in hids):
        raise ValueError("transverse_ids must contain only family-0 tracks")
    hset = set(hids.tolist())
    geometry, adjacency = {}, {}

    def geom(t):
        if t not in geometry:
            a = np.asarray(coordinates_for(t), dtype=float)
            if a.shape != (int(length_for(t)), 3) or not len(a):
                raise ValueError(f"track {t}: coordinate shape/length mismatch")
            if not np.isfinite(a).all() or np.any(a[:, 0] != a[0, 0]):
                raise ValueError(f"track {t}: family-0 coordinates must have constant finite z")
            z = float(a[0, 0])
            if not z0 - dz <= z <= z0 + dz:
                raise ValueError(f"track {t}: transverse track is outside the requested slab")
            cy, cx = np.interp(z, uz, uy), np.interp(z, uz, ux)
            yy, xx = a[:, 1] - cy, a[:, 2] - cx
            radius = np.hypot(yy, xx)
            if np.any(radius == 0):
                raise ValueError(f"track {t}: image angle is undefined at the umbilicus")
            raw = np.arctan2(yy, mirror * xx)
            preliminary = np.unwrap(raw)
            delta = preliminary[-1] - preliminary[0]
            # Endpoint tie makes zero-span handling storage invariant.
            forward = delta > 0 or (delta == 0 and tuple(a[0]) <= tuple(a[-1]))
            step = 1 if forward else -1
            theta = np.unwrap(raw[::step])
            if step < 0:
                theta = theta[::-1]
            geometry[t] = z, theta, radius, step
        return geometry[t]

    def cross(t):
        if t not in adjacency:
            arrays = tuple(np.asarray(v) for v in crossings_for(t))
            if len(arrays) != 3 or any(a.ndim != 1 for a in arrays):
                raise ValueError(f"track {t}: crossing arrays must be 1D triples")
            if len({len(a) for a in arrays}) != 1:
                raise ValueError(f"track {t}: crossing-array lengths differ")
            if any(a.size and not np.issubdtype(a.dtype, np.integer) for a in arrays):
                raise ValueError(f"track {t}: crossing IDs and indices must be integers")
            partners, local, other = arrays
            if np.any(local < 0) or np.any(local >= int(length_for(t))):
                raise ValueError(f"track {t}: crossing self index out of bounds")
            for partner, index in zip(partners, other):
                if int(partner) < 0:
                    raise ValueError(f"track {t}: crossing partner ID must be nonnegative")
                if int(index) < 0 or int(index) >= int(length_for(int(partner))):
                    raise ValueError(f"track {t}: crossing partner index out of bounds")
            adjacency[t] = arrays
        return adjacency[t]

    candidates = np.asarray([t for t in hids if int(length_for(int(t))) >= 40], dtype=np.int64)
    # Preserve the legacy sentinel: the old default skipped the radius test.
    if rmax < 1e8:
        candidates = np.asarray([t for t in candidates if np.median(geom(int(t))[2]) < rmax], dtype=np.int64)
    starts = np.random.default_rng(seed).choice(candidates, min(nstart, len(candidates)), replace=False)
    results, completed_ids, broken = [], [], []
    outcome_hasher = hashlib.sha256(OUTCOME_SCHEMA.encode("ascii") + b"\0")

    for start_id in starts:
        start_id = int(start_id)
        z, theta, radius, step = geom(start_id)
        entry = 0 if step > 0 else len(theta) - 1
        start_radius, start_z = radius[entry], z
        accumulated, track, hops = 0.0, start_id, 0
        visited = set()
        reason = "hop_limit"
        while hops < max_hops:
            if (track, entry) in visited:
                reason = "cycle"
                break
            visited.add((track, entry))
            z, theta, radius, step = geom(track)
            endpoint = len(theta) - 1 if step > 0 else 0
            span = theta[endpoint] - theta[entry]
            travel = theta - theta[entry]
            if accumulated + span >= 2 * np.pi:
                need = 2 * np.pi - accumulated
                indices = np.arange(entry, endpoint + step, step)
                order = np.argsort(travel[indices], kind="stable")
                r1 = np.interp(need, travel[indices][order], radius[indices][order])
                results.append((float(r1 - start_radius), float(z - start_z), hops))
                completed_ids.append(start_id)
                reason = None
                break
            best, routes = None, set()
            for v, il, jv in zip(*cross(track)):
                v, il, jv = int(v), int(il), int(jv)
                if int(family_for(v)) != 1 or (il - entry) * step < 0:
                    continue
                if travel[il] < front * span:
                    continue
                for h2, jv2, i2 in zip(*cross(v)):
                    h2, jv2, i2 = int(h2), int(jv2), int(i2)
                    if h2 == track or h2 not in hset or abs(jv2 - jv) > jmax:
                        continue
                    z2, theta2, radius2, step2 = geom(h2)
                    if abs(z2 - z) > zstep:
                        continue
                    dtheta = float(np.angle(np.exp(1j * (theta2[i2] - theta[il]))))
                    if abs(dtheta) > 0.05 or abs(radius2[i2] - radius[il]) > rtol:
                        continue
                    end2 = len(theta2) - 1 if step2 > 0 else 0
                    advance = theta2[end2] - theta2[i2]
                    progress = travel[il] + dtheta
                    gain = progress + advance - span
                    if gain <= 0.02:
                        continue
                    ci = il if step > 0 else len(theta) - 1 - il
                    cj = i2 if step2 > 0 else len(theta2) - 1 - i2
                    route = h2, cj, ci
                    if route in routes:
                        continue
                    routes.add(route)
                    cost = abs(radius2[i2] - radius[il]) + 0.1 * abs(z2 - z0)
                    key = float(cost), h2, cj, ci
                    if best is None or key < best[0]:
                        best = key, h2, i2, float(progress)
            if best is None:
                reason = "no_continuation"
                break
            accumulated += best[3]
            track, entry = best[1], best[2]
            hops += 1
        if reason is not None:
            broken.append((start_id, hops, accumulated, reason))
            reason_bytes = reason.encode("utf-8")
            outcome_hasher.update(b"B" + struct.pack("<qqdI", start_id, hops, accumulated, len(reason_bytes)) + reason_bytes)
        else:
            result = results[-1]
            outcome_hasher.update(b"C" + struct.pack("<qddq", start_id, result[0], result[1], hops))

    rows = np.asarray(results, dtype=float).reshape(-1, 3)
    dr, dzz = rows[:, 0], rows[:, 1]
    same = dzz == 0
    same_dr = dr[same]
    summary = dict(
        z0=z0, DZ=dz, transverse_tracks_in_slab=int(len(hids)), chains_started=int(len(starts)),
        chains_full_turn=int(len(rows)), dr_positive=int((dr > 0).sum()), dr_negative=int((dr < 0).sum()),
        median_dr=float(np.median(dr)) if len(dr) else None,
        median_abs_dr=float(np.median(np.abs(dr))) if len(dr) else None,
        n_abs_dr_5_60_pos=int(((dr > 5) & (dr < 60)).sum()),
        n_abs_dr_5_60_neg=int(((dr < -5) & (dr > -60)).sum()),
        median_abs_dz=float(np.median(np.abs(dzz))) if len(dzz) else None,
        median_hops=float(np.median(rows[:, 2])) if len(rows) else None,
        broken=len(broken),
        broken_median_acc_deg=float(np.degrees(np.median([b[2] for b in broken]))) if broken else None,
        broken_max_acc_deg=float(np.degrees(max(b[2] for b in broken))) if broken else None,
        broken_reasons={reason: sum(b[3] == reason for b in broken) for reason in sorted({b[3] for b in broken})},
        completed_start_ids=completed_ids,
        exact_same_z=dict(
            chains_full_turn=int(same.sum()), dr_positive=int((same_dr > 0).sum()), dr_negative=int((same_dr < 0).sum()),
            n_abs_dr_5_60_pos=int(((same_dr > 5) & (same_dr < 60)).sum()),
            n_abs_dr_5_60_neg=int(((same_dr < -5) & (same_dr > -60)).sum()),
            median_dr=float(np.median(same_dr)) if len(same_dr) else None,
        ),
        nonzero_dz_closures=int((~same).sum()),
        algorithm="increasing-image-angle-v2",
        winding_sense_verdict=None,
        per_start_outcome_schema=OUTCOME_SCHEMA,
        per_start_outcome_sha256=outcome_hasher.hexdigest(),
    )
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("track_store", "crossings", "umbilicus", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("z0", type=int)
    for name, typ, default in (("DZ", float, 40), ("NSTART", int, 3000), ("RTOL", float, 6),
                               ("FRONT", float, .3), ("JMAX", int, 200), ("ZSTEP", float, 1e9),
                               ("RMAX", float, 1e9), ("MIR", int, 1)):
        parser.add_argument(name, type=typ, nargs="?", default=default)
    args = parser.parse_args(argv)
    offsets = np.fromfile(args.track_store / "offsets.i64", "<i8")
    bounds = np.fromfile(args.track_store / "z_bounds.i32", "<i4").reshape(-1, 2)
    families = np.fromfile(args.track_store / "family_codes.i8", "i1")
    coords = np.memmap(args.track_store / "coordinates.i32", "<i4", "r").reshape(-1, 3)
    if len(offsets) != len(families) + 1 or len(bounds) != len(families) or offsets[0] != 0:
        raise ValueError("track metadata lengths/initial offset are inconsistent")
    if np.any(np.diff(offsets) < 0) or offsets[-1] != len(coords):
        raise ValueError("track coordinate offsets are inconsistent")
    lengths = np.diff(offsets)
    hids = np.flatnonzero((families == 0) & (bounds[:, 0] >= args.z0 - args.DZ) & (bounds[:, 1] <= args.z0 + args.DZ))
    with np.load(args.crossings) as archive:
        coff = archive["offsets"]
        partners, local, other = (archive[k] for k in ("partners", "self_local", "partner_local"))
        if len(coff) != len(families) + 1 or coff[0] != 0 or np.any(np.diff(coff) < 0):
            raise ValueError("crossing offsets are inconsistent")
        if any(len(a) != coff[-1] for a in (partners, local, other)):
            raise ValueError("crossing data lengths are inconsistent")

        def crossings_for(t):
            start, stop = coff[t:t + 2]
            return partners[start:stop], local[start:stop], other[start:stop]

        summary, rows = trace_winding_sense(
            hids, lambda t: coords[offsets[t]:offsets[t + 1]], crossings_for,
            lambda t: int(families[t]), lambda t: int(lengths[t]),
            json.loads(args.umbilicus.read_text())["control_points"],
            z0=args.z0, dz=args.DZ, nstart=args.NSTART, rtol=args.RTOL, front=args.FRONT,
            jmax=args.JMAX, zstep=args.ZSTEP, rmax=args.RMAX, mirror=args.MIR,
        )
    print(json.dumps(summary, sort_keys=True))
    np.save(args.output, rows)


if __name__ == "__main__":
    main()
