#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/spiral_oncheck.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths, job names and every location of unread material
# were removed; the numerical part is unchanged.
# Ledger entry: 01_surfaces.md S4 (gate), S3, S7
# NOTE FOR REVIEWERS: the docstring below describes a control shift scaled to the local winding gap.
# The code does NOT do that: the control shift is uniform in [-8, +8] voxels for every vertex
# (line `ctrl = rng.uniform(-8, 8, len(sel))`). All published control numbers use the fixed +-8.
# "on m7" = m7 prediction > 127 at the vertex; "within 2 voxels" = nearest m7 run centre (peak with
# prominence 60) at |offset| <= 2; "CT z-score" = (CT at the vertex - mean of its +-16-voxel profile) / sd.
# ---------------------------------------------------------------------------------------------------
"""oncheck.py MESHDIR OUTDIR [--ct CTZARR --ct-level 1 --m7 M7ZARR --n 300 --seed 0]
On-sheet check of tifxyz windings against the CT. For N random valid vertices per winding: local normal from the
grid, CT profile (trilinear, at the given pyramid level) along the normal over +-K full-res voxels, distance from the
surface (offset 0) to the nearest CT intensity peak; same for the m7 prediction (distance to nearest predicted-sheet
run centre). A control re-samples the same vertices after a random offset uniform in [-P/2, P/2] (P = local winding
gap estimated from neighbouring windings, default 16), i.e. what an unrelated surface would score.
Also reports normal sign relative to the umbilicus (outward fraction). Writes oncheck.tsv / oncheck.json."""
import argparse, glob, json, os, sys
import numpy as np, tifffile, zarr
from scipy import ndimage
from scipy.signal import find_peaks
ap = argparse.ArgumentParser()
ap.add_argument('meshes'); ap.add_argument('out')
ap.add_argument('--ct', required=True); ap.add_argument('--ct-level', type=int, default=1)
ap.add_argument('--m7', default=None); ap.add_argument('--m7-level', type=int, default=0)
ap.add_argument('--umb', default=None)
ap.add_argument('--n', type=int, default=300); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--K', type=float, default=16.0); ap.add_argument('--glob', default='w[0-9][0-9][0-9]*')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
ct = zarr.open(a.ct, mode='r')[str(a.ct_level)]; cs = 2 ** a.ct_level
m7 = zarr.open(a.m7, mode='r')[str(a.m7_level)] if a.m7 else None; ms = 2 ** a.m7_level
umb = None
if a.umb:
    cp = sorted(json.load(open(a.umb))['control_points'], key=lambda p: p['z'])
    uz = np.array([p['z'] for p in cp], float); uy = np.array([p['y'] for p in cp], float); ux = np.array([p['x'] for p in cp], float)
    umb = lambda z: (np.interp(z, uz, uy), np.interp(z, uz, ux))
offs = np.arange(-a.K, a.K + 0.01, 0.5)

class Cache:  # chunk cache for a zarr array (reads whole chunks, keeps them)
    def __init__(s, arr): s.a = arr; s.c = arr.chunks; s.d = {}
    def get(s, key):
        if key not in s.d:
            sl = tuple(slice(k * c, min((k + 1) * c, n)) for k, c, n in zip(key, s.c, s.a.shape))
            s.d[key] = np.asarray(s.a[sl]).astype(np.float32)
            if len(s.d) > 400: s.d.pop(next(iter(s.d)))
        return s.d[key]
    def sample(s, zyx):  # zyx (M,3) in this array's voxel coords, trilinear, nan outside
        out = np.full(len(zyx), np.nan, np.float32)
        f = np.floor(zyx).astype(np.int64); t = zyx - f
        ok = np.all((f >= 0) & (f + 1 < np.array(s.a.shape)), axis=1)
        idx = np.nonzero(ok)[0]
        if not len(idx): return out
        acc = np.zeros(len(idx), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    p = f[idx] + [dz, dy, dx]
                    w = (t[idx, 0] if dz else 1 - t[idx, 0]) * (t[idx, 1] if dy else 1 - t[idx, 1]) * (t[idx, 2] if dx else 1 - t[idx, 2])
                    keys = p // np.array(s.c)
                    v = np.empty(len(idx), np.float32)
                    for key in {tuple(k) for k in keys}:
                        m = np.all(keys == key, axis=1); loc = p[m] - np.array(key) * np.array(s.c)
                        v[m] = s.get(key)[loc[:, 0], loc[:, 1], loc[:, 2]]
                    acc += w * v
        out[idx] = acc; return out
CT = Cache(ct); M7 = Cache(m7) if m7 is not None else None

def normals(X, Y, Z, valid):
    P = np.stack([Z, Y, X], -1).astype(np.float64); P[~valid] = np.nan
    gu = np.gradient(P, axis=1); gv = np.gradient(P, axis=0)
    n = np.cross(gv, gu); nn = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.where(nn > 1e-6, nn, np.nan)   # zyx; convention: cross(d/drow, d/dcol)

def nearest_peak(prof, prominence):
    sm = ndimage.gaussian_filter1d(np.nan_to_num(prof, nan=np.nanmin(prof) if np.isfinite(prof).any() else 0), 1.0)
    pk, _ = find_peaks(sm, prominence=prominence)
    if not len(pk): return np.nan
    return offs[pk[np.argmin(np.abs(offs[pk]))]]

rng = np.random.default_rng(a.seed); rows = []; summary = {}
dirs = sorted(d for d in glob.glob(os.path.join(a.meshes, a.glob)) if os.path.isfile(os.path.join(d, 'x.tif')) and '_spliced' not in d)
for d in dirs:
    w = os.path.basename(d)
    X = tifffile.imread(d + '/x.tif').astype(np.float32); Y = tifffile.imread(d + '/y.tif').astype(np.float32); Z = tifffile.imread(d + '/z.tif').astype(np.float32)
    valid = (X > -0.5) & (Y > -0.5) & (Z > -0.5)
    N = normals(X, Y, Z, valid); ok = valid & np.isfinite(N).all(-1)
    vy, vx = np.nonzero(ok)
    if len(vy) < 10: continue
    sel = rng.choice(len(vy), min(a.n, len(vy)), replace=False)
    base = np.stack([Z[vy[sel], vx[sel]], Y[vy[sel], vx[sel]], X[vy[sel], vx[sel]]], -1)
    nv = N[vy[sel], vx[sel]]
    ctrl = rng.uniform(-8, 8, len(sel))
    res = {}
    for name, shift in (('fit', np.zeros(len(sel))), ('ctrl', ctrl)):
        S = base[:, None, :] + (offs[None, :, None] + shift[:, None, None]) * nv[:, None, :]
        prof = CT.sample((S / cs).reshape(-1, 3)).reshape(len(sel), -1)
        dct = np.array([nearest_peak(p, 8.0) for p in prof])
        res[name + '_ct'] = dct
        mu = np.nanmean(prof, 1); sd = np.nanstd(prof, 1) + 1e-3
        res[name + '_ctz'] = (prof[:, len(offs) // 2] - mu) / sd   # CT z-score at the surface within its +-K profile
        if M7 is not None:
            pm = M7.sample((S / ms).reshape(-1, 3)).reshape(len(sel), -1)
            res[name + '_m7'] = np.array([nearest_peak(p, 60.0) for p in pm])
            res[name + '_m7on'] = pm[:, len(offs) // 2] > 127
    out_frac = np.nan
    if umb is not None:
        cy, cx = umb(base[:, 0]); radial = np.stack([np.zeros(len(sel)), base[:, 1] - cy, base[:, 2] - cx], -1)
        out_frac = float(np.mean(np.sum(radial * nv, -1) > 0))
    meta = json.load(open(d + '/meta.json')) if os.path.isfile(d + '/meta.json') else {}
    s = dict(winding=w, area_cm2=meta.get('area_cm2'), n=int(len(sel)), normal_outward_frac=out_frac)
    for k, v in res.items():
        v = np.asarray(v, float)
        if k.endswith('m7on'): s[k] = float(np.mean(v)); continue
        if k.endswith('ctz'): s[k + '_mean'] = float(np.nanmean(v)); s[k + '_frac_pos'] = float(np.nanmean(v > 0)); continue
        f = np.isfinite(v); av = np.abs(v[f])
        s[k + '_found'] = float(f.mean()); s[k + '_med_abs'] = float(np.median(av)) if f.any() else None
        s[k + '_le2'] = float(np.mean(av <= 2)) if f.any() else None; s[k + '_le4'] = float(np.mean(av <= 4)) if f.any() else None
    rows.append(s); print(json.dumps(s)); sys.stdout.flush()
    np.savez_compressed(os.path.join(a.out, f'samples_{w}.npz'), base=base, normal=nv, **{k: np.asarray(v) for k, v in res.items()})
keys = sorted({k for r in rows for k in r})
with open(os.path.join(a.out, 'oncheck.tsv'), 'w') as f:
    f.write('\t'.join(keys) + '\n')
    for r in rows: f.write('\t'.join(str(r.get(k)) for k in keys) + '\n')
json.dump(rows, open(os.path.join(a.out, 'oncheck.json'), 'w'), indent=1)
print('done', len(rows), 'windings')
