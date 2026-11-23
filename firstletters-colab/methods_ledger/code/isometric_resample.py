#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/isometric_resample.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths, job names and every location of unread material
# were removed; the output metadata
# records the source folder name instead of its absolute path. Numerical part unchanged.
# Ledger entry: 01_surfaces.md S8 (non-isometry of fitted windings)
# ---------------------------------------------------------------------------------------------------
"""Resample a tifxyz grid so that vertices are ~equally spaced in 3D (default 20 voxels), without shear.
Spiral fits parametrise by angle/height, so cells can be 11-60 voxels wide; rendered at a fixed px per cell, such a
mesh would stretch or squash the papyrus up to 3x. One common column map (median over rows of the cumulative arc
length) and one common row map (median over columns) are applied to the whole grid, so straight lines stay straight.
usage: isometric_resample.py SRC DST [--step 20] [--per-row]
(--per-row is the variant we REJECTED: it removes the residual scale error but shears the grid.)"""
import sys, os, json, numpy as np, tifffile, cv2
src, dst = sys.argv[1], sys.argv[2]
step = float(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 20.0
P = np.stack([tifffile.imread(f"{src}/{c}.tif") for c in "xyz"], -1).astype(np.float32)
v = (P[..., 2] > 0) & (P[..., 0] != -1)
def cum(axis):
    d = np.linalg.norm(np.diff(P, axis=axis), axis=-1)
    ok = v[:, 1:] & v[:, :-1] if axis == 1 else v[1:] & v[:-1]
    d = np.where(ok, d, np.nan)
    med = np.nanmedian(d, axis=0 if axis == 1 else 1)          # per column gap (axis 1) / per row gap (axis 0)
    med = np.where(np.isfinite(med), med, np.nanmedian(med))
    return np.concatenate([[0], np.cumsum(med)])
cu, cv_ = cum(1), cum(0)
nu, nv = int(cu[-1] // step) + 1, int(cv_[-1] // step) + 1
iu = np.interp(np.arange(nu) * step, cu, np.arange(len(cu))).astype(np.float32)
iv = np.interp(np.arange(nv) * step, cv_, np.arange(len(cv_))).astype(np.float32)
if "--per-row" not in sys.argv:
    mx, my = np.meshgrid(iu, iv)
    Q = cv2.remap(P, mx, my, cv2.INTER_LINEAR)
    vv = cv2.remap(v.astype(np.float32), mx, my, cv2.INTER_LINEAR) > 0.999
else:
    # per row: arc length along u, anchored at the grid's centre column (so shear between rows stays symmetric);
    # rows first (common v map), then each resampled row re-parametrised by its own arc length
    H, W = v.shape; c0 = W // 2
    R = cv2.remap(P, *np.meshgrid(np.arange(W, dtype=np.float32), iv), cv2.INTER_LINEAR)
    Rv = cv2.remap(v.astype(np.float32), *np.meshgrid(np.arange(W, dtype=np.float32), iv), cv2.INTER_LINEAR) > 0.999
    half = int(max(c0, W - c0) * 3)                       # generous output width in steps on each side
    Q = np.full((nv, 2 * half + 1, 3), -1, np.float32); vv = np.zeros((nv, 2 * half + 1), bool)
    for i in range(nv):
        d = np.linalg.norm(np.diff(R[i], axis=0), axis=-1); ok = Rv[i, 1:] & Rv[i, :-1]
        d = np.where(ok, d, np.nanmedian(np.where(ok, d, np.nan)) if ok.any() else step)
        s_ = np.concatenate([[0], np.cumsum(d)]); s_ -= s_[c0]
        k = np.arange(-half, half + 1) * step
        inside = (k >= s_[0]) & (k <= s_[-1])
        fi = np.interp(k[inside], s_, np.arange(W))
        j0 = np.floor(fi).astype(int); t = (fi - j0)[:, None]; j1 = np.minimum(j0 + 1, W - 1)
        Q[i, inside] = R[i, j0] * (1 - t) + R[i, j1] * t
        vv[i, inside] = Rv[i, j0] & Rv[i, j1]
    cols = np.where(vv.any(0))[0]; Q = Q[:, cols.min():cols.max() + 1]; vv = vv[:, cols.min():cols.max() + 1]
    nu = Q.shape[1]
Q[~vv] = -1
os.makedirs(dst, exist_ok=True)
for i, c in enumerate("xyz"):
    tifffile.imwrite(f"{dst}/{c}.tif", Q[..., i])
m = json.load(open(f"{src}/meta.json")); m["scale"] = [1 / step, 1 / step]
m["isometric_resample"] = {"src": os.path.basename(os.path.normpath(src)), "step_vox": step, "grid_in": list(v.shape), "grid_out": [nv, nu]}
json.dump(m, open(f"{dst}/meta.json", "w"), indent=1)
d1 = np.linalg.norm(np.diff(Q, axis=1), axis=-1)[vv[:, 1:] & vv[:, :-1]]; d0 = np.linalg.norm(np.diff(Q, axis=0), axis=-1)[vv[1:] & vv[:-1]]
print(f"{os.path.basename(src)}: grid {v.shape} -> {(nv, nu)}; spacing u p5/50/95 {np.percentile(d1,[5,50,95]).round(1)}, v {np.percentile(d0,[5,50,95]).round(1)}")
