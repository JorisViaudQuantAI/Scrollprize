#!/usr/bin/env python3
"""ctlayer.py MESHDIR OUT --ct CTZARR --level 1 --windings w020,w040,... [--up 4]
Single CT layer sampled ON the mesh (offset 0), grid upsampled by --up (step 20 vox / 4 = 5 vox per px ~ 21 px/mm),
trilinear from the given pyramid level. Writes ct_<w>.png (contrast-stretched p1-p99) and ct_montage.jpg."""
import argparse, os, json, numpy as np, tifffile, zarr, cv2
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument('meshes'); ap.add_argument('out')
ap.add_argument('--ct', required=True); ap.add_argument('--level', type=int, default=1)
ap.add_argument('--windings', required=True); ap.add_argument('--up', type=int, default=4)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
arr = zarr.open(a.ct, mode='r')[str(a.level)]; s = 2 ** a.level; ch = np.array(arr.chunks)
cache = {}
def chunk(k):
    if k not in cache:
        sl = tuple(slice(i * c, min((i + 1) * c, n)) for i, c, n in zip(k, ch, arr.shape))
        cache[k] = np.asarray(arr[sl]).astype(np.float32)
        if len(cache) > 600: cache.pop(next(iter(cache)))
    return cache[k]
def sample(zyx):
    out = np.full(len(zyx), np.nan, np.float32); f = np.floor(zyx).astype(np.int64); t = zyx - f
    ok = np.all((f >= 0) & (f + 1 < np.array(arr.shape)), 1); idx = np.nonzero(ok)[0]
    acc = np.zeros(len(idx), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                p = f[idx] + [dz, dy, dx]; w = (t[idx,0] if dz else 1-t[idx,0])*(t[idx,1] if dy else 1-t[idx,1])*(t[idx,2] if dx else 1-t[idx,2])
                keys = p // ch; v = np.empty(len(idx), np.float32)
                kk, inv = np.unique(keys, axis=0, return_inverse=True)
                for j, k in enumerate(kk):
                    m = inv.ravel() == j; loc = p[m] - k * ch; v[m] = chunk(tuple(k))[loc[:,0], loc[:,1], loc[:,2]]
                acc += w * v
    out[idx] = acc; return out
panels = []
for w in a.windings.split(','):
    d = os.path.join(a.meshes, w)
    if not os.path.isfile(d + '/x.tif'): print('missing', w); continue
    X, Y, Z = (tifffile.imread(f'{d}/{c}.tif').astype(np.float32) for c in 'xyz')
    valid = (X > -0.5) & (Y > -0.5) & (Z > -0.5)
    up = lambda A: ndimage.zoom(np.where(valid, A, 0), a.up, order=1)
    V = ndimage.zoom(valid.astype(np.float32), a.up, order=1) > 0.99
    P = np.stack([up(Z), up(Y), up(X)], -1)
    img = np.full(V.shape, np.nan, np.float32); img[V] = sample(P[V] / s)
    f = np.isfinite(img)
    lo, hi = np.percentile(img[f], [1, 99]) if f.any() else (0, 1)
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8); g[~f] = 0
    cv2.putText(g, w, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.imwrite(os.path.join(a.out, f'ct_{w}.png'), g); panels.append(g)
    print(w, g.shape, 'valid', round(float(f.mean()), 3), flush=True)
if panels:
    W = max(p.shape[1] for p in panels)
    m = np.concatenate([np.pad(p, ((0, 20), (0, W - p.shape[1]))) for p in panels], 0)
    if m.shape[1] > 2400: m = cv2.resize(m, (2400, int(m.shape[0] * 2400 / m.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.out, 'ct_montage.jpg'), m, [cv2.IMWRITE_JPEG_QUALITY, 88])
