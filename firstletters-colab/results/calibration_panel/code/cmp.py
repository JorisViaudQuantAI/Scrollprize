#!/usr/bin/env python3
"""cmp.py REF.tif OUTDIR MAP [MAP ...]  : correlate maps with the reference (winner raw TIFF).
Each MAP: .tif/.png/.npy (2D). Both binned 4x (mean) over the common area; the 8 dihedral variants and
integer shifts of +-3 binned px are tried; best r reported. Also writes 1/4-scale PNGs (p1-p99.5 stretch)."""
import sys, os, json, numpy as np, tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def load(p):
    if p.endswith('.npy'): a = np.load(p).astype(np.float32)
    elif p.endswith(('.tif', '.tiff')): a = tifffile.imread(p).astype(np.float32)
    else: a = np.asarray(Image.open(p).convert('L'), np.float32)
    while a.ndim > 2: a = a[0]
    return a

def bin4(a):
    h, w = a.shape[0] // 4 * 4, a.shape[1] // 4 * 4
    return a[:h, :w].reshape(h // 4, 4, w // 4, 4).mean((1, 3))

def png(a, path, m=None):
    v = a[m] if m is not None and m.any() else a.ravel()
    lo, hi = np.percentile(v, (1, 99.5))
    Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)).save(path)

ref_p, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
R = bin4(load(ref_p)); rm = R > 0
png(R, f'{out}/ref_q.png', rm)
res = {}
for p in sys.argv[3:]:
    try:
        A = bin4(load(p))
    except Exception as e:
        res[p] = {'err': repr(e)}; continue
    best = (-2, None)
    for k in range(8):
        B = np.rot90(A, k % 4); B = B[:, ::-1] if k >= 4 else B
        h, w = min(B.shape[0], R.shape[0]), min(B.shape[1], R.shape[1])
        if min(h, w) < 0.8 * min(R.shape): continue
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                r0, r1 = max(0, dy), min(h, h + dy); c0, c1 = max(0, dx), min(w, w + dx)
                a = R[r0:r1, c0:c1]; b = B[r0 - dy:r1 - dy, c0 - dx:c1 - dx]; m = (a > 0) & (b > 0)
                if m.sum() < 1000: continue
                r = float(np.corrcoef(a[m], b[m])[0, 1])
                if r > best[0]: best = (r, (k, dy, dx), b)
    if best[1] is None:
        res[p] = {'err': 'no overlap', 'shape': A.shape}; continue
    res[p] = {'r': round(best[0], 4), 'variant': best[1], 'shape4': list(A.shape)}
    name = os.path.basename(os.path.dirname(p)) + '__' + os.path.splitext(os.path.basename(p))[0]
    png(best[2], f'{out}/{name}_q.png', best[2] > 0)
    print(f'{best[0]:.4f} {best[1]} {p}', flush=True)
json.dump(res, open(f'{out}/cmp.json', 'w'), indent=1)
