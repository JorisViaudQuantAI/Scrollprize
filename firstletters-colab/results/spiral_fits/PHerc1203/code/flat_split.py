#!/usr/bin/env python3
"""flat_split.py FLAT_TIFXYZ RAW_WINDINGS_DIR OUT_DIR [--windings w018,w025,...] [--max-dist 6]
Cut the combined Lasagna-flattened spiral surface (one tifxyz, flatten_spiral_checkpoint.py) into per-winding tifxyz:
every flattened vertex gets the winding id of the nearest raw fitted-winding vertex (KD-tree, 3-D, level-0 voxels;
vertices farther than --max-dist x the raw grid step from any raw vertex are dropped), then each winding keeps the
bounding box of its rows/cols with other vertices set to -1. Also prints vertex-spacing statistics (row / col step
length p5/p50/p95 in voxels, and |cos| of the angle between the row and col directions = shear) for the raw and the
flattened per-winding meshes. Writes OUT_DIR/wNNN/{x,y,z}.tif + meta.json and OUT_DIR/spacing.tsv."""
import argparse, glob, json, os
import numpy as np, tifffile
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument('flat'); ap.add_argument('raw'); ap.add_argument('out')
ap.add_argument('--windings', default=''); ap.add_argument('--max-dist', type=float, default=1.5)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)


def load(d):
    X, Y, Z = (tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz')
    v = (X > -0.5) & (Y > -0.5) & (Z > -0.5)
    return np.stack([X, Y, Z], -1), v


def spacing(P, v):
    dr = np.linalg.norm(P[1:, :] - P[:-1, :], axis=-1)[(v[1:, :] & v[:-1, :])]
    dc = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[(v[:, 1:] & v[:, :-1])]
    gr = P[1:, :-1] - P[:-1, :-1]; gc = P[:-1, 1:] - P[:-1, :-1]
    m = v[1:, :-1] & v[:-1, :-1] & v[:-1, 1:]
    cs = np.abs(np.sum(gr * gc, -1)) / (np.linalg.norm(gr, axis=-1) * np.linalg.norm(gc, axis=-1) + 1e-9)
    q = lambda x: np.percentile(x, [5, 50, 95]).round(2).tolist() if len(x) else [None] * 3
    return q(dr), q(dc), q(cs[m])


raw_dirs = sorted(d for d in glob.glob(os.path.join(a.raw, 'w[0-9][0-9][0-9]')) if os.path.isfile(d + '/x.tif'))
pts, lab = [], []
for d in raw_dirs:
    P, v = load(d); pts.append(P[v]); lab.append(np.full(v.sum(), int(os.path.basename(d)[1:4])))
pts = np.concatenate(pts); lab = np.concatenate(lab); tree = cKDTree(pts)
F, fv = load(a.flat); fmeta = json.load(open(a.flat + '/meta.json'))
print('flat grid', F.shape[:2], 'valid', int(fv.sum()), 'meta scale', fmeta.get('scale'))
dist, idx = tree.query(F[fv], k=1)
L = np.full(fv.shape, -1, int); L[fv] = np.where(dist <= a.max_dist * 20, lab[idx], -1)
print('assigned', int((L >= 0).sum()), 'dropped (far from any raw winding)', int(((L < 0) & fv).sum()))
want = [w.strip() for w in a.windings.split(',') if w.strip()] or [os.path.basename(d) for d in raw_dirs]
rows = ['winding\tvariant\tn_valid\trow_p5\trow_p50\trow_p95\tcol_p5\tcol_p50\tcol_p95\tshear_cos_p5\tshear_cos_p50\tshear_cos_p95\tarea_cm2']
um = float(fmeta.get('voxel_size_um', 9.362))
for w in want:
    k = int(w[1:4]); m = L == k
    if m.sum() < 50: print(w, 'too few flattened vertices', int(m.sum())); continue
    r = np.nonzero(m.any(1))[0]; c = np.nonzero(m.any(0))[0]
    sl = (slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1))
    P = F[sl].copy(); vv = m[sl]; P[~vv] = -1
    od = os.path.join(a.out, w); os.makedirs(od, exist_ok=True)
    for i, ch in enumerate('xyz'): tifffile.imwrite(f'{od}/{ch}.tif', P[..., i].astype(np.float32))
    q = vv[:-1, :-1] & vv[1:, :-1] & vv[:-1, 1:] & vv[1:, 1:]
    e1 = P[1:, :-1] - P[:-1, :-1]; e2 = P[:-1, 1:] - P[:-1, :-1]
    area = float(np.linalg.norm(np.cross(e1, e2), axis=-1)[q].sum() * (um * 1e-4) ** 2)
    meta = dict(fmeta); meta.update(uuid=f'{w}_flat', area_cm2=area,
                                    bbox=[[float(P[..., i][vv].min()) for i in range(3)], [float(P[..., i][vv].max()) for i in range(3)]],
                                    source=f'Lasagna flatten_fast_nofilter of the combined spiral surface, cut to winding {w} by nearest raw vertex')
    json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
    for variant, (PP, VV) in (('raw', load(os.path.join(a.raw, w))), ('flat', (P, vv))):
        s = spacing(PP, VV)
        rows.append('\t'.join(map(str, [w, variant, int(VV.sum()), *s[0], *s[1], *s[2], round(area, 3) if variant == 'flat' else ''])))
open(os.path.join(a.out, 'spacing.tsv'), 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))
