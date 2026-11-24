"""windist.py WINNER_DIR MESHES_DIR OUT.tsv [--up 4] : distance (voxels) from the winner mesh's vertices in the letter rows
to each fitted winding (densified by bilinear upsampling of the tifxyz grid), per winding and row band.
Rows (z bands of the winner mesh): top 7600-7800, awarded 8300-8500, bottom 8950-9150, all 7650-9150."""
import sys, glob, os, argparse, numpy as np, tifffile
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
ap = argparse.ArgumentParser(); ap.add_argument('win'); ap.add_argument('meshes'); ap.add_argument('out')
ap.add_argument('--up', type=int, default=8); ap.add_argument('--glob', default='w[0-9][0-9][0-9]*')
a = ap.parse_args()
def load(d):
    x, y, z = [tifffile.imread(f'{d}/{c}.tif').astype(np.float32) for c in 'xyz']
    return x, y, z, (z > 0) & (x > 0)
wx, wy, wz, wv = load(a.win)
# densify the winner too (its grid step is 20 vox)
def dens(x, y, z, v, up):
    X, Y, Z = [zoom(c, up, order=1) for c in (x, y, z)]; V = zoom(v.astype(np.float32), up, order=0) > 0.5
    return np.c_[X[V], Y[V], Z[V]]
W = dens(wx, wy, wz, wv, 2)
bands = {'top': (7600, 7800), 'awarded': (8300, 8500), 'bottom': (8950, 9150), 'all': (7650, 9150)}
rows = []
for d in sorted(glob.glob(os.path.join(a.meshes, a.glob))):
    if not os.path.isfile(d + '/x.tif'): continue
    x, y, z, v = load(d)
    if not v.any(): continue
    P = dens(x, y, z, v, a.up)
    sel = (P[:, 2] > 7500) & (P[:, 2] < 9300)
    if sel.sum() < 10: continue
    t = cKDTree(P[sel])
    r = [os.path.basename(d)]
    for k, (z0, z1) in bands.items():
        q = W[(W[:, 2] >= z0) & (W[:, 2] < z1)]
        dd, _ = t.query(q, distance_upper_bound=400); dd[~np.isfinite(dd)] = 400
        r += [np.median(dd), np.percentile(dd, 10), (dd <= 2).mean(), (dd <= 4).mean(), (dd <= 8).mean()]
    rows.append(r)
hdr = ['winding'] + [f'{k}_{s}' for k in bands for s in ('med', 'p10', 'le2', 'le4', 'le8')]
with open(a.out, 'w') as f:
    f.write('\t'.join(hdr) + '\n')
    for r in rows: f.write('\t'.join([r[0]] + ['%.3f' % v for v in r[1:]]) + '\n')
rows.sort(key=lambda r: r[6])
for r in rows[:8]: print(r[0], 'awarded med %.1f p10 %.1f le2 %.2f le4 %.2f le8 %.2f | all med %.1f le4 %.2f' % (r[6], r[7], r[8], r[9], r[10], r[16], r[19]))
