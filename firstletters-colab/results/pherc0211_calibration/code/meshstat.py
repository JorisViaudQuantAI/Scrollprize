"""meshstat.py DIR UMB.json : per-winding stats of tifxyz windings: area, z range, radius from umbilicus, local stretch, gap to next winding"""
import sys, glob, os, json, numpy as np, tifffile
from scipy.spatial import cKDTree
D, U = sys.argv[1], sys.argv[2]
cp = json.load(open(U))['control_points']; uz = np.array([p['z'] for p in cp]); ux = np.array([p['x'] for p in cp]); uy = np.array([p['y'] for p in cp])
o = np.argsort(uz); uz, ux, uy = uz[o], ux[o], uy[o]
def load(w):
    x, y, z = [tifffile.imread(f'{w}/{c}.tif').astype(np.float64) for c in 'xyz']
    return x, y, z, (z > 0) & (x > 0)
ws = sorted(glob.glob(D + '/w[0-9][0-9][0-9]'))
P = {}; rows = []
for w in ws:
    x, y, z, v = load(w); n = os.path.basename(w)
    cx = np.interp(z, uz, ux); cy = np.interp(z, uy, uy) if False else np.interp(z, uz, uy)
    r = np.hypot(x - cx, y - cy)
    # stretch: 3D distance between neighbouring vertices / nominal 20 vox
    dc = np.sqrt((x[:, 1:] - x[:, :-1])**2 + (y[:, 1:] - y[:, :-1])**2 + (z[:, 1:] - z[:, :-1])**2); vc = v[:, 1:] & v[:, :-1]
    dr = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2); vr = v[1:] & v[:-1]
    P[n] = np.c_[x[v], y[v], z[v]]
    rows.append([n, v.sum(), z[v].min(), z[v].max(), np.median(r[v]), np.percentile(dc[vc], 5) / 20, np.median(dc[vc]) / 20, np.percentile(dc[vc], 95) / 20, np.median(dr[vr]) / 20])
names = [r[0] for r in rows]
print('w nverts z0 z1 r_med stretchC_p5 p50 p95 stretchR_p50 gap_next_med gap_next_p10 frac_lt5')
for i, r in enumerate(rows):
    g = ['', '', '']
    if i + 1 < len(rows):
        t = cKDTree(P[names[i + 1]]); d, _ = t.query(P[names[i]][::3]); g = [np.median(d), np.percentile(d, 10), (d < 5).mean()]
    print(r[0], r[1], '%.0f %.0f %.0f %.2f %.2f %.2f %.2f' % tuple(r[2:]), *(('%.1f' % a) if a != '' else '' for a in g))
