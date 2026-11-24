"""normal_side.py TIFXYZ [UMB.json] : sign of n = cross(dP/drow, dP/dcol), P = (z, y, x) (oncheck/xsect convention) relative to
(a) the centre of curvature of each grid row's cross-section (least-squares circle fit in x,y), (b) an umbilicus if given.
Prints the fraction of vertices whose n points AWAY from the centre (outward)."""
import sys, json, numpy as np, tifffile
d = sys.argv[1]
X = tifffile.imread(d + '/x.tif').astype(float); Y = tifffile.imread(d + '/y.tif').astype(float); Z = tifffile.imread(d + '/z.tif').astype(float)
v = (X > -0.5) & (Y > -0.5) & (Z > -0.5)
P = np.stack([Z, Y, X], -1); P[~v] = np.nan
n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
out_c = []; 
for r in range(0, X.shape[0], max(1, X.shape[0] // 20)):
    m = v[r] & np.isfinite(n[r]).all(-1)
    if m.sum() < 20: continue
    x, y = X[r, m], Y[r, m]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]; c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    rad = np.stack([np.zeros_like(x), y - c[1], x - c[0]], -1)
    out_c.append(np.mean(np.sum(rad * n[r, m], -1) > 0))
print('outward fraction vs row curvature centres: median %.3f (rows %d)' % (np.median(out_c), len(out_c)))
if len(sys.argv) > 2:
    cp = sorted(json.load(open(sys.argv[2]))['control_points'], key=lambda p: p['z'])
    uz = np.array([p['z'] for p in cp]); uy = np.array([p['y'] for p in cp]); ux = np.array([p['x'] for p in cp])
    m = v & np.isfinite(n).all(-1)
    rad = np.stack([np.zeros(m.sum()), Y[m] - np.interp(Z[m], uz, uy), X[m] - np.interp(Z[m], uz, ux)], -1)
    print('outward fraction vs umbilicus: %.3f' % np.mean(np.sum(rad * n[m], -1) > 0))
