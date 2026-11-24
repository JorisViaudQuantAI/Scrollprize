"""normside.py UMB.json MESH... : fraction of vertices whose grid normal n = cross(dP/drow, dP/dcol), P=(x,y,z), points
towards the umbilicus axis (horizontal component). Same formula for every mesh -> same rendered layer order meaning."""
import sys, json, numpy as np, tifffile
cp = json.load(open(sys.argv[1]))['control_points']; uz = np.array([p['z'] for p in cp]); o = np.argsort(uz)
ux = np.array([p['x'] for p in cp])[o]; uy = np.array([p['y'] for p in cp])[o]; uz = uz[o]
for d in sys.argv[2:]:
    x, y, z = [tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']; P = np.stack([x, y, z], -1); v = z > 0
    dr = np.gradient(P, axis=0); dc = np.gradient(P, axis=1); n = np.cross(dr, dc)
    tx = np.interp(z, uz, ux) - x; ty = np.interp(z, uz, uy) - y
    dot = n[..., 0] * tx + n[..., 1] * ty
    vv = v & np.roll(v, 1, 0) & np.roll(v, -1, 0) & np.roll(v, 1, 1) & np.roll(v, -1, 1)
    print(d, 'towards-umbilicus fraction %.3f' % (dot[vv] > 0).mean(), 'n', vv.sum())
