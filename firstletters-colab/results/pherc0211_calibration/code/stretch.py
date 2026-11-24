"""stretch.py MESHDIR... : grid shape, scale, and 3D vertex spacing / nominal (1/scale) along columns and rows, p5/p50/p95"""
import sys, json, numpy as np, tifffile
for d in sys.argv[1:]:
    x, y, z = [tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']; v = (z > 0)
    sc = json.load(open(f'{d}/meta.json')).get('scale', [0.05, 0.05]); nom = 1 / sc[0]
    def sp(ax):
        P = np.stack([x, y, z], -1); d_ = np.linalg.norm(np.diff(P, axis=ax), axis=-1)
        vv = v[1:] & v[:-1] if ax == 0 else v[:, 1:] & v[:, :-1]; return np.percentile(d_[vv] / nom, [5, 50, 95])
    c, r = sp(1), sp(0)
    print(d.split('/')[-1], x.shape, 'scale', sc, 'valid', v.sum(), 'along-turn p5/50/95 %.2f/%.2f/%.2f' % tuple(c), 'along-z %.2f/%.2f/%.2f' % tuple(r))
