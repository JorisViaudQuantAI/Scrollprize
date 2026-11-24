"""gridq.py MESH... : grid quality of a tifxyz as rendered (grid -> rectangle):
 shear8 = signed angle between the grid axes minus 90 deg, axes spanning 8 cells (~1.5 mm at 20 vox), p5/p50/p95
 stretch = 3D vertex spacing / nominal (1/scale) along the turn (cols) and along rows, p5/p50/p95
 rough = angle of the cell normal to its 9x9 neighbourhood mean, median / p95 / share > 20 deg
Optional: --mask-pts file.npy (Nx3 xyz) -> also the same statistics restricted to cells within --rad vox of those points."""
import sys, json, argparse, numpy as np, tifffile
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('meshes', nargs='+'); ap.add_argument('--mask-pts', default=''); ap.add_argument('--rad', type=float, default=10)
ap.add_argument('--k', type=int, default=8); ap.add_argument('--json', default='')
a = ap.parse_args(); out = {}
def q(P, ok, nom, sel=None):
    k = a.k
    A = P[:-k, k:] - P[:-k, :-k]; B = P[k:, :-k] - P[:-k, :-k]
    mk = ok[:-k, :-k] & ok[:-k, k:] & ok[k:, :-k]
    ca = (A * B).sum(-1) / np.maximum(np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1), 1e-9)
    sh = 90 - np.degrees(np.arccos(np.clip(ca, -1, 1)))
    a1 = P[:-1, 1:] - P[:-1, :-1]; b1 = P[1:, :-1] - P[:-1, :-1]; m1 = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1]
    sc = np.linalg.norm(a1, axis=-1) / nom; sr = np.linalg.norm(b1, axis=-1) / nom
    n = np.cross(a1, b1); nn = n / np.maximum(np.linalg.norm(n, axis=-1), 1e-9)[..., None]; nn[~m1] = 0
    sm = np.stack([uniform_filter(nn[..., i], 9, mode='nearest') for i in range(3)], -1); w = uniform_filter(m1.astype(float), 9, mode='nearest')
    sm = sm / np.maximum(np.linalg.norm(sm, axis=-1), 1e-9)[..., None]; ang = np.degrees(np.arccos(np.clip((nn * sm).sum(-1), -1, 1)))
    mr = m1 & (w > 0.6)
    if sel is not None: mk = mk & sel[:-k, :-k]; m1 = m1 & sel[:-1, :-1]; mr = mr & sel[:-1, :-1]
    if mk.sum() < 5 or m1.sum() < 5: return None
    r = lambda v: [round(float(t), 2) for t in v]
    return dict(cells=int(m1.sum()), shear8=r(np.percentile(sh[mk], [5, 50, 95])), shear8_abs_le8=round(float((abs(sh[mk]) <= 8).mean()), 3),
                stretch_turn=r(np.percentile(sc[m1], [5, 50, 95])), stretch_z=r(np.percentile(sr[m1], [5, 50, 95])),
                rough=r([np.median(ang[mr]), np.percentile(ang[mr], 95)]) + [round(float((ang[mr] > 20).mean()), 3)])
MP = np.load(a.mask_pts) if a.mask_pts else None
for d in a.meshes:
    x, y, z = [tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']; P = np.stack([x, y, z], -1); ok = (z > 0) & np.isfinite(z)
    try: nom = 1 / json.load(open(f'{d}/meta.json'))['scale'][0]
    except Exception: nom = 20.0
    res = {'all': q(P, ok, nom)}
    if MP is not None:
        dd, _ = cKDTree(MP).query(P.reshape(-1, 3)); sel = (dd.reshape(ok.shape) <= a.rad) & ok
        res['place'] = q(P, ok, nom, sel); res['place_cells'] = int(sel.sum())
    out[d] = res; print(d.split('/')[-1], json.dumps(res))
if a.json: json.dump(out, open(a.json, 'w'), indent=1)
