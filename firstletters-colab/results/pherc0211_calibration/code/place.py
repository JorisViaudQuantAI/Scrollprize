"""place.py --win WINNER --mesh OURFLAT [--raw OURRAW] --ours a.png,b.png --theirs a.png,b.png --out PREFIX
Correspondence between the winner mesh and one of our (flattened) windings near the letter rows:
 - per winner vertex (rows z 7600-9150): distance to our winding (densified x8), our grid (r,c)
 - affine map winner-z20 -> ours-z20 from pairs with d <= DMAX, our maps warped into the winner frame, side by side
 - local stretch of OUR RAW (fitter-grid) mesh at the matched place (3D spacing / 20 vox along the turn and along z)
z20 frame: col*20/f, row*20/f with f = 1000/um/20 (render px = grid*20 for an uncropped render)."""
import argparse, json, numpy as np, tifffile, cv2
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
ap = argparse.ArgumentParser()
ap.add_argument('--win', required=True); ap.add_argument('--mesh', required=True); ap.add_argument('--raw', default='')
ap.add_argument('--ours', default=''); ap.add_argument('--theirs', default=''); ap.add_argument('--labels', default='')
ap.add_argument('--out', required=True); ap.add_argument('--dmax', type=float, default=6.0); ap.add_argument('--um', type=float, default=9.362)
ap.add_argument('--z0', type=float, default=7600); ap.add_argument('--z1', type=float, default=9150)
a = ap.parse_args(); F = 1000.0 / a.um / 20.0
def load(d):
    x, y, z = [tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']; return np.stack([x, y, z], -1), (z > 0)
def dens(P, v, up):
    R, C = np.meshgrid(np.arange(P.shape[0]), np.arange(P.shape[1]), indexing='ij')
    Q = np.stack([zoom(P[..., k], up, order=1) for k in range(3)], -1); V = zoom(v.astype(float), up, order=0) > 0.5
    RR = zoom(R.astype(float), up, order=1); CC = zoom(C.astype(float), up, order=1)
    return Q[V], RR[V], CC[V]
Pw, vw = load(a.win); Po, vo = load(a.mesh)
Wq, Wr, Wc = dens(Pw, vw, 2); s = (Wq[:, 2] >= a.z0) & (Wq[:, 2] < a.z1); Wq, Wr, Wc = Wq[s], Wr[s], Wc[s]
Oq, Or, Oc = dens(Po, vo, 8); t = cKDTree(Oq); d, i = t.query(Wq)
res = {'n_winner_pts': int(len(d))}
for name, (z0, z1) in {'top': (7600, 7800), 'awarded': (8300, 8500), 'bottom': (8950, 9150), 'all': (a.z0, a.z1)}.items():
    m = (Wq[:, 2] >= z0) & (Wq[:, 2] < z1)
    if m.any(): res[name] = dict(med=float(np.median(d[m])), p10=float(np.percentile(d[m], 10)), le2=float((d[m] <= 2).mean()), le4=float((d[m] <= 4).mean()), le8=float((d[m] <= 8).mean()))
ok = d <= a.dmax; res['matched_frac'] = float(ok.mean())
if ok.sum() >= 20:
    src = np.c_[Wc[ok] * 20 / F, Wr[ok] * 20 / F].astype(np.float32); dst = np.c_[Oc[i[ok]] * 20 / F, Or[i[ok]] * 20 / F].astype(np.float32)
    A, inl = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=4.0)
    res['affine_w2o'] = A.tolist(); res['affine_inliers'] = float(inl.mean())
    sv = np.linalg.svd(A[:, :2], compute_uv=False); res['affine_scales'] = sv.tolist()
    res['our_box_z20'] = [float(dst[:, 0].min()), float(dst[:, 1].min()), float(dst[:, 0].max()), float(dst[:, 1].max())]
    if a.ours and a.theirs:
        tiles = []; labs = a.labels.split(',') if a.labels else []
        th = [cv2.imread(f, 0) for f in a.theirs.split(',')]; H, W = th[0].shape
        Ai = cv2.invertAffineTransform(A)
        cover = np.zeros((H, W), np.uint8)
        for p in src.astype(int): cv2.circle(cover, tuple(p), 3, 255, -1)
        for k, f in enumerate(a.ours.split(',')):
            o = cv2.imread(f, 0); wimg = cv2.warpAffine(o, Ai, (W, H), flags=cv2.INTER_LINEAR)
            wimg = np.where(cover > 0, wimg, (wimg * 0.35).astype(np.uint8))
            tiles.append(('ours ' + (labs[k] if k < len(labs) else f.split('/')[-1]), wimg))
        for k, im in enumerate(th): tiles.append(('winner ' + (labs[len(a.ours.split(',')) + k] if len(labs) > len(a.ours.split(',')) + k else ''), im))
        rows = []
        for lab, im in tiles:
            c = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); cv2.putText(c, lab, (5, 22), 0, 0.7, (0, 0, 255), 2); rows.append(c)
        cv2.imwrite(a.out + '_side.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
if a.raw:
    Pr, vr = load(a.raw)
    Rq, Rr, Rc = dens(Pr, vr, 4); tr = cKDTree(Rq); dr, ir = tr.query(Wq[ok]) if ok.any() else (np.array([]), np.array([]))
    rr = np.clip(np.rint(Rr[ir]).astype(int), 0, Pr.shape[0] - 2); cc = np.clip(np.rint(Rc[ir]).astype(int), 0, Pr.shape[1] - 2)
    sc = np.linalg.norm(Pr[rr, cc + 1] - Pr[rr, cc], axis=-1) / 20; sr = np.linalg.norm(Pr[rr + 1, cc] - Pr[rr, cc], axis=-1) / 20
    g = vr[rr, cc + 1] & vr[rr, cc] & vr[rr + 1, cc]
    if g.any(): res['raw_stretch_at_place'] = dict(turn=np.percentile(sc[g], [5, 50, 95]).tolist(), z=np.percentile(sr[g], [5, 50, 95]).tolist(), raw_dist_med=float(np.median(dr)))
json.dump(res, open(a.out + '.json', 'w'), indent=1); print(json.dumps(res)[:1500])
