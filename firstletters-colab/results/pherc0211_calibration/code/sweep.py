"""sweep.py DIR NAME ORACLE.npy|none META.json OUTDIR : depth-sweep composites from per-shift maps DIR/sw__*_z{s}.png
 max      : plain max over shifts
 stitch   : per block (1.25 mm) shift maximising the hot fraction (pixels > 0.5 prob), block field median-filtered 3x3
 oracleP / oracleM : per pixel the shift nearest to +/- the oracle signed distance (NaN -> 0)
Writes OUTDIR/<det>__x_z+0.png (full res, for at_atlas post), OUTDIR/<NAME>_<det>_v.jpg (20 px/mm view), stats json."""
import sys, glob, re, json, numpy as np, cv2
from scipy.ndimage import median_filter
D, name, orc, meta, out = sys.argv[1:6]
fs = sorted(glob.glob(f'{D}/sw__*_z*.png'), key=lambda f: float(re.search(r'_z([+-]?\d+)', f).group(1)))
sh = np.array([float(re.search(r'_z([+-]?\d+)', f).group(1)) for f in fs])
A = np.stack([cv2.imread(f, 0) for f in fs]).astype(np.uint8)  # S,H,W
S, H, W = A.shape; um = 9.362; B = int(round(1250 / um))
res = {'shifts': sh.tolist(), 'block_px': B}
mx = A.max(0)
# stitch
nby, nbx = -(-H // B), -(-W // B); score = np.zeros((S, nby, nbx))
for k in range(S):
    hot = (A[k] > 127).astype(np.float32)
    for by in range(nby):
        for bx in range(nbx):
            score[k, by, bx] = hot[by * B:(by + 1) * B, bx * B:(bx + 1) * B].mean()
idx = score.argmax(0); idx = median_filter(idx, size=3, mode='nearest')
full = cv2.resize(idx.astype(np.uint8), (nbx * B, nby * B), interpolation=cv2.INTER_NEAREST)[:H, :W]
st = np.take_along_axis(A, full[None].astype(np.int64), 0)[0]
res['stitch_shift_hist'] = {str(int(sh[i])): int((idx == i).sum()) for i in range(S)}
outs = {'swmax': mx, 'swstitch': st}
if orc != 'none':
    O = np.load(orc); m = json.load(open(meta)); gx0, gy0, gw, gh = m['grid_crop_xywh']; O = O[gy0:gy0 + gh, gx0:gx0 + gw]
    Ou = cv2.resize(np.nan_to_num(O, nan=0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    Vu = cv2.resize(np.isfinite(O).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    for sg, tag in ((1, 'oracleP'), (-1, 'oracleM')):
        k = np.abs(sh[:, None, None] - sg * Ou[None]).argmin(0); k[~Vu] = int(np.abs(sh).argmin())
        outs['sw' + tag] = np.take_along_axis(A, k[None], 0)[0]
        # agreement of stitched shift with oracle (blocks with oracle info), within one step (3 vox)
        ob = cv2.resize(np.where(Vu, sg * Ou, np.nan), (nbx, nby), interpolation=cv2.INTER_AREA)
        ok = np.isfinite(ob); res[f'stitch_vs_{tag}_within3'] = float((np.abs(sh[idx][ok] - ob[ok]) <= 3).mean()) if ok.any() else None
        res[f'oracle_block_frac'] = float(ok.mean())
z0 = int(np.abs(sh).argmin()); outs['sw0'] = A[z0]
f20 = 1000 / um / 20
for det, im in outs.items():
    cv2.imwrite(f'{out}/{det}__x_z+0.png', im)
    v = cv2.resize(im, (int(W / f20), int(H / f20)), interpolation=cv2.INTER_AREA); cv2.imwrite(f'{out}/{name}_{det}_v.jpg', v)
    res[det + '_hot'] = float((im > 127).mean())
json.dump(res, open(f'{out}/{name}_sweep.json', 'w'), indent=1); print(json.dumps(res))
