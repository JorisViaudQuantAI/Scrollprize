"""pwin.py : Task P window tools (run on the VM, inside wl_env).
 windows C NAME DET MAPDIR OUTJSON   : cut the winding's render canvas into 20 mm (along the turn) x full-height windows,
                                      keep those with valid >= 50 % of the window, run at_atlas post per window
                                      (det map cropped), write per-window line/row/hot/glyph to OUTJSON
 crop C NAME K OUTNAME               : crop render layers + mask + dst of window K into renders/OUTNAME and dst/OUTNAME
Window K of NAME: columns [x0 + K*WPX, x0 + (K+1)*WPX), x0 = first valid column; WPX = 20 mm / 9.362 um."""
import sys, os, json, glob, subprocess, shutil, numpy as np, cv2, tifffile
UM = 9.362; WPX = int(round(20000 / UM))
def wins(mask):
    cols = np.where(mask.any(0))[0]; x0 = cols.min(); W = mask.shape[1]; out = []
    k = 0
    while x0 + k * WPX < W:
        a, b = x0 + k * WPX, min(W, x0 + (k + 1) * WPX)
        rows = np.where(mask.any(1))[0]; H = rows.max() - rows.min() + 1
        v = mask[:, a:b].sum() / float(WPX * H)
        out.append((k, a, b, v)); k += 1
    return out
cmd = sys.argv[1]
if cmd == 'windows':
    C, n, det, mapdir, oj = sys.argv[2:7]
    D = f'{C}/dst/{n}'; mask = cv2.imread(f'{C}/renders/{n}/mask.png', 0) > 0
    mf = sorted(glob.glob(f'{mapdir}/{det}__*_z*.png'))[0]; M = cv2.imread(mf, 0)
    if M.shape != mask.shape: M = cv2.resize(M, mask.shape[::-1])
    res = []
    for k, a, b, v in wins(mask):
        if v < 0.5: continue
        wn = f'{n}_x{k}'; WD = f'{C}/pw/dst/{wn}_{det}'; WM = f'{C}/pw/maps/{wn}_{det}'
        for d in (WD, WM): os.makedirs(d, exist_ok=True)
        for f in ('mask.png', 'ct15.png', 'flag.png'):
            if os.path.exists(f'{D}/{f}'): cv2.imwrite(f'{WD}/{f}', cv2.imread(f'{D}/{f}', 0)[:, a:b])
        pj = json.load(open(f'{D}/prep.json')); pj['mesh'] = wn; json.dump(pj, open(f'{WD}/prep.json', 'w'))
        cv2.imwrite(f'{WM}/{det}__x_z+0.png', M[:, a:b])
        o = f'{C}/pw/out/{wn}_{det}'; os.makedirs(o, exist_ok=True)
        subprocess.run(['python', f'{C}/code_v3/at_atlas.py', 'post', WD, WM, o, '--um', str(UM), '--det', det, '--red', f'{C}/pw/red'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: p = json.load(open(f'{o}/post_{det}.json'))
        except Exception: p = {}
        res.append(dict(window=wn, winding=n, k=k, x0=int(a), x1=int(b), valid_share=round(float(v), 3), det=det,
                        line=p.get('line_max'), row=p.get('row_max'), hot=p.get('hot_frac_z2'), glyph=p.get('glyph_p_max'),
                        valid_cm2=p.get('valid_cm2')))
    json.dump(res, open(oj, 'w'), indent=1); print(n, det, len(res), 'windows')
elif cmd == 'crop':
    C, n, k, on = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    D = f'{C}/dst/{n}'; mask = cv2.imread(f'{C}/renders/{n}/mask.png', 0) > 0
    _, a, b, v = [w for w in wins(mask) if w[0] == k][0]
    R = f'{C}/renders/{n}'; O = f'{C}/renders/{on}'; os.makedirs(f'{O}/layers', exist_ok=True)
    for f in sorted(glob.glob(f'{R}/layers/*.tif')): tifffile.imwrite(f'{O}/layers/{os.path.basename(f)}', tifffile.imread(f)[:, a:b])
    for f in ('mask.png', 'layer_mid.png'):
        if os.path.exists(f'{R}/{f}'): cv2.imwrite(f'{O}/{f}', cv2.imread(f'{R}/{f}', 0)[:, a:b])
    m = json.load(open(f'{R}/render_meta.json')); m['window_cols'] = [int(a), int(b)]; json.dump(m, open(f'{O}/render_meta.json', 'w'))
    OD = f'{C}/dst/{on}'; os.makedirs(OD, exist_ok=True)
    for f in ('mask.png', 'ct15.png', 'flag.png'):
        if os.path.exists(f'{D}/{f}'): cv2.imwrite(f'{OD}/{f}', cv2.imread(f'{D}/{f}', 0)[:, a:b])
    pj = json.load(open(f'{D}/prep.json')); pj['mesh'] = on; json.dump(pj, open(f'{OD}/prep.json', 'w'))
    print('crop', n, k, a, b, round(v, 3))
