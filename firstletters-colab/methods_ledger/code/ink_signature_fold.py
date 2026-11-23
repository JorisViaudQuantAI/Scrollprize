# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/ink_signature_fold.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths were removed; the published surface volume, the
# line grid, the published ink map and the two crop boxes became arguments; the plotting code was dropped.
# The folding, the two depth references and the noise estimate are unchanged.
# Ledger entry: 05_label_free_and_physics.md L6 (ink signature in CT units)
# ---------------------------------------------------------------------------------------------------
"""Line-folded ink signature of a READ scroll (PHerc. Paris 4, a text column of a published 2.4 um surface volume).

Input stack: level 2 of the surface volume (9.6 um in XY, 109 layers of 2.4 um, layer 54 = mesh).
Two depth references: (a) mesh layer as published; (b) per-pixel sheet-aligned: 0 = brightest layer of the
depth-smoothed (sigma 2 layers) profile within +-30 layers of the mesh.
Fold at the column's line grid (pitch P in mm, phase, y origin; all measured on the PUBLISHED ink map).
Sample = one text line x one ~1 mm strip along the line (104 px). For each sample and each depth:
   difference = mean(CT | line-centre band, phase within +-0.1 of the centre) - mean(CT | interline band, phase 0.4-0.6)
Reported: mean and s.e.m. of that difference over samples, in grey levels of the uint8 volume; the same pooled
4 layers at a time (9.6 um in depth); and the per-pixel noise sigma on a blank crop after removing trends
(Gaussian high-pass, sigma 25 px in both lateral axes), at level 2 and at level 0.
Per-ink-pixel conversion used in the ledger: difference / (ink share of centre band - ink share of interline band),
with the shares taken from the published ink map (> 0.5): 0.30 and 0.027.

usage: ink_signature_fold.py SURFACE_VOLUME_URL GRID.json OUT.json  YA YB XA XB  BY0 BY1 BX0 BX1
   SURFACE_VOLUME_URL  the .zarr of a published surface volume on the open-data bucket (levels '0' and '2' are read)
   GRID.json           {"P": pitch_mm, "phase": cycles, "y0": origin in level-3 px}
   YA YB XA XB         text crop in level-2 px;   BY0..BX1  blank crop in level-2 px (x4 for level 0)
"""
import sys, json
import numpy as np, zarr, fsspec
from scipy.ndimage import gaussian_filter1d
URL = sys.argv[1].rstrip('/') + '/'
g = json.load(open(sys.argv[2])); OUT = sys.argv[3]
P, ph, Y0G = g['P'], g['phase'], g['y0']
ya, yb, xa, xb, by0, by1, bx0, bx1 = [int(v) for v in sys.argv[4:12]]
arr = zarr.open_array(store=fsspec.get_mapper(URL + '2'), mode='r')
R = 30
accA = {}; accB = {}
samplesA = []; samplesB = []
for y0 in range(ya, yb, 400):
    a = np.asarray(arr[:, y0:y0 + 400, xa:xb]).astype(np.float32)   # (109, 400, 2000)
    v = (a[54] > 0) & (a[20] > 0) & (a[88] > 0)
    s = gaussian_filter1d(a, 2.0, axis=0)
    c = np.argmax(s[54 - R:54 + R + 1], 0) + 54 - R               # sheet centre per pixel
    # aligned stack: offsets -40..40 around c
    offs = np.arange(-40, 41)
    idx = np.clip(c[None] + offs[:, None, None], 0, 108)
    al = np.take_along_axis(a, idx, 0)                               # (81, 400, 2000)
    yy = (np.arange(y0, y0 + 400) / 2.0 - Y0G) * 0.0192             # mm in the grid frame (L3 px -> mm)
    phase = ((yy / P - ph) % 1.0)
    lineno = np.floor(yy / P - ph).astype(int)
    for st in range(0, xb - xa, 104):                                # ~1 mm strips
        sl = slice(st, st + 104)
        for ln in np.unique(lineno):
            rows = lineno == ln
            cen = rows & ((phase < 0.1) | (phase > 0.9)); itl = rows & (phase > 0.4) & (phase < 0.6)
            if cen.sum() < 5 or itl.sum() < 5:
                continue
            vc = v[cen][:, sl]; vi = v[itl][:, sl]
            if vc.mean() < 0.8 or vi.mean() < 0.8:
                continue
            A_c = (a[:, cen][:, :, sl] * vc).sum((1, 2)) / vc.sum(); A_i = (a[:, itl][:, :, sl] * vi).sum((1, 2)) / vi.sum()
            B_c = (al[:, cen][:, :, sl] * vc).sum((1, 2)) / vc.sum(); B_i = (al[:, itl][:, :, sl] * vi).sum((1, 2)) / vi.sum()
            samplesA.append(A_c - A_i); samplesB.append(B_c - B_i)
    print(y0, len(samplesA), flush=True)
SA = np.array(samplesA); SB = np.array(samplesB)
# per-pixel noise on blank papyrus (agraphon) at L2 and L0, at the mesh layer and at the sheet centre
blank2 = np.asarray(arr[:, by0:by1, bx0:bx1]).astype(np.float32)
arr0 = zarr.open_array(store=fsspec.get_mapper(URL + '0'), mode='r')
blank0 = np.asarray(arr0[:, 4 * by0:4 * by0 + 800, 4 * bx0:4 * bx0 + 2000]).astype(np.float32)
def noise(b):
    hp = b - gaussian_filter1d(gaussian_filter1d(b, 25, axis=1), 25, axis=2)   # remove > 0.2-1 mm trends
    return [float(hp[l].std()) for l in range(109)]
n2, n0 = noise(blank2), noise(blank0)
res = dict(n_samples=len(SA), mesh_mean=SA.mean(0).tolist(), mesh_sem=(SA.std(0) / np.sqrt(len(SA))).tolist(),
           al_offsets=list(range(-40, 41)), al_mean=SB.mean(0).tolist(), al_sem=(SB.std(0) / np.sqrt(len(SB))).tolist(),
           noise_L2=n2, noise_L0=n0)
# pooled to 9.6 um in depth (4 layers)
SA4 = np.stack([SA[:, 4 * k + 1:4 * k + 5].mean(1) for k in range(27)], 1)
res['mesh_pooled_mean'] = SA4.mean(0).tolist(); res['mesh_pooled_sem'] = (SA4.std(0) / np.sqrt(len(SA4))).tolist()
json.dump(res, open(OUT, 'w'))
k = int(np.argmax(np.abs(SB.mean(0)))); print('aligned max |diff| at offset', k - 40, SB.mean(0)[k], '+-', SB.std(0)[k] / np.sqrt(len(SB)))
k2 = int(np.argmax(np.abs(SA.mean(0)))); print('mesh max |diff| at layer', k2, SA.mean(0)[k2], '+-', SA.std(0)[k2] / np.sqrt(len(SA)))
print('noise L2 at layer 54', n2[54], 'L0', n0[54])
