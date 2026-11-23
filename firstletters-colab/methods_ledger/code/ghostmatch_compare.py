# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/ghostmatch_compare.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths were removed, the hard-coded canvas boxes and map
# origins became arguments, and the figure-drawing part was dropped. The correspondence field, the
# smoothing, the NCC and the list of nulls are unchanged.
# Ledger entry: 04_review_and_false_leads.md F5 (ghost text from the neighbouring wrap, PHerc. Paris 4)
# ---------------------------------------------------------------------------------------------------
"""Ghost letter-match check, steps 3-4: warp the outer turn's own ink map into the coordinates of the tested
strip and compare it with the map in which letter rows appeared at an outward depth shift.

TARGET  = detector map of the tested strip with the reading window moved outward (the 'ghost' map),
          on a regular grid: origin (T_X0, T_Y0) in canvas px, T_PX canvas px per map pixel.
SOURCE  = an ink map of the outer turn at its OWN depth (we used two: reader-v3 at shift 0, and the
          organisers' published 2-D ink map), origin (S_X0, S_Y0), S_PX canvas px per map pixel.
GEOM    = output of ghostmatch_geom.py (point of the strip + shift along the outward normal -> nearest point
          on the outer turn).
The correspondence is median-cleaned (nearest-point jumps inside the fold) and Gaussian-smoothed at ~2 mm.
Orientation: the map was orientation-preserving (Jacobian ~ identity), so NO mirror is applied to the 'real'
comparison; a mirrored source is one of the nulls.
NCC on Gaussian-smoothed maps (sigma 0.15 mm) on an 8-canvas-px grid.
Nulls (19): source shifted by +-1, +-2 line pitches in y; by +-3, 4, 5, 6, 8, 10 mm in x; mirrored in x about
the strip; mirrored and shifted by +-1 pitch.
Each statistic is reported at zero residual shift (r0) and as the best over a +-1 mm residual search (rb).

usage: ghostmatch_compare.py GEOM.npz TARGET.npy T_X0 T_Y0 T_PX SOURCE.npy S_X0 S_Y0 S_PX
                             STRIP_X0 STRIP_X1 STRIP_Y0 STRIP_Y1 PITCH_PX [--voxel-um 2.4]
"""
import sys
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, median_filter

a = sys.argv
G = np.load(a[1])
Tm = np.load(a[2]).astype(np.float32); T_X0, T_Y0, T_PX = float(a[3]), float(a[4]), float(a[5])
Sm = np.load(a[6]).astype(np.float32); S_X0, S_Y0, S_PX = float(a[7]), float(a[8]), float(a[9])
X0, X1, Y0, Y1 = [int(v) for v in a[10:14]]; pitch = float(a[14])
VOX = float(a[a.index('--voxel-um') + 1]) if '--voxel-um' in a else 2.4
if Sm.max() > 1.5: Sm /= 255.
S = 8                                   # canvas px per comparison pixel (19.2 um at 2.4 um voxels)
ty, tx = np.mgrid[Y0:Y1:S, X0:X1:S].astype(np.float32)
T = map_coordinates(Tm, [(ty - T_Y0) / T_PX, (tx - T_X0) / T_PX], order=1)


def field(arr, ref):
    off = arr - ref
    m = np.isfinite(off); med = np.nanmedian(off)
    o = np.where(m, off, med)
    md = median_filter(o, size=21, mode="nearest")
    bad = np.abs(o - md) > 60          # canvas px
    print('field outliers replaced: %.2f %%' % (100 * bad.mean()))
    o = np.where(bad, md, o)
    return gaussian_filter(o, 4.0)     # ~2 mm; the residual +-1 mm search covers the rest


ox = field(G['xo'], G['gx']); oy = field(G['yo'], G['gy'])
gy0, gx0 = G['gy'][0, 0], G['gx'][0, 0]
ci = [(ty - gy0) / 40, (tx - gx0) / 40]
XO = tx + map_coordinates(ox, ci, order=1, mode='nearest'); YO = ty + map_coordinates(oy, ci, order=1, mode='nearest')


def sample(dx=0., dy=0., mirror=False):
    xs = XO + dx; ys = YO + dy
    if mirror:
        xc = np.median(XO); xs = 2 * xc - xs
    return map_coordinates(Sm, [(ys - S_Y0) / S_PX, (xs - S_X0) / S_PX], order=1, cval=np.nan)


SIG = 0.15e3 / (S * VOX)   # px


def sm(arr):
    m = np.isfinite(arr).astype(np.float32)
    w = gaussian_filter(m, SIG)
    o = gaussian_filter(np.nan_to_num(arr), SIG) / np.maximum(w, 1e-3)
    o[w < 0.5] = np.nan
    return o


def ncc(p, q, m):
    ok = m & np.isfinite(p) & np.isfinite(q)
    if ok.sum() < 1000: return np.nan
    p = p[ok] - p[ok].mean(); q = q[ok] - q[ok].mean()
    return float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum() + 1e-12))


region = np.ones(tx.shape, bool)
mm = 1000 / VOX   # canvas px per mm
res_shift = np.arange(-1.0, 1.01, 0.25) * mm
Ts = sm(T)


def stat(dx=0, dy=0, mirror=False):
    r0 = ncc(Ts, sm(sample(dx, dy, mirror)), region)
    rb = max(ncc(Ts, sm(sample(dx + ex, dy + ey, mirror)), region) for ex in res_shift[::2] for ey in res_shift[::2])
    return r0, rb


r0, rb = stat(); print(f'REAL (mapped, no mirror): r0 {r0:.3f}  rbest(+-1mm) {rb:.3f}')
nul0, nulb = [], []
for k in (-2, -1, 1, 2):
    p0, pb = stat(dy=k * pitch); nul0.append(p0); nulb.append(pb); print(f'  null dy {k:+d} pitch: r0 {p0:.3f} rb {pb:.3f}')
for s in (-10, -8, -6, -5, -4, -3, 3, 4, 5, 6, 8, 10):
    p0, pb = stat(dx=s * mm); nul0.append(p0); nulb.append(pb); print(f'  null dx {s:+d} mm: r0 {p0:.3f} rb {pb:.3f}')
p0, pb = stat(mirror=True); nul0.append(p0); nulb.append(pb); print(f'  null mirrored: r0 {p0:.3f} rb {pb:.3f}')
for k in (-1, 1):
    p0, pb = stat(dy=k * pitch, mirror=True); nul0.append(p0); nulb.append(pb)
    print(f'  null mirrored + dy {k:+d} pitch: r0 {p0:.3f} rb {pb:.3f}')
n0 = np.array(nul0); nb = np.array(nulb)
print(f'NULL r0: mean {np.nanmean(n0):.3f} sd {np.nanstd(n0):.3f} max {np.nanmax(n0):.3f} ; '
      f'NULL rbest: mean {np.nanmean(nb):.3f} sd {np.nanstd(nb):.3f} max {np.nanmax(nb):.3f}')
print(f'z(real r0) = {(r0 - np.nanmean(n0)) / np.nanstd(n0):.1f} ; z(real rbest) = {(rb - np.nanmean(nb)) / np.nanstd(nb):.1f}')
