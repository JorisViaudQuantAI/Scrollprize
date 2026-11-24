#!/usr/bin/env python3
"""cmp_tracks.py MINE_STORE PUB_STORE --z0 --z1 --core0 --core1 --zov Z --bits BITS.npy --out DIR
Compare our packed track store with the published one in the same sub-band.
Horizontal (family 0): tracks whose z_bounds lie inside [z0, z1): count, points, points/track, arclength.
Vertical (family 1): points with z inside the core [core0, core1) (away from our sub-band ends), per family and per slab
position (slab = y//16*16 for zx slabs / x for zy slabs is not stored, so we bin by the constant coordinate).
Overlay: horizontal tracks of ribbon z = zov (mine green, published red, both = yellow-ish) + vertical points at zov."""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument('mine'); ap.add_argument('pub')
for k in ('z0', 'z1', 'core0', 'core1', 'zov'): ap.add_argument('--' + k, type=int)
ap.add_argument('--bits'); ap.add_argument('--out'); ap.add_argument('--crop', type=int, nargs=4, default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)


def load(d):
    off = np.fromfile(d + '/offsets.i64', '<i8'); zb = np.fromfile(d + '/z_bounds.i32', '<i4').reshape(-1, 2)
    fam = np.fromfile(d + '/family_codes.i8', 'i1'); arc = np.fromfile(d + '/arclengths.f64', '<f8')
    co = np.memmap(d + '/coordinates.i32', '<i4', 'r').reshape(-1, 3)
    return off, zb, fam, arc, co


R = {}; S = {}
for name, d in (('mine', a.mine), ('pub', a.pub)):
    off, zb, fam, arc, co = load(d)
    h = np.nonzero((fam == 0) & (zb[:, 0] >= a.z0) & (zb[:, 1] < a.z1))[0]
    n = off[h + 1] - off[h]
    r = dict(h_tracks=int(len(h)), h_points=int(n.sum()))
    for q in (10, 50, 90, 99): r[f'h_pts_p{q}'] = float(np.percentile(n, q)) if len(n) else None
    for q in (10, 50, 90): r[f'h_arc_p{q}'] = float(np.percentile(arc[h], q)) if len(h) else None
    hz = {}
    for t in h:
        z = int(co[off[t], 0]); hz[z] = hz.get(z, 0) + 1
    r['h_tracks_per_ribbon_median'] = float(np.median(list(hz.values()))) if hz else None
    # vertical: tracks overlapping the core
    v = np.nonzero((fam == 1) & (zb[:, 1] >= a.core0) & (zb[:, 0] < a.core1))[0]
    pts = []; nv = 0
    ys = []; xs = []
    for t in v:
        c = np.asarray(co[off[t]:off[t + 1]])
        m = (c[:, 0] >= a.core0) & (c[:, 0] < a.core1)
        if m.any(): nv += 1; pts.append(c[m])
    P = np.concatenate(pts) if pts else np.zeros((0, 3), np.int32)
    r['v_tracks_touching_core'] = int(nv); r['v_points_in_core'] = int(len(P))
    vl = off[v + 1] - off[v]
    for q in (10, 50, 90): r[f'v_pts_p{q}_whole_track'] = float(np.percentile(vl, q)) if len(vl) else None
    # per-slab point counts: zx slabs have y in [w, w+4) (bin y//16), zy slabs x in [w, w+4) (bin x//16)
    R[name] = r; S[name] = (h, off, co, P)
    print(name, json.dumps(r))
json.dump(R, open(os.path.join(a.out, 'cmp_summary.json'), 'w'), indent=1)

# vertical per-slab comparison: split the core points by which coordinate is slab-constant: a point belongs to a zx slab
# if y % 16 < 4 relative to lo; ambiguous points are rare. Use the published lo (384, 0) and ours from run_params.json.
def slab_hist(P, lo_y, lo_x):
    iy = (P[:, 1] - lo_y) % 16 < 4; ix = (P[:, 2] - lo_x) % 16 < 4
    hy = np.bincount(((P[iy, 1] - lo_y) // 16).astype(int), minlength=700)
    hx = np.bincount(((P[ix, 2] - lo_x) // 16).astype(int), minlength=700)
    return hy[:700], hx[:700]
rp = json.load(open(os.path.join(os.path.dirname(a.mine.rstrip('/')), 'run_params.json'))) if os.path.isfile(os.path.join(os.path.dirname(a.mine.rstrip('/')), 'run_params.json')) else {'lo': [384, 0]}
my_y, my_x = slab_hist(S['mine'][3], *rp['lo']); pu_y, pu_x = slab_hist(S['pub'][3], 384, 0)
cy = np.corrcoef(my_y, pu_y)[0, 1]; cx = np.corrcoef(my_x, pu_x)[0, 1]
print(f'per-slab vertical point counts in core: corr zx {cy:.3f} zy {cx:.3f}; totals mine {my_y.sum()+my_x.sum()} pub {pu_y.sum()+pu_x.sum()}')
with open(os.path.join(a.out, 'cmp_summary.json'), 'a') as f: f.write(f'\n# per-slab corr zx {cy:.3f} zy {cx:.3f}\n')

# overlay at ribbon z = zov
bits = np.load(a.bits, mmap_mode='r'); meta = json.load(open(a.bits + '.json'))
bg = np.unpackbits(bits[a.zov - meta['z0']], axis=-1, count=meta['shape'][2]).astype(np.uint8) * 90
if a.crop: y0, x0, hh, ww = a.crop
else: y0, x0, hh, ww = 0, 0, bg.shape[0], bg.shape[1]
base = Image.fromarray(bg[y0:y0 + hh, x0:x0 + ww]).convert('RGB'); panels = []
for name, col, vcol in (('pub', (255, 60, 60), (60, 160, 255)), ('mine', (60, 255, 60), (255, 255, 0))):
    im = base.copy(); dr = ImageDraw.Draw(im)
    h, off, co, P = S[name]
    for t in h:
        c = np.asarray(co[off[t]:off[t + 1]])
        if c[0, 0] not in (a.zov, a.zov + 1): continue
        pts = [(int(x - x0), int(y - y0)) for z, y, x in c]
        if len(pts) > 1: dr.line(pts, fill=col, width=1)
    m = np.abs(P[:, 0] - a.zov) <= 1
    for z, y, x in P[m]:
        dr.point((int(x - x0), int(y - y0)), fill=vcol)
    dr.text((5, 5), name, fill=(255, 255, 255)); panels.append(im)
W = Image.new('RGB', (2 * base.size[0] + 10, base.size[1]))
W.paste(panels[0], (0, 0)); W.paste(panels[1], (base.size[0] + 10, 0))
fn = os.path.join(a.out, f'overlay_z{a.zov}.png'); W.save(fn); print(fn, W.size)
