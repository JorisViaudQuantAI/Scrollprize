#!/usr/bin/env python3
"""xsect.py MESHDIR OUTDIR --ct CTZARR --level L --z Z [Z ...] [--crop y0 x0 h w (full-res)] : axial CT slices with the
intersection of every winding drawn on top (points of the tifxyz grid within +-dz of the slice; consecutive windings
alternate colours). Writes xsect_z<Z>_L<L>[_crop].jpg"""
import argparse, glob, os, colorsys
import numpy as np, tifffile, zarr
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument('meshes'); ap.add_argument('out')
ap.add_argument('--ct', required=True); ap.add_argument('--level', type=int, default=2)
ap.add_argument('--z', type=int, nargs='+', required=True); ap.add_argument('--crop', type=int, nargs=4, default=None)
ap.add_argument('--glob', default='w[0-9][0-9][0-9]*'); ap.add_argument('--tag', default=''); ap.add_argument('--maxpx', type=int, default=1800); ap.add_argument('--up', type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
s = 2 ** a.level; ct = zarr.open(a.ct, mode='r')[str(a.level)]
dirs = sorted(d for d in glob.glob(os.path.join(a.meshes, a.glob)) if os.path.isfile(d + '/x.tif') and '_spliced' not in d)
W = []
for d in dirs:
    X = tifffile.imread(d + '/x.tif'); Y = tifffile.imread(d + '/y.tif'); Z = tifffile.imread(d + '/z.tif')
    W.append((os.path.basename(d), X, Y, Z))
for z in a.z:
    if a.crop: y0, x0, h, w = a.crop
    else: y0, x0, h, w = 0, 0, ct.shape[1] * s, ct.shape[2] * s
    img = np.asarray(ct[z // s, y0 // s:(y0 + h) // s, x0 // s:(x0 + w) // s]).astype(np.uint8)
    fct = min(1.0, a.maxpx / max(img.shape)) * a.up
    im = Image.fromarray(img).convert('RGB')
    if fct != 1: im = im.resize((int(img.shape[1] * fct), int(img.shape[0] * fct)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for i, (name, X, Y, Z) in enumerate(W):
        # interpolate along each grid column (fixed theta) to the slice z
        r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1, 1, 1); col = (int(r * 255), int(g * 255), int(b * 255))
        valid = (X > -0.5) & (Y > -0.5) & (Z > -0.5)
        Zm = np.where(valid, Z, np.nan)
        pts = []
        for j in range(Z.shape[1]):
            zc = Zm[:, j]; f = np.isfinite(zc)
            if f.sum() < 2: pts.append(None); continue
            k = np.searchsorted(zc[f], z)
            if k <= 0 or k >= f.sum(): pts.append(None); continue
            zi = zc[f]; t = (z - zi[k - 1]) / max(zi[k] - zi[k - 1], 1e-6)
            yy = Y[f, j][k - 1] * (1 - t) + Y[f, j][k] * t; xx = X[f, j][k - 1] * (1 - t) + X[f, j][k] * t
            pts.append(((xx - x0) / s * fct, (yy - y0) / s * fct))
        seg = []
        for p in pts + [None]:
            if p is None:
                if len(seg) > 1: dr.line(seg, fill=col, width=1)
                seg = []
            else: seg.append(p)
        if pts and any(p is not None for p in pts):
            p = next(p for p in pts if p is not None); dr.text(p, name[:4], fill=col)
    fn = os.path.join(a.out, f'xsect{a.tag}_z{z}_L{a.level}' + ('_crop' if a.crop else '') + '.jpg')
    im.save(fn, quality=88); print(fn, im.size)
