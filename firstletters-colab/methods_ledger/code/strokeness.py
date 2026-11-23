#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/strokeness.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths, job names and every location of unread material
# were removed. The private script hard-coded one row band per target crop; here the band is an argument
# (row range of the map). The per-map arithmetic (function `comps` and the summary) is unchanged.
# Ledger entry: 04_review_and_false_leads.md F1 (strokeness sweep)
# ---------------------------------------------------------------------------------------------------
"""Strokeness of the marks inside a row band of one detector map.

z          = per-map z-score over valid pixels (48 px border set to 0)
hot_band   = share of band pixels with z > 2
component  = 8-connected component of z > 2 after a 0.15 mm elliptical opening, area >= 0.15 mm^2,
             centroid inside the band
strokeness = mean over the 6 largest components of (skeleton length / equivalent diameter),
             skeleton length = number of skeleton pixels x pixel size,
             equivalent diameter = 2 * sqrt(area / pi)
             (a round dot gives about 0.6-1; a stroke or a connected letter group gives well above 1)
skel_mm    = mean skeleton length of those 6 components

Two choices were made while evaluating, after the first run but before the unread targets' numbers were
compared with the control's: the component threshold was lowered from z > 2.5 to z > 2 (z > 2.5 was empty
on the text-dense control crop), and the control's band was defined as all rows spanned by its letter
boxes (connected letters have centroids between boxes). A reviewer should treat both as post-hoc.

usage: strokeness.py MAP.npy VALID.png UM_PER_PX BAND_Y0 BAND_Y1
"""
import sys, json
import numpy as np, cv2
from skimage.morphology import skeletonize


def to_z(prob, valid, border=48):
    p = prob.astype(np.float32); v = valid
    if v.sum() < 100:
        return np.zeros_like(p)
    z = (p - p[v].mean()) / (p[v].std() + 1e-6); z[~valid] = 0
    if border:
        z[:border] = 0; z[-border:] = 0; z[:, :border] = 0; z[:, -border:] = 0
    return z.astype(np.float32)


def comps(z, band, um):
    b = (z > 2.0).astype(np.uint8); r = int(round(150 / um)) | 1
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, 8); out = []
    for i in range(1, n):
        cx, cy = cen[i]
        if not band[int(cy), int(cx)] or st[i][4] * (um / 1000) ** 2 < 0.15: continue
        x, y, w, h, a = st[i]; m = (lab[y:y + h, x:x + w] == i)
        sk = skeletonize(m).sum() * um / 1000; ed = 2 * np.sqrt(a / np.pi) * um / 1000
        mo = cv2.moments(m.astype(np.uint8)); cov = np.array([[mo['mu20'], mo['mu11']], [mo['mu11'], mo['mu02']]]) / max(mo['m00'], 1)
        ev = np.sort(np.linalg.eigvalsh(cov)); el = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        out.append(dict(area=a * (um / 1000) ** 2, skel=sk, ed=ed, el=el))
    return out


def evaluate(p, valid, um, y0, y1):
    v = valid & np.isfinite(p) & (p > 0); z = to_z(np.where(v, p, 0), v)
    band = np.zeros(z.shape, bool); band[y0:y1] = True; band &= v
    allc = comps(z, band, um)
    C = sorted(allc, key=lambda c: -c['area'])[:6]
    return dict(hot_band=round(float((z[band] > 2).mean()), 4), hot_out=round(float((z[v & ~band] > 2).mean()), 4),
                n_comp=len(allc), el=round(float(np.mean([c['el'] for c in C])), 2) if C else None,
                skel_mm=round(float(np.mean([c['skel'] for c in C])), 2) if C else None,
                strokeness=round(float(np.mean([c['skel'] / max(c['ed'], 1e-6) for c in C])), 2) if C else None)


if __name__ == '__main__':
    f, mk, um, y0, y1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    print(json.dumps(evaluate(np.load(f).astype(np.float32), cv2.imread(mk, 0) > 0, um, y0, y1)))
