#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/row_criterion.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths, job names and every location of unread material
# were removed; the z-score helper `to_z` was inlined from our helper module. Numerical part unchanged.
# Ledger entry: 04_review_and_false_leads.md F1 (rows of dots in Studio at stride 48)
# ---------------------------------------------------------------------------------------------------
"""Dot-row criterion, version 2 (the version every published count uses).

row event = >= 6 separate marks, each 0.6-4 mm in largest extent (convex-hull diameter),
            centres on one straight line within +-0.7 mm,
            consecutive along-line gaps <= 6 mm and median gap 2-5 mm,
            present in one plane order and absent in the other
            (< 3 other-order marks within +-0.7 mm of the line over the run, +-1 mm at the ends).
mark      = connected component (8-connectivity) of z > 2.5 after a 0.3 mm elliptical opening,
            z = per-map z-score over valid pixels (48 px border set to 0).

History, stated because it matters for a reviewer: version 1 required marks of 1.5-4 mm. It found 0 events
on the place that prompted the test, so the lower bound was changed to 0.6 mm ON THAT PLACE, before any
control map was counted. Everything else is as first written.

usage: row_criterion.py OUT.json NAME MASK.png ORDER_A.npy ORDER_B.npy [um_per_px]
       (probability maps of the two plane orders on the same grid as the mask)
"""
import sys, json
import numpy as np, cv2


def to_z(prob, valid, border=48):
    """z-score over valid px; 0 off-valid and in a border strip"""
    p = prob.astype(np.float32); v = valid
    if v.sum() < 100:
        return np.zeros_like(p)
    z = (p - p[v].mean()) / (p[v].std() + 1e-6); z[~valid] = 0
    if border:
        z[:border] = 0; z[-border:] = 0; z[:, :border] = 0; z[:, -border:] = 0
    return z.astype(np.float32)


def marks(p, valid, um):
    v = valid & np.isfinite(p) & (p > 0); z = to_z(np.where(v, p, 0), v)
    b = (z > 2.5).astype(np.uint8); r = int(round(300 / um)) | 1
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        pts = np.stack([xs + x, ys + y], 1).astype(np.float32)
        ext = 0.0
        if len(pts) > 2:
            hull = cv2.convexHull(pts); d = hull.reshape(-1, 2); ext = float(np.max(np.linalg.norm(d[:, None] - d[None], axis=2)))
        out.append(dict(cx=float(cen[i][0]) * um / 1000, cy=float(cen[i][1]) * um / 1000, ext_mm=ext * um / 1000, area_mm2=a * (um / 1000) ** 2))
    return out


def events(M, O, tol=0.7):
    C = np.array([[m['cx'], m['cy']] for m in M]) if M else np.zeros((0, 2))
    OC = np.array([[m['cx'], m['cy']] for m in O]) if O else np.zeros((0, 2))
    found = []; seen = set()
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            d = C[j] - C[i]; L_ = np.linalg.norm(d)
            if L_ < 2 or L_ > 40: continue
            e = d / L_; nrm = np.array([-e[1], e[0]])
            off = (C - C[i]) @ nrm; inl = np.nonzero(np.abs(off) <= tol)[0]
            if len(inl) < 6: continue
            s = (C[inl] - C[i]) @ e; o = np.argsort(s); s = s[o]; inl = inl[o]
            # maximal runs with gaps <= 6 mm
            start = 0
            for k in range(1, len(s) + 1):
                if k == len(s) or s[k] - s[k - 1] > 6:
                    run = inl[start:k]; ss = s[start:k]
                    if len(run) >= 6:
                        g = np.diff(ss); mg = float(np.median(g))
                        key = tuple(sorted(run.tolist()))
                        if 2 <= mg <= 5 and key not in seen:
                            seen.add(key)
                            if len(OC):
                                oo = (OC - C[i]) @ nrm; os_ = (OC - C[i]) @ e
                                n_other = int(np.sum((np.abs(oo) <= tol) & (os_ >= ss[0] - 1) & (os_ <= ss[-1] + 1)))
                            else: n_other = 0
                            found.append(dict(n=len(run), median_gap_mm=round(mg, 2), other_order_marks=n_other, key=key,
                                              angle=round(float(np.degrees(np.arctan2(e[1], e[0]))) % 180, 1)))
                    start = k
    # keep maximal (drop runs that are subsets of others)
    found.sort(key=lambda f: -f['n']); keep = []
    for f in found:
        if not any(set(f['key']) <= set(g['key']) for g in keep): keep.append(f)
    for f in keep: f.pop('key')
    return keep


if __name__ == '__main__':
    out, name, mk, fa, fd = sys.argv[1:6]; um = float(sys.argv[6]) if len(sys.argv) > 6 else 9.362
    valid = cv2.imread(mk, 0) > 0
    pa = np.load(fa).astype(np.float32); pd = np.load(fd).astype(np.float32)
    keepmark = lambda ms: [m for m in ms if 0.6 <= m['ext_mm'] <= 4.0]   # version 2
    MA, MD = marks(pa, valid, um), marks(pd, valid, um)
    A, D = keepmark(MA), keepmark(MD)
    ea = events(A, D); ed = events(D, A)
    res = dict(name=name, valid_cm2=round(float(valid.sum()) * (um * 1e-4) ** 2, 3), marks_a=len(A), marks_b=len(D),
               events_a=ea, events_b=ed,
               n_event_a=sum(1 for e in ea if e['other_order_marks'] < 3), n_event_b=sum(1 for e in ed if e['other_order_marks'] < 3))
    try: allr = json.load(open(out))
    except Exception: allr = {}
    allr[name] = res; json.dump(allr, open(out, 'w'), indent=1)
    # The private version also stored the position of every event; positions are deliberately not written here.
    # "Distinct" events in the ledger = events merged by hand when their x ranges overlap and their y ranges
    # lie within 1.5 mm (the merge was done from the stored positions, not by this script).
    print(name, 'cm2', res['valid_cm2'], 'marks a/b', len(A), len(D), 'EVENTS a', res['n_event_a'], 'b', res['n_event_b'])
