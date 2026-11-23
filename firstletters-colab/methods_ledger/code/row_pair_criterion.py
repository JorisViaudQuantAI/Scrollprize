# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/row_pair_criterion.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths, job names and every location of unread material
# were removed; the numerical part is unchanged.
# Ledger entry: 04_review_and_false_leads.md F2
# ---------------------------------------------------------------------------------------------------
"""row_pair_criterion.py - fixed row-pair criterion (written before any base-rate map was evaluated):
>= 2 parallel rows, each >= 10 mm of connected or closely spaced (gap <= 1.5 mm) stroke elements 1-3 mm high,
row pitch 3-8 mm, in one order and not the other (other order at the same tile and angle +-6 deg must not pass),
present at >= 2 of the 3 depth shifts available (when only s0 exists, reported as 'single-shift pass').
Implementation: Studio stride-48 probability, area-mean to ~37 um, hot = p > 0.5; 20 x 20 mm tiles, step 2.5 mm; line angle
0..177 step 3 deg; components of hot with perpendicular extent 1-3 mm and area >= 0.3 mm^2 are stroke elements; rows = peaks
of the element profile (smoothed 0.5 mm, >= 3 mm apart); a row's length = longest run of element x-intervals merged with
gaps <= 1.5 mm inside a +-1.5 mm band.
"""
import numpy as np, cv2, math

PX_WORK = 37.0


def prep(p, um):
    f = max(1, int(round(PX_WORK / um)))
    h, w = p.shape[0] // f, p.shape[1] // f
    q = cv2.resize(p[:h * f, :w * f].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    return (q > 0.5).astype(np.uint8), um * f


def rows_in(hot, umw, theta):
    pxmm = 1000.0 / umw
    h, w = hot.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    r = cv2.warpAffine(hot, M, (w, h), flags=cv2.INTER_NEAREST)
    n, lab, st, cen = cv2.connectedComponentsWithStats(r, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        hh = st[i, cv2.CC_STAT_HEIGHT] / pxmm
        if 1.0 <= hh <= 3.0 and st[i, cv2.CC_STAT_AREA] / pxmm ** 2 >= 0.3:
            keep[i] = True
    if keep.sum() < 4:
        return []
    el = keep[lab]
    prof = el.sum(1).astype(np.float32)
    prof = cv2.GaussianBlur(prof[:, None], (1, 0), sigmaX=0, sigmaY=0.5 * pxmm).ravel()
    pk = [i for i in range(len(prof)) if prof[i] > 0 and prof[i] == prof[max(0, i - int(3 * pxmm)):i + int(3 * pxmm) + 1].max()]
    out = []
    for y in pk:
        band = [i for i in range(1, n) if keep[i] and abs(cen[i][1] - y) <= 1.5 * pxmm]
        iv = sorted((st[i, 0], st[i, 0] + st[i, 2]) for i in band)
        best, cur = 0, None
        for a, b in iv:
            if cur and a - cur[1] <= 1.5 * pxmm:
                cur = (cur[0], max(cur[1], b))
            else:
                cur = (a, b)
            best = max(best, cur[1] - cur[0])
        out.append((y / pxmm, best / pxmm))
    return out


def tile_pass(hot, umw, theta):
    rows = [r for r in rows_in(hot, umw, theta) if r[1] >= 10.0]
    rows.sort()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if 3.0 <= rows[j][0] - rows[i][0] <= 8.0:
                return True, rows
    return False, rows_in(hot, umw, theta)


def evaluate(maps, um, main_key, other_key, shift_keys=(), valid=None):
    """maps: dict of 2-D prob maps (same grid). Returns list of tile results and summary."""
    hot = {k: prep(np.where(valid, m, 0) if valid is not None else m, um) for k, m in maps.items()}
    H0, umw = hot[main_key]
    pxmm = 1000.0 / umw
    T, S = int(20 * pxmm), int(2.5 * pxmm)
    res = []
    h, w = H0.shape
    for y in range(0, max(1, h - T + 1), S):
        for x in range(0, max(1, w - T + 1), S):
            t = H0[y:y + T, x:x + T]
            if t.mean() < 0.005:
                continue
            best = None; pair = (0.0, None)
            for th in range(0, 180, 3):
                ok, rows = tile_pass(t, umw, th)
                rr = sorted(rows_in(t, umw, th))
                for i in range(len(rr)):
                    for j in range(i + 1, len(rr)):
                        if 3.0 <= rr[j][0] - rr[i][0] <= 8.0 and min(rr[i][1], rr[j][1]) > pair[0]:
                            pair = (float(min(rr[i][1], rr[j][1])), th)
                long_rows = sorted([r for r in rows if r[1] >= 10.0])
                score = (int(ok), len(long_rows), max([r[1] for r in rows] or [0]))
                if best is None or score > best[0]:
                    best = (score, th, ok, rows)
            score, th, ok, rows = best
            rec = dict(y_mm=round(y / pxmm, 1), x_mm=round(x / pxmm, 1), theta=th, main_pass=ok, n_long_rows=score[1],
                       longest_mm=round(score[2], 1), pair_mm=round(pair[0], 1), pair_theta=pair[1])
            if ok:
                ot = hot[other_key][0][y:y + T, x:x + T]
                rec["other_pass"] = any(tile_pass(ot, umw, (th + d) % 180)[0] for d in range(-6, 7, 3))
                sp = []
                for k in shift_keys:
                    stt = hot[k][0][y:y + T, x:x + T]
                    sp.append(any(tile_pass(stt, umw, (th + d) % 180)[0] for d in range(-6, 7, 3)))
                rec["shift_pass"] = sp
                rec["criterion"] = (not rec["other_pass"]) and ((1 + sum(sp)) >= 2 if shift_keys else None)
            res.append(rec)
    return res
