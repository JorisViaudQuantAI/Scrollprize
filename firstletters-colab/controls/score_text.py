#!/usr/bin/env python
"""Objective text-likeness score for an ink-probability map (First Letters triage).

Given an ink map (0..1 after rescaling), a validity mask, the pixel size and the
papyrus fibre directions of the render, every square window (default 20 mm = the
4 cm^2 prize area) gets these components:

  period   row periodicity. The map is rotated so that a candidate writing direction is
           horizontal, the masked mean over each row gives a profile across the lines,
           and its detrended power spectrum is searched for a peak at a line pitch of
           4-7 mm. period = peak power / mean power of the other frequencies (pitch
           0.8-20 mm). Candidate directions: each fibre direction +-6 deg (or 0..175 deg
           when no fibre direction is known). The best direction is kept.
  rowmod   modulation depth of that profile: std of the band-passed profile / mean ink
           level, i.e. how much more ink sits in the rows than between them.
  stroke   fraction of the thresholded ink area that lies in stroke-sized components
           (length 0.6-5 mm, width 0.12-1.5 mm, not a fibre-aligned streak).
  orient   fraction of stroke area whose orientation is NOT within 12 deg of a fibre
           direction (letters have verticals, diagonals and curves; fibre artefacts
           follow the grid).
  cover    fraction of the window above the ink threshold.
  stab_*   stability: Dice overlap of the thresholded map with each companion map
           (other models, neighbouring depth offsets).
  asym     direction asymmetry (cover_fwd - cover_rev) / (cover_fwd + cover_rev) when the
           reverse-order map is given. Its sign depends on the checkpoint, surface and depth order.

The combined score is documented in README.md (it is a calibrated z-score sum, see
calibrate_scorer.py). This file only depends on numpy / scipy / opencv / scikit-image.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import measure

PITCH_MM = (4.0, 7.0)
BG_PITCH_MM = (0.8, 20.0)
STROKE_LEN_MM = (0.8, 4.5)
STROKE_W_MM = (0.25, 3.5)


# ----------------------------------------------------------------------------- io
def load_map(path, rescale=None):
    """uint8/uint16 image -> float32 0..1. rescale=(lo, hi) maps raw probability lo..hi to 0..1."""
    a = cv2.imread(str(path), -1)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
    if rescale:
        lo, hi = rescale
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    return a


def fibre_angles(ct, mask=None, px_um=9.0):
    """Two dominant fibre orientations (deg, 0..180, image x axis = 0, y down) of a CT surface image."""
    img = ct.astype(np.float32)
    s = max(1.0, 40.0 / px_um)  # ~40 um
    img = cv2.GaussianBlur(img, (0, 0), s)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    S = max(2.0, 150.0 / px_um)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), S)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), S)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), S)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # gradient direction
    fib = (np.degrees(theta) + 90.0) % 180.0  # fibre runs perpendicular to the gradient
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = lam / (jxx + jyy + 1e-6)
    w = coh * np.sqrt(jxx + jyy)
    if mask is not None:
        w = w * (mask > 0)
    h, _ = np.histogram(fib, bins=180, range=(0, 180), weights=w)
    h = np.convolve(np.r_[h[-5:], h, h[:5]], np.ones(5) / 5, "same")[5:-5]
    a1 = int(np.argmax(h))
    d = np.abs(((np.arange(180) - a1 + 90) % 180) - 90)
    h2 = np.where(d >= 45, h, -1)
    a2 = int(np.argmax(h2))
    return float(a1), float(a2), float(h[a2] / max(h[a1], 1e-9))


# ----------------------------------------------------------------------------- components
def _rotate(a, angle_deg, border=0.0):
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def row_periodicity(p, m, px_mm, angle_deg):
    """Return (period, rowmod, pitch_mm) for writing direction angle_deg (deg, image coords)."""
    # cv2 rotates counter-clockwise for positive angles with y down -> rotate by +angle makes a line at angle horizontal
    pr = _rotate(p * m, angle_deg)
    mr = _rotate(m.astype(np.float32), angle_deg)
    rs = mr.sum(1)
    good = rs > 0.4 * mr.shape[1]
    rows = np.flatnonzero(good)
    if rows.size * px_mm < 1.9 * PITCH_MM[1]:
        return 0.0, 0.0, 0.0
    # Trim only the outer margins. One array index still represents one physical row.
    span = slice(rows[0], rows[-1] + 1)
    good = good[span]
    row_sum = pr.sum(1)[span]
    prof = np.zeros(row_sum.shape, dtype=np.result_type(row_sum, rs))
    np.divide(row_sum, rs[span], out=prof, where=good)
    n = len(prof)
    sig = 2.0 * PITCH_MM[1] / px_mm / 2.355

    def smooth(values):
        column = values.astype(np.float32).reshape(-1, 1)
        return cv2.GaussianBlur(column, (1, 0), sig, borderType=cv2.BORDER_REFLECT).ravel()

    trend = smooth(prof)
    if not good.all():
        # Normalize the trend by its observed support so a tear does not become a dark stripe.
        trend[good] /= smooth(good)[good]
    residual = prof - trend
    residual[~good] = 0
    x = residual * np.hanning(n)
    nfft = 1 << int(math.ceil(math.log2(n * 4)))
    P = np.abs(np.fft.rfft(x, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, d=px_mm)  # cycles per mm
    with np.errstate(divide="ignore"):
        pitch = np.where(f > 0, 1.0 / f, np.inf)
    band = (pitch >= PITCH_MM[0]) & (pitch <= PITCH_MM[1])
    bg = (pitch >= BG_PITCH_MM[0]) & (pitch <= BG_PITCH_MM[1]) & ~band
    if band.sum() == 0 or bg.sum() == 0:
        return 0.0, 0.0, 0.0
    k = np.argmax(np.where(band, P, -1))
    period = float(P[k] / (P[bg].mean() + 1e-12))
    # modulation of the line-pitch component
    lo, hi = PITCH_MM
    fp = np.fft.rfft(residual, nfft)
    fp[~band] = 0
    bp = np.fft.irfft(fp, nfft)[:n]
    rowmod = float(bp[good].std() / (prof[good].mean() + 1e-6))
    return period, rowmod, float(pitch[k])


def threshold_map(p, m, thr):
    b = (cv2.GaussianBlur(p, (0, 0), 1.0) >= thr) & (m > 0)
    return b


def stroke_stats(b, px_mm, fibre=None):
    lab = measure.label(b, connectivity=2)
    tot = float(b.sum())
    if tot == 0:
        return 0.0, 0.0, 0
    stroke_area = 0.0
    off_axis = 0.0
    n_stroke = 0
    for r in measure.regionprops(lab):
        if r.area < 4:
            continue
        L = r.axis_major_length * px_mm
        W = max(r.axis_minor_length * px_mm, px_mm)
        ang = (-math.degrees(r.orientation) + 90.0) % 180.0  # skimage orientation: angle between row axis and major axis
        aligned = False
        if fibre:
            aligned = any(abs(((ang - fa + 90) % 180) - 90) < 12 for fa in fibre)
        streak = L > STROKE_LEN_MM[1] or (L / W > 6 and aligned)
        if STROKE_LEN_MM[0] <= L <= STROKE_LEN_MM[1] and STROKE_W_MM[0] <= W <= STROKE_W_MM[1] and not streak:
            stroke_area += r.area
            n_stroke += 1
            if not aligned or L / W < 2:
                off_axis += r.area
    return stroke_area / tot, (off_axis / stroke_area if stroke_area else 0.0), n_stroke


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 0.0


# ----------------------------------------------------------------------------- windows
def score_windows(p, m, px_um, fibre=None, companions=(), reverse=None, thr=0.5, win_mm=20.0, step_mm=5.0,
                  min_valid=0.5, angle_step=3.0):
    """Score every window. p, m: arrays on the same grid. companions: list of (name, map). reverse: map or None."""
    px_mm = px_um / 1000.0
    H, W = p.shape
    win = int(round(win_mm / px_mm))
    winy, winx = min(win, H), min(win, W)
    step = max(1, int(round(step_mm / px_mm)))
    ys = list(range(0, max(1, H - winy + 1), step)) or [0]
    xs = list(range(0, max(1, W - winx + 1), step)) or [0]
    if ys[-1] + winy < H:
        ys.append(H - winy)
    if xs[-1] + winx < W:
        xs.append(W - winx)
    if fibre:
        angles = sorted({round((fa + d) % 180, 1) for fa in fibre for d in np.arange(-6, 6.01, angle_step)})
    else:
        angles = list(np.arange(0, 180, 5.0))
    m = (m > 0).astype(np.float32)
    B = threshold_map(p, m, thr)
    Bc = [(n, threshold_map(c, m, thr)) for n, c in companions]
    Br = threshold_map(reverse, m, thr) if reverse is not None else None
    out = []
    for y in ys:
        for x in xs:
            sl = (slice(y, y + winy), slice(x, x + winx))
            mw = m[sl]
            valid = float(mw.mean())
            if valid < min_valid:
                continue
            pw = p[sl]
            best = (0.0, 0.0, 0.0, 0.0)
            for a in angles:
                per, mod, pitch = row_periodicity(pw, mw, px_mm, a)
                if mod > best[1]:  # the direction is chosen by modulation amplitude (period ratio is unstable on flat profiles)
                    best = (per, mod, pitch, a)
            b = B[sl]
            cover = float(b.sum() / max(mw.sum(), 1))
            sfrac, orient, nstroke = stroke_stats(b, px_mm, fibre)
            rec = dict(y=y, x=x, h=winy, w=winx, valid=round(valid, 3), area_cm2=round(float(mw.sum()) * px_mm * px_mm / 100, 3),
                       period=round(best[0], 3), rowmod=round(best[1], 4), pitch_mm=round(best[2], 2), line_angle=best[3],
                       cover=round(cover, 4), stroke=round(sfrac, 3), orient=round(orient, 3), n_stroke=nstroke,
                       mean=round(float((pw * mw).sum() / mw.sum()), 4))
            for n, bc in Bc:
                rec["stab_" + n] = round(dice(b, bc[sl]), 3)
            if Br is not None:
                cr = float(Br[sl].sum() / max(mw.sum(), 1))
                rec["cover_rev"] = round(cr, 4)
                rec["asym"] = round((cover - cr) / (cover + cr + 1e-6), 3)
            out.append(rec)
    return out


# Combined score: z-scores against the control windows (blank papyrus + reversed + noise + off-sheet
# of PHerc0841 w00; see calibration.json written by calibrate_scorer.py). Only components that a
# single map can provide are mandatory; stab/asym enter when present.
DEFAULT_CAL = Path(__file__).with_name("calibration.json")


def combined(rec, cal):
    z = 0.0
    used = []
    for k, (mu, sd) in cal["controls"].items():
        v = rec.get(k)
        if v is None:
            continue
        if k == "period":
            v = math.log(max(v, 1e-6))
        zk = (v - mu) / max(sd, 1e-6)
        z += float(np.clip(zk, -3, 6))  # a single runaway component cannot dominate
        used.append(k)
    return round(z / math.sqrt(max(len(used), 1)), 3), used


def calibration_warnings(cal, win_mm, step_mm):
    messages = []
    window = cal.get("window_mm")
    step = cal.get("step_mm")
    if window is None or step is None:
        messages.append("Calibration window and step are unknown. Treat the combined score as a ranking only.")
    elif not (math.isclose(window, win_mm) and math.isclose(step, step_mm)):
        messages.append(
            f"Calibration used {window:g} mm windows and {step:g} mm steps. "
            f"Requested {win_mm:g} mm windows and {step_mm:g} mm steps. "
            "Historical cutoffs do not transfer to this grid."
        )
    if cal.get("row_sampling") != "physical":
        messages.append("Calibration of the corrected spectrum on internally gapped windows is unverified.")
    return messages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="ink map (uint8 png/tif); forward / chosen direction")
    ap.add_argument("--mask", help="validity mask on the same grid (default: all valid)")
    ap.add_argument("--px-um", type=float, required=True)
    ap.add_argument("--rescale", type=float, nargs=2, help="raw lo hi -> 0..1 (ink_9um raw maps: 0.25 0.75)")
    ap.add_argument("--ct", help="CT surface image for fibre directions (e.g. <render>/composite_mean.png)")
    ap.add_argument("--fibre", type=float, nargs="+", help="fibre directions in degrees (overrides --ct)")
    ap.add_argument("--companion", nargs=2, action="append", default=[], metavar=("NAME", "MAP"),
                    help="another model or depth offset on the same grid (stability)")
    ap.add_argument("--reverse", help="map from the reversed layer order (asymmetry)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--win-mm", type=float, default=20.0)
    ap.add_argument("--step-mm", type=float, default=5.0)
    ap.add_argument("--calibration", default=str(DEFAULT_CAL))
    ap.add_argument("--json", help="write all window records here")
    a = ap.parse_args()
    p = load_map(a.map, a.rescale)
    m = cv2.imread(a.mask, 0) if a.mask else np.ones_like(p, np.uint8)
    fibre = a.fibre
    if fibre is None and a.ct:
        ct = cv2.imread(a.ct, 0)
        f1, f2, _ = fibre_angles(ct, m, a.px_um)
        fibre = [f1, f2]
    comps = [(n, load_map(f, a.rescale)) for n, f in a.companion]
    rev = load_map(a.reverse, a.rescale) if a.reverse else None
    recs = score_windows(p, m, a.px_um, fibre, comps, rev, a.thr, a.win_mm, a.step_mm)
    cal = json.load(open(a.calibration)) if Path(a.calibration).exists() else None
    messages = calibration_warnings(cal, a.win_mm, a.step_mm) if cal else []
    for message in messages:
        print(message, file=sys.stderr)
    if cal:
        for r in recs:
            r["score"], _ = combined(r, cal)
    key = "score" if cal else "period"
    recs.sort(key=lambda r: -r[key])
    res = {"map": a.map, "fibre_deg": fibre, "n_windows": len(recs), "best": recs[0] if recs else None}
    extra = {"calibration_warnings": messages} if messages else {}
    print(json.dumps({**res, **extra}))
    if a.json:
        json.dump({"fibre_deg": fibre, "windows": recs, **extra}, open(a.json, "w"), indent=0)


if __name__ == "__main__":
    main()
