#!/usr/bin/env python
"""Controls on PHerc0841 w00 (known text): model / combination / depth-offset agreement with the 2.4 um answer key,
and calibration of score_text.py on positive vs negative windows. Runs on the GPU VM (CPU part) or locally.

Map files (any of the --maps dirs), all on the render grid:
  <stem>_<forward|reverse>__ckpt-<model>_z<shift>.png   (infer_ink.py --save-each, already rescaled 0..1)
  <model>_<fwd|rev>_z<shift>.png                         (single runs, raw probability; rescaled here)
Model ids are taken from the checkpoint tag: reader -> rv3, s42 -> i42, s43 -> i43, hecate -> hec.
Directory names decide the data set: a dir whose name contains 'noise' holds maps computed on a pure-noise stack.
|shift| >= 16 counts as an off-sheet control.
"""
import argparse, glob, json, math, os, re, sys

import cv2
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_text as st  # noqa: E402

RESCALE = {"rv3": (0.25, 0.75), "i42": (0.25, 0.75), "i43": (0.25, 0.75), "hec": (0.0, 1.0)}


def model_id(s):
    s = s.lower()
    for k, v in (("reader", "rv3"), ("rv3", "rv3"), ("s42", "i42"), ("i42", "i42"), ("s43", "i43"), ("i43", "i43"), ("hecate", "hec"), ("hec", "hec")):
        if k in s:
            return v
    return None


def collect(dirs):
    maps = {}
    for d in dirs:
        ds = "noise" if "noise" in os.path.basename(os.path.normpath(d)) else "w00"
        for f in sorted(glob.glob(os.path.join(d, "*.png"))):
            b = os.path.basename(f)[:-4]
            if "__" in b:
                stem, rest = b.split("__", 1)
                mz = re.search(r"_z([+-]?\d+(?:\.\d+)?)$", rest)
                if not mz:
                    continue
                z = float(mz.group(1)); m = model_id(rest); raw = False
                dr = "fwd" if ("forward" in stem or stem.endswith("_fwd")) else "rev" if ("reverse" in stem or stem.endswith("_rev")) else None
            else:
                mz = re.match(r"^([a-z0-9]+)_(fwd|rev)_z([+-]?\d+)$", b)
                if not mz:
                    continue
                m, dr, z, raw = model_id(mz.group(1)), mz.group(2), float(mz.group(3)), True
            if m is None or dr is None:
                continue
            maps[(ds, m, dr, z)] = (f, raw)
    return maps


def load(entry, m):
    f, raw = entry
    a = cv2.imread(f, 0).astype(np.float32) / 255.0
    if raw:
        lo, hi = RESCALE[m]
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    return a


def auc(score, label):
    r = rankdata(score)
    npos = label.sum(); nneg = len(label) - npos
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def zscore(a, m):
    v = a[m]
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
    return (a - med) / mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--canon", required=True, help="answer key on the 9.366 um surface-volume grid (4220 x 4760)")
    ap.add_argument("--offset", type=int, nargs=2, default=(1320, 800), help="render origin on the canon grid (row col)")
    ap.add_argument("--textroi", type=int, nargs=4, default=(88, 1120, 1752, 2912))
    ap.add_argument("--px-um", type=float, default=9.366)
    ap.add_argument("--win-mm", type=float, default=14.0)
    ap.add_argument("--step-mm", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mask = cv2.imread(os.path.join(a.render, "mask.png"), 0) > 0
    H, W = mask.shape
    canon = cv2.imread(a.canon, 0)
    oy, ox = a.offset
    canon = canon[oy:oy + H, ox:ox + W].astype(np.float32) / 255.0
    gt = canon >= 0.5
    y0, x0, y1, x1 = a.textroi
    roi = np.zeros_like(mask); roi[y0:y1, x0:x1] = True
    sel_roi = (mask & roi)[::3, ::3]; sel_all = mask[::3, ::3]
    ct = cv2.imread(os.path.join(a.render, "composite_mean.png"), 0)
    fib = st.fibre_angles(ct, mask.astype(np.uint8), a.px_um)
    fibre = [fib[0], fib[1]]
    res = {"fibre_deg": fib, "auc": [], "combos": []}
    maps = collect(a.maps)
    print("maps found:", len(maps), sorted({k[:2] for k in maps}))

    def A(arr, sel):
        s = arr[::3, ::3][sel]; l = gt[::3, ::3][sel]
        return round(auc(s, l), 4), round(float(np.corrcoef(s, canon[::3, ::3][sel])[0, 1]), 3)

    # ---- A: single maps
    for k in sorted(maps):
        ds, m, dr, z = k
        if ds != "w00":
            continue
        arr = load(maps[k], m)
        ar, rr = A(arr, sel_roi); aa, ra = A(arr, sel_all)
        res["auc"].append(dict(model=m, dir=dr, z=z, auc_roi=ar, r_roi=rr, auc_all=aa, r_all=ra, mean=round(float(arr[mask].mean()), 4)))
        print("AUC", m, dr, f"{z:+g}", ar, rr, aa)
    # ---- B: combinations (forward and reverse), per shift and depth-averaged
    cache = {}

    def get(m, dr, z):
        k = ("w00", m, dr, z)
        if k not in maps:
            return None
        if k not in cache:
            cache[k] = load(maps[k], m)
        return cache[k]

    models = sorted({k[1] for k in maps if k[0] == "w00"})
    zs = sorted({k[3] for k in maps if k[0] == "w00" and abs(k[3]) < 16})
    sets = [s for s in (("rv3",), ("rv3", "i42"), ("rv3", "hec"), ("rv3", "i42", "hec"), ("rv3", "i42", "i43", "hec"), ("i42", "i43"), ("rv3", "i42", "i43")) if all(x in models for x in s)]
    best = None
    for dr in ("fwd", "rev"):
        for s in sets:
            for zc in zs:
                for depth in ((zc,), (zc - 2, zc, zc + 2)):
                    per = []
                    for m in s:
                        ms = [get(m, dr, z) for z in depth]
                        if any(x is None for x in ms):
                            per = None
                            break
                        per.append(np.mean(ms, 0))
                    if not per:
                        continue
                    for rule in ("mean", "geomean", "zmean"):
                        if len(s) == 1 and rule != "mean":
                            continue
                        if rule == "mean":
                            c = np.mean(per, 0)
                        elif rule == "geomean":
                            c = np.exp(np.mean([np.log(np.clip(p, 1e-3, 1)) for p in per], 0))
                        else:
                            c = np.mean([zscore(p, mask) for p in per], 0)
                        ar, rr = A(c, sel_roi)
                        rec = dict(dir=dr, models="+".join(s), rule=rule, z=zc, depth="avg3" if len(depth) == 3 else "single", auc_roi=ar, r_roi=rr)
                        res["combos"].append(rec)
                        if dr == "fwd" and (best is None or ar > best["auc_roi"]):
                            best = rec
    res["best"] = best
    print("BEST", best)
    cb = sorted([c for c in res["combos"] if c["dir"] == "fwd"], key=lambda c: -c["auc_roi"])[:15]
    for c in cb:
        print("TOPCOMBO", c)

    # ---- C: scorer windows. Scored map = mean of rescaled maps of the chosen set (0..1), depth handled as chosen.
    def scored(ds, s, dr, depth):
        per = []
        for m in s:
            ms = []
            for z in depth:
                k = (ds, m, dr, z)
                if k not in maps:
                    return None
                ms.append(load(maps[k], m))
            per.append(np.mean(ms, 0))
        return np.mean(per, 0)

    cfgs = {"rv3_z0": (("rv3",), (0.0,))}
    if best:
        bs = tuple(best["models"].split("+")); zc = best["z"]
        cfgs["best"] = (bs, (zc - 2, zc, zc + 2) if best["depth"] == "avg3" else (zc,))
    win_recs = []
    for cname, (s, depth) in cfgs.items():
        for dr in ("fwd", "rev"):
            p = scored("w00", s, dr, depth)
            if p is None:
                continue
            comp = []
            for dz in (-2, 2):
                q = scored("w00", s, dr, tuple(z + dz for z in depth))
                if q is not None:
                    comp.append((f"z{dz:+d}", q))
            q = scored("w00", ("i42",) if len(s) == 1 else s[1:], dr, depth)
            if q is not None:
                comp.append(("model", q))
            rev = scored("w00", s, "rev" if dr == "fwd" else "fwd", depth)
            recs = st.score_windows(p, mask.astype(np.uint8), a.px_um, fibre, comp, rev, 0.5, a.win_mm, a.step_mm, 0.6)
            for r in recs:
                sl = (slice(r["y"], r["y"] + r["h"]), slice(r["x"], r["x"] + r["w"]))
                mm = mask[sl]
                r["canon_cover"] = round(float((gt[sl] & mm).sum() / max(mm.sum(), 1)), 4)
                r["cls"] = ("text" if dr == "fwd" else "reversed")
                r["cfg"] = cname
            win_recs += recs
        # off-sheet: |z| >= 16 of rv3 (both directions) and noise stack maps
        for z in sorted({k[3] for k in maps if abs(k[3]) >= 16}):
            for dr in ("fwd", "rev"):
                p = scored("w00", ("rv3",), dr, (z,))
                if p is None:
                    continue
                recs = st.score_windows(p, mask.astype(np.uint8), a.px_um, fibre, [], None, 0.5, a.win_mm, a.step_mm, 0.6)
                for r in recs:
                    r["cls"] = "offsheet"; r["cfg"] = cname; r["z"] = z; r["dir"] = dr
                win_recs += recs
        for dr in ("fwd", "rev"):
            p = scored("noise", s, dr, (0.0,))
            if p is None:
                p = scored("noise", ("rv3",), dr, (0.0,))
            if p is None:
                continue
            m0 = mask.astype(np.uint8) if p.shape == mask.shape else np.ones(p.shape, np.uint8)
            recs = st.score_windows(p, m0, a.px_um, fibre, [], None, 0.5, a.win_mm, a.step_mm, 0.6)
            for r in recs:
                r["cls"] = "noise"; r["cfg"] = cname; r["dir"] = dr
            win_recs += recs
    # classes on forward maps by answer-key coverage
    fw = [r for r in win_recs if r["cls"] == "text"]
    if fw:
        cc = np.array([r["canon_cover"] for r in fw])
        q25, q75 = np.percentile(cc, [25, 75])
        for r in fw:
            r["cls"] = "text_hi" if r["canon_cover"] >= q75 else "text_lo" if r["canon_cover"] <= q25 else "text_mid"
    json.dump(win_recs, open(os.path.join(a.out, "windows.json"), "w"))
    # calibration: controls = reversed + offsheet + noise windows of the 'best' config (fall back to rv3_z0)
    use = "best" if any(r["cfg"] == "best" for r in win_recs) else "rv3_z0"
    keys = ["period", "rowmod", "stroke", "orient", "cover"]
    ctrl = [r for r in win_recs if r["cfg"] == use and r["cls"] in ("reversed", "offsheet", "noise")]
    cal = {"config": use, "controls": {}, "n_controls": len(ctrl), "note": "mean/sd of each component over control windows; period in log"}
    for k in keys:
        v = np.array([math.log(max(r[k], 1e-6)) if k == "period" else r[k] for r in ctrl])
        cal["controls"][k] = [float(v.mean()), float(v.std() + 1e-6)]
    json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    summ = {}
    for r in win_recs:
        if r["cfg"] != use:
            continue
        r["score"], _ = st.combined(r, cal)
        summ.setdefault(r["cls"], []).append(r)
    res["window_classes"] = {}
    for c, rs in sorted(summ.items()):
        d = {"n": len(rs)}
        for k in keys + ["score", "asym", "stab_z-2", "stab_z+2", "stab_model", "pitch_mm"]:
            v = [r[k] for r in rs if k in r]
            if v:
                d[k] = [round(float(np.percentile(v, q)), 3) for q in (10, 50, 90)] + [round(float(np.max(v)), 3)]
        res["window_classes"][c] = d
        print("CLASS", c, json.dumps(d))
    json.dump(res, open(os.path.join(a.out, "controls_summary.json"), "w"), indent=1)

    # ---- D: preview panel (1/4 scale)
    tiles = [("answer key 2.4um", canon)]
    for lab, s, dr, depth in (("rv3 fwd z0", ("rv3",), "fwd", (0.0,)), ("rv3 REVERSE z0", ("rv3",), "rev", (0.0,)),
                              ("i42 fwd z0", ("i42",), "fwd", (0.0,)), ("hecate fwd z0", ("hec",), "fwd", (0.0,))):
        p = scored("w00", s, dr, depth)
        if p is not None:
            tiles.append((lab, p))
    if best:
        bs = tuple(best["models"].split("+")); zc = best["z"]
        dp = (zc - 2, zc, zc + 2) if best["depth"] == "avg3" else (zc,)
        for dr in ("fwd", "rev"):
            p = scored("w00", bs, dr, dp)
            if p is not None:
                tiles.append((f"BEST {best['models']} z{zc:+g} {best['depth']} {dr}", p))
    for z in (-6.0, 6.0):
        p = scored("w00", ("rv3",), "fwd", (z,))
        if p is not None:
            tiles.append((f"rv3 fwd z{z:+g}", p))
    ims = []
    for lab, p in tiles:
        im = cv2.resize((np.clip(p, 0, 1) * 255).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        im[cv2.resize(mask.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) == 0] = 40
        bar = np.zeros((22, im.shape[1]), np.uint8)
        cv2.putText(bar, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
        ims.append(np.vstack([bar, im]))
    cols = 3
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack([np.pad(x, ((0, 3), (0, 3))) for x in ims[i:i + cols]]) for i in range(0, len(ims), cols)]
    cv2.imwrite(os.path.join(a.out, "controls_panel.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("DONE analyze_controls")


if __name__ == "__main__":
    main()
