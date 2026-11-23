#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/letterbox_auc.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised extract of a private lane script: local paths, the preview code and the call of a third-party
# line finder were removed. The definition of positives, negatives, z, AUC and 'hot' is unchanged.
# Ledger entry: 03_controls.md C2, C3 (letter-box AUC on the PHerc. 0343 and PHerc. 0306B positives)
# ---------------------------------------------------------------------------------------------------
"""Letter-box AUC of one detector map on a public positive.

positives = valid pixels inside any published letter box (or the claim box)
negatives = valid pixels at least 1.5 mm from every box (letters outside the boxes are possible: kept out)
z         = per-map z-score over valid pixels (48 px border set to 0)
AUC       = rank AUC of z, positives against negatives (400,000 pixels of each at most, seed 0)
hot       = share of pixels with z > 2 (reported inside the boxes and in the negatives)

Boxes are DILUTED positives: most pixels inside a letter box are blank papyrus. An AUC of 0.6-0.75 on boxes
therefore corresponds to clearly visible letters, and must not be compared with the per-pixel AUCs of the
PHerc. 0841 control (0.86-0.92), which use a per-pixel ink map as the answer key.

usage: letterbox_auc.py MAP.png MASK.png BOXES.json UM_PER_PX      (BOXES = [[x0, y0, x1, y1], ...] in map px)
"""
import sys, json
import numpy as np, cv2
from scipy.stats import rankdata


def to_z(prob, valid, border=48):
    p = prob.astype(np.float32); v = valid
    if v.sum() < 100:
        return np.zeros_like(p)
    z = (p - p[v].mean()) / (p[v].std() + 1e-6); z[~valid] = 0
    if border:
        z[:border] = 0; z[-border:] = 0; z[:, :border] = 0; z[:, -border:] = 0
    return z.astype(np.float32)


def auc(pos, neg):
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


if __name__ == '__main__':
    fmap, fmask, fbox, um = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    valid = cv2.imread(fmask, 0) > 0
    H, W = valid.shape
    box = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in json.load(open(fbox)):
        box[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))] = True
    far = cv2.dilate(box.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * int(1500 / um) + 1,) * 2)) == 0
    pos_m, neg_m = box & valid, far & valid
    rng = np.random.default_rng(0)
    sub = lambda m: np.flatnonzero(m.ravel())[rng.permutation(int(m.sum()))[:400000]] if m.any() else np.zeros(0, int)
    ip, ineg = sub(pos_m), sub(neg_m)
    p = cv2.imread(fmap, 0).astype(np.float32)
    if p.shape != (H, W):
        p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
    zr = to_z(p, valid).ravel()
    print(json.dumps(dict(auc_box=round(auc(zr[ip], zr[ineg]), 4), z_box=round(float(zr[ip].mean()), 3),
                          z_far=round(float(zr[ineg].mean()), 3), hot_box=round(float((zr[ip] > 2).mean()), 4),
                          hot_far=round(float((zr[ineg] > 2).mean()), 4), n_pos=int(pos_m.sum()), n_neg=int(neg_m.sum()))))
