#!/usr/bin/env python3
"""One fine->coarse correspondence by normalised cross-correlation of CT z-slices.

A square region of one level-2 z-slice of the fine (moving) volume is downscaled to the coarse
voxel size, band-pass filtered, rotated by -3..+3 degrees and matched (OpenCV TM_CCOEFF_NORMED)
against level-0 z-slices of the coarse (fixed) volume in [zc0, zc1]. Slices are read from the public
open-data bucket with HTTP range requests (one 16 KB row-block per 128^3 chunk), so a point costs
a few hundred small requests and runs on a laptop CPU in a few minutes.

Prints the best matches: ncc, coarse z, angle, fine level-0 xyz -> coarse level-0 xyz.
Copy the best line into landmarks.json and refit with fit_landmarks.py.

example (PHerc1203):
  reg_point.py --fine PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr \
               --coarse PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr \
               --fine-um 2.403 --coarse-um 9.362 --prior 13246.5 13246.5 3380 3418 \
               --z2 900 --cx2 4000 --cy2 3311 --half 300 --zc 8850 8878 2
"""
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
_tl = threading.local()


def sess():
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
    return _tl.s


def zslice(base, volkey, level, z, y0, y1, x0, x1, workers=8):
    """uint8 [y0:y1, x0:x1] of slice z at a level of an uncompressed, 128^3-chunked OME-Zarr."""
    meta = sess().get(f"{base}/{volkey}/{level}/.zarray", timeout=60).json()
    assert meta["compressor"] is None and meta["dtype"] == "|u1", "needs an uncompressed uint8 level"
    Z, Y, X = meta["shape"]
    cz, cy, cx = meta["chunks"]
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(y1, Y), min(x1, X)
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    kz, oz = z // cz, z % cz
    jobs = [(j, i) for j in range(y0 // cy, (y1 - 1) // cy + 1) for i in range(x0 // cx, (x1 - 1) // cx + 1)]

    def fetch(ji):
        j, i = ji
        for attempt in range(4):
            try:
                r = sess().get(f"{base}/{volkey}/{level}/{kz}/{j}/{i}",
                               headers={"Range": f"bytes={oz * cy * cx}-{(oz + 1) * cy * cx - 1}"}, timeout=60)
                if r.status_code == 404:  # chunk absent from the store: fill value 0
                    return ji, None
                r.raise_for_status()  # 403, 500...: never read a refusal as an empty chunk
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {volkey}/{level}/{kz}/{j}/{i}")
                return ji, np.frombuffer(r.content, np.uint8).reshape(cy, cx)
            except Exception:
                if attempt == 3:
                    raise

    with ThreadPoolExecutor(workers) as ex:
        for (j, i), a in ex.map(fetch, jobs):
            if a is None:
                continue
            ys, xs = j * cy, i * cx
            a0, a1 = max(y0, ys), min(y1, ys + cy)
            b0, b1 = max(x0, xs), min(x1, xs + cx)
            out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = a[a0 - ys:a1 - ys, b0 - xs:b1 - xs]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--fine", required=True, help="fine volume key, <Sample>/volumes/<id>.zarr")
    ap.add_argument("--coarse", required=True, help="coarse volume key")
    ap.add_argument("--fine-um", type=float, required=True)
    ap.add_argument("--coarse-um", type=float, required=True)
    ap.add_argument("--prior", type=float, nargs=4, required=True, metavar=("FX", "FY", "CX", "CY"),
                    help="rough correspondence: fine level-0 (x,y) <-> coarse level-0 (x,y), e.g. the two slice centres")
    ap.add_argument("--z2", type=int, required=True, help="fine level-2 slice index")
    ap.add_argument("--cx2", type=int, required=True, help="region centre x, fine level-2 px")
    ap.add_argument("--cy2", type=int, required=True, help="region centre y, fine level-2 px")
    ap.add_argument("--half", type=int, default=300, help="region half-size, fine level-2 px")
    ap.add_argument("--zc", type=int, nargs=3, required=True, metavar=("Z0", "Z1", "STEP"),
                    help="coarse level-0 z range to search")
    ap.add_argument("--angles", type=float, nargs=3, default=(-3, 3, 0.5), metavar=("A0", "A1", "STEP"))
    ap.add_argument("--cache", default="reg_cache", help="folder for cached slice PNGs")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    cv2.setNumThreads(2)
    os.makedirs(a.cache, exist_ok=True)

    def get(key, lvl, z, y0, y1, x0, x1):
        tag = key.replace("/", "_")
        p = os.path.join(a.cache, f"{tag}_L{lvl}_z{z}_{y0}_{y1}_{x0}_{x1}.png")
        if os.path.exists(p):
            return cv2.imread(p, 0)
        img = zslice(a.bucket, key, lvl, z, y0, y1, x0, x1)
        cv2.imwrite(p, img)
        return img

    s = a.fine_um * 4 / a.coarse_um  # fine level-2 px -> coarse level-0 px
    fx, fy, cxp, cyp = a.prior
    F = get(a.fine, 2, a.z2, a.cy2 - a.half, a.cy2 + a.half, a.cx2 - a.half, a.cx2 + a.half).astype(np.float32)
    pcx = cxp + s * (a.cx2 - fx / 4)
    pcy = cyp + s * (a.cy2 - fy / 4)
    hc = int(a.half * s * 1.25)
    X0, Y0 = int(pcx) - hc, int(pcy) - hc
    Fr = cv2.resize(F, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def prep(img):
        b = cv2.GaussianBlur(img, (0, 0), 1.0) - cv2.GaussianBlur(img, (0, 0), 8.0)
        return b * (cv2.erode((img > 3).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)

    Fp = prep(Fr)
    n = 2 * hc
    best = []
    for zc in range(a.zc[0], a.zc[1] + 1, a.zc[2]):
        Cp = prep(get(a.coarse, 0, zc, Y0, Y0 + n, X0, X0 + n).astype(np.float32))
        for ang in np.arange(a.angles[0], a.angles[1] + 1e-6, a.angles[2]):
            R = cv2.warpAffine(Fp, cv2.getRotationMatrix2D((Fp.shape[1] / 2, Fp.shape[0] / 2), ang, 1.0),
                               Fp.shape[::-1])
            res = cv2.matchTemplate(Cp, R, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            best.append((mx, zc, ang, X0 + loc[0] + R.shape[1] / 2, Y0 + loc[1] + R.shape[0] / 2))
    best.sort(reverse=True)
    for b in best[:a.top]:
        print(f"ncc={b[0]:.3f} zc={b[1]} ang={b[2]:+.1f} fine_L0({a.cx2 * 4},{a.cy2 * 4},z{a.z2 * 4}) -> "
              f"coarse({b[3]:.1f},{b[4]:.1f},{b[1]})  [pred {pcx:.0f},{pcy:.0f}]", flush=True)


if __name__ == "__main__":
    main()
