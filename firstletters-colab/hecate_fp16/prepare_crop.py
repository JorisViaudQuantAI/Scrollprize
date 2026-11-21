#!/usr/bin/env python3
"""Cut a crop of a public 9.366 um surface volume and resample it to Hecate's 9.6 um grid (.npy).

Default: PHerc. 0841 w00 (known text), the region used for the T4 measurement in the README.
Reads raw chunks over HTTPS (the level-0 array is uncompressed uint8), no zarr needed.
A chunk answered with HTTP 404 is absent from the store and read as zeros (Zarr's fill value); any other
status than 200/404 (e.g. 403 from a proxy or a permission change) stops the script, so a refused download
can never turn into silently black input. The sha256 of the saved crop array, and of the source crop before
resampling, are printed and written next to it (OUT.npy.sha256) so a timing run can be tied to its exact input.

usage: prepare_crop.py OUT.npy [--key KEY] [--y0 Y --x0 X --size N] [--src-um 9.366]
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import requests
from scipy.ndimage import zoom

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
KEY = "PHerc0841/segments/20260220213127-w00/surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--y0", type=int, default=1408)
    ap.add_argument("--x0", type=int, default=2048)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--src-um", type=float, default=9.366)
    ap.add_argument("--dst-um", type=float, default=9.6)
    a = ap.parse_args()
    s = requests.Session()
    r = s.get(f"{B}/{a.key}/0/.zarray", timeout=60)
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code} for {a.key}/0/.zarray: refusing to continue")
    meta = r.json()
    assert meta["compressor"] is None and meta["dtype"] == "|u1"
    Z, Y, X = meta["shape"]
    cz, cy, cx = meta["chunks"]
    sep = meta.get("dimension_separator", ".")
    y1, x1 = min(a.y0 + a.size, Y), min(a.x0 + a.size, X)
    vol = np.zeros((Z, y1 - a.y0, x1 - a.x0), np.uint8)
    n_ok = n_missing = 0
    for j in range(a.y0 // cy, (y1 - 1) // cy + 1):
        for i in range(a.x0 // cx, (x1 - 1) // cx + 1):
            for k in range(0, (Z - 1) // cz + 1):
                ck = f"{a.key}/0/{sep.join(map(str, (k, j, i)))}"
                r = s.get(f"{B}/{ck}", timeout=120)
                if r.status_code == 404:  # chunk absent from the store: fill value 0
                    n_missing += 1
                    continue
                r.raise_for_status()  # 403, 500...: a refusal is never an empty chunk
                if r.status_code != 200:
                    sys.exit(f"HTTP {r.status_code} for {ck}: refusing to treat it as an empty chunk")
                if len(r.content) != cz * cy * cx:
                    sys.exit(f"{ck}: {len(r.content)} bytes, expected {cz * cy * cx}")
                n_ok += 1
                c = np.frombuffer(r.content, np.uint8).reshape(cz, cy, cx)
                ys, xs, zs = j * cy, i * cx, k * cz
                ya, yb = max(a.y0, ys), min(y1, ys + cy)
                xa, xb = max(a.x0, xs), min(x1, xs + cx)
                zb = min(Z, zs + cz)
                vol[zs:zb, ya - a.y0:yb - a.y0, xa - a.x0:xb - a.x0] = c[:zb - zs, ya - ys:yb - ys, xa - xs:xb - xs]
    f = a.src_um / a.dst_um
    out = np.clip(np.rint(zoom(vol.astype(np.float32), f, order=1)), 0, 255).astype(np.uint8)
    np.save(a.out, out)
    digest = hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()
    src_digest = hashlib.sha256(np.ascontiguousarray(vol).tobytes()).hexdigest()  # before resampling
    with open(a.out + ".sha256", "w") as fh:
        fh.write(f"{digest}  array bytes of {a.out} (uint8, shape {list(out.shape)})\n"
                 f"{src_digest}  source crop before resampling (uint8, shape {list(vol.shape)})\n")
    print(json.dumps({"key": a.key, "crop_yx": [a.y0, a.x0, a.size], "src_shape": vol.shape, "out_shape": out.shape,
                      "zoom": round(f, 5), "nonzero_fraction": round(float((out > 0).mean()), 3),
                      "chunks_read": n_ok, "chunks_absent_404": n_missing, "sha256_array": digest,
                      "sha256_source": src_digest}))


if __name__ == "__main__":
    main()
