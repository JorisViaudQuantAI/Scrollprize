#!/usr/bin/env python3
"""Move a tifxyz mesh between the 2.4 um rescan and the 9.36 um volume of the same scroll with a
transform.json (bucket layout, schema 1.0.0: fixed_xyz = M @ [moving_xyz, 1], level-0 voxels).

  forward  (default): mesh in the moving (fine) volume, coordinates at fine level --level -> fixed (coarse) level 0
  --inverse         : mesh in the fixed (coarse) volume, level 0                    -> moving (fine) level 0

Invalid vertices (z <= 0 or -1) stay invalid. --zrange keeps only vertices whose *input* z is
inside [zmin, zmax] (use it to cut a 9 um mesh to the part covered by the 2.4 um band).
meta.json 'scale' is divided by the voxel scale change, so the grid spacing in voxels is preserved.
No densification: a 9 um mesh mapped to 2.4 um keeps its vertex spacing (the renderer interpolates).

usage: apply_transform_tifxyz.py transform.json SRC_DIR DST_DIR [--inverse] [--level L] [--zrange ZMIN ZMAX]
"""
import argparse
import json
import os

import numpy as np
import tifffile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transform")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--level", type=int, default=0, help="pyramid level of the fine mesh coordinates (forward only)")
    ap.add_argument("--zrange", type=float, nargs=2, metavar=("ZMIN", "ZMAX"))
    a = ap.parse_args()
    M = np.array(json.load(open(a.transform))["transformation_matrix"], float)
    A = np.vstack([M, [0, 0, 0, 1]])
    if a.inverse:
        A = np.linalg.inv(A)
    elif a.level:
        A = A @ np.diag([2 ** a.level] * 3 + [1])
    X, Y, Z = [tifffile.imread(os.path.join(a.src, f"{c}.tif")).astype(np.float64) for c in "xyz"]
    valid = (Z > 0) & np.isfinite(Z)
    if a.zrange:
        valid &= (Z >= a.zrange[0]) & (Z <= a.zrange[1])
    P = np.stack([X, Y, Z, np.ones_like(X)], -1) @ A.T
    os.makedirs(a.dst, exist_ok=True)
    for i, c in enumerate("xyz"):
        tifffile.imwrite(os.path.join(a.dst, f"{c}.tif"), np.where(valid, P[..., i], -1).astype(np.float32))
    f = float(np.cbrt(abs(np.linalg.det(A[:3, :3]))))
    m = json.load(open(os.path.join(a.src, "meta.json")))
    m["scale"] = [m["scale"][0] / f, m["scale"][1] / f]
    m.pop("bbox", None)
    if valid.any():
        m["bbox"] = [[float(P[..., i][valid].min()) for i in range(3)], [float(P[..., i][valid].max()) for i in range(3)]]
    m["uuid"] = os.path.basename(os.path.normpath(a.dst))
    m["transform_note"] = (f"{'inverse of ' if a.inverse else ''}{os.path.basename(a.transform)} "
                           f"(fixed volume {json.load(open(a.transform))['fixed_volume']}), level {a.level}")
    json.dump(m, open(os.path.join(a.dst, "meta.json"), "w"), indent=1)
    print("wrote", a.dst, "valid vertices", int(valid.sum()), "scale", m["scale"])


if __name__ == "__main__":
    main()
