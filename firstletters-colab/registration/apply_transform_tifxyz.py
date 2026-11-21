#!/usr/bin/env python3
"""Move a tifxyz mesh between the 2.4 um rescan and the 9.36 um volume of the same scroll with a
transform.json (bucket layout, schema 1.0.0: fixed_xyz = M @ [moving_xyz, 1], level-0 voxels).

  forward  (default): mesh in the moving (fine) volume, coordinates at fine level --level -> fixed (coarse) level 0
  --inverse         : mesh in the fixed (coarse) volume, level 0                    -> moving (fine) level 0

A vertex is valid only if x, y and z are all finite and all >= 0 and it is not (0, 0, 0); every other
vertex (the tifxyz -1 marker, NaN, inf, any negative channel) is written as -1 in all three channels.
--zrange keeps only vertices whose *input* z is inside [zmin, zmax] (use it to cut a 9 um mesh to the
part covered by the 2.4 um band). The output bbox is computed over valid vertices only.
meta.json 'scale' is divided by the voxel scale change, so the grid spacing in voxels is preserved.
No densification: a 9 um mesh mapped to 2.4 um keeps its vertex spacing (the renderer interpolates).

Frame of the output: meta.json fields that name the volume of the coordinates ('volume', 'volume_id')
are set to the destination volume: transform.json 'fixed_volume' (forward) or 'moving_volume' (inverse;
read from transform.json or from the provenance.json of the same <moving>-to-<fixed> folder), or
--dst-volume-id when given. Fields that name a source-frame dataset ('target_volume', e.g. the surface
prediction a mesh was grown on) are removed. The original values are kept under 'source_frame'.

usage: apply_transform_tifxyz.py transform.json SRC_DIR DST_DIR [--inverse] [--level L] [--zrange ZMIN ZMAX]
                                 [--dst-volume-id ID]
"""
import argparse
import json
import os

import numpy as np
import tifffile


VOLUME_KEYS = ("volume", "volume_id")       # name the volume the coordinates live in -> destination id
SOURCE_ONLY_KEYS = ("target_volume",)       # name a dataset of the source frame -> dropped


def valid_mask(X, Y, Z):
    """True where all three channels are finite and >= 0 and the vertex is not (0, 0, 0)."""
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    with np.errstate(invalid="ignore"):
        ok &= (X >= 0) & (Y >= 0) & (Z >= 0)
        ok &= ~((X == 0) & (Y == 0) & (Z == 0))
    return ok


def destination_volume(transform_path, T, inverse):
    """Volume id of the output frame, or None if it cannot be found."""
    if not inverse:
        return T.get("fixed_volume")
    if T.get("moving_volume"):
        return T["moving_volume"]
    d = os.path.dirname(os.path.abspath(transform_path))
    for cand in (os.path.join(d, "provenance.json"), os.path.join(os.path.dirname(d), "provenance.json")):
        if os.path.exists(cand):
            mv = json.load(open(cand)).get("moving_volume")
            if mv:
                return mv
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transform")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--level", type=int, default=0, help="pyramid level of the fine mesh coordinates (forward only)")
    ap.add_argument("--zrange", type=float, nargs=2, metavar=("ZMIN", "ZMAX"))
    ap.add_argument("--dst-volume-id", help="volume id written into meta.json for the output frame "
                    "(default: inferred from transform.json / provenance.json)")
    a = ap.parse_args(argv)
    T = json.load(open(a.transform))
    M = np.array(T["transformation_matrix"], float)
    A = np.vstack([M, [0, 0, 0, 1]])
    if a.inverse:
        A = np.linalg.inv(A)
    elif a.level:
        A = A @ np.diag([2 ** a.level] * 3 + [1])
    dst_vol = a.dst_volume_id or destination_volume(a.transform, T, a.inverse)
    m = json.load(open(os.path.join(a.src, "meta.json")))
    if dst_vol is None and any(k in m for k in VOLUME_KEYS):
        ap.error("meta.json names a volume but the destination volume cannot be inferred; pass --dst-volume-id")
    X, Y, Z = [tifffile.imread(os.path.join(a.src, f"{c}.tif")).astype(np.float64) for c in "xyz"]
    valid = valid_mask(X, Y, Z)
    if a.zrange:
        valid &= (Z >= a.zrange[0]) & (Z <= a.zrange[1])
    P = np.stack([np.where(valid, X, 0), np.where(valid, Y, 0), np.where(valid, Z, 0), np.ones_like(X)], -1) @ A.T
    os.makedirs(a.dst, exist_ok=True)
    for i, c in enumerate("xyz"):
        tifffile.imwrite(os.path.join(a.dst, f"{c}.tif"), np.where(valid, P[..., i], -1).astype(np.float32))
    f = float(np.cbrt(abs(np.linalg.det(A[:3, :3]))))
    m["scale"] = [m["scale"][0] / f, m["scale"][1] / f]
    m.pop("bbox", None)
    if valid.any():
        m["bbox"] = [[float(P[..., i][valid].min()) for i in range(3)], [float(P[..., i][valid].max()) for i in range(3)]]
    src_frame = {k: m.pop(k) for k in VOLUME_KEYS + SOURCE_ONLY_KEYS if k in m}
    for k in VOLUME_KEYS:
        if k in src_frame:
            m[k] = dst_vol
    if src_frame:
        m["source_frame"] = src_frame
    m["uuid"] = os.path.basename(os.path.normpath(a.dst))
    m["transform_note"] = (f"{'inverse of ' if a.inverse else ''}{os.path.basename(a.transform)} "
                           f"(fixed volume {T.get('fixed_volume')}), level {a.level}; "
                           f"output frame: {dst_vol or 'unknown'}")
    json.dump(m, open(os.path.join(a.dst, "meta.json"), "w"), indent=1)
    print("wrote", a.dst, "valid vertices", int(valid.sum()), "of", valid.size, "scale", m["scale"],
          "frame", dst_vol or "unknown")
    return int(valid.sum())


if __name__ == "__main__":
    main()
