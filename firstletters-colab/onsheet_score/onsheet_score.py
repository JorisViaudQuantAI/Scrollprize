#!/usr/bin/env python3
"""Layer-coherence score: is a rendered surface stack parallel to the papyrus sheet, or does it cut through wraps?

Two planes of the render, `dk` layers either side of the mesh layer, are high-passed and phase-correlated in
tiles. On a render parallel to the sheet the two planes sample different fibre layers and do not correlate.
Where the render cuts through sheets at an angle, the same sheet/gap bands appear in both planes, laterally
shifted. Each tile with enough valid pixels gets one class:

    parallel  phase-correlation response <  r_t                (no shared pattern)
    drift     response >= r_t and shift >= s_t pixels          (cut-through)
    still     response >= r_t and shift <  s_t pixels          (shared pattern, no shift)

`f_parallel` is the share of tiles in the first class. It is an upper bound on the on-sheet share: a tile with
no CT structure at all also counts as parallel, and the score says nothing about depth offset or which face is
up. See README.md for the measured numbers and limits.

Needs numpy and opencv-python. Reading TIFF layers needs tifffile; reading a zarr needs zarr.

    python onsheet_score.py --layers RENDER/layers --out score.json [--map tiles.npy]
    python onsheet_score.py --npy stack.npy --mid 20 --out score.json
    python onsheet_score.py --zarr surface-volumes/x.zarr --out score.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

DEFAULTS = dict(dk=2, tile=256, step=128, r_t=0.10, s_t=1.5, hp_sigma=6.0, min_valid=0.9)
PARALLEL, DRIFT, STILL, SKIPPED = 0, 1, 2, 255


def layer_coherence(lower, middle, upper, tile=256, step=128, r_t=0.10, s_t=1.5, hp_sigma=6.0, min_valid=0.9,
                    return_map=False):
    """Score three 2-D planes of one render: `lower` and `upper` are `dk` layers either side of `middle`.

    A pixel is valid when all three planes are non-zero. Tiles with less than `min_valid` valid pixels are
    skipped. Returns a dict; with `return_map` also a uint8 array of tile classes (rows, cols) on the
    `step` grid, 255 where the tile was skipped.
    """
    import cv2

    a = np.asarray(lower, dtype=np.float32)
    b = np.asarray(upper, dtype=np.float32)
    if a.shape != b.shape or a.shape != np.asarray(middle).shape or a.ndim != 2:
        raise ValueError("the three planes must be 2-D and of equal shape")
    valid = (np.asarray(middle) > 0) & (a > 0) & (b > 0)
    a = a - cv2.GaussianBlur(a, (0, 0), hp_sigma)
    b = b - cv2.GaussianBlur(b, (0, 0), hp_sigma)
    window = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    height, width = a.shape
    ys = list(range(0, height - tile + 1, step))
    xs = list(range(0, width - tile + 1, step))
    classes = np.full((len(ys), len(xs)), SKIPPED, np.uint8)
    shifts, responses = [], []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if valid[y:y + tile, x:x + tile].mean() < min_valid:
                continue
            (dx, dy), response = cv2.phaseCorrelate(a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile], window)
            shift = float(np.hypot(dx, dy))
            shifts.append(shift)
            responses.append(float(response))
            if response < r_t:
                classes[i, j] = PARALLEL
            elif shift >= s_t:
                classes[i, j] = DRIFT
            else:
                classes[i, j] = STILL
    n = len(shifts)
    out = dict(tiles=n, tile=tile, step=step, r_t=r_t, s_t=s_t, hp_sigma=hp_sigma, min_valid=min_valid)
    if n:
        scored = classes[classes != SKIPPED]
        confident = np.array(responses) >= r_t
        out.update(
            f_parallel=round(float((scored == PARALLEL).mean()), 4),
            f_drift=round(float((scored == DRIFT).mean()), 4),
            f_still=round(float((scored == STILL).mean()), 4),
            response_median=round(float(np.median(responses)), 3),
            shift_median_confident=round(float(np.median(np.array(shifts)[confident])), 2) if confident.any() else None,
        )
    return (out, classes) if return_map else out


def score_stack(stack, mid=None, dk=2, **kw):
    """Score a [layers, H, W] stack. `mid` is the mesh layer (default: the central layer)."""
    stack = np.asarray(stack) if not hasattr(stack, "shape") else stack
    depth = stack.shape[0]
    mid = (depth - 1) // 2 if mid is None else mid
    if mid - dk < 0 or mid + dk >= depth:
        raise ValueError(f"layers {mid - dk} and {mid + dk} are outside a stack of {depth}")
    res = layer_coherence(np.asarray(stack[mid - dk]), np.asarray(stack[mid]), np.asarray(stack[mid + dk]), **kw)
    meta = dict(layers=int(depth), mid=int(mid), dk=int(dk))
    if isinstance(res, tuple):
        res[0].update(meta)
        return res
    res.update(meta)
    return res


def _load(args):
    """Return the three planes (lower, middle, upper) and the stack depth without loading the whole stack."""
    if args.layers:
        import tifffile

        files = sorted(glob.glob(os.path.join(args.layers, "*.tif")) + glob.glob(os.path.join(args.layers, "*.tiff")))
        if not files:
            sys.exit(f"no TIFF layers in {args.layers}")
        depth = len(files)
        mid = (depth - 1) // 2 if args.mid is None else args.mid
        if mid - args.dk < 0 or mid + args.dk >= depth:
            sys.exit(f"layers {mid - args.dk} and {mid + args.dk} are outside a stack of {depth}")
        planes = [tifffile.imread(files[k]) for k in (mid - args.dk, mid, mid + args.dk)]
        return planes, depth, mid
    if args.npy:
        stack = np.load(args.npy, mmap_mode="r")
    else:
        import zarr

        group = zarr.open(args.zarr, mode="r")
        stack = group["0"] if hasattr(group, "keys") and "0" in group else group
    depth = stack.shape[0]
    mid = (depth - 1) // 2 if args.mid is None else args.mid
    if mid - args.dk < 0 or mid + args.dk >= depth:
        sys.exit(f"layers {mid - args.dk} and {mid + args.dk} are outside a stack of {depth}")
    return [np.asarray(stack[k]) for k in (mid - args.dk, mid, mid + args.dk)], depth, mid


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--layers", help="directory of per-layer TIFF files, sorted by name")
    src.add_argument("--npy", help=".npy file holding a [layers, H, W] stack")
    src.add_argument("--zarr", help="zarr array or OME-Zarr group (level 0) holding a [layers, H, W] stack")
    p.add_argument("--mid", type=int, help="index of the mesh layer (default: the central layer)")
    p.add_argument("--dk", type=int, default=DEFAULTS["dk"], help="layers either side of --mid (default 2)")
    p.add_argument("--tile", type=int, default=DEFAULTS["tile"])
    p.add_argument("--step", type=int, default=DEFAULTS["step"])
    p.add_argument("--r-t", type=float, default=DEFAULTS["r_t"], help="phase-correlation response threshold")
    p.add_argument("--s-t", type=float, default=DEFAULTS["s_t"], help="shift threshold in pixels")
    p.add_argument("--out", required=True, help="JSON file to write (never overwritten)")
    p.add_argument("--map", help="optional .npy for the tile-class map (0 parallel, 1 drift, 2 still, 255 skipped)")
    args = p.parse_args(argv)
    for path in (args.out, args.map):
        if path and os.path.exists(path):
            sys.exit(f"refusing to overwrite {path}")
    (lower, middle, upper), depth, mid = _load(args)
    res, classes = layer_coherence(lower, middle, upper, tile=args.tile, step=args.step, r_t=args.r_t, s_t=args.s_t,
                                   return_map=True)
    res.update(layers=int(depth), mid=int(mid), dk=int(args.dk), shape=[int(v) for v in middle.shape])
    with open(args.out, "x") as f:
        json.dump(res, f, indent=1)
    if args.map:
        np.save(args.map, classes)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
