"""Run the pinned R50 ink detector on a local uint8 NumPy stack, using CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from r50_model import (DEPTH, MODEL, REVISION, WEIGHT_SHA, WIDTH, file_sha256,
                       load_model, predict_tile, verified_cache)


FIXED_PERMUTATION = (
    26, 40, 0, 4, 5, 11, 12, 50, 24, 58, 56, 57, 6, 55, 10, 34,
    22, 9, 48, 33, 19, 30, 46, 29, 2, 17, 52, 35, 32, 39, 21, 31,
    28, 38, 20, 61, 45, 60, 53, 44, 1, 43, 14, 15, 23, 25, 54, 51,
    13, 3, 36, 47, 16, 41, 42, 7, 8, 59, 18, 27, 37, 49,
)


def depth_order(name, filename=None):
    if filename is not None:
        if name != "permutation":
            raise ValueError("--permutation-file requires --order permutation")
        values = json.loads(Path(filename).read_text(encoding="utf-8"))
        if (not isinstance(values, list) or len(values) != DEPTH or
                any(type(i) is not int for i in values) or sorted(values) != list(range(DEPTH))):
            raise ValueError(f"Expected a permutation of the integers 0..{DEPTH-1}")
        return tuple(values)
    if name == "asc":
        return tuple(range(DEPTH))
    if name == "desc":
        return tuple(reversed(range(DEPTH)))
    if name == "permutation":
        return FIXED_PERMUTATION
    raise ValueError(f"Unknown depth order: {name}")


def tile_starts(length, stride):
    if length < WIDTH:
        raise ValueError(f"At least {WIDTH} real pixels are required on each spatial axis")
    if not (1 <= stride <= WIDTH):
        raise ValueError(f"Stride must be between 1 and {WIDTH}")
    last = length - WIDTH
    starts = list(range(0, last+1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def coverage_axis(length, starts):
    counts = np.zeros(length, dtype=np.uint16)
    for start in starts:
        counts[start:start+WIDTH] += 1
    return counts


def array_sha256(a):
    # Hash logical C-order bytes, even when the input .npy uses Fortran order.
    digest = hashlib.sha256()
    for part in np.nditer(a, flags=["external_loop", "buffered"],
                          op_flags=[["readonly"]], order="C", buffersize=1 << 20):
        digest.update(part.tobytes())
    return digest.hexdigest()


def mosaic(stack, order, stride, infer):
    if stack.ndim != 3 or stack.shape[0] != DEPTH or stack.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 array with shape ({DEPTH}, H, W)")
    height, width = stack.shape[1:]
    ys, xs = tile_starts(height, stride), tile_starts(width, stride)
    cy, cx = coverage_axis(height, ys), coverage_axis(width, xs)
    sums = np.zeros((height, width), dtype=np.float64)
    identity = tuple(range(DEPTH))
    for y in ys:
        for x in xs:
            tile = stack[:, y:y+WIDTH, x:x+WIDTH]
            if order != identity:
                tile = tile[list(order)]
            sums[y:y+WIDTH, x:x+WIDTH] += infer(tile)
    # Divide in row bands to avoid another full-resolution coverage array.
    image = np.empty((height, width), dtype=np.float32)
    for first in range(0, height, 256):
        band = slice(first, min(height, first+256))
        counts = cy[band, None].astype(np.uint32) * cx[None, :]
        image[band] = sums[band] / counts
    geometry = {"stride": stride, "tile_size": WIDTH, "starts_y": ys, "starts_x": xs,
                "tiles": len(ys)*len(xs), "coverage_min": int(cy.min())*int(cx.min()),
                "coverage_max": int(cy.max())*int(cx.max()), "blend": "arithmetic_mean"}
    return image, geometry


def write_bundle(path, probability, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            np.savez(f, probability=probability, manifest=json.dumps(manifest, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        bundle_hash = file_sha256(temporary)
        temporary.replace(path)
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return bundle_hash
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_identity(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("weights", help="Download and verify the pinned public model only")
    prepare.add_argument("--cache", required=True, type=Path)
    predict = commands.add_parser("predict", help="Read local input and the verified local model cache")
    predict.add_argument("input", type=Path, help="A NumPy .npy file with uint8 values, shape (62,H,W)")
    predict.add_argument("output", type=Path, help="Atomic .npz with float32 probability and JSON manifest")
    predict.add_argument("--cache", required=True, type=Path)
    predict.add_argument("--order", choices=("asc", "desc", "permutation"), default="asc")
    predict.add_argument("--permutation-file", type=Path, help="Optional local JSON list of 62 input layer indices")
    predict.add_argument("--stride", type=int, default=128)
    predict.add_argument("--threads", type=int, default=min(4, torch.get_num_threads()))
    args = parser.parse_args()
    if args.command == "weights":
        verified_cache(args.cache, download=True)
        print(json.dumps({"model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA}))
        return
    if args.output.suffix.lower() != ".npz" or args.output.resolve() == args.input.resolve():
        parser.error("Choose a distinct .npz output")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    identity = file_identity(args.input)
    order = depth_order(args.order, args.permutation_file)
    stack = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if not isinstance(stack, np.ndarray) or stack.ndim != 3 or stack.shape[0] != DEPTH or stack.dtype != np.uint8:
        parser.error("Expected a uint8 .npy with axes (62, H, W)")
    tile_starts(stack.shape[1], args.stride)
    tile_starts(stack.shape[2], args.stride)
    file_hash = file_sha256(args.input)
    raw_hash = array_sha256(stack)
    model = load_model(args.cache)
    probability, geometry = mosaic(stack, order, args.stride, lambda tile: predict_tile(model, tile))
    if file_identity(args.input) != identity:
        raise RuntimeError("The input file changed during inference")
    manifest = {"model": MODEL, "model_revision": REVISION, "weights_sha256": WEIGHT_SHA,
                "input_npy_sha256": file_hash, "input_c_order_sha256": raw_hash,
                "input_shape": list(stack.shape), "input_dtype": str(stack.dtype),
                "order": args.order, "layer_indices": list(order), "threads": args.threads,
                "torch": str(torch.__version__), "numpy": np.__version__,
                "probability_dtype": str(probability.dtype), "probability_shape": list(probability.shape),
                "probability_c_order_sha256": array_sha256(probability), **geometry}
    bundle_hash = write_bundle(args.output, probability, manifest)
    print(json.dumps({"bundle_sha256": bundle_hash, **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
