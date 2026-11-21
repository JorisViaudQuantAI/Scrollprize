"""Check the local R50 entry point against the frozen public PHerc. 841 study."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from r50_model import DEPTH, WEIGHT_SHA, WIDTH, file_sha256, load_model, predict_tile
from r50_stack import FIXED_PERMUTATION, array_sha256, depth_order, mosaic, tile_starts, write_bundle
from r50_tile import VOLUME, array


SCRIPT = Path(__file__).with_name("r50_stack.py")
PUBLIC_TILE_SHA = "5fb599f069e38581e346beed228fff371d0e7e1376b6259c107eed1d06fe1952"


def capture_original(source, cache, work):
    # The original control already computes the whole-resolution map before metrics.
    spec = importlib.util.spec_from_file_location("frozen_public_r50", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original_metrics = original.metrics
    saved = False

    def capture(probability, ink, mask):
        nonlocal saved
        if not saved:
            np.save(work / "original-map.npy", probability)
            saved = True
        return original_metrics(probability, ink, mask)

    original.metrics = capture
    sys.argv = [str(source), "--cache", str(cache)]
    original.main()
    if not saved:
        raise RuntimeError("The frozen control did not return its public probability map")


def call_local(raw, basename, cache, work):
    source, dest = work / f"{basename}.npy", work / f"{basename}.npz"
    np.save(source, raw)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = subprocess.run([sys.executable, str(SCRIPT), "predict", str(source), str(dest),
                             "--cache", str(cache), "--threads", "2", "--stride", "128"],
                            check=True, capture_output=True, text=True, env=env)
    reported = json.loads(result.stdout.strip().splitlines()[-1])
    with np.load(dest, allow_pickle=False) as archive:
        probability = archive["probability"]
        meta = json.loads(str(archive["manifest"]))
    assert reported.pop("bundle_sha256") == file_sha256(dest)
    assert reported == meta
    assert meta["weights_sha256"] == WEIGHT_SHA
    assert meta["input_npy_sha256"] == file_sha256(source)
    assert meta["input_c_order_sha256"] == hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    assert meta["probability_c_order_sha256"] == array_sha256(probability)
    return probability, meta


def check_orders_and_atomic_write(work):
    ramp = np.broadcast_to(np.arange(DEPTH, dtype=np.uint8)[:, None, None], (DEPTH, 257, 258))
    local_order = work / "example-permutation.json"
    custom = list(range(0, DEPTH, 2)) + list(range(1, DEPTH, 2))
    local_order.write_text(json.dumps(custom))
    options = [("asc", None, tuple(range(DEPTH))), ("desc", None, tuple(reversed(range(DEPTH)))),
               ("permutation", None, FIXED_PERMUTATION), ("permutation", local_order, tuple(custom))]
    for name, file, expected in options:
        order = depth_order(name, file)
        assert order == expected and sorted(order) == list(range(DEPTH))

        def inspect(tile):
            assert tile.shape == (DEPTH, WIDTH, WIDTH)
            np.testing.assert_array_equal(tile[:, 0, 0], expected)
            return np.ones((WIDTH, WIDTH), dtype=np.float32)

        field, geometry = mosaic(ramp, order, 128, inspect)
        assert np.all(field == 1) and geometry["tiles"] == 4
    for invalid in ([0]*DEPTH, list(range(DEPTH-1)) + [True]):
        local_order.write_text(json.dumps(invalid))
        try:
            depth_order("permutation", local_order)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid permutation accepted")
    assert tile_starts(539, 128) == [0, 128, 256, 283]
    try:
        tile_starts(255, 128)
    except ValueError:
        pass
    else:
        raise AssertionError("A short axis was accepted")
    keep = work / "atomic-test.npz"
    keep.write_bytes(b"prior output")
    prior_files = set(work.iterdir())
    try:
        with patch("r50_stack.np.savez", side_effect=OSError("simulated write failure")):
            write_bundle(keep, np.ones((1, 1), dtype=np.float32), {})
    except OSError:
        pass
    else:
        raise AssertionError("Expected a simulated write failure")
    assert keep.read_bytes() == b"prior output" and set(work.iterdir()) == prior_files


def check_public(cache, work):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # The frozen public control's released w00 coordinates, plus 16 rows and 22 columns.
    surface = array(VOLUME, 0)
    raw = np.asarray(surface[23:85, 11264:11264+272, 12544:12544+278])
    assert raw.shape == (62, 272, 278)
    reference = np.load(work / "original-map.npy", allow_pickle=False)
    single, meta1 = call_local(raw[:, :256, :256], "public-single", cache, work)
    assert meta1["input_c_order_sha256"] == PUBLIC_TILE_SHA
    assert meta1["coverage_min"] == meta1["coverage_max"] == meta1["tiles"] == 1
    np.testing.assert_array_equal(single, reference)
    stitched, meta4 = call_local(raw, "public-overlap", cache, work)
    assert meta4["starts_y"] == [0, 16] and meta4["starts_x"] == [0, 22]
    model = load_model(cache)
    total = np.zeros((272, 278), dtype=np.float64)
    count = np.zeros(total.shape, dtype=np.uint8)
    for y, x in ((0, 0), (0, 22), (16, 0), (16, 22)):
        region = np.s_[y:y+256, x:x+256]
        tile = reference if (y, x) == (0, 0) else predict_tile(model, raw[:, region[0], region[1]])
        total[region] += tile
        count[region] += 1
    expected = (total/count).astype(np.float32)
    np.testing.assert_array_equal(stitched, expected)
    coverage = {str(int(number)): int((count == number).sum()) for number in np.unique(count)}
    assert meta4["coverage_min"] == 1 and meta4["coverage_max"] == 4 and meta4["tiles"] == 4
    check_orders_and_atomic_write(work)
    print("R50_LOCAL_PARITY " + json.dumps({
        "old_public_pixels": int(reference.size), "old_public_map_sha256": array_sha256(reference),
        "new_public_map_sha256": array_sha256(single),
        "single_max_abs_error": float(np.abs(reference-single).max()),
        "overlap_shape": list(stitched.shape), "overlap_pixels": int(stitched.size),
        "overlap_map_sha256": array_sha256(stitched), "coverage_pixels_by_number_of_tiles": coverage,
        "overlap_max_abs_error": float(np.abs(expected-stitched).max()),
        "orders_verified": ["asc", "desc", "permutation", "explicit local permutation"],
        "atomic_write_failure_preserved_prior_output": True,
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--capture-original", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.capture_original:
        capture_original(args.capture_original, args.cache, args.work)
    else:
        check_public(args.cache, args.work)


if __name__ == "__main__":
    main()
