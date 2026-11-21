#!/usr/bin/env python3
"""Combine three comparable depth maps for private review, never a find verdict.

Input: JSON {"maps": [{"offset_planes": ..., "map": ..., "mask": ...,
"provenance": {...}}, ...]}. NPY score maps must be floating-point HW; masks
must be boolean HW. Paths are relative to the manifest or absolute. Output
must be a new directory outside this repository. See README.md for the contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHA = re.compile(r"[0-9a-f]{64}\Z")
SHARED = ("checkpoint_sha256", "mesh_sha256", "frame", "normal_sha256",
          "normal_convention", "plane_order", "xy_pitch_um", "depth_pitch_um",
          "inference", "preprocess")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive_vector(value, length):
    return isinstance(value, list) and len(value) == length and all(finite(v) and v > 0 for v in value)


def check_provenance(provenance):
    if not isinstance(provenance, list) or len(provenance) != 3:
        raise ValueError("Three per-map provenance objects are required")
    for p in provenance:
        if not isinstance(p, dict) or any(key not in p for key in SHARED):
            raise ValueError("Incomplete per-map provenance")
        json.dumps(p, allow_nan=False)
        for key in ("checkpoint_sha256", "mesh_sha256", "normal_sha256"):
            if not isinstance(p[key], str) or not SHA.fullmatch(p[key]):
                raise ValueError("Checkpoint, mesh and normal hashes must be SHA256 strings")
        if not positive_vector(p["xy_pitch_um"], 2) or not finite(p["depth_pitch_um"]) or p["depth_pitch_um"] <= 0:
            raise ValueError("Explicit positive XY and depth sampling pitches are required")
        for key in ("normal_convention", "plane_order"):
            if not isinstance(p[key], str) or not p[key]:
                raise ValueError("Explicit normal convention and ordered-plane description are required")
        for key in ("inference", "preprocess"):
            if not isinstance(p[key], dict) or not p[key]:
                raise ValueError("Nonempty inference and preprocessing records are required")
        frame = p["frame"]
        if not isinstance(frame, dict) or not all(k in frame for k in ("acquisition", "axes", "units", "level", "voxel_um_xyz", "origin_native_xyz")):
            raise ValueError("Complete native-frame metadata is required")
        if frame["axes"] != "XYZ" or frame["units"] != "native_voxels" or type(frame["level"]) is not int or frame["level"] != 0:
            raise ValueError("All maps must declare the same level-zero native XYZ frame")
        if not isinstance(frame["acquisition"], str) or not frame["acquisition"] or not positive_vector(frame["voxel_um_xyz"], 3):
            raise ValueError("Invalid native acquisition or voxel spacing")
        origin = frame["origin_native_xyz"]
        if not isinstance(origin, list) or len(origin) != 3 or not all(finite(v) for v in origin):
            raise ValueError("Explicit finite native XYZ origin is required")
    for p in provenance[1:]:
        for key in SHARED:
            if p[key] != provenance[0][key]:
                raise ValueError("Depth maps differ in shared provenance field: " + key)


def combine(scores, masks, offsets, provenance):
    """Return unscaled maps on common coverage and descriptive comparisons."""
    check_provenance(provenance)
    if len(scores) != 3 or len(masks) != 3 or len(offsets) != 3:
        raise ValueError("Exactly three score maps, masks and offsets are required")
    if not all(finite(x) for x in offsets) or len(set(offsets)) != 3 or offsets.count(0) != 1:
        raise ValueError("Three distinct finite plane offsets including exactly zero are required")
    if not isinstance(scores[0], np.ndarray):
        raise ValueError("Score maps must be NumPy arrays")
    shape = scores[0].shape
    for score, mask in zip(scores, masks):
        if not isinstance(score, np.ndarray) or score.ndim != 2 or not np.issubdtype(score.dtype, np.floating) or 0 in score.shape:
            raise ValueError("Score maps must be nonempty floating-point HW arrays")
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != shape or score.shape != shape:
            raise ValueError("Every mask must be boolean HW on the same grid as every score map")
        if not np.isfinite(score[mask]).all():
            raise ValueError("A score marked valid is nonfinite; repair its mask or source")
    common = np.logical_and.reduce(masks)
    if not common.any():
        raise ValueError("No common valid pixels; no composite can be produced")
    values = np.stack([np.asarray(score[common]) for score in scores])
    middle = offsets.index(0)
    maximum, median, central = values.max(0), np.median(values, axis=0), values[middle]
    # Exact ties prefer the central input, then ascending physical offset. This
    # prevents arbitrary input order from choosing a fake noncentral winner.
    priority = [middle] + sorted((i for i in range(3) if i != middle), key=lambda i: offsets[i])
    choices = np.asarray(priority, dtype=np.int8)[np.argmax(values[priority], axis=0)]
    tie_counts = np.sum(values == maximum[None, :], axis=0).astype(np.uint8)
    output = {"common_valid": common}
    for key, v in (("maximum", maximum), ("median", median), ("central", central)):
        arr = np.full(shape, np.nan, dtype=v.dtype)
        arr[common] = v
        output[key] = arr
    winners = np.full(shape, -1, dtype=np.int8)
    winners[common] = choices
    ties = np.zeros(shape, dtype=np.uint8)
    ties[common] = tie_counts
    output.update(winner_index=winners, maximum_tie_count=ties)
    descriptions = []
    for i, v in enumerate(values):
        delta = v.astype(np.float64) - central.astype(np.float64)
        correlation = None
        if np.std(v.astype(np.float64)) > 0 and np.std(central.astype(np.float64)) > 0:
            correlation = float(np.corrcoef(v.astype(np.float64), central.astype(np.float64))[0, 1])
        descriptions.append(dict(index=i, offset_planes=offsets[i],
                                 offset_um=offsets[i]*provenance[i]["depth_pitch_um"],
                                 valid_pixels_in_own_mask=int(masks[i].sum()),
                                 score_quantiles_0_25_50_75_100=np.quantile(v, [0, .25, .5, .75, 1]).tolist(),
                                 mean_difference_from_central=float(delta.mean()),
                                 mean_absolute_difference_from_central=float(np.abs(delta).mean()),
                                 correlation_with_central=correlation,
                                 winning_pixels_after_tie_rule=int((choices == i).sum())))
    report = dict(shape_yx=list(shape), common_valid_pixels=int(common.sum()),
                  excluded_pixels=int(common.size-common.sum()), central_input_index=middle,
                  index_to_offset_planes=list(offsets),
                  maximum_strictly_above_central_pixels=int((maximum > central).sum()),
                  exact_maximum_tie_pixels=int((tie_counts > 1).sum()),
                  exact_tie_rule="Central input, then ascending offset; ties are also saved explicitly",
                  per_depth=descriptions,
                  transform="No rescaling, calibration, thresholding or image interpolation",
                  provenance_status="Declared shared metadata checked; actual acquisition geometry and sheet identity not certified",
                  interpretation="Private review products only. No candidate verdict or calibrated significance. Recalibrate any downstream score on controls through this same three-depth operation.")
    return output, report


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outside_git(path):
    resolved = Path(path).resolve()
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise ValueError("Generated depth products must stay outside every Git worktree")


def run(manifest_path, output_dir):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir).resolve()
    outside_git(output_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory exists; use a new private output directory")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    rows = manifest.get("maps") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) != 3 or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Manifest must contain exactly three map records")
    scores, masks, receipts, loaded_paths = [], [], [], []
    for row in rows:
        if any(k not in row for k in ("map", "mask", "offset_planes", "provenance")):
            raise ValueError("Incomplete map record")
        paths = [manifest_path.parent / row[k] for k in ("map", "mask")]
        loaded_paths.append(paths)
        scores.append(np.load(paths[0], mmap_mode="r", allow_pickle=False))
        masks.append(np.load(paths[1], mmap_mode="r", allow_pickle=False))
        receipts.append(dict(map_sha256=digest(paths[0]), mask_sha256=digest(paths[1])))
    output, report = combine(scores, masks, [r["offset_planes"] for r in rows], [r["provenance"] for r in rows])
    for paths, receipt in zip(loaded_paths, receipts):
        if digest(paths[0]) != receipt["map_sha256"] or digest(paths[1]) != receipt["mask_sha256"]:
            raise ValueError("An input changed during combination; wait for the producer to finish")
    output_dir.mkdir(parents=True)
    hashes = {}
    for name, array in output.items():
        path = output_dir / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        hashes[path.name] = digest(path)
    report.update(input_manifest_sha256=hashlib.sha256(payload).hexdigest(),
                  script_sha256=digest(__file__), input_receipts=receipts,
                  output_sha256=hashes, shared_provenance=rows[0]["provenance"],
                  status="complete")
    with (output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.manifest, args.output_dir)
    print(json.dumps({k: result[k] for k in ("status", "shape_yx", "common_valid_pixels", "excluded_pixels", "exact_maximum_tie_pixels")}, indent=2))


if __name__ == "__main__":
    main()
