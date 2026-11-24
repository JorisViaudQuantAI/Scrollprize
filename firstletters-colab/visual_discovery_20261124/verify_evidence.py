#!/usr/bin/env python3
"""Verify the published numerical evidence using Python and NumPy only.

No network, GPU, model weights, inference, image enhancement, or letter recognition
is performed. This verifies saved evidence, not the original physical acquisition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def near(actual, expected, name, atol=1e-8):
    ensure(np.isclose(actual, expected, rtol=0, atol=atol),
           f"{name}: {actual!r} != {expected!r}")


def verify_manifest(root):
    manifest = load_json(root / "manifest.json")
    paths = []
    total = 0
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        ensure(path.is_relative_to(root) and path.is_file(), "Invalid artifact path: " + row["path"])
        ensure(path != root / "manifest.json", "Manifest cannot hash itself")
        ensure(path.stat().st_size == row["bytes"], "Size mismatch: " + row["path"])
        ensure(sha(path) == row["sha256"], "Hash mismatch: " + row["path"])
        paths.append(row["path"])
        total += row["bytes"]
    ensure(len(paths) == len(set(paths)), "Duplicate manifest paths")
    ensure(total == manifest["listed_bytes"], "Listed size mismatch")
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*")
                    if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    ensure(set(paths) == actual_files, "Unlisted or missing public files")
    for relative, origin in manifest["origins"].items():
        ensure(relative in paths, "Historical origin references absent public file")
        if origin["transformation"] == "byte-for-byte copy":
            ensure(sha(root / relative) == origin["source_sha256"], "Original copy mismatch: " + relative)
    all_bytes = total + (root / "manifest.json").stat().st_size
    ensure(all_bytes < 10_000_000, "Publication exceeds 10 MB budget")
    return {"verified_files": len(paths), "total_bytes_including_manifest": all_bytes}


def verify_campaign(root):
    campaign = load_json(root / "receipts/campaign_summary.json")
    records = campaign["records"]
    ensure(len(records) == len({r["name"] for r in records}) == 13, "Job count is not 13 distinct maps")
    for record in records:
        receipt_path = root / record["receipt"]
        ensure(sha(receipt_path) == record["published_receipt_sha256"], "Published receipt mismatch")
        receipt = load_json(receipt_path)
        ensure(receipt["sentinel_bitwise_equal"] is True, "Historical sentinel did not pass")
        ensure(receipt["statistics"]["windows"] == record["windows"], "Window count mismatch")
        ensure(receipt["new_model_forwards_including_sentinel"] == record["forwards"], "Forward count mismatch")
        ensure(record["sentinel"] == 1 and record["forwards"] == record["windows"] + 1,
               "Each job must include exactly one sentinel forward")
        origin = load_json(root / "manifest.json")["origins"][record["receipt"]]
        ensure(origin["source_sha256"] == record["historical_receipt_sha256"], "Historical receipt mismatch")
    totals = {"maps": len(records), "windows": sum(r["windows"] for r in records),
              "sentinels": sum(r["sentinel"] for r in records),
              "forwards": sum(r["forwards"] for r in records)}
    ensure(totals == {"maps": 13, "windows": 1078, "sentinels": 13, "forwards": 1091},
           "Campaign total mismatch")
    ensure(campaign["completed_maps"] == totals["maps"] and campaign["prediction_windows"] == totals["windows"]
           and campaign["identity_sentinels"] == totals["sentinels"]
           and campaign["actual_model_forwards"] == totals["forwards"], "Campaign headline mismatch")
    ensure(sum(r["role"] == "known_public_control" for r in records) == 2, "Control map count mismatch")
    return totals


def verify_crop_arrays(root):
    manifest = load_json(root / "data/crop_manifest.json")
    with np.load(root / "data/candidate_probability_crops.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    ensure(set(arrays) == {r["key"] for r in manifest["selections"]}, "Unexpected crop array")
    for record in manifest["selections"]:
        array = arrays[record["key"]]
        ensure(list(array.shape) == record["published_shape"] and str(array.dtype) == record["dtype"],
               "Crop shape or dtype mismatch: " + record["key"])
        ensure(hashlib.sha256(array.tobytes(order="C")).hexdigest() == record["published_array_sha256"],
               "Crop array content mismatch: " + record["key"])
        source_name = Path(record["source_relative_path"]).name
        if source_name.endswith(".prob.npy"):
            receipt_name = source_name.removesuffix(".prob.npy") + ".receipt.json"
            reference = load_json(root / "receipts/inference" / receipt_name)["outputs"]["prob"]
            ensure(np.isfinite(array).all() and ((0 <= array) & (array <= 1)).all(), "Invalid probability")
        else:
            receipt_name = source_name.removesuffix(".context_valid.npy") + ".receipt.json"
            reference = load_json(root / "receipts/inference" / receipt_name)["extra_masks"]["context_valid"]
            ensure(array.dtype == bool, "Context mask must be bool")
        ensure(reference["sha256"] == record["source_sha256"], "Crop source hash differs from inference receipt")
    return arrays


def verify_context(root, arrays):
    old, new = arrays["0800_old_probability"], arrays["0800_new_probability"]
    old_context, new_context = arrays["0800_old_context_valid"], arrays["0800_new_context_valid"]
    ensure(old.shape == new.shape == (240, 304), "Wrong candidate box")
    ensure(new_context.all() and np.all(~old_context | new_context), "Context did not improve as recorded")
    difference = np.abs(new.astype(np.float64) - old.astype(np.float64))
    new_pixels = new_context & ~old_context
    ensure(old_context.sum() == 57600 and new_pixels.sum() == 15360, "Context pixel count mismatch")
    ensure(np.array_equal(old[old_context], new[old_context]), "Already valid candidate values changed")
    metrics = load_json(root / "metrics/0800_reader_context.json")["metrics"]
    for name, mask in [("fixed_roi", np.ones(old.shape, dtype=bool)),
                       ("previously_valid_in_roi", old_context), ("newly_valid_in_roi", new_pixels)]:
        reference = metrics[name]
        ensure(int(mask.sum()) == reference["pixels"], "Context report count mismatch")
        near(float(difference[mask].mean()), reference["mean_abs_probability_change"], name + " mean", 1e-12)
        near(float(difference[mask].max()), reference["max_abs_probability_change"], name + " max", 1e-12)
    return {"pixels": int(old.size), "unchanged_previously_valid": int(old_context.sum()),
            "newly_valid": int(new_pixels.sum()), "new_context_complete": True,
            "newly_valid_mean_abs_probability_change": float(difference[new_pixels].mean()),
            "newly_valid_max_abs_probability_change": float(difference[new_pixels].max()),
            "scope": "Fixed candidate crop only; full-frame 416-column identity remains receipt evidence"}


def seam_metrics(probability):
    # Native publication preview: round(probability * 255), without contrast adjustment.
    preview = np.rint(probability * 255).astype(np.float32)
    dx, dy = np.abs(np.diff(preview, axis=1)), np.abs(np.diff(preview, axis=0))
    grid_x = np.arange(531, 840) % 96 == 0
    grid_y = np.arange(181, 580) % 96 == 0
    grid = np.concatenate([dx[:, grid_x].ravel(), dy[grid_y, :].ravel()])
    other = np.concatenate([dx[:, ~grid_x].ravel(), dy[~grid_y, :].ravel()])
    return {"grid_pair_count": int(grid.size), "other_pair_count": int(other.size),
            "at_original_grid_mean_abs_uint8_step": float(grid.mean()),
            "elsewhere_mean_abs_uint8_step": float(other.mean())}


def verify_seams(root, arrays):
    reference = load_json(root / "metrics/0125_seams.json")["results"]
    result = {}
    for arm in ["raw", "snapped"]:
        result[arm] = {}
        for index, stride in enumerate(["s96", "s48"]):
            prefix = "0125_" + arm + "_" + stride
            ensure(arrays[prefix + "_context_valid"].all(), "PHerc0125 crop is not fully in context")
            probability = arrays[prefix + "_probability"]
            ensure(probability.shape == (400, 310), "Wrong PHerc0125 box")
            metrics = seam_metrics(probability)
            for key, value in metrics.items():
                near(value, reference[arm][index][key], arm + " " + stride + " " + key, 1e-8)
            result[arm][stride] = metrics
    result["interpretation"] = "Recropping, phase, stride and blending changed together; no isolated Hann effect or glyph claim"
    return result


def verify_native(root):
    with (root / "data/sample_values.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    official = np.array([int(r["official_uint8"]) for r in rows])
    floating = np.array([float(r["native_interpolated_float64"]) for r in rows])
    replay = np.array([int(r["replay_truncated_uint8"]) for r in rows])
    recorded_difference = np.array([int(r["signed_difference"]) for r in rows])
    ensure(np.isfinite(floating).all() and ((floating >= 0) & (floating <= 255)).all(), "Invalid native value")
    ensure(np.array_equal(np.trunc(floating).astype(np.int64), replay), "Truncation mismatch")
    difference = replay - official
    ensure(np.array_equal(difference, recorded_difference), "Signed difference mismatch")
    ensure(len(rows) == 7776 and int((difference == 0).sum()) == 7744 and np.abs(difference).max() == 1,
           "Native replay headline mismatch")
    ensure(len({(r["cohort"], r["patch"], r["y"], r["x"], r["stored_layer"]) for r in rows}) == len(rows),
           "Duplicate native sample")
    holdout_rows = [r for r in rows if r["cohort"] == "New spatial check"]
    ensure(len(holdout_rows) == 3888, "Wrong spatial check count")
    holdout_info = load_json(root / "metrics/native_holdout.json")
    results = {}
    with np.load(root / "data/historic_holdout_profiles.npz", allow_pickle=False) as profiles:
        reference = profiles["reference"]
        ensure(np.array_equal(reference.ravel(), np.array([int(r["official_uint8"]) for r in holdout_rows])),
               "CSV and holdout profile reference disagree")
        ensure(np.array_equal(profiles["predictions"][0].ravel(),
                              np.array([float(r["native_interpolated_float64"]) for r in holdout_rows])),
               "CSV and holdout profile prediction disagree")
        for index, description in enumerate(holdout_info["holdout_results"]):
            prediction = profiles["predictions"][index]
            quantized = np.rint(prediction) if description["quantizer"] == "rint" else np.floor(prediction)
            delta = quantized.astype(np.int64) - reference.astype(np.int64)
            mae, exact = float(np.abs(delta).mean()), float((delta == 0).mean())
            near(mae, description["overall"]["mae_rounded"], description["name"] + " MAE", 1e-12)
            near(exact, description["overall"]["fraction_exact_rounded"], description["name"] + " exact", 1e-12)
            results[description["name"]] = {"mae_uint8": mae, "fraction_exact": exact}
    return {"samples": len(rows), "exact": int((difference == 0).sum()),
            "nonexact": int((difference != 0).sum()), "max_abs_uint8_error": int(np.abs(difference).max()),
            "holdout": results, "scope": "Saved interpolated samples; no fresh CT download or geometry replay"}


def binary_auc_and_ap(labels, scores):
    """Exact ranking metrics with tied-score groups; no threshold selection."""
    positive = labels == 2
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    ensure(n_positive > 0 and n_negative > 0, "Both known classes are required")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    rank_sum = ranks[positive[order]].sum()
    auc = (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    descending = order[::-1]
    sorted_scores = scores[descending]
    tp = np.cumsum(positive[descending], dtype=np.int64)
    endpoints = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)-1]
    tp_at_threshold = tp[endpoints]
    precision = tp_at_threshold / (endpoints + 1)
    recall_increment = np.diff(np.r_[0, tp_at_threshold]) / n_positive
    ap = np.sum(recall_increment * precision)
    return float(auc), float(ap)


def verify_control(root):
    manifest = load_json(root / "data/0841_manifest.json")
    with np.load(root / "data/0841_known_predictions.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    ensure(set(arrays) == set(manifest["arrays"]), "Unexpected control array")
    for key, array in arrays.items():
        entry = manifest["arrays"][key]
        ensure(str(array.dtype) == entry["dtype"] and list(array.shape) == entry["shape"], "Control dtype/shape mismatch")
        ensure(hashlib.sha256(array.tobytes(order="C")).hexdigest() == entry["raw_c_order_sha256"], "Control array hash mismatch")
    indices, labels, probability, context = [arrays[k] for k in
        ["known_flat_index", "known_label", "known_probability", "known_context_valid"]]
    ensure(indices.size == 178423 and np.all(np.diff(indices) > 0), "Invalid known pixel indices")
    total = int(np.prod(manifest["shape"]))
    ensure(indices[0] >= 0 and indices[-1] < total and total - indices.size == 608009, "Control extent mismatch")
    ensure(np.isin(labels, [1, 2]).all() and context.all(), "Unknown or invalid-context pixel in known subset")
    ensure(np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all(), "Invalid control score")
    ensure(int((labels == 2).sum()) == 67113 and int((labels == 1).sum()) == 111310, "Control class count mismatch")
    auc, ap = binary_auc_and_ap(labels, probability)
    expected = load_json(root / "metrics/0841_calibration.json")
    near(auc, expected["roc_auc_known_pixels"], "Control AUC", 1e-12)
    near(ap, expected["ap_known_pixels"], "Control AP", 1e-12)
    for label in [1, 2]:
        near(float(probability[labels == label].mean()), expected["class_summaries"][str(label)]["mean_probability"],
             "Control class mean", 1e-8)
    return {"known_ink_pixels": 67113, "known_background_pixels": 111310, "unknown_excluded": 608009,
            "auc": auc, "average_precision": ap,
            "scope": "Known public control only; not independent generalization or discovery"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {"status": "verified_saved_evidence", "new_model_forwards": 0}
    result["manifest"] = verify_manifest(root)
    result["historical_campaign"] = verify_campaign(root)
    arrays = verify_crop_arrays(root)
    result["0800_context"] = verify_context(root, arrays)
    result["0125_seams"] = verify_seams(root, arrays)
    result["native_replay"] = verify_native(root)
    result["0841_known_control"] = verify_control(root)
    candidate = load_json(root / "candidate.json")
    ensure(candidate["status"] == "unconfirmed_morphological_hypothesis" and
           candidate["novel_text_claim"] is False and candidate["ground_truth_eligible"] is False,
           "Candidate status was promoted")
    result["new_validated_letters"] = 0
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
