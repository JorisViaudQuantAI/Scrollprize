#!/usr/bin/env python3
"""Independently audit the completed, frozen, zero-forward Q16 control analysis.

This reviewer does not import either experimental runner or its metrics module.
It uses explicit nearest-even rounding, strided integer sums, grouped rank
formulae and scikit-learn. It loads no checkpoint and makes no network request.
Only an aggregate receipt is written, outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
from PIL import Image
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "5cdfe2de2deee727808d9258edb20875185ba8ca"
SOURCE_FREEZE = "7bb9cd9d02cef0ed2c95fc694da90a5a2c09481c"
FROZEN = ("protocol.json", "run.py", "test_run.py", "README.md")
SOURCE_FROZEN = ("protocol.json", "run.py", "metrics.py", "test_metrics.py", "test_stitch.py", "README.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def exact(first, second, name):
    first, second = np.asarray(first), np.asarray(second)
    require(first.dtype == second.dtype and first.shape == second.shape,
            "Array type or shape differs: " + name)
    require(first.tobytes(order="C") == second.tobytes(order="C"),
            "Array bytes differ: " + name)


class Audit:
    def __init__(self):
        self.files = {}
        self.scalar_checks = 0
        self.max_float_error = 0.

    def file(self, path, expected):
        path = Path(path)
        require(sha(path) == expected, "File hash differs: " + path.name)
        self.files[path] = expected

    def scalar(self, observed, expected, name):
        self.scalar_checks += 1
        if isinstance(expected, float):
            error = abs(float(observed) - expected)
            self.max_float_error = max(self.max_float_error, error)
            require(np.isfinite(observed) and error <= 1e-12, "Scalar differs: " + name)
        else:
            require(observed == expected, "Scalar differs: " + name)

    def array(self, path, expected, dtype, shape):
        self.file(path, expected)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        require(value.dtype == np.dtype(dtype) and value.shape == shape,
                "Unexpected source array: " + Path(path).name)
        return value

    def recheck(self):
        for path, expected in self.files.items():
            require(sha(path) == expected, "File changed during review: " + path.name)


def freeze_files(folder, names, reference, audit):
    for name in names:
        path = folder / name
        published = subprocess.check_output(
            ["git", "show", f"{reference}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(path.read_bytes() == published, "Published freeze differs: " + name)
        audit.file(path, hashlib.sha256(published).hexdigest())


def round_even(values, dtype):
    """Explicit integer rounding, independent of np.rint used by the producer."""
    values = np.asarray(values, np.float64)
    lower = np.floor(values).astype(np.uint64)
    fraction = values - lower.astype(np.float64)
    increment = (fraction > .5) | ((fraction == .5) & ((lower & 1) == 1))
    return (lower + increment).astype(dtype)


def strided_sum(values, dtype):
    """Add the sixteen disjoint residue classes, without reshape/axis mean."""
    total = np.zeros((256, 256), dtype=dtype)
    for y in range(4):
        for x in range(4):
            total += values[y::4, x::4].astype(dtype)
    return total


def integer_reduction(values):
    sums = strided_sum(values, np.uint64)
    maximum = int(np.iinfo(values.dtype).max)
    common = (sums.astype(np.float64) / (16 * maximum)).astype(np.float32)
    # For these bounded denominators distinct sums must remain distinct FP32
    # scores; equal integer numerators must produce exactly equal scores.
    unique_sum, inverse = np.unique(sums, return_inverse=True)
    unique_score = (unique_sum.astype(np.float64) / (16 * maximum)).astype(np.float32)
    require(len(unique_sum) == len(np.unique(common)), "Reduction lost a distinct integer sum")
    exact(common, unique_score[inverse].reshape(common.shape), "integer tie reconstruction")
    return common, {
        "integer_sum_sha256": hashlib.sha256(sums.tobytes()).hexdigest(),
        "distinct_integer_sums": int(unique_sum.size),
        "repeated_sum_cells_beyond_first": int(sums.size - unique_sum.size),
        "minimum_integer_sum": int(sums.min()),
        "maximum_integer_sum": int(sums.max()),
        "equal_sum_ties_preserved_exactly": True,
    }


def independent_metrics(values, labels, support, audit):
    ink = support & (labels == 2)
    background = support & (labels == 1)
    known = ink | background
    truth, score = ink[known], values[known]
    nink, nbg = int(ink.sum()), int(background.sum())
    require(nink > 0 and nbg > 0 and np.isfinite(score).all(), "Invalid rank metric domain")
    require(((values >= 0) & (values <= 1)).all(), "Common probabilities out of range")
    ap = float(average_precision_score(truth, score))
    auc = float(roc_auc_score(truth, score))
    unique, inverse, totals = np.unique(score, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse[truth], minlength=unique.size).astype(np.int64)
    negatives = totals - positives
    cumulative_positive = positives[::-1].cumsum()
    cumulative_total = totals[::-1].cumsum()
    grouped_ap = float(np.sum((positives[::-1] / nink) *
                              (cumulative_positive / cumulative_total), dtype=np.float64))
    negative_below = negatives.cumsum() - negatives
    twice_pair_credit = int(np.dot(positives, 2 * negative_below + negatives))
    grouped_auc = twice_pair_credit / (2 * nink * nbg)
    audit.scalar(ap, grouped_ap, "AP versus independent grouped definition")
    audit.scalar(auc, grouped_auc, "AUC versus exact integer pair credit")
    budget = nbg // 20
    background_values = values[background]
    ascending_index = nbg - budget - 1
    threshold = float(np.partition(background_values, ascending_index)[ascending_index])
    bg_positive = int((background_values > threshold).sum())
    ink_positive = int((values[ink] > threshold).sum())
    result = {
        "shape": [256, 256], "seam_period": 32,
        "known": nink + nbg, "ink": nink, "background": nbg,
        "unknown_or_unsupported": int(known.size - known.sum()),
        "supported_unknown": int((support & (labels == 0)).sum()),
        "unsupported": int((~support).sum()),
        "ap": ap, "auc": auc,
        "threshold_strict_greater": threshold,
        "threshold_background_rank_zero_based": budget,
        "allowed_background_count": budget,
        "background_positive_count": bg_positive,
        "background_tied_at_threshold_count": int((background_values == threshold).sum()),
        "unused_background_budget_due_to_ties": budget - bg_positive,
        "ink_positive_count": ink_positive,
        "ink_tied_at_threshold_count": int((values[ink] == threshold).sum()),
        "recall_at_background_5pct": ink_positive / nink,
        "background_exceedance": bg_positive / nbg,
    }
    details = {
        "unique_known_scores": int(unique.size),
        "mixed_class_tied_positive_negative_pairs": int(np.dot(positives, negatives)),
        "twice_AUC_pair_credit": twice_pair_credit,
        "AUC_pair_denominator": nink * nbg,
        "grouped_AP": grouped_ap, "integer_pair_AUC": grouped_auc,
    }
    return result, details


def verify(args):
    started = time.monotonic()
    output = args.output or args.run_dir / "independent_review.json"
    require(not output.resolve().is_relative_to(ROOT.resolve()), "Receipt must remain outside Git")
    require(not output.exists(), "Refusing to overwrite an existing review receipt")
    audit = Audit()
    result_path = args.run_dir / "result.json"
    result = read_json(result_path)
    require(result["status"] == "complete" and result["mode"] == "controls" and
            result["freeze_reference"] == FREEZE, "Expected completed frozen precision controls")
    require(all(result[key] == 0 for key in ("detector_forwards", "network_requests", "new_CT_requests", "training_steps")),
            "Analysis scope changed")
    require(result["old_gates_changed"] is False and result["source_artifacts_unchanged"] is True,
            "Gate or source scope changed")
    audit.file(result_path, sha(result_path))
    protocol = read_json(HERE / "protocol.json")
    freeze_files(HERE, FROZEN, FREEZE, audit)
    freeze_files(ROOT / "firstletters-colab/hecate_blending", SOURCE_FROZEN, SOURCE_FREEZE, audit)
    audit.file(HERE / "protocol.json", result["protocol_sha256"])
    audit.file(HERE / "run.py", result["runner_sha256"])
    for name, expected in protocol["dependency_sha256"].items():
        audit.file(ROOT / name, expected)
    audit.file(args.source_dir / "result.json", result["source_result_sha256"])
    source = read_json(args.source_dir / "result.json")
    require(source["status"] == "complete" and source["freeze_reference"] == SOURCE_FREEZE and
            source["forwards_completed"] == 128, "Expected completed original control run")
    settings = protocol["controls"]["restoration"]
    require(settings["absolute_recall_difference_from_float_maximum"] == .01 and
            settings["absolute_AP_difference_from_float_maximum"] == .005 and
            settings["require_both_Hann_fields"] is True and
            settings["diagnostics_cannot_replace_Hann"] is True, "Fidelity decision changed")
    original = {row["scroll"]: row for row in source["fields"]}
    observed = {row["scroll"]: row for row in result["analysis"]["fields"]}
    require(len(original) == len(observed) == 2 and set(original) == set(observed) == {"PHerc0841", "PHerc0814"},
            "Unexpected donor list")
    source_count, output_count = 0, 0
    fields = []
    for scroll in ("PHerc0841", "PHerc0814"):
        folder, hashes = args.source_dir / scroll, original[scroll]["output_sha256"]
        labels = audit.array(folder / "labels.npy", hashes["labels.npy"], np.uint8, (256, 256))
        support = audit.array(folder / "common_support.npy", hashes["common_support.npy"], bool, (256, 256))
        require(np.isin(labels, (0, 1, 2)).all(), "Categorical labels differ")
        source_count += 2
        row = {"scroll": scroll, "assemblies": {}}
        require(set(observed[scroll]["assemblies"]) == {"hann", "central", "uniform"}, "Unexpected assemblies")
        for arm in ("hann", "central", "uniform"):
            measured = observed[scroll]["assemblies"][arm]
            fine = audit.array(folder / f"{arm}_fine.npy", hashes[f"{arm}_fine.npy"], np.float32, (1024, 1024))
            actual_q8 = audit.array(folder / f"{arm}_uint8.npy", hashes[f"{arm}_uint8.npy"], np.uint8, (1024, 1024))
            require(np.isfinite(fine).all() and fine.min() >= -1e-6 and fine.max() <= 1 + 1e-6, "Invalid fine values")
            product16 = np.clip(fine.astype(np.float64), 0., 1.) * 65535.
            q16 = round_even(product16, np.uint16)
            q8 = round_even(np.clip(fine, 0, 1) * np.float32(255), np.uint8)
            exact(q8, actual_q8, "original fine Q8 quantizer")
            q8_common, q8_ties = integer_reduction(q8)
            q16_common, q16_ties = integer_reduction(q16)
            common = {"float": (strided_sum(fine, np.float64) / 16).astype(np.float32),
                      "q8": q8_common, "q16": q16_common}
            for source_name, representation in (("float", "float"), ("uint8", "q8")):
                name = f"{arm}_{source_name}_common.npy"
                saved = audit.array(folder / name, hashes[name], np.float32, (256, 256))
                exact(saved, common[representation], "original common representation")
            source_count += 4
            restored_metrics, ranks = {}, {}
            for representation, values in common.items():
                independent, ranks[representation] = independent_metrics(values, labels, support, audit)
                for key, value in independent.items():
                    audit.scalar(measured["metrics"][representation][key], value, f"{scroll}/{arm}/{representation}/{key}")
                restored_metrics[representation] = independent
            saturation = {}
            for representation, export in (("q8", q8), ("q16", q16)):
                maximum = int(np.iinfo(export.dtype).max)
                saturation[representation] = {
                    "maximum_code": maximum,
                    "fine_pixels_at_maximum": int((export == maximum).sum()),
                    "fine_pixels_at_zero": int((export == 0).sum()),
                    "background_common_cells_at_maximum": int((support & (labels == 1) & (common[representation] == 1)).sum()),
                    "ink_common_cells_at_maximum": int((support & (labels == 2) & (common[representation] == 1)).sum()),
                }
                for key, value in saturation[representation].items():
                    audit.scalar(measured["saturation"][representation][key], value, "saturation/" + key)
            recall_difference = abs(restored_metrics["float"]["recall_at_background_5pct"] - restored_metrics["q16"]["recall_at_background_5pct"])
            ap_difference = abs(restored_metrics["float"]["ap"] - restored_metrics["q16"]["ap"])
            decision = {"pass": recall_difference <= .01 and ap_difference <= .005,
                        "absolute_recall_difference": recall_difference, "absolute_AP_difference": ap_difference}
            for key, value in decision.items():
                audit.scalar(measured["restoration"][key], value, "restoration/" + key)
            eps = float(np.finfo(np.float32).eps)
            clipped = np.clip(fine.astype(np.float64), eps, 1 - eps)
            grey64 = np.clip((np.log(clipped) - np.log1p(-clipped) + 8) / 24, 0, 1)
            views = {"float": fine, "actual_q8": q8, "q16": q16, "fixed_logit_display": grey64.astype(np.float32),
                     "float_common": common["float"], "q8_common": common["q8"], "q16_common": common["q16"]}
            pngs = {"actual_q8": q8, "q16": q16, "fixed_logit_display": round_even(grey64 * 255, np.uint8)}
            expected_names = {f"{arm}_{name}.npy" for name in views} | {f"{arm}_{name}.png" for name in pngs}
            require(set(measured["output_sha256"]) == expected_names, "Unexpected retained output inventory")
            for suffix, expected in views.items():
                name = f"{arm}_{suffix}.npy"
                saved = audit.array(args.run_dir / scroll / name, measured["output_sha256"][name], expected.dtype, expected.shape)
                exact(saved, expected, name)
            for suffix, expected in pngs.items():
                name = f"{arm}_{suffix}.png"
                path = args.run_dir / scroll / name
                audit.file(path, measured["output_sha256"][name])
                with Image.open(path) as image:
                    decoded = np.array(image)
                exact(decoded, expected, name)
            output_count += len(expected_names)
            row["assemblies"][arm] = {
                "metrics": restored_metrics, "rank_definition_crosscheck": ranks,
                "integer_reduction": {"q8": q8_ties, "q16": q16_ties},
                "q16_fine_exact_halfway_values": int(((product16 - np.floor(product16)) == .5).sum()),
                "saturation": saturation, "restoration": decision,
                "all_quantizations_common_grids_output_copies_PNGs_and_fixed_displays_exact": True,
            }
        row["Hann_restoration_pass"] = row["assemblies"]["hann"]["restoration"]["pass"]
        audit.scalar(observed[scroll]["Hann_restoration_pass"], row["Hann_restoration_pass"], "field Hann fidelity")
        fields.append(row)
    decision = all(row["Hann_restoration_pass"] for row in fields)
    audit.scalar(result["analysis"]["Hann_restoration_pass"], decision, "overall Hann fidelity")
    audit.scalar(result["consumed_source_artifact_count"], source_count, "source artifact count")
    require(source_count == 28 and output_count == 60, "Review did not cover the complete output inventory")
    audit.recheck()
    receipt = {
        "schema": "scrollprize.prediction-precision-independent-review.v1", "status": "pass",
        "freeze_reference": FREEZE, "original_freeze_reference": SOURCE_FREEZE,
        "reviewer_sha256": sha(Path(__file__)), "reviewed_result_sha256": sha(result_path),
        "source_result_sha256": result["source_result_sha256"],
        "frozen_analysis_files_checked": 4, "frozen_original_files_checked": 6,
        "dependency_pins_checked": len(protocol["dependency_sha256"]),
        "source_arrays_checked": source_count, "output_assets_checked": output_count,
        "fine_quantizations_checked": 6, "common_rank_operating_points_checked": 18,
        "scalar_checks": audit.scalar_checks, "maximum_scalar_absolute_difference": audit.max_float_error,
        "source_output_and_pins_unchanged_after_review": True,
        "Hann_serialization_fidelity_pass": decision, "fields": fields,
        "runtime": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "wall_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "reviewer_network_requests": 0, "reviewer_detector_forwards": 0, "checkpoint_loads": 0,
        "scope": "Audit of six retained serialization arms. Independent nearest-even quantizers, exact integer sums/ties, rank definitions, strict cutoffs, fidelity decision, output/display pixels and hashes. Pair-connectivity metrics and original model outputs are covered by the separate primary review, not recomputed here. No image interpretation, calibration transport, unread promotion or prize claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "scalar_checks", "maximum_scalar_absolute_difference", "Hann_serialization_fidelity_pass", "wall_seconds")}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/scrollprize-prediction-precision-controls-v1"))
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/scrollprize-hecate-blending-v1"))
    parser.add_argument("--output", type=Path)
    return verify(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
