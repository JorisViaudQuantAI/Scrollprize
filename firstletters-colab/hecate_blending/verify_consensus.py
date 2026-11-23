#!/usr/bin/env python3
"""Independently verify a completed context-consensus secondary, offline.

This reviewer imports neither the secondary runner nor the primary metrics.
It reads only retained public-control arrays. Cutoffs, pixel/adjacency counts,
tie-aware AP/AUC, and all gates are recalculated independently. Raw geometric
assembly is spot-checked at fixed geometry-only pixel addresses, using an
analytic Hann formula rather than the production accumulation. No detector,
checkpoint, torch import, network request, or trained forward is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import time

for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCROLLS = ("PHerc0841", "PHerc0814")
ARMS = ("hann", "consensus")
REPRESENTATIONS = ("float", "uint8")
SECONDARY_FROZEN = ("context_consensus.py", "consensus_protocol.json", "test_consensus.py")
PRIMARY_INPUTS = ("fine_context_uint8.npy", "labels.npy", "common_support.npy",
                  "patch_probabilities.npy", "hann_fine.npy", "hann_uint8.npy",
                  "hann_float_common.npy", "hann_uint8_common.npy")
COUNT_KEYS = ("eligible_pairs", "both_positive_pairs", "exactly_one_positive_pairs", "both_negative_pairs")
# Includes core corners, old central-crop seams, their immediate neighbours,
# and ordinary cell interiors. These addresses do not depend on scores/labels.
SPOT_ADDRESSES_YX = ((0, 0), (0, 1023), (1023, 0), (1023, 1023),
                     (63, 63), (64, 64), (64, 192), (127, 127),
                     (127, 128), (128, 127), (128, 128), (129, 129),
                     (255, 511), (256, 512), (257, 513), (511, 511),
                     (512, 512), (513, 513), (700, 901), (943, 22))


class VerificationFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationFailure(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outside_git(path):
    path = Path(path).expanduser().resolve()
    require(path != ROOT and ROOT not in path.parents, "Evidence paths must remain outside Git")
    return path


def load_array(path, dtype, shape):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.dtype == np.dtype(dtype) and value.shape == shape, path.name + ": wrong dtype/shape")
    if value.dtype.kind == "f":
        require(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(),
                path.name + ": invalid probabilities")
    return value


def freeze_check(secondary, primary, protocol):
    reference = secondary["freeze_reference"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", reference)), "Invalid secondary freeze identity")
    for name in SECONDARY_FROZEN:
        file = HERE / name
        pinned = subprocess.check_output(["git", "show", f"{reference}:{file.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(file.read_bytes() == pinned, "Frozen secondary source changed: " + name)
    for name, expected in protocol["dependency_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen dependency changed: " + name)
    require(secondary["protocol_sha256"] == sha(HERE / "consensus_protocol.json"), "Secondary protocol receipt differs")
    require(secondary["runner_sha256"] == sha(HERE / "context_consensus.py"), "Secondary runner receipt differs")
    require(primary["freeze_reference"] == protocol["primary_freeze_reference"], "Primary freeze differs")
    require(primary["protocol_sha256"] == protocol["dependency_sha256"]["firstletters-colab/hecate_blending/protocol.json"],
            "Primary protocol identity differs")
    require(primary["status"] == secondary["status"] == "complete", "Both experiments must be complete")
    require(primary["forwards_completed"] == 128, "Primary forward count differs")
    for name in ("new_detector_forwards", "new_CT_requests", "new_network_requests", "training_steps"):
        require(secondary[name] == 0, "Secondary unexpectedly changed execution scope")


class Comparison:
    def __init__(self):
        self.scalars = 0
        self.max_absolute_float_difference = 0.0

    def check(self, calculated, recorded, where):
        if isinstance(calculated, dict):
            require(isinstance(recorded, dict) and set(calculated) == set(recorded), where + ": mapping keys differ")
            for key in calculated:
                self.check(calculated[key], recorded[key], where + "/" + key)
        elif isinstance(calculated, (list, tuple)):
            require(isinstance(recorded, (list, tuple)) and len(calculated) == len(recorded), where + ": sequence differs")
            for i, (a, b) in enumerate(zip(calculated, recorded)):
                self.check(a, b, where + "/" + str(i))
        elif isinstance(calculated, (bool, int, str)) or calculated is None:
            require(calculated == recorded, where + ": exact value differs")
            self.scalars += 1
        else:
            require(isinstance(recorded, (int, float)) and not isinstance(recorded, bool), where + ": nonnumeric value")
            difference = abs(float(calculated) - float(recorded))
            require(math.isfinite(difference) and difference <= 1e-12, where + ": numeric value differs")
            self.max_absolute_float_difference = max(self.max_absolute_float_difference, difference)
            self.scalars += 1


def independently_reduce(fine, quantized):
    """Sum the 16 interleaved subgrids, rather than the runner's block reshape."""
    values = np.rint(np.asarray(fine, np.float32) * np.float32(255)).astype(np.uint8) if quantized else fine
    total = np.zeros((256, 256), dtype=np.uint32 if quantized else np.float64)
    for dy in range(4):
        for dx in range(4):
            total += values[dy::4, dx::4]
    return (total / (16 * 255 if quantized else 16)).astype(np.float32)


def masks(labels, support):
    require(labels.shape == support.shape == (256, 256), "Unexpected scoring geometry")
    require(labels.dtype.kind in "iu" and support.dtype == np.dtype(bool), "Unexpected label/support dtype")
    require(np.isin(labels, (0, 1, 2)).all(), "Unexpected label encoding")
    ink, background = support & (labels == 2), support & (labels == 1)
    require(ink.any() and background.any(), "Missing supported known class")
    return ink, background, ink | background


def calibrate(score, labels, support):
    _, background, _ = masks(labels, support)
    values = score[background]
    count = int(values.size)
    allowed = count // 20
    # Ascending order-statistic partition, independent of the descending sort.
    ordinal = count - allowed - 1
    cutoff = float(np.partition(values, ordinal)[ordinal])
    positives = int(np.count_nonzero(values > cutoff))
    return {"background": count, "allowed_background_count": allowed,
            "background_positive_count": positives,
            "background_tied_at_threshold_count": int(np.count_nonzero(values == cutoff)),
            "threshold_strict_greater": cutoff,
            "unused_background_budget_due_to_ties": allowed - positives}


def rank_metrics(score, labels, support):
    """Explicit tied-score groups: stepwise AP and tie-aware Mann–Whitney AUC."""
    ink, background, known = masks(labels, support)
    values, truth = score[known], ink[known]
    _, inverse, totals = np.unique(values, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=truth.astype(np.uint8), minlength=totals.size)
    negatives = totals.astype(np.float64) - positives
    n_positive, n_negative = int(ink.sum()), int(background.sum())
    # Each positive ties with half the negatives in its own ascending group.
    earlier_negatives = np.cumsum(negatives) - negatives
    auc = float(np.sum(positives * (earlier_negatives + negatives / 2)) / (n_positive * n_negative))
    descending_positive = positives[::-1]
    precision = np.cumsum(descending_positive) / np.cumsum(totals[::-1])
    ap = float(np.sum((descending_positive / n_positive) * precision))
    return ap, auc


def adjacency_counts(labels, support, positive):
    """Enumerate row/column boundaries; histogram the two endpoint decisions."""
    classes = np.where(support, labels, 0)
    summaries = {}
    for value, class_name in ((2, "ink"), (1, "background")):
        for seam, prefix in ((True, "seam"), (False, "within_cell")):
            by_axis = {}
            for vertical, axis_name in ((True, "vertical"), (False, "horizontal")):
                histogram = np.zeros(3, dtype=np.int64)
                for destination in range(1, 256):
                    if (destination % 32 == 0) != seam:
                        continue
                    if vertical:
                        first, second = (destination - 1, slice(None)), (destination, slice(None))
                    else:
                        first, second = (slice(None), destination - 1), (slice(None), destination)
                    eligible = (classes[first] == value) & (classes[second] == value)
                    endpoint_count = positive[first][eligible].astype(np.uint8) + positive[second][eligible].astype(np.uint8)
                    histogram += np.bincount(endpoint_count, minlength=3)
                by_axis[axis_name] = {"eligible_pairs": int(histogram.sum()),
                                      "both_positive_pairs": int(histogram[2]),
                                      "exactly_one_positive_pairs": int(histogram[1]),
                                      "both_negative_pairs": int(histogram[0])}
            total = {key: sum(axis[key] for axis in by_axis.values()) for key in COUNT_KEYS}
            denominator = total["eligible_pairs"]
            summaries[prefix + "_" + class_name] = {
                **total,
                "both_positive_fraction": total["both_positive_pairs"] / denominator if denominator else None,
                "exactly_one_positive_fraction": total["exactly_one_positive_pairs"] / denominator if denominator else None,
                "by_axis": by_axis}
    return summaries


def evaluate(score, labels, support, cutoff):
    ink, background, known = masks(labels, support)
    require(math.isfinite(cutoff) and 0 <= cutoff <= 1, "Invalid transferred cutoff")
    positive = known & (score > cutoff)
    n_ink, n_background = int(ink.sum()), int(background.sum())
    pos_ink, pos_background = int(np.count_nonzero(positive & ink)), int(np.count_nonzero(positive & background))
    ap, auc = rank_metrics(score, labels, support)
    return {"known": n_ink + n_background, "ink": n_ink, "background": n_background,
            "threshold_strict_greater": cutoff, "threshold_refit_on_evaluation_donor": False,
            "ink_positive_count": pos_ink, "background_positive_count": pos_background,
            "recall": pos_ink / n_ink, "background_exceedance": pos_background / n_background,
            "ap": ap, "auc": auc, "pairs": adjacency_counts(labels, support, positive)}


def independently_decide(before, after, limits):
    b_bg, a_bg = before["pairs"]["seam_background"], after["pairs"]["seam_background"]
    b_ink, a_ink = before["pairs"]["seam_ink"], after["pairs"]["seam_ink"]
    require(b_bg["eligible_pairs"] == a_bg["eligible_pairs"] and b_ink["eligible_pairs"] == a_ink["eligible_pairs"],
            "Arms have different seam pair populations")
    count = b_bg["both_positive_pairs"]
    answer = {"pass": False, "assessable": False,
              "reason": "insufficient_pair_pool_or_baseline_false_connections",
              "baseline_false_background_pairs": count}
    if count < limits["minimum_baseline_false_background_pairs"]:
        return answer
    if any(pool["eligible_pairs"] < limits["minimum_eligible_pair_pool"] for pool in (b_bg, b_ink)):
        return answer
    removed = count - a_bg["both_positive_pairs"]
    relative, absolute = removed / count, removed / b_bg["eligible_pairs"]
    recall_change = after["recall"] - before["recall"]
    pair_change = a_ink["both_positive_fraction"] - b_ink["both_positive_fraction"]
    ap_change = after["ap"] - before["ap"]
    fpr_change = after["background_exceedance"] - before["background_exceedance"]
    checks = {
        "relative_false_background_pair_reduction": relative >= limits["relative_false_background_pair_reduction_minimum"],
        "absolute_false_background_pair_reduction": absolute >= limits["absolute_false_background_pair_reduction_minimum"],
        "ink_recall_preserved": recall_change >= -limits["ink_recall_maximum_loss"],
        "ink_pair_recall_preserved": pair_change >= -limits["ink_pair_recall_maximum_loss"],
        "AP_preserved": ap_change >= -limits["AP_maximum_loss"],
        "whole_background_pixel_limit": fpr_change <= limits["whole_background_FPR_maximum_increase"],
    }
    passed = all(checks.values())
    answer.update({"pass": passed, "assessable": True,
                   "reason": "all_secondary_conditions_pass" if passed else "secondary_condition_failed",
                   "checks": checks, "relative_false_background_pair_reduction": relative,
                   "absolute_false_background_pair_reduction": absolute, "ink_recall_change": recall_change,
                   "ink_pair_recall_change": pair_change, "AP_change": ap_change,
                   "whole_background_FPR_change": fpr_change})
    return answer


def assembly_spots(patches, raw_support, fine_hann, fine_consensus):
    """Independent scalar contributor enumeration; no full-map reassembly."""
    hann = [0.5 * (1 - math.cos(2 * math.pi * k / 255)) for k in range(256)]
    maximum_squared = max(hann) ** 2
    epsilon = float(np.finfo(np.float32).eps)
    rows = []
    max_geo, max_arithmetic = 0.0, 0.0
    for fy, fx in SPOT_ADDRESSES_YX:
        y, x = fy + 64, fx + 64
        terms = []
        for py in range(8):
            for px in range(8):
                iy, ix = y - 128 * py, x - 128 * px
                if 0 <= iy < 256 and 0 <= ix < 256:
                    # Independent analytic weights are not claimed byte-equal
                    # to Torch's FP32 cosine/window construction.
                    weight = max(hann[iy] * hann[ix] / maximum_squared, 0.001)
                    probability = float(patches[8 * py + px, iy, ix])
                    terms.append((weight, probability))
        require(bool(terms), "Spot lies outside patch support")
        denominator = math.fsum(w for w, _ in terms)
        arithmetic = math.fsum(w * p for w, p in terms) / denominator
        geometric = math.exp(math.fsum(w * math.log(max(p, epsilon)) for w, p in terms) / denominator)
        if not raw_support[y, x]:
            arithmetic = geometric = 0.0
        arithmetic, geometric = float(np.float32(arithmetic)), float(np.float32(geometric))
        ea = abs(arithmetic - float(fine_hann[fy, fx]))
        eg = abs(geometric - float(fine_consensus[fy, fx]))
        require(max(ea, eg) <= 1e-6, "Independent raw-patch assembly spot differs")
        max_geo, max_arithmetic = max(max_geo, eg), max(max_arithmetic, ea)
        rows.append({"core_yx": [fy, fx], "contributors": len(terms), "supported": bool(raw_support[y, x]),
                     "geometric_absolute_error": eg, "arithmetic_absolute_error": ea})
    return {"method": "Independent scalar patch enumeration and analytic Hann weights; no production assembly import",
            "full_raw_geometric_map_reconstruction": False, "probability_absolute_tolerance": 1e-6,
            "maximum_geometric_absolute_error": max_geo, "maximum_arithmetic_absolute_error": max_arithmetic,
            "addresses": rows}


def run(args):
    started = time.monotonic()
    primary_dir, secondary_dir, output = map(outside_git, (args.primary_dir, args.secondary_dir, args.output))
    require(primary_dir != secondary_dir, "Primary and secondary directories must differ")
    require(not output.exists(), "Retain old reviews; choose a fresh output file")
    primary_path, secondary_path = primary_dir / "result.json", secondary_dir / "result.json"
    primary, secondary = read_json(primary_path), read_json(secondary_path)
    protocol = read_json(HERE / "consensus_protocol.json")
    freeze_check(secondary, primary, protocol)
    require(sha(primary_path) == secondary["primary_result_sha256"], "Secondary primary-result pin differs")
    verified = {primary_path: sha(primary_path), secondary_path: sha(secondary_path)}
    compare, retained, field_reviews = Comparison(), {}, []
    primary_fields = {f["scroll"]: f for f in primary["fields"]}
    secondary_fields = {f["scroll"]: f for f in secondary["fields"]}
    require(len(primary["fields"]) == len(secondary["fields"]) == 2 and
            set(primary_fields) == set(secondary_fields) == set(SCROLLS), "Unexpected field population")
    for scroll in SCROLLS:
        pdir, sdir = primary_dir / scroll, secondary_dir / scroll
        for name in PRIMARY_INPUTS:
            path = pdir / name
            expected = primary_fields[scroll]["output_sha256"][name]
            require(sha(path) == expected, "Primary artifact identity differs: " + name)
            verified[path] = expected
        require(set(secondary_fields[scroll]["output_sha256"]) == {
            "consensus_fine.npy", "consensus_uint8.npy", "hann_float_common.npy", "hann_uint8_common.npy",
            "consensus_float_common.npy", "consensus_uint8_common.npy"}, "Unexpected secondary artifacts")
        for name, expected in secondary_fields[scroll]["output_sha256"].items():
            path = sdir / name
            require(sha(path) == expected, "Secondary artifact identity differs: " + name)
            verified[path] = expected
        labels = load_array(pdir / "labels.npy", np.uint8, (256, 256))
        support = load_array(pdir / "common_support.npy", bool, (256, 256))
        raw = load_array(pdir / "fine_context_uint8.npy", np.uint8, (64, 1152, 1152))
        raw_support = np.any(raw != 0, axis=0)
        fine_support = raw_support[64:1088, 64:1088]
        independently_supported = np.ones((256, 256), bool)
        for dy in range(4):
            for dx in range(4):
                independently_supported &= fine_support[dy::4, dx::4]
        require(np.array_equal(support, independently_supported), "Common support is not the all16 source-column intersection")
        patches = load_array(pdir / "patch_probabilities.npy", np.float32, (64, 256, 256))
        fine = {"hann": load_array(pdir / "hann_fine.npy", np.float32, (1024, 1024)),
                "consensus": load_array(sdir / "consensus_fine.npy", np.float32, (1024, 1024))}
        spot_review = assembly_spots(patches, raw_support, fine["hann"], fine["consensus"])
        maps = {}
        for arm in ARMS:
            original_folder = pdir if arm == "hann" else sdir
            exported = load_array(original_folder / (arm + "_uint8.npy"), np.uint8, (1024, 1024))
            require(np.array_equal(exported, np.rint(fine[arm] * np.float32(255)).astype(np.uint8)), "Fine uint8 export differs")
            require(np.count_nonzero(fine[arm][~fine_support]) == 0, "Fine unsupported pixels are not masked")
            for representation in REPRESENTATIONS:
                name = arm + "_" + representation + "_common.npy"
                score = load_array(sdir / name, np.float32, (256, 256))
                expected = independently_reduce(fine[arm], representation == "uint8")
                require(np.array_equal(score, expected), "Independent common reduction differs: " + name)
                if arm == "hann":
                    actual_primary = load_array(pdir / name, np.float32, (256, 256))
                    require(np.array_equal(score, actual_primary), "Operational baseline differs from primary")
                maps[(arm, representation)] = np.asarray(score)
        retained[scroll] = {"labels": np.asarray(labels), "support": np.asarray(support), "maps": maps}
        ink, background, known = masks(labels, support)
        field_reviews.append({"scroll": scroll, "known_pixels": int(known.sum()), "ink_pixels": int(ink.sum()),
                              "background_pixels": int(background.sum()), "all16_supported_common_cells": int(support.sum()),
                              "primary_baseline_common_arrays_unchanged": True,
                              "all_common_reductions_bytewise_equal": True, "fine_uint8_exports_bytewise_equal": True,
                              "raw_assembly_spot_review": spot_review})
        del raw, raw_support, fine_support, fine, patches, exported
    expected_keys = {(a, b, r) for a, b in (SCROLLS, SCROLLS[::-1]) for r in REPRESENTATIONS}
    recorded_transfers = {(t["calibration_scroll"], t["evaluation_scroll"], t["representation"]): t for t in secondary["transfers"]}
    require(len(secondary["transfers"]) == 4 and set(recorded_transfers) == expected_keys, "Wrong transfer population")
    reviews = []
    for a, b in (SCROLLS, SCROLLS[::-1]):
        calibration, target = retained[a], retained[b]
        for representation in REPRESENTATIONS:
            expected = recorded_transfers[(a, b, representation)]
            source_metrics, target_metrics = {}, {}
            for arm in ARMS:
                source_metrics[arm] = calibrate(calibration["maps"][(arm, representation)], calibration["labels"], calibration["support"])
                target_metrics[arm] = evaluate(target["maps"][(arm, representation)], target["labels"], target["support"],
                                               source_metrics[arm]["threshold_strict_greater"])
            gate = independently_decide(target_metrics["hann"], target_metrics["consensus"], protocol["gate"])
            label = a + "->" + b + "/" + representation
            compare.check(source_metrics, expected["calibration"], label + "/calibration")
            compare.check(target_metrics, expected["evaluation"], label + "/evaluation")
            compare.check(gate, expected["gate"], label + "/gate")
            compact = {}
            for arm, entry in target_metrics.items():
                compact[arm] = {key: entry[key] for key in ("known", "ink", "background", "ink_positive_count",
                                "background_positive_count", "recall", "background_exceedance", "ap", "auc")}
                compact[arm]["pairs"] = {key: {name: value[name] for name in COUNT_KEYS}
                                          for key, value in entry["pairs"].items()}
            reviews.append({"calibration_scroll": a, "evaluation_scroll": b, "representation": representation,
                            "source_calibration": source_metrics, "target": compact, "gate": gate})
    all_pass = all(entry["gate"]["pass"] for entry in reviews)
    compare.check(all_pass, secondary["numeric_gate_pass"], "aggregate_gate")
    require(all(sha(path) == expected for path, expected in verified.items()), "An input changed during verification")
    primary_review_path = primary_dir / "independent_review.json"
    primary_review_identity = sha(primary_review_path) if primary_review_path.is_file() else None
    receipt = {"schema": "scrollprize.context-consensus-independent-review.v1", "status": "verified",
               "reviewer_sha256": sha(__file__), "primary_result_sha256": verified[primary_path],
               "secondary_result_sha256": verified[secondary_path], "secondary_freeze_reference": secondary["freeze_reference"],
               "primary_independent_review_sha256": primary_review_identity,
               "source_artifact_count_including_results": len(verified), "source_artifacts_unchanged": True,
               "compared_scalar_count": compare.scalars, "maximum_metric_absolute_difference": compare.max_absolute_float_difference,
               "new_detector_forwards": 0, "new_CT_requests": 0, "new_network_requests": 0, "training_steps": 0,
               "production_measurement_functions_imported": False, "torch_imported": False,
               "all_four_gate_outcomes_confirmed": True, "numeric_gate_pass": all_pass,
               "fields": field_reviews, "transfers": reviews,
               "primary_reviewer_checker_repair": {
                   "scope": "Reviewer assumption only; no primary inference or frozen-file change",
                   "actual_released_predictor_support_dtype": "torch.uint8 in Torch2.6, from torch.any(uint8)",
                   "historical_direct_forward_support_dtype": "torch.bool",
                   "value_semantics": "Identical 0/1 support values; Hecate converts both to float internally",
                   "source": "firstletters-colab/hecate_blending/verify_result.py and its separate primary review",
                   "independently_replayed_again_in_this_review": False},
               "limitations": ["Raw geometric reconstruction was spot-checked, not rebuilt over every fine pixel",
                               "Analytic Hann spot weights are numerically independent, not claimed byte-equal to Torch FP32 construction",
                               "Pixel/pair labels are model-assisted public control labels; no physical-ink or complete-glyph claim",
                               "Adjacent pairs and the two transfer directions are dependent; no independent-sample significance claim"],
               "wall_seconds": time.monotonic() - started,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({key: receipt[key] for key in ("status", "all_four_gate_outcomes_confirmed", "numeric_gate_pass",
                                                   "compared_scalar_count", "maximum_metric_absolute_difference", "wall_seconds")}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", required=True, type=Path)
    parser.add_argument("--secondary-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
