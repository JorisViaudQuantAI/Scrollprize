#!/usr/bin/env python3
"""Frozen, zero-forward context-agreement secondary on two public controls.

This script consumes a completed, separately frozen Hann experiment. It never
loads a detector, requests data, fits model weights, chooses a field, or changes
the primary experiment. Every empirical cutoff is calibrated on the other
physical donor before being transferred unchanged to its evaluation donor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ("context_consensus.py", "consensus_protocol.json", "test_consensus.py")
SCROLLS = ("PHerc0841", "PHerc0814")
EPS = float(np.finfo(np.float32).eps)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_freeze(reference, protocol):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("A complete public freeze commit is required")
    for name in FROZEN:
        path = HERE / name
        reference_bytes = subprocess.check_output(
            ["git", "show", f"{reference}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        if path.read_bytes() != reference_bytes:
            raise ValueError("Secondary file differs from its freeze")
    for name, digest in protocol["dependency_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen secondary dependency changed")


def assemble(patches, support_context, window):
    """Return weighted arithmetic diagnostic and geometric candidate cores.

    The input probabilities are unmasked. Both outputs receive the same exact
    raw.any(Z) column mask. FP64 is used for both diagnostic accumulations;
    the operational baseline remains the saved primary FP32 Hann assembly.
    The log floor is the released model's FP32 epsilon, not a fitted parameter.
    """
    patches, support_context, window = map(np.asarray, (patches, support_context, window))
    if patches.shape != (64, 256, 256) or patches.dtype != np.float32:
        raise ValueError("Expected all64 FP32 full-patch probabilities")
    if support_context.shape != (1152, 1152) or support_context.dtype != bool:
        raise ValueError("Expected the exact boolean1152x1152 source-column mask")
    if window.shape != (256, 256) or window.dtype != np.float32:
        raise ValueError("Expected the released FP32 Hann window")
    if not np.isfinite(window).all() or not (window > 0).all():
        raise ValueError("Every window weight must be finite and positive")
    numerator = np.zeros((1152, 1152), np.float64)
    log_numerator = np.zeros_like(numerator)
    denominator = np.zeros_like(numerator)
    weights = window.astype(np.float64)
    floored_supported_predictions = 0
    for index in range(64):
        probability = patches[index].astype(np.float64)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("All retained patch probabilities must be finite and in[0,1]")
        y, x = index // 8 * 128, index % 8 * 128
        region = np.s_[y:y+256, x:x+256]
        mask = support_context[region]
        numerator[region] += probability * mask * weights
        log_numerator[region] += np.log(np.maximum(probability, EPS)) * weights
        denominator[region] += weights
        floored_supported_predictions += int(np.count_nonzero((probability < EPS) & mask))
    if not (denominator > 0).all():
        raise ValueError("Incomplete accumulation")
    arithmetic = numerator / denominator
    geometric = np.exp(log_numerator / denominator)
    geometric[~support_context] = 0.
    core = np.s_[64:1088, 64:1088]
    support = support_context[core]
    if not support.any():
        raise ValueError("No supported core columns")
    excess = float(np.max(geometric[core][support] - arithmetic[core][support]))
    # Epsilon flooring can raise a truly zero input by at most epsilon.
    if excess > 2 * EPS:
        raise ValueError("Geometric/arithmetic inequality failed beyond epsilon floor")
    diagnostic = {
        "log_floor": EPS,
        "floored_supported_prediction_count": floored_supported_predictions,
        "maximum_geometric_minus_arithmetic_supported": excess,
        "core_mean_arithmetic_minus_geometric_supported": float(np.mean(arithmetic[core][support] - geometric[core][support])),
    }
    return arithmetic[core].astype(np.float32), geometric[core].astype(np.float32), diagnostic


def measure_transferred(score, labels, support, threshold, metric_module):
    """Evaluate a supplied donor-external cutoff without fitting on this field."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Transferred probability cutoff must be finite and in[0,1]")
    score, labels, support, known, ink, background = metric_module._validate_field(score, labels, support)
    positive = np.zeros(score.shape, bool)
    positive[known] = score[known] > threshold
    pos_ink = int(np.count_nonzero(positive & ink))
    pos_background = int(np.count_nonzero(positive & background))
    return {
        "known": int(np.count_nonzero(known)),
        "ink": int(np.count_nonzero(ink)),
        "background": int(np.count_nonzero(background)),
        "threshold_strict_greater": float(threshold),
        "threshold_refit_on_evaluation_donor": False,
        "ink_positive_count": pos_ink,
        "background_positive_count": pos_background,
        "recall": float(pos_ink / np.count_nonzero(ink)),
        "background_exceedance": float(pos_background / np.count_nonzero(background)),
        "ap": float(metric_module.average_precision_score(ink[known], score[known])),
        "auc": float(metric_module.roc_auc_score(ink[known], score[known])),
        "pairs": {
            "seam_ink": metric_module._pair_summary(positive, ink, True, 32),
            "seam_background": metric_module._pair_summary(positive, background, True, 32),
            "within_cell_ink": metric_module._pair_summary(positive, ink, False, 32),
            "within_cell_background": metric_module._pair_summary(positive, background, False, 32),
        },
    }


def compare_transfer(baseline, consensus, gate):
    """A conservative veto-method gate, not a probability or discovery claim."""
    for key in ("known", "ink", "background"):
        if baseline[key] != consensus[key]:
            raise ValueError("Transferred arms must share evaluation labels and support")
    for key in ("seam_ink", "seam_background"):
        if baseline["pairs"][key]["eligible_pairs"] != consensus["pairs"][key]["eligible_pairs"]:
            raise ValueError("Transferred arms have different pair populations")
    base_bg = baseline["pairs"]["seam_background"]
    new_bg = consensus["pairs"]["seam_background"]
    base_ink = baseline["pairs"]["seam_ink"]
    new_ink = consensus["pairs"]["seam_ink"]
    outcome = {"pass": False, "assessable": False,
               "reason": "insufficient_pair_pool_or_baseline_false_connections",
               "baseline_false_background_pairs": base_bg["both_positive_pairs"]}
    if (min(base_bg["eligible_pairs"], base_ink["eligible_pairs"]) < gate["minimum_eligible_pair_pool"]
            or base_bg["both_positive_pairs"] < gate["minimum_baseline_false_background_pairs"]):
        return outcome
    before, after = base_bg["both_positive_fraction"], new_bg["both_positive_fraction"]
    if before is None or after is None or before <= 0 or base_ink["both_positive_fraction"] is None or new_ink["both_positive_fraction"] is None:
        raise ValueError("Inconsistent eligible pair counts and rates")
    # Paired pools have the same denominator. Use their integer counts before
    # division, avoiding cancellation and artificial boundary changes from
    # subtracting rounded rates.
    false_pairs_removed = base_bg["both_positive_pairs"] - new_bg["both_positive_pairs"]
    absolute_reduction = false_pairs_removed / base_bg["eligible_pairs"]
    relative_reduction = false_pairs_removed / base_bg["both_positive_pairs"]
    checks = {
        "relative_false_background_pair_reduction": relative_reduction >= gate["relative_false_background_pair_reduction_minimum"],
        "absolute_false_background_pair_reduction": absolute_reduction >= gate["absolute_false_background_pair_reduction_minimum"],
        "ink_recall_preserved": consensus["recall"] >= baseline["recall"] - gate["ink_recall_maximum_loss"],
        "ink_pair_recall_preserved": new_ink["both_positive_fraction"] >= base_ink["both_positive_fraction"] - gate["ink_pair_recall_maximum_loss"],
        "AP_preserved": consensus["ap"] >= baseline["ap"] - gate["AP_maximum_loss"],
        "whole_background_pixel_limit": consensus["background_exceedance"] <= baseline["background_exceedance"] + gate["whole_background_FPR_maximum_increase"],
    }
    outcome.update({"pass": bool(all(checks.values())), "assessable": True,
                    "reason": "all_secondary_conditions_pass" if all(checks.values()) else "secondary_condition_failed",
                    "checks": checks, "relative_false_background_pair_reduction": relative_reduction,
                    "absolute_false_background_pair_reduction": absolute_reduction,
                    "ink_recall_change": consensus["recall"] - baseline["recall"],
                    "ink_pair_recall_change": new_ink["both_positive_fraction"] - base_ink["both_positive_fraction"],
                    "AP_change": consensus["ap"] - baseline["ap"],
                    "whole_background_FPR_change": consensus["background_exceedance"] - baseline["background_exceedance"]})
    return outcome


def run(args):
    protocol = json.loads((HERE / "consensus_protocol.json").read_text())
    # No input result or probability is opened before this exact-file check.
    verify_freeze(args.freeze_reference, protocol)
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    if source == out or out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a fresh output directory outsideGit and outside the source directory")
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Keep source and output directory trees separate")
    out.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema": "scrollprize.context-consensus.v1", "status": "started",
              "freeze_reference": args.freeze_reference,
              "protocol_sha256": sha(HERE / "consensus_protocol.json"),
              "runner_sha256": sha(__file__), "fields": [], "transfers": [],
              "new_detector_forwards": 0, "new_CT_requests": 0, "new_network_requests": 0,
              "training_steps": 0, "old_or_primary_gate_changed": False,
              "candidate_autogate": False, "numeric_gate_pass": False}
    prior_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("secondary_time_cap")))
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["wall_seconds"])
    code = 1
    stage = "source_validation"
    try:
        primary = json.loads((source / "result.json").read_text())
        if primary["status"] != "complete" or primary["forwards_completed"] != 128:
            raise ValueError("Both complete frozen primary fields are required, irrespective of their gate")
        if primary["freeze_reference"] != protocol["primary_freeze_reference"]:
            raise ValueError("Unexpected primary experiment freeze")
        if primary["protocol_sha256"] != protocol["dependency_sha256"]["firstletters-colab/hecate_blending/protocol.json"]:
            raise ValueError("Unexpected primary protocol")
        result["primary_result_sha256"] = sha(source / "result.json")
        by_scroll = {row["scroll"]: row for row in primary["fields"]}
        if set(by_scroll) != set(SCROLLS) or len(primary["fields"]) != 2:
            raise ValueError("Unexpected physical donors")
        metric_module = module(HERE / "metrics.py", "context_consensus_metrics")
        # Only a deterministic window constructor is used from torch; no model
        # is imported or loaded. Its exact byte hash must match the primary.
        import torch
        window = torch.hann_window(256, periodic=False, dtype=torch.float32)
        window = window[:, None] * window[None, :]
        window = (window / window.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
        if hashlib.sha256(window.tobytes()).hexdigest() != primary["window_array_sha256"]:
            raise ValueError("Released Hann window bytes changed")
        complete, verified = {}, []
        for scroll in SCROLLS:
            stage = "secondary_assembly"
            folder = source / scroll
            row = by_scroll[scroll]
            required = ("fine_context_uint8.npy", "labels.npy", "common_support.npy", "patch_probabilities.npy",
                        "hann_fine.npy", "hann_uint8.npy", "hann_float_common.npy", "hann_uint8_common.npy")
            for name in required:
                path = folder / name
                if sha(path) != row["output_sha256"][name]:
                    raise ValueError("A retained primary input/output hash changed")
                verified.append((path, row["output_sha256"][name]))
            labels = np.load(folder / "labels.npy", allow_pickle=False)
            support = np.load(folder / "common_support.npy", allow_pickle=False)
            ct = np.load(folder / "fine_context_uint8.npy", mmap_mode="r", allow_pickle=False)
            if ct.shape != (64, 1152, 1152) or ct.dtype != np.uint8:
                raise ValueError("Unexpected complete source context")
            column_support = np.any(ct, axis=0)
            expected_support = column_support[64:1088, 64:1088].reshape(256, 4, 256, 4).all((1, 3))
            if support.dtype != bool or not np.array_equal(support, expected_support):
                raise ValueError("Common support does not match all16 source columns")
            patches = np.load(folder / "patch_probabilities.npy", mmap_mode="r", allow_pickle=False)
            arithmetic, geometric, diagnostics = assemble(patches, column_support, window)
            baseline_fine = np.load(folder / "hann_fine.npy", allow_pickle=False)
            baseline_export = np.load(folder / "hann_uint8.npy", allow_pickle=False)
            if baseline_fine.dtype != np.float32 or baseline_fine.shape != (1024, 1024):
                raise ValueError("Unexpected actual primary Hann core")
            exact_export = np.rint(baseline_fine.clip(0, 1) * np.float32(255)).astype(np.uint8)
            if baseline_export.dtype != np.uint8 or not np.array_equal(baseline_export, exact_export):
                raise ValueError("Saved primary export is not the actual fine uint8 export")
            diagnostics["maximum_FP64_arithmetic_vs_saved_FP32_Hann_difference"] = float(np.max(np.abs(arithmetic.astype(np.float64) - baseline_fine)))
            if diagnostics["maximum_FP64_arithmetic_vs_saved_FP32_Hann_difference"] > protocol["arithmetic_reconstruction_maximum_difference"]:
                raise ValueError("Arithmetic diagnostic does not reproduce saved Hann within numeric allowance")
            maps = {"float": {}, "uint8": {}}
            for representation in maps:
                quantized = representation == "uint8"
                baseline = np.load(folder / f"hann_{representation}_common.npy", allow_pickle=False)
                expected = metric_module.common_grid(baseline_fine, uint8_export=quantized)
                if not np.array_equal(baseline, expected):
                    raise ValueError("Saved primary common grid differs from exact reduction")
                maps[representation]["hann"] = baseline
                maps[representation]["consensus"] = metric_module.common_grid(geometric, uint8_export=quantized)
            target_folder = out / scroll
            target_folder.mkdir()
            np.save(target_folder / "consensus_fine.npy", geometric, allow_pickle=False)
            np.save(target_folder / "consensus_uint8.npy", np.rint(geometric.clip(0, 1) * np.float32(255)).astype(np.uint8), allow_pickle=False)
            for representation, arms in maps.items():
                for arm, values in arms.items():
                    np.save(target_folder / f"{arm}_{representation}_common.npy", values, allow_pickle=False)
            complete[scroll] = {"labels": labels, "support": support, "maps": maps}
            result["fields"].append({"scroll": scroll, "diagnostics": diagnostics,
                                     "output_sha256": {path.name: sha(path) for path in sorted(target_folder.glob("*.npy"))}})
            del ct, patches, column_support, baseline_fine, baseline_export, exact_export, arithmetic, geometric
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > protocol["limits"]["peak_rss_soft_kib"]:
                raise MemoryError("secondary_RSS_soft_cap")
        stage = "cross_donor_threshold_transfer"
        for calibration_scroll, evaluation_scroll in (SCROLLS, SCROLLS[::-1]):
            calibration, evaluation = complete[calibration_scroll], complete[evaluation_scroll]
            for representation in ("float", "uint8"):
                entry = {"calibration_scroll": calibration_scroll, "evaluation_scroll": evaluation_scroll,
                         "representation": representation, "calibration": {}, "evaluation": {}}
                for arm in ("hann", "consensus"):
                    calibration_metric = metric_module.measure(calibration["maps"][representation][arm], calibration["labels"], calibration["support"])
                    entry["calibration"][arm] = {name: calibration_metric[name] for name in (
                        "background", "allowed_background_count", "background_positive_count", "background_tied_at_threshold_count",
                        "threshold_strict_greater", "unused_background_budget_due_to_ties")}
                    entry["evaluation"][arm] = measure_transferred(
                        evaluation["maps"][representation][arm], evaluation["labels"], evaluation["support"],
                        calibration_metric["threshold_strict_greater"], metric_module)
                entry["gate"] = compare_transfer(entry["evaluation"]["hann"], entry["evaluation"]["consensus"], protocol["gate"])
                result["transfers"].append(entry)
        stage = "final_integrity"
        if sha(source / "result.json") != result["primary_result_sha256"] or any(sha(path) != digest for path, digest in verified):
            raise ValueError("A primary artifact changed during analysis")
        result.update({"status": "complete", "numeric_gate_pass": all(item["gate"]["pass"] for item in result["transfers"]),
                       "source_artifacts_unchanged": True, "verified_source_artifact_count": len(verified)})
        code = 0
    except Exception as error:
        result.update({"status": "resource_unassessable" if isinstance(error, (TimeoutError, MemoryError)) else "failed",
                       "stage": stage, "error_type": type(error).__name__, "reason": str(error)})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_alarm)
        result.update({"wall_seconds": time.monotonic() - started,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        save_json(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "numeric_gate_pass", "wall_seconds")}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
