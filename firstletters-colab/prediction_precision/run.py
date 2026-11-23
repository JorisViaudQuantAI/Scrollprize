#!/usr/bin/env python3
"""Zero-forward serialization and fixed-display analysis of retained predictions."""
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
FROZEN = ("protocol.json", "run.py", "test_run.py", "README.md")
EPS = float(np.finfo(np.float32).eps)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_freeze(reference, protocol):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("A full published freeze commit is required")
    for name in FROZEN:
        path = HERE / name
        if path.read_bytes() != subprocess.check_output(
                ["git", "show", f"{reference}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT):
            raise ValueError("A precision-analysis file differs from its freeze")
    for name, expected in protocol["dependency_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("A pinned dependency changed")


def probability_array(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Expected a finite two-dimensional float32 probability map")
    # An FP32 weighted sum/division can differ infinitesimally from the exact
    # convex combination. Official export clips; retain and report raw floats.
    if value.min() < -1e-6 or value.max() > 1 + 1e-6:
        raise ValueError("Probability map exceeds numeric roundoff allowance")
    return value


def quantize16(probability):
    probability = probability_array(probability)
    return np.rint(np.clip(probability.astype(np.float64), 0, 1) * 65535.).astype(np.uint16)


def integer_common(export):
    export = np.asarray(export)
    if export.ndim != 2 or export.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("Expected a two-dimensional uint8 or uint16 export")
    if any(size == 0 or size % 4 for size in export.shape):
        raise ValueError("Export dimensions must be positive multiples of four")
    h, w = export.shape
    sums = export.reshape(h//4, 4, w//4, 4).sum((1, 3), dtype=np.uint64)
    denominator = 16 * int(np.iinfo(export.dtype).max)
    return (sums.astype(np.float64) / denominator).astype(np.float32)


def fixed_display(probability):
    probability = probability_array(probability)
    clipped = np.clip(probability.astype(np.float64), EPS, 1. - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    grey = np.clip((logits + 8.) / 24., 0., 1.)
    return grey.astype(np.float32), np.rint(grey * 255.).astype(np.uint8)


def restoration(float_metric, q16_metric, settings):
    recall_difference = abs(q16_metric["recall_at_background_5pct"] - float_metric["recall_at_background_5pct"])
    ap_difference = abs(q16_metric["ap"] - float_metric["ap"])
    return {"pass": bool(recall_difference <= settings["absolute_recall_difference_from_float_maximum"]
                         and ap_difference <= settings["absolute_AP_difference_from_float_maximum"]),
            "absolute_recall_difference": float(recall_difference),
            "absolute_AP_difference": float(ap_difference),
            "scope": "Serialization fidelity only; no unread or detector gate"}


def save_views(folder, name, fine, q8, q16):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    grey, image = fixed_display(fine)
    arrays = {"float": fine, "actual_q8": q8, "q16": q16, "fixed_logit_display": grey}
    for suffix, value in arrays.items():
        np.save(folder / f"{name}_{suffix}.npy", value, allow_pickle=False)
    Image.fromarray(q8).save(folder / f"{name}_actual_q8.png")
    Image.fromarray(q16).save(folder / f"{name}_q16.png")
    Image.fromarray(image).save(folder / f"{name}_fixed_logit_display.png")
    return {path.name: sha(path) for path in sorted(folder.glob(name + "_*"))}


def axis_starts(length, patch, stride):
    if length < patch or not 0 < stride <= patch:
        raise ValueError("Only complete unpadded patches are accepted")
    return list(range(0, length-patch, stride)) + [length-patch]


def assemble_hann(patches, support, window, stride):
    """Exact FP32 probability assembly from cached outputs; no model call."""
    patches, support, window = map(np.asarray, (patches, support, window))
    if support.ndim != 2 or support.dtype != bool or window.ndim != 2 or window.shape[0] != window.shape[1]:
        raise ValueError("Unexpected support/window layout")
    if window.dtype != np.float32 or not np.isfinite(window).all() or not (window > 0).all():
        raise ValueError("Expected finite positive FP32 weights")
    patch = window.shape[0]
    ys, xs = axis_starts(support.shape[0], patch, stride), axis_starts(support.shape[1], patch, stride)
    if patches.shape != (len(ys)*len(xs), patch, patch) or patches.dtype != np.float32:
        raise ValueError("All complete row-major patch probabilities are required")
    numerator = np.zeros(support.shape, np.float32)
    denominator = np.zeros_like(numerator)
    for index, (y, x) in enumerate((y, x) for y in ys for x in xs):
        probability = patches[index]
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Invalid cached probability")
        region = np.s_[y:y+patch, x:x+patch]
        # Match released probability*support*win and row-major FP32 updates.
        numerator[region] += probability * support[region] * window
        denominator[region] += window
    if not (denominator > 0).all():
        raise ValueError("Incomplete Hann coverage")
    return probability_array(numerator / denominator)


class Inputs:
    def __init__(self):
        self.verified = {}

    def array(self, path, expected):
        path = Path(path)
        if sha(path) != expected:
            raise ValueError("A consumed source artifact changed")
        self.verified[path] = expected
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def recheck(self):
        if any(sha(path) != expected for path, expected in self.verified.items()):
            raise ValueError("A consumed source artifact changed during analysis")


def control_analysis(source, out, primary, protocol, inputs, check):
    if primary["status"] != "complete" or primary["forwards_completed"] != 128 or primary["freeze_reference"] != protocol["control_source_freeze"]:
        raise ValueError("Expected the completed frozen two-field primary run")
    by_scroll = {row["scroll"]: row for row in primary["fields"]}
    if set(by_scroll) != set(protocol["controls"]["fields"]) or len(primary["fields"]) != 2:
        raise ValueError("Unexpected control donors")
    metric_module = module(ROOT / "firstletters-colab/hecate_blending/metrics.py", "precision_control_metrics")
    rows = []
    for scroll in protocol["controls"]["fields"]:
        folder, receipt = source / scroll, by_scroll[scroll]
        hashes = receipt["output_sha256"]
        labels = inputs.array(folder / "labels.npy", hashes["labels.npy"])
        support = inputs.array(folder / "common_support.npy", hashes["common_support.npy"])
        row = {"scroll": scroll, "assemblies": {}}
        for arm in ("hann", "central", "uniform"):
            check()
            fine = probability_array(inputs.array(folder / f"{arm}_fine.npy", hashes[f"{arm}_fine.npy"]))
            q8 = inputs.array(folder / f"{arm}_uint8.npy", hashes[f"{arm}_uint8.npy"])
            actual_q8 = np.rint(fine.clip(0, 1) * np.float32(255)).astype(np.uint8)
            if q8.dtype != np.uint8 or not np.array_equal(q8, actual_q8):
                raise ValueError("Actual Q8 differs from the original export operation")
            q16 = quantize16(fine)
            common = {"float": metric_module.common_grid(fine), "q8": integer_common(q8), "q16": integer_common(q16)}
            for source_rep, our_rep in (("float", "float"), ("uint8", "q8")):
                name = f"{arm}_{source_rep}_common.npy"
                saved = inputs.array(folder / name, hashes[name])
                if not np.array_equal(saved, common[our_rep]):
                    raise ValueError("Historical common-grid representation did not reproduce exactly")
            measured = {rep: metric_module.measure(values, labels, support) for rep, values in common.items()}
            saturation = {}
            for rep, export in (("q8", q8), ("q16", q16)):
                maximum = int(np.iinfo(export.dtype).max)
                saturation[rep] = {
                    "maximum_code": maximum,
                    "fine_pixels_at_maximum": int(np.count_nonzero(export == maximum)),
                    "fine_pixels_at_zero": int(np.count_nonzero(export == 0)),
                    "background_common_cells_at_maximum": int(np.count_nonzero(support & (labels == 1) & (common[rep] == 1))),
                    "ink_common_cells_at_maximum": int(np.count_nonzero(support & (labels == 2) & (common[rep] == 1))),
                }
            row["assemblies"][arm] = {
                "metrics": measured, "saturation": saturation,
                "restoration": restoration(measured["float"], measured["q16"], protocol["controls"]["restoration"]),
                "output_sha256": save_views(out / scroll, arm, fine, q8, q16),
            }
            for rep, values in common.items():
                common_path = out / scroll / f"{arm}_{rep}_common.npy"
                np.save(common_path, values, allow_pickle=False)
                row["assemblies"][arm]["output_sha256"][common_path.name] = sha(common_path)
            del fine, q8, q16, actual_q8, common, saved
            check()
        row["Hann_restoration_pass"] = row["assemblies"]["hann"]["restoration"]["pass"]
        rows.append(row)
    return {"fields": rows, "Hann_restoration_pass": all(row["Hann_restoration_pass"] for row in rows),
            "scope": "Exploratory serialization comparison on previously seen development controls"}


def glyph_analysis(source, out, primary, protocol, inputs, check):
    if primary["status"] != "complete" or primary["completed_fixture_forwards"] != 450 or primary["freeze_reference"] != protocol["glyph_source_freeze"]:
        raise ValueError("Both complete frozen published-glyph orders are required")
    by_order = {row["order"]: row for row in primary["orders"]}
    if set(by_order) != {"source", "reverse"} or len(primary["orders"]) != 2:
        raise ValueError("Both physical depth orders must be retained")
    import torch
    if torch.__version__ != primary["runtime"]["torch"]:
        raise ValueError("Use the original Torch runtime for exact window construction")
    window = torch.hann_window(256, periodic=False, dtype=torch.float32)
    window = window[:, None] * window[None, :]
    window = (window / window.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
    orders = []
    for order in ("source", "reverse"):
        check()
        row = by_order[order]
        patch_record = row["depth_archive"]["arrays"]["patch_probability"]
        if patch_record["file"] != f"{order}_patch_probability.npy" or row["forwards"] != 225:
            raise ValueError("Unexpected archive origin or patch count")
        patches = inputs.array(source / patch_record["file"], patch_record["sha256"])
        support = inputs.array(source / f"ct_nonempty_support_{order}.npy", primary["source_support"]["file_sha256"][order])
        if support.shape != (2048, 2048):
            raise ValueError("Unexpected full context support")
        float_context = assemble_hann(patches, support, window, 128)
        original_context = inputs.array(source / f"{order}_context_uint8.npy", row["output_sha256"][f"{order}_context_uint8.npy"])
        reconstructed_q8 = np.rint(float_context.clip(0, 1) * np.float32(255)).astype(np.uint8)
        if original_context.dtype != np.uint8 or not np.array_equal(original_context, reconstructed_q8):
            raise ValueError("Canonical whole-context Q8 replay failed; no display accepted")
        fine = float_context[64:1984, 64:1984].copy()
        original_core = inputs.array(source / f"{order}_core_uint8.npy", row["output_sha256"][f"{order}_core_uint8.npy"])
        if original_core.dtype != np.uint8 or not np.array_equal(original_core, original_context[64:1984, 64:1984]):
            raise ValueError("Canonical glyph core differs from the retained export")
        q16 = quantize16(fine)
        order_result = {
            "order": order, "actual_forward_precision": row["actual_forward_precision"],
            "canonical_context_Q8_replay_exact": True,
            "replayed_context_pixels": int(original_context.size),
            "core_float_minimum": float(fine.min()), "core_float_maximum": float(fine.max()),
            "source_z_halfopen": row["source_z_halfopen"],
            "output_sha256": save_views(out, order, fine, original_core, q16),
            "legibility": "unassessed; fixed display is not a new detector or inferred reading",
        }
        orders.append(order_result)
        del patches, support, float_context, original_context, reconstructed_q8, fine, original_core, q16
        check()
    return {"orders": orders, "both_orders_retained": True,
            "scope": "Fixed visibility and serialization changes only; original Q8 files unchanged"}


def run(args):
    protocol = json.loads((HERE / "protocol.json").read_text())
    verify_freeze(args.freeze_reference, protocol)
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT) or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Use a fresh directory outsideGit and separate from the source tree")
    out.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema": "scrollprize.prediction-precision.v1", "status": "started", "mode": args.mode,
              "freeze_reference": args.freeze_reference, "protocol_sha256": sha(HERE / "protocol.json"),
              "runner_sha256": sha(__file__), "detector_forwards": 0, "network_requests": 0,
              "new_CT_requests": 0, "training_steps": 0, "old_gates_changed": False}
    def check():
        if time.monotonic() - started >= protocol["limits"]["wall_seconds"]:
            raise TimeoutError("Analysis wall-time cap reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > protocol["limits"]["peak_rss_soft_kib"]:
            raise MemoryError("Analysis RSS soft cap reached")
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Analysis wall-time cap reached")))
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["wall_seconds"])
    code = 1
    inputs = Inputs()
    try:
        result_path = source / "result.json"
        result["source_result_sha256"] = sha(result_path)
        primary = json.loads(result_path.read_text())
        analyze = control_analysis if args.mode == "controls" else glyph_analysis
        result["analysis"] = analyze(source, out, primary, protocol, inputs, check)
        inputs.recheck()
        if sha(result_path) != result["source_result_sha256"]:
            raise ValueError("Source receipt changed during analysis")
        check()
        result.update({"status": "complete", "source_artifacts_unchanged": True,
                       "consumed_source_artifact_count": len(inputs.verified)})
        code = 0
    except Exception as error:
        result.update({"status": "resource_unassessable" if isinstance(error, (TimeoutError, MemoryError)) else "failed",
                       "error_type": type(error).__name__, "reason": str(error)})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result.update({"wall_seconds": time.monotonic()-started,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        save_json(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "mode", "wall_seconds")}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--mode", choices=("controls", "glyph"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
