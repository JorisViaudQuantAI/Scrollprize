#!/usr/bin/env python3
"""One frozen, zero-CNN depth counterfactual after exact original replay."""
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
ORDERS = ("source", "reverse")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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
        raise ValueError("A complete public freeze commit is required")
    for name in FROZEN:
        path = HERE / name
        if path.read_bytes() != subprocess.check_output(
                ["git", "show", f"{reference}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT):
            raise ValueError("A depth-bias file differs from its freeze")
    for name, expected in protocol["dependency_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("A frozen dependency changed")


class BaselineMismatch(RuntimeError):
    pass


def require_probability_equal(actual, reconstructed):
    if actual.shape != reconstructed.shape or actual.dtype != np.float32 or reconstructed.dtype != np.float32:
        raise BaselineMismatch("Baseline probability shape or dtype differs")
    if not np.array_equal(actual.view(np.uint32), reconstructed.view(np.uint32)):
        raise BaselineMismatch("Baseline float32 probability bits differ; no tolerance allowed")


def source_planes(order):
    if order == "source":
        return np.arange(23, 85, dtype=np.int64)
    if order == "reverse":
        return np.arange(85, 23, -1, dtype=np.int64)
    raise ValueError("Unknown physical input order")


def project(logits_array, attention_array, raw_support, alpha, planes, torch):
    """Exact post-CNN path plus separate non-predictive weight diagnostics."""
    import torch.nn.functional as functional
    if (np.shape(logits_array) != (62, 64, 64) or np.shape(attention_array) != (62, 64, 64)
            or np.asarray(logits_array).dtype != np.float32 or np.asarray(attention_array).dtype != np.float32):
        raise ValueError("Unexpected archived internal tensor")
    if np.shape(raw_support) != (256, 256) or np.asarray(raw_support).dtype != bool:
        raise ValueError("Expected exact boolean fine-column support")
    if np.shape(planes) != (62,) or not np.isfinite(alpha):
        raise ValueError("Invalid physical depth mapping or coefficient")
    logits = torch.from_numpy(np.array(logits_array, dtype=np.float32, copy=True))[None, None]
    scores = torch.from_numpy(np.array(attention_array, dtype=np.float32, copy=True))[None, None]
    if not torch.isfinite(logits).all().item() or not torch.isfinite(scores).all().item():
        raise ValueError("Nonfinite archive values")
    # torch.any(uint8) in the released runtime is uint8, not boolean.
    mask = torch.from_numpy(np.asarray(raw_support, dtype=np.uint8).copy())
    valid = mask[None, None, None].expand(1, 1, 64, 256, 256)
    coordinate = (torch.arange(62, dtype=torch.float32) - 31) / 31
    scale = torch.tensor(alpha, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        biased = scores + scale * coordinate.view(1, 1, -1, 1, 1)
        support = functional.max_pool3d(valid[:, :, 1:63].float(), (1, 4, 4)).bool()
        weights = biased.masked_fill(~support, -1e4).softmax(2) * support
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
        collapsed = (weights * logits).sum(2).sigmoid()
        resized = functional.interpolate(collapsed, size=(256, 256), mode="bilinear", align_corners=False)
        epsilon = torch.finfo(torch.float32).eps
        final_logits = torch.logit(resized.clamp(epsilon, 1 - epsilon))
        probability = final_logits.float().sigmoid()[0, 0].numpy().copy()
        weight_values = weights[0, 0].numpy().astype(np.float64)
        eligible = support[0, 0].any(0).numpy()
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite reconstructed probability")
    # This normalization is solely for descriptive entropy/physical means.
    # It never changes the weights used by the original projection above.
    selected = weight_values[:, eligible]
    if selected.shape[1]:
        selected /= selected.sum(axis=0, keepdims=True)
    logs = np.zeros_like(selected)
    np.log(selected, out=logs, where=selected > 0)
    entropy = -(selected * logs).sum(axis=0)
    depth_um = ((np.asarray(planes, dtype=np.float64) - 54.) * 2.4)[:, None]
    diagnostics = {"entropy_nats": entropy, "effective_plane_count": np.exp(entropy),
                   "mean_depth_relative_to_plane54_um": (selected * depth_um).sum(axis=0)}
    return probability, diagnostics


def summarize_attention(chunks):
    result = {"population": "Supported internal positions in all225 fixed tiles; physical positions overlap",
              "quantiles": [0.05, 0.5, 0.95], "not_ink_localization_truth": True, "values": {}}
    for name in ("entropy_nats", "effective_plane_count", "mean_depth_relative_to_plane54_um"):
        values = np.concatenate([row[name] for row in chunks])
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Missing or nonfinite supported attention population")
        result["values"][name] = {"count": int(values.size),
                                   "percentiles_5_50_95": [float(value) for value in np.quantile(values, (.05, .5, .95))]}
    return result


class Assembly:
    def __init__(self, window, support):
        if window.shape != (256, 256) or window.dtype != np.float32 or support.shape != (2048, 2048) or support.dtype != bool:
            raise ValueError("Unexpected fixed Hann assembly geometry")
        self.window, self.support = window, support
        self.numerator = np.zeros((2048, 2048), np.float32)
        self.denominator = np.zeros_like(self.numerator)
        self.count = 0

    def add(self, index, probability):
        if index != self.count or probability.shape != (256, 256) or probability.dtype != np.float32:
            raise ValueError("Unexpected row-major prediction")
        y, x = index // 15 * 128, index % 15 * 128
        region = np.s_[y:y+256, x:x+256]
        self.numerator[region] += probability * self.support[region] * self.window
        self.denominator[region] += self.window
        self.count += 1

    def finish(self):
        if self.count != 225 or not (self.denominator > 0).all():
            raise ValueError("Incomplete fixed-grid projection")
        result = self.numerator / self.denominator
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite assembled map")
        return result


def q8(value):
    return np.rint(value.clip(0, 1) * np.float32(255)).astype(np.uint8)


def probability_changes(baseline, alternative, support):
    if baseline.shape != alternative.shape or baseline.shape != support.shape or support.dtype != bool:
        raise ValueError("Unpaired output geometry")
    if not np.isfinite(baseline).all() or not np.isfinite(alternative).all():
        raise ValueError("Nonfinite output comparison")
    delta = alternative.astype(np.float64) - baseline.astype(np.float64)
    changed_q8 = q8(baseline) != q8(alternative)
    original_positive, alternative_positive = baseline > .5, alternative > .5
    result = {"threshold_meaning": "Strict P>0.5 is descriptive only, not an ink/background label"}
    for name, mask in (("all_core", np.ones(baseline.shape, bool)), ("nonempty_CT_columns", support)):
        count = int(mask.sum())
        if not count:
            raise ValueError("Empty comparison population")
        values = delta[mask]
        result[name] = {
            "pixels": count,
            "mean_signed_probability_change": float(values.mean()),
            "mean_absolute_probability_change": float(np.abs(values).mean()),
            "maximum_absolute_probability_change": float(np.abs(values).max()),
            "changed_Q8_pixels": int(changed_q8[mask].sum()),
            "changed_Q8_fraction": float(changed_q8[mask].mean()),
            "upward_0_5_crossings": int(np.count_nonzero(mask & ~original_positive & alternative_positive)),
            "downward_0_5_crossings": int(np.count_nonzero(mask & original_positive & ~alternative_positive)),
        }
    return result


def save_maps(out, name, context, display_module):
    from PIL import Image
    core = context[64:1984, 64:1984].copy()
    display, display_png = display_module.fixed_display(core)
    arrays = {"context_float": context, "core_float": core, "core_Q8": q8(core), "fixed_logit_display": display}
    for suffix, value in arrays.items():
        np.save(out / f"{name}_{suffix}.npy", value, allow_pickle=False)
    Image.fromarray(arrays["core_Q8"]).save(out / f"{name}_core_Q8.png")
    Image.fromarray(display_png).save(out / f"{name}_fixed_logit_display.png")
    return {path.name: sha(path) for path in sorted(out.glob(name + "_*"))}


class Inputs:
    def __init__(self):
        self.verified = {}

    def verify(self, path, expected):
        path = Path(path)
        if sha(path) != expected:
            raise ValueError("A consumed source artifact changed")
        self.verified[path] = expected
        return path

    def array(self, path, expected):
        return np.load(self.verify(path, expected), mmap_mode="r", allow_pickle=False)

    def recheck(self):
        if any(sha(path) != expected for path, expected in self.verified.items()):
            raise ValueError("A source artifact changed during analysis")


def run(args):
    protocol = json.loads((HERE / "protocol.json").read_text())
    verify_freeze(args.freeze_reference, protocol)
    source, out = args.input_dir.resolve(), args.output_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT) or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Use a fresh output directory outsideGit, separate from source")
    out.mkdir(parents=True)
    started = time.monotonic()
    result = {"schema": "scrollprize.depth-bias-control.v1", "status": "started",
              "freeze_reference": args.freeze_reference, "protocol_sha256": sha(HERE / "protocol.json"),
              "runner_sha256": sha(__file__), "CNN_forwards": 0, "checkpoint_loads": 0,
              "network_requests": 0, "new_CT_requests": 0, "training_steps": 0,
              "projection_replays": 0, "all_baselines_exact_before_counterfactual": False,
              "old_gates_changed": False, "unread_promotion": False, "orders": []}
    stage = "source_validation"
    inputs = Inputs()
    code = 1

    def check():
        if time.monotonic()-started >= protocol["limits"]["wall_seconds"]:
            raise TimeoutError("Depth diagnostic wall-time cap")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > protocol["limits"]["peak_rss_soft_kib"]:
            raise MemoryError("Depth diagnostic RSS soft cap")
        if result["projection_replays"] > protocol["limits"]["maximum_projection_replays"]:
            raise ValueError("Projection replay budget exceeded")

    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Depth diagnostic wall-time cap")))
    signal.setitimer(signal.ITIMER_REAL, protocol["limits"]["wall_seconds"])
    try:
        result_path = source / "result.json"
        result["source_result_sha256"] = sha(result_path)
        primary = json.loads(result_path.read_text())
        if (primary["status"] != "complete" or primary["freeze_reference"] != protocol["source_freeze"]
                or primary["completed_fixture_forwards"] != 450 or primary["completed_forwards"] != 454):
            raise ValueError("Both complete original glyph orders are required")
        rows = {row["order"]: row for row in primary["orders"]}
        if set(rows) != set(ORDERS) or len(primary["orders"]) != 2:
            raise ValueError("Unexpected original order population")
        contract_path = source / "depth_archive_contract.json"
        contract = json.loads(contract_path.read_text())
        if contract != primary["depth_archive_contract"]:
            raise ValueError("Archive contract differs from original receipt")
        inputs.verify(contract_path, sha(contract_path))
        expected_planes = {order: source_planes(order).tolist() for order in ORDERS}
        if (contract["source_feature_z_indices_by_order"] != expected_planes
                or contract["model_input_depth_margin"] != 1 or contract["spatial_stride"] != 4
                or contract["attention_z_coordinate"] != "(arange(62)-31)/31"
                or not contract["raw_attention_scores_before_coordinate_bias"]):
            raise ValueError("Archive depth/support convention differs")
        alpha = np.float32(contract["depth_coordinate_scale"])
        if (not np.isfinite(alpha) or contract["depth_coordinate_scale_originating_dtype"] != "torch.float32"
                or array_sha(alpha) != contract["depth_coordinate_scale_array_sha256"]):
            raise ValueError("Recorded scalar identity differs")
        result["actual_alpha_read_after_freeze"] = float(alpha)
        result["zero_bias_is_parameter_noop"] = bool(alpha == 0)
        result["depth_mapping"] = expected_planes
        union = inputs.array(source / "source_union65_uint8.npy", primary["source_union_file_sha256"])
        if union.shape != (65, 2048, 2048) or union.dtype != np.uint8:
            raise ValueError("Unexpected native source union")
        supports, archive_paths = {}, {}
        for order, depth_slice in (("source", slice(0, 64)), ("reverse", slice(1, 65))):
            derived = np.any(union[depth_slice], axis=0)
            saved = inputs.array(source / f"ct_nonempty_support_{order}.npy", primary["source_support"]["file_sha256"][order])
            if saved.dtype != bool or not np.array_equal(saved, derived):
                raise ValueError("Saved support differs from original selected source slab")
            supports[order] = derived
            row = rows[order]
            if row["forwards"] != 225 or row["source_z_halfopen"] != ([22, 86] if order == "source" else [23, 87]):
                raise ValueError("Unexpected physical order or patch count")
            paths = {}
            for name, shape in (("pre_collapse_logits", (225, 62, 64, 64)),
                                ("attention_scores", (225, 62, 64, 64)),
                                ("patch_probability", (225, 256, 256))):
                record = row["depth_archive"]["arrays"][name]
                if record["file"] != f"{order}_{name}.npy":
                    raise ValueError("Unexpected archive filename")
                value = inputs.array(source / record["file"], record["sha256"])
                if value.shape != shape or value.dtype != np.float32:
                    raise ValueError("Unexpected archive shape or stored dtype")
                paths[name] = source / record["file"]
                del value
            archive_paths[order] = paths
            del saved
        del union
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if torch.__version__ != primary["runtime"]["torch"]:
            raise ValueError("Use the original Torch runtime for exact projection replay")
        result["runtime"] = {"torch": torch.__version__, "numpy": np.__version__, "CPU_threads": 4}
        display_module = module(ROOT / "firstletters-colab/prediction_precision/run.py", "depth_bias_fixed_display")
        window = torch.hann_window(256, periodic=False, dtype=torch.float32)
        window = window[:, None] * window[None, :]
        window = (window/window.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
        result["baseline_orders"] = []
        # Finish both original-order replays before computing ANY alpha=0 map.
        for order in ORDERS:
            stage = "baseline_replay_" + order
            values = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in archive_paths[order].items()}
            assembly = Assembly(window, supports[order])
            attention_chunks = []
            for index in range(225):
                if index % 15 == 0:
                    check()
                y, x = index // 15 * 128, index % 15 * 128
                probability, diagnostics = project(values["pre_collapse_logits"][index], values["attention_scores"][index],
                    supports[order][y:y+256, x:x+256], float(alpha), source_planes(order), torch)
                result["projection_replays"] += 1
                require_probability_equal(values["patch_probability"][index], probability)
                assembly.add(index, probability)
                attention_chunks.append(diagnostics)
            context = assembly.finish()
            original = inputs.array(source / f"{order}_context_uint8.npy", rows[order]["output_sha256"][f"{order}_context_uint8.npy"])
            if original.dtype != np.uint8 or not np.array_equal(original, q8(context)):
                raise BaselineMismatch("Canonical full-context Q8 replay differs")
            core_original = inputs.array(source / f"{order}_core_uint8.npy", rows[order]["output_sha256"][f"{order}_core_uint8.npy"])
            if not np.array_equal(core_original, original[64:1984, 64:1984]):
                raise BaselineMismatch("Original saved core differs from its context")
            summary = {"order": order, "patch_probabilities_bitwise_exact": 225,
                       "canonical_Q8_context_exact": True, "attention": summarize_attention(attention_chunks),
                       "output_sha256": save_maps(out, order + "_baseline", context, display_module)}
            result["baseline_orders"].append(summary)
            del values, assembly, attention_chunks, context, original, core_original, probability, diagnostics
            check()
        if result["projection_replays"] != 450:
            raise ValueError("Incomplete baseline replay")
        result["all_baselines_exact_before_counterfactual"] = True
        save_json(out / "baseline_replay.json", {"exact": True, "orders": result["baseline_orders"], "projection_replays": 450})
        for order in ORDERS:
            stage = "zero_bias_projection_" + order
            values = {name: np.load(archive_paths[order][name], mmap_mode="r", allow_pickle=False)
                      for name in ("pre_collapse_logits", "attention_scores")}
            assembly = Assembly(window, supports[order])
            attention_chunks = []
            for index in range(225):
                if index % 15 == 0:
                    check()
                y, x = index // 15 * 128, index % 15 * 128
                probability, diagnostics = project(values["pre_collapse_logits"][index], values["attention_scores"][index],
                    supports[order][y:y+256, x:x+256], 0., source_planes(order), torch)
                result["projection_replays"] += 1
                assembly.add(index, probability)
                attention_chunks.append(diagnostics)
            context = assembly.finish()
            baseline_core = np.load(out / f"{order}_baseline_core_float.npy", mmap_mode="r", allow_pickle=False)
            row = {"order": order, "actual_forward_precision": rows[order]["actual_forward_precision"],
                   "originating_archive_dtypes": rows[order]["depth_archive"]["originating_dtypes"],
                   "counterfactual_alpha": 0., "source_z_halfopen": rows[order]["source_z_halfopen"],
                   "probability_changes": probability_changes(baseline_core, context[64:1984, 64:1984], supports[order][64:1984, 64:1984]),
                   "attention": summarize_attention(attention_chunks),
                   "output_sha256": save_maps(out, order + "_zero_bias", context, display_module),
                   "legibility": "unassessed; no automatic preferred order or projection"}
            result["orders"].append(row)
            del values, assembly, attention_chunks, context, baseline_core, probability, diagnostics
            check()
        stage = "final_integrity"
        inputs.recheck()
        if sha(result_path) != result["source_result_sha256"] or result["projection_replays"] != 900:
            raise ValueError("A source receipt changed or projection count differs")
        check()
        result.update({"status": "complete", "source_artifacts_unchanged": True,
                       "consumed_source_artifact_count": len(inputs.verified), "both_orders_retained": True})
        code = 0
    except Exception as error:
        result.update({"status": "unassessable" if isinstance(error, (BaselineMismatch, TimeoutError, MemoryError)) else "failed",
                       "stage": stage, "error_type": type(error).__name__, "reason": str(error),
                       "no_tolerance_relaxation": True})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result.update({"wall_seconds": time.monotonic()-started,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        save_json(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "projection_replays", "all_baselines_exact_before_counterfactual", "wall_seconds")}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
