#!/usr/bin/env python3
"""Independent, zero-CNN review of the frozen depth-bias diagnostic.

Reuses only the pinned released Hecate.forward BODY, with saved head outputs
supplied in place of learned layers. Its return is extended with weights/support
for an independent descriptive calculation. No producer projection, assembly,
summary, display or comparison helper is imported. PNGs are decoded for exact
pixel verification, never interpreted as letters. All data/report paths are CLI
arguments and stay outside Git. No network, checkpoint load or training occurs.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "5b48d79fde5a41f8239b8548f1609d4349de70ff"
ORDERS = ("source", "reverse")
DEPTH_REPLAY_LIMIT = 900
WALL_SECONDS = 300
RSS_KIB = 1_572_864


class ReviewFailure(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise ReviewFailure(reason)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def exact(first, second, label):
    first, second = np.asarray(first), np.asarray(second)
    require(first.dtype == second.dtype and first.shape == second.shape, label + ": shape/dtype mismatch")
    # Viewing all bytes also distinguishes +0/-0 and every float32 ULP.
    a = np.ascontiguousarray(first).view(np.uint8)
    b = np.ascontiguousarray(second).view(np.uint8)
    require(np.array_equal(a, b), label + ": byte mismatch; no tolerance allowed")


def array(path, dtype, shape):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.dtype == np.dtype(dtype) and value.shape == tuple(shape), path.name + ": shape/dtype mismatch")
    return value


def close_array(value):
    mapped = getattr(value, "_mmap", None)
    if mapped is not None:
        mapped.close()


def q8(value):
    return np.rint(np.clip(value, 0, 1) * np.float32(255)).astype(np.uint8)


class Review:
    def __init__(self):
        self.start = time.monotonic()
        self.checked = {}
        self.scalars = 0
        self.max_scalar_difference = 0.
        self.replays = 0

    def budget(self):
        require(time.monotonic() - self.start < WALL_SECONDS, "Independent review time cap")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= RSS_KIB, "Independent review RSS cap")
        require(self.replays <= DEPTH_REPLAY_LIMIT, "Independent projection count cap")

    def file(self, label, path, expected):
        require(digest(path) == expected, label + ": SHA256 mismatch")
        if label in self.checked:
            require(self.checked[label][1] == expected, "Conflicting file identities")
        self.checked[label] = (Path(path), expected)
        self.budget()

    def scalar(self, actual, reported, label):
        self.scalars += 1
        if isinstance(actual, (int, bool, str)) or actual is None:
            require(actual == reported, label + ": scalar mismatch")
        else:
            error = abs(float(actual) - float(reported))
            require(math.isfinite(error) and error <= 1e-10, label + ": numeric summary mismatch")
            self.max_scalar_difference = max(self.max_scalar_difference, error)


def released_projection(torch, source_path):
    """Return official post-CNN algebra, with only the return tuple augmented."""
    tree = ast.parse(source_path.read_text())
    classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Hecate"]
    require(len(classes) == 1, "Released Hecate class is ambiguous")
    functions = [x for x in classes[0].body if isinstance(x, ast.FunctionDef) and x.name == "forward"]
    require(len(functions) == 1, "Released forward method is ambiguous")
    method = copy.deepcopy(functions[0])
    returns = [x for x in ast.walk(method) if isinstance(x, ast.Return)]
    require(len(returns) == 1, "Released projection return contract changed")
    returns[0].value = ast.Tuple(elts=[returns[0].value, ast.Name(id="weights", ctx=ast.Load()), ast.Name(id="support", ctx=ast.Load())], ctx=ast.Load())
    method.name = "released_postcnn"
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(module, "<pinned-released-postcnn>", "exec"), namespace)
    return namespace["released_postcnn"], hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()


def replay(torch, official, logits, attention, support, alpha):
    """No learned layer: the two head callables return the archived tensors."""
    tensor_logits = torch.from_numpy(np.array(logits, dtype=np.float32, copy=True))[None, None]
    tensor_attention = torch.from_numpy(np.array(attention, dtype=np.float32, copy=True))[None, None]
    require(bool(torch.isfinite(tensor_logits).all()) and bool(torch.isfinite(tensor_attention).all()), "Nonfinite archived tensors")
    stub = SimpleNamespace(
        features_logits=lambda _: (None, tensor_logits),
        canonical=SimpleNamespace(decoder=SimpleNamespace(depth_collapse=SimpleNamespace(attn_conv=lambda _: tensor_attention))),
        depth_coordinate_scale=torch.tensor(alpha, dtype=torch.float32),
        patch_size=(64, 256, 256), margin=1, xy_stride=4,
    )
    # Only image.shape is used after the two archived head outputs are supplied.
    image = SimpleNamespace(shape=(1, 1, 64, 256, 256))
    valid = torch.from_numpy(support.astype(np.uint8, copy=True))[None, None, None].expand(1, 1, 64, 256, 256)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        model_logits, weights, internal_support = official(stub, image, valid)
        probability = model_logits.float().sigmoid()[0, 0].numpy().copy()
        emitted = weights[0, 0].numpy().copy()
        eligible = internal_support[0, 0].any(dim=0).numpy().copy()
    require(np.isfinite(probability).all() and np.isfinite(emitted).all(), "Nonfinite released projection")
    return probability, emitted, eligible


def descriptive_weights(weights, eligible, physical_depth):
    # Each ROW is a supported XY location, unlike the producer's depth-first
    # diagnostic array. Normalisation and summation occur independently here.
    rows = np.ascontiguousarray(weights[:, eligible].T, dtype=np.float64)
    require(rows.shape[1] == 62, "Attention depth population changed")
    sums = rows.sum(axis=1, keepdims=True)
    require(np.all(sums > 0), "Nonpositive weight mass on supported cells")
    rows /= sums
    logs = np.zeros_like(rows)
    np.log(rows, out=logs, where=rows > 0)
    entropy = -np.sum(rows * logs, axis=1)
    depth = np.sum(rows * physical_depth[None], axis=1)
    return entropy, np.exp(entropy), depth


def attention_summary(chunks, reported, review):
    names = ("entropy_nats", "effective_plane_count", "mean_depth_relative_to_plane54_um")
    result = {}
    require(reported["quantiles"] == [0.05, 0.5, 0.95] and reported["not_ink_localization_truth"] is True, "Attention reporting contract differs")
    for index, name in enumerate(names):
        values = np.concatenate([row[index] for row in chunks])
        require(values.size and np.isfinite(values).all(), "Empty/nonfinite attention summary")
        # Independently implement the specified linear order-statistic quantile.
        ordered = np.sort(values)
        quantiles = []
        for q in (0.05, 0.5, 0.95):
            position = (len(ordered) - 1) * q
            lower, upper = math.floor(position), math.ceil(position)
            fraction = position - lower
            quantiles.append(float(ordered[lower] + fraction * (ordered[upper] - ordered[lower])))
        review.scalar(int(values.size), reported["values"][name]["count"], name + " count")
        for measured, saved in zip(quantiles, reported["values"][name]["percentiles_5_50_95"]):
            review.scalar(measured, saved, name + " quantile")
        result[name] = {"count": int(values.size), "percentiles_5_50_95": quantiles}
    return result


def compare_exports(folder, prefix, context, row, review):
    from PIL import Image
    expected_names = {prefix + suffix for suffix in (
        "_context_float.npy", "_core_float.npy", "_core_Q8.npy", "_core_Q8.png",
        "_fixed_logit_display.npy", "_fixed_logit_display.png",
    )}
    require(set(row["output_sha256"]) == expected_names, "Unexpected saved export population")
    core = context[64:1984, 64:1984]
    require(core.shape == (1920, 1920), "Core extent differs")
    eps = float(np.finfo(np.float32).eps)
    clipped = np.clip(core.astype(np.float64), eps, 1. - eps)
    grey = np.clip((np.log(clipped) - np.log1p(-clipped) + 8.) / 24., 0., 1.)
    expected = {
        "_context_float.npy": context,
        "_core_float.npy": core,
        "_core_Q8.npy": q8(core),
        "_core_Q8.png": q8(core),
        "_fixed_logit_display.npy": grey.astype(np.float32),
        "_fixed_logit_display.png": np.rint(grey * 255.).astype(np.uint8),
    }
    for suffix, value in expected.items():
        name = prefix + suffix
        path = folder / name
        review.file("diagnostic/" + name, path, row["output_sha256"][name])
        if path.suffix == ".png":
            with Image.open(path) as image:
                require(image.mode == "L", "Non-grayscale diagnostic image")
                actual = np.asarray(image)
                exact(actual, value, name)
        else:
            actual = array(path, value.dtype, value.shape)
            exact(actual, value, name)
            close_array(actual)


def compare_probability_change(base, alternative, supported, reported, review):
    require(base.shape == alternative.shape == supported.shape == (1920, 1920), "Probability comparison geometry differs")
    difference = alternative.astype(np.float64) - base.astype(np.float64)
    changed = q8(base) != q8(alternative)
    upward = (base <= 0.5) & (alternative > 0.5)
    downward = (base > 0.5) & (alternative <= 0.5)
    actual = {}
    for name, mask in (("all_core", np.ones(base.shape, dtype=bool)), ("nonempty_CT_columns", supported)):
        count = int(np.count_nonzero(mask))
        require(count > 0, "Empty probability comparison population")
        delta = difference[mask]
        values = {
            "pixels": count,
            "mean_signed_probability_change": float(np.sum(delta, dtype=np.float64) / count),
            "mean_absolute_probability_change": float(np.sum(np.abs(delta), dtype=np.float64) / count),
            "maximum_absolute_probability_change": float(np.max(np.abs(delta))),
            "changed_Q8_pixels": int(np.count_nonzero(changed & mask)),
            "changed_Q8_fraction": float(np.count_nonzero(changed & mask) / count),
            "upward_0_5_crossings": int(np.count_nonzero(upward & mask)),
            "downward_0_5_crossings": int(np.count_nonzero(downward & mask)),
        }
        for key, value in values.items():
            review.scalar(value, reported[name][key], name + "/" + key)
        actual[name] = values
    return actual


def run(args):
    source, folder, output = (path.resolve() for path in (args.source_dir, args.result_dir, args.output))
    for path in (source, folder, output):
        require(not path.is_relative_to(ROOT), "Source, diagnostic and review output must remain outside Git")
    require(source != folder and not source.is_relative_to(folder) and not folder.is_relative_to(source), "Source and diagnostic directories must be separate")
    require(not output.exists(), "Refusing to overwrite an independent review")
    review = Review()
    report = {"schema": "scrollprize.depth-bias-independent-review.v1", "status": "started", "CNN_forwards": 0,
              "checkpoint_loads": 0, "network_requests": 0, "new_CT_requests": 0, "training_steps": 0,
              "image_interpretation": False, "efficacy_gate_created": False, "producer_helpers_imported": False}
    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(ReviewFailure("Independent review time cap")))
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    exit_code = 1
    try:
        result = read_json(folder / "result.json")
        require(result["status"] in {"complete", "failed", "unassessable"}, "Producer result is not terminal")
        report["producer_status"] = result["status"]
        if result["status"] != "complete":
            report["status"] = "original_failure_retained_without_counterfactual_review"
            return_code = 0
            return return_code
        require(result["freeze_reference"] == FREEZE, "Depth diagnostic freeze differs")
        git_env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
        for name in ("run.py", "protocol.json", "test_run.py", "README.md"):
            path = HERE / name
            frozen = subprocess.check_output(["git", "show", FREEZE + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT, env=git_env)
            require(path.read_bytes() == frozen, "Published diagnostic source changed")
        protocol = read_json(HERE / "protocol.json")
        review.file("code/protocol.json", HERE / "protocol.json", result["protocol_sha256"])
        review.file("code/run.py", HERE / "run.py", result["runner_sha256"])
        for relative, expected in protocol["dependency_sha256"].items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe dependency path")
            review.file("code/" + relative, ROOT / relative, expected)
        review.file("diagnostic/result.json", folder / "result.json", digest(folder / "result.json"))
        review.file("glyph/result.json", source / "result.json", result["source_result_sha256"])
        primary = read_json(source / "result.json")
        require(primary["status"] == "complete" and primary["freeze_reference"] == protocol["source_freeze"], "Original fixture incomplete or wrong freeze")
        require(primary["completed_fixture_forwards"] == 450 and primary["completed_forwards"] == 454, "Original forward population differs")
        primary_rows = {row["order"]: row for row in primary["orders"]}
        baselines = {row["order"]: row for row in result["baseline_orders"]}
        counterfactuals = {row["order"]: row for row in result["orders"]}
        require(all(set(rows) == set(ORDERS) for rows in (primary_rows, baselines, counterfactuals)), "Both orders required")
        require(all(len(rows) == 2 for rows in (primary["orders"], result["baseline_orders"], result["orders"])), "Duplicate order")
        contract = read_json(source / "depth_archive_contract.json")
        require(contract == primary["depth_archive_contract"], "Archive contract differs from primary receipt")
        review.file("glyph/depth_archive_contract.json", source / "depth_archive_contract.json", digest(source / "depth_archive_contract.json"))
        alpha = np.float32(contract["depth_coordinate_scale"])
        require(np.isfinite(alpha) and contract["depth_coordinate_scale_originating_dtype"] == "torch.float32", "Invalid originating scalar")
        require(hashlib.sha256(alpha.tobytes()).hexdigest() == contract["depth_coordinate_scale_array_sha256"], "Scalar byte hash differs")
        review.scalar(float(alpha), result["actual_alpha_read_after_freeze"], "Actual alpha")
        review.scalar(bool(alpha == 0), result["zero_bias_is_parameter_noop"], "No-op status")
        require(contract["model_input_depth_margin"] == 1 and contract["spatial_stride"] == 4 and contract["raw_attention_scores_before_coordinate_bias"] is True, "Archived projection geometry differs")
        require(contract["attention_z_coordinate"] == "(arange(62)-31)/31", "Attention coordinate convention differs")
        geometry = {}
        for order in ORDERS:
            first, last = ([22, 86] if order == "source" else [23, 87])
            full = np.arange(first, last, dtype=np.int64)
            if order == "reverse":
                full = full[::-1]
            planes = full[1:-1]
            require(primary_rows[order]["source_z_halfopen"] == [first, last] and primary_rows[order]["forwards"] == 225, "Physical slab/patch population differs")
            require(planes.tolist() == result["depth_mapping"][order] == contract["source_feature_z_indices_by_order"][order], "Physical feature plane mapping differs")
            geometry[order] = {"original_planes": planes.tolist(), "relative_to_plane54_um": (planes.astype(np.float64) - 54.) * 2.4}
        review.file("glyph/source_union65_uint8.npy", source / "source_union65_uint8.npy", primary["source_union_file_sha256"])
        union = array(source / "source_union65_uint8.npy", np.uint8, (65, 2048, 2048))
        supports = {}
        for order, slab in (("source", slice(0, 64)), ("reverse", slice(1, 65))):
            support = np.any(union[slab], axis=0)
            path = source / ("ct_nonempty_support_" + order + ".npy")
            review.file("glyph/" + path.name, path, primary["source_support"]["file_sha256"][order])
            saved = array(path, np.bool_, (2048, 2048))
            exact(saved, support, "Derived native column support")
            close_array(saved)
            supports[order] = support
        close_array(union)
        del union
        archive_paths = {}
        for order in ORDERS:
            archive_paths[order] = {}
            for name, shape in (("pre_collapse_logits", (225, 62, 64, 64)), ("attention_scores", (225, 62, 64, 64)), ("patch_probability", (225, 256, 256))):
                record = primary_rows[order]["depth_archive"]["arrays"][name]
                require(record["file"] == order + "_" + name + ".npy", "Unexpected archive name")
                path = source / record["file"]
                require(path.stat().st_size == record["bytes"], "Archive byte count differs")
                review.file("glyph/" + path.name, path, record["sha256"])
                value = array(path, np.float32, shape)
                close_array(value)
                archive_paths[order][name] = path
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require(torch.__version__ == result["runtime"]["torch"] == primary["runtime"]["torch"], "Original Torch runtime required")
        require(np.__version__ == result["runtime"]["numpy"], "Original NumPy runtime required")
        official, projection_ast_sha = released_projection(torch, ROOT / "firstletters-colab/ink/third_party/hecate/hecate.py")
        one_d = torch.hann_window(256, periodic=False, dtype=torch.float32)
        two_d = one_d[:, None] * one_d[None, :]
        window = (two_d / two_d.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
        details = {order: {} for order in ORDERS}
        for phase, coefficient, records in (("baseline", float(alpha), baselines), ("zero_bias", 0., counterfactuals)):
            if phase == "zero_bias":
                require(review.replays == 450, "Counterfactual attempted before every exact baseline")
                print("Both original orders match all archived patch bits; checking alpha=0.", flush=True)
            for order in ORDERS:
                row = records[order]
                if phase == "zero_bias":
                    require(row["counterfactual_alpha"] == 0., "Counterfactual must change only alpha to zero")
                    require(row["actual_forward_precision"] == primary_rows[order]["actual_forward_precision"] and row["originating_archive_dtypes"] == primary_rows[order]["depth_archive"]["originating_dtypes"], "Originating precision provenance differs")
                values = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in archive_paths[order].items() if phase == "baseline" or name != "patch_probability"}
                total = np.zeros((2048, 2048), dtype=np.float32)
                mass = np.zeros_like(total)
                diagnostic_chunks = []
                for index, (top, left) in enumerate((y, x) for y in range(0, 1793, 128) for x in range(0, 1793, 128)):
                    if index % 15 == 0:
                        review.budget()
                    region = np.s_[top:top+256, left:left+256]
                    probability, weights, eligible = replay(torch, official, values["pre_collapse_logits"][index], values["attention_scores"][index], supports[order][region], coefficient)
                    review.replays += 1
                    if phase == "baseline":
                        exact(probability, values["patch_probability"][index], "Released baseline patch")
                    diagnostic_chunks.append(descriptive_weights(weights, eligible, geometry[order]["relative_to_plane54_um"]))
                    np.add(total[region], probability * supports[order][region] * window, out=total[region])
                    np.add(mass[region], window, out=mass[region])
                require(index == 224 and np.all(mass > 0), "Incomplete row-major assembly")
                context = total / mass
                require(np.isfinite(context).all(), "Nonfinite assembled context")
                if phase == "baseline":
                    review.scalar(225, row["patch_probabilities_bitwise_exact"], "Baseline patch count")
                    require(row["canonical_Q8_context_exact"] is True, "Original Q8 replay not claimed exact")
                    for suffix, expected in (("context_uint8.npy", q8(context)), ("core_uint8.npy", q8(context[64:1984, 64:1984]))):
                        name = order + "_" + suffix
                        path = source / name
                        review.file("glyph/" + name, path, primary_rows[order]["output_sha256"][name])
                        saved = array(path, np.uint8, expected.shape)
                        exact(saved, expected, "Original canonical " + name)
                        close_array(saved)
                compare_exports(folder, order + "_" + phase, context, row, review)
                attention = attention_summary(diagnostic_chunks, row["attention"], review)
                details[order][phase] = {"projection_replays": 225, "all_context_and_core_exports_exact": True, "attention": attention}
                if phase == "zero_bias":
                    baseline_core = array(folder / (order + "_baseline_core_float.npy"), np.float32, (1920, 1920))
                    details[order][phase]["probability_changes"] = compare_probability_change(baseline_core, context[64:1984, 64:1984], supports[order][64:1984, 64:1984], row["probability_changes"], review)
                    close_array(baseline_core)
                for value in values.values():
                    close_array(value)
                del values, total, mass, context, diagnostic_chunks, probability, weights, eligible
                review.budget()
        baseline_receipt = read_json(folder / "baseline_replay.json")
        require(baseline_receipt == {"exact": True, "orders": result["baseline_orders"], "projection_replays": 450}, "Baseline completion receipt differs")
        review.file("diagnostic/baseline_replay.json", folder / "baseline_replay.json", digest(folder / "baseline_replay.json"))
        require(result["projection_replays"] == review.replays == 900 and result["all_baselines_exact_before_counterfactual"] is True, "Projection count/sequence contract differs")
        for name in ("CNN_forwards", "checkpoint_loads", "network_requests", "new_CT_requests", "training_steps"):
            require(result[name] == 0, "Producer nonzero " + name)
        require(result["both_orders_retained"] is True and result["old_gates_changed"] is False and result["unread_promotion"] is False, "Unexpected promotion/selection")
        require(result["source_artifacts_unchanged"] is True and result["consumed_source_artifact_count"] == 14, "Producer consumed-source accounting differs")
        require(result["wall_seconds"] < protocol["limits"]["wall_seconds"] and result["peak_rss_kib"] <= protocol["limits"]["peak_rss_soft_kib"], "Producer resource receipt exceeded its declared cap")
        for label, (path, expected) in review.checked.items():
            require(digest(path) == expected, label + ": changed during review")
            review.budget()
        report.update({
            "status": "passed", "freeze_reference": FREEZE,
            "baseline_patch_probabilities_bitwise_exact": 450,
            "both_baselines_completed_before_review_counterfactual": True,
            "zero_bias_projection_replays": 450,
            "saved_output_artifacts_verified": 24,
            "pngs_compared_as_pixels_without_interpretation": 8,
            "actual_alpha": float(alpha), "zero_bias_is_parameter_noop": bool(alpha == 0),
            "depth_mapping": {order: geometry[order]["original_planes"] for order in ORDERS},
            "orders": details, "released_projection_augmented_return_ast_sha256": projection_ast_sha,
            "runtime": {"torch": torch.__version__, "numpy": np.__version__, "CPU_threads": 4, "interop_threads": 1},
            "producer_timing_source": "Saved producing-run receipt; not independently re-timed or system-traced",
            "limits": {"wall_seconds": WALL_SECONDS, "peak_rss_soft_kib": RSS_KIB, "projection_replays": DEPTH_REPLAY_LIMIT},
            "scope_limits": [
                "Release-source projection is reused with archived head outputs; no CNN or checkpoint is evaluated.",
                "Patch and exported-map bits require exact equality. Float64 descriptive summaries use absolute tolerance 1e-10 for independent reduction order.",
                "Source support is recomputed from the saved union; raw CT chunks, acquisition and training provenance are not re-audited here.",
                "Physical depth is relative to source plane54, not an independently established writing surface.",
                "All supported internal positions are included; overlapping patches are not independent physical samples.",
                "No image interpretation, efficacy threshold, preferred order, reading or unread promotion is made.",
            ],
        })
        exit_code = 0
    except Exception as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "reason": str(error), "no_tolerance_relaxation": True})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        report.update({"projection_replays": review.replays, "checked_scalar_count": review.scalars,
                       "maximum_summary_absolute_difference": review.max_scalar_difference,
                       "verified_file_count": len(review.checked),
                       "verified_sha256": {label: item[1] for label, item in review.checked.items()},
                       "checker_sha256": digest(__file__), "wall_seconds": time.monotonic() - review.start,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "projection_replays", "checked_scalar_count", "maximum_summary_absolute_difference", "wall_seconds", "peak_rss_kib")}), flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
