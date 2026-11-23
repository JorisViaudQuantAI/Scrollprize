#!/usr/bin/env python3
"""Independent, zero-forward audit of the completed glyph precision outputs.

Reassemble archived probabilities without importing the production runner;
verify all source/output bytes, original Q8 contexts, Q16 and fixed displays.
Only an aggregate receipt is written. No image interpretation is performed.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "5cdfe2de2deee727808d9258edb20875185ba8ca"
SOURCE_FREEZE = "cada2a5497a47868d8dfde68f04e1a333a7723fb"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def identical(first, second, name):
    first, second = np.asarray(first), np.asarray(second)
    require(first.shape == second.shape and first.dtype == second.dtype,
            "Array type or shape differs: " + name)
    require(first.tobytes(order="C") == second.tobytes(order="C"),
            "Array bytes differ: " + name)


def round_even(value, dtype):
    """Explicit integer nearest-even operation, without producer np.rint."""
    value = np.asarray(value, np.float64)
    lower = np.floor(value).astype(np.uint64)
    residual = value - lower.astype(np.float64)
    increment = (residual > .5) | ((residual == .5) & ((lower & 1) != 0))
    return (lower + increment).astype(dtype)


class Inputs:
    def __init__(self):
        self.files = {}

    def check(self, path, expected):
        path = Path(path)
        require(sha(path) == expected, "Hash differs: " + path.name)
        self.files[path] = expected

    def array(self, path, expected, dtype, shape):
        self.check(path, expected)
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        require(value.shape == shape and value.dtype == np.dtype(dtype),
                "Unexpected array: " + Path(path).name)
        return value

    def recheck(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, "File changed during review: " + path.name)


def verify(args):
    started = time.monotonic()
    output = args.output or args.run_dir / "independent_precision_glyph_review.json"
    require(not output.resolve().is_relative_to(ROOT.resolve()), "Receipt must remain outside Git")
    require(not output.exists(), "Refusing to replace an existing review receipt")
    inputs = Inputs()
    result_path = args.run_dir / "result.json"
    result = read_json(result_path)
    require(result["status"] == "complete" and result["mode"] == "glyph" and
            result["freeze_reference"] == FREEZE, "Expected completed frozen glyph precision analysis")
    inputs.check(result_path, sha(result_path))
    protocol = read_json(HERE / "protocol.json")
    for name in ("protocol.json", "run.py", "test_run.py", "README.md"):
        path = HERE / name
        original = subprocess.check_output(
            ["git", "show", f"{FREEZE}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(path.read_bytes() == original, "Frozen precision source changed")
        inputs.check(path, hashlib.sha256(original).hexdigest())
    for name, digest in protocol["dependency_sha256"].items():
        inputs.check(ROOT / name, digest)
    inputs.check(HERE / "protocol.json", result["protocol_sha256"])
    inputs.check(HERE / "run.py", result["runner_sha256"])
    for name in ("network_requests", "new_CT_requests", "detector_forwards", "training_steps"):
        require(result[name] == 0, "Producing analysis changed zero-forward scope")
    require(result["old_gates_changed"] is False and result["source_artifacts_unchanged"] is True,
            "Producing analysis changed gates or inputs")
    primary_path = args.source_dir / "result.json"
    inputs.check(primary_path, result["source_result_sha256"])
    primary = read_json(primary_path)
    require(primary["status"] == "complete" and primary["freeze_reference"] == SOURCE_FREEZE and
            primary["completed_fixture_forwards"] == 450 and primary["completed_forwards"] == 454,
            "Incomplete or different original glyph experiment")
    original_review_path = args.source_dir / "independent_review.json"
    original_review = read_json(original_review_path)
    inputs.check(original_review_path, sha(original_review_path))
    require(original_review["status"] == "pass" and original_review["all_projection_reconstructions_bitwise_exact"]
            and original_review["run_result_sha256"] == result["source_result_sha256"],
            "Original glyph source has not passed exact independent verification")
    analysis = result["analysis"]
    require(set(analysis) == {"orders", "both_orders_retained", "scope"} and analysis["both_orders_retained"] is True,
            "Unexpected analysis scope or omitted order")
    require(analysis["scope"] == "Fixed visibility and serialization changes only; original Q8 files unchanged",
            "Analysis adds an unsupported effectiveness claim")
    original_rows = {row["order"]: row for row in primary["orders"]}
    observed_rows = {row["order"]: row for row in analysis["orders"]}
    require(len(primary["orders"]) == len(analysis["orders"]) == 2 and
            set(original_rows) == set(observed_rows) == {"source", "reverse"}, "Unexpected order population")
    import torch
    torch.set_num_threads(1)
    require(torch.__version__ == primary["runtime"]["torch"], "Window construction requires original Torch runtime")
    line = torch.hann_window(256, periodic=False, dtype=torch.float32)
    weight = line[:, None] * line[None, :]
    weight = (weight / weight.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
    require(hashlib.sha256(weight.tobytes()).hexdigest() == original_review["Hann_window_array_sha256"],
            "Independent Hann window differs from original reviewed window")
    orders = []
    for order in ("source", "reverse"):
        original, observed = original_rows[order], observed_rows[order]
        require(set(observed) == {"order", "actual_forward_precision", "canonical_context_Q8_replay_exact",
                "replayed_context_pixels", "core_float_minimum", "core_float_maximum", "source_z_halfopen",
                "output_sha256", "legibility"}, "Order adds an unreviewed endpoint")
        require(observed["legibility"] == "unassessed; fixed display is not a new detector or inferred reading",
                "Order asserts unsupported legibility")
        require(observed["actual_forward_precision"] == original["actual_forward_precision"] == "fp32",
                "Producing precision differs")
        expected_depth = [22, 86] if order == "source" else [23, 87]
        require(observed["source_z_halfopen"] == original["source_z_halfopen"] == expected_depth,
                "Physical depth slab differs")
        require(original["forwards"] == 225 and observed["canonical_context_Q8_replay_exact"] is True and
                observed["replayed_context_pixels"] == 2048 * 2048, "Incomplete original context replay")
        archive = original["depth_archive"]["arrays"]["patch_probability"]
        require(archive["file"] == f"{order}_patch_probability.npy", "Archive identity differs")
        patches = inputs.array(args.source_dir / archive["file"], archive["sha256"], np.float32, (225, 256, 256))
        support = inputs.array(args.source_dir / f"ct_nonempty_support_{order}.npy",
                               primary["source_support"]["file_sha256"][order], bool, (2048, 2048))
        total, mass = np.zeros((2048, 2048), np.float32), np.zeros((2048, 2048), np.float32)
        index = 0
        for y in range(0, 1793, 128):
            for x in range(0, 1793, 128):
                probability = np.asarray(patches[index])
                require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
                        "Invalid original patch probability")
                region = np.s_[y:y+256, x:x+256]
                total[region] += (probability * support[region]) * weight
                mass[region] += weight
                index += 1
        require(index == 225 and (mass > 0).all(), "Incomplete context assembly")
        context = total / mass
        context_q8 = round_even(np.clip(context, 0, 1) * np.float32(255), np.uint8)
        canonical_context = inputs.array(args.source_dir / f"{order}_context_uint8.npy",
                original["output_sha256"][f"{order}_context_uint8.npy"], np.uint8, (2048, 2048))
        identical(context_q8, canonical_context, "entire original Q8 context")
        canonical_core = inputs.array(args.source_dir / f"{order}_core_uint8.npy",
                original["output_sha256"][f"{order}_core_uint8.npy"], np.uint8, (1920, 1920))
        identical(canonical_core, canonical_context[64:1984, 64:1984], "original core from context")
        fine = context[64:1984, 64:1984]
        require(float(fine.min()) == observed["core_float_minimum"] and
                float(fine.max()) == observed["core_float_maximum"], "Reported core range differs")
        q16 = round_even(np.clip(fine.astype(np.float64), 0., 1.) * 65535., np.uint16)
        eps = float(np.finfo(np.float32).eps)
        clipped = np.clip(fine.astype(np.float64), eps, 1 - eps)
        grey64 = np.clip((np.log(clipped) - np.log1p(-clipped) + 8) / 24, 0, 1)
        arrays = {"float": fine, "actual_q8": canonical_core, "q16": q16,
                  "fixed_logit_display": grey64.astype(np.float32)}
        images = {"actual_q8": canonical_core, "q16": q16,
                  "fixed_logit_display": round_even(grey64 * 255, np.uint8)}
        names = {f"{order}_{name}.npy" for name in arrays} | {f"{order}_{name}.png" for name in images}
        require(set(observed["output_sha256"]) == names, "Unexpected output inventory")
        for name, expected in arrays.items():
            filename = f"{order}_{name}.npy"
            retained = inputs.array(args.run_dir / filename, observed["output_sha256"][filename], expected.dtype, expected.shape)
            identical(retained, expected, filename)
        for name, expected in images.items():
            filename = f"{order}_{name}.png"
            path = args.run_dir / filename
            inputs.check(path, observed["output_sha256"][filename])
            with Image.open(path) as image:
                decoded = np.asarray(image)
            identical(decoded, expected, filename)
        orders.append({"order": order, "physical_source_z_halfopen": expected_depth,
                       "archived_patches_used": index, "whole_context_Q8_pixels_exact": 2048 * 2048,
                       "core_float_pixels_exact": 1920 * 1920, "core_Q16_pixels_exact": 1920 * 1920,
                       "core_fixed_display_float_and_PNG_pixels_exact": 1920 * 1920,
                       "maximum_core_float_absolute_error": 0.0, "different_Q8_Q16_or_display_pixels": 0,
                       "source_Q8_copies_exact": True, "decoded_PNGs_exact": 3,
                       "output_file_hashes_checked": 7, "source_artifact_hashes_checked": 4,
                       "effectiveness_or_legibility_assessed": False})
        del patches, support, total, mass, context, context_q8, canonical_context, canonical_core, fine, q16, clipped, grey64, arrays, images, retained, expected
    require(result["consumed_source_artifact_count"] == 8, "Producing source count differs")
    inputs.recheck()
    review = {"schema": "scrollprize.prediction-precision-glyph-independent-review.v1", "status": "pass",
              "freeze_reference": FREEZE, "source_freeze_reference": SOURCE_FREEZE,
              "reviewed_result_sha256": sha(result_path), "original_result_sha256": sha(primary_path),
              "original_independent_review_sha256": sha(original_review_path), "reviewer_sha256": sha(__file__),
              "frozen_files_verified": 4, "dependency_pins_verified": 3,
              "source_artifacts_verified": 8, "output_assets_verified": 14, "orders": orders,
              "all_retained_values_and_exports_exact": True, "source_output_and_pins_unchanged_after_review": True,
              "label_inputs_used": 0, "CNN_forwards": 0, "checkpoint_loads": 0, "network_requests": 0,
              "new_CT_arrays_read": 0, "training_steps": 0, "effectiveness_claim": False,
              "image_interpretation_performed": False, "old_gates_changed": False,
              "wall_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "runtime": {"numpy": np.__version__, "torch": torch.__version__},
              "scope": "Independent archive assembly, serialization, fixed-display and integrity audit only. Original post-CNN projection/source acquisition was verified separately. No glyph or ink truth, accuracy, physical-face identity, global generalization or prize claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: review[key] for key in ("status", "source_artifacts_verified", "output_assets_verified", "all_retained_values_and_exports_exact", "wall_seconds")}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/scrollprize-prediction-precision-glyph-v1"))
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/scrollprize-public-glyph-control-v1"))
    parser.add_argument("--output", type=Path)
    return verify(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
