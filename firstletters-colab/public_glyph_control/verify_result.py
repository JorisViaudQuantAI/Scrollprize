#!/usr/bin/env python3
"""Offline independent review of the completed published-glyph fixture.

This utility never loads trained weights, downloads data, or evaluates a CNN.
It reconstructs the acquired CT union from every retained raw source chunk,
checks the saved precision pilot, and repeats only the explicit FP32 depth
projection and Hann assembly from the archived intermediate tensors. Numerical
differences are measured, retained and reported; approximate equality is never
called exact. No image interpretation or new reading is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "cada2a5497a47868d8dfde68f04e1a333a7723fb"
FROZEN = ("run.py", "protocol.json", "selection.json", "README.md", "test_geometry.py")
ORDERS = ("source", "reverse")


class ReviewFailure(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise ReviewFailure(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def outside_git(path):
    path = Path(path).expanduser().resolve()
    require(path != ROOT and ROOT not in path.parents, "Retained data and review output must stay outside Git")
    return path


def same_bytes(first, second, name):
    first, second = np.asarray(first), np.asarray(second)
    require(first.dtype == second.dtype and first.shape == second.shape, name + ": dtype/shape differs")
    first = np.ascontiguousarray(first).view(np.uint8)
    second = np.ascontiguousarray(second).view(np.uint8)
    require(np.array_equal(first, second), name + ": bytes differ")


def load_array(path, dtype, shape):
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    require(result.dtype == np.dtype(dtype) and result.shape == tuple(shape), path.name + ": unexpected dtype/shape")
    return result


def same_scalar(expected, observed, name):
    if isinstance(expected, (bool, int, str)) or expected is None:
        require(expected == observed, name + ": scalar differs")
    else:
        difference = abs(float(expected) - float(observed))
        require(math.isfinite(difference) and difference <= 1e-12, name + ": numeric scalar differs")


def local_inputs(args, protocol):
    checkpoint = outside_git(args.fine_checkpoint)
    expected = protocol["checkpoint"]
    require(checkpoint.is_file() and checkpoint.stat().st_size == expected["bytes"] and sha(checkpoint) == expected["sha256"],
            "Local checkpoint identity differs")
    paths = {"PHerc0841": outside_git(args.pilot_0841_context), "PHerc0814": outside_git(args.pilot_0814_context)}
    for donor, path in paths.items():
        require(sha(path) == protocol["precision_pilot"]["inputs"][donor]["fine_context_sha256"], "Historical pilot input identity differs")
        load_array(path, np.uint8, protocol["precision_pilot"]["input_shape_zyx"])
    return {"checkpoint_identity_verified_without_loading_weights": True,
            "historical_pilot_inputs_verified": 2}


def review_pilot(run_dir, result, protocol):
    pilot = read_json(run_dir / "precision_pilot.json")
    require(pilot == result["precision_pilot"], "Pilot receipt differs from final result")
    settings = protocol["precision_pilot"]
    records = pilot["fixed_order_results"]
    require(len(records) == settings["pilot_forwards"] == 4, "Pilot forward count differs")
    values, seconds, files = {}, {}, []
    for index, (record, fixed) in enumerate(zip(records, settings["fixed_order"])):
        require(record["index"] == index and record["scroll"] == fixed["scroll"] and record["precision"] == fixed["precision"],
                "Pilot order differs from the frozen four calls")
        donor, precision = fixed["scroll"], fixed["precision"]
        name = f"pilot_{donor}_{precision}.npy"
        require(record["full_score_file"] == name and sha(run_dir / name) == record["full_score_file_sha256"], "Pilot score digest differs")
        score = load_array(run_dir / name, np.float32, (256, 256))
        require(np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all(), "Pilot score outside probability range")
        values[donor, precision] = score
        seconds[donor, precision] = float(record["forward_seconds"])
        require(math.isfinite(seconds[donor, precision]) and seconds[donor, precision] > 0, "Invalid pilot time")
        files.append(name)
    donors = sorted(settings["inputs"])
    comparisons = {}
    for donor in donors:
        difference = np.abs(values[donor, "fp32"].astype(np.float64) - values[donor, "bf16"].astype(np.float64))
        maximum, mean = float(difference.max()), float(difference.mean())
        passed = maximum <= settings["full_score_max_abs_error_inclusive"] and mean <= settings["full_score_mean_abs_error_inclusive"]
        comparisons[donor] = {"max_full_score_abs_error": maximum, "mean_full_score_abs_error": mean, "numerical_pass": passed}
        for name, value in comparisons[donor].items():
            same_scalar(value, pilot["by_donor"][donor][name], "Pilot numerical " + donor + "/" + name)
    means = {precision: sum(seconds[donor, precision] for donor in donors) / len(donors) for precision in ("fp32", "bf16")}
    for precision in means:
        same_scalar(means[precision], pilot["mean_seconds"][precision], "Pilot timing mean")
    timing_pass = means["bf16"] < means["fp32"]
    accepted = timing_pass and all(value["numerical_pass"] for value in comparisons.values())
    selected = "bf16" if accepted else "fp32"
    for name, value in (("timing_pass", timing_pass), ("bf16_accepted", accepted), ("selected_precision", selected),
                        ("warmup_forwards", 0), ("first_unwarmed_forward", settings["first_unwarmed_forward"]),
                        ("not_an_independent_speed_benchmark", True)):
        same_scalar(value, pilot[name], "Pilot decision " + name)
    worst = max(seconds[donor, "fp32"] for donor in donors)
    same_scalar(worst, pilot["fp32_worst_pilot_seconds"], "Pilot worst FP32 time")
    return {"pilot_scores_verified": 4, "selected_precision": selected, "bf16_accepted": accepted,
            "by_donor": comparisons, "mean_seconds": means, "timing_pass": timing_pass,
            "timing_source": "Saved producing-run measurements, not independently re-timed",
            "pilot_receipt_sha256": sha(run_dir / "precision_pilot.json")}, set(files)


def independent_plan(selection, source):
    y0, x0 = selection["context"]["origin_yx"]
    height, width = selection["context"]["shape_yx"]
    y1, x1 = y0 + height, x0 + width
    depth, source_height, source_width = source["shape_zyx"]
    chunk_depth, chunk_height, chunk_width = source["chunks_zyx"]
    require(depth == chunk_depth == 109 and height == width == 2048, "Unexpected native source geometry")
    require(0 <= y0 < y1 <= source_height and 0 <= x0 < x1 <= source_width, "Context leaves source")
    require(selection["context"]["end_yx_exclusive"] == [y1, x1], "Context extent inconsistent")
    require(selection["core"]["origin_yx"] == [y0 + 64, x0 + 64] and selection["core"]["shape_yx"] == [1920, 1920],
            "Core/halo geometry differs")
    plan = []
    for top in range((y0 // chunk_height) * chunk_height, y1, chunk_height):
        for left in range((x0 // chunk_width) * chunk_width, x1, chunk_width):
            start_y, end_y = max(y0, top), min(y1, top + chunk_height)
            start_x, end_x = max(x0, left), min(x1, left + chunk_width)
            plan.append({"key": f"0/{top // chunk_height}/{left // chunk_width}",
                         "source_yx": [start_y-top, end_y-top, start_x-left, end_x-left],
                         "target_yx": [start_y-y0, end_y-y0, start_x-x0, end_x-x0]})
    require(len(plan) == source["complete_chunk_count"] == 289, "Expected exactly 289 native chunks")
    return plan


def review_source(run_dir, result, protocol, selection):
    source = protocol["source"]
    plan = independent_plan(selection, source)
    saved_plan = read_json(run_dir / "acquisition_plan.json")
    require(saved_plan["chunk_plan"] == plan and saved_plan["chunk_count"] == 289, "Acquisition plan differs from independent geometry")
    require(saved_plan["complete_chunk_reads"] is True and saved_plan["no_resampling"] is True and saved_plan["missing_objects_are_fatal"] is True,
            "Acquisition policy differs")
    base = selection["source_base"].rstrip("/")
    require(saved_plan["source_base"] == base and saved_plan["retained_source_z_halfopen"] == [22, 87], "Acquisition identity/depth differs")
    receipt = read_json(run_dir / "source_receipt.json")
    require(not receipt["failed_assets"], "Completed run has failed source acquisitions")
    rows = receipt["completed_assets"]
    by_url = {row["url"]: row for row in rows}
    require(len(rows) == len(by_url) == 291, "Expected 289 chunks and two metadata objects")
    expected_urls = {base + "/" + entry["suffix"] for entry in source["metadata"]}
    expected_urls.update(base + "/0/" + entry["key"] for entry in plan)
    require(set(by_url) == expected_urls, "Acquisition included missing or additional sources")
    paths, transferred = {}, 0
    for url, row in by_url.items():
        parts = urlsplit(url)
        require(parts.scheme == "https" and not parts.query and not parts.username and not parts.password, "Noncanonical source URL")
        require(row.get("reused_old_cache") is False, "Unexpected source-cache reuse")
        path = run_dir / "source_cache" / hashlib.sha256(url.encode()).hexdigest()
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Retained source object hash/size differs")
        paths[url] = path
        transferred += row["bytes"]
    require(transferred == result["new_application_body_bytes"] == receipt["new_application_body_bytes"] == source["metadata_and_chunk_body_bytes"],
            "Complete source-body accounting differs")
    require(transferred <= protocol["limits"]["new_body_bytes"], "Source body cap exceeded")
    require(receipt["requests_including_redirects"] == result["requests_including_redirects"] <= protocol["limits"]["requests_including_redirects"],
            "Source request count differs or exceeds cap")
    metadata = {}
    for entry in source["metadata"]:
        url = base + "/" + entry["suffix"]
        require(by_url[url]["bytes"] == entry["bytes"] and by_url[url]["sha256"] == entry["sha256"], "Pinned source metadata differs")
        metadata[entry["suffix"]] = read_json(paths[url])
    zarray, attrs = metadata["0/.zarray"], metadata[".zattrs"]
    require(zarray["zarr_format"] == 2, "Unexpected Zarr version")
    keys = {"shape": "shape_zyx", "chunks": "chunks_zyx", "dtype": "dtype", "order": "order", "compressor": "compressor",
            "filters": "filters", "fill_value": "fill_value", "dimension_separator": "dimension_separator"}
    require(all(zarray[key] == source[value] for key, value in keys.items()), "Decoded source metadata mismatch")
    require(len(attrs["multiscales"]) == 1 and attrs["num_slices"] == 109, "Ambiguous multiscale/depth metadata")
    scale = attrs["multiscales"][0]
    require([axis["name"] for axis in scale["axes"]] == ["z", "y", "x"] and all(axis["unit"] == "micrometer" for axis in scale["axes"]), "Source axes/units differ")
    transforms = [entry["coordinateTransformations"] for entry in scale["datasets"] if entry["path"] == "0"]
    require(transforms == [[{"scale": [2.4, 2.4, 2.4], "type": "scale"}, {"translation": [0.0, 0.0, 0.0], "type": "translation"}]],
            "Native physical spacing/translation differs")
    volume_path = run_dir / "source_union65_uint8.npy"
    require(sha(volume_path) == result["source_union_file_sha256"], "Source union digest differs")
    volume = load_array(volume_path, np.uint8, (65, 2048, 2048))
    coverage = np.zeros((2048, 2048), dtype=np.uint8)
    compared = 0
    for item in plan:
        url = base + "/0/" + item["key"]
        require(by_url[url]["bytes"] == source["complete_chunk_bytes"] == 109 * 128 * 128, "Incomplete native chunk")
        chunk = np.memmap(paths[url], mode="r", dtype=np.uint8, shape=(109, 128, 128))
        sy0, sy1, sx0, sx1 = item["source_yx"]
        ty0, ty1, tx0, tx1 = item["target_yx"]
        require(not coverage[ty0:ty1, tx0:tx1].any(), "Independent chunk union overlaps")
        same_bytes(volume[:, ty0:ty1, tx0:tx1], chunk[22:87, sy0:sy1, sx0:sx1], "Native source-to-union voxel copy")
        coverage[ty0:ty1, tx0:tx1] = 1
        compared += 65 * (ty1-ty0) * (tx1-tx0)
        del chunk
    require((coverage == 1).all() and compared == 65 * 2048 * 2048, "Incomplete native union reconstruction")
    same_bytes(load_array(run_dir / "acquisition_support.npy", bool, (2048, 2048)), coverage.astype(bool), "Acquisition support")
    pre = read_json(run_dir / "pre_fixture_inference.json")
    require(pre["source_union_file_sha256"] == result["source_union_file_sha256"] and pre["pilot_completed_forwards"] == 4 and pre["fixture_forwards_before_freeze"] == 0,
            "Pre-fixture identity/count record differs")
    require(pre["selected_forward_precision"] == result["precision_pilot"]["selected_precision"], "Pre-fixture precision changed")
    require(pre["source_support"] == result["source_support"] and pre["source_z_halfopen_by_order"] == protocol["inference"]["source_z_halfopen_by_order"],
            "Pre-fixture depth/support changed")
    same_scalar(pre["new_application_body_bytes"], transferred, "Pre-fixture body count")
    return volume, {"retained_source_assets_verified": 291, "native_CT_chunks_verified": 289,
                    "native_union_voxels_compared_bytewise": compared, "native_union_complete_and_exact": True,
                    "source_body_bytes": transferred, "source_requests_including_redirects": result["requests_including_redirects"],
                    "source_receipt_sha256": sha(run_dir / "source_receipt.json"),
                    "acquisition_support_sha256": sha(run_dir / "acquisition_support.npy")}


def reconstruct_probability(logits_array, attention_array, raw_support, coordinate_scale, torch):
    """Repeat only the pinned post-CNN FP32 projection, in its exact order."""
    import torch.nn.functional as functional
    logits = torch.from_numpy(np.array(logits_array, dtype=np.float32, copy=True))[None, None]
    scores = torch.from_numpy(np.array(attention_array, dtype=np.float32, copy=True))[None, None]
    require(tuple(logits.shape) == tuple(scores.shape) == (1, 1, 62, 64, 64), "Projection archive shape differs")
    require(torch.isfinite(logits).all().item() and torch.isfinite(scores).all().item(), "Nonfinite projection archive")
    # Official predict's torch.any(uint8) is uint8 in Torch2.6. Make that
    # representation explicit, then reproduce the model's float conversion.
    mask = torch.from_numpy(np.asarray(raw_support, dtype=np.uint8).copy())
    valid = mask[None, None, None].expand(1, 1, 64, 256, 256)
    coordinate = (torch.arange(62, dtype=torch.float32) - 31) / 31
    scale = torch.tensor(coordinate_scale, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        biased_scores = scores + scale * coordinate.view(1, 1, -1, 1, 1)
        support = functional.max_pool3d(valid[:, :, 1:63].float(), (1, 4, 4)).bool()
        weights = biased_scores.masked_fill(~support, -1e4).softmax(2) * support
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
        collapsed = (weights * logits).sum(2).sigmoid()
        resized = functional.interpolate(collapsed, size=(256, 256), mode="bilinear", align_corners=False)
        epsilon = torch.finfo(torch.float32).eps
        final_logits = torch.logit(resized.clamp(epsilon, 1 - epsilon))
        probability = final_logits.float().sigmoid()[0, 0].numpy().copy()
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), "Reconstructed probability invalid")
    return probability


class DifferenceSummary:
    def __init__(self):
        self.arrays = self.differing_arrays = self.values = self.differing_values = 0
        self.maximum = self.absolute_sum = 0.0
        self.maximum_ulp_difference = 0
        self.worst_index = None

    def add(self, actual, reconstructed, index):
        require(actual.dtype == reconstructed.dtype == np.float32 and actual.shape == reconstructed.shape, "Difference shape/dtype differs")
        require(np.isfinite(actual).all() and np.isfinite(reconstructed).all(), "Nonfinite difference input")
        delta = np.abs(actual.astype(np.float64) - reconstructed.astype(np.float64))
        bits_a, bits_b = np.ascontiguousarray(actual).view(np.uint32), np.ascontiguousarray(reconstructed).view(np.uint32)
        differing = int(np.count_nonzero(bits_a != bits_b))
        maximum = float(delta.max())
        if maximum > self.maximum:
            self.maximum, self.worst_index = maximum, index
        # All compared values are nonnegative probabilities, making ordered
        # uint32 representations a valid ULP-distance calculation.
        ulps = np.abs(bits_a.astype(np.int64) - bits_b.astype(np.int64))
        self.maximum_ulp_difference = max(self.maximum_ulp_difference, int(ulps.max()))
        self.arrays += 1
        self.differing_arrays += bool(differing)
        self.values += actual.size
        self.differing_values += differing
        self.absolute_sum += float(delta.sum())

    def report(self):
        return {"arrays_checked": self.arrays, "arrays_with_byte_differences": self.differing_arrays,
                "values_checked": self.values, "values_with_byte_differences": self.differing_values,
                "maximum_absolute_error": self.maximum, "mean_absolute_error": self.absolute_sum / self.values,
                "maximum_ULP_difference": self.maximum_ulp_difference, "worst_patch_index": self.worst_index,
                "bitwise_exact": self.differing_values == 0,
                "within_1e_minus_6_absolute_error_diagnostic": self.maximum <= 1e-6}


def review_order(order, row, run_dir, volume, contract, protocol, selected_precision, window, torch):
    require(row["order"] == order and row["forwards"] == 225 and len(row["forward_seconds"]) == 225, "Fixture forward/order count differs")
    require(row["actual_forward_precision"] == selected_precision and row["official_predict_precision_argument"] == "fp32", "Order precision differs")
    require(all(math.isfinite(value) and value > 0 for value in row["forward_seconds"]), "Invalid fixture timing record")
    require(read_json(run_dir / (order + "_result.json")) == row, "Per-order receipt differs")
    if order == "source":
        indices = np.arange(64)
        expected_source_interval = [22, 86]
    else:
        indices = np.arange(64, 0, -1)
        expected_source_interval = [23, 87]
    require(row["source_z_halfopen"] == expected_source_interval == protocol["inference"]["source_z_halfopen_by_order"][order], "Physical depth order differs")
    require(row["is_pure_reversal_of_same_physical_slab"] is False, "Odd-depth slab distinction lost")
    require(contract["source_feature_z_indices_by_order"][order] == (indices[1:-1] + 22).tolist(), "Feature depth mapping differs")
    slab = np.asarray(volume[int(indices.min()):int(indices.max())+1])
    support = slab.any(axis=0)
    support_name = f"ct_nonempty_support_{order}.npy"
    same_bytes(load_array(run_dir / support_name, bool, (2048, 2048)), support, "Order-specific CT support")
    saved_support = read_json(run_dir / "result.json")["source_support"]
    require(sha(run_dir / support_name) == saved_support["file_sha256"][order], "Order support digest differs")
    counts = {"context_nonempty_columns": int(support.sum()), "context_total_columns": support.size,
              "core_nonempty_columns": int(support[64:1984, 64:1984].sum()), "core_total_columns": 1920 * 1920}
    require(counts == saved_support["counts"][order], "Order support counts differ")
    archive = row["depth_archive"]
    require(archive["patch_count"] == 225 and archive["stored_dtype"] == "float32", "Archive count/dtype differs")
    require(archive["storage_order"] == "row-major XY tiles; depth follows each model input order", "Archive index contract differs")
    expected_shapes = {"pre_collapse_logits": (225, 62, 64, 64), "attention_scores": (225, 62, 64, 64), "patch_probability": (225, 256, 256)}
    require(set(archive["arrays"]) == set(expected_shapes), "Archive tensor set differs")
    arrays, archive_bytes = {}, 0
    for name, shape in expected_shapes.items():
        item = archive["arrays"][name]
        filename = f"{order}_{name}.npy"
        require(item["file"] == filename, "Archive filename/order differs")
        path = run_dir / filename
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Archive size/hash differs")
        arrays[name] = load_array(path, np.float32, shape)
        archive_bytes += path.stat().st_size
    types = archive["originating_dtypes"]
    require(types["pre_collapse_logits"] == types["final_model_logits"] == "torch.float32" and types["patch_probability"] == "float32", "Projection originating dtype differs")
    require(types["attention_scores"] in ("torch.float32", "torch.bfloat16"), "Unsupported originating attention precision")
    expected_outputs = {f"{order}_{kind}{suffix}" for kind in ("context", "core") for suffix in ("_uint8.npy", ".png")}
    require(set(row["output_sha256"]) == expected_outputs, "Order output inventory differs")
    for filename, digest in row["output_sha256"].items():
        require(sha(run_dir / filename) == digest, "Output image/array digest differs")
    saved_context = load_array(run_dir / f"{order}_context_uint8.npy", np.uint8, (2048, 2048))
    saved_core = load_array(run_dir / f"{order}_core_uint8.npy", np.uint8, (1920, 1920))
    same_bytes(saved_core, saved_context[64:1984, 64:1984], "Core crop from complete official context")
    from PIL import Image
    for kind, expected in (("context", saved_context), ("core", saved_core)):
        with Image.open(run_dir / f"{order}_{kind}.png") as image:
            require(image.mode == "L", "Official PNG is not grayscale")
            same_bytes(np.asarray(image), expected, "PNG pixels versus retained uint8 array")
    numerator_actual = np.zeros((2048, 2048), dtype=np.float32)
    numerator_reconstructed = np.zeros_like(numerator_actual)
    denominator = np.zeros_like(numerator_actual)
    differences = DifferenceSummary()
    input_support_digest = hashlib.sha256()
    scale = contract["depth_coordinate_scale"]
    with torch.inference_mode():
        for index in range(225):
            row_index, column_index = divmod(index, 15)
            y, x = 128 * row_index, 128 * column_index
            region = (slice(y, y+256), slice(x, x+256))
            mask = support[region]
            require(mask.any(), "An empty fixture tile contradicts the exact forward count")
            input_support_digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
            actual = np.asarray(arrays["patch_probability"][index])
            require(np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all(), "Archived probability invalid")
            attention = np.asarray(arrays["attention_scores"][index])
            if types["attention_scores"] == "torch.bfloat16":
                bf16_roundtrip = torch.from_numpy(attention.copy()).to(torch.bfloat16).float().numpy()
                same_bytes(attention, bf16_roundtrip, "BF16-origin attention was not preserved exactly")
            reconstructed = reconstruct_probability(arrays["pre_collapse_logits"][index], attention, mask, scale, torch)
            differences.add(actual, reconstructed, index)
            # Preserve released FP32 operation and summation order. These are
            # two independently assembled maps, never an edited output image.
            numerator_actual[region] += (actual * mask) * window
            numerator_reconstructed[region] += (reconstructed * mask) * window
            denominator[region] += window
    require((denominator > 0).all(), "Hann coverage incomplete")
    actual_map = numerator_actual / denominator
    reconstructed_map = numerator_reconstructed / denominator
    actual_export = np.rint(actual_map.clip(0, 1) * np.float32(255)).astype(np.uint8)
    reconstructed_export = np.rint(reconstructed_map.clip(0, 1) * np.float32(255)).astype(np.uint8)
    same_bytes(actual_export, saved_context, "Archive-probability independent official Hann export")
    same_bytes(actual_export[64:1984, 64:1984], saved_core, "Independent Hann core export")
    float_delta = np.abs(actual_map.astype(np.float64) - reconstructed_map.astype(np.float64))
    integer_delta = np.abs(reconstructed_export.astype(np.int16) - saved_context.astype(np.int16))
    report = {
        "order": order, "fixture_patch_count_verified": 225, "originating_dtypes": types,
        "archive_bytes_verified": archive_bytes, "archive_files_verified": 3, "output_files_verified": 4,
        "raw_CT_support_counts": counts, "tile_XY_support_bytes_sha256": input_support_digest.hexdigest(),
        "projection_reconstruction": differences.report(),
        "actual_archived_probability_Hann_export_bitwise_exact": True,
        "reconstructed_probability_Hann_float_maximum_absolute_error": float(float_delta.max()),
        "reconstructed_probability_Hann_float_mean_absolute_error": float(float_delta.mean()),
        "reconstructed_Hann_context_uint8_different_pixels": int(np.count_nonzero(integer_delta)),
        "reconstructed_Hann_context_uint8_maximum_byte_error": int(integer_delta.max()),
        "reconstructed_Hann_core_uint8_different_pixels": int(np.count_nonzero(integer_delta[64:1984, 64:1984])),
        "decoded_PNGs_equal_retained_uint8_arrays": True,
        "independent_actual_Hann_context_array_sha256": array_sha(actual_export),
    }
    names = {item["file"] for item in archive["arrays"].values()} | expected_outputs | {support_name}
    return report, archive_bytes, names


def verify(args):
    started = time.monotonic()
    run_dir = outside_git(args.run_dir)
    result = read_json(run_dir / "result.json")
    # Refuse live/partial output before opening any prediction archive.
    require(result.get("status") == "complete", "Run is incomplete; do not inspect live archives")
    require(result["freeze_reference"] == FREEZE and result["forward_attempts"] == result["completed_forwards"] == 454 and result["completed_fixture_forwards"] == 450,
            "Complete fixture counts or freeze differ")
    for name in ("training_steps", "unread_CT_requests", "new_checkpoint_downloads"):
        require(result[name] == 0, "Experiment scope differs: " + name)
    require(result["new_reading_claim"] is False and result["unread_promotion"] is False and result["automatic_order_selection"] is False, "Reading/promotion scope changed")
    protocol, selection = read_json(HERE / "protocol.json"), read_json(HERE / "selection.json")
    for name in FROZEN:
        path = HERE / name
        frozen = subprocess.check_output(["git", "show", f"{FREEZE}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(path.read_bytes() == frozen and sha(path) == result["file_sha256"][name], "Frozen glyph file differs: " + name)
    require(sha(HERE / "selection.json") == protocol["pins"]["selection_sha256"], "Selection digest differs")
    for relative, digest in protocol["dependency_sha256"].items():
        require(sha(ROOT / relative) == digest, "Pinned dependency differs")
    require(result["core"] == selection["core"] and result["context"] == selection["context"], "Final result geometry differs")
    local = local_inputs(args, protocol)
    pilot, names = review_pilot(run_dir, result, protocol)
    volume, sources = review_source(run_dir, result, protocol, selection)
    names |= {"source_union65_uint8.npy", "acquisition_support.npy"}
    contract = read_json(run_dir / "depth_archive_contract.json")
    require(contract == result["depth_archive_contract"], "Depth archive contract differs from final result")
    require(contract["raw_attention_scores_before_coordinate_bias"] is True and contract["model_input_depth_margin"] == 1 and contract["spatial_stride"] == 4,
            "Unexpected archive projection contract")
    require(contract["same_450_fixture_forwards_only"] is True and contract["feature_tensors_archived"] is False and contract["no_alternative_projection_selected"] is True,
            "Archive changed producing-run scope")
    require(contract["attention_z_coordinate"] == "(arange(62)-31)/31", "Attention coordinate differs")
    scale = np.array(contract["depth_coordinate_scale"], dtype=np.float32)
    require(np.isfinite(scale).all() and array_sha(scale) == contract["depth_coordinate_scale_array_sha256"] and contract["depth_coordinate_scale_originating_dtype"] == "torch.float32",
            "Coordinate-scale identity differs")
    require([row["order"] for row in result["orders"]] == list(ORDERS), "Both fixed orders were not retained")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    one_dimensional = torch.hann_window(256, periodic=False, dtype=torch.float32)
    window_tensor = one_dimensional[:, None] * one_dimensional[None, :]
    window = (window_tensor / window_tensor.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
    orders, archive_bytes = [], 0
    for row in result["orders"]:
        report, count, files = review_order(row["order"], row, run_dir, volume, contract, protocol, pilot["selected_precision"], window, torch)
        orders.append(report)
        archive_bytes += count
        names |= files
    require(archive_bytes == result["additional_depth_archive_bytes"] <= protocol["archive"]["maximum_additional_disk_bytes"], "Archive disk count differs")
    require(sum(row["fixture_patch_count_verified"] for row in orders) == 450, "Review did not check every fixture patch")
    required_arrays = {name for name in names if name.endswith(".npy")}
    require(required_arrays == {path.name for path in run_dir.glob("*.npy")}, "Retained primary-array inventory differs")
    require(result["both_orders_retained"] is True and result["all_source_hashes_reverified"] is True and result["model_buffers_unchanged"] is True,
            "Producing final-integrity flags differ")
    exact = all(row["projection_reconstruction"]["bitwise_exact"] and row["reconstructed_Hann_context_uint8_different_pixels"] == 0 for row in orders)
    error_sum = sum(row["projection_reconstruction"]["mean_absolute_error"] * row["projection_reconstruction"]["values_checked"] for row in orders)
    total_values = sum(row["projection_reconstruction"]["values_checked"] for row in orders)
    return {
        "schema": "scrollprize.public-glyph-control.independent-review.v1",
        "status": "pass" if exact else "complete_with_projection_numerical_differences",
        "all_projection_reconstructions_bitwise_exact": exact,
        "freeze_reference": FREEZE, "reviewer_sha256": sha(__file__),
        "run_result_sha256": sha(run_dir / "result.json"), "protocol_sha256": sha(HERE / "protocol.json"),
        "frozen_files_verified": 5, "pinned_dependencies_verified": len(protocol["dependency_sha256"]),
        **local, **sources, "precision_pilot": pilot,
        "retained_npy_arrays_verified": len(required_arrays), "PNG_files_verified": 4,
        "fixture_forward_count_verified": 450, "total_producing_forward_count_verified": 454,
        "pre_collapse_logit_values_verified": 450 * 62 * 64 * 64,
        "raw_attention_values_verified": 450 * 62 * 64 * 64,
        "projected_2D_probability_values_compared": total_values,
        "maximum_projection_probability_absolute_error": max(row["projection_reconstruction"]["maximum_absolute_error"] for row in orders),
        "mean_projection_probability_absolute_error": error_sum / total_values,
        "additional_depth_archive_bytes_verified": archive_bytes,
        "depth_archive_contract_sha256": sha(run_dir / "depth_archive_contract.json"),
        "Hann_window_array_sha256": array_sha(window), "orders": orders,
        "CNN_forwards_by_reviewer": 0, "trained_checkpoint_loaded_by_reviewer": False,
        "network_requests_by_reviewer": 0, "training_steps_by_reviewer": 0,
        "image_interpretation_performed": False, "new_reading_or_prize_claim": False,
        "review_wall_seconds": time.monotonic() - started,
        "limitations": [
            "CNN feature computation and pilot timings are not independently rerun; the reviewer validates retained data, reported timings and the explicit post-CNN projection.",
            "Archived tile indexing and the logged coordinate scale rely on the frozen instrumenter; all reconstructed probabilities and exports expose any detected inconsistency.",
            "Source chunks are individually hash-verified and their exact native voxel union is checked; this is not an authenticated atomic snapshot of a mutable remote bucket.",
            "The two odd-depth selections differ by one physical source plane and are not pure reversals of the same slab.",
            "A complete archive and numerical reconstruction do not establish legibility, independent glyph truth or unread-scroll generalisation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fine-checkpoint", required=True, type=Path)
    parser.add_argument("--pilot-0841-context", required=True, type=Path)
    parser.add_argument("--pilot-0814-context", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = outside_git(args.output or args.run_dir / "independent_review.json")
    require(not output.exists(), "Review output already exists; preserve earlier attempts")
    review = verify(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    require(not partial.exists(), "Partial review output already exists")
    partial.write_text(json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(output)
    print(json.dumps({key: review[key] for key in ("status", "fixture_forward_count_verified", "retained_source_assets_verified",
        "maximum_projection_probability_absolute_error", "all_projection_reconstructions_bitwise_exact", "review_wall_seconds")}))
    # The diagnostic tolerance is not a pass criterion. A nonexact reconstruction
    # is explicitly surfaced even if every discrepancy is numerically tiny.
    return 0 if review["all_projection_reconstructions_bitwise_exact"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewFailure as error:
        print(json.dumps({"status": "review_failed", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(1)
