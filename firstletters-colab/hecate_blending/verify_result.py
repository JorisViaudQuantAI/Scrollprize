#!/usr/bin/env python3
"""Independently check a completed Hecate blending run, entirely offline.

No checkpoint is loaded and no trained network is executed. The untouched
released predictor receives a small replay module returning the actual saved
logits. Every input tensor and support tensor is checked byte for byte first.
This file deliberately does not import this experiment's runner or metrics.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "7bb9cd9d02cef0ed2c95fc694da90a5a2c09481c"
FROZEN = ("run.py", "protocol.json", "metrics.py", "test_metrics.py", "test_stitch.py", "README.md")
ARMS = ("central", "hann", "uniform")
COUNT_KEYS = ("eligible_pairs", "both_positive_pairs", "exactly_one_positive_pairs", "both_negative_pairs")


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def json_read(path):
    return json.loads(Path(path).read_text())


def same_bytes(actual, expected, reason):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.dtype == expected.dtype and actual.shape == expected.shape, reason + ": dtype/shape")
    actual = np.ascontiguousarray(actual).view(np.uint8)
    expected = np.ascontiguousarray(expected).view(np.uint8)
    require(np.array_equal(actual, expected), reason + ": byte mismatch")


def outside_repository(path):
    path = Path(path).expanduser().resolve()
    require(path != ROOT and ROOT not in path.parents, "Review data/output must stay outside Git")
    return path


def checked_array(path, dtype, shape):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.dtype == np.dtype(dtype) and value.shape == shape, path.name + ": dtype/shape")
    if value.dtype.kind == "f":
        require(np.isfinite(value).all(), path.name + ": nonfinite value")
    return value


class ScalarComparison:
    def __init__(self):
        self.count = 0
        self.maximum_float_difference = 0.0

    def check(self, expected, recorded, name):
        if isinstance(expected, dict):
            require(isinstance(recorded, dict), name + ": not a mapping")
            for key, value in expected.items():
                require(key in recorded, name + ": missing " + key)
                self.check(value, recorded[key], name + "/" + key)
        elif isinstance(expected, (tuple, list)):
            require(isinstance(recorded, (tuple, list)) and len(expected) == len(recorded), name + ": list mismatch")
            for index, (a, b) in enumerate(zip(expected, recorded)):
                self.check(a, b, name + "/" + str(index))
        elif isinstance(expected, (bool, int)) or expected is None or isinstance(expected, str):
            require(expected == recorded, name + ": exact scalar mismatch")
            self.count += 1
        else:
            require(isinstance(recorded, (int, float)) and not isinstance(recorded, bool), name + ": numeric type")
            difference = abs(float(expected) - float(recorded))
            require(math.isfinite(difference) and difference <= 1e-12, name + ": numeric mismatch")
            self.maximum_float_difference = max(self.maximum_float_difference, difference)
            self.count += 1


def independent_pairs(labels, support, positive):
    """Enumerate directed-down/right edges; each undirected neighbour pair once."""
    ids = np.arange(256 * 256).reshape(256, 256)
    edges = (
        ("vertical", ids[:-1].ravel(), ids[1:].ravel(), np.repeat(np.arange(1, 256), 256)),
        ("horizontal", ids[:, :-1].ravel(), ids[:, 1:].ravel(), np.tile(np.arange(1, 256), 256)),
    )
    labels, support, positive = labels.ravel(), support.ravel(), positive.ravel()
    summaries = {}
    for class_value, class_name in ((2, "ink"), (1, "background")):
        for crosses, location in ((True, "seam"), (False, "within_cell")):
            totals = dict.fromkeys(COUNT_KEYS, 0)
            by_axis = {}
            for axis_name, a, b, destination_coordinate in edges:
                eligible = support[a] & support[b] & (labels[a] == class_value) & (labels[b] == class_value)
                eligible &= (destination_coordinate % 32 == 0) if crosses else (destination_coordinate % 32 != 0)
                pa, pb = positive[a[eligible]], positive[b[eligible]]
                codes = pa.astype(np.uint8) + pb.astype(np.uint8)
                histogram = np.bincount(codes, minlength=3)
                counts = {
                    "eligible_pairs": int(codes.size),
                    "both_positive_pairs": int(histogram[2]),
                    "exactly_one_positive_pairs": int(histogram[1]),
                    "both_negative_pairs": int(histogram[0]),
                }
                by_axis[axis_name] = counts
                for key in COUNT_KEYS:
                    totals[key] += counts[key]
            denominator = totals["eligible_pairs"]
            summaries[location + "_" + class_name] = {
                **totals,
                "both_positive_fraction": totals["both_positive_pairs"] / denominator if denominator else None,
                "exactly_one_positive_fraction": totals["exactly_one_positive_pairs"] / denominator if denominator else None,
                "by_axis": by_axis,
            }
    return summaries


def independent_metrics(score, labels, support, native_measure):
    require(score.shape == labels.shape == support.shape == (256, 256), "Metric grid mismatch")
    require(support.dtype == bool and np.isin(labels, (0, 1, 2)).all(), "Metric support/labels invalid")
    require(np.isfinite(score[support]).all(), "Metric supported score nonfinite")
    require(((score[support] >= 0) & (score[support] <= 1)).all(), "Metric probability outside [0,1]")
    ink = support & (labels == 2)
    background = support & (labels == 1)
    known = ink | background
    n_ink, n_background = int(ink.sum()), int(background.sum())
    require(n_ink > 0 and n_background > 0, "Metric class missing")
    budget = n_background // 20
    # Ascending partition is independent of the production descending sort.
    ordinal = n_background - budget - 1
    threshold = float(np.partition(score[background], ordinal)[ordinal])
    positive = known & (score > threshold)
    p_ink, p_background = int(positive[ink].sum()), int(positive[background].sum())
    # Only AP/AUC use the earlier native-detail implementation. All thresholds,
    # counts, support decisions, adjacency pools, and gates below are independent.
    ranks = native_measure(score, labels, support)
    return {
        "shape": [256, 256], "seam_period": 32,
        "known": n_ink + n_background, "ink": n_ink, "background": n_background,
        "unknown_or_unsupported": int(labels.size - known.sum()),
        "supported_unknown": int((support & (labels == 0)).sum()),
        "unsupported": int((~support).sum()),
        "ap": ranks["ap"], "auc": ranks["auc"],
        "threshold_strict_greater": threshold,
        "threshold_background_rank_zero_based": budget,
        "allowed_background_count": budget,
        "background_positive_count": p_background,
        "background_tied_at_threshold_count": int((score[background] == threshold).sum()),
        "unused_background_budget_due_to_ties": budget - p_background,
        "ink_positive_count": p_ink,
        "ink_tied_at_threshold_count": int((score[ink] == threshold).sum()),
        "recall_at_background_5pct": p_ink / n_ink,
        "background_exceedance": p_background / n_background,
        "pairs": independent_pairs(labels, support, positive),
    }


def independent_decision(central, hann, protocol_gate):
    relevant = ("seam_ink", "seam_background", "within_cell_ink")
    counts = {name: central["pairs"][name]["eligible_pairs"] for name in relevant}
    require(all(counts[name] == hann["pairs"][name]["eligible_pairs"] for name in relevant), "Gate pool differs")
    limits = {
        "minimum_pair_count": protocol_gate["minimum_relevant_pair_pool"],
        "minimum_seam_ink_gain": protocol_gate["seam_ink_pair_absolute_gain_minimum"],
        "maximum_regression": protocol_gate["field_AP_maximum_loss"],
        "maximum_seam_background_increase": protocol_gate["seam_background_pair_maximum_increase"],
    }
    result = {"pass": False, "reason": "insufficient_eligible_labelled_pairs", "eligible_pair_counts": counts, "limits": limits}
    if min(counts.values()) < limits["minimum_pair_count"]:
        return result
    def value(row, name):
        return row["pairs"][name]["both_positive_fraction"] if name in relevant else row[name]
    values = (*relevant, "ap", "recall_at_background_5pct")
    c = {name: value(central, name) for name in values}
    h = {name: value(hann, name) for name in values}
    if any(v is None or not math.isfinite(v) for v in (*c.values(), *h.values())):
        result["reason"] = "nonfinite_or_missing_endpoint"
        return result
    checks = {
        "seam_ink_gain": h["seam_ink"] >= c["seam_ink"] + protocol_gate["seam_ink_pair_absolute_gain_minimum"],
        "ap_noninferiority": h["ap"] >= c["ap"] - protocol_gate["field_AP_maximum_loss"],
        "field_recall_noninferiority": h["recall_at_background_5pct"] >= c["recall_at_background_5pct"] - protocol_gate["field_recall_maximum_loss"],
        "within_cell_ink_noninferiority": h["within_cell_ink"] >= c["within_cell_ink"] - protocol_gate["within_ink_pair_maximum_loss"],
        "seam_background_pair_limit": h["seam_background"] <= c["seam_background"] + protocol_gate["seam_background_pair_maximum_increase"],
    }
    result.update({"pass": all(checks.values()), "checks": checks,
                   "reason": "all_new_method_conditions_pass" if all(checks.values()) else "new_method_condition_failed",
                   "hann_minus_central": {name: h[name] - c[name] for name in values}})
    return result


def replay_type(torch):
    class ActualLogitReplay(torch.nn.Module):
        patch_size = (64, 256, 256)
        divisor = 200.0

        def __init__(self, field, logits, probabilities):
            super().__init__()
            self.stub = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.field, self.logits, self.probabilities = field, logits, probabilities
            self.calls = 0
            self.input_digest = hashlib.sha256()
            self.support_digest = hashlib.sha256()
            self.support_dtypes = set()

        def forward(self, image, valid):
            require(self.calls < 64, "Released predictor requested too many cached logits")
            y, x = divmod(self.calls, 8)
            raw = np.asarray(self.field[:, 128*y:128*y+256, 128*x:128*x+256])
            expected_image = (raw[::-1].astype(np.float32) / np.float32(200))[None, None]
            actual_image = image.detach().cpu().numpy()
            same_bytes(actual_image, expected_image, "Released predictor reverse-normalized patch")
            xy_valid = raw.any(axis=0)
            # Torch 2.6 preserves uint8 for torch.any(uint8), unlike NumPy's
            # boolean result. Check the official representation, not a falsely
            # assumed bool dtype. The original model converts support to float.
            expected_valid = np.broadcast_to(xy_valid.astype(np.uint8)[None, None, None], (1, 1, 64, 256, 256))
            same_bytes(valid.detach().cpu().numpy(), expected_valid, "Released predictor support tensor")
            self.support_dtypes.add(str(valid.dtype))
            self.input_digest.update(np.ascontiguousarray(actual_image).tobytes())
            self.support_digest.update(np.ascontiguousarray(xy_valid).tobytes())
            saved = torch.from_numpy(np.array(self.logits[self.calls], copy=True))[None, None]
            same_bytes(saved.sigmoid()[0, 0].numpy(), self.probabilities[self.calls], "Saved actual logits/probabilities")
            self.calls += 1
            return saved

    return ActualLogitReplay


def independent_assembly(field, patches, window):
    coordinates = np.arange(1024)
    tile = (coordinates[:, None] // 128) * 8 + coordinates[None, :] // 128
    unmasked = patches[tile, 64 + coordinates[:, None] % 128, 64 + coordinates[None, :] % 128]
    support_fine = field.any(axis=0)[64:1088, 64:1088]
    maps = {"central": unmasked * support_fine}
    numerator = {name: np.zeros((1152, 1152), dtype=np.float32) for name in ("hann", "uniform")}
    denominator = {name: np.zeros((1152, 1152), dtype=np.float32) for name in ("hann", "uniform")}
    for index in range(64):
        ty, tx = divmod(index, 8)
        region = (slice(128*ty, 128*ty+256), slice(128*tx, 128*tx+256))
        patch_support = field[(slice(None), *region)].any(axis=0)
        masked = patches[index] * patch_support
        numerator["hann"][region] += masked * window
        denominator["hann"][region] += window
        numerator["uniform"][region] += masked
        denominator["uniform"][region] += np.float32(1)
    for name in ("hann", "uniform"):
        require((denominator[name] > 0).all(), "Independent assembly has uncovered cells")
        maps[name] = (numerator[name] / denominator[name])[64:1088, 64:1088].copy()
    counts = support_fine.reshape(256, 4, 256, 4).sum(axis=(1, 3), dtype=np.uint8)
    return maps, unmasked, support_fine, counts == 16, counts > 0


def verify_sources(run_dir, result, protocol, donors):
    receipt_path = run_dir / "source_receipt.json"
    require(sha(receipt_path) == result["source_receipt_sha256"], "Acquisition receipt digest mismatch")
    receipt = json_read(receipt_path)
    require(not receipt["failed_assets"], "Completed run contains failed acquisitions")
    records = receipt["completed_assets"]
    by_url = {row["url"]: row for row in records}
    require(len(by_url) == len(records), "Duplicate acquisition objects")
    total = 0
    for row in records:
        url = urlsplit(row["url"])
        require(url.scheme == "https" and not url.query and not url.username and not url.password, "Noncanonical acquisition URL")
        require(row.get("reused_old_cache") is False, "Cold start unexpectedly reused old cache")
        path = run_dir / "source_cache" / hashlib.sha256(row["url"].encode()).hexdigest()
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Retained source bytes changed")
        total += row["bytes"]
    require(total == receipt["new_application_body_bytes"] == result["new_application_body_bytes"], "Source body accounting mismatch")
    require(receipt["requests_including_redirects"] == result["requests_including_redirects"], "Source request accounting mismatch")
    require(total <= protocol["limits"]["new_body_bytes"], "Source body budget exceeded")
    require(result["requests_including_redirects"] <= protocol["limits"]["requests_including_redirects"], "Request budget exceeded")
    expected_assets = [(protocol["fine_checkpoint"]["url"], protocol["fine_checkpoint"]["sha256"], protocol["fine_checkpoint"]["bytes"])]
    for donor in donors:
        expected_assets.extend([
            (donor["source_zarr_url"].rstrip("/") + "/.zarray", donor["source_zarr_metadata_sha256"], None),
            (donor["label_url"], donor["label_sha256"], donor["label_size_bytes"]),
            (donor["geometry_mask_url"], donor["geometry_mask_sha256"], donor["geometry_mask_size_bytes"]),
        ])
    # Restrict the retained acquisition inventory to the two frozen controls;
    # a self-reported zero unread-request count is not an adequate scope audit.
    allowed_urls = {url for url, _, _ in expected_assets}
    donor_map = {donor["scroll"]: donor for donor in donors}
    selection = json_read(ROOT / "firstletters-colab/native_resolution/selection.json")
    for entry in selection["results"]:
        choice = entry["choice"]
        donor = donor_map[choice["scroll"]]
        lower, upper = choice["source_box_zyx_exclusive"]
        chunks = donor["source_zarr_metadata"]["chunks"]
        separator = donor["source_zarr_metadata"].get("dimension_separator", ".")
        axes = [range(a // size, (b - 1) // size + 1) for a, b, size in zip(lower, upper, chunks)]
        count = 0
        for z in axes[0]:
            for y in axes[1]:
                for x in axes[2]:
                    key = separator.join(map(str, (z, y, x)))
                    allowed_urls.add(donor["source_zarr_url"].rstrip("/") + "/" + key)
                    count += 1
        require(count == choice["unique_full_chunks"] == 100, "Frozen source footprint differs")
    require(set(by_url) == allowed_urls, "Acquisition inventory differs from the exact two-control footprint")
    for url, expected_sha, expected_bytes in expected_assets:
        require(url in by_url and by_url[url]["sha256"] == expected_sha, "Pinned source asset absent or changed")
        if expected_bytes is not None:
            require(by_url[url]["bytes"] == expected_bytes, "Pinned source asset size changed")
    return {"retained_assets_verified": len(records), "retained_application_body_bytes": total,
            "exact_two_control_acquisition_scope_verified": True,
            "checkpoint_loaded_for_review": False, "network_requests_by_reviewer": 0}


def verify(args):
    started = time.monotonic()
    run_dir = outside_repository(args.run_dir)
    result = json_read(run_dir / "result.json")
    # This first check prevents inspection of partial or live prediction arrays.
    require(result.get("status") == "complete" and result.get("forwards_completed") == 128, "Run is not complete; do not review partial arrays")
    require(result.get("freeze_reference") == FREEZE and len(result.get("fields", [])) == 2, "Unexpected freeze or field count")
    require(result.get("training_steps") == 0 and result.get("unread_CT_requests") == 0 and result.get("old_native_gate_changed") is False,
            "Experiment scope differs from freeze")
    protocol = json_read(HERE / "protocol.json")
    for name in FROZEN:
        path = HERE / name
        published = subprocess.check_output(["git", "show", f"{FREEZE}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(path.read_bytes() == published, "Frozen experiment file changed: " + name)
    for relative, expected in protocol["dependency_sha256"].items():
        require(sha(ROOT / relative) == expected, "Pinned experiment dependency changed")
    require(sha(HERE / "protocol.json") == result["protocol_sha256"] and sha(HERE / "run.py") == result["runner_sha256"], "Run code/protocol digest mismatch")
    require(protocol["pins"]["prior_receipt_sha256"] == [], "Unexpected historical cache permission")
    identities = protocol["historical_field_identity"]
    require([row["scroll"] for row in result["fields"]] == ["PHerc0841", "PHerc0814"], "Field order/identity changed")
    pre = json_read(run_dir / "pre_inference_inputs.json")
    require(pre["historical_field_identity"] == identities and pre["all_complete_field_hashes_match"] is True and pre["forwards_before_identity_check"] == 0,
            "Input identity was not recorded before forwards")
    manifest = json_read(ROOT / "firstletters-colab/transversal_lab/hecate_donor_manifest.json")
    donors = [d for d in manifest["donors"] if d["scroll"] in identities]
    sources = verify_sources(run_dir, result, protocol, donors)
    native = load_module(ROOT / "firstletters-colab/native_resolution/run.py", "review_native_rank_metrics")
    hecate = load_module(ROOT / "firstletters-colab/ink/third_party/hecate/hecate.py", "review_untouched_hecate")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    replay_class = replay_type(torch)
    one_dimensional = torch.hann_window(256, periodic=False, dtype=torch.float32)
    window_tensor = one_dimensional[:, None] * one_dimensional[None, :]
    window = (window_tensor / window_tensor.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
    require(array_sha(window) == result["window_array_sha256"], "Released window digest mismatch")
    comparison = ScalarComparison()
    reviewed = []
    array_count = 0
    prior = json_read(ROOT / "firstletters-colab/native_resolution/evidence/result.json")
    prior_fields = {row["scroll"]: row for row in prior["fields"]}
    for row in result["fields"]:
        scroll = row["scroll"]
        folder = run_dir / scroll
        require(json_read(folder / "result.json") == row, "Per-field receipt differs from aggregate")
        recorded = row["output_sha256"]
        require(set(recorded) == {path.name for path in folder.glob("*.npy")}, "Retained array inventory differs")
        for name, expected in recorded.items():
            require(Path(name).name == name and sha(folder / name) == expected, "Retained array digest mismatch")
            array_count += 1
        for name, key in (("fine_context_uint8.npy", "fine_context_sha256"), ("labels.npy", "labels_sha256")):
            require(recorded[name] == identities[scroll][key] == row[key], "Historical field identity mismatch")
        require(row["patch_probabilities_sha256"] == recorded["patch_probabilities.npy"], "Patch probability digest mismatch")
        field = checked_array(folder / "fine_context_uint8.npy", np.uint8, (64, 1152, 1152))
        labels = checked_array(folder / "labels.npy", np.uint8, (256, 256))
        patches = checked_array(folder / "patch_probabilities.npy", np.float32, (64, 256, 256))
        logits = checked_array(folder / "patch_logits.npy", np.float32, (64, 256, 256))
        require(((patches >= 0) & (patches <= 1)).all(), "Cached probability outside [0,1]")
        maps, unmasked, fine_support, support, historical_support = independent_assembly(field, patches, window)
        same_bytes(checked_array(folder / "common_support.npy", bool, (256, 256)), support, "Fully supported common cells")
        same_bytes(checked_array(folder / "historical_unmasked_central.npy", np.float32, (1024, 1024)), unmasked, "Historical central reconstruction")
        comparison.check(int(support.sum()), row["fully_supported_common_cells"], scroll + "/full support")
        comparison.check(int(historical_support.sum()), row["any_supported_common_cells"], scroll + "/any support")
        comparison.check(int((~fine_support).sum()), row["fine_support_holes"], scroll + "/support holes")
        checked_metrics = {"float": {}, "uint8": {}}
        for arm in ARMS:
            fine = checked_array(folder / (arm + "_fine.npy"), np.float32, (1024, 1024))
            same_bytes(fine, maps[arm], scroll + "/" + arm + "/assembly")
            export = np.rint(np.clip(fine, 0, 1) * np.float32(255)).astype(np.uint8)
            same_bytes(checked_array(folder / (arm + "_uint8.npy"), np.uint8, (1024, 1024)), export, scroll + "/" + arm + "/export")
            common = fine.reshape(256, 4, 256, 4).sum(axis=(1, 3), dtype=np.float64) / 16.0
            integer_sums = export.reshape(256, 4, 256, 4).sum(axis=(1, 3), dtype=np.uint32)
            representations = {"float": common.astype(np.float32), "uint8": (integer_sums.astype(np.float64) / 16.0 / 255.0).astype(np.float32)}
            for representation, values in representations.items():
                saved = checked_array(folder / f"{arm}_{representation}_common.npy", np.float32, (256, 256))
                same_bytes(saved, values, scroll + "/" + arm + "/" + representation + "/common reduction")
                calculated = independent_metrics(values, labels, support, native.measure)
                comparison.check(calculated, row["metrics"][representation][arm], scroll + "/" + arm + "/" + representation + "/metrics")
                checked_metrics[representation][arm] = calculated
        historical_common = (unmasked.reshape(256, 4, 256, 4).sum(axis=(1, 3), dtype=np.float64) / 16).astype(np.float32)
        historical = native.measure(historical_common, labels, historical_support)
        comparison.check(historical, row["historical_unmasked_central_metric"], scroll + "/historical metric")
        decisions = {name: independent_decision(checked_metrics[name]["central"], checked_metrics[name]["hann"], protocol["gate"])
                     for name in ("float", "uint8")}
        comparison.check({"pass": all(value["pass"] for value in decisions.values()), "representations": decisions}, row["gate"], scroll + "/gate")
        replay = replay_class(field, logits, patches).eval()
        official = np.empty((1152, 1152), dtype=np.uint8)
        with contextlib.redirect_stdout(io.StringIO()):
            hecate.predict(replay, field, official, reverse=True, stride=None, batch_size=1, precision="fp32")
        require(replay.calls == 64, "Released predictor did not consume exactly 64 cached logits")
        saved_hann = checked_array(folder / "hann_uint8.npy", np.uint8, (1024, 1024))
        same_bytes(official[64:1088, 64:1088], saved_hann, scroll + "/released predictor uint8 crop")
        old = prior_fields[scroll]["metrics"]["field"]["A_native_fine"]
        reviewed.append({
            "scroll": scroll, "retained_arrays_verified": len(recorded),
            "official_predictor_cached_logit_calls": replay.calls,
            "reverse_normalized_input_and_support_checks": replay.calls,
            "official_predictor_support_dtypes": sorted(replay.support_dtypes),
            "historical_direct_forward_support_dtype": "torch.bool; identical 0/1 values, converted to float inside Hecate",
            "actual_logits_sigmoid_match_cached_probabilities": True,
            "official_Hann_uint8_core_bytewise_equal": True,
            "official_Hann_core_array_sha256": array_sha(official[64:1088, 64:1088]),
            "all_replayed_normalized_inputs_sha256": replay.input_digest.hexdigest(),
            "all_replayed_XY_support_sha256": replay.support_digest.hexdigest(),
            "fully_supported_common_cells": int(support.sum()),
            "representations": {name: {"pass": decisions[name]["pass"], "checks": decisions[name].get("checks", {}),
                "hann_minus_central": decisions[name].get("hann_minus_central", {})} for name in decisions},
            "new_method_gate_pass": row["gate"]["pass"],
            "historical_unmasked_central_metric_deltas": {key: historical[key] - old[key]
                for key in ("ap", "auc", "recall_at_background_5pct")},
        })
        del field, patches, logits, maps, replay, official
    gate = all(row["new_method_gate_pass"] for row in reviewed)
    comparison.check(gate, result["numeric_gate_pass"], "Aggregate decision")
    require(result["model_buffers_unchanged"] is True, "Producing run did not report unchanged model buffers")
    return {
        "schema": "scrollprize.hecate-blending.independent-review.v1", "status": "pass",
        "freeze_reference": FREEZE, "reviewer_sha256": sha(__file__),
        "run_result_sha256": sha(run_dir / "result.json"), "protocol_sha256": sha(HERE / "protocol.json"),
        "frozen_experiment_files_verified": len(FROZEN), "pinned_dependencies_verified": len(protocol["dependency_sha256"]),
        **sources, "retained_arrays_verified": array_count,
        "numeric_and_exact_scalar_comparisons": comparison.count,
        "maximum_scalar_float_difference": comparison.maximum_float_difference,
        "real_model_forwards_by_reviewer": 0, "cached_official_predictor_calls": 128,
        "metrics_module_imported": False, "old_native_resolution_gate_changed": False,
        "new_method_gate_pass": gate, "fields": reviewed, "review_wall_seconds": time.monotonic() - started,
        "limitations": [
            "Cached actual logits are checked against saved probabilities, but the trained network is not independently rerun.",
            "Historical complete input hashes bind CT reconstruction; this checker does not independently repeat CT regridding.",
            "Source hashes verify retained acquisition objects, not an immutable atomic snapshot of the mutable bucket.",
            "Thresholds and dependent pixel pairs are descriptive development metrics; no new letters, independent glyph truth or unread-scroll generalisation follows.",
            "The old native-detail advancement decision remains failed; this review cannot authorize unread-region work.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = outside_repository(args.output or args.run_dir / "independent_review.json")
    require(not output.exists(), "Review output already exists; preserve earlier reviews")
    review = verify(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    require(not partial.exists(), "Partial review output already exists")
    partial.write_text(json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(output)
    print(json.dumps({"status": review["status"], "retained_arrays_verified": review["retained_arrays_verified"],
                      "cached_official_predictor_calls": 128, "real_model_forwards": 0,
                      "new_method_gate_pass": review["new_method_gate_pass"], "wall_seconds": review["review_wall_seconds"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewFailure as error:
        print(json.dumps({"status": "review_failed", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(1)
