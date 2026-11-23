#!/usr/bin/env python3
"""Independent review of saved T5 outputs; no acquisition or feature extraction.

This post-freeze reviewer imports none of the experiment's numerical modules.
Observed AUC uses scipy.stats.rankdata. All saved spatial-null AUCs are checked
by independently sorting negative scores and counting positive/negative pairs
with searchsorted. Paired block bootstrap is independently reconstructed from
20 x 20 block-pair Mann-Whitney U matrices, never a pixel-pair matrix.

Only a terminal result (complete, unassessable, or stopped) is reviewed. Cache
verification hashes opaque acquired bodies; it does not decode CT or regenerate
feature maps. Original feature values, CT-to-reference registration, and the
legacy peak/profile masks cannot be independently established from these saved
outputs and are explicitly outside this review's numerical claim.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import rankdata


FREEZE_COMMIT = "b5079245a4e0965a36b6f99ea50921f5c9bc4f6a"
RELATIVE_ROOT = "firstletters-colab/t5_replication/"
FEATURES = ("raw_ct_surface", "oct_hilbert_env_peak", "wst_s1_scale_slope")
SIGNS = (1, 1, -1)
SHAPE = (512, 640)
NUMERICAL_TOLERANCE = 1e-12


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _label_masks(reference, valid):
    return valid & (reference >= 179), valid & (reference >= 1) & (reference <= 64)


def _rank_auc(score, positive, negative):
    """Independent observed AUC: SciPy average ranks on included pixels only."""
    p = np.asarray(score[positive], dtype=np.float64)
    n = np.asarray(score[negative], dtype=np.float64)
    if not len(p) or not len(n):
        return float("nan")
    ranks = rankdata(np.concatenate((p, n)), method="average")
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def _pair_u(positive_scores, sorted_negative_scores):
    """Pair-count U, with exact half credit for ties, without a pair matrix."""
    left = np.searchsorted(sorted_negative_scores, positive_scores, side="left")
    right = np.searchsorted(sorted_negative_scores, positive_scores, side="right")
    return float(np.sum(left + right, dtype=np.int64) / 2)


def _pair_auc(score, positive, negative):
    p = np.asarray(score[positive], dtype=np.float64)
    n = np.sort(np.asarray(score[negative], dtype=np.float64))
    return _pair_u(p, n) / (len(p) * len(n)) if len(p) and len(n) else float("nan")


def _block_pair_bootstrap(scores, reference, valid, draws, block_size):
    """Return all paired draws using one small block-pair U matrix per feature."""
    height, width = valid.shape
    rows, columns = np.indices(valid.shape)
    blocks = (rows // block_size) * (width // block_size) + columns // block_size
    block_count = (height // block_size) * (width // block_size)
    positive, negative = _label_masks(reference, valid)
    positive_masks = [positive & (blocks == b) for b in range(block_count)]
    negative_masks = [negative & (blocks == b) for b in range(block_count)]
    n_positive = np.array([mask.sum() for mask in positive_masks], dtype=np.int64)
    n_negative = np.array([mask.sum() for mask in negative_masks], dtype=np.int64)
    multiplicity = np.stack([np.bincount(draw, minlength=block_count) for draw in draws])
    counts = np.column_stack((multiplicity @ n_positive, multiplicity @ n_negative))
    denominators = counts[:, 0].astype(np.float64) * counts[:, 1]
    auc = np.full((len(draws), len(scores)), np.nan)
    matrices = []
    for index, score in enumerate(scores):
        p = [np.asarray(score[mask], dtype=np.float64) for mask in positive_masks]
        n = [np.sort(np.asarray(score[mask], dtype=np.float64)) for mask in negative_masks]
        matrix = np.array([[_pair_u(pa, nb) for nb in n] for pa in p], dtype=np.float64)
        matrices.append(matrix)
        numerator = np.einsum("bi,ij,bj->b", multiplicity, matrix, multiplicity)
        np.divide(numerator, denominators, out=auc[:, index], where=denominators > 0)
    return auc, counts, matrices


def _rectangle_gap(a, b):
    y_gap = max(0, int(a[0]) - int(b[1]), int(b[0]) - int(a[1]))
    x_gap = max(0, int(a[2]) - int(b[3]), int(b[2]) - int(a[3]))
    return y_gap * y_gap + x_gap * x_gap


class Review:
    def __init__(self):
        self.checks = []
        self.details = {}

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "pass": bool(passed), **_json_safe(details)})

    def same(self, name, observed, expected):
        self.check(name, observed == expected)

    def close(self, name, observed, expected, tolerance=NUMERICAL_TOLERANCE):
        actual = np.asarray(observed, dtype=float)
        target = np.asarray(expected, dtype=float)
        shape_ok = actual.shape == target.shape
        passed = shape_ok and np.allclose(actual, target, atol=tolerance, rtol=0, equal_nan=True)
        finite = np.isfinite(actual) & np.isfinite(target) if shape_ok else np.array([], bool)
        error = float(np.max(np.abs(actual[finite] - target[finite]))) if finite.any() else 0.0
        self.check(name, passed, max_absolute_error=error, tolerance=tolerance)


def _verify_freeze(review, result):
    repo = Path(__file__).resolve().parents[2]
    review.same("freeze_commit", result["freeze_commit"], FREEZE_COMMIT)

    def blob(name):
        return subprocess.check_output(["git", "show", f"{FREEZE_COMMIT}:{RELATIVE_ROOT}{name}"], cwd=repo)

    raw = blob("protocol.json")
    protocol = json.loads(raw)
    review.same("protocol_sha256", hashlib.sha256(raw).hexdigest(), result["protocol_sha256"])
    for name, expected in protocol["frozen_files_sha256"].items():
        review.same("frozen_git_sha256:" + name, hashlib.sha256(blob(name)).hexdigest(), expected)
        review.same("frozen_local_sha256:" + name, _sha(repo / RELATIVE_ROOT / name), expected)
    review.same("fixed_feature_polarities", protocol["feature_polarities"], dict(zip(FEATURES, SIGNS)))
    return protocol, json.loads(blob("source_manifest.json"))


def _verify_artifacts(review, directory, result):
    names = set()
    for item in result["artifacts"]:
        name = item["name"]
        safe = Path(name).name == name and name not in ("", ".", "..")
        review.check("artifact_safe_unique_name:" + str(name), safe and name not in names)
        if not safe:
            raise ValueError("Unsafe artifact name")
        names.add(name)
        path = directory / name
        exists = path.is_file()
        review.check("artifact_exists:" + name, exists)
        if exists:
            review.same("artifact_bytes:" + name, path.stat().st_size, item["bytes"])
            review.same("artifact_sha256:" + name, _sha(path), item["sha256"])
    review.details["recorded_artifact_count"] = len(names)
    return names


def _verify_acquisition(review, result, protocol, manifest, cache):
    acquisition = result["acquisition"]
    events = acquisition["network_events"]
    preflight = acquisition["metadata_preflight"]
    review.same("acquisition_request_accounting", acquisition["total_requests_including_preflight"],
                len(events) + preflight["requests"])
    review.same("acquisition_body_byte_accounting", acquisition["total_response_body_bytes_including_preflight"],
                sum(event["body_bytes"] for event in events) + preflight["response_body_bytes"])
    review.check("acquisition_byte_limit", acquisition["total_response_body_bytes_including_preflight"] <=
                 protocol["limits"]["total_response_body_bytes_including_preflight"])
    review.check("acquisition_request_limit", acquisition["total_requests_including_preflight"] <=
                 protocol["limits"]["total_requests_including_preflight"])
    urls = [event["url"] for event in events]
    review.check("no_repeated_network_asset_requests", len(urls) == len(set(urls)))
    assets = acquisition["whole_asset_receipts"]
    ct_prefix = manifest["zarr"]["array_url"] + "/0/"
    ct_assets = [asset for asset in assets if asset["url"].startswith(ct_prefix)]
    review.check("CT_chunk_limit", len(ct_assets) <= protocol["limits"]["maximum_unique_CT_chunks"])
    authorized_chunks = set()
    for y0, y1, x0, x1 in acquisition["authorized_regions_y0_y1_x0_x1"]:
        for cy in range(y0 // 128, y1 // 128):
            for cx in range(x0 // 128, x1 // 128):
                authorized_chunks.add(f'{manifest["zarr"]["array_url"]}/0/{cy}/{cx}')
    acquired_chunks = {asset["url"] for asset in ct_assets}
    review.check("no_CT_acquisition_outside_selected_regions", acquired_chunks <= authorized_chunks)
    allowed_urls = authorized_chunks | {manifest["reference"]["url"], manifest["zarr"]["metadata_url"]}
    review.check("all_requested_assets_inside_public_control_scope", set(urls) <= allowed_urls)
    if result["status"] in ("complete", "unassessable"):
        review.check("terminal_full_acquisition_contains_all_authorized_chunks", acquired_chunks == authorized_chunks)
        review.check("terminal_full_acquisition_has_80_CT_chunks", len(acquired_chunks) == 80)
        review.check("all_network_events_completed", all(event["complete"] and event.get("status") == 200 for event in events))
    for index, asset in enumerate(assets):
        tag = "acquired_asset_" + str(index)
        review.same(tag + ":protocol", asset["protocol_sha256"], result["protocol_sha256"])
        if asset["url"] == manifest["reference"]["url"]:
            review.same(tag + ":reference_etag", asset["etag"], manifest["reference"]["etag"])
            review.same(tag + ":reference_bytes", asset["body_bytes"], manifest["reference"]["content_length"])
        if asset["url"] == manifest["zarr"]["metadata_url"]:
            review.same(tag + ":metadata_sha256", asset["sha256"], manifest["zarr"]["sha256"])
        if asset["url"].startswith(ct_prefix):
            review.same(tag + ":CT_chunk_bytes", asset["body_bytes"], manifest["zarr"]["full_chunk_bytes"])
        if cache is not None:
            key = hashlib.sha256(asset["url"].encode()).hexdigest()
            body = cache / (key + ".bin")
            cached_receipt = cache / (key + ".json")
            found = body.is_file() and cached_receipt.is_file()
            review.check(tag + ":cache_pair_exists", found)
            if not found:
                continue
            receipt = json.loads(cached_receipt.read_text())
            review.same(tag + ":cache_bytes", body.stat().st_size, asset["body_bytes"])
            review.same(tag + ":cache_sha256", _sha(body), asset["sha256"])
            for field in ("url", "body_bytes", "etag", "sha256", "protocol_sha256"):
                review.same(tag + ":cache_receipt_" + field, receipt[field], asset[field])
    review.details["acquisition"] = {
        "asset_count": len(assets), "CT_chunk_count": len(ct_assets),
        "opaque_cache_body_hashes_verified": cache is not None,
        "CT_arrays_decoded_or_features_recomputed": False,
    }


def _verify_selection(review, selection, protocol):
    params = protocol["selection"]
    eligible = selection["eligible_candidates"]
    expected = []
    anchors = [selection["development_bounds_y0_y1_x0_x1"]]
    for _ in range(params["count"]):
        choices = []
        for item in eligible:
            bounds = item["bounds_y0_y1_x0_x1"]
            gap = min(_rectangle_gap(bounds, anchor) for anchor in anchors)
            if gap >= params["min_gap_px"] ** 2 and bounds not in anchors:
                choices.append((-gap, bounds[0], bounds[2], item))
        if not choices:
            break
        gap, _, _, chosen = sorted(choices, key=lambda value: value[:3])[0]
        expected.append({**chosen, "min_gap_squared_px_at_selection": -gap})
        anchors.append(chosen["bounds_y0_y1_x0_x1"])
    review.same("selection_greedy_rule_from_saved_eligible_list", selection["selected"], expected)
    review.same("selection_status", selection["status"], "complete" if len(expected) == params["count"] else "unassessable")
    for index, item in enumerate(eligible):
        bounds = item["bounds_y0_y1_x0_x1"]
        valid = ((bounds[1] - bounds[0], bounds[3] - bounds[2]) == tuple(params["shape"])
                 and all(value % params["alignment"] == 0 for value in bounds))
        if not valid:
            review.check("selection_candidate_geometry", False, candidate_index=index)
    review.details["selection"] = {
        "greedy_choice_recomputed_from_saved_eligible_list": True,
        "full_reference_map_and_candidate_list_completeness_recomputed": False,
        "selected_crop_label_counts_recomputed": True,
    }


def _load_case(review, directory, case, protocol, selection):
    name = case["name"]
    with np.load(directory / (name + ".npz"), allow_pickle=False) as saved:
        review.same(name + ":npz_keys", set(saved.files), set(FEATURES) | {"reference", "common_valid", "eval_valid"})
        maps = [saved[feature].copy() for feature in FEATURES]
        reference, common, valid = (saved[key].copy() for key in ("reference", "common_valid", "eval_valid"))
    shape_ok = all(array.shape == SHAPE for array in (*maps, reference, common, valid))
    review.check(name + ":all_shapes", shape_ok)
    if not shape_ok:
        raise ValueError("Invalid saved array shape")
    review.check(name + ":array_dtypes", all(array.dtype == np.float32 for array in maps)
                 and reference.dtype == np.uint8 and common.dtype == bool and valid.dtype == bool)
    finite = np.logical_and.reduce([np.isfinite(array) for array in maps])
    review.check(name + ":common_support_is_finite_in_all_features", not (common & ~finite).any())
    development = name == "development_replay"
    guard = protocol["replay"]["border_guard_px"] if development else protocol["selection"]["border_guard_px"]
    expected_mask = common.copy()
    expected_mask[:guard, :] = False
    expected_mask[-guard:, :] = False
    expected_mask[:, :guard] = False
    expected_mask[:, -guard:] = False
    review.check(name + ":exact_border_mask", np.array_equal(expected_mask, valid))
    scores = [array.astype(np.float64) * sign for array, sign in zip(maps, SIGNS)]
    positive, negative = _label_masks(reference, valid)
    auc = np.array([_rank_auc(score, positive, negative) for score in scores])
    observed = case["observed"]
    counts = [int(positive.sum()), int(negative.sum())]
    review.same(name + ":observed_class_counts", [observed["ink_count"], observed["blank_count"]], counts)
    review.close(name + ":scipy_rank_AUC", auc, [observed["auc"][feature] for feature in FEATURES])
    support = case["support"]
    diagnostics = case["diagnostics"]["support"]
    n_sheet = diagnostics["legacy_sheet_valid_columns"]
    whole_fraction = float(common.sum() / n_sheet) if n_sheet else 0.0
    potential = (SHAPE[0] - 2 * guard) * (SHAPE[1] - 2 * guard)
    guarded_fraction = float(valid.sum() / potential)
    review.same(name + ":support_guard", support["guard_px"], guard)
    review.same(name + ":potential_core_pixels", support["potential_guarded_pixels"], potential)
    review.same(name + ":common_mask_count", int(common.sum()), diagnostics["common_valid_columns"])
    review.same(name + ":guarded_mask_count", int(valid.sum()), support["guarded_common_valid_pixels"])
    review.close(name + ":guarded_fraction", guarded_fraction, support["guarded_common_valid_fraction"])
    review.close(name + ":whole_common_fraction_arithmetic", whole_fraction, support["whole_common_valid_fraction_of_sheet"])
    threshold = protocol["support_gates"]
    class_pass = counts[0] >= threshold["minimum_high_count_after_guard_and_support"] and counts[1] >= threshold["minimum_low_count_after_guard_and_support"]
    support_pass = (guarded_fraction >= threshold["minimum_guarded_common_valid_fraction"]
                    and whole_fraction >= threshold["minimum_whole_common_valid_fraction_of_legacy_sheet"] and class_pass)
    review.same(name + ":class_gate", support["class_counts_at_least_10000_each"], class_pass)
    review.same(name + ":support_gate", support["pass"], support_pass)
    review.same(name + ":fixed_polarities", case["diagnostics"]["fixed_polarities"], dict(zip(FEATURES, SIGNS)))
    if development:
        review.same("development_bounds", case["bounds_y0_y1_x0_x1"], protocol["replay"]["bounds_y0_y1_x0_x1"])
    elif selection is not None:
        item = next(item for item in selection["selected"] if item["bounds_y0_y1_x0_x1"] == case["bounds_y0_y1_x0_x1"])
        core = reference[guard:-guard, guard:-guard]
        for field, array in (("full_counts_high_low_nonzero", reference), ("core_counts_high_low_nonzero", core)):
            recomputed = [int((array >= 179).sum()), int(((array >= 1) & (array <= 64)).sum()), int((array > 0).sum())]
            review.same(name + ":selection_" + field, recomputed, item[field])
    return {"name": name, "scores": scores, "reference": reference, "valid": valid,
            "common": common, "auc": auc, "counts": counts, "support_pass": support_pass}


def _verify_statistics(review, statistics, cases, protocol):
    config = protocol["statistics"]
    n_null, n_boot = config["null_replicates"], config["bootstrap_replicates"]
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(config["seed"]).spawn(6)]
    review.same("statistical_case_order", statistics["case_names"], [case["name"] for case in cases])
    review.same("statistical_feature_order", statistics["feature_names"], list(FEATURES))
    observed = np.stack([case["auc"] for case in cases])
    for index, report in enumerate(statistics["observed"]["cases"]):
        review.close("statistics_observed:" + cases[index]["name"],
                     [report["auc"][feature] for feature in FEATURES], observed[index])
        review.same("statistics_counts:" + cases[index]["name"],
                    [report["ink_count"], report["blank_count"]], cases[index]["counts"])
    recomputed_null = np.full((n_null, 3, 3), np.nan)
    null_counts = np.zeros((n_null, 3, 2), dtype=np.int64)
    recomputed_boot = np.full((n_boot, 3, 3), np.nan)
    boot_counts = np.zeros((n_boot, 3, 2), dtype=np.int64)
    block_matrices = {}
    for index, case in enumerate(cases):
        flat = streams[index].choice(np.prod(SHAPE) - 1, size=n_null, replace=False) + 1
        offsets = np.column_stack((flat // SHAPE[1], flat % SHAPE[1]))
        review.check("null_offset_seed_reproduction:" + case["name"],
                     np.array_equal(offsets, statistics["null"]["offsets_yx"][index]))
        for draw, (dy, dx) in enumerate(offsets):
            shifted = np.roll(case["reference"], (int(dy), int(dx)), axis=(0, 1))
            positive, negative = _label_masks(shifted, case["valid"])
            null_counts[draw, index] = [positive.sum(), negative.sum()]
            recomputed_null[draw, index] = [_pair_auc(score, positive, negative) for score in case["scores"]]
        draws = streams[index + 3].integers(0, 20, size=(n_boot, 20))
        review.check("bootstrap_draw_seed_reproduction:" + case["name"],
                     np.array_equal(draws, statistics["bootstrap"]["resampled_block_indices"][index]))
        boot, counts, matrices = _block_pair_bootstrap(case["scores"], case["reference"], case["valid"], draws, config["block_size"])
        recomputed_boot[:, index], boot_counts[:, index] = boot, counts
        block_matrices[case["name"]] = dict(zip(FEATURES, matrices))
        # The same U matrices independently reconstruct the unweighted AUC.
        denominator = case["counts"][0] * case["counts"][1]
        matrix_auc = [matrix.sum() / denominator if denominator else np.nan for matrix in matrices]
        review.close("observed_block_pair_U_identity:" + case["name"], matrix_auc, case["auc"])

    review.close("all_null_pixel_pair_AUC", recomputed_null, statistics["null"]["case_auc"])
    review.check("all_null_class_counts", np.array_equal(null_counts, statistics["null"]["class_counts_ink_blank"]))
    review.close("all_bootstrap_block_pair_AUC", recomputed_boot, statistics["bootstrap"]["case_auc"])
    review.check("all_bootstrap_class_counts", np.array_equal(boot_counts, statistics["bootstrap"]["class_counts_ink_blank"]))
    observed_macro = observed.mean(axis=0)
    null_macro = recomputed_null.mean(axis=1)
    boot_macro = recomputed_boot.mean(axis=1)
    delta = observed_macro[1:] - observed_macro[0]
    boot_delta = boot_macro[:, 1:] - boot_macro[:, :1]
    null_max = np.max(null_macro[:, 1:], axis=1)
    review.close("observed_macro_AUC", observed_macro, [statistics["observed"]["macro_auc"][feature] for feature in FEATURES])
    review.close("observed_macro_delta", delta, [statistics["observed"]["macro_delta_vs_raw"][feature] for feature in FEATURES[1:]])
    review.close("null_macro_AUC", null_macro, statistics["null"]["macro_auc"])
    review.close("null_family_max_AUC", null_max, statistics["null"]["family_max_auc"])
    review.close("bootstrap_macro_AUC", boot_macro, statistics["bootstrap"]["macro_auc"])
    review.close("bootstrap_paired_macro_delta", boot_delta, statistics["bootstrap"]["macro_delta_vs_raw"])
    failures = []
    for index, case in enumerate(cases):
        if min(case["counts"]) == 0:
            failures.append({"stage": "observed", "case": case["name"], "class_counts_ink_blank": case["counts"]})
    for stage, counts in (("null", null_counts), ("bootstrap", boot_counts)):
        for draw, index in np.argwhere((counts == 0).any(axis=2)):
            failures.append({"stage": stage, "case": cases[index]["name"], "replicate": int(draw),
                             "class_counts_ink_blank": counts[draw, index].tolist()})
    review.same("statistical_failures", statistics["failures"], failures)
    review.same("statistical_assessability", statistics["status"], "unassessable" if failures else "assessable")
    gates, ranks, intervals = {}, {}, {}
    for index, feature in enumerate(FEATURES[1:]):
        rank = float((1 + np.count_nonzero(null_max >= observed_macro[index + 1])) / (n_null + 1)) if np.isfinite(null_max).all() and np.isfinite(observed_macro[index + 1]) else np.nan
        interval = np.quantile(boot_delta[:, index], [.025, .975], method="linear") if np.isfinite(boot_delta[:, index]).all() else np.array([np.nan, np.nan])
        review.close("family_max_spatial_rank:" + feature, rank, statistics["null"]["spatial_rank"][feature])
        review.close("paired_block_interval:" + feature, interval, statistics["bootstrap"]["percentile_interval"][feature])
        thresholds = protocol["replication_gates_per_feature"]
        checks = {
            "each_case_auc_ge_0_70": bool((observed[:, index + 1] >= thresholds["every_heldout_field_auc_at_least"]).all()),
            "macro_delta_ge_0_05": bool(delta[index] >= thresholds["macro_auc_improvement_over_raw_at_least"]),
            "bootstrap_lower_gt_0": bool(interval[0] > thresholds["paired_block_lower_interval_strictly_above"]),
            "spatial_rank_le_0_05": bool(rank <= thresholds["family_max_spatial_rank_at_most"]),
            "all_assessable": not failures,
        }
        gates[feature] = {**checks, "pass": all(checks.values())}
        review.same("replication_gate:" + feature, gates[feature], statistics["gates"][feature])
        ranks[feature], intervals[feature] = rank, interval
    review.details["statistics"] = {
        "observed_macro_auc": dict(zip(FEATURES, observed_macro)),
        "macro_delta_vs_raw": dict(zip(FEATURES[1:], delta)),
        "spatial_rank": ranks, "paired_block_interval": intervals, "gates": gates,
        "recomputed_null_case_feature_AUC_count": int(recomputed_null.size),
        "recomputed_bootstrap_case_feature_AUC_count": int(recomputed_boot.size),
        "block_pair_U_matrices": block_matrices,
        "empty_class_failures": failures,
    }
    return any(gate["pass"] for gate in gates.values())


def review_result(directory, cache=None):
    directory = Path(directory).resolve()
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if result["status"] not in ("complete", "unassessable", "stopped"):
        raise RuntimeError("Result is not terminal; no saved arrays were opened")
    start = time.monotonic()
    review = Review()
    report = {
        "schema": "t5-spatial-replication.independent-review.v1",
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_sha256": _sha(__file__), "input_result_sha256": _sha(result_path),
        "freeze_commit": FREEZE_COMMIT, "protocol_sha256": result["protocol_sha256"],
        "experiment_status": result["status"], "experiment_failure_retained": result.get("failure"),
        "independence": {
            "experiment_numerical_modules_imported": False,
            "network_requests": 0, "CT_decoded": False, "feature_extraction_rerun": False,
            "observed_auc": "SciPy rankdata on signed included scores",
            "null_auc": "All draws; independently sorted negative scores plus searchsorted pair counts",
            "bootstrap_auc": "All draws; independent 20x20 block-pair U matrices and paired multiplicities",
            "source_cache": "Opaque body byte lengths and SHA-256 only",
        },
        "limitations": [
            "This tests numerical integrity of saved outputs, not independently identified ink, text, or a reading.",
            "Original CT-to-feature arithmetic is not rerun. The earlier generated-array parity test is a separate check.",
            "Legacy sheet masks and full-depth profile bounds are not stored separately; their reported counts are checked arithmetically, not rederived from CT.",
            "Recorded source-stack array hashes are not reconstructed; acquired chunk bodies are verified as opaque bytes when a cache is supplied.",
            "Full-reference candidate-list completeness and source/reference physical registration are not independently recomputed.",
            "Spatial ranks remain misalignment diagnostics and block intervals remain descriptive within-field uncertainty.",
        ],
    }
    try:
        protocol, manifest = _verify_freeze(review, result)
        artifact_names = _verify_artifacts(review, directory, result)
        _verify_acquisition(review, result, protocol, manifest, Path(cache).resolve() if cache else None)
        selection = None
        if "selection.json" in artifact_names:
            selection = json.loads((directory / "selection.json").read_text())
            _verify_selection(review, selection, protocol)
        cases = [_load_case(review, directory, case, protocol, selection) for case in result["cases"]]
        review.details["observed_cases"] = [{"name": case["name"], "counts_ink_blank": case["counts"],
                                              "auc": dict(zip(FEATURES, case["auc"])), "support_pass": case["support_pass"]}
                                             for case in cases]
        development = next((case for case in cases if case["name"] == "development_replay"), None)
        replay_pass = False
        if development is not None and "development_replay" in result:
            replay = result["development_replay"]
            expected = protocol["replay"]
            errors = np.abs(development["auc"] - np.array([expected["expected_auc"][feature] for feature in FEATURES]))
            replay_pass = bool(development["common"].all() and development["counts"] == expected["expected_class_counts"]
                               and np.isfinite(errors).all() and (errors <= expected["auc_tolerance"]).all())
            review.same("development_replay_gate", replay_pass, replay["pass"])
            review.close("development_replay_AUC_errors", errors, [replay["absolute_auc_errors"][feature] for feature in FEATURES])
        heldout = [case for case in cases if case["name"] != "development_replay"]
        final_gate = False
        if "statistics.json" in artifact_names:
            review.same("heldout_case_count", len(heldout), 3)
            if len(heldout) != 3:
                raise ValueError("Statistics require exactly three heldout cases")
            statistics = json.loads((directory / "statistics.json").read_text())
            statistical_pass = _verify_statistics(review, statistics, heldout, protocol)
            for field in ("status", "observed", "gates", "failures"):
                review.same("result_statistics_summary:" + field, result["statistics_summary"][field], statistics[field])
            review.same("result_statistics_summary:spatial_rank", result["statistics_summary"]["spatial_rank"], statistics["null"]["spatial_rank"])
            review.same("result_statistics_summary:paired_block_interval", result["statistics_summary"]["paired_block_interval"], statistics["bootstrap"]["percentile_interval"])
            final_gate = bool(replay_pass and all(case["support_pass"] for case in heldout) and statistical_pass
                              and result["status"] != "stopped")
        review.same("whole_experiment_gate", final_gate, result["any_feature_passes_replication_gate"])
        if result["status"] == "complete":
            review.check("complete_run_all_four_cases", len(cases) == 4 and len(heldout) == 3)
            review.check("complete_run_support_and_replay", replay_pass and all(case["support_pass"] for case in heldout))
            review.check("complete_run_statistics_present", "statistics.json" in artifact_names)
            review.check("complete_run_not_acquisition_failed", not result["acquisition"]["failed"])
            review.check("complete_run_elapsed_inside_frozen_limit", result["elapsed_seconds"] <= protocol["limits"]["wall_seconds"])
            review.check("complete_run_peak_RSS_inside_frozen_limit", result["peak_rss_bytes"] <= protocol["limits"]["peak_rss_bytes"])
        review.details["independent_final_replication_gate"] = final_gate
    except Exception as exc:
        # Never leak a private directory through an exception string.
        review.check("review_execution_completed", False, error_type=type(exc).__name__)
    report["checks"] = review.checks
    report["details"] = review.details
    report["check_count"] = len(review.checks)
    report["passed_count"] = sum(check["pass"] for check in review.checks)
    report["failed_checks"] = [check["name"] for check in review.checks if not check["pass"]]
    report["verification_status"] = "pass" if not report["failed_checks"] else "fail"
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    output = directory / "review.json"
    output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    output.chmod(0o600)
    return report


def _self_test():
    """Small generated arrays only; independently compare both pair identities."""
    rng = np.random.default_rng(82719)
    for _ in range(20):
        values = rng.integers(-3, 4, size=(16, 20)).astype(float)
        reference = rng.choice(np.array([0, 32, 100, 220], dtype=np.uint8), size=(16, 20))
        valid = rng.random((16, 20)) > .15
        positive, negative = _label_masks(reference, valid)
        np.testing.assert_allclose(_pair_auc(values, positive, negative), _rank_auc(values, positive, negative), rtol=0, atol=1e-14)
    draws = rng.integers(0, 20, size=(15, 20))
    scores = [values, -values, values * 0.5]
    auc, counts, _ = _block_pair_bootstrap(scores, reference, valid, draws, 4)
    rows, columns = np.indices(valid.shape)
    ids = (rows // 4) * 5 + columns // 4
    for replicate, draw in enumerate(draws):
        multiplicity = np.bincount(draw, minlength=20)[ids]
        for index, score in enumerate(scores):
            p = np.repeat(score[positive], multiplicity[positive])
            n = np.repeat(score[negative], multiplicity[negative])
            joined = np.concatenate((p, n))
            ranked = rankdata(joined, method="average")
            expected = (ranked[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
            np.testing.assert_allclose(auc[replicate, index], expected, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(counts[replicate], [len(p), len(n)])
    print(json.dumps({"generated_fixture_pair_AUC_checks": 20, "generated_fixture_bootstrap_AUC_checks": 45, "status": "pass"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir")
    parser.add_argument("--cache")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.result_dir:
        parser.error("--result-dir is required unless --self-test is used")
    report = review_result(args.result_dir, args.cache)
    print(json.dumps({"verification_status": report["verification_status"], "check_count": report["check_count"],
                      "failed_checks": report["failed_checks"], "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["verification_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
