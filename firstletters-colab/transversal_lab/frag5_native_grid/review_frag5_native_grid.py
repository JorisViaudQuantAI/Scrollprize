#!/usr/bin/env python3
"""Independent local-only review of the separately frozen Frag5 follow-up.

The old stage-1 checker is imported unchanged. Its legacy-render identity gate
is never changed or rescored. Only its numerical saved-array anatomy checker is
called with an explicit adapter for the NEW native-integer gate. No downloads,
model fits, phase searches or footprint reintegration occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

OLD_CHECKER_SHA = "a78c35a71954923eb3e2bb8d7619b0279ea5793e47d3c14e57fada180a157918"
NEW_RUNNER_SHA = "6d9f0eee1ddaf6e4563a0ec790441141df0e36591e0be5c4c5ae836ea80cab29"
NEW_PROTOCOL_SHA = "b0d1cd17d9e651d39f3e199d8a354ad8c9aacb2e2f657c5df6fc6479f49a6745"
NEW_FREEZE = "1dae1cbf3c9eeb8c7713fa0e50795608afc823ec"
TOLERANCE = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1048576), b""):
            value.update(data)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def import_checker(path):
    require(digest(path) == OLD_CHECKER_SHA, "Original independent checker changed")
    spec = importlib.util.spec_from_file_location("unchanged_frag5_stage1_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected, message):
    require(np.isfinite(actual) and abs(float(actual) - float(expected)) <= TOLERANCE, message)


def audit_new_acquisition(root, protocol, plan, checker):
    if not (root / "acquisition.json").exists():
        return {"available": False}, {}, {}
    ledger = read_json(root / "acquisition.json")
    old_raw = {item["url"] for item in protocol["cached_fine_chunks"]}
    all_raw = set(protocol["all_source_expected_heads"])
    allowed_raw = all_raw - old_raw
    native_by_url = {s["url"]: s for s in protocol["native_slices"]}
    metadata = {plan["volumes"][scan]["metadata_url"]: plan["volumes"][scan]["metadata_sha256"] for scan in ("fine", "coarse")}
    expected_native_headers = {(s["url"], tuple(r["range"])): r["sha256"]
                               for s in protocol["native_slices"] for r in s["header_ranges"]}
    expected_native_pixels = {(r["url"], tuple(r["byte_range"])) for r in protocol["corner_rows"]}
    required_heads = all_raw | set(native_by_url) | set(metadata)
    allowed_requests = required_heads - old_raw
    pinned_heads = {**protocol["all_source_expected_heads"], **{url: s["head"] for url, s in native_by_url.items()}}
    for url, actual in ledger["heads"].items():
        require(url in required_heads and actual["length"] > 0 and bool(actual["etag"]), "Unexpected/malformed HEAD record")
        if url in pinned_heads:
            require({k: actual[k] for k in ("length", "etag")} == pinned_heads[url], "Source length/ETag differs from pinned preflight")
    objects, positions = {}, {}
    charged = Counter()
    headers, pixels, raw_indices = [], [], []
    complete_headers = set()
    for index, row in enumerate(ledger["responses"]):
        url = row["url"]
        require(url in allowed_requests, "Unplanned or repeated old-raw GET")
        interval = None if row["range"] is None else tuple(row["range"])
        key = (url, interval)
        if url in native_by_url:
            require(row["category"] == "tiff_metadata", "Native TIFF response has wrong category")
            require(key in expected_native_headers or key in expected_native_pixels, "Unplanned native TIFF range")
            (headers if key in expected_native_headers else pixels).append(index)
        elif url in allowed_raw:
            require(interval is None and row["category"] == "source_chunk", "Raw chunk must be a full object")
            raw_indices.append(index)
        else:
            require(url in metadata and interval is None and row["category"] == "tiff_metadata", "Unexpected source-metadata request")
        expected = ledger["heads"][url]["length"] if interval is None else interval[1] - interval[0] + 1
        require(expected == row["expected_bytes"] and 0 <= row["body_bytes_read"] <= expected, "Response-byte accounting differs")
        charged[row["category"]] += expected
        if row["status"] == "complete":
            cached = checker.public_path(root, row["cache_file"])
            require(cached.stat().st_size == expected and row["body_bytes_read"] == expected, "Complete response length differs")
            require(digest(cached) == row["sha256"], "New cached-object hash differs")
            require(key not in objects, "Unplanned duplicate completed GET")
            objects[key], positions[key] = cached, index
            if key in expected_native_headers:
                require(row["sha256"] == expected_native_headers[key], "Native header bytes differ from frozen preflight")
                complete_headers.add(key)
            if url in metadata:
                require(row["sha256"] == metadata[url], "Pinned Zarr metadata differs")
    new_bytes = {category: charged[category] for category in ("source_chunk", "tiff_metadata")}
    require(new_bytes == ledger["body_bytes"], "New ledger totals differ")
    prior = protocol["prior_charges"]
    cumulative = {"source_chunk": prior["source_chunk_bytes"] + new_bytes["source_chunk"],
                  "tiff_metadata": prior["tiff_metadata_bytes"] + new_bytes["tiff_metadata"]}
    for category in cumulative:
        require(cumulative[category] <= protocol["cumulative_caps"][category + "_body_bytes"], "Cumulative byte cap exceeded")
    require(ledger["cpu_seconds"] + prior["reserved_cpu_seconds"] <= protocol["cumulative_caps"]["cpu_seconds"], "Cumulative CPU reservation exceeded")
    if pixels or raw_indices:
        require(complete_headers == set(expected_native_headers), "New pixel request without all eight complete native headers")
        require(max(headers) < min(pixels + raw_indices), "New pixel request preceded native-header barrier")
        require(read_json(root / "native_header_gate.json")["all_eight_native_headers_match"], "Native header attestation missing")
    if raw_indices:
        require(expected_native_pixels <= set(objects), "New raw chunk without all 32 native integer values")
        require(max(pixels) < min(raw_indices), "New raw chunk request preceded native equality inputs")
    if (root / "new_heads_freeze.json").exists():
        freeze = read_json(root / "new_heads_freeze.json")
        require(set(ledger["heads"]) == required_heads and freeze["heads"] == ledger["heads"], "HEAD snapshot changed")
        require(freeze["freeze_reference"] == NEW_FREEZE, "HEAD freeze commit differs")
        expected_new_source = sum(ledger["heads"][url]["length"] for url in allowed_raw)
        require(freeze["planned_new_source_bytes"] == expected_new_source, "New-source budget sum differs")
        require(freeze["new_native_or_raw_pixel_bodies_before_this_receipt"] == 0, "HEAD barrier attestation differs")
    return {"available": True, "responses": len(ledger["responses"]), "cached_objects_verified": len(objects),
            "new_body_bytes": new_bytes, "cumulative_body_bytes": cumulative,
            "complete_native_header_ranges": len(complete_headers),
            "complete_native_pixel_ranges": len(set(objects) & expected_native_pixels),
            "new_raw_chunk_responses": len(raw_indices),
            "all_fixed_native_ranges_charged_bytes": sum(r["expected_bytes"] for r in ledger["responses"]
                                                        if (r["url"], None if r["range"] is None else tuple(r["range"])) in expected_native_pixels),
            "ordering_scope": "Request ledger checked with frozen-code completion barriers; receipt has no individual completion timestamps."}, objects, positions


def audit_native_gate(root, protocol, plan, old_objects, new_objects, checker):
    if not (root / "native_source_gate.json").exists():
        return {"available": False}, None
    saved = read_json(root / "native_source_gate.json")
    require(len(saved["values"]) == 32, "Native gate has wrong sample count")
    integer_reader = checker.raw_value_reader(plan, old_objects)
    points = [xyz for row in protocol["corner_rows"] for xyz in row["xyz"]]
    old_values = integer_reader(np.asarray(points, dtype=np.int64)).astype(np.int64)
    reconstructed = []
    for row in protocol["corner_rows"]:
        values = struct.unpack("<HH", new_objects[(row["url"], tuple(row["byte_range"]))].read_bytes())
        for xyz, native in zip(row["xyz"], values):
            old = int(old_values[len(reconstructed)])
            reconstructed.append({"center_id": row["center_id"], "xyz": xyz, "native_uint16": native,
                                  "zarr_uint16": old, "signed_native_minus_zarr": native - old})
    require(reconstructed == saved["values"], "Native gate values or differences do not reproduce")
    equal = sum(row["native_uint16"] == row["zarr_uint16"] for row in reconstructed)
    passed = equal == 32
    maximum = max(abs(row["signed_native_minus_zarr"]) for row in reconstructed)
    require(equal == saved["equal_integer_values"] and passed == saved["passed_exact_32_of_32"]
            and maximum == saved["maximum_absolute_error"], "Native equality gate does not reproduce")
    return {"available": True, "passed_exact_32_of_32": passed, "equal_values": equal,
            "maximum_absolute_integer_difference": maximum, "all32_values_recomputed_from_cached_bytes": True,
            "scope": "Local equality at these fixed integer corners only."}, reconstructed


def audit_secondary(root, prior, protocol, plan, rows, old_objects, checker):
    if rows is None or not (root / "native_center_diagnostic.json").exists():
        return {"available": False}
    saved = read_json(root / "native_center_diagnostic.json")
    old = read_json(prior / "identity_result.json")
    require(old["classification"] == "neither", "Old rendering result was reclassified")
    old_by_id = {row["id"]: row for row in old["profiles"]}
    require(saved["adaptive_secondary_diagnostic_only"] and not saved["affects_gate"], "Secondary diagnostic scope changed")
    require([row["id"] for row in saved["centers"]] == [center["id"] for center in plan["centers"]], "Secondary center coverage differs")
    native_map = {tuple(row["xyz"]): row["native_uint16"] for row in rows}
    old_reader = checker.raw_value_reader(plan, old_objects)
    def native_reader(xyz):
        return np.asarray([native_map[tuple(map(int, q))] for q in xyz], dtype=np.float64)
    results = []
    for center, diagnostic in zip(plan["centers"], saved["centers"]):
        p = np.asarray(center["fine_xyz"], dtype=np.float64)[None, :]
        native = float(checker.independent_interpolate(p, native_reader)[0])
        old_raw = float(checker.independent_interpolate(p, old_reader)[0])
        old_row = old_by_id[center["id"]]
        y, x = center["surface_yx"]
        offset = 8 + 2 * (4560 * y + x)
        surface = struct.unpack("<H", old_objects[(plan["surface_base_url"] + "/surface_volume/32.tif", (offset, offset + 1))].read_bytes())[0]
        require(surface == old_row["tiff_values"][32] == diagnostic["previous_surface_tiff_k32"], "Previous TIFF center value differs")
        close(native, diagnostic["native_corner_trilinear"], "Native center interpolation differs")
        close(old_raw, diagnostic["previous_zarr_trilinear_center"], "Old raw center interpolation differs")
        close(native - surface, diagnostic["native_minus_previous_surface_unrounded"], "Native center signed error differs")
        rounded = int(np.rint(native))
        native_error = abs(rounded - surface)
        old_error = abs(int(np.rint(old_raw)) - surface)
        require(rounded == diagnostic["native_corner_trilinear_rounded"], "Native center rounding differs")
        require(native_error == diagnostic["native_vs_previous_surface_rounded_absolute_error"], "Native center rounded error differs")
        require(old_error == old_row["primary_absolute_errors"][32] == diagnostic["previous_zarr_vs_surface_rounded_absolute_error"], "Old center rounded error differs")
        require((native_error == old_error) == diagnostic["reproduces_previous_rounded_center_error"], "Center reproduction flag differs")
        results.append({"id": center["id"], "native_center_unrounded": native, "previous_surface_center": surface,
                        "native_vs_surface_rounded_absolute_error": native_error, "old_zarr_vs_surface_rounded_absolute_error": old_error,
                        "reproduces_previous_rounded_center_error": native_error == old_error})
    return {"available": True, "all_four_recomputed": True, "affects_source_or_anatomy_gate": False, "centers": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--stage1-protocol", type=Path, required=True)
    parser.add_argument("--old-checker", type=Path, default=Path(__file__).with_name("review_frag5_stage1.py"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Preserve earlier review: output must be new")
    started = time.process_time()
    report = {"schema": "frag5-adaptive-native-grid-independent-result-review-v1", "checker_sha256": digest(Path(__file__)),
              "unchanged_stage1_checker_sha256": OLD_CHECKER_SHA, "freeze_reference": NEW_FREEZE,
              "actual_network_requests": 0, "new_source_acquisition": False, "models_or_fits": 0,
              "label_or_photo_access": False, "footprint_integration_rerun": False,
              "python_version": sys.version, "numpy_version": np.__version__, "numeric_tolerance": TOLERANCE}
    try:
        checker = import_checker(args.old_checker)
        require(digest(args.runner) == NEW_RUNNER_SHA and digest(args.protocol) == NEW_PROTOCOL_SHA, "Follow-up source/protocol differs from published freeze")
        require(digest(args.stage1_protocol) == checker.PROTOCOL_SHA, "Original scientific protocol changed")
        require((args.run_dir / "result.json").exists() or (args.run_dir / "error.json").exists(), "Follow-up run has not finished")
        protocol, plan = read_json(args.protocol), read_json(args.stage1_protocol)
        for name, expected in protocol["prior_receipt_sha256"].items():
            require(digest(args.prior_run / name) == expected, "Prior failed-run receipt changed")
        old_result = read_json(args.prior_run / "result.json")
        require(old_result["identity"]["classification"] == "neither" and not old_result["anatomy"], "Original failure was not preserved")
        report["old_stage1_classification_preserved"] = "neither"
        report["static_support"] = checker.verify_support(plan)
        old_review, old_objects, old_positions = checker.inspect_acquisition(args.prior_run, plan)
        report["old_cached_objects_verified"] = old_review["complete_cached_objects_verified"]
        report["acquisition"], objects, positions = audit_new_acquisition(args.run_dir, protocol, plan, checker)
        start = read_json(args.run_dir / "start.json")
        require(start["freeze_reference"] == NEW_FREEZE and start["runner_sha256"] == NEW_RUNNER_SHA
                and start["protocol_sha256"] == NEW_PROTOCOL_SHA, "Follow-up runtime identity differs")
        report["native_gate"], rows = audit_native_gate(args.run_dir, protocol, plan, old_objects, objects, checker)
        report["secondary_center_diagnostic"] = audit_secondary(args.run_dir, args.prior_run, protocol, plan, rows, old_objects, checker)
        native_passed = report["native_gate"].get("passed_exact_32_of_32", False)
        if report["acquisition"].get("new_raw_chunk_responses", 0):
            require(native_passed, "New raw source payload acquired without exact native integer gate")
        if (args.run_dir / "result.json").exists():
            result = read_json(args.run_dir / "result.json")
            require(result["freeze_reference"] == NEW_FREEZE and result["adaptive_after_original_failure"]
                    and result["original_surface_contract_remains"] == "neither", "Separate adaptive question was not preserved")
            require(not result["labels_or_photographs_accessed"] and not result["registration_fitted"]
                    and not result["best_shift_selected"] and not result["automatic_label_stage_go"], "Scientific scope changed")
            require(result["native_source_gate"] == read_json(args.run_dir / "native_source_gate.json")
                    and result["native_center_trilinear_diagnostic"] == read_json(args.run_dir / "native_center_diagnostic.json"), "Embedded diagnostics differ")
            # This adapter authorizes ONLY the separately frozen raw-grid question.
            # It neither calls nor changes the old rendering identity classifier.
            report["anatomy_gate_adapter"] = "New native 32-value equality gate only; old rendering identity remains neither."
            if native_passed:
                report["anatomy"] = checker.review_anatomy(args.run_dir, plan, result, {"anatomy_allowed": True})
            else:
                require(result["anatomy"] == [], "Anatomy exists after a failed native equality gate")
                report["anatomy"] = {"performed": False, "reason": "The new native 32-value equality gate failed; old rendering gate remains unchanged."}
            resources = result["resources"]
            require(resources["new_body_bytes_prospectively_charged"] == report["acquisition"]["new_body_bytes"], "Final new-byte resources differ")
            require(resources["cumulative_source_chunk_bytes"] == report["acquisition"]["cumulative_body_bytes"]["source_chunk"]
                    and resources["cumulative_tiff_metadata_bytes_with_declared_preflights"] == report["acquisition"]["cumulative_body_bytes"]["tiff_metadata"], "Final cumulative resources differ")
            require(resources["cumulative_cpu_seconds_with_prior_reservation"] <= protocol["cumulative_caps"]["cpu_seconds"], "Final cumulative CPU exceeds cap")
            raw_new = sum(1 for (url, interval) in objects if url in protocol["all_source_expected_heads"])
            require(resources["decoded_unique_chunks"] == len(protocol["cached_fine_chunks"]) + raw_new <= 64, "Final unique-chunk count differs")
            expected_outcome = "completed_separate_adaptive_raw_scanner_grid_anatomy" if native_passed else "native_source_fidelity_failed_anatomy_not_run"
            require(result["outcome"] == expected_outcome, "Follow-up outcome contradicts its gate")
            report["run_outcome"] = result["outcome"]
            report["run_result_sha256"] = digest(args.run_dir / "result.json")
            report["resources"] = resources
            report["status"] = "passed_completed_adaptive_artifact_review"
        else:
            report["run_error_sha256"] = digest(args.run_dir / "error.json")
            report["run_error_type"] = read_json(args.run_dir / "error.json").get("error_type")
            report["status"] = "verified_available_artifacts_of_incomplete_adaptive_run"
            report["partial_anatomy_metrics_rechecked"] = False
        require(digest(args.old_checker) == OLD_CHECKER_SHA, "Original checker changed during review")
        report["interpretation_limits"] = [
            "Adaptive follow-up after a known failed legacy-render hypothesis; not an independent confirmation or rescue of that hypothesis.",
            "Exact native/Zarr agreement establishes only the32fixed integer values, not whole-volume identity, source history or physical ink face.",
            "Four PPM points serve only as previously observed geometric selectors; no photographic or label correspondence is validated.",
            "All raw-grid shifts are reported descriptively within one physical fragment; no best-shift selection or registration fit.",
            "The stored125midpoint footprint arrays are hash-checked but not reintegrated in this review; their metrics are independently recomputed.",
            "No outcome authorizes a label stage automatically.",
        ]
    except Exception as error:
        report["status"] = "review_failed"
        report["failure_type"] = type(error).__name__
        report["failure_message"] = str(error) if isinstance(error, ValueError) else "Local artifact review stopped; inspect saved artifacts and checker."
    report["review_cpu_seconds"] = time.process_time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "receipt_sha256": digest(args.out), "review_cpu_seconds": report["review_cpu_seconds"]}))
    if report["status"] == "review_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
