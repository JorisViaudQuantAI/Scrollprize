#!/usr/bin/env python3
"""Independent, local-only review of the frozen Frag5 stage-1 artifacts.

This checker makes no network requests, fits no model and selects no shifts.
It reads an already finished run, validates its acquisition ledger/cache, then
recomputes identity from cached raw chunks and all anatomy metrics from saved
arrays. It does not regenerate the 125-point footprint arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

RUNNER_SHA = "e18bdff3a75729dbbb2785f4caa56892388f36c2b5603d3c86d082cb21877600"
PROTOCOL_SHA = "3cae8c53394bd92de8f07bcc3eb81afaf9db81c3cfe039616cc27566b18c18a2"
FREEZE_REFERENCE = "4ea8eedb90d5757272cef59144c94d0b82cbb5e0"
CHUNK_BYTES = 128**3 * 2
NUMERIC_TOLERANCE = 1e-10


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def close(actual, expected, description):
    if expected is None:
        require(actual is None, description)
    else:
        require(actual is not None and np.isfinite(actual), description)
        require(abs(float(actual) - float(expected)) <= NUMERIC_TOLERANCE, description)


def public_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path leaves run directory")
    return path


def ranks(values):
    """Average 1-based ranks, implemented independently without scipy.rankdata."""
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    edges = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    averages = (edges[:-1] + edges[1:] + 1.0) / 2.0
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.repeat(averages, np.diff(edges))
    return result


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    da, db = a - np.mean(a), b - np.mean(b)
    return float(np.dot(da, db) / np.sqrt(np.dot(da, da) * np.dot(db, db)))


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    require(a.size == b.size and np.isfinite(a).all() and np.isfinite(b).all(), "Invalid metric arrays")
    if a.max() == a.min() or b.max() == b.min():
        return {"n": int(a.size), "status": "constant_unassessable", "pearson": None, "spearman": None}
    return {"n": int(a.size), "status": "assessable", "pearson": pearson(a, b),
            "spearman": pearson(ranks(a), ranks(b))}


def compare_metric(actual, stored):
    require(actual["n"] == stored["n"] and actual["status"] == stored["status"], "Metric support/status mismatch")
    for name in ("pearson", "spearman"):
        close(actual[name], stored[name], "Stored correlation differs from independent result")


def discrete_laplacian(a):
    # Preserve the declared floating-point addition order because changing it
    # can change exact ties before rank correlations. Metrics are independent.
    core = a[1:-1, 1:-1, 1:-1]
    neighbors = (a[2:, 1:-1, 1:-1], a[:-2, 1:-1, 1:-1], a[1:-1, 2:, 1:-1],
                 a[1:-1, :-2, 1:-1], a[1:-1, 1:-1, 2:], a[1:-1, 1:-1, :-2])
    result = neighbors[0].copy()
    for neighbor in neighbors[1:]:
        result += neighbor
    return result - 6 * core


def all_box_keys(lo, hi, shape):
    lo, hi = np.asarray(lo), np.asarray(hi)
    require(np.all(lo >= 0) and np.all(hi < np.asarray(shape)[::-1]), "Static support outside source")
    return set(tuple(xyz[::-1]) for xyz in itertools.product(
        *[range(int(a) // 128, int(b) // 128 + 1) for a, b in zip(lo, hi)]))


def verify_support(plan):
    A = np.asarray(plan["affine_xyz"], dtype=np.float64)
    inverse = np.linalg.inv(A[:3, :3])
    union = {"fine": set(), "coarse": set()}
    identities = set()
    for center, stored in zip(plan["centers"], plan["support_plan"]["centers"]):
        p = np.asarray(center["fine_xyz"])
        normal = np.asarray(center["stored_normal_xyz"])
        c = np.rint(A[:3, :3] @ p + A[:3, 3]).astype(np.int64)
        require(c.tolist() == stored["coarse_center_xyz"], "Frozen center differs")
        ray = p + np.arange(-32, 33)[:, None] * normal
        vertex = np.asarray(list(itertools.product(*[(v - 16.4, v + 16.4) for v in c])))
        fine_points = (vertex - A[:3, 3]) @ inverse.T
        expected = {
            "identity_fine_box_xyz": [np.floor(ray.min(0)).astype(int).tolist(), np.ceil(ray.max(0)).astype(int).tolist()],
            "anatomy_fine_box_xyz": [np.floor(np.vstack((ray, fine_points)).min(0)).astype(int).tolist(),
                                     np.ceil(np.vstack((ray, fine_points)).max(0)).astype(int).tolist()],
            "anatomy_coarse_box_xyz": [(c - 17).tolist(), (c + 17).tolist()],
        }
        for field, value in expected.items():
            require(value == stored[field], "Frozen support box differs")
        identities |= all_box_keys(*expected["identity_fine_box_xyz"], plan["volumes"]["fine"]["metadata"]["shape"])
        for scan in ("fine", "coarse"):
            union[scan] |= all_box_keys(*expected[f"anatomy_{scan}_box_xyz"], plan["volumes"][scan]["metadata"]["shape"])
    for scan in union:
        require(union[scan] == {tuple(k) for k in plan["support_plan"][f"{scan}_chunk_keys_zyx"]}, "Chunk union mismatch")
    require(identities == {tuple(k) for k in plan["support_plan"]["identity_fine_chunk_keys_zyx"]}, "Identity chunk union mismatch")
    count = sum(map(len, union.values()))
    require(count == plan["support_plan"]["decoded_chunks"] and count <= 64, "Shared chunk cap mismatch")
    return {"fine_chunks": len(union["fine"]), "coarse_chunks": len(union["coarse"]),
            "shared_chunks": count, "decoded_bytes_upper_bound": count * CHUNK_BYTES,
            "all_cartesian_support_corners_checked": True}


def source_url(plan, scan, key):
    return plan["volumes"][scan]["base_url"] + "/" + "/".join(map(str, key))


def inspect_acquisition(root, plan):
    path = root / "acquisition.json"
    if not path.exists():
        return {"available": False}, {}, {}
    ledger = read_json(path)
    objects, positions = {}, {}
    expected_sources = {source_url(plan, scan, key) for scan in ("fine", "coarse")
                        for key in plan["support_plan"][f"{scan}_chunk_keys_zyx"]}
    source_metadata = {plan["volumes"][scan]["metadata_url"]: plan["volumes"][scan]["metadata_sha256"]
                       for scan in ("fine", "coarse")}
    expected_tiff = {plan["surface_base_url"] + f"/surface_volume/{k:02}.tif" for k in range(65)}
    charged = Counter()
    header_indices, pixel_indices, source_indices, coarse_indices = [], [], [], []
    complete_header_ranges = Counter()
    complete_pixel_ranges = Counter()
    for index, response in enumerate(ledger["responses"]):
        url, category = response["url"], response["category"]
        require(url in expected_sources | expected_tiff | set(source_metadata), "Unexpected acquisition URL")
        require(category in ("source_chunk", "tiff_metadata"), "Unexpected acquisition category")
        head = ledger["heads"][url]
        require(head["length"] > 0 and bool(head["etag"]), "Missing source length or ETag")
        requested = response["range"]
        expected = head["length"] if requested is None else requested[1] - requested[0] + 1
        require(response["expected_bytes"] == expected, "Incorrect prospective byte charge")
        require(0 <= response["body_bytes_read"] <= expected, "Invalid observed body count")
        charged[category] += expected
        if url in expected_tiff:
            require(head["length"] == 66658265, "Unexpected TIFF object length")
            if requested in ([0, 7], [66658088, 66658237]):
                header_indices.append(index)
                if response["status"] == "complete":
                    complete_header_ranges[(url, tuple(requested))] += 1
            else:
                allowed = [[8 + 2 * (4560 * c["surface_yx"][0] + c["surface_yx"][1]),
                            9 + 2 * (4560 * c["surface_yx"][0] + c["surface_yx"][1])] for c in plan["centers"]]
                require(requested in allowed, "Unexpected TIFF pixel range")
                pixel_indices.append(index)
                if response["status"] == "complete":
                    complete_pixel_ranges[(url, tuple(requested))] += 1
        elif url in expected_sources:
            require(category == "source_chunk" and requested is None, "Raw source must be whole pinned object")
            source_indices.append(index)
            if url.startswith(plan["volumes"]["coarse"]["base_url"] + "/"):
                coarse_indices.append(index)
        if response["status"] == "complete":
            cached = public_path(root, response["cache_file"])
            require(cached.stat().st_size == expected, "Cached response length differs")
            require(response["body_bytes_read"] == expected, "Completed response byte count differs")
            require(digest(cached) == response["sha256"], "Cached response hash differs")
            key = (url, None if requested is None else tuple(requested))
            require(key not in objects, "Undeclared repeated complete acquisition")
            objects[key] = cached
            positions[key] = index
            if url in source_metadata:
                require(response["sha256"] == source_metadata[url], "Pinned source metadata differs")
    for category in ("source_chunk", "tiff_metadata"):
        require(charged[category] == ledger["body_bytes"][category], "Ledger charged-byte total differs")
        require(charged[category] <= plan["caps"][category + "_body_bytes"], "Body cap exceeded")
    require(ledger["cpu_seconds"] <= plan["caps"]["cpu_seconds"], "CPU cap exceeded")
    if pixel_indices or source_indices:
        require(len(complete_header_ranges) == 130 and all(v == 1 for v in complete_header_ranges.values()),
                "Pixel source acquired before complete set of header responses")
        first_pixel_source = min(pixel_indices + source_indices)
        require(max(header_indices) < first_pixel_source, "Pixel-source request preceded last header request")
        header_result = read_json(root / "all_tiff_headers.json")
        require(header_result["all_65_headers_checked_before_any_CT_or_TIFF_pixel_body"], "Header barrier attestation absent")
        require(sorted(h["depth"] for h in header_result["headers"]) == list(range(65)), "Header depth coverage differs")
    freeze = root / "acquisition_heads_freeze.json"
    if freeze.exists():
        frozen = read_json(freeze)
        require(frozen["heads"] == ledger["heads"], "HEAD identities differ from initial freeze")
        require(frozen["freeze_reference"] == FREEZE_REFERENCE, "HEAD freeze reference differs")
        require(frozen["planned_source_chunk_bytes"] == sum(ledger["heads"][u]["length"] for u in expected_sources),
                "Planned source-byte sum differs")
    return {"available": True, "responses": len(ledger["responses"]), "complete_cached_objects_verified": len(objects),
            "body_bytes_prospectively_charged": dict(charged), "cpu_seconds": ledger["cpu_seconds"],
            "complete_tiff_header_ranges": len(complete_header_ranges), "complete_tiff_pixel_ranges": len(complete_pixel_ranges),
            "coarse_response_count": len(coarse_indices), "all_pixel_requests_after_header_request_set": bool(pixel_indices),
            "first_coarse_request_index": min(coarse_indices) if coarse_indices else None,
            "last_tiff_pixel_request_index": max(pixel_indices) if pixel_indices else None,
            "ordering_scope": "Ordered request ledger plus published code completion barriers; receipt has no per-response completion timestamps."}, objects, positions


def raw_value_reader(plan, objects):
    import numcodecs
    numcodecs.blosc.set_nthreads(1)
    codec = numcodecs.get_codec(plan["volumes"]["fine"]["metadata"]["compressor"])
    cache = {}
    def values(xyz):
        xyz = np.asarray(xyz, dtype=np.int64)
        result = np.empty(len(xyz), dtype=np.float64)
        chunks = xyz // 128
        for key_xyz in np.unique(chunks, axis=0):
            key = tuple(map(int, key_xyz[::-1]))
            if key not in cache:
                raw = objects[(source_url(plan, "fine", key), None)].read_bytes()
                require(len(raw) >= 16, "Truncated cached Blosc object")
                header = struct.unpack("<4B3I", raw[:16])
                require(header[4] == CHUNK_BYTES and header[6] == len(raw), "Invalid cached Blosc size")
                decoded = codec.decode(raw)
                require(len(decoded) == CHUNK_BYTES, "Unexpected decoded raw length")
                cache[key] = np.frombuffer(decoded, dtype="<u2").reshape((128, 128, 128))
            selected = np.all(chunks == key_xyz, axis=1)
            local = xyz[selected] - 128 * key_xyz
            result[selected] = cache[key][local[:, 2], local[:, 1], local[:, 0]]
        return result
    return values


def independent_interpolate(points, integer_values):
    points = np.asarray(points, dtype=np.float64)
    lo = np.floor(points).astype(np.int64)
    fraction = points - lo
    result = np.zeros(len(points), dtype=np.float64)
    for direction in itertools.product((0, 1), repeat=3):
        d = np.asarray(direction, dtype=np.int64)
        weight = np.prod(np.where(d, fraction, 1 - fraction), axis=1)
        active = weight != 0
        result[active] += weight[active] * integer_values(lo[active] + d)
    return result


def review_identity(root, plan, objects):
    identity_path = root / "identity_result.json"
    if not identity_path.exists():
        return {"available": False}
    stored = read_json(identity_path)
    require(len(stored["profiles"]) == 4, "Identity has wrong number of profiles")
    get_values = raw_value_reader(plan, objects)
    primary_pass, reverse_pass = [], []
    checked = []
    for center, row in zip(plan["centers"], stored["profiles"]):
        require(center["id"] == row["id"], "Identity center order differs")
        p, n = np.asarray(center["fine_xyz"]), np.asarray(center["stored_normal_xyz"])
        predicted = independent_interpolate(p + np.arange(-32, 33)[:, None] * n, get_values)
        original_predicted = np.asarray(row["fine_trilinear_values"])
        require(original_predicted.shape == (65,), "Wrong identity depth count")
        max_difference = float(np.max(np.abs(predicted - original_predicted)))
        require(max_difference <= NUMERIC_TOLERANCE, "Independent raw interpolation differs")
        y, x = center["surface_yx"]
        start = 8 + 2 * (4560 * y + x)
        observed = np.array([struct.unpack("<H", objects[(plan["surface_base_url"] + f"/surface_volume/{k:02}.tif",
                                                          (start, start + 1))].read_bytes())[0] for k in range(65)], dtype=np.int64)
        require(observed.tolist() == row["tiff_values"], "Cached TIFF values differ")
        rounded = np.rint(predicted).astype(np.int64)
        informative = bool(np.ptp(rounded) > 0 and np.ptp(observed) > 0 and np.any(rounded) and np.any(observed))
        require(informative == row["nonconstant_nonzero"], "Informative-profile flag differs")
        errors = {}
        for label, version in (("primary", rounded), ("reverse", rounded[::-1])):
            error = np.abs(version - observed)
            require(error.tolist() == row[label + "_absolute_errors"], "Rounded error vector differs")
            require(int(error.max()) == row[label + "_max_abs_error_uint16"], "Rounded error maximum differs")
            errors[label] = int(error.max())
        primary_pass.append(informative and errors["primary"] <= 1)
        reverse_pass.append(informative and errors["reverse"] <= 1)
        checked.append({"id": center["id"], "source_samples": 65, "maximum_raw_interpolation_difference": max_difference,
                        "primary_max_abs_error_uint16": errors["primary"], "reverse_max_abs_error_uint16": errors["reverse"]})
    primary, reverse = all(primary_pass), all(reverse_pass)
    classification = "both_ambiguous" if primary and reverse else "primary_matched" if primary else "reverse_only" if reverse else "neither"
    allowed = primary != reverse
    require(classification == stored["classification"] and allowed == stored["anatomy_allowed"], "Independent classification differs")
    return {"available": True, "classification": classification, "anatomy_allowed": allowed,
            "raw_source_and_tiff_recomputed": True, "profiles": checked,
            "claim_scope": "Local consistency at four fixed rays with the declared source/order/phase hypothesis only."}


def review_anatomy(root, plan, result, identity):
    rows = result.get("anatomy", [])
    if not identity.get("anatomy_allowed", False):
        require(rows == [], "Anatomy exists without unique identity")
        return {"performed": False, "reason": "No unique accepted identity"}
    require([r["id"] for r in rows] == [c["id"] for c in plan["centers"]], "Incomplete anatomy center set")
    expected_shifts = set(itertools.product((-1, 0, 1), repeat=3))
    expected_arms = {"fine_point", "fine_nominal_coarse_footprint"}
    checked = []
    maximum_difference = 0.0
    for row, geometry in zip(rows, plan["support_plan"]["centers"]):
        path = public_path(root, row["arrays_file"])
        require(digest(path) == row["arrays_sha256"], "Anatomy archive hash differs")
        with zipfile.ZipFile(path) as archive:
            require(sum(info.file_size for info in archive.infolist()) < 3000000, "Unexpected anatomy archive decoded size")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        require(set(arrays) == expected_arms | {"coarse_neighborhood", "coarse_center_xyz", "coarse_neighborhood_origin_xyz"}, "Unexpected anatomy archive members")
        for arm in expected_arms:
            require(arrays[arm].shape == (33, 33, 33) and np.isfinite(arrays[arm]).all(), "Unexpected fine anatomy shape")
        require(arrays["coarse_neighborhood"].shape == (35, 35, 35), "Unexpected coarse neighborhood")
        require(arrays["coarse_center_xyz"].tolist() == geometry["coarse_center_xyz"], "Saved center differs")
        require(arrays["coarse_neighborhood_origin_xyz"].tolist() == geometry["anatomy_coarse_box_xyz"][0], "Saved origin differs")
        require(np.isfinite(arrays["coarse_neighborhood"]).all(), "Nonfinite coarse neighborhood")
        lookup = {(tuple(r["shift_xyz_coarse_voxels"]), r["arm"]): r for r in row["all_shifts"]}
        require(len(row["all_shifts"]) == 54 and set(lookup) == set(itertools.product(expected_shifts, expected_arms)), "Missing/duplicate shift-arm comparison")
        zeros = []
        for shift in sorted(expected_shifts):
            x, y, z = shift
            native = arrays["coarse_neighborhood"][1+z:34+z, 1+y:34+y, 1+x:34+x]
            for arm in sorted(expected_arms):
                stored = lookup[(shift, arm)]
                require(stored["original_transform_primary"] == (shift == (0, 0, 0)), "Primary-shift marker differs")
                for measure, a, b in (("intensity", arrays[arm], native),
                                      ("laplacian_interior", discrete_laplacian(arrays[arm]), discrete_laplacian(native))):
                    actual = metric(a, b)
                    compare_metric(actual, stored[measure])
                    for name in ("pearson", "spearman"):
                        if actual[name] is not None:
                            maximum_difference = max(maximum_difference, abs(actual[name] - stored[measure][name]))
                if shift == (0, 0, 0):
                    zeros.append({"arm": arm, "intensity": stored["intensity"], "laplacian_interior": stored["laplacian_interior"]})
        compare_metric(metric(arrays["fine_point"], arrays["fine_nominal_coarse_footprint"]), row["within_fine_point_vs_footprint"])
        checked.append({"id": row["id"], "shifts": 27, "arms": 2, "comparisons_verified": 54, "zero_shift": zeros})
    return {"performed": True, "centers": checked, "total_shift_arm_comparisons_verified": 216,
            "correlation_metrics_per_comparison": 4, "maximum_correlation_difference": maximum_difference,
            "footprint_array_generation_repeated": False, "all_fixed_shifts_checked_no_best_shift_selection": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Preserve prior review; output must be new")
    started = time.process_time()
    report = {"schema": "frag5-stage1-independent-result-review-v1", "checker_sha256": digest(Path(__file__)),
              "freeze_reference": FREEZE_REFERENCE, "actual_network_requests": 0, "models_or_fits": 0,
              "new_source_acquisition": False, "label_or_photo_access": False, "numeric_tolerance": NUMERIC_TOLERANCE,
              "python_version": sys.version, "numpy_version": np.__version__}
    try:
        require(digest(args.runner) == RUNNER_SHA, "Published runner bytes differ")
        require(digest(args.protocol) == PROTOCOL_SHA, "Published protocol bytes differ")
        plan = read_json(args.protocol)
        require((args.run_dir / "result.json").exists() or (args.run_dir / "error.json").exists(), "Run has not finished")
        start = read_json(args.run_dir / "start.json")
        require(start["freeze_reference"] == FREEZE_REFERENCE and start["runner_sha256"] == RUNNER_SHA
                and start["plan_sha256"] == PROTOCOL_SHA, "Runtime freeze identity differs")
        report["static_support"] = verify_support(plan)
        report["acquisition"], objects, positions = inspect_acquisition(args.run_dir, plan)
        report["identity"] = review_identity(args.run_dir, plan, objects)
        if report["acquisition"].get("coarse_response_count", 0):
            require(report["identity"].get("anatomy_allowed", False), "Coarse acquisition without unique identity")
            coarse_start = report["acquisition"]["first_coarse_request_index"]
            tiff_end = report["acquisition"]["last_tiff_pixel_request_index"]
            require(tiff_end is not None, "Coarse acquisition lacks preceding TIFF inputs")
            require(coarse_start > tiff_end, "Coarse source requested before identity inputs complete")
        if (args.run_dir / "result.json").exists():
            result = read_json(args.run_dir / "result.json")
            report["run_result_sha256"] = digest(args.run_dir / "result.json")
            require(result["freeze_reference"] == FREEZE_REFERENCE, "Result freeze reference differs")
            require(not result["labels_or_photographs_accessed"] and not result["registration_fitted"]
                    and not result["best_shift_selected"] and not result["automatic_label_stage_go"], "Declared scientific scope changed")
            require(result["identity"] == read_json(args.run_dir / "identity_result.json"), "Embedded identity differs")
            report["anatomy"] = review_anatomy(args.run_dir, plan, result, report["identity"])
            report["run_outcome"] = result["outcome"]
            require(result["resources"]["cpu_seconds"] <= plan["caps"]["cpu_seconds"], "Run CPU cap exceeded")
            require(result["resources"]["body_bytes_prospectively_charged"] == report["acquisition"]["body_bytes_prospectively_charged"],
                    "Final resource totals differ from acquisition ledger")
            raw_objects = sum(1 for url, interval in objects if any(url.startswith(plan["volumes"][scan]["base_url"] + "/")
                              and not url.endswith("/.zarray") for scan in ("fine", "coarse")))
            require(result["resources"]["decoded_chunk_count"] == raw_objects, "Final decoded-chunk count differs")
            report["status"] = "passed_completed_artifact_review"
        else:
            report["run_error_sha256"] = digest(args.run_dir / "error.json")
            report["run_error_type"] = read_json(args.run_dir / "error.json").get("error_type")
            report["status"] = "verified_available_artifacts_of_incomplete_run"
        report["interpretation_limits"] = [
            "Identity is local consistency at four frozen rays; not whole-surface provenance or physical ink-face authentication.",
            "Anatomy correlations are descriptive within one physical fragment; all27shifts are retained and no affine is fitted.",
            "The footprint is125midpoint samples of the interpolated fine field, not an authenticated coarse acquisition PSF.",
            "No result gives automatic permission for a label stage.",
        ]
    except Exception as exc:
        report["status"] = "review_failed"
        report["failure_type"] = type(exc).__name__
        report["failure_message"] = str(exc) if isinstance(exc, ValueError) else "Local artifact review stopped; inspect checker and saved artifacts."
    report["review_cpu_seconds"] = time.process_time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "receipt_sha256": digest(args.out), "review_cpu_seconds": report["review_cpu_seconds"]}))
    if report["status"] == "review_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
