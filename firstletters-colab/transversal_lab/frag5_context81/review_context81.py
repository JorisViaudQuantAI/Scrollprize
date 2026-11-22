#!/usr/bin/env python3
"""Offline independent verification of the fixed-center Frag5 context study.

No primary scientific helper is imported. All published metrics are recomputed
with direct centered dot products, independently ranked ties and full binomial
kernels. Raw trilinear values and 125-midpoint integrals are checked at exactly
30 predefined geometric positions, not over the complete new grids.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import time
import traceback

import numpy as np

FREEZE = "7f20a7796431af44e922ffa0af82d0810f1eb189"
RUNNER_SHA = "0626d8497302edd85134bf6fb0a8740aaab2783c6a50409bd626712f2125a689"
PROTOCOL_SHA = "5a311410a8999eaf03a8db93e20662bf1fcf920640b6d12a5e74de9e20b181dc"
BASE_SHA = "3cae8c53394bd92de8f07bcc3eb81afaf9db81c3cfe039616cc27566b18c18a2"
PRIOR_SHA = "352e0bf0c8214f839689adb2a6b79e6375cff795c9b3c09e042e67e7196253c5"
RAW_TOLERANCE = 1e-6  # Declared before observing these raw-check outcomes.
IDS = (20, 25)
ARMS = ("fine_point", "fine_nominal_coarse_footprint")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, label):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {label}")
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"List differs: {label}")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        need(isinstance(actual, (int, float)) and np.isfinite(actual)
             and np.isclose(actual, expected, rtol=2e-11, atol=2e-10),
             f"Numeric mismatch: {label}: {actual!r} versus {expected!r}")
    else:
        need(actual == expected, f"Discrete mismatch: {label}: {actual!r} versus {expected!r}")


def center(a, side, xyz=(0, 0, 0)):
    slices = []
    for length, offset in zip(a.shape, xyz[::-1]):
        first = (length - side) // 2 + offset
        need(length % 2 == 1 and side % 2 == 1 and first >= 0 and first + side <= length,
             "Incomplete finite support")
        slices.append(slice(first, first + side))
    return a[tuple(slices)]


def binomial(a, passes):
    weights = np.array([math.comb(2 * passes, k) for k in range(2 * passes + 1)], float) / 4 ** passes
    result = np.asarray(a, float)
    for axis in range(3):
        result = np.apply_along_axis(lambda row: np.convolve(row, weights, mode="valid"), axis, result)
    return result


def average_ranks(a):
    values = np.asarray(a, float).ravel()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], values.size]
    result = np.empty(values.size, float)
    result[order] = np.repeat((starts + ends + 1) / 2., ends - starts)
    return result


def prepare(a):
    values = np.asarray(a, float).ravel()
    need(values.size > 1 and np.isfinite(values).all(), "Invalid metric input")
    if values.min() == values.max():
        return {"n": values.size, "constant": True}
    centered = values - values.mean()
    ranks = average_ranks(values)
    ranks -= ranks.mean()
    return {"n": values.size, "constant": False, "centered": centered,
            "ss": float(np.sum(centered ** 2)), "ranks": ranks, "rank_ss": float(np.sum(ranks ** 2))}


def correlate_prepared(a, b):
    need(a["n"] == b["n"], "Different comparison support")
    if a["constant"] or b["constant"]:
        return {"n": int(a["n"]), "pearson": None, "spearman": None, "status": "constant_unassessable"}
    return {"n": int(a["n"]),
            "pearson": float(np.sum(a["centered"] * b["centered"]) / math.sqrt(a["ss"] * b["ss"])),
            "spearman": float(np.sum(a["ranks"] * b["ranks"]) / math.sqrt(a["rank_ss"] * b["rank_ss"])),
            "status": "assessable"}


def describe(a):
    values = np.asarray(a, float)
    constant = values.min() == values.max()
    mean = float(values.flat[0]) if constant else float(values.mean())
    return {"n": int(values.size), "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": mean, "sd": 0. if constant else float(np.sqrt(np.mean((values - mean) ** 2))),
            "nonzero": int(np.count_nonzero(values)), "zeros": int(np.count_nonzero(values == 0)),
            "at_65535": int(np.count_nonzero(values == 65535))}


def source_path(root, relative):
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), "Cache path escapes root")
    return path


def verify_transport(args, plan, base, result):
    roots = {"stage1": args.stage1_cache, "native_grid": args.native_grid_cache}
    old_ledgers, old_urls = {}, set()
    for group, root in roots.items():
        ledger_path = root / "acquisition.json"
        need(sha(ledger_path) == plan["canonical_acquisition_sha256"][group], "Prior source ledger changed")
        old_ledgers[group] = json.loads(ledger_path.read_bytes())
        old_urls.update(row["url"] for row in old_ledgers[group]["responses"] if row["category"] == "source_chunk")
    acquisition = json.loads((args.run_dir / "acquisition.json").read_bytes())
    snapshot = json.loads((args.run_dir / "heads_before_new_data.json").read_bytes())
    sources = plan["source_objects"]
    target_urls = {row["url"] for row in sources}
    metadata_urls = {base["volumes"][volume]["metadata_url"] for volume in ("fine", "coarse")}
    need(len(old_urls) == 14 and len(target_urls) == 49 and len(target_urls & old_urls) == 6
         and len(target_urls - old_urls) == 43 and len(target_urls | old_urls) == 57, "Chunk union differs")
    need(set(acquisition["heads"]) == target_urls | metadata_urls, "HEAD object set differs")
    same(snapshot["heads"], acquisition["heads"], "admission.heads")
    need(snapshot["admitted"] is True and snapshot["new_raw_objects"] == 43
         and snapshot["raw_objects_reused"] == 6, "Admission record differs")
    head_rows = acquisition["head_requests"]
    need(len(head_rows) == 51 and {r["url"] for r in head_rows} == target_urls | metadata_urls,
         "HEAD request count/set differs")
    need(all(r["status"] == "complete" and r["http_status"] == 200 for r in head_rows), "Incomplete HEAD")
    rows = acquisition["responses"]
    need(len(rows) == 45 and all(r["status"] == "complete" for r in rows), "Unexpected or incomplete body responses")
    need(len({r["url"] for r in rows}) == 45, "Repeated body acquisition")
    need({r["url"] for r in rows[:2]} == metadata_urls and all(r["category"] == "tiff_metadata" for r in rows[:2]),
         "Metadata bodies were not first")
    need(all(r["category"] == "source_chunk" for r in rows[2:])
         and {r["url"] for r in rows[2:]} == target_urls - old_urls, "Unexpected raw bodies")
    by_url = {r["url"]: r for r in rows}
    raw_bytes = metadata_bytes = 0
    checked_files = []
    raw_paths = {}
    for row in rows:
        head = acquisition["heads"][row["url"]]
        need(row["range"] is None and row["expected_bytes"] == row["body_bytes_read"] == head["length"],
             "Body length/range mismatch")
        path = source_path(args.run_dir, row["cache_file"])
        need(path.stat().st_size == head["length"] and sha(path) == row["sha256"], "New source body hash/size mismatch")
        checked_files.append({"category": row["category"], "sha256": row["sha256"], "bytes": row["expected_bytes"]})
        if row["category"] == "source_chunk":
            raw_bytes += row["expected_bytes"]
        else:
            metadata_bytes += row["expected_bytes"]
    planned_bytes = 0
    for source in sources:
        volume, key = source["volume"], tuple(source["key_zyx"])
        expected_url = base["volumes"][volume]["base_url"].rstrip("/") + "/" + "/".join(map(str, key))
        need(source["url"] == expected_url, "Source key/URL mismatch")
        head = acquisition["heads"][source["url"]]
        need(head["length"] > 0 and bool(head["etag"]), "Invalid HEAD identity")
        cached = source["cached"]
        need((cached is not None) == (source["url"] in old_urls), "Cached/new split differs")
        if cached is None:
            need(head["length"] <= 4259840, "Per-object admission cap exceeded")
            planned_bytes += head["length"]
            row = by_url[source["url"]]
            path = source_path(args.run_dir, row["cache_file"])
        else:
            group = cached["cache_group"]
            prior = {r["url"]: r for r in old_ledgers[group]["responses"] if r["category"] == "source_chunk"}[source["url"]]
            need(prior["status"] == "complete" and prior["sha256"] == cached["sha256"]
                 and prior["expected_bytes"] == cached["bytes"] and prior["cache_file"] == cached["cache_file"],
                 "Cached raw source lineage differs")
            for name in ("length", "etag"):
                need(head[name] == cached["head"][name] == old_ledgers[group]["heads"][source["url"]][name],
                     "Retained source identity changed")
            path = source_path(roots[group], cached["cache_file"])
            need(path.stat().st_size == cached["bytes"] and sha(path) == cached["sha256"], "Retained raw source hash/size mismatch")
            checked_files.append({"category": "cached_source_chunk", "sha256": cached["sha256"], "bytes": cached["bytes"]})
        raw_paths[volume, key] = path
    need(planned_bytes == raw_bytes == snapshot["planned_new_raw_bytes"], "Raw admission/transfer mismatch")
    need(raw_bytes + 57064081 <= 268435456 and metadata_bytes <= 1048576, "Body budget exceeded")
    same(acquisition["body_bytes"], {"source_chunk": raw_bytes, "tiff_metadata": metadata_bytes}, "acquisition.bytes")
    for volume in ("fine", "coarse"):
        spec = base["volumes"][volume]
        row = by_url[spec["metadata_url"]]
        need(row["sha256"] == spec["metadata_sha256"], "Raw volume metadata hash changed")
    resources = result["resources"]
    expected = {"new_raw_body_bytes": raw_bytes, "cumulative_raw_body_bytes": raw_bytes + 57064081,
                "new_metadata_body_bytes": metadata_bytes, "new_raw_chunks": 43,
                "target_decoded_chunks": 49, "cumulative_unique_raw_chunks": 57}
    for key, value in expected.items():
        same(resources[key], value, "resources." + key)
    need(0 < resources["process_cpu_seconds"] <= 360 and resources["wall_seconds"] > 0, "Runtime budget/record invalid")
    return raw_paths, {"new_body_files_verified": 45, "retained_raw_files_verified": 6,
                       "head_requests_verified": 51, "raw_bytes": raw_bytes,
                       "cumulative_raw_bytes": raw_bytes + 57064081, "metadata_bytes": metadata_bytes,
                       "target_unique_chunks": 49, "cumulative_unique_chunks": 57,
                       "source_file_hashes": checked_files}


class RawReader:
    def __init__(self, paths):
        self.paths, self.decoded = paths, {}
        self.integer_reads = 0

    def at(self, volume, xyz):
        x, y, z = map(int, xyz)
        key = (z // 128, y // 128, x // 128)
        identity = (volume, key)
        need(identity in self.paths, "Manual interpolation requested an unplanned chunk")
        if identity not in self.decoded:
            import numcodecs.blosc
            numcodecs.blosc.set_nthreads(1)
            raw = self.paths[identity].read_bytes()
            need(len(raw) >= 16, "Short raw Blosc header")
            header = struct.unpack("<4B3I", raw[:16])
            need(header[4] == 128 ** 3 * 2 and header[6] == len(raw), "Unexpected raw Blosc sizes")
            decoded = numcodecs.blosc.decompress(raw)
            need(len(decoded) == 128 ** 3 * 2, "Unexpected raw decoded bytes")
            self.decoded[identity] = np.frombuffer(decoded, dtype="<u2").reshape((128,) * 3)
        self.integer_reads += 1
        return int(self.decoded[identity][z % 128, y % 128, x % 128])


def manual_trilinear(get_value, xyz):
    xyz = np.asarray(xyz, float)
    lower = np.floor(xyz).astype(int)
    fraction = xyz - lower
    weighted = []
    for bits in itertools.product((0, 1), repeat=3):
        weight = math.prod(fraction[i] if bit else 1 - fraction[i] for i, bit in enumerate(bits))
        if weight != 0:
            weighted.append(weight * get_value(lower + np.array(bits)))
    return math.fsum(weighted)


def raw_positions():
    positions = list(itertools.product((-40, 40), repeat=3))
    for axis in range(3):
        for value in (-40, 40):
            offset = [0, 0, 0]
            offset[axis] = value
            positions.append(tuple(offset))
    positions.append((0, 0, 0))
    need(len(positions) == len(set(positions)) == 15, "Fixed position enumeration changed")
    return positions


def verify_raw_samples(arrays, plan, base, reader):
    A = np.array(base["affine_xyz"], float)
    matrix, t = A[:3, :3], A[:3, 3]
    nodes = (-.4, -.2, 0., .2, .4)
    midpoint_offsets = np.array(list(itertools.product(nodes, repeat=3)))
    rows = []
    for item in plan["centers"]:
        i, center_xyz = item["id"], np.array(item["center_xyz"], int)
        data = arrays[i]
        for offset in raw_positions():
            q = center_xyz + np.array(offset)
            point_xyz = np.linalg.solve(matrix, q - t)
            point = manual_trilinear(lambda xyz: reader.at("fine", xyz), point_xyz)
            samples = []
            for u in midpoint_offsets:
                xyz = np.linalg.solve(matrix, q + u - t)
                samples.append(manual_trilinear(lambda v: reader.at("fine", v), xyz))
            footprint = math.fsum(samples) / 125
            index = tuple(np.array(offset)[::-1] + 40)
            coarse_index = tuple(np.array(offset)[::-1] + 41)
            recorded_point = float(data["fine_point"][index])
            recorded_footprint = float(data["fine_nominal_coarse_footprint"][index])
            native = reader.at("coarse", q)
            recorded_native = float(data["coarse_neighborhood"][coarse_index])
            rows.append({"id": i, "offset_xyz": list(offset),
                         "manual_point": point, "recorded_point": recorded_point,
                         "point_absolute_error": abs(point - recorded_point),
                         "manual_125_midpoint_mean": footprint, "recorded_footprint": recorded_footprint,
                         "footprint_absolute_error": abs(footprint - recorded_footprint),
                         "native_uint16": native, "recorded_native": recorded_native,
                         "native_absolute_error": abs(native - recorded_native)})
    return {"positions_per_site": 15, "sites": 2, "positions": 30,
            "geometry": "Eight corners at +/-40, six face centers at +/-40 along one axis, and the center, per unchanged site.",
            "predeclared_fine_absolute_tolerance_counts": RAW_TOLERANCE,
            "fine_point_evaluations": 30, "fine_midpoint_evaluations": 3750,
            "native_integer_evaluations": 30, "raw_integer_reads": reader.integer_reads,
            "decoded_source_chunks_for_fixed_checks": len(reader.decoded),
            "all_checks_passed": all(r["point_absolute_error"] <= RAW_TOLERANCE
                                     and r["footprint_absolute_error"] <= RAW_TOLERANCE
                                     and r["native_absolute_error"] == 0 for r in rows),
            "maximum_point_absolute_error": max(r["point_absolute_error"] for r in rows),
            "maximum_footprint_absolute_error": max(r["footprint_absolute_error"] for r in rows),
            "maximum_native_absolute_error": max(r["native_absolute_error"] for r in rows),
            "rows": rows,
            "limit": "All reported correlations are recomputed, but raw point/integration validation samples only these30 positions; the complete81-cubed integrations are not independently repeated."}


def recompute_metrics(arrays, check_cpu):
    nested, smoothed, shell, quality = [], [], [], []
    shell_mask = np.ones((81, 81, 81), bool)
    shell_mask[24:57, 24:57, 24:57] = False
    need(shell_mask.sum() == 495504, "Outer-shell count differs")
    for i in IDS:
        data = arrays[i]
        coarse = data["coarse_neighborhood"]
        for side in (17, 33, 65, 81):
            fine_prepared = {arm: prepare(center(data[arm], side)) for arm in ARMS}
            for shift in itertools.product((-1, 0, 1), repeat=3):
                b = prepare(center(coarse, side, shift))
                for arm in ARMS:
                    nested.append({"id": i, "side": side, "arm": arm, "shift_xyz": list(shift),
                                   "zero_shift_primary": shift == (0, 0, 0),
                                   **correlate_prepared(fine_prepared[arm], b)})
                check_cpu()
        zero = center(coarse, 81)
        native_shell = prepare(zero[shell_mask])
        for arm in ARMS:
            shell.append({"id": i, "arm": arm, "support": "81cube_excluding_central33cube",
                          **correlate_prepared(prepare(data[arm][shell_mask]), native_shell)})
        for passes in (0, 1, 4):
            native_filtered = center(binomial(zero, passes), 65)
            native_prepared = prepare(native_filtered)
            for name in (*ARMS, "native_coarse"):
                raw = data[name] if name in ARMS else zero
                a, raw_center = center(binomial(raw, passes), 65), center(raw, 65)
                retained = None if raw_center.min() == raw_center.max() else (
                    0. if a.min() == a.max() else float(np.var(a) / np.var(raw_center)))
                quality.append({"id": i, "array": name, "binomial_passes": passes, "side": 65,
                                "variance_retention": retained, **describe(a)})
                if name in ARMS:
                    smoothed.append({"id": i, "arm": name, "side": 65, "binomial_passes": passes,
                                     **correlate_prepared(prepare(a), native_prepared)})
            check_cpu()
    return {"nested_contexts_all_shifts": nested, "fixed65_smoothing_zero_shift": smoothed,
            "outer_shell_zero_shift": shell, "fixed65_structure": quality}


def review(args, cpu_start):
    def check_cpu():
        need(time.process_time() - cpu_start <= 360, "Independent review CPU budget exceeded")
    for path, expected in ((args.protocol, PROTOCOL_SHA), (args.runner, RUNNER_SHA),
                           (args.stage1_protocol, BASE_SHA), (args.prior_context_result, PRIOR_SHA)):
        need(sha(path) == expected, "Pinned input code/protocol/provenance changed")
    plan = json.loads(args.protocol.read_bytes())
    base = json.loads(args.stage1_protocol.read_bytes())
    start = json.loads((args.run_dir / "start.json").read_bytes())
    result = json.loads((args.run_dir / "result.json").read_bytes())
    need(start["freeze_reference"] == result["freeze_reference"] == FREEZE, "Freeze differs")
    need(start["runner_sha256"] == RUNNER_SHA and start["protocol_sha256"] == PROTOCOL_SHA
         and start["prior_context_result_sha256"] == PRIOR_SHA, "Start provenance differs")
    raw_paths, transport = verify_transport(args, plan, base, result)
    arrays, overlap = {}, []
    artifacts = {r["id"]: r for r in result["arrays"]}
    need(set(artifacts) == set(IDS) and len(result["arrays"]) == 2, "Unexpected array outputs")
    for item in plan["prior_arrays"]:
        i = item["id"]
        old_path = source_path(args.native_grid_cache, item["file"])
        need(sha(old_path) == item["sha256"], "Old anatomy array changed")
        record = artifacts[i]
        need(record["file"] == f"context81_{i}.npz", "Unexpected new array filename")
        new_path = source_path(args.run_dir, record["file"])
        need(sha(new_path) == record["sha256"], "New anatomy array hash mismatch")
        with np.load(new_path, allow_pickle=False) as archive:
            values = {arm: np.asarray(archive[arm], float) for arm in (*ARMS, "coarse_neighborhood")}
            c = np.asarray(archive["center_xyz"])
        expected_center = next(x["center_xyz"] for x in plan["centers"] if x["id"] == i)
        need(np.array_equal(c, expected_center), "New center changed")
        need(values["coarse_neighborhood"].shape == (83,) * 3 and all(values[arm].shape == (81,) * 3 for arm in ARMS),
             "New array support differs")
        need(all(np.isfinite(a).all() for a in values.values()), "Nonfinite new arrays")
        with np.load(old_path, allow_pickle=False) as old:
            need(np.array_equal(old["coarse_center_xyz"], c)
                 and np.array_equal(old["coarse_center_xyz"] - old["coarse_neighborhood_origin_xyz"], [17, 17, 17]),
                 "Old native center/origin differs")
            errors = {}
            for arm in (*ARMS, "coarse_neighborhood"):
                side = 35 if arm == "coarse_neighborhood" else 33
                need(old[arm].shape == (side,) * 3, "Old array support differs")
                errors[arm] = float(np.max(np.abs(center(values[arm], side) - old[arm])))
        passed = errors["coarse_neighborhood"] == 0 and all(errors[arm] <= 1e-8 for arm in ARMS)
        overlap.append({"passed": passed, "maximum_absolute_errors": errors,
                        "tolerance_fine_counts": 1e-8, "native_requires_exact": True, "id": i})
        arrays[i] = values
    same(result["overlap_gates"], overlap, "overlap")
    same(json.loads((args.run_dir / "overlap_gates.json").read_bytes()), overlap, "overlap_file")
    need(all(row["passed"] for row in overlap), "Primary overlap failed; full interpretation verification stops")
    need(result["outcome"] == "completed_fixed_context_comparison", "Unexpected primary outcome")
    for key in ("adaptive_pair", "centers_and_affine_unchanged"):
        need(result[key] is True, f"Missing provenance flag: {key}")
    for key in ("labels_or_models_accessed", "registration_fitted", "material_or_same_sheet_identity_established"):
        need(result[key] is False, f"Unexpected scientific claim: {key}")
    expected = recompute_metrics(arrays, check_cpu)
    maximum_correlation_error = 0.
    correlation_count = 0
    for name, rows in expected.items():
        same(result[name], rows, name)
        if name != "fixed65_structure":
            for actual, independent in zip(result[name], rows):
                for metric in ("pearson", "spearman"):
                    correlation_count += 1
                    if independent[metric] is not None:
                        maximum_correlation_error = max(maximum_correlation_error, abs(actual[metric] - independent[metric]))
    need(correlation_count == 896 and len(expected["fixed65_structure"]) == 18, "Metric counts differ")
    reader = RawReader(raw_paths)
    raw_check = verify_raw_samples(arrays, plan, base, reader)
    write(args.review_dir / "fixed_raw_sample_checks.json", raw_check)
    need(raw_check["all_checks_passed"], "A predefined raw sample failed its declared tolerance")
    check_cpu()
    return {"schema": "frag5-context81-independent-result-review-v1", "status": "passed",
            "freeze_reference": FREEZE, "checker_sha256": sha(__file__), "runner_sha256": RUNNER_SHA,
            "protocol_sha256": PROTOCOL_SHA, "result_sha256": sha(args.run_dir / "result.json"),
            "start_sha256": sha(args.run_dir / "start.json"),
            "acquisition_sha256": sha(args.run_dir / "acquisition.json"),
            "admission_sha256": sha(args.run_dir / "heads_before_new_data.json"),
            "prior_context_result_sha256": PRIOR_SHA, "transport": transport,
            "overlap_gates_recomputed": overlap, "new_arrays_hash_verified": result["arrays"],
            "nested_rows_recomputed": len(expected["nested_contexts_all_shifts"]),
            "smoothed_rows_recomputed": len(expected["fixed65_smoothing_zero_shift"]),
            "outer_shell_rows_recomputed": len(expected["outer_shell_zero_shift"]),
            "correlations_recomputed": correlation_count,
            "maximum_correlation_absolute_error": maximum_correlation_error,
            "structure_records_recomputed": len(expected["fixed65_structure"]),
            "raw_sample_checks_file": "fixed_raw_sample_checks.json",
            "raw_sample_checks_sha256": sha(args.review_dir / "fixed_raw_sample_checks.json"),
            "raw_positions_checked": 30, "raw_point_tolerance_counts": RAW_TOLERANCE,
            "maximum_manual_point_absolute_error": raw_check["maximum_point_absolute_error"],
            "maximum_manual_footprint_absolute_error": raw_check["maximum_footprint_absolute_error"],
            "maximum_manual_native_integer_error": raw_check["maximum_native_absolute_error"],
            "scientific_runner_imported": False, "new_network_bytes": 0,
            "labels_or_models_accessed": False, "cpu_seconds": time.process_time() - cpu_start,
            "limits": ["All896 correlations and18 structure records were recomputed; only30 fixed raw integrations were checked.",
                       "Raw coordinates use independent affine solves and manual8-corner sums;1e-6 count tolerance was declared before outcomes.",
                       "The original1e-8 fine overlap tolerance and exact native overlap requirement remain unchanged.",
                       "Verifies existing acquisition records/caches without recontacting source servers.",
                       "Does not certify material, physical sheet, face, labels, detector behavior or ink improvement."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "runner", "stage1-protocol", "prior-context-result", "stage1-cache", "native-grid-cache", "run-dir", "review-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.review_dir.exists():
        parser.error("Choose a fresh review directory; prior attempts are preserved")
    args.review_dir.mkdir(parents=True)
    cpu_start = time.process_time()
    write(args.review_dir / "review_start.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                                "checker_sha256": sha(__file__), "raw_tolerance_counts": RAW_TOLERANCE,
                                                "network_allowed": False})
    try:
        receipt = review(args, cpu_start)
        write(args.review_dir / "independent_result_review.json", receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "transport"}, indent=2))
    except Exception as exc:
        write(args.review_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                 "traceback": traceback.format_exc(),
                                                 "cpu_seconds": time.process_time() - cpu_start})
        raise


if __name__ == "__main__":
    main()
