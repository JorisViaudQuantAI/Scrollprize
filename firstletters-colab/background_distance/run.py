#!/usr/bin/env python3
"""Frozen offline diagnosis of background-labelled Hann disagreement.

Original label centres, unchanged thresholds, complete populations. No detector,
CT read, network access, label repair, fitted threshold, or promotion decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import time

PROCESS_STARTED = time.monotonic()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ("protocol.json", "run.py", "test_run.py", "README.md")
SCROLLS = ("PHerc0841", "PHerc0814")
ENDPOINTS = ("own_float", "own_q8", "cross_float", "cross_q8", "float_ge_0_99", "q8_equal_one")
EDGES = np.array([240, 480, 960, 1920], dtype=np.int64)
SIDE = 256
COMMON_PITCH = 240
LABEL_PITCH = 234
HALO = 1920 + 234


class ContractError(RuntimeError):
    pass


class ResourceStop(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise ContractError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def outside_git(path):
    path = Path(path).expanduser().resolve()
    require(not any(os.path.lexists(parent / ".git") for parent in (path, *path.parents)),
            "Detailed input/output must remain outside every Git worktree")
    return path


def verify_freeze(reference, protocol):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", reference)), "A full published freeze commit is required")
    for name in FROZEN:
        path = HERE / name
        committed = subprocess.check_output(["git", "show", f"{reference}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(path.read_bytes() == committed, "A diagnostic file differs from its freeze")
    for name, digest in protocol["dependency_sha256"].items():
        require(sha(ROOT / name) == digest, "A pinned provenance dependency changed")


class Budget:
    def __init__(self, limits, started):
        self.limits, self.started = limits, started

    def check(self):
        if time.monotonic() - self.started >= self.limits["whole_run_wall_seconds"]:
            raise ResourceStop("whole_run_time_cap_reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > self.limits["peak_rss_soft_kib"]:
            raise ResourceStop("peak_RSS_soft_cap_reached")


class Inputs:
    def __init__(self, budget):
        self.budget, self.records = budget, {}

    def check(self, path, digest, role, size=None, canonical_url=None):
        self.budget.check()
        path = outside_git(path)
        require(path.is_file(), "A required cached input is absent")
        actual_size = path.stat().st_size
        require(size is None or actual_size == size, "A cached input size differs")
        require(sha(path) == digest, "A consumed input hash differs: " + role)
        self.records[path] = {"role": role, "sha256": digest, "bytes": actual_size,
                              **({"canonical_url": canonical_url} if canonical_url else {})}
        self.budget.check()
        return path

    def final_check(self):
        for path, record in self.records.items():
            self.budget.check()
            require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                    "An input changed during the diagnostic")


def halo_bounds(origin, label_shape, core_shape=(SIDE, SIDE)):
    """Complete source cells covering the core plus the fixed physical halo."""
    require(len(origin) == len(label_shape) == len(core_shape) == 2, "Wrong halo dimensions")
    bounds = []
    for o, size, source_size in zip(origin, core_shape, label_shape):
        require(isinstance(o, (int, np.integer)) and o >= 0, "Invalid common-grid origin")
        low = (int(o) * COMMON_PITCH - HALO) // LABEL_PITCH
        high = -(-((int(o) + int(size)) * COMMON_PITCH + HALO) // LABEL_PITCH)
        require(0 <= low < high <= source_size, "Complete original-label halo unavailable")
        bounds.append([low, high])
    return bounds


def common_points(origin, shape=(SIDE, SIDE)):
    ys = (2 * (int(origin[0]) + np.arange(shape[0], dtype=np.int64)) + 1) * 120
    xs = (2 * (int(origin[1]) + np.arange(shape[1], dtype=np.int64)) + 1) * 120
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack((yy.ravel(), xx.ravel()))


def project_categories(original, origin, shape=(SIDE, SIDE)):
    indices = common_points(origin, shape) // LABEL_PITCH
    require((indices >= 0).all() and (indices[:, 0] < original.shape[0]).all()
            and (indices[:, 1] < original.shape[1]).all(), "Categorical mapping leaves source")
    return original[indices[:, 0], indices[:, 1]].reshape(shape)


def label_points(original_crop, source_origin, category):
    indices = np.argwhere(original_crop == category).astype(np.int64)
    indices += np.asarray(source_origin, np.int64)
    return (2 * indices + 1) * 117


def squared_distance_to_centres(queries, references):
    """Nearest-centre query; exact integer differences decide physical bands."""
    queries, references = np.asarray(queries), np.asarray(references)
    require(queries.ndim == references.ndim == 2 and queries.shape[1] == references.shape[1] == 2,
            "Centres must be YX coordinate pairs")
    require(queries.dtype == references.dtype == np.dtype(np.int64), "Centres must use exact integer units")
    require(np.max(np.abs(queries), initial=0) < 100_000_000
            and np.max(np.abs(references), initial=0) < 100_000_000, "Coordinates exceed safe squared-integer range")
    if not len(references):
        return np.full(len(queries), -1, np.int64)  # no class centre within the complete halo
    from scipy.spatial import cKDTree
    tree = cKDTree(references.astype(np.float64), balanced_tree=True, compact_nodes=True)
    _, nearest = tree.query(queries.astype(np.float64), k=1, eps=0, workers=1)
    delta = queries - references[nearest]
    return (delta * delta).sum(axis=1, dtype=np.int64)


def distance_bins(squared):
    squared = np.asarray(squared)
    require(squared.dtype == np.dtype(np.int64), "Squared distances must remain exact integers")
    require((squared >= -1).all(), "Unexpected absent-distance encoding")
    result = np.searchsorted(EDGES * EDGES, squared, side="left").astype(np.uint8)
    result[squared == -1] = 4
    return result


def support_boundary_censored(shape=(SIDE, SIDE)):
    yy, xx = np.indices(shape, dtype=np.int64)
    distance = np.minimum.reduce(((2 * yy + 1) * 120,
                                  (2 * (shape[0] - yy) - 1) * 120,
                                  (2 * xx + 1) * 120,
                                  (2 * (shape[1] - xx) - 1) * 120))
    return distance <= EDGES[-1]


def label_geometry(original, origin, support):
    require(original.dtype == np.uint8 and original.ndim == 2 and np.isin(original, (0, 1, 2)).all(),
            "Unexpected original categorical source")
    require(support.shape == (SIDE, SIDE) and support.dtype == bool, "Unexpected saved common support")
    bounds = halo_bounds(origin, original.shape)
    (y0, y1), (x0, x1) = bounds
    crop = original[y0:y1, x0:x1]
    query = common_points(origin)
    ink_points = label_points(crop, (y0, x0), 2)
    unknown_points = label_points(crop, (y0, x0), 0)
    unsupported_points = query[~support.ravel()]
    ink = distance_bins(squared_distance_to_centres(query, ink_points)).reshape(support.shape)
    unknown = distance_bins(squared_distance_to_centres(query, unknown_points)).reshape(support.shape)
    holes = distance_bins(squared_distance_to_centres(query, unsupported_points)).reshape(support.shape)
    censored = support_boundary_censored()
    robust = (ink == 4) & (unknown == 4) & (holes == 4) & ~censored
    status = np.full(support.shape, 2, np.uint8)
    status[censored] = 1
    status[robust] = 0
    return {"distance_to_known_ink_label": ink,
            "distance_to_unknown_label": unknown,
            "observed_support_hole_distance": holes,
            "support_distance_censorship": censored.astype(np.uint8),
            "robust_interior_status": status}, {
                "halo_yx_halfopen": bounds, "halo_shape_yx": list(crop.shape),
                "complete_original_label_halo": True,
                "ink_reference_centres": len(ink_points), "unknown_reference_centres": len(unknown_points),
                "unsupported_core_reference_centres": len(unsupported_points),
                "source_acquisition_revalidated_from_CT": False,
                "support_is_inherited_nonzero_column_mask": True,
                "support_beyond_scoring_core_observed": False,
                "all_scoring_cells_support_distance_censored": int(censored.sum()),
                "no_cells_removed_for_censorship": True}


def endpoint_masks(float_score, q8_score, own, cross):
    require(float_score.shape == q8_score.shape == (SIDE, SIDE), "Score geometry differs")
    for value in (float_score, q8_score):
        require(value.dtype == np.float32 and np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(),
                "Scores must be finite saved float32 probabilities")
    f64, q64 = float_score.astype(np.float64), q8_score.astype(np.float64)
    for cutoffs in (own, cross):
        for representation in ("float", "q8"):
            cutoff = cutoffs[representation]["threshold_strict_greater"]
            require(isinstance(cutoff, (int, float)) and np.isfinite(cutoff) and 0 <= cutoff <= 1,
                    "Invalid frozen probability cutoff")
    return {"own_float": f64 > own["float"]["threshold_strict_greater"],
            "own_q8": q64 > own["q8"]["threshold_strict_greater"],
            "cross_float": f64 > cross["float"]["threshold_strict_greater"],
            "cross_q8": q64 > cross["q8"]["threshold_strict_greater"],
            "float_ge_0_99": f64 >= .99,
            "q8_equal_one": q64 == 1.0}


def population_summary(population, endpoints, whole_positive):
    n = int(population.sum())
    counts = {name: int(np.count_nonzero(population & value)) for name, value in endpoints.items()}
    return {"background_denominator": n,
            "endpoints": {name: {"positive_count": count,
                                  "positive_rate": count / n if n else None,
                                  "share_of_all_endpoint_positives": count / whole_positive[name] if whole_positive[name] else None}
                          for name, count in counts.items()}}


def partition_summary(background, endpoints, assignments, names):
    require(assignments.shape == background.shape and assignments.dtype.kind in "iu", "Invalid partition geometry")
    require((assignments >= 0).all() and (assignments < len(names)).all(), "Partition leaves cells unassigned")
    positives = {name: int(np.count_nonzero(background & value)) for name, value in endpoints.items()}
    entries = [{"category": name, **population_summary(background & (assignments == index), endpoints, positives)}
               for index, name in enumerate(names)]
    require(sum(row["background_denominator"] for row in entries) == int(background.sum()), "A partition lost background cells")
    for name, total in positives.items():
        require(sum(row["endpoints"][name]["positive_count"] for row in entries) == total,
                "A partition lost original background-labelled positives")
    return entries


def read_array(path, dtype):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.shape == (SIDE, SIDE) and value.dtype == np.dtype(dtype), "Saved common array dtype/shape differs")
    return value


def validate_provenance(protocol, primary, secondary, source_receipt):
    pins = protocol["inputs"]
    require(primary["status"] == secondary["status"] == "complete", "Complete inherited experiments are required")
    require(primary["freeze_reference"] == pins["primary_freeze_reference"]
            and secondary["freeze_reference"] == pins["secondary_freeze_reference"], "Inherited freeze differs")
    require(primary["source_receipt_sha256"] == pins["primary_source_receipt_sha256"]
            and secondary["primary_result_sha256"] == pins["primary_result_sha256"], "Inherited receipt chain differs")
    require(primary["forwards_completed"] == 128 and primary["all_complete_field_hashes_match"], "Primary acquisition/inference incomplete")
    require(not source_receipt["failed_assets"], "Primary source receipt contains acquisition failures")
    rows = {row["scroll"]: row for row in primary["fields"]}
    require(len(primary["fields"]) == 2 and set(rows) == set(SCROLLS), "Unexpected primary fields")
    manifest = read_json(ROOT / "firstletters-colab/transversal_lab/hecate_donor_manifest.json")
    donors = {d["scroll"]: d for d in manifest["donors"]}
    selection = read_json(ROOT / "firstletters-colab/native_resolution/selection.json")
    choices = {row["scroll"]: row["choice"] for row in selection["results"]}
    require([f["scroll"] for f in protocol["fields"]] == list(SCROLLS), "Frozen field order differs")
    for field in protocol["fields"]:
        scroll, row = field["scroll"], rows[field["scroll"]]
        require(field["core_origin_target_yx"] == choices[scroll]["core_origin_target_yx"], "Inherited physical field moved")
        require(all(donors[scroll][key] == value for key, value in field["original_label"].items()), "Original-label provenance differs")
        require(halo_bounds(field["core_origin_target_yx"], field["original_label"]["label_shape_yx"])
                == field["label_halo_yx_halfopen"], "Frozen source-label halo differs")
        for name, digest in field["primary_array_sha256"].items():
            require(row["output_sha256"][name] == digest, "Inherited primary array identity differs")
        for representation, key in (("float", "float"), ("uint8", "q8")):
            metric = row["metrics"][representation]["hann"]
            require(all(metric[name] == value for name, value in field["expected_counts"].items()), "Inherited class counts differ")
            require(all(metric[name] == value for name, value in field["own_cutoffs"][key].items()), "Own-field cutoff/counts changed")
            other = field["cross_cutoffs"][key]["calibration_scroll"]
            matches = [t for t in secondary["transfers"] if t["calibration_scroll"] == other
                       and t["evaluation_scroll"] == scroll and t["representation"] == representation]
            require(len(matches) == 1, "Cross-donor transfer identity differs")
            require(all(matches[0]["evaluation"]["hann"][name] == value
                        for name, value in field["cross_cutoffs"][key].items() if name != "calibration_scroll"),
                    "Cross-donor cutoff/counts changed")
            require(field["cross_cutoffs"][key]["threshold_strict_greater"]
                    == rows[other]["metrics"][representation]["hann"]["threshold_strict_greater"],
                    "Cross-donor cutoff is not the other donor's unchanged operating point")
    return rows


def run(args):
    protocol = read_json(HERE / "protocol.json")
    # No inherited arrays, including predictions, are opened before this check.
    verify_freeze(args.freeze_reference, protocol)
    primary_dir, secondary_path, out = map(outside_git, (args.primary_dir, args.consensus_result, args.output_dir))
    require(not out.exists() and not out.is_relative_to(primary_dir) and not primary_dir.is_relative_to(out),
            "Choose a fresh diagnostic directory outside the primary tree")
    require(not secondary_path.is_relative_to(out), "Output cannot contain the inherited secondary")
    out.mkdir(parents=True)
    budget = Budget(protocol["limits"], PROCESS_STARTED)
    inputs = Inputs(budget)
    result = {"schema": "scrollprize.background-distance.v1", "status": "started", "fields": [],
              "freeze_reference": args.freeze_reference,
              "file_sha256": {name: sha(HERE / name) for name in FROZEN},
              "new_network_requests": 0, "new_CT_requests": 0, "CT_arrays_opened": 0,
              "new_detector_forwards": 0, "training_steps": 0,
              "diagnostic_only": True, "new_promotion_gate": None, "true_ink_adjudication": False,
              "labels_changed": False, "thresholds_refit": False, "primary_or_secondary_gate_changed": False,
              "model_exposure": protocol["reporting"]["model_exposure"]}
    save_json(out / "start.json", result)
    stage, code = "inherited_identity_validation", 1
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(ResourceStop("whole_run_time_cap_reached")))
    signal.setitimer(signal.ITIMER_REAL, max(.001, protocol["limits"]["whole_run_wall_seconds"] - (time.monotonic() - PROCESS_STARTED)))
    try:
        budget.check()
        pins = protocol["inputs"]
        primary_path = inputs.check(primary_dir / "result.json", pins["primary_result_sha256"], "primary_result")
        receipt_path = inputs.check(primary_dir / "source_receipt.json", pins["primary_source_receipt_sha256"], "primary_source_receipt")
        secondary_path = inputs.check(secondary_path, pins["secondary_result_sha256"], "secondary_result")
        primary, secondary, source_receipt = map(read_json, (primary_path, secondary_path, receipt_path))
        primary_fields = validate_provenance(protocol, primary, secondary, source_receipt)
        for source, name in ((primary_path, "inherited_primary_result.json"),
                             (receipt_path, "inherited_primary_source_receipt.json"),
                             (secondary_path, "inherited_secondary_result.json")):
            shutil.copyfile(source, out / name)
            require(sha(out / name) == inputs.records[source]["sha256"], "Preserved inherited receipt differs")
        result["inherited_result_sha256"] = {"primary": pins["primary_result_sha256"], "secondary": pins["secondary_result_sha256"]}
        result["inherited_gate_objects"] = {"primary_numeric_gate_pass": primary["numeric_gate_pass"],
                                            "primary_fields": {s: row["gate"] for s, row in primary_fields.items()},
                                            "secondary_numeric_gate_pass": secondary["numeric_gate_pass"],
                                            "secondary_transfers": [{key: t[key] for key in ("calibration_scroll", "evaluation_scroll", "representation", "gate")}
                                                                    for t in secondary["transfers"]]}
        for field in protocol["fields"]:
            scroll = field["scroll"]
            stage = scroll + "_label_geometry"
            original = field["original_label"]
            matches = [record for record in source_receipt["completed_assets"] if record["url"] == original["label_url"]]
            require(len(matches) == 1 and matches[0]["sha256"] == original["label_sha256"]
                    and matches[0]["bytes"] == original["label_size_bytes"], "Original categorical source lacks its inherited receipt")
            key = hashlib.sha256(original["label_url"].encode()).hexdigest()
            original_path = inputs.check(primary_dir / "source_cache" / key, original["label_sha256"],
                                         scroll + "/original_label", original["label_size_bytes"], original["label_url"])
            with Image.open(original_path) as source_image:
                labels_source = np.array(source_image)
            require(labels_source.shape == tuple(original["label_shape_yx"]), "Original categorical canvas changed")
            paths = {name: inputs.check(primary_dir / scroll / name, digest, scroll + "/" + name)
                     for name, digest in field["primary_array_sha256"].items()}
            labels = read_array(paths["labels.npy"], np.uint8)
            support = read_array(paths["common_support.npy"], bool)
            mapped = project_categories(labels_source, field["core_origin_target_yx"])
            require(np.array_equal(labels, mapped), "Original label-centre mapping does not reproduce inherited categories")
            partitions, geometry_receipt = label_geometry(labels_source, field["core_origin_target_yx"], support)
            require(geometry_receipt["halo_yx_halfopen"] == field["label_halo_yx_halfopen"], "Runtime halo changed")
            budget.check()
            stage = scroll + "_unchanged_endpoint_accounting"
            float_score = read_array(paths["hann_float_common.npy"], np.float32)
            q8_score = read_array(paths["hann_uint8_common.npy"], np.float32)
            endpoints = endpoint_masks(float_score, q8_score, field["own_cutoffs"], field["cross_cutoffs"])
            require(tuple(endpoints) == ENDPOINTS, "Endpoint population changed")
            background, ink = support & (labels == 1), support & (labels == 2)
            actual_counts = {"background": int(background.sum()), "ink": int(ink.sum()), "known": int(background.sum() + ink.sum())}
            require(actual_counts == field["expected_counts"], "Original scoring population changed")
            totals = {name: int(np.count_nonzero(background & positive)) for name, positive in endpoints.items()}
            positive_ink = {name: int(np.count_nonzero(ink & endpoints[name])) for name in ENDPOINTS[:4]}
            for kind, cutoffs in (("own", field["own_cutoffs"]), ("cross", field["cross_cutoffs"])):
                for representation in ("float", "q8"):
                    name = kind + "_" + representation
                    require(totals[name] == cutoffs[representation]["background_positive_count"]
                            and positive_ink[name] == cutoffs[representation]["ink_positive_count"],
                            "Original endpoint count was not preserved")
            for representation, score in (("float", float_score), ("q8", q8_score)):
                cutoff = field["own_cutoffs"][representation]
                require(int(np.count_nonzero(background & (score.astype(np.float64) == cutoff["threshold_strict_greater"])))
                        == cutoff["background_tied_at_threshold_count"], "Original own-field tie count changed")
            names = {name: protocol["geometry"]["band_names"] for name in list(partitions)[:3]}
            names["support_distance_censorship"] = ["uncensored_within_76.8um", "boundary_censored_within_76.8um"]
            names["robust_interior_status"] = ["beyond_76.8um_from_all_observed_boundaries_uncensored",
                                                "support_boundary_censored",
                                                "remaining_uncensored_cells"]
            summaries = {name: partition_summary(background, endpoints, assignments, names[name])
                         for name, assignments in partitions.items()}
            result["fields"].append({"scroll": scroll, "counts": actual_counts,
                                     "unknown_or_unsupported": int(labels.size - actual_counts["known"]),
                                     "unsupported_common_cells": int(np.count_nonzero(~support)),
                                     "geometry": geometry_receipt,
                                     "unchanged_own_cutoffs": field["own_cutoffs"], "unchanged_cross_cutoffs": field["cross_cutoffs"],
                                     "all_background_totals": population_summary(background, endpoints, totals),
                                     "unchanged_ink_positive_counts": positive_ink,
                                     "all_partition_totals_reconcile": True,
                                     "partitions": summaries})
            save_json(out / "progress.json", {"fields_completed": len(result["fields"]), "stage": stage,
                                               "wall_seconds": time.monotonic() - PROCESS_STARTED})
            del labels_source, labels, support, mapped, partitions, float_score, q8_score, endpoints, background, ink
            budget.check()
        stage = "final_integrity"
        require([row["scroll"] for row in result["fields"]] == list(SCROLLS), "A complete original field is missing")
        inputs.final_check()
        require(all(sha(HERE / name) == digest for name, digest in result["file_sha256"].items()), "Frozen code changed during analysis")
        result.update({"status": "complete", "source_artifacts_unchanged": True,
                       "verified_consumed_asset_count": len(inputs.records),
                       "every_original_background_cell_and_positive_retained": True,
                       "partition_rates_are_descriptive_not_independent_samples": True})
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        result.update({"status": "resource_unassessable" if isinstance(error, (ResourceStop, MemoryError)) else "failed",
                       "stage": stage, "error_type": type(error).__name__, "reason": str(error)[:500]})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result.update({"wall_seconds": time.monotonic() - PROCESS_STARTED,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        save_json(out / "consumed_assets.json", {"assets": list(inputs.records.values()), "source_paths_omitted": True})
        save_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "fields_completed": len(result["fields"]),
                      "new_detector_forwards": 0, "new_network_requests": 0,
                      "diagnostic_only": True, "wall_seconds": result["wall_seconds"]}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", required=True)
    parser.add_argument("--primary-dir", required=True, type=Path)
    parser.add_argument("--consensus-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
