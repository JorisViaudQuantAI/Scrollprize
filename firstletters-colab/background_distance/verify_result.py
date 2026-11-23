#!/usr/bin/env python3
"""Independent, single-thread review of the completed boundary diagnostic.

No runner helper is imported. Original-label distances are obtained by exhaustive
local cell enumeration in rationally derived 1/50-micrometre coordinates, not a
nearest-neighbour tree. Only the two authorised old public controls are read.
No CT, glyph archive, detector, checkpoint, or network access is involved.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "5b48d79fde5a41f8239b8548f1609d4349de70ff"
SCROLLS = ("PHerc0841", "PHerc0814")
FILES = ("protocol.json", "run.py", "test_run.py", "README.md")
COMMON = Fraction(48, 5)
LABEL = Fraction(234, 25)
RADII = tuple(Fraction(str(value)) for value in (9.6, 19.2, 38.4, 76.8))
SCALE = math.lcm(2 * COMMON.denominator, 2 * LABEL.denominator)  # 50, not producer's25
SIDE = 256
ENDPOINTS = ("own_float", "own_q8", "cross_float", "cross_q8", "float_ge_0_99", "q8_equal_one")


class ReviewFailure(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise ReviewFailure(reason)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outside_git(path):
    path = Path(path).expanduser().resolve()
    require(not any(os.path.lexists(parent / ".git") for parent in (path, *path.parents)), "Review evidence must stay outside Git")
    return path


class Compare:
    def __init__(self):
        self.count = 0
        self.max_float_difference = 0.0

    def check(self, expected, observed, where):
        if isinstance(expected, dict):
            require(isinstance(observed, dict) and set(expected) == set(observed), where + ": mapping keys differ")
            for key in expected:
                self.check(expected[key], observed[key], where + "/" + key)
        elif isinstance(expected, (tuple, list)):
            require(isinstance(observed, (tuple, list)) and len(expected) == len(observed), where + ": sequence length differs")
            for i, (a, b) in enumerate(zip(expected, observed)):
                self.check(a, b, where + "/" + str(i))
        elif isinstance(expected, (int, bool, str)) or expected is None:
            require(expected == observed, where + ": exact value differs")
            self.count += 1
        else:
            require(isinstance(observed, (int, float)), where + ": numeric value absent")
            difference = abs(float(expected) - float(observed))
            require(math.isfinite(difference) and difference <= 1e-14, where + ": numeric value differs")
            self.max_float_difference = max(self.max_float_difference, difference)
            self.count += 1


def exhaustive_label_bands(original, origin, declared_halo, check):
    """Every potential source centre within the maximum radius is enumerated."""
    query_um = [[(Fraction(o + i) + Fraction(1, 2)) * COMMON for i in range(SIDE)] for o in origin]
    label_index = [np.array([math.floor(q / LABEL) for q in axis], np.int64) for axis in query_um]
    query_units = [np.array([int(q * SCALE) for q in axis], np.int64) for axis in query_um]
    halo = [[math.floor((o * COMMON - RADII[-1] - LABEL) / LABEL),
             math.ceil(((o + SIDE) * COMMON + RADII[-1] + LABEL) / LABEL)] for o in origin]
    require(halo == declared_halo, "Independent rational halo differs")
    require(all(0 <= a < b <= size for (a, b), size in zip(halo, original.shape)), "Full original-label halo unavailable")
    core = original[np.ix_(*label_index)]
    (y0, y1), (x0, x1) = halo
    crop = original[y0:y1, x0:x1]
    radii_squared = np.array([int(r * SCALE) ** 2 for r in RADII], np.int64)
    cap = int(radii_squared[-1]) + 1
    nearest = {value: np.full((SIDE, SIDE), cap, np.int64) for value in (0, 2)}
    steps = math.ceil(RADII[-1] / LABEL + Fraction(1, 2))
    # Every omitted centre is at least (steps+0.5)*LABEL away along one axis.
    require((Fraction(steps) + Fraction(1, 2)) * LABEL > RADII[-1], "Enumeration radius is insufficient")
    pitch = int(LABEL * SCALE)
    half = int(LABEL * SCALE / 2)
    for dy in range(-steps, steps + 1):
        check()
        ys = label_index[0] + dy
        require(ys.min() >= 0 and ys.max() < original.shape[0], "Enumeration leaves source label canvas")
        delta_y_squared = (query_units[0] - (ys * pitch + half)) ** 2
        for dx in range(-steps, steps + 1):
            xs = label_index[1] + dx
            require(xs.min() >= 0 and xs.max() < original.shape[1], "Enumeration leaves source label canvas")
            delta_x_squared = (query_units[1] - (xs * pitch + half)) ** 2
            squared = delta_y_squared[:, None] + delta_x_squared[None, :]
            outside_halo = ((ys < y0) | (ys >= y1))[:, None] | ((xs < x0) | (xs >= x1))[None, :]
            require(not np.any(outside_halo & (squared <= radii_squared[-1])), "A tested-radius centre escapes the declared halo")
            category = original[np.ix_(ys, xs)]
            for value in (0, 2):
                np.minimum(nearest[value], squared, out=nearest[value], where=category == value)
    # Counting exceeded thresholds is independent of production searchsorted.
    bands = {value: (distance[:, :, None] > radii_squared).sum(axis=2).astype(np.uint8)
             for value, distance in nearest.items()}
    geometry = {"halo_yx_halfopen": halo, "halo_shape_yx": list(crop.shape),
                "complete_original_label_halo": True,
                "ink_reference_centres": int(np.count_nonzero(crop == 2)),
                "unknown_reference_centres": int(np.count_nonzero(crop == 0))}
    method = {"integer_coordinate_unit_um": float(Fraction(1, SCALE)),
              "enumerated_offsets_per_axis": list(range(-steps, steps + 1)),
              "source_centres_considered_per_query": (2 * steps + 1) ** 2,
              "omitted_nearest_possible_axis_distance_um": float((Fraction(steps) + Fraction(1, 2)) * LABEL),
              "all_possible_tested_radius_centres_inside_frozen_halo": True}
    return core, bands[2], bands[0], geometry, method


def exhaustive_support_bands(support):
    """Observed-hole offsets only; outside-core positions are not references."""
    step = RADII[-1] / COMMON
    require(step.denominator == 1, "Support radius does not match the fixed physical contract")
    radius = int(step)
    pixel_edges = np.array([int(r / COMMON) ** 2 for r in RADII], np.int64)
    minimum = np.full(support.shape, radius * radius + 1, np.int64)
    if (~support).any():
        observed_holes = np.pad(~support, radius, constant_values=False)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                squared = dy * dy + dx * dx
                if squared > radius * radius:
                    continue
                holes = observed_holes[radius+dy:radius+dy+SIDE, radius+dx:radius+dx+SIDE]
                np.minimum(minimum, squared, out=minimum, where=holes)
    bins = (minimum[:, :, None] > pixel_edges).sum(axis=2).astype(np.uint8)
    yy, xx = np.indices(support.shape)
    # Centre-to-domain-edge distances are half-integer multiples of COMMON.
    censored = (yy < radius) | (xx < radius) | (yy >= SIDE-radius) | (xx >= SIDE-radius)
    return bins, censored


def positives(float_map, q8_map, field):
    f, q = float_map.astype(np.float64), q8_map.astype(np.float64)
    return {"own_float": f > field["own_cutoffs"]["float"]["threshold_strict_greater"],
            "own_q8": q > field["own_cutoffs"]["q8"]["threshold_strict_greater"],
            "cross_float": f > field["cross_cutoffs"]["float"]["threshold_strict_greater"],
            "cross_q8": q > field["cross_cutoffs"]["q8"]["threshold_strict_greater"],
            "float_ge_0_99": f >= .99, "q8_equal_one": q == 1.0}


def independent_population(mask, hits, totals):
    denominator = int(np.count_nonzero(mask))
    items = {}
    for name in ENDPOINTS:
        count = int(np.count_nonzero(hits[name][mask]))
        items[name] = {"positive_count": count,
                       "positive_rate": count / denominator if denominator else None,
                       "share_of_all_endpoint_positives": count / totals[name] if totals[name] else None}
    return {"background_denominator": denominator, "endpoints": items}


def independent_partition(background, hits, assignment, names):
    require(assignment.shape == background.shape and np.isin(assignment, np.arange(len(names))).all(),
            "Independent partition has an unassigned cell")
    totals = {name: int(np.count_nonzero(background & hit)) for name, hit in hits.items()}
    rows = [{"category": name, **independent_population(background & (assignment == index), hits, totals)}
            for index, name in enumerate(names)]
    require(sum(row["background_denominator"] for row in rows) == int(background.sum()), "Independent partition lost background")
    for name, count in totals.items():
        require(sum(row["endpoints"][name]["positive_count"] for row in rows) == count,
                "Independent partition lost a positive")
    return rows


def run(args):
    started = time.monotonic()
    primary_dir, diagnostic_dir, output = map(outside_git, (args.primary_dir, args.diagnostic_dir, args.output))
    require(not output.exists(), "Retain existing reviews; choose a new output file")
    protocol = read_json(HERE / "protocol.json")
    result_path = diagnostic_dir / "result.json"
    result = read_json(result_path)
    require(result["status"] == "complete" and result["freeze_reference"] == FREEZE, "Wrong or incomplete frozen diagnostic")
    for name in FILES:
        path = HERE / name
        frozen = subprocess.check_output(["git", "show", f"{FREEZE}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(path.read_bytes() == frozen and sha(path) == result["file_sha256"][name], "Frozen diagnostic source changed")
    for name, digest in protocol["dependency_sha256"].items():
        require(sha(ROOT / name) == digest, "A frozen provenance dependency changed")
    require(protocol["geometry"]["band_edges_um"] == [float(v) for v in RADII], "Physical bands changed")
    verified, consumed = {}, []

    def check():
        if time.monotonic() - started > 60:
            raise TimeoutError("Independent review time cap")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1048576:
            raise MemoryError("Independent review RSS cap")

    def verify(path, digest, role=None, canonical_url=None):
        check()
        path = outside_git(path)
        require(path.is_file() and sha(path) == digest, "Consumed source identity differs")
        verified[path] = digest
        if role:
            consumed.append({"role": role, "sha256": digest, "bytes": path.stat().st_size,
                             **({"canonical_url": canonical_url} if canonical_url else {})})
        return path

    alarm_before = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Independent review time cap")))
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        verify(result_path, sha(result_path))
        consumed_path = diagnostic_dir / "consumed_assets.json"
        verify(consumed_path, sha(consumed_path))
        pins = protocol["inputs"]
        primary_path = verify(primary_dir / "result.json", pins["primary_result_sha256"], "primary_result")
        source_path = verify(primary_dir / "source_receipt.json", pins["primary_source_receipt_sha256"], "primary_source_receipt")
        secondary_path = verify(diagnostic_dir / "inherited_secondary_result.json", pins["secondary_result_sha256"], "secondary_result")
        verify(diagnostic_dir / "inherited_primary_result.json", pins["primary_result_sha256"])
        verify(diagnostic_dir / "inherited_primary_source_receipt.json", pins["primary_source_receipt_sha256"])
        primary, secondary, source_receipt = map(read_json, (primary_path, secondary_path, source_path))
        require(primary["status"] == secondary["status"] == "complete", "Inherited source incomplete")
        compare = Compare()
        expected_gates = {"primary_numeric_gate_pass": primary["numeric_gate_pass"],
                          "primary_fields": {row["scroll"]: row["gate"] for row in primary["fields"]},
                          "secondary_numeric_gate_pass": secondary["numeric_gate_pass"],
                          "secondary_transfers": [{key: row[key] for key in ("calibration_scroll", "evaluation_scroll", "representation", "gate")}
                                                  for row in secondary["transfers"]]}
        compare.check(expected_gates, result["inherited_gate_objects"], "unchanged gates")
        reported = {row["scroll"]: row for row in result["fields"]}
        require(len(result["fields"]) == 2 and set(reported) == set(SCROLLS), "Diagnostic field population differs")
        reviewed = []
        for field in protocol["fields"]:
            scroll, original = field["scroll"], field["original_label"]
            old = reported[scroll]
            entries = [item for item in source_receipt["completed_assets"] if item["url"] == original["label_url"]]
            require(len(entries) == 1 and entries[0]["sha256"] == original["label_sha256"], "Original label receipt differs")
            name = hashlib.sha256(original["label_url"].encode()).hexdigest()
            label_path = verify(primary_dir / "source_cache" / name, original["label_sha256"], scroll + "/original_label", original["label_url"])
            require(label_path.stat().st_size == original["label_size_bytes"], "Original label size differs")
            with Image.open(label_path) as image:
                categorical = np.array(image)
            require(categorical.shape == tuple(original["label_shape_yx"]) and categorical.dtype == np.uint8
                    and np.isin(categorical, (0, 1, 2)).all(), "Original label schema differs")
            arrays = {}
            for name, digest in field["primary_array_sha256"].items():
                path = verify(primary_dir / scroll / name, digest, scroll + "/" + name)
                arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
                require(arrays[name].shape == (SIDE, SIDE), "Wrong retained common geometry")
            labels, support = arrays["labels.npy"], arrays["common_support.npy"]
            require(labels.dtype == np.uint8 and support.dtype == bool, "Wrong saved label/support dtype")
            core, ink_bins, unknown_bins, geometry, method = exhaustive_label_bands(
                categorical, field["core_origin_target_yx"], field["label_halo_yx_halfopen"], check)
            require(np.array_equal(core, labels), "Independent rational label-centre mapping differs")
            hole_bins, censored = exhaustive_support_bands(support)
            geometry.update({"unsupported_core_reference_centres": int(np.count_nonzero(~support)),
                             "source_acquisition_revalidated_from_CT": False,
                             "support_is_inherited_nonzero_column_mask": True,
                             "support_beyond_scoring_core_observed": False,
                             "all_scoring_cells_support_distance_censored": int(censored.sum()),
                             "no_cells_removed_for_censorship": True})
            compare.check(geometry, old["geometry"], scroll + "/geometry")
            robust = (ink_bins == 4) & (unknown_bins == 4) & (hole_bins == 4) & ~censored
            status = np.where(censored, 1, np.where(robust, 0, 2)).astype(np.uint8)
            f, q = arrays["hann_float_common.npy"], arrays["hann_uint8_common.npy"]
            for score in (f, q):
                require(score.dtype == np.float32 and np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all(), "Bad saved probability")
            hits = positives(f, q, field)
            background, ink = support & (labels == 1), support & (labels == 2)
            counts = {"background": int(background.sum()), "ink": int(ink.sum()), "known": int(background.sum() + ink.sum())}
            compare.check(counts, old["counts"], scroll + "/counts")
            compare.check(field["expected_counts"], counts, scroll + "/original counts retained")
            compare.check(field["own_cutoffs"], old["unchanged_own_cutoffs"], scroll + "/own cutoffs unchanged")
            compare.check(field["cross_cutoffs"], old["unchanged_cross_cutoffs"], scroll + "/cross cutoffs unchanged")
            positive_totals = {name: int(np.count_nonzero(background & mask)) for name, mask in hits.items()}
            totals = independent_population(background, hits, positive_totals)
            compare.check(totals, old["all_background_totals"], scroll + "/all background totals")
            ink_counts = {name: int(np.count_nonzero(ink & hits[name])) for name in ENDPOINTS[:4]}
            compare.check(ink_counts, old["unchanged_ink_positive_counts"], scroll + "/ink totals unchanged")
            for kind in ("own", "cross"):
                for representation in ("float", "q8"):
                    endpoint = kind + "_" + representation
                    require(positive_totals[endpoint] == field[kind + "_cutoffs"][representation]["background_positive_count"]
                            and ink_counts[endpoint] == field[kind + "_cutoffs"][representation]["ink_positive_count"],
                            "Inherited endpoint count did not survive")
            assignments = {"distance_to_known_ink_label": ink_bins,
                           "distance_to_unknown_label": unknown_bins,
                           "observed_support_hole_distance": hole_bins,
                           "support_distance_censorship": censored.astype(np.uint8),
                           "robust_interior_status": status}
            names = {key: protocol["geometry"]["band_names"] for key in list(assignments)[:3]}
            names["support_distance_censorship"] = ["uncensored_within_76.8um", "boundary_censored_within_76.8um"]
            names["robust_interior_status"] = ["beyond_76.8um_from_all_observed_boundaries_uncensored",
                                                "support_boundary_censored", "remaining_uncensored_cells"]
            partitions = {key: independent_partition(background, hits, assignment, names[key]) for key, assignment in assignments.items()}
            compare.check(partitions, old["partitions"], scroll + "/all partitions")
            compare.check(int(labels.size - counts["known"]), old["unknown_or_unsupported"], scroll + "/original omitted population")
            compare.check(int((~support).sum()), old["unsupported_common_cells"], scroll + "/support count")
            require(old["all_partition_totals_reconcile"], "Original diagnostic did not claim complete accounting")
            reviewed.append({"scroll": scroll, "all_geometry_and_partition_checks_match": True,
                             "independent_method": method, "counts": counts,
                             "all_background_positive_counts": positive_totals,
                             "uncensored_far_from_all_observed_boundaries": partitions["robust_interior_status"][0],
                             "all_fixed_radius_bins_recomputed": True,
                             "all_original_totals_retained": True})
            del categorical, arrays, labels, support, core, ink_bins, unknown_bins, hole_bins, censored, robust, status, hits, f, q
            check()
        recorded_consumed = read_json(consumed_path)["assets"]
        require(len(consumed) == result["verified_consumed_asset_count"], "Consumed input count differs")
        compare.check(sorted(consumed, key=lambda row: row["role"]), sorted(recorded_consumed, key=lambda row: row["role"]), "consumed identities")
        require(all(sha(path) == digest for path, digest in verified.items()), "An input changed during independent review")
        receipt = {"schema": "scrollprize.background-distance-independent-review.v1", "status": "verified",
                   "source_freeze_reference": FREEZE, "reviewer_sha256": sha(__file__),
                   "diagnostic_result_sha256": verified[result_path], "consumed_assets_receipt_sha256": verified[consumed_path],
                   "production_runner_or_helpers_imported": False,
                   "nearest_neighbour_tree_used": False,
                   "method": "Exhaustive original-label cell offsets; Fraction-derived physical centres in1/50um units; exact squared comparisons; independent complete count partitions",
                   "compared_scalar_count": compare.count, "maximum_float_difference": compare.max_float_difference,
                   "source_files_verified_unchanged": len(verified), "inherited_gates_unchanged": True,
                   "all_original_background_cells_and_positives_retained": True, "fields": reviewed,
                   "CNN_forwards": 0, "CT_arrays_opened": 0, "new_network_requests": 0,
                   "glyph_arrays_images_or_actual_alpha_opened": False, "threads": 1,
                   "true_ink_adjudication": False, "wall_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)
        print(json.dumps({key: receipt[key] for key in ("status", "compared_scalar_count", "maximum_float_difference", "wall_seconds", "peak_rss_kib")}))
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, alarm_before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
