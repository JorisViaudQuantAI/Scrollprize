#!/usr/bin/env python3
"""Independent, offline recomputation of the frozen Frag5 local diagnostics.

No scientific function is imported from the analysis runner. Filtering uses
full binomial coefficients and direct valid convolutions. Every correlation
uses explicitly selected values and a direct centered dot product, not FFTs.
The four input arrays remain adaptive development data, not new observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np


FREEZE = "32a13933a4e811973a79809f20c6dd54de19488f"
RUNNER_SHA = "65f9fe9b6bb50c6c4f3e98f03a43b5b3910d77f398c26ef5907fc9f75750c6f4"
PROTOCOL_SHA = "fbe930aec8c79d5141748d58198e59b2ce56e5adf553ad141c1b07ffe68793b0"
PRIOR_SHA = "c0e57d1deea4dabaea850ee7e0f111cf48a83f5df73482507c55d1647d7a6c7c"
IDS = (20, 25, 65, 70)
ARMS = ("fine_nominal_coarse_footprint", "fine_point")
FILTERS = (0, 1, 4)
MASKS = ("full", "negative_x_slab", "positive_x_slab")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def centered(a, side=17, xyz=(0, 0, 0)):
    """Independently index a centered XYZ-shifted cube in ZYX storage."""
    offsets = (xyz[2], xyz[1], xyz[0])
    slices = []
    for dim, shift in zip(a.shape, offsets):
        first = (dim - side) // 2 + shift
        need(dim % 2 == 1 and side % 2 == 1 and first >= 0 and first + side <= dim,
             "Centered support exceeds cached data")
        slices.append(slice(first, first + side))
    return a[tuple(slices)]


def binomial_valid(a, passes):
    coeff = np.array([math.comb(2 * passes, k) for k in range(2 * passes + 1)], float)
    coeff /= 4 ** passes
    result = np.asarray(a, dtype=np.float64).copy()
    for axis in range(3):
        result = np.apply_along_axis(lambda row: np.convolve(row, coeff, mode="valid"), axis, result)
    return result


def direct_correlation(a, b):
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    need(x.shape == y.shape and x.size >= 2, "Different correlation support")
    if x.max() == x.min() or y.max() == y.min():
        return None
    dx, dy = x - np.mean(x), y - np.mean(y)
    denominator = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
    return float(np.sum(dx * dy) / denominator) if denominator > 0 else None


def selectors():
    result = [np.ones((17, 17, 17), bool)]
    for columns in (slice(0, 4), slice(13, 17)):
        mask = np.zeros((17, 17, 17), bool)
        mask[:, :, columns] = True
        result.append(mask)
    return result


def direct_field(fine, coarse, mask):
    template = centered(fine)[mask]
    result = np.full((9, 9, 9), np.nan)
    for iz, iy, ix in itertools.product(range(9), repeat=3):
        candidate = centered(coarse, xyz=(ix - 4, iy - 4, iz - 4))[mask]
        value = direct_correlation(template, candidate)
        if value is not None:
            result[iz, iy, ix] = value
    return result


def peak(field):
    finite = np.isfinite(field)
    if not finite.any():
        return {"status": "undefined", "zero_shift": None, "maximum": None,
                "argmax_shift_xyz": None, "boundary_maximum": None,
                "within_0_01_of_maximum": 0, "within_0_05_of_maximum": 0}
    flat_index = int(np.argmax(np.where(finite, field, -np.inf).ravel()))
    iz, iy, ix = np.unravel_index(flat_index, field.shape)
    maximum = float(field[iz, iy, ix])
    zero = float(field[4, 4, 4]) if np.isfinite(field[4, 4, 4]) else None
    shift = [int(ix - 4), int(iy - 4), int(iz - 4)]
    return {"status": "assessable", "zero_shift": zero, "maximum": maximum,
            "argmax_shift_xyz": shift, "boundary_maximum": max(map(abs, shift)) == 4,
            "within_0_01_of_maximum": int(np.count_nonzero(finite & (field >= maximum - .01))),
            "within_0_05_of_maximum": int(np.count_nonzero(finite & (field >= maximum - .05)))}


def slab_summaries(fields):
    output = []
    for fit, check in ((1, 2), (2, 1)):
        selected = peak(fields[fit])
        if selected["status"] == "undefined":
            output.append({"fit": MASKS[fit], "check": MASKS[check], "status": "undefined"})
            continue
        x, y, z = selected["argmax_shift_xyz"]
        value = fields[check][z + 4, y + 4, x + 4]
        zero = fields[check][4, 4, 4]
        output.append({"fit": MASKS[fit], "check": MASKS[check],
                       "status": "descriptive_same_site_check",
                       "selected_shift_xyz": [x, y, z], "fit_score": selected["maximum"],
                       "check_score_at_selected_shift": float(value) if np.isfinite(value) else None,
                       "check_score_at_zero_shift": float(zero) if np.isfinite(zero) else None,
                       "boundary_selected": max(abs(x), abs(y), abs(z)) == 4})
    return output


def quality(raw, filtered, zero):
    center, before = centered(filtered), centered(raw)
    variance = 0. if center.max() == center.min() else float(np.var(center))
    prior_variance = 0. if before.max() == before.min() else float(np.var(before))
    lags, gradient = {}, []
    for axis, name in enumerate(("x", "y", "z")):
        unit = [0, 0, 0]
        unit[axis] = 1
        negative = [-v for v in unit]
        gradient.append((centered(filtered, xyz=unit) - centered(filtered, xyz=negative)).ravel() / 2)
        for lag in (1, 2, 4):
            shift = [lag * v for v in unit]
            lags[f"{name}_{lag}"] = direct_correlation(center, centered(filtered, xyz=shift))
    tensor = np.array([[np.mean(a * b) for b in gradient] for a in gradient])
    trace = float(tensor.trace())
    fractions = (np.linalg.eigvalsh(tensor)[::-1] / trace).tolist() if trace > 0 else None
    return {"samples": int(center.size), "mean": float(np.mean(center)), "sd": math.sqrt(variance),
            "recorded_reconstruction_zero": zero, "mean_minus_recorded_zero": float(np.mean(center) - zero),
            "variance_retention": variance / prior_variance if prior_variance > 0 else None,
            "lag_pearson_positive_xyz": lags, "structure_tensor_xyz": tensor.tolist(),
            "gradient_energy_counts_squared_per_grid_voxel_squared": trace,
            "tensor_eigenvalue_fractions_descending": fractions,
            "material_or_noise_classification": "not_identified"}


def same(actual, expected, label):
    """Recursive comparison: exact discrete values, tight float tolerance."""
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {label}")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"List size differs: {label}")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        need(isinstance(actual, (int, float)) and np.isfinite(actual)
             and np.isclose(actual, expected, rtol=2e-11, atol=2e-10),
             f"Numeric mismatch: {label}: actual={actual!r}, expected={expected!r}")
    else:
        need(actual == expected, f"Discrete mismatch: {label}: actual={actual!r}, expected={expected!r}")


def review(args, cpu_start):
    need(sha(args.runner) == RUNNER_SHA, "Runner differs from independently reviewed freeze")
    need(sha(args.protocol) == PROTOCOL_SHA, "Protocol differs from independently reviewed freeze")
    need(sha(args.prior_result) == PRIOR_SHA, "Published prior result changed")
    plan = json.loads(args.protocol.read_bytes())
    prior = json.loads(args.prior_result.read_bytes())
    start = json.loads((args.run_dir / "start.json").read_bytes())
    result = json.loads((args.run_dir / "result.json").read_bytes())
    for item in (start, result):
        same(item["freeze_reference"], FREEZE, "freeze")
        same(item["runner_sha256"], RUNNER_SHA, "runner")
        same(item["protocol_sha256"], PROTOCOL_SHA, "protocol")
        same(item["prior_result_sha256"], PRIOR_SHA, "prior")
    same(result["input_arrays"], plan["inputs"], "result.inputs")
    same(start["source_arrays"], plan["inputs"], "start.inputs")
    need([item["id"] for item in plan["inputs"]] == list(IDS), "Changed input sites")
    parent_rows = {row["id"]: row for row in prior["anatomy"]}
    data, file_receipts = {}, []
    for item in plan["inputs"]:
        path = args.input_dir / item["file"]
        need(path.name == item["file"] and sha(path) == item["sha256"], "Input hash/path mismatch")
        need(parent_rows[item["id"]]["arrays_file"] == item["file"]
             and parent_rows[item["id"]]["arrays_sha256"] == item["sha256"], "Input parent mismatch")
        with np.load(path, allow_pickle=False) as npz:
            fine = {arm: np.asarray(npz[arm], dtype=float) for arm in ARMS}
            coarse = np.asarray(npz["coarse_neighborhood"], dtype=float)
            center = np.asarray(npz["coarse_center_xyz"])
            origin = np.asarray(npz["coarse_neighborhood_origin_xyz"])
        need(coarse.shape == (35, 35, 35) and np.array_equal(center - origin, [17, 17, 17]),
             "Coarse source extraction changed")
        need(all(a.shape == (33, 33, 33) for a in fine.values()), "Fine source extraction changed")
        fine["native_coarse"] = coarse[1:34, 1:34, 1:34]
        need(all(np.isfinite(a).all() for a in fine.values()), "Nonfinite input")
        data[item["id"]] = fine
        file_receipts.append({"file": item["file"], "sha256": item["sha256"], "bytes": path.stat().st_size})
    score_path = args.run_dir / result["scores"]["file"]
    need(score_path.name == result["scores"]["file"] and sha(score_path) == result["scores"]["sha256"],
         "Stored correlation array hash/path mismatch")
    with np.load(score_path, allow_pickle=False) as npz:
        stored_fields = np.asarray(npz["values"])
        need(np.array_equal(npz["ids"], IDS) and np.array_equal(npz["binomial_passes"], FILTERS),
             "Stored array axis metadata differs")
    need(stored_fields.shape == (4, 2, 3, 3, 9, 9, 9), "Stored array dimensions differ")
    same(result["scores"]["shape"], [4, 2, 3, 3, 9, 9, 9], "score.shape")
    same(result["scores"]["number_of_correlations"], 52488, "score.count")
    same(result["scores"]["fine_arm_order"], list(ARMS), "score.arm_order")
    same(result["scores"]["mask_order"], list(MASKS), "score.mask_order")
    same(result["scores"]["shift_axis_values"], list(range(-4, 5)), "score.shifts")
    same(result["scores"]["axis_order"], ["site", "fine_arm", "binomial_passes", "mask", "shift_z", "shift_y", "shift_x"], "score.axes")
    filtered = {(i, name, passes): binomial_valid(array, passes)
                for i, arrays in data.items() for name, array in arrays.items() for passes in FILTERS}
    expected_quality = []
    for i in IDS:
        for name, raw in data[i].items():
            zero = 13506 if name == "native_coarse" else 20244
            for passes in FILTERS:
                expected_quality.append({"id": i, "array": name, "binomial_passes": passes,
                                         **quality(raw, filtered[i, name, passes], zero)})
    same(result["quality"], expected_quality, "quality")
    expected_pairs, expected_matched = [], []
    max_ncc_error, evaluated = 0., 0
    max_site_pair_error = 0.
    for si, i in enumerate(IDS):
        for ai, arm in enumerate(ARMS):
            for pi, passes in enumerate(FILTERS):
                a, b = filtered[i, arm, passes], filtered[i, "native_coarse", passes]
                fields = np.stack([direct_field(a, b, mask) for mask in selectors()])
                original = stored_fields[si, ai, pi]
                need(np.array_equal(np.isfinite(original), np.isfinite(fields)), "Undefined score mask differs")
                finite = np.isfinite(fields)
                if finite.any():
                    error = float(np.max(np.abs(original[finite] - fields[finite])))
                    max_ncc_error = max(max_ncc_error, error)
                    need(error <= 2e-10, f"Direct correlation error exceeded tolerance: {error}")
                evaluated += fields.size
                expected_matched.append({"id": i, "arm": arm, "binomial_passes": passes,
                                         "correlation_surfaces": {name: peak(field) for name, field in zip(MASKS, fields)},
                                         "slab_checks": slab_summaries(fields)})
                for j in IDS:
                    expected_pairs.append({"fine_id": i, "coarse_id": j, "arm": arm,
                                           "binomial_passes": passes, "same_site": i == j,
                                           "zero_shift_pearson": direct_correlation(centered(a), centered(filtered[j, "native_coarse", passes]))})
                need(time.process_time() - cpu_start <= 120, "Independent checker CPU cap exceeded")
    same(result["matched_site_comparisons"], expected_matched, "matched")
    same(result["all_site_pair_zero_shift_comparisons"], expected_pairs, "site_pairs")
    for actual, expected in zip(result["all_site_pair_zero_shift_comparisons"], expected_pairs):
        if expected["zero_shift_pearson"] is not None:
            max_site_pair_error = max(max_site_pair_error, abs(actual["zero_shift_pearson"] - expected["zero_shift_pearson"]))
    for key in ("new_ct_or_label_access", "model_run", "registration_update_applied", "material_identity_established"):
        need(result[key] is False, f"Unexpected scientific action claim: {key}")
    need(result["adaptively_chosen_after_prior_outcomes"] is True, "Adaptive provenance missing")
    need(result["resources"]["new_network_bytes"] == 0 and result["resources"]["new_raw_chunks"] == 0,
         "Unexpected source acquisition")
    return {"schema": "frag5-local-structure-independent-result-review-v1", "status": "passed",
            "freeze_reference": FREEZE, "runner_sha256": RUNNER_SHA, "protocol_sha256": PROTOCOL_SHA,
            "prior_result_sha256": PRIOR_SHA, "result_sha256": sha(args.run_dir / "result.json"),
            "start_sha256": sha(args.run_dir / "start.json"), "checker_sha256": sha(__file__),
            "input_arrays_verified": file_receipts, "score_array_sha256": sha(score_path),
            "direct_correlations_recomputed": evaluated, "maximum_correlation_absolute_error": max_ncc_error,
            "quality_records_recomputed": len(expected_quality), "quality_lag_correlations_recomputed": len(expected_quality) * 9,
            "site_pair_correlations_recomputed": len(expected_pairs), "maximum_site_pair_absolute_error": max_site_pair_error,
            "peak_summaries_recomputed": len(expected_matched) * 3, "slab_checks_recomputed": len(expected_matched) * 2,
            "method": "Independent full binomial coefficients and valid direct convolution; explicit support and masked direct centered dot products for every shift; independent gradients, tensor entries, eigenvalues, peaks and slab checks.",
            "new_network_bytes": 0, "new_raw_chunks": 0, "labels_or_models_accessed": False,
            "scientific_runner_imported": False, "cpu_seconds": time.process_time() - cpu_start,
            "limits": ["Verifies computations on existing development caches, not acquisition truth or anatomical correspondence.",
                       "Does not repeat the earlier fine-footprint integration or source acquisition.",
                       "No statistical independence, anatomical identity, label transfer or ink improvement is established."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("input-dir", "prior-result", "protocol", "runner", "run-dir", "review-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    if args.review_dir.exists():
        parser.error("Choose a fresh review directory; failures are preserved")
    args.review_dir.mkdir(parents=True)
    cpu_start = time.process_time()
    write(args.review_dir / "review_start.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                                  "checker_sha256": sha(__file__), "network_allowed": False})
    try:
        receipt = review(args, cpu_start)
        write(args.review_dir / "independent_result_review.json", receipt)
        print(json.dumps(receipt, indent=2))
    except Exception as exc:
        write(args.review_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                 "traceback": traceback.format_exc(),
                                                 "cpu_seconds": time.process_time() - cpu_start})
        raise


if __name__ == "__main__":
    main()
