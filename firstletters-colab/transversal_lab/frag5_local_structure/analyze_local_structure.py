#!/usr/bin/env python3
"""Adaptive, cache-only Frag5 structure and bounded-translation diagnostic.

No network, labels, detector, registration update, or automatic tissue verdict.
All four sites are already-observed development data. The protocol is frozen
before computing these additional metrics, not before the original CT access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np
import scipy
from scipy.ndimage import convolve1d
from scipy.signal import correlate

CORE = 17
SEARCH = 4
PASSES = (0, 1, 4)
ARMS = ("fine_nominal_coarse_footprint", "fine_point")
MASK_NAMES = ("full", "negative_x_slab", "positive_x_slab")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scalar(value):
    return float(value) if np.isfinite(value) else None


def crop(a, side, shift_xyz=(0, 0, 0)):
    """An odd, centered cube; positive shift means sampling at q + shift."""
    if a.ndim != 3 or side % 2 != 1 or any(n % 2 != 1 for n in a.shape):
        raise ValueError("Odd three-dimensional arrays and odd support required")
    c = np.array(a.shape) // 2 + np.asarray(shift_xyz, dtype=int)[::-1]
    lo, hi = c - side // 2, c + side // 2 + 1
    if np.any(lo < 0) or np.any(hi > a.shape):
        raise ValueError("Incomplete support; no padding or wraparound allowed")
    return a[tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))]


def smooth_valid(a, passes):
    """Separable binomial filtering, deleting every boundary it touched."""
    a = np.asarray(a, dtype=np.float64)
    for _ in range(passes):
        for axis in range(3):
            a = convolve1d(a, np.array([.25, .5, .25]), axis=axis,
                           mode="constant", cval=0.)
        a = a[1:-1, 1:-1, 1:-1]
    return a


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("Unequal comparison support")
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return scalar(np.sum(a * b) / den) if den > 0 else None


def masks():
    full = np.ones((CORE,) * 3, dtype=np.float64)
    left, right = np.zeros_like(full), np.zeros_like(full)
    left[:, :, :4] = 1
    right[:, :, -4:] = 1
    return (full, left, right)


def correlation_field(template, candidate, mask):
    """NCC for a 17³ template and 25³ search field; output axes are Z,Y,X.

    Index (z,y,x) denotes coarse XYZ shift (x-4,y-4,z-4). Means and variances
    are recomputed on exactly the selected mask at every shift. FFT is only
    an implementation of finite dot products, with no cyclic wrapping.
    """
    n = float(mask.sum())
    if template.shape != mask.shape or candidate.shape != (CORE + 2 * SEARCH,) * 3:
        raise ValueError("Correlation support differs from frozen design")
    selected = template[mask.astype(bool)]
    if selected.size < 2 or np.all(selected == selected[0]) or np.all(candidate == candidate.flat[0]):
        return np.full((2 * SEARCH + 1,) * 3, np.nan)
    t = (template - np.sum(template * mask) / n) * mask
    t_ss = float(np.sum(t * t))
    b = candidate - candidate.mean()
    if t_ss == 0 or not np.any(b):
        return np.full((2 * SEARCH + 1,) * 3, np.nan)
    b_sum = correlate(b, mask, mode="valid", method="fft")
    b_sq = correlate(b * b, mask, mode="valid", method="fft")
    b_ss = b_sq - b_sum * b_sum / n
    roundoff = 128 * np.finfo(np.float64).eps * n * float(np.max(np.abs(b)) ** 2)
    if np.min(b_ss) < -roundoff:
        raise ValueError("Negative variance exceeds numerical tolerance")
    valid = b_ss > roundoff
    out = np.full(b_ss.shape, np.nan)
    numerator = correlate(b, t, mode="valid", method="fft")
    out[valid] = numerator[valid] / np.sqrt(t_ss * b_ss[valid])
    if np.any(np.abs(out[valid]) > 1 + 1e-10):
        raise ValueError("NCC exceeds rounding tolerance")
    out[valid] = np.clip(out[valid], -1, 1)
    return out


def peak_summary(field):
    valid = np.isfinite(field)
    if not np.any(valid):
        return {"status": "undefined", "zero_shift": None, "maximum": None,
                "argmax_shift_xyz": None, "boundary_maximum": None,
                "within_0_01_of_maximum": 0, "within_0_05_of_maximum": 0}
    # First maximum in ZYX lexicographic order is only a reporting convention.
    arg = np.unravel_index(np.nanargmax(field), field.shape)
    delta = [int(v - SEARCH) for v in arg[::-1]]
    maximum = float(field[arg])
    return {"status": "assessable", "zero_shift": scalar(field[(SEARCH,) * 3]),
            "maximum": maximum, "argmax_shift_xyz": delta,
            "boundary_maximum": any(abs(v) == SEARCH for v in delta),
            "within_0_01_of_maximum": int(np.sum(valid & (field >= maximum - .01))),
            "within_0_05_of_maximum": int(np.sum(valid & (field >= maximum - .05)))}


def slab_check(fields):
    out = []
    for fit, check in ((1, 2), (2, 1)):
        if not np.any(np.isfinite(fields[fit])):
            out.append({"fit": MASK_NAMES[fit], "check": MASK_NAMES[check], "status": "undefined"})
            continue
        arg = np.unravel_index(np.nanargmax(fields[fit]), fields[fit].shape)
        out.append({"fit": MASK_NAMES[fit], "check": MASK_NAMES[check],
                    "status": "descriptive_same_site_check",
                    "selected_shift_xyz": [int(v - SEARCH) for v in arg[::-1]],
                    "fit_score": scalar(fields[fit][arg]),
                    "check_score_at_selected_shift": scalar(fields[check][arg]),
                    "check_score_at_zero_shift": scalar(fields[check][(SEARCH,) * 3]),
                    "boundary_selected": any(v in (0, 2 * SEARCH) for v in arg)})
    return out


def structure_metrics(raw, filtered, recorded_zero):
    center = crop(filtered, CORE)
    raw_center = crop(raw, CORE)
    constant, raw_constant = np.all(center == center.flat[0]), np.all(raw_center == raw_center.flat[0])
    mean = float(center.flat[0]) if constant else float(center.mean())
    var = 0. if constant else float(np.var(center))
    raw_var = 0. if raw_constant else float(np.var(raw_center))
    gradients = []
    lags = {}
    for axis, name in enumerate(("x", "y", "z")):
        unit = np.zeros(3, dtype=int)
        unit[axis] = 1
        gradients.append((crop(filtered, CORE, unit) - crop(filtered, CORE, -unit)) / 2)
        for lag in (1, 2, 4):
            lags[f"{name}_{lag}"] = pearson(center, crop(filtered, CORE, lag * unit))
    g = np.stack(gradients, axis=-1).reshape(-1, 3)
    tensor = g.T @ g / len(g)
    eigenvalues = np.linalg.eigvalsh(tensor)[::-1]
    trace = float(np.trace(tensor))
    return {"samples": center.size, "mean": mean, "sd": float(np.sqrt(var)),
            "recorded_reconstruction_zero": recorded_zero,
            "mean_minus_recorded_zero": mean - recorded_zero,
            "variance_retention": var / raw_var if raw_var > 0 else None,
            "lag_pearson_positive_xyz": lags, "structure_tensor_xyz": tensor.tolist(),
            "gradient_energy_counts_squared_per_grid_voxel_squared": trace,
            "tensor_eigenvalue_fractions_descending": (eigenvalues / trace).tolist() if trace > 0 else None,
            "material_or_noise_classification": "not_identified"}


def validate_protocol(plan):
    if plan["schema"] != "frag5-adaptive-local-structure-v1":
        raise ValueError("Unsupported protocol")
    expected = {"core_side": CORE, "search_radius_coarse_voxels": SEARCH,
                "binomial_passes": list(PASSES), "fine_arms": list(ARMS),
                "mask_names": list(MASK_NAMES), "slab_width": 4}
    if plan["numerical_design"] != expected or [r["id"] for r in plan["inputs"]] != [20, 25, 65, 70]:
        raise ValueError("Frozen design changed")
    if plan["caps"] != {"new_network_bytes": 0, "cpu_seconds": 120, "sites": 4}:
        raise ValueError("Unsupported resource limits")


def verify_inputs(input_dir, prior_result, plan):
    if sha(prior_result) != plan["prior_result_sha256"]:
        raise ValueError("Prior result changed")
    prior = json.loads(prior_result.read_bytes())
    known = {row["id"]: row for row in prior["anatomy"]}
    for item in plan["inputs"]:
        path = input_dir / item["file"]
        if path.name != item["file"] or sha(path) != item["sha256"]:
            raise ValueError("Input array missing or changed")
        if (known[item["id"]]["arrays_sha256"] != item["sha256"]
                or known[item["id"]]["arrays_file"] != item["file"]):
            raise ValueError("Array provenance does not match published result")


def load_inputs(input_dir, plan):
    data = {}
    for item in plan["inputs"]:
        with np.load(input_dir / item["file"], allow_pickle=False) as archive:
            a = {arm: np.array(archive[arm], dtype=np.float64) for arm in ARMS}
            coarse = np.array(archive["coarse_neighborhood"], dtype=np.float64)
            center = np.array(archive["coarse_center_xyz"])
            origin = np.array(archive["coarse_neighborhood_origin_xyz"])
            if coarse.shape != (35,) * 3 or np.any(center - origin != 17):
                raise ValueError("Coarse cache is not the recorded centered 35³ neighborhood")
            if any(v.shape != (33,) * 3 for v in a.values()):
                raise ValueError("Fine cache is not the recorded 33³ grid")
            # Identical centered 33³ supports for every representation.
            a["native_coarse"] = crop(coarse, 33).copy()
            if any(not np.all(np.isfinite(v)) for v in a.values()):
                raise ValueError("Nonfinite input values")
            data[item["id"]] = a
    return data


def execute(data, plan, out, freeze, source_hash, protocol_hash, cpu_start):
    ids = [r["id"] for r in plan["inputs"]]
    filtered = {(i, name, passes): smooth_valid(a, passes)
                for i, arrays in data.items() for name, a in arrays.items() for passes in PASSES}
    quality, comparisons, mismatches = [], [], []
    score_arrays = np.empty((4, 2, 3, 3, 9, 9, 9), dtype=np.float64)
    for si, i in enumerate(ids):
        for name, raw in data[i].items():
            zero = plan["recorded_reconstruction_zero"]["coarse" if name == "native_coarse" else "fine"]
            for passes in PASSES:
                quality.append({"id": i, "array": name, "binomial_passes": passes,
                                **structure_metrics(raw, filtered[i, name, passes], zero)})
        for ai, arm in enumerate(ARMS):
            for pi, passes in enumerate(PASSES):
                template = crop(filtered[i, arm, passes], CORE)
                candidate = crop(filtered[i, "native_coarse", passes], CORE + 2 * SEARCH)
                fields = np.stack([correlation_field(template, candidate, m) for m in masks()])
                score_arrays[si, ai, pi] = fields
                comparisons.append({"id": i, "arm": arm, "binomial_passes": passes,
                                    "correlation_surfaces": {name: peak_summary(f) for name, f in zip(MASK_NAMES, fields)},
                                    "slab_checks": slab_check(fields)})
                for j in ids:
                    mismatches.append({"fine_id": i, "coarse_id": j, "arm": arm,
                                       "binomial_passes": passes, "same_site": i == j,
                                       "zero_shift_pearson": pearson(template, crop(filtered[j, "native_coarse", passes], CORE))})
        if time.process_time() - cpu_start > plan["caps"]["cpu_seconds"]:
            raise RuntimeError("CPU cap exceeded")
    np.savez_compressed(out / "all_correlation_surfaces.npz", values=score_arrays,
                        ids=np.array(ids), binomial_passes=np.array(PASSES))
    result = {"schema": plan["schema"], "freeze_reference": freeze,
              "runner_sha256": source_hash, "protocol_sha256": protocol_hash,
              "prior_result_sha256": plan["prior_result_sha256"], "input_arrays": plan["inputs"],
              "adaptively_chosen_after_prior_outcomes": True,
              "new_ct_or_label_access": False, "model_run": False,
              "registration_update_applied": False, "material_identity_established": False,
              "quality": quality, "matched_site_comparisons": comparisons,
              "all_site_pair_zero_shift_comparisons": mismatches,
              "scores": {"file": "all_correlation_surfaces.npz", "sha256": sha(out / "all_correlation_surfaces.npz"),
                         "shape": list(score_arrays.shape), "axis_order": ["site", "fine_arm", "binomial_passes", "mask", "shift_z", "shift_y", "shift_x"],
                         "fine_arm_order": list(ARMS), "mask_order": list(MASK_NAMES),
                         "shift_axis_values": list(range(-SEARCH, SEARCH + 1)),
                         "number_of_correlations": int(score_arrays.size)},
              "resources": {"new_network_bytes": 0, "new_raw_chunks": 0,
                            "process_cpu_seconds": time.process_time() - cpu_start},
              "environment": {"numpy": np.__version__, "scipy": scipy.__version__}}
    write_json(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cpu_start = time.process_time()
    plan = json.loads(args.protocol.read_bytes())
    validate_protocol(plan)
    verify_inputs(args.input_dir, args.prior_result, plan)
    if args.check_only:
        print("Protocol and input hashes verified; no CT arrays decoded.")
        return
    if not args.freeze_reference or not re.fullmatch(r"[0-9a-f]{40}", args.freeze_reference):
        parser.error("A public 40-character freeze commit is required")
    if args.out is None or args.out.exists():
        parser.error("Choose a fresh output directory")
    args.out.mkdir(parents=True)
    start = {"started_utc": datetime.now(timezone.utc).isoformat(), "freeze_reference": args.freeze_reference,
             "runner_sha256": sha(__file__), "protocol_sha256": sha(args.protocol),
             "prior_result_sha256": sha(args.prior_result), "source_arrays": plan["inputs"],
             "scope": "Adaptive additional diagnostics on already observed CT caches; no new acquisition"}
    write_json(args.out / "start.json", start)
    try:
        data = load_inputs(args.input_dir, plan)
        result = execute(data, plan, args.out, args.freeze_reference, sha(__file__), sha(args.protocol), cpu_start)
        print(json.dumps({"outcome": "completed", "correlations": result["scores"]["number_of_correlations"],
                          "resources": result["resources"]}))
    except Exception as exc:
        write_json(args.out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                   "process_cpu_seconds": time.process_time() - cpu_start})
        raise


if __name__ == "__main__":
    main()
