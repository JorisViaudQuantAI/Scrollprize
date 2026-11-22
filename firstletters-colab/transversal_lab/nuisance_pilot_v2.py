#!/usr/bin/env python3
"""Separate support amendment using normalized masked Gaussian moments.

This adaptively follows v1's support-only failure on the same public controls.
It preserves invalid centres, all ranking settings, source denominators and
continuation gates. It is not independent validation or physical mask auditing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

import nuisance_pilot as v1

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "nuisance_protocol_v2.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())


def verify_unchanged_settings():
    """The only intended numerical amendment is the support/moment operator."""
    for key in ("cases", "work_um", "views", "local_sigma_mm", "gaussian_truncate",
                "standard_deviation_floor", "standardized_mapping",
                "minimum_original_centreline_support_fraction", "weak_row_id",
                "strong_row_id", "localization_continuation_gate",
                "time_limit_seconds_per_input_case"):
        if PROTOCOL[key] != v1.PROTOCOL[key]:
            raise ValueError("Unexpected change from v1 setting: " + key)
    if v1.sha(v1.PROTOCOL_PATH) != PROTOCOL["prior_v1"]["protocol_sha256"]:
        raise ValueError("Published v1 protocol changed")
    if v1.sha(v1.__file__) != PROTOCOL["prior_v1"]["runner_sha256"]:
        raise ValueError("Published v1 runner changed")


def gaussian(array, sigma):
    # Missing or exterior material remains missing validity mass. There is no
    # in-bounds-only renormalization of this operator or its coverage threshold.
    return gaussian_filter(array, sigma=sigma,
        truncate=PROTOCOL["gaussian_truncate"], mode="constant", cval=0)


def grid_context(score, valid, source_spacing):
    grid, observed, spacing_mm = v1.work_grid(score, valid, source_spacing, v1.Config())
    sigma = np.array([PROTOCOL["local_sigma_mm"] / d for d in spacing_mm])
    mass = gaussian(observed.astype(np.float32), sigma)
    common = observed & (mass >= PROTOCOL["minimum_gaussian_observed_mass"])
    # A work cell must retain its original conservative native-centre validity.
    # Neighbours estimate a background; they never supply a missing centre score.
    filled = np.where(observed, grid, 0).astype(np.float32)
    return filled, observed, common, spacing_mm, sigma, mass


def standardized_views(grid, common, sigma, mass):
    if not common.any():
        return None, {"status": "unassessable_no_observed_centres_with_required_gaussian_mass"}
    floor = PROTOCOL["standard_deviation_floor"]
    values = grid[common].astype(np.float64)
    global_mean, global_sd = float(values.mean()), float(values.std())
    # grid is already zero at originally unobserved work cells. Use that
    # original observed mask, not common, in both local moment numerators.
    first = gaussian(grid, sigma)
    second = gaussian(grid * grid, sigma)
    local_mean = np.divide(first, mass, out=np.zeros_like(first), where=mass > 0)
    local_second = np.divide(second, mass, out=np.zeros_like(second), where=mass > 0)
    local_sd = np.sqrt(np.maximum(local_second - local_mean * local_mean, 0))
    global_q = np.clip((grid - global_mean) / max(global_sd, floor) / 4, 0, 1).astype(np.float32)
    local_q = np.clip((grid - local_mean) / np.maximum(local_sd, floor) / 4, 0, 1).astype(np.float32)
    return dict(raw=grid, global_standardized=global_q, local_standardized=local_q), {
        "global_mean": global_mean, "global_population_sd": global_sd,
        "global_sd_after_floor": max(global_sd, floor),
        "local_sd_quantiles_on_common": np.quantile(local_sd[common], [0,.25,.5,.75,1]).tolist(),
        "local_sd_floor_fraction_on_common": float((local_sd[common] < floor).mean()),
        "gaussian_mass_quantiles_on_common": np.quantile(mass[common], [0,.25,.5,.75,1]).tolist(),
        "coverage_interpretation": "Gaussian weight of the released tissue/model-support proxy, not audited acquisition support"}


def explicit_row_metrics(rows):
    """Clarify exported denominators AFTER computing the unchanged v1 gate.

    This is a reporting amendment only. The inherited half-coverage rank and
    strong-row preservation criterion use conditional supported-pixel coverage.
    New full-source fractions and ranks are descriptive, never new gate inputs.
    """
    exported = []
    for row in rows:
        def coverage(value):
            record = {k: v for k, v in value.items()
                      if k not in ("centreline_coverage", "box_coverage")}
            record["conditional_centreline_fraction"] = value["centreline_coverage"]
            record["conditional_box_fraction"] = value["box_coverage"]
            record["observed_covered_fraction_of_full_source_centreline"] = (
                value["covered_source_centreline_pixels"] / row["full_source_centreline_pixels"])
            record["observed_covered_fraction_of_full_source_box"] = (
                value["covered_source_box_pixels"] / row["full_source_box_pixels"])
            return record

        result = {k: v for k, v in row.items() if k not in
                  ("top_k", "individual", "first_individual_half_rank",
                   "best_individual_centreline_coverage")}
        result["top_k"] = {k: coverage(v) for k, v in row["top_k"].items()}
        result["individual"] = [coverage(v) for v in row["individual"]]
        result["first_half_supported_centreline_rank"] = row["first_individual_half_rank"]
        result["best_individual_conditional_centreline_fraction"] = row["best_individual_centreline_coverage"]
        first = row["first_individual_half_rank"]
        result["first_half_supported_centreline_proposal"] = next(
            (v for v in result["individual"] if v["rank"] == first), None)
        result["first_half_full_source_centreline_rank"] = next(
            (v["rank"] for v in result["individual"]
             if v["observed_covered_fraction_of_full_source_centreline"] >= .5), None)
        result["coverage_semantics"] = PROTOCOL["coverage_metric_semantics"]
        exported.append(result)
    return exported


def run_case(case, source_root, output):
    start = time.perf_counter()
    folder = source_root / "cases" / case["id"]
    paths = {"score": folder / "score.npy", "mask": folder / "validity.npy"}
    expected = {"score": case["score_sha256"], "mask": case["mask_sha256"]}
    for key, path in paths.items():
        if v1.sha(path) != expected[key]:
            raise ValueError("Pinned source hash mismatch: " + case["id"] + "/" + key)
    score = np.load(paths["score"], allow_pickle=False)
    valid = np.load(paths["mask"], allow_pickle=False)
    v1.validate_inputs(score, valid, case["spacing_um_yx"], "model_score")
    if list(score.shape) != case["shape"]:
        raise ValueError("Unexpected source grid")
    grid, observed, common, spacing_mm, sigma, mass = grid_context(score, valid, case["spacing_um_yx"])
    del score, valid
    arrays, stats = standardized_views(grid, common, sigma, mass)
    out = output / case["id"]
    out.mkdir()
    np.save(out / "common_valid.npy", common, allow_pickle=False)
    np.save(out / "gaussian_observed_mass.npy", mass, allow_pickle=False)
    descriptor = [v1.target_descriptor(t, case["shape"], common.shape, common)[0]
                  for t in case["targets"]]
    result = dict(id=case["id"], role=case["role"], source_hashes=expected,
        source_shape=case["shape"], work_shape_yx=list(grid.shape),
        effective_work_spacing_mm_yx=list(spacing_mm),
        original_source_valid_pixels=case["valid_pixels"], work_observed_pixels=int(observed.sum()),
        common_gaussian_mass_pixels=int(common.sum()),
        common_fraction_of_observed_work=float(common.sum() / max(1, observed.sum())),
        minimum_gaussian_observed_mass=PROTOCOL["minimum_gaussian_observed_mass"],
        sigma_work_pixels=sigma.tolist(),
        gaussian_mass_quantiles_on_observed_centres=np.quantile(mass[observed], [0,.25,.5,.75,1]).tolist() if observed.any() else None,
        source_validity_semantics=PROTOCOL["source_validity_semantics"],
        source_target_support=descriptor, statistics=stats, views=[])
    if arrays is not None:
        for name in PROTOCOL["views"]:
            p = arrays[name]
            report = v1.rank_map(p, common, [1000*d for d in spacing_mm],
                "model_score" if name == "raw" else "display_signal", v1.Config())
            if report["work_shape_yx"] != list(grid.shape):
                raise ValueError("Unexpected second spatial resampling")
            rows = v1.row_metrics(report["proposals"], case["targets"], case["shape"], common)
            np.save(out / (name + ".npy"), np.where(common, p, np.nan).astype(np.float32), allow_pickle=False)
            result["views"].append(dict(name=name, proposal_count=len(report["proposals"]),
                search_truncated=report["search_truncated"], seconds=report["seconds"],
                score_quantiles=np.quantile(p[common], [0,.25,.5,.75,.95,.99,1]).tolist(),
                rows=rows, proposals=report["proposals"], ranker_trials=report["trials"],
                ranker_work_shape_yx=report["work_shape_yx"],
                ranker_work_spacing_mm_yx=report["work_spacing_mm_yx"]))
        v1.plot_views(result["views"], arrays, common, case, spacing_mm, out / "comparison.png")
    result["localization_gate"] = v1.localization_gate(result["views"], case["role"])
    if not result["views"] and case["role"] != "matched-depth-shuffle-process-control":
        # Status wording reflects the amended support operator; gate values and
        # every numerical continuation condition remain those of frozen v1.
        result["localization_gate"]["status"] = "unassessable_no_observed_centres_with_required_gaussian_mass"
    result["localization_gate"]["coverage_denominator"] = "Supported original centreline samples; conditional, not full-original-source coverage"
    result["localization_gate"]["coverage_limitation"] = PROTOCOL["coverage_metric_semantics"]["gate_limitation"]
    # The gate above receives exactly v1's unmodified numerical row metrics.
    # Export explicit names plus full-source fractions only after it is fixed.
    for view in result["views"]:
        view["rows"] = explicit_row_metrics(view["rows"])
    result["target_support_assessment"] = "adequate_for_localization_gate" if all(
        d["source_centreline_support_fraction"] >= PROTOCOL["minimum_original_centreline_support_fraction"]
        for d in descriptor) else "unassessable_at_least_one_original_centreline_below_half_support"
    for key, path in paths.items():
        if v1.sha(path) != expected[key]:
            raise ValueError("Source changed during pilot")
    result["status"] = "completed"
    result["seconds_total"] = time.perf_counter() - start
    result["output_sha256"] = {p.name: v1.sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    v1.save_json(out / "report.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-controls-root", type=Path, required=True)
    ap.add_argument("--v1-result", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    verify_unchanged_settings()
    v1.outside_git(args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError("Use a new external directory distinct from v1")
    prior_result_sha = v1.sha(args.v1_result)
    if prior_result_sha != PROTOCOL["prior_v1"]["result_sha256"]:
        raise ValueError("Prior v1 receipt differs from the frozen amendment motivation")
    inventory = json.loads(v1.INVENTORY_PATH.read_text())
    cases = {c["id"]: c for c in inventory["cases"]}
    cv2.setNumThreads(4)
    args.output_dir.mkdir(parents=True)
    signature = dict(pilot_version="v2", protocol_status=PROTOCOL["status"],
        experimental_status="adaptive_support_amendment_on_reused_published_controls_not_independent_validation",
        row_metric_semantics=PROTOCOL["coverage_metric_semantics"],
        protocol_sha256=v1.sha(PROTOCOL_PATH), script_sha256=v1.sha(__file__),
        prior_v1_result_sha256=prior_result_sha,
        prior_v1_protocol_sha256=v1.sha(v1.PROTOCOL_PATH),
        reused_v1_runner_sha256=v1.sha(v1.__file__),
        ranker_sha256=v1.sha(HERE.parent / "line_review/review.py"),
        inventory_sha256=v1.sha(v1.INVENTORY_PATH),
        runtime={"python": sys.version.split()[0], "numpy": np.__version__,
                 "scipy": scipy.__version__, "opencv": cv2.__version__},
        config=asdict(v1.Config()), started_utc=datetime.now(timezone.utc).isoformat(),
        network_forward_calls=0)
    v1.save_json(args.output_dir / "run_started.json", signature)

    def expire(_a, _b):
        raise TimeoutError("Frozen 180-second input-case budget exhausted")

    old = signal.signal(signal.SIGALRM, expire)
    results = []
    try:
        for cid in PROTOCOL["cases"]:
            signal.setitimer(signal.ITIMER_REAL, PROTOCOL["time_limit_seconds_per_input_case"])
            try:
                r = run_case(cases[cid], args.public_controls_root, args.output_dir)
            except TimeoutError:
                results.append({"id": cid, "status": "incomplete_timeout"})
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            results.append(r)
            print(json.dumps({"case": cid, "common_work_pixels": r["common_gaussian_mass_pixels"],
                "target_support_assessment": r["target_support_assessment"],
                "views": [{"name": v["name"], "proposals": v["proposal_count"]} for v in r["views"]],
                "localization_gate": r["localization_gate"]}), flush=True)
    finally:
        signal.signal(signal.SIGALRM, old)
    result = dict(signature,
        status="completed" if len(results) == len(PROTOCOL["cases"]) and all(r["status"] == "completed" for r in results) else "incomplete",
        cases=results,
        operational_gate="not_established_adaptive_amendment_single_previously_inspected_published_source",
        limitations=["Adaptive amendment motivated by v1 support-only failure; same reused known rows",
            "Gaussian proxy-mask coverage is not audited physical acquisition support",
            "Unobserved centres remain excluded; structured missingness can still bias normalized moments",
            "Coarse published rows are localization targets, not ink truth", "All unlabelled pixels remain unknown",
            "No new model inference or independent detector", "No population false-positive estimate",
            "Local standardization can enhance crack and blob rims",
            "Any continuation gate requires independent controls before operational use"])
    v1.save_json(args.output_dir / "result.json", result)
    return 0 if result["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
