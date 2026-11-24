"""Reproduce synthetic A02 planning comparisons and their mathematical validation.

Run from industrial_funnel:
    python -m followup_a03.run_comparison --out followup_a03/comparison_result.json

No real scan/model output or adjudication is loaded. Nothing is dispatched.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import platform
import sys
from time import perf_counter
import unittest

from .joint_zero_bound import joint_zero_upper_bound
from . import test_joint_zero_bound

ROOT = Path(__file__).resolve().parents[1]


def compare_case(name, strata, *, source, unknown_paired_units=0):
    from survey import zero_upper_count

    strata = tuple(strata)
    legacy = sum(zero_upper_count(N, n, 0.05 / len(strata)) for N, n in strata)
    exact = joint_zero_upper_bound(strata)
    return {
        "name": name,
        "source": source,
        "strata": [{"N": N, "n": n} for N, n in strata],
        "unknown_order_census_units_excluded_from_estimand": unknown_paired_units,
        "current_sum_bonferroni_upper_count": legacy,
        "joint_total": exact.as_dict(),
    }


def build_comparison():
    from order_audit import audit_summary, freeze_plan
    from survey import zero_upper_count
    import test_order_audit as a02_fixtures

    plan = freeze_plan(a02_fixtures.population(), a02_fixtures.recipes(), 7401)
    labels = {
        str(i): ("NO_SINGLE_ORDER_EXPOSURE" if plan["population"][i]["normal_order"] == "unknown"
                 else "NO_ADDITIONAL_DETECTABLE_TEXT")
        for i in a02_fixtures.pair_indices(plan)
    }
    legacy = audit_summary(plan, labels)
    known = [(row["N"], row["n"]) for _, row in sorted(legacy["strata"].items())]
    first = compare_case(
        "existing_A02_test_fixture_known_orders_only", known,
        source="Existing test_order_audit.population() synthetic fixture; all-negative outcomes imposed for planning.",
        unknown_paired_units=legacy["unknown_order_paired_units"],
    )
    assert first["current_sum_bonferroni_upper_count"] == legacy["simultaneous_zero_event_upper_count"]
    assert first["joint_total"]["population_units"] == 40
    assert first["joint_total"]["sample_units"] == 6
    assert first["unknown_order_census_units_excluded_from_estimand"] == 2

    cases = [first]
    for name, strata in (
        ("hypothetical_120_units_one_stratum", [(120, 18)]),
        ("hypothetical_120_units_two_equal_strata", [(60, 9)] * 2),
        ("hypothetical_120_units_six_equal_strata", [(20, 3)] * 6),
        ("hypothetical_120_units_twelve_equal_strata_rounding", [(10, 2)] * 12),
        ("hypothetical_120_units_unequal_sizes_equal_fraction", [(100, 15), (20, 3)]),
    ):
        cases.append(compare_case(name, strata, source="Synthetic design; assumes all sampled outcomes zero, no observed evidence."))

    planning = []
    for H, N in ((1, 120), (2, 60), (6, 20), (12, 10)):
        joint_n = next(n for n in range(1, N + 1)
                       if joint_zero_upper_bound([(N, n)] * H).upper_count < 6)
        legacy_n = next(n for n in range(1, N + 1)
                        if H * zero_upper_count(N, n, 0.05 / H) < 6)
        planning.append({
            "strata": H, "population_per_stratum": N,
            "target": "If zero events, exclude total missed >= 6 among 120 known-order units at alpha=0.05.",
            "allocation_constraint": "Same integer n in every stratum, fixed before outcomes.",
            "joint_min_audits_per_stratum": joint_n,
            "joint_min_total_audits": H * joint_n,
            "bonferroni_min_audits_per_stratum": legacy_n,
            "bonferroni_min_total_audits": H * legacy_n,
            "observed_evidence": False,
        })

    p0 = Fraction(comb(100 - 50, 1), comb(100, 1))
    anti_pooling = {
        "actual_design": [{"N": 100, "n": 1}, {"N": 20, "n": 20}],
        "fixed_missed_allocation": [50, 0],
        "actual_probability_zero": str(p0),
        "correct_joint_upper_count": joint_zero_upper_bound(((100, 1), (20, 20))).upper_count,
        "invalid_pooled_N120_n21_upper_count": zero_upper_count(120, 21, 0.05),
        "implication": "Pooling would exclude the true total 50 with probability 0.5, violating a 0.05 error level.",
    }
    paths = [ROOT / name for name in ("survey.py", "order_audit.py", "test_order_audit.py")]
    paths += [Path(__file__).with_name(name) for name in ("joint_zero_bound.py", "test_joint_zero_bound.py", "run_comparison.py")]
    return {
        "schema": "funnel.optional_joint_zero_bound_comparison.v1",
        "review_context_public_ref": "52f90c8713d28d98be64262499deb67f10d062a6",
        "source_file_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "scope": "Synthetic design comparison; fixed disjoint unit populations, independent SRSWOR within strata, complete zero-event review.",
        "observed_scan_evidence": False,
        "new_model_inferences": 0,
        "promotion_allowed": False,
        "gate_changes": False,
        "baseline_amendment074_modified": False,
        "alpha": "1/20",
        "inversion_convention": "Largest M whose exact worst-allocation zero-event probability is >= alpha; excludes strict < alpha.",
        "cases": cases,
        "predeclared_sample_size_planning": planning,
        "anti_pooling_counterexample": anti_pooling,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("comparison_result.json"))
    args = parser.parse_args()
    start = perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_joint_zero_bound)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    elapsed = perf_counter() - start
    if not result.wasSuccessful():
        raise SystemExit("Mathematical verification failed; comparison not written")
    report = build_comparison()
    report["verification"] = {
        "python": platform.python_version(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "seconds": elapsed,
        "exhaustive_ordered_plans": 5511,
        "oracle": "All allocations evaluated with math.comb and Fraction; every probability threshold equality and midpoint tested.",
        "additional_checks": ["unconditional design noncoverage", "sample-size and confidence monotonicity", "n=0 and census", "empty populations", "unequal-design pooling counterexample"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out), "tests_run": result.testsRun,
                      "fixture_upper_before": report["cases"][0]["current_sum_bonferroni_upper_count"],
                      "fixture_upper_after": report["cases"][0]["joint_total"]["upper_count"]}, indent=2))


if __name__ == "__main__":
    main()
