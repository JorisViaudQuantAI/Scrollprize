"""Exact finite-population zero-event upper bound on a stratified TOTAL.

This is a mathematical comparator, not evidence admission. It assumes independent
simple random samples without replacement within fixed, disjoint strata, fixed
binary unit outcomes, and complete zero-event reviews. It does not authenticate
any of those premises. No control-plane or order-audit gate imports this module.

For total M, the nuisance parameter is its allocation m_h across strata. The
maximum zero-event probability is max(prod(C(N_h-m_h,n_h)/C(N_h,n_h))) subject
to sum(m_h)=M. Successive factors (N_h-n_h-m_h)/(N_h-m_h) are nonincreasing,
so merging them with a max heap gives the exact maximizing allocation for every
M. Fractions avoid rounding at the confidence threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
from typing import Iterable


def _strata(values: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as exc:
        raise ValueError("Strata must be an iterable of (N, n) pairs") from exc
    for row in rows:
        if len(row) != 2:
            raise ValueError("Each stratum must contain N and n")
        N, n = row
        if type(N) is not int or type(n) is not int or not 0 <= n <= N:
            raise ValueError("Each stratum requires integers 0 <= n <= N")
    return rows


def _alpha(value: Fraction | float | str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, float, str, int)):
        raise ValueError("Alpha must be a finite rational or decimal probability")
    try:
        # A decimal float such as .05 means 5/100, not its binary expansion.
        alpha = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Alpha must be a finite rational or decimal probability") from exc
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie strictly between zero and one")
    return alpha


def _fraction_record(value: Fraction | None):
    if value is None:
        return None
    return {"exact": str(value), "decimal": float(value)}


@dataclass(frozen=True)
class JointZeroBound:
    population_units: int
    sample_units: int
    alpha: Fraction
    upper_count: int
    maximizing_allocation_at_upper: tuple[int, ...]
    probability_at_upper: Fraction
    first_excluded_count: int | None
    maximizing_allocation_at_first_excluded: tuple[int, ...] | None
    probability_at_first_excluded: Fraction | None

    def as_dict(self) -> dict:
        return {
            "method": "exact_stratified_zero_event_total_v1",
            "population_units": self.population_units,
            "sample_units": self.sample_units,
            "alpha": _fraction_record(self.alpha),
            "upper_count": self.upper_count,
            "maximizing_allocation_at_upper": list(self.maximizing_allocation_at_upper),
            "probability_at_upper": _fraction_record(self.probability_at_upper),
            "first_excluded_count": self.first_excluded_count,
            "maximizing_allocation_at_first_excluded": (
                None if self.maximizing_allocation_at_first_excluded is None
                else list(self.maximizing_allocation_at_first_excluded)
            ),
            "probability_at_first_excluded": _fraction_record(self.probability_at_first_excluded),
            "scope": "Total count of fixed detectable unit outcomes; no simultaneous per-stratum bounds.",
            "evidence_verified": False,
            "promotion_allowed": False,
        }


def joint_zero_upper_bound(
    strata: Iterable[tuple[int, int]],
    alpha: Fraction | float | str = Fraction(1, 20),
) -> JointZeroBound:
    """Return max M with worst-allocation P(all samples zero | M) >= alpha.

    The returned certificate records maximizing allocations and exact
    probabilities at the accepted/excluded boundary. Empty populations have
    mathematical upper bound zero. A stratum with n=0 contributes all its N
    units to the bound: this is conservative arithmetic, not an eligible A02
    audit. A census contributes no possible missed positives after zero events.

    Only the total bound has 1-alpha design coverage. It is not a conditional
    probability that a fixed population contains fewer positives. Complete
    zero-event reviews and the sampling assumptions must be checked elsewhere.
    With any observed positive, this zero-event-only result must not be used.
    """
    rows = _strata(strata)
    threshold = _alpha(alpha)
    population = sum(N for N, _ in rows)
    sample = sum(n for _, n in rows)

    # All n=0 marginal factors equal one. Credit those units in O(H) instead
    # of iterating over a potentially enormous entirely unobserved population.
    allocation = [N if n == 0 else 0 for N, n in rows]
    count = sum(allocation)
    probability = Fraction(1)
    heap = [(-Fraction(N - n, N), h) for h, (N, n) in enumerate(rows) if 0 < n < N]
    heapq.heapify(heap)

    def result(excluded_count, excluded_allocation, excluded_probability):
        return JointZeroBound(
            population, sample, threshold, count, tuple(allocation), probability,
            excluded_count, excluded_allocation, excluded_probability,
        )

    while heap:
        negative_ratio, h = heapq.heappop(heap)
        next_probability = probability * -negative_ratio
        if next_probability < threshold:
            excluded = allocation.copy()
            excluded[h] += 1
            return result(count + 1, tuple(excluded), next_probability)

        probability = next_probability
        allocation[h] += 1
        count += 1
        N, n = rows[h]
        m = allocation[h]
        if m < N - n:
            heapq.heappush(heap, (-Fraction(N - n - m, N - m), h))

    if count < population:
        # Any additional positive crosses one stratum's zero-event support.
        # Every allocation at this total therefore has probability zero.
        excluded = allocation.copy()
        h = next(h for h, (N, _) in enumerate(rows) if excluded[h] < N)
        excluded[h] += 1
        return result(count + 1, tuple(excluded), Fraction(0))
    return result(None, None, None)
