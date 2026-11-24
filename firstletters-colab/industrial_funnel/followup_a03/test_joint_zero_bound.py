"""Independent combinatorial oracle and design-coverage checks for the comparator."""
from fractions import Fraction
from itertools import product
from math import comb, prod
import unittest

from .joint_zero_bound import joint_zero_upper_bound


def allocation_probability(strata, allocation):
    """Direct hypergeometric combinatorics, with no greedy recurrence."""
    return prod((Fraction(comb(N - m, n), comb(N, n)) if N - m >= n else Fraction(0))
                for (N, n), m in zip(strata, allocation))


def enumerate_oracle(strata):
    """Enumerate all finite-population allocations, including zero-probability ones."""
    total = sum(N for N, _ in strata)
    maximum = [Fraction(0)] * (total + 1)
    for allocation in product(*(range(N + 1) for N, _ in strata)):
        M = sum(allocation)
        p0 = allocation_probability(strata, allocation)
        maximum[M] = max(maximum[M], p0)
    return maximum


def small_plans():
    # All ordered plans, including empty strata, nonaudited strata and censuses.
    yield ()
    for H, max_N in ((1, 9), (2, 6), (3, 4), (4, 2)):
        kinds = [(N, n) for N in range(max_N + 1) for n in range(N + 1)]
        yield from product(kinds, repeat=H)


class ExactJointZeroBoundTests(unittest.TestCase):
    def test_exhaustive_rational_oracle_and_all_threshold_jumps(self):
        plans = 0
        for strata in small_plans():
            oracle = enumerate_oracle(strata)
            # Test every possible threshold equality and every interval between
            # consecutive attainable probabilities, plus the intended 95% use.
            masses = sorted(set(oracle) | {Fraction(0), Fraction(1)})
            thresholds = {Fraction(1, 20), Fraction(1, 2)}
            thresholds.update(p for p in masses if 0 < p < 1)
            thresholds.update((a + b) / 2 for a, b in zip(masses, masses[1:]))
            for alpha in thresholds:
                q = joint_zero_upper_bound(strata, alpha)
                expected = max(M for M, p0 in enumerate(oracle) if p0 >= alpha)
                self.assertEqual(q.upper_count, expected, (strata, alpha))
                self.assertEqual(sum(q.maximizing_allocation_at_upper), expected)
                self.assertEqual(q.probability_at_upper, oracle[expected])
                self.assertEqual(allocation_probability(strata, q.maximizing_allocation_at_upper), oracle[expected])
                self.assertGreaterEqual(q.probability_at_upper, alpha)
                if expected < len(oracle) - 1:
                    self.assertEqual(q.first_excluded_count, expected + 1)
                    self.assertEqual(sum(q.maximizing_allocation_at_first_excluded), expected + 1)
                    self.assertEqual(q.probability_at_first_excluded, oracle[expected + 1])
                    self.assertEqual(allocation_probability(strata, q.maximizing_allocation_at_first_excluded), oracle[expected + 1])
                    self.assertLess(q.probability_at_first_excluded, alpha)
                else:
                    self.assertIsNone(q.first_excluded_count)
                    self.assertIsNone(q.probability_at_first_excluded)
            plans += 1
        self.assertEqual(plans, 5511)

    def test_frequentist_noncoverage_by_all_allocations(self):
        # This is the whole procedure: use U0 for an all-zero observation and
        # the trivial population bound otherwise. Noncoverage is possible only
        # on the all-zero event and is checked for every fixed allocation.
        for strata in (((5, 1), (4, 2)), ((3, 0), (4, 2), (2, 2)), ((4, 1),) * 3):
            for alpha in (Fraction(1, 20), Fraction(1, 3), Fraction(1, 2)):
                q = joint_zero_upper_bound(strata, alpha)
                for allocation in product(*(range(N + 1) for N, _ in strata)):
                    noncoverage = allocation_probability(strata, allocation) if sum(allocation) > q.upper_count else Fraction(0)
                    self.assertLess(noncoverage, alpha)

    def test_monotone_in_sampling_and_confidence(self):
        for populations in ((5,), (4, 5), (2, 3, 4)):
            for samples in product(*(range(N + 1) for N in populations)):
                strata = tuple(zip(populations, samples))
                upper = joint_zero_upper_bound(strata).upper_count
                for h, N in enumerate(populations):
                    if samples[h] < N:
                        increased = list(samples)
                        increased[h] += 1
                        self.assertLessEqual(joint_zero_upper_bound(zip(populations, increased)).upper_count, upper)
                self.assertGreaterEqual(joint_zero_upper_bound(strata, "0.01").upper_count, upper)
                self.assertLessEqual(joint_zero_upper_bound(strata, "0.1").upper_count, upper)

    def test_stratum_permutation_and_maximizing_allocation_ties(self):
        strata = ((12, 2), (10, 3), (7, 0), (8, 8))
        q = joint_zero_upper_bound(strata)
        r = joint_zero_upper_bound(reversed(strata))
        self.assertEqual(q.upper_count, r.upper_count)
        self.assertEqual(q.probability_at_upper, r.probability_at_upper)
        tied = joint_zero_upper_bound(((20, 3),) * 6)
        self.assertEqual(tied.upper_count, 17)
        self.assertLessEqual(max(tied.maximizing_allocation_at_upper) - min(tied.maximizing_allocation_at_upper), 1)

    def test_census_and_unobserved_and_empty_boundaries(self):
        for strata, expected in (((), 0), (((0, 0),), 0), (((10, 10),), 0),
                                 (((10, 0), (20, 20)), 10), (((10, 0),), 10),
                                 (((10**12, 0), (20, 3)), 10**12 + 11)):
            q = joint_zero_upper_bound(strata)
            self.assertEqual(q.upper_count, expected)
            self.assertGreaterEqual(q.upper_count, sum(N for N, n in strata if n == 0))
        q = joint_zero_upper_bound(((10, 10),))
        self.assertEqual(q.probability_at_first_excluded, 0)
        self.assertEqual(q.first_excluded_count, 1)
        q = joint_zero_upper_bound(((10**12, 0),))
        self.assertIsNone(q.first_excluded_count)

    def test_exact_threshold_equality_and_decimal_alpha(self):
        # For N=4,n=1,M=3 the exact zero-event probability is 1/4.
        self.assertEqual(joint_zero_upper_bound(((4, 1),), "0.25").upper_count, 3)
        self.assertEqual(joint_zero_upper_bound(((4, 1),), Fraction(250001, 1000000)).upper_count, 2)
        self.assertEqual(joint_zero_upper_bound(((120, 18),), 0.05).alpha, Fraction(1, 20))

    def test_known_a02_design_values_and_evidence_status(self):
        q = joint_zero_upper_bound(((20, 3),) * 2)
        self.assertEqual(q.upper_count, 14)
        self.assertEqual(q.maximizing_allocation_at_upper, (7, 7))
        self.assertEqual(joint_zero_upper_bound(((120, 18),)).upper_count, 17)
        self.assertFalse(q.as_dict()["evidence_verified"])
        self.assertFalse(q.as_dict()["promotion_allowed"])

    def test_pooling_unequal_sampling_fractions_is_invalid(self):
        strata = ((100, 1), (20, 20))
        q = joint_zero_upper_bound(strata)
        self.assertEqual(q.upper_count, 95)
        self.assertEqual(q.probability_at_upper, Fraction(1, 20))
        self.assertEqual(allocation_probability(strata, (50, 0)), Fraction(1, 2))
        pooled = joint_zero_upper_bound(((120, 21),))
        self.assertEqual(pooled.upper_count, 14)
        # This proves the tempting pooled answer would exclude the true total
        # 50 with probability 1/2 under the actual stratified sampling design.
        self.assertLess(pooled.upper_count, 50)

    def test_invalid_parameters_fail_instead_of_silent_coercion(self):
        for strata in (None, [1], [(1,)], [(1, 0, 0)], [(-1, 0)], [(2, -1)], [(2, 3)], [(True, 0)], [(2, False)], [(2.0, 1)]):
            with self.assertRaises(ValueError):
                joint_zero_upper_bound(strata)
        for alpha in (0, 1, -0.1, True, None, float("nan"), float("inf"), "1/0", "nan"):
            with self.assertRaises(ValueError):
                joint_zero_upper_bound(((2, 1),), alpha)


if __name__ == "__main__":
    unittest.main()
