"""Generated numerical fixtures only; no public cases, model maps or acquisition."""
import json
import unittest

import numpy as np

from statistics import FEATURES, SHAPE, _RankedScore, _labels, _translation_offsets, evaluate


def pairwise_auc(scores, labels, weights=None):
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    denominator = weights[positive].sum() * weights[negative].sum()
    if denominator == 0:
        return float("nan")
    return sum(weights[p] * weights[n] * ((scores[p] > scores[n]) + .5 * (scores[p] == scores[n]))
               for p in positive for n in negative) / denominator


class AucTests(unittest.TestCase):
    def test_exact_ties_exclusion_and_fixed_orientation(self):
        score = np.array([0., 1., 1., 2., 9.])
        labels = np.array([0, 1, 0, 1, -1])
        self.assertEqual(_RankedScore(score).auc(labels), .875)
        self.assertEqual(_RankedScore(-score).auc(labels), .125)
        self.assertEqual(_RankedScore(np.ones(4)).auc(np.array([0, 1, 0, 1])), .5)
        nearby = np.array([1., np.nextafter(1., 2.)])
        self.assertEqual(_RankedScore(nearby).auc(np.array([0, 1])), 1.)

    def test_integer_weighting_equals_explicit_block_repetition(self):
        score = np.array([2., 1., 1., 3., 0., 4.])
        labels = np.array([1, 0, 1, 0, -1, 1])
        weights = np.array([2, 3, 1, 0, 4, 2])
        result = _RankedScore(score).auc(labels, weights)
        self.assertAlmostEqual(result, pairwise_auc(score, labels, weights))
        self.assertEqual(result, _RankedScore(np.repeat(score, weights)).auc(np.repeat(labels, weights)))

    def test_weighted_pairwise_reference_and_empty_classes(self):
        rng = np.random.default_rng(41)
        for _ in range(20):
            scores = rng.integers(-2, 3, 17)
            labels = np.resize(np.array([-1, 0, 1]), 17)
            weights = rng.random(17)
            self.assertAlmostEqual(_RankedScore(scores).auc(labels, weights), pairwise_auc(scores, labels, weights))
        rank = _RankedScore(np.array([0., 1.]))
        self.assertTrue(np.isnan(rank.auc(np.array([0, 1]), [1, 0])))
        self.assertTrue(np.isnan(_RankedScore(np.array([])).auc(np.array([]))))
        with self.assertRaises(ValueError):
            rank.auc(np.array([0, 1]), [1, -1])

    def test_reference_thresholds(self):
        np.testing.assert_array_equal(_labels(np.array([0, 1, 64, 65, 178, 179, 255], np.uint8)),
                                      [-1, 0, 0, -1, -1, 1, 1])

    def test_unique_nonzero_uniform_population_and_roll_reproducibility(self):
        first = _translation_offsets((3, 4), 11, np.random.default_rng(83))
        second = _translation_offsets((3, 4), 11, np.random.default_rng(83))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(map(tuple, first)), {(y, x) for y in range(3) for x in range(4)} - {(0, 0)})
        reference = np.arange(12).reshape(3, 4)
        for dy, dx in first:
            expected = np.array([[reference[(y - dy) % 3, (x - dx) % 4] for x in range(4)] for y in range(3)])
            np.testing.assert_array_equal(np.roll(reference, (dy, dx), axis=(0, 1)), expected)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def cases(empty=False):
        y, x = np.indices(SHAPE)
        positive = (x + y) % 2 == 0
        reference = np.where(positive, 200, 30).astype(np.uint8)
        if empty:
            reference[:] = 0
        features = {FEATURES[0]: np.zeros(SHAPE), FEATURES[1]: positive.astype(float),
                    FEATURES[2]: (~positive).astype(float)}
        return [{"name": f"generated_{i}", "features": features, "valid": np.ones(SHAPE, bool),
                 "reference": reference} for i in range(3)]

    def test_paired_macro_polarity_counts_and_seed_reproducibility(self):
        cases = self.cases()
        cases[0]["features"] = dict(cases[0]["features"])
        cases[0]["features"][FEATURES[1]] = cases[0]["features"][FEATURES[1]].copy()
        cases[0]["features"][FEATURES[1]][0, 0] = np.nan
        first = evaluate(cases, seed=19, null_replicates=7, bootstrap_replicates=5)
        second = evaluate(cases, seed=19, null_replicates=7, bootstrap_replicates=5)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "assessable")
        json.dumps(first, allow_nan=False)
        self.assertEqual(first["observed"]["cases"][0]["common_valid_count"], np.prod(SHAPE) - 1)
        self.assertEqual(first["observed"]["macro_auc"], dict(zip(FEATURES, [.5, 1., 1.])))
        np.testing.assert_allclose(first["bootstrap"]["macro_delta_vs_raw"], .5)
        for feature in FEATURES[1:]:
            self.assertEqual(first["bootstrap"]["percentile_interval"][feature], [.5, .5])
            expected = (1 + np.count_nonzero(np.array(first["null"]["family_max_auc"]) >= 1.)) / 8
            self.assertEqual(first["null"]["spatial_rank"][feature], expected)
        self.assertEqual(np.array(first["null"]["case_auc"]).shape, (7, 3, 3))
        self.assertEqual(np.array(first["bootstrap"]["resampled_block_indices"]).shape, (3, 5, 20))
        for offsets in first["null"]["offsets_yx"]:
            self.assertEqual(len(set(map(tuple, offsets))), 7)
            self.assertNotIn([0, 0], offsets)

    def test_no_empty_class_redraw_or_hidden_drop(self):
        result = evaluate(self.cases(empty=True), null_replicates=3, bootstrap_replicates=2)
        self.assertEqual(result["status"], "unassessable")
        self.assertEqual(len(result["failures"]), 3 + 3 * 3 + 2 * 3)
        self.assertEqual(result["null"]["family_max_auc"], [None] * 3)
        self.assertEqual(result["bootstrap"]["macro_delta_vs_raw"], [[None, None]] * 2)
        for feature in FEATURES[1:]:
            self.assertIsNone(result["null"]["spatial_rank"][feature])
            self.assertEqual(result["bootstrap"]["percentile_interval"][feature], [None, None])
            self.assertFalse(result["gates"][feature]["pass"])
        json.dumps(result, allow_nan=False)

    def test_macro_has_equal_case_weight_despite_unequal_pixel_counts(self):
        cases = self.cases()
        for index, width in enumerate((16, 4, 8)):
            cases[index]["valid"] = np.zeros(SHAPE, bool)
            cases[index]["valid"][0, :width] = True
            features = dict(cases[index]["features"])
            features[FEATURES[1]] = (features[FEATURES[1]] if index == 0 else
                                     1 - features[FEATURES[1]] if index == 1 else np.zeros(SHAPE))
            cases[index]["features"] = features
        result = evaluate(cases, null_replicates=2, bootstrap_replicates=2)
        self.assertEqual([case["auc"][FEATURES[1]] for case in result["observed"]["cases"]], [1., 0., .5])
        self.assertEqual(result["observed"]["macro_auc"][FEATURES[1]], .5)
        self.assertEqual(result["observed"]["macro_delta_vs_raw"][FEATURES[1]], 0.)

    def test_empty_shift_does_not_disappear_when_observed_is_assessable(self):
        cases = self.cases(empty=True)
        for case in cases:
            case["valid"] = np.zeros(SHAPE, bool)
            case["valid"][0, :2] = True
            case["reference"][0, :2] = [200, 30]
        result = evaluate(cases, seed=77, null_replicates=3, bootstrap_replicates=2)
        self.assertTrue(all(case["ink_count"] == case["blank_count"] == 1 for case in result["observed"]["cases"]))
        self.assertEqual(result["status"], "unassessable")
        self.assertEqual(len(result["null"]["case_auc"]), 3)
        self.assertTrue(any(failure["stage"] == "null" for failure in result["failures"]))
        self.assertTrue(all(rank is None for rank in result["null"]["spatial_rank"].values()))

    def test_fixed_shape_case_count_and_block_definition(self):
        with self.assertRaisesRegex(ValueError, "three"):
            evaluate([])
        with self.assertRaisesRegex(ValueError, "block_size=128"):
            evaluate(self.cases(), block_size=64)
        cases = self.cases()
        cases[0]["valid"] = np.ones((2, 2), bool)
        with self.assertRaisesRegex(ValueError, "512, 640"):
            evaluate(cases)


if __name__ == "__main__":
    unittest.main()
