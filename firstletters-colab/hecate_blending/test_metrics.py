import importlib.util
import copy
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location("hecate_blending_metrics", Path(__file__).with_name("metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def sparse_fixture():
    labels = np.zeros((256, 256), np.uint8)
    labels[31:33, 31:33] = 2
    labels[63:65, 63:65] = 1
    score = np.full((256, 256), .1, np.float32)
    score[31:33, 31:33] = .9
    score[32, 32] = .05
    return score, labels, np.ones((256, 256), bool)


class BlendingMetricTests(unittest.TestCase):
    def test_true_seam_connections_and_missing_endpoint(self):
        score, labels, support = sparse_fixture()
        result = metrics.field_metrics(score, labels, support)
        pairs = result["pairs"]["seam_ink"]
        self.assertEqual(pairs["eligible_pairs"], 4)
        self.assertEqual(pairs["both_positive_pairs"], 2)
        self.assertEqual(pairs["exactly_one_positive_pairs"], 2)
        self.assertEqual(pairs["both_negative_pairs"], 0)
        self.assertEqual(pairs["both_positive_fraction"], .5)
        self.assertEqual(pairs["by_axis"]["horizontal"]["eligible_pairs"], 2)
        self.assertEqual(pairs["by_axis"]["vertical"]["eligible_pairs"], 2)
        self.assertEqual(result["pairs"]["seam_background"]["eligible_pairs"], 4)
        self.assertEqual(result["pairs"]["seam_background"]["both_positive_fraction"], 0.)
        self.assertIsNone(result["pairs"]["within_cell_ink"]["both_positive_fraction"])
        self.assertEqual(result["recall_at_background_5pct"], .75)

    def test_support_excludes_pairs_with_either_endpoint_missing(self):
        score, labels, support = sparse_fixture()
        support[31, 31] = False
        score[31, 31] = np.nan
        result = metrics.field_metrics(score, labels, support)
        self.assertEqual(result["pairs"]["seam_ink"]["eligible_pairs"], 2)
        self.assertEqual(result["pairs"]["seam_ink"]["both_positive_pairs"], 0)
        self.assertEqual(result["ink"], 3)
        self.assertEqual(result["unsupported"], 1)

    def test_seven_internal_lines_per_axis_without_wraparound(self):
        labels = np.full((256, 256), 2, np.uint8)
        labels[0, 0] = 1
        score = np.full((256, 256), .9, np.float32)
        score[0, 0] = .1
        result = metrics.field_metrics(score, labels, np.ones((256, 256), bool))
        self.assertEqual(result["pairs"]["seam_ink"]["eligible_pairs"], 2 * 7 * 256)
        # There are2*256*255 total neighbour edges; two touch the single background pixel.
        self.assertEqual(result["pairs"]["within_cell_ink"]["eligible_pairs"], 2 * 256 * 255 - 2 * 7 * 256 - 2)
        self.assertEqual(result["pairs"]["seam_ink"]["both_positive_fraction"], 1.)

    def test_whole_ties_and_actual_background_budget(self):
        labels = np.zeros((256, 256), np.uint8)
        score = np.zeros((256, 256), np.float32)
        labels.flat[:10] = 2
        labels.flat[10:110] = 1
        score.flat[:8] = .9
        score.flat[8:10] = .1
        score.flat[10:14] = .8
        score.flat[14:110] = .2
        result = metrics.field_metrics(score, labels, np.ones((256, 256), bool))
        self.assertEqual(result["background"], 100)
        self.assertEqual(result["allowed_background_count"], 5)
        self.assertEqual(result["threshold_background_rank_zero_based"], 5)
        self.assertEqual(result["background_positive_count"], 4)
        self.assertEqual(result["background_tied_at_threshold_count"], 96)
        self.assertEqual(result["unused_background_budget_due_to_ties"], 1)
        self.assertEqual(result["background_exceedance"], .04)
        self.assertEqual(result["recall_at_background_5pct"], .8)

    def test_integer_floor_budget_at_small_sample_boundary(self):
        labels = np.zeros((256, 256), np.uint8)
        score = np.zeros((256, 256), np.float64)
        labels[0, 0] = 2
        score[0, 0] = 1.
        for count, expected_budget in ((19, 0), (20, 1), (39, 1), (40, 2)):
            labels.flat[1:] = 0
            labels.flat[1:count + 1] = 1
            score.flat[1:count + 1] = np.arange(count) / 100
            result = metrics.field_metrics(score, labels, np.ones((256, 256), bool))
            self.assertEqual(result["allowed_background_count"], expected_budget)
            self.assertEqual(result["background_positive_count"], expected_budget)

    def test_strict_monotone_transform_preserves_rank_and_pair_endpoints(self):
        rng = np.random.default_rng(219)
        score = rng.uniform(.1, .9, (256, 256))
        labels = np.where(np.indices((256, 256))[0] < 128, 1, 2).astype(np.uint8)
        support = np.ones((256, 256), bool)
        first = metrics.field_metrics(score, labels, support)
        second = metrics.field_metrics(score ** 2, labels, support)
        for name in ("ap", "auc", "recall_at_background_5pct", "background_positive_count", "pairs"):
            self.assertEqual(first[name], second[name])
        self.assertAlmostEqual(second["threshold_strict_greater"], first["threshold_strict_greater"] ** 2)

    def test_unknown_or_background_bridge_is_never_true_ink_pair(self):
        score, labels, support = sparse_fixture()
        labels[31, 32] = 0
        labels[32, 31] = 1
        score[31:33, 31:33] = .99
        result = metrics.field_metrics(score, labels, support)
        self.assertEqual(result["pairs"]["seam_ink"]["eligible_pairs"], 0)
        self.assertIsNone(result["pairs"]["seam_ink"]["both_positive_fraction"])

    def test_quantization_happens_before_area_reduction(self):
        fine = np.zeros((1024, 1024), np.float32)
        fine[:2, :4] = .49 / 255
        fine[2:4, :4] = 1.49 / 255
        result = metrics.common_grid(fine, uint8_export=True)
        self.assertAlmostEqual(float(result[0, 0]), .5 / 255)
        wrong_order = np.rint(metrics.common_grid(fine)[0, 0] * 255) / 255
        self.assertNotEqual(float(result[0, 0]), float(wrong_order))

    def test_export_ties_are_retained_and_not_interpolated_away(self):
        labels = np.where(np.indices((256, 256))[0] < 128, 1, 2).astype(np.uint8)
        common = np.where(labels == 2, .5002, .5001).astype(np.float32)
        fine = common.repeat(4, axis=0).repeat(4, axis=1)
        result = metrics.evaluate_fine_core(fine, labels, np.ones((256, 256), bool))
        self.assertEqual(result["float_probability"]["recall_at_background_5pct"], 1.)
        self.assertEqual(result["uint8_export"]["recall_at_background_5pct"], 0.)
        self.assertEqual(result["uint8_export"]["auc"], .5)
        self.assertEqual(result["uint8_export"]["background_tied_at_threshold_count"], 32768)
        self.assertEqual(result["uint8_export"]["unused_background_budget_due_to_ties"], 32768 // 20)

    def test_equal_export_sums_remain_exactly_tied(self):
        # Dividing each uint8 pixel in float32 before averaging breaks this tie
        # by one final-float32 ULP. Such a fake rank change can alter strict cuts.
        first = np.array([65, 32, 36, 235, 71, 51, 180, 236,
                          242, 195, 8, 124, 249, 24, 82, 29], np.uint8)
        second = first.copy()
        second[:2] = (66, 31)
        self.assertEqual(int(first.sum()), 1859)
        self.assertEqual(int(second.sum()), 1859)
        fine = np.zeros((1024, 1024), np.float32)
        fine[:4, :4] = first.reshape(4, 4).astype(np.float32) / 255
        fine[:4, 4:8] = second.reshape(4, 4).astype(np.float32) / 255
        result = metrics.common_grid(fine, uint8_export=True)
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertEqual(result[0, 0], np.float32(1859 / (16 * 255)))

    def test_invalid_or_missing_classes_fail_explicitly(self):
        score, labels, support = sparse_fixture()
        cases = [
            (score[:128], labels[:128], support[:128]),
            (score, labels, support.astype(np.uint8)),
            (score, labels.astype(float), support),
            (score, np.where(labels == 2, 3, labels), support),
            (score, np.ones_like(labels), support),
            (score, labels, np.zeros_like(support)),
        ]
        for args in cases:
            with self.assertRaises(ValueError):
                metrics.field_metrics(*args)
        for bad in (np.nan, np.inf, -1., 1.01):
            changed = score.copy()
            changed[0, 0] = bad  # Even a supported unknown-label cell must be finite/in range.
            with self.assertRaises(ValueError):
                metrics.field_metrics(changed, labels, support)
        with self.assertRaises(ValueError):
            metrics.common_grid(np.zeros((1023, 1024)))
        with self.assertRaises(ValueError):
            metrics.common_grid(np.full((1024, 1024), np.nan))

    def test_new_decision_rejects_cosmetic_gain_regression_and_small_pools(self):
        base = {
            "shape": [256, 256], "seam_period": 32, "known": 1000,
            "ink": 500, "background": 500, "unsupported": 0,
            "supported_unknown": 64536, "ap": .8,
            "recall_at_background_5pct": .7,
            "pairs": {name: {"eligible_pairs": 100, "both_positive_fraction": fraction}
                      for name, fraction in (("seam_ink", .5), ("seam_background", .02), ("within_cell_ink", .6))},
        }
        improved = copy.deepcopy(base)
        improved["pairs"]["seam_ink"]["both_positive_fraction"] = .55
        self.assertTrue(metrics.compare(base, improved)["pass"])
        near = copy.deepcopy(improved)
        near["pairs"]["seam_ink"]["both_positive_fraction"] = np.nextafter(.55, 0.)
        self.assertFalse(metrics.compare(base, near)["pass"])
        for endpoint in ("ap", "recall_at_background_5pct"):
            bad = copy.deepcopy(improved)
            bad[endpoint] -= .02
            self.assertFalse(metrics.compare(base, bad)["pass"])
        for endpoint, bad_value in (("seam_background", .04), ("within_cell_ink", .58)):
            bad = copy.deepcopy(improved)
            bad["pairs"][endpoint]["both_positive_fraction"] = bad_value
            self.assertFalse(metrics.compare(base, bad)["pass"])
        small_base, small_improved = copy.deepcopy(base), copy.deepcopy(improved)
        for item in (small_base, small_improved):
            item["pairs"]["seam_ink"]["eligible_pairs"] = 19
        self.assertEqual(metrics.compare(small_base, small_improved)["reason"], "insufficient_eligible_labelled_pairs")
        mismatched = copy.deepcopy(improved)
        mismatched["ink"] += 1
        with self.assertRaises(ValueError):
            metrics.compare(base, mismatched)

    def test_seam_period_is_explicit_and_validated(self):
        score, labels, support = sparse_fixture()
        for invalid in (0, 7, 256, 32., True):
            with self.assertRaises(ValueError):
                metrics.measure(score, labels, support, seam_period=invalid)
        result = metrics.measure(score, labels, support, seam_period=64)
        self.assertEqual(result["pairs"]["seam_ink"]["eligible_pairs"], 0)
        self.assertEqual(result["pairs"]["within_cell_ink"]["eligible_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
