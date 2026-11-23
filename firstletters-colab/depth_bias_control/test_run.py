"""Synthetic projection tests; no checkpoint, real archive, CT or image access."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("depth_bias_test_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ComparisonTests(unittest.TestCase):
    def test_one_ulp_mismatch_stops_even_when_Q8_is_identical(self):
        first = np.full((2, 2), .5, np.float32)
        second = first.copy()
        second[0, 0] = np.nextafter(second[0, 0], np.float32(1))
        np.testing.assert_array_equal(runner.q8(first), runner.q8(second))
        with self.assertRaises(runner.BaselineMismatch):
            runner.require_probability_equal(first, second)
        runner.require_probability_equal(first, first.copy())

    def test_fixed_strict_crossings_and_support_populations(self):
        first = np.array([[0, .5, .51, 1]], np.float32)
        second = np.array([[0, .51, .49, 1]], np.float32)
        support = np.array([[False, True, True, True]])
        result = runner.probability_changes(first, second, support)
        self.assertEqual(result["all_core"]["pixels"], 4)
        self.assertEqual(result["nonempty_CT_columns"]["pixels"], 3)
        self.assertEqual(result["all_core"]["upward_0_5_crossings"], 1)
        self.assertEqual(result["all_core"]["downward_0_5_crossings"], 1)
        self.assertEqual(result["all_core"]["changed_Q8_pixels"], 2)

    def test_source_depth_maps_keep_the_odd_slab_difference(self):
        self.assertEqual((runner.source_planes("source")[0], runner.source_planes("source")[-1]), (23, 84))
        self.assertEqual((runner.source_planes("reverse")[0], runner.source_planes("reverse")[-1]), (85, 24))
        self.assertFalse(np.array_equal(runner.source_planes("source")[::-1], runner.source_planes("reverse")))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch needed for generated projection tests")
class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(4)

    def test_bias_changes_known_depth_gradient_and_uniform_effective_count(self):
        coordinate = (np.arange(62, dtype=np.float32)-31)/31
        logits = np.broadcast_to((4*coordinate)[:, None, None], (62, 64, 64)).copy()
        attention = np.zeros_like(logits)
        support = np.ones((256, 256), bool)
        zero, zero_stats = runner.project(logits, attention, support, 0., runner.source_planes("source"), self.torch)
        biased, biased_stats = runner.project(logits, attention, support, 2., runner.source_planes("source"), self.torch)
        expected_zero = 1/(1+np.exp(-np.mean(4*coordinate.astype(np.float64))))
        self.assertAlmostEqual(float(zero[0, 0]), float(expected_zero), places=6)
        self.assertTrue((biased > zero).all())
        np.testing.assert_allclose(zero_stats["effective_plane_count"], 62., atol=1e-10, rtol=0)
        np.testing.assert_allclose(zero_stats["mean_depth_relative_to_plane54_um"], -1.2, atol=1e-10, rtol=0)
        self.assertTrue((biased_stats["effective_plane_count"] < 62).all())

    def test_empty_support_does_not_receive_depth_weights(self):
        logits = np.full((62, 64, 64), 10., np.float32)
        probability, stats = runner.project(logits, np.zeros_like(logits), np.zeros((256, 256), bool), 1.,
                                             runner.source_planes("reverse"), self.torch)
        self.assertTrue((probability == .5).all())  # Outer assembly then zeros these columns.
        self.assertEqual(stats["entropy_nats"].size, 0)

    def test_one_supported_column_pools_one_internal_cell_without_mutation(self):
        logits = np.zeros((62, 64, 64), np.float32)
        attention = np.zeros_like(logits)
        support = np.zeros((256, 256), bool)
        support[1, 1] = True
        original = (logits.copy(), attention.copy(), support.copy())
        first, stats = runner.project(logits, attention, support, 0., runner.source_planes("source"), self.torch)
        second, _ = runner.project(logits, attention, support, 0., runner.source_planes("source"), self.torch)
        runner.require_probability_equal(first, second)
        self.assertEqual(stats["entropy_nats"].size, 1)
        for before, after in zip(original, (logits, attention, support)):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
