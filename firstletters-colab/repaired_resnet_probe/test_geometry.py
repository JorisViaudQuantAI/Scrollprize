"""Generated geometry/arithmetic checks; no real arrays, downloads, or models."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("resnet_probe_geometry", Path(__file__).with_name("run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class GeometryTests(unittest.TestCase):
    def test_fixed_depth_and_xy_bounds(self):
        depth, rows = probe.axis_overlap(64, 18)
        self.assertEqual((depth.start, depth.stop), (2, 62))
        self.assertEqual(len(rows), 18)
        xy, rows = probe.axis_overlap(1152, 256)
        self.assertEqual((xy.start, xy.stop), (154, 998))
        self.assertEqual(len(rows), 256)
        for indices, weights in rows:
            self.assertTrue(np.all(weights > 0))
            self.assertEqual(int(weights.sum()), 79)
            self.assertGreaterEqual(int(indices.min()), 0)
            self.assertLess(int(indices.max()), xy.stop - xy.start)

    def test_exact_interval_overlap_against_fractions(self):
        for n, m in ((64, 18), (1152, 256), (32, 6)):
            source, rows = probe.axis_overlap(n, m)
            for j, (indices, weights) in enumerate(rows):
                left = Fraction(n, 2) + Fraction(79, 24) * (j - m // 2)
                right = left + Fraction(79, 24)
                exact = [max(Fraction(0), min(right, int(i) + source.start + 1) - max(left, int(i) + source.start)) for i in indices]
                self.assertEqual(exact, [Fraction(int(w), 24) for w in weights])

    def test_three_axis_constant_is_preserved(self):
        array = np.full((32, 32, 32), 173, dtype=np.uint8)
        specs = [probe.axis_overlap(32, 6) for _ in range(3)]
        value = array[tuple(s for s, _ in specs)]
        for axis, (_, rows) in enumerate(specs):
            value = probe.weighted_axis(value, rows, axis)
        self.assertTrue(np.all(value == 173 * 79 ** 3))
        self.assertTrue(np.all(probe.divide_nearest_even(value, 79 ** 3) == 173))

    def test_nearest_even_and_support_rejection(self):
        np.testing.assert_array_equal(probe.divide_nearest_even(np.array([1, 3, 5, 7]), 2), [0, 2, 2, 4])
        with self.assertRaises(ValueError):
            probe.axis_overlap(64, 20)

    def test_original_label_centres_independently(self):
        for origin in (0, 17, 160, 928, 4096):
            expected = []
            for j in range(16, 48):
                centre_um = Fraction(48, 5) * (origin + 128) + Fraction(79, 5) * (2 * j + 1 - 64)
                label_cell = centre_um / Fraction(234, 25)
                expected.append(label_cell.numerator // label_cell.denominator)
            np.testing.assert_array_equal(probe.label_centres(origin), expected)
            local = np.arange(32, dtype=np.int64)
            np.testing.assert_array_equal(probe.label_centres(origin), (240 * origin + 18475 + 790 * local) // 234)

    def test_physical_scoring_aperture(self):
        fine_pitch = Fraction(12, 5)
        target_pitch = Fraction(79, 10)
        start = Fraction(464, 3)
        core_left = start + 64 * Fraction(79, 24)
        core_right = start + 192 * Fraction(79, 24)
        self.assertEqual((core_left, core_right), (Fraction(1096, 3), Fraction(2360, 3)))
        self.assertEqual((core_right - core_left) * fine_pitch, Fraction(5056, 5))
        self.assertEqual(18 * target_pitch, Fraction(711, 5))
        self.assertEqual((64 * fine_pitch - 18 * target_pitch) / 2, Fraction(57, 10))


if __name__ == "__main__":
    unittest.main()
