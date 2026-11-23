"""Synthetic serialization/replay checks only; no source arrays or model calls."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prediction_precision_test", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


class PrecisionTests(unittest.TestCase):
    def test_q16_distinguishes_values_hidden_by_q8(self):
        fine = np.full((8, 8), .9997, np.float32)
        fine[:4] = .9999
        q8 = np.rint(fine * np.float32(255)).astype(np.uint8)
        q16 = runner.quantize16(fine)
        self.assertTrue((q8 == 255).all())
        self.assertGreater(int(q16[0, 0]), int(q16[-1, 0]))
        self.assertLess(int(q16[0, 0]), 65535)

    def test_quantization_precedes_common_reduction_and_preserves_sum_ties(self):
        fine = np.zeros((4, 8), np.float32)
        fine[:2, :4] = .49 / 65535
        fine[2:, :4] = 1.49 / 65535
        first = np.array([65, 32, 36, 235, 71, 51, 180, 236,
                          242, 195, 8, 124, 249, 24, 82, 29], np.uint16)
        second = first.copy()
        second[:2] = (66, 31)
        exported = np.zeros((4, 8), np.uint16)
        exported[:, :4], exported[:, 4:] = first.reshape(4, 4), second.reshape(4, 4)
        common = runner.integer_common(exported)
        self.assertEqual(common[0, 0], common[0, 1])
        self.assertEqual(common[0, 0], np.float32(1859 / (16 * 65535)))
        self.assertAlmostEqual(float(runner.integer_common(runner.quantize16(fine))[0, 0]), .5 / 65535)

    def test_q16_all_maximum_does_not_overflow_integer_sum(self):
        result = runner.integer_common(np.full((4, 4), 65535, np.uint16))
        self.assertEqual(float(result[0, 0]), 1.)
        self.assertEqual(float(runner.integer_common(np.zeros((4, 4), np.uint16))[0, 0]), 0.)

    def test_fixed_display_is_global_and_pointwise(self):
        first = np.array([[0., .5], [.9999, 1.]], np.float32)
        grey, png = runner.fixed_display(first)
        self.assertEqual(float(grey[0, 0]), 0.)
        self.assertEqual(int(png[0, 1]), 85)
        self.assertLess(float(grey[1, 0]), float(grey[1, 1]))
        self.assertLess(float(grey[1, 1]), 1.)
        changed = np.full((17, 19), .5, np.float32)
        self.assertTrue((runner.fixed_display(changed)[1] == png[0, 1]).all())
        self.assertTrue(np.array_equal(first, np.array([[0., .5], [.9999, 1.]], np.float32)))

    def test_fp32_hann_replay_and_source_support(self):
        support = np.ones((6, 6), bool)
        support[3, 3] = False
        patches = np.stack([np.full((4, 4), value, np.float32) for value in (.1, .3, .5, .7)])
        result = runner.assemble_hann(patches, support, np.ones((4, 4), np.float32), 2)
        self.assertAlmostEqual(float(result[2, 2]), .4)
        self.assertEqual(float(result[3, 3]), 0.)
        self.assertEqual(float(result[0, 0]), float(patches[0, 0, 0]))
        self.assertEqual(float(result[-1, -1]), float(patches[-1, -1, -1]))
        with self.assertRaises(ValueError):
            runner.assemble_hann(patches[:-1], support, np.ones((4, 4), np.float32), 2)

    def test_restoration_means_absolute_agreement_not_just_improvement(self):
        settings = PROTOCOL["controls"]["restoration"]
        baseline = {"recall_at_background_5pct": .5, "ap": .8}
        self.assertTrue(runner.restoration(baseline, dict(baseline), settings)["pass"])
        for altered in ({"recall_at_background_5pct": .52, "ap": .8},
                        {"recall_at_background_5pct": .48, "ap": .8},
                        {"recall_at_background_5pct": .5, "ap": .79}):
            self.assertFalse(runner.restoration(baseline, altered, settings)["pass"])

    def test_nonfinite_and_invalid_export_inputs_fail(self):
        with self.assertRaises(ValueError):
            runner.quantize16(np.full((4, 4), np.nan, np.float32))
        with self.assertRaises(ValueError):
            runner.quantize16(np.full((4, 4), 2., np.float32))
        with self.assertRaises(ValueError):
            runner.integer_common(np.zeros((4, 4), np.int16))
        with self.assertRaises(ValueError):
            runner.integer_common(np.zeros((3, 4), np.uint16))


if __name__ == "__main__":
    unittest.main()
