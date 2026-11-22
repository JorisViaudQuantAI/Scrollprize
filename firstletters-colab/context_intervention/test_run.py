import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("context_intervention_tests", Path(__file__).with_name("run.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class InterventionTests(unittest.TestCase):
    def test_core_unchanged_and_halo_definitions(self):
        raw = np.arange(16 * 64 * 64, dtype=np.uint16).reshape(16, 64, 64).astype(np.uint8)
        original = raw.copy()
        for name in r.CONDITIONS:
            out = r.intervention(raw, name)
            np.testing.assert_array_equal(out[:, 16:48, 16:48], raw[:, 16:48, 16:48])
        zero = r.intervention(raw, "zero_halo")
        self.assertFalse(zero[:, :16].any())
        self.assertFalse(zero[:, 48:].any())
        self.assertFalse(zero[:, :, :16].any())
        reflected = r.intervention(raw, "reflect_halo")
        np.testing.assert_array_equal(reflected[:, 15, 16:48], raw[:, 17, 16:48])
        np.testing.assert_array_equal(raw, original)

    def test_wrong_shape_dtype_condition_fail(self):
        for x in (np.ones((16, 32, 32), np.uint8), np.ones((16, 64, 64), np.float32)):
            with self.assertRaises(ValueError): r.intervention(x, "original")
        with self.assertRaises(ValueError): r.intervention(np.ones((16, 64, 64), np.uint8), "adaptive")

    def test_logits_become_probabilities_before_half_threshold(self):
        probability = r.sigmoid(np.array([-2., 0., 2.]))
        got = r.metrics(probability, np.array([1, 2, 2]), np.ones(3, bool))
        self.assertEqual(got["true_positive_at_0_5"], 2)
        self.assertEqual(got["false_positive_at_0_5"], 0)
        self.assertEqual(got["auroc"], 1.)
        self.assertEqual(got["average_precision"], 1.)

    def test_ties_unknowns_and_one_class(self):
        got = r.metrics(np.array([.5, .5, .99]), np.array([1, 2, 0]), np.ones(3, bool))
        self.assertEqual(got["auroc"], .5)
        self.assertEqual(got["average_precision"], .5)
        self.assertEqual(got["known_pixels"], 2)
        empty = r.metrics(np.array([.5]), np.array([0]), np.ones(1, bool))
        self.assertIsNone(empty["auroc"])
        self.assertIsNone(empty["average_precision"])
        one = r.metrics(np.array([.6]), np.array([1]), np.ones(1, bool))
        self.assertIsNone(one["auroc"])
        self.assertEqual(one["background_exceedance_at_0_5"], 1.)

    def test_supported_nan_rejected_unknown_support_explicit(self):
        with self.assertRaises(ValueError): r.sigmoid([np.nan])
        with self.assertRaises(ValueError): r.metrics(np.array([np.nan]), np.array([1]), np.ones(1, bool))
        with self.assertRaises(ValueError): r.metrics(np.array([.2]), np.array([1]), np.ones(1, np.uint8))

    def test_paired_drift_has_correct_direction(self):
        a = np.array([.1, .4, .6]); b = np.array([.3, .6, .8])
        got = r.paired_drift(a, b, np.ones(3, bool))
        self.assertAlmostEqual(got["mean_signed_change"], .2)
        self.assertAlmostEqual(got["mean_absolute_change"], .2)
        self.assertEqual(got["probability_0_5_disagreements"], 1)

    def test_cache_never_falls_back_to_network(self):
        url = "https://example.invalid/public-fixture"
        raw = b"source"
        row = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as temp:
            cache = r.LocalVerifiedCache(temp, {"completed_assets": [row]})
            with self.assertRaises(ValueError): cache.get(url)
            path = Path(temp) / hashlib.sha256(url.encode()).hexdigest()
            path.write_bytes(raw)
            self.assertEqual(cache.get(url), path)
            with self.assertRaises(ValueError): cache.get(url, expected_sha256="0" * 64)
            with self.assertRaises(ValueError): cache.get(url + "/other")


if __name__ == "__main__":
    unittest.main()
