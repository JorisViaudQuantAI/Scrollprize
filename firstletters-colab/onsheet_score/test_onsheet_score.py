"""CPU tests for onsheet_score.py on synthetic planes: no network, no scroll data."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("onsheet_score", HERE / "onsheet_score.py")
osc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osc)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def texture(seed, shape=(768, 1024), sigma=3.0):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    t = (t - t.min()) / (t.max() - t.min())
    return (40 + 180 * t).astype(np.uint8)


@unittest.skipIf(cv2 is None, "opencv-python not installed")
class LayerCoherence(unittest.TestCase):
    def test_independent_planes_are_parallel(self):
        r = osc.layer_coherence(texture(1), texture(2), texture(3))
        self.assertGreater(r["tiles"], 20)
        self.assertGreater(r["f_parallel"], 0.95)

    def test_shifted_copy_is_drift(self):
        a = texture(4)
        b = np.roll(a, 5, axis=1)
        r = osc.layer_coherence(a, a, b)
        self.assertGreater(r["f_drift"], 0.95)
        self.assertAlmostEqual(r["shift_median_confident"], 5.0, delta=0.3)

    def test_identical_planes_are_still(self):
        a = texture(5)
        r = osc.layer_coherence(a, a, a)
        self.assertGreater(r["f_still"], 0.95)

    def test_fractions_sum_to_one(self):
        a = texture(6)
        b = a.copy()
        b[:, :512] = texture(7)[:, :512]
        r = osc.layer_coherence(a, a, b)
        self.assertAlmostEqual(r["f_parallel"] + r["f_drift"] + r["f_still"], 1.0, places=3)
        self.assertGreater(r["f_parallel"], 0.2)
        self.assertGreater(r["f_still"], 0.2)

    def test_invalid_tiles_are_skipped_and_mapped(self):
        a, m, b = texture(8), texture(9), texture(10)
        m[:, :600] = 0
        r, classes = osc.layer_coherence(a, m, b, return_map=True)
        self.assertEqual(int((classes != osc.SKIPPED).sum()), r["tiles"])
        self.assertTrue((classes[:, :2] == osc.SKIPPED).all())
        empty = osc.layer_coherence(a, np.zeros_like(m), b)
        self.assertEqual(empty["tiles"], 0)
        self.assertNotIn("f_parallel", empty)

    def test_score_stack_picks_planes_around_mid(self):
        a = texture(11)
        stack = np.stack([texture(20 + k) for k in range(9)])
        stack[2] = a
        stack[6] = np.roll(a, 4, axis=0)
        r = osc.score_stack(stack)
        self.assertEqual((r["mid"], r["dk"], r["layers"]), (4, 2, 9))
        self.assertGreater(r["f_drift"], 0.95)
        with self.assertRaises(ValueError):
            osc.score_stack(stack, mid=1)

    def test_cli_writes_once(self):
        a = texture(12)
        stack = np.stack([a, texture(13), texture(14), texture(15), np.roll(a, 6, axis=1)])
        with tempfile.TemporaryDirectory() as d:
            npy, out, tiles = os.path.join(d, "s.npy"), os.path.join(d, "s.json"), os.path.join(d, "t.npy")
            np.save(npy, stack)
            osc.main(["--npy", npy, "--out", out, "--map", tiles])
            with open(out) as f:
                res = json.load(f)
            self.assertGreater(res["f_drift"], 0.95)
            self.assertEqual(res["shape"], [768, 1024])
            self.assertEqual(np.load(tiles).dtype, np.uint8)
            with self.assertRaises(SystemExit):
                osc.main(["--npy", npy, "--out", out])


if __name__ == "__main__":
    unittest.main()
