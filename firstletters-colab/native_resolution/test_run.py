import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native_resolution_run", HERE/"run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class NativeResolutionTests(unittest.TestCase):
    def test_degradation_identity_and_shared_mask(self):
        raw = np.arange(64*256*256, dtype=np.uint32).reshape(64, 256, 256)
        raw = ((raw*13+7) % 256).astype(np.uint8)
        raw[:, :4, :4] = 0
        before = raw.copy()
        expanded, coarse, fine_valid, coarse_valid = run.degrade(raw)
        expected = raw.astype(np.uint32).reshape(16, 4, 64, 4, 64, 4).sum((1, 3, 5))/64
        np.testing.assert_array_equal(coarse, np.rint(expected).astype(np.uint8))
        np.testing.assert_array_equal(expanded[::4, ::4, ::4], coarse)
        np.testing.assert_array_equal(expanded, coarse.repeat(4, 0).repeat(4, 1).repeat(4, 2))
        np.testing.assert_array_equal(fine_valid, raw.any(0))
        np.testing.assert_array_equal(coarse_valid, fine_valid.reshape(64, 4, 64, 4).any((1, 3)))
        np.testing.assert_array_equal(raw, before)
        self.assertFalse(coarse_valid[0, 0])

    def test_nearest_even_rounding(self):
        raw = np.zeros((64, 256, 256), np.uint8)
        raw[:2, :4, :4] = 1
        raw[:4, :4, 4:8] = 1
        raw[:2, :4, 4:8] = 2
        _, coarse, _, _ = run.degrade(raw)
        self.assertEqual(int(coarse[0, 0, 0]), 0)
        self.assertEqual(int(coarse[0, 0, 1]), 2)

    def test_fine_to_coarse_score_cells(self):
        values = np.arange(128*128, dtype=np.float32).reshape(128, 128)
        got = run.reduce_scores(values)
        self.assertEqual(got.shape, (32, 32))
        self.assertEqual(float(got[5, 9]), float(values[20:24, 36:40].mean()))

    def test_sparse_box_average_matches_dense(self):
        rng = np.random.default_rng(17)
        raw = rng.integers(0, 256, size=(7, 9, 11), dtype=np.uint8)
        weights = []
        for n in raw.shape:
            w = np.zeros((n-1, n), np.float32)
            for i in range(n-1):
                w[i, i:i+2] = (.25, .75)
            weights.append(w)
        dense = np.einsum("az,by,cx,zyx->abc", *weights, raw.astype(np.float32), optimize=True)
        np.testing.assert_array_equal(run.resample_fine(raw, weights), np.rint(dense).astype(np.uint8))

    def test_physical_grid_and_context_halos(self):
        helper = run.load_module(HERE.parent/"transversal_lab/hecate_probe.py", "native_resolution_test_helper")
        donor = {"source_pitch_um": 2.403, "source_shape_zyx": [65, 10000, 10000],
                 "crop_xyxy": [0, 0, 10000, 10000]}
        origin = [144, 912]
        fine = run.fine_geometry(donor, origin, 64, helper.box_axis_weights)
        coarse = helper.context_geometry(donor, {"origin_target_yx": origin},
                     {"sampling": {"target_pitch_um": 9.6, "z_center_source_index": 31.5}})
        for (sf, wf), (sc, wc) in zip(fine, coarse):
            self.assertEqual((sf.start, sf.stop), (sc.start, sc.stop))
            np.testing.assert_allclose(wf.reshape(wc.shape[0], 4, -1).mean(1), wc, atol=1e-7, rtol=0)
        full = run.fine_geometry(donor, origin, 288, helper.box_axis_weights)
        self.assertEqual(full[1][1].shape[0], 1152)
        self.assertEqual(full[0][1].shape[0], 64)
        with self.assertRaises(ValueError):
            run.fine_geometry(donor, [-1, 0], 64, helper.box_axis_weights)

    def test_exact_categorical_alignment_and_pilot_exclusion(self):
        labels = (np.indices((600, 600)).sum(0) % 3).astype(np.uint8)
        full = run.categorical_core({}, [32, 64], labels, 256)
        for j in range(8):
            for i in range(8):
                patch = run.categorical_core({}, [32+32*j, 64+32*i], labels, 32)
                np.testing.assert_array_equal(patch, full[32*j:32*j+32, 32*i:32*i+32])
        self.assertTrue(run.rectangles_overlap([10, 10], [73, 10], 64, 288))
        self.assertFalse(run.rectangles_overlap([10, 10], [75, 10], 64, 288, guard=1))
        self.assertTrue(run.rectangles_overlap([10, 10], [74, 10], 64, 288, guard=1))

    def test_ties_and_within_region_background_operating_point(self):
        labels = np.array([2]*10 + [1]*100, np.uint8)
        scores = np.array([.9]*8 + [.1]*2 + [.8]*4 + [.2]*96, np.float32)
        metric = run.measure(scores, labels, np.ones_like(labels, bool))
        self.assertEqual(metric["ink_positive_count"], 8)
        self.assertEqual(metric["background_positive_count"], 4)
        self.assertAlmostEqual(metric["recall_at_background_5pct"], .8)
        self.assertLessEqual(metric["background_exceedance"], .05)
        tied = run.measure(np.ones(110), labels, np.ones(110, bool))
        self.assertEqual(tied["recall_at_background_5pct"], 0.)
        self.assertEqual(tied["auc"], .5)
        self.assertAlmostEqual(tied["ap"], 10/110)
        pure = run.measure(np.ones(10), np.full(10, 2, np.uint8), np.ones(10, bool))
        self.assertIsNone(pure["auc"])

    def test_gate_uses_macro_not_pooled_and_requires_both_comparisons(self):
        def fixture(a=.8, b=.7, c=.6):
            return {q: {arm: {"recall_at_background_5pct": value, "ap": .8}
                        for arm, value in zip(run.ARMS, (a, b, c))}
                    for q in ("q00", "q01", "q10", "q11")}
        self.assertTrue(run.gate(fixture())["pass"])
        self.assertFalse(run.gate(fixture(a=.64, b=.5, c=.6))["pass"])
        self.assertFalse(run.gate(fixture(a=.8, b=.81, c=.6))["pass"])
        self.assertFalse(run.gate(fixture(a=.99, b=.9, c=.97))["pass"])
        value = fixture()
        value["field"] = {arm: {"recall_at_background_5pct": 0, "ap": 0} for arm in run.ARMS}
        self.assertTrue(run.gate(value)["pass"])
        value["q00"][run.ARMS[0]]["ap"] = .1
        value["q01"][run.ARMS[0]]["ap"] = .1
        self.assertFalse(run.gate(value)["pass"])

    def test_invalid_shapes_and_labels_fail(self):
        with self.assertRaises(ValueError):
            run.degrade(np.zeros((16, 64, 64), np.uint8))
        with self.assertRaises(ValueError):
            run.reduce_scores(np.ones((3, 4)))
        with self.assertRaises(ValueError):
            run.measure(np.ones(2), np.array([1, 3]), np.ones(2, bool))


if __name__ == "__main__":
    unittest.main()
