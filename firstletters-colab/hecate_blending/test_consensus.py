import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


consensus = load("test_context_consensus", HERE / "context_consensus.py")
metrics = load("test_context_consensus_metrics", HERE / "metrics.py")
PROTOCOL = json.loads((HERE / "consensus_protocol.json").read_text())


class ContextConsensusTests(unittest.TestCase):
    def test_identical_predictions_preserve_constant_supported_signal(self):
        patches = np.full((64, 256, 256), .36, np.float32)
        support = np.ones((1152, 1152), bool)
        support[300, 400] = False
        arithmetic, geometric, diagnostic = consensus.assemble(patches, support, np.ones((256, 256), np.float32))
        np.testing.assert_array_equal(arithmetic, geometric)
        self.assertEqual(geometric[300-64, 400-64], 0.)
        self.assertEqual(geometric[0, 0], np.float32(.36))
        self.assertEqual(diagnostic["floored_supported_prediction_count"], 0)

    def test_equal_arithmetic_mean_has_lower_score_when_contexts_disagree(self):
        patches = np.full((64, 256, 256), .5, np.float32)
        # At context(192,192), the four overlappingorigins are0,1,8,9.
        patches[0] = .9
        patches[1] = .1
        patches[8] = .9
        patches[9] = .1
        arithmetic, geometric, _ = consensus.assemble(patches, np.ones((1152, 1152), bool), np.ones((256, 256), np.float32))
        self.assertAlmostEqual(float(arithmetic[128, 128]), .5)
        self.assertAlmostEqual(float(geometric[128, 128]), .3)
        self.assertLess(geometric[128, 128], geometric[600, 600])

    def test_invalid_and_empty_support_fail(self):
        patches = np.full((64, 256, 256), .5, np.float32)
        support = np.ones((1152, 1152), bool)
        window = np.ones((256, 256), np.float32)
        with self.assertRaises(ValueError):
            consensus.assemble(patches, np.zeros_like(support), window)
        patches[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            consensus.assemble(patches, support, window)
        with self.assertRaises(ValueError):
            consensus.assemble(patches[:63], support, window)

    def test_transferred_threshold_is_not_refit_to_evaluation_background(self):
        labels = np.ones((256, 256), np.uint8)
        labels[:128] = 2
        score = np.full((256, 256), .7, np.float32)
        measured = consensus.measure_transferred(score, labels, np.ones_like(labels, bool), .5, metrics)
        self.assertEqual(measured["background_exceedance"], 1.)
        self.assertEqual(measured["recall"], 1.)
        self.assertFalse(measured["threshold_refit_on_evaluation_donor"])
        # A target-refit strict5%cutoff would select nothing for this tiedmap.
        self.assertEqual(metrics.measure(score, labels, np.ones_like(labels, bool))["background_exceedance"], 0.)

    def test_cutoff_ties_unknowns_and_invalid_classes(self):
        labels = np.zeros((256, 256), np.uint8)
        labels[31:33, 31:33] = 2
        labels[63:65, 63:65] = 1
        score = np.full((256, 256), .5, np.float32)
        support = np.ones_like(labels, bool)
        measured = consensus.measure_transferred(score, labels, support, .5, metrics)
        self.assertEqual(measured["background_exceedance"], 0.)
        self.assertEqual(measured["pairs"]["seam_ink"]["eligible_pairs"], 4)
        self.assertEqual(measured["pairs"]["seam_ink"]["both_positive_pairs"], 0)
        with self.assertRaises(ValueError):
            consensus.measure_transferred(score, np.ones_like(labels), support, .5, metrics)
        with self.assertRaises(ValueError):
            consensus.measure_transferred(score, labels, support, np.nan, metrics)

    def test_veto_gate_rejects_recall_damage_and_false_positive_relocation(self):
        baseline = {"known": 1000, "ink": 500, "background": 500,
                    "recall": .8, "ap": .9, "background_exceedance": .2,
                    "pairs": {"seam_ink": {"eligible_pairs": 100, "both_positive_fraction": .8, "both_positive_pairs": 80},
                              "seam_background": {"eligible_pairs": 100, "both_positive_fraction": .3, "both_positive_pairs": 30}}}
        good = copy.deepcopy(baseline)
        good["pairs"]["seam_background"].update(both_positive_fraction=.2, both_positive_pairs=20)
        self.assertTrue(consensus.compare_transfer(baseline, good, PROTOCOL["gate"])["pass"])
        exact_boundary = copy.deepcopy(good)
        exact_boundary["pairs"]["seam_background"].update(both_positive_fraction=.24, both_positive_pairs=24)
        boundary_result = consensus.compare_transfer(baseline, exact_boundary, PROTOCOL["gate"])
        self.assertEqual(boundary_result["relative_false_background_pair_reduction"], .2)
        self.assertEqual(boundary_result["absolute_false_background_pair_reduction"], .06)
        self.assertTrue(boundary_result["pass"])
        bad = copy.deepcopy(good)
        bad["recall"] = .74
        self.assertFalse(consensus.compare_transfer(baseline, bad, PROTOCOL["gate"])["pass"])
        bad = copy.deepcopy(good)
        bad["background_exceedance"] = .22
        self.assertFalse(consensus.compare_transfer(baseline, bad, PROTOCOL["gate"])["pass"])
        ceiling = copy.deepcopy(baseline)
        ceiling["pairs"]["seam_background"].update(both_positive_fraction=.19, both_positive_pairs=19)
        self.assertFalse(consensus.compare_transfer(ceiling, good, PROTOCOL["gate"])["assessable"])


if __name__ == "__main__":
    unittest.main()
