"""Physical row spacing and calibration metadata for the First Letters triage scorer."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


CONTROLS = Path(__file__).resolve().parents[1] / "controls"
spec = importlib.util.spec_from_file_location("score_text", CONTROLS / "score_text.py")
score_text = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_text)

PX_MM = 0.1
PITCH_MM = 5.5


def stripe_fixture():
    rows = (np.arange(200) + 0.5) * PX_MM
    profile = 0.55 + 0.35 * np.cos(2 * np.pi * rows / PITCH_MM)
    image = np.tile(profile.astype(np.float32)[:, None], (1, 48))
    mask = np.ones_like(image)
    mask[86:114] = 0  # 2.8 mm of unobserved papyrus, across the middle of a 20 mm field
    return image, mask


class RowPeriodicityTests(unittest.TestCase):
    def test_pitch_survives_an_internal_tear(self):
        image, torn = stripe_fixture()
        for mask in (np.ones_like(torn), torn):
            with self.subTest(observed_rows=int(mask[:, 0].sum())):
                _, modulation, pitch = score_text.row_periodicity(image, mask, PX_MM, 0)
                self.assertAlmostEqual(pitch, PITCH_MM, delta=0.2)
                self.assertGreater(modulation, 0.15)

    def test_contiguous_support_agrees_with_trimming_outer_margins(self):
        image, _ = stripe_fixture()
        mask = np.zeros_like(image)
        mask[5:-7] = 1
        actual = score_text.row_periodicity(image, mask, PX_MM, 0)
        expected = score_text.row_periodicity(image[5:-7], mask[5:-7], PX_MM, 0)
        np.testing.assert_array_equal(actual, expected)

    def test_mask_alone_creates_no_row_modulation(self):
        image, mask = stripe_fixture()
        _, modulation, _ = score_text.row_periodicity(np.full_like(image, 0.5), mask, PX_MM, 0)
        self.assertLess(modulation, 1e-5)
        mask[55:145] = 0  # The surviving rows cover just 11 mm.
        self.assertEqual(score_text.row_periodicity(image, mask, PX_MM, 0), (0., 0., 0.))

    def test_cli_reports_historical_calibration_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "blank.png"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((24, 24), np.uint8)))
            records = folder / "windows.json"
            argv = ["score_text.py", str(image), "--px-um", "1000", "--fibre", "0", "--json", str(records)]
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                score_text.main()
            summary = json.loads(stdout.getvalue())
            warnings = summary["calibration_warnings"]
            self.assertIn("14 mm windows and 1.5 mm steps", warnings[0])
            self.assertIn("20 mm windows and 5 mm steps", stderr.getvalue())
            self.assertEqual(json.loads(records.read_text())["calibration_warnings"], warnings)
            self.assertIn("score", summary["best"])
        self.assertEqual(score_text.calibration_warnings(
            {"window_mm": 14., "step_mm": 1.5, "row_sampling": "physical"}, 14., 1.5), [])


if __name__ == "__main__":
    unittest.main()
