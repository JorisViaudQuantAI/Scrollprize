# SPDX-License-Identifier: MIT
"""CPU-only unit tests for cloud_rv3_screen and Colab L4 Batch C1 receipt."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cloud_rv3_screen as CRS  # noqa: E402
import leak_guard as LG  # noqa: E402


class TestCloudRv3Screen(unittest.TestCase):
    def test_lazy_import_does_not_load_torch(self) -> None:
        code = (
            "import sys; sys.path.insert(0, r'" + str(HERE) + "'); "
            "import cloud_rv3_screen; "
            "print('TORCH_LOADED=' + str('torch' in sys.modules))"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        out = subprocess.check_output([sys.executable, "-c", code], text=True, env=env)
        self.assertIn("TORCH_LOADED=False", out)

    def test_summarize_and_comparison_quiet_vs_alert(self) -> None:
        h, w = 512, 512
        valid = np.ones((h, w), dtype=bool)

        # 1. Quiet null mesh
        pf_null = np.full((h, w), 0.25, dtype=np.float32)
        pr_null = np.full((h, w), 0.24, dtype=np.float32)
        st = CRS.summarize_prob_map(pf_null, valid)
        cmp_null = CRS.compute_direction_comparison(pf_null, pr_null, valid)
        self.assertEqual(st["valid_pixels"], h * w)
        self.assertGreater(st["valid_cm2"], 0.2)
        self.assertEqual(cmp_null["hits_ge_07_forward"], 0)
        self.assertFalse(cmp_null["hit_alert"])

        # 2. Asymmetric unilateral regional patch (>= 1500 hits, ratio >= 5)
        pf_asym = pf_null.copy()
        pf_asym[50:110, 50:110] = 0.85  # 3600 px
        cmp_asym = CRS.compute_direction_comparison(pf_asym, pr_null, valid)
        self.assertEqual(cmp_asym["dominant_direction"], "forward")
        self.assertEqual(cmp_asym["hits_ge_07_forward"], 3600)
        self.assertEqual(cmp_asym["bilateral_ge_07_px"], 0)
        self.assertEqual(cmp_asym["max_regional_dom_hits"], 3600)
        self.assertTrue(cmp_asym["hit_alert"])

        # 3. Bilateral mottle suppression (>= 5% bilateral overlap blocks regional alert)
        pr_bil = pr_null.copy()
        pr_bil[50:60, 50:100] = 0.85  # 500 bilateral px -> 500/3600 = 0.1389 > 0.05
        cmp_bil = CRS.compute_direction_comparison(pf_asym, pr_bil, valid)
        self.assertEqual(cmp_bil["bilateral_ge_07_px"], 500)
        self.assertGreater(cmp_bil["bilateral_overlap_frac"], 0.05)
        self.assertFalse(cmp_bil["hit_alert"])

    def test_batch_c1_receipt_integrity_and_privacy(self) -> None:
        receipt_path = HERE / "evidence" / "colab_l4_rv3_batch_c1_receipt.json"
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows = data["rows"]
        totals = data["totals"]

        self.assertEqual(data["schema"], "colab_l4_rv3_batch_c1_receipt/v1")
        self.assertEqual(len(rows), 5)
        self.assertEqual(totals["meshes_completed"], 5)
        self.assertEqual(totals["candidates_claimed"], 0)
        self.assertEqual(totals["visual_confirmed_ink_candidates"], 0)

        sum_cm2 = round(sum(float(r["valid_ct_cm2"]) for r in rows), 4)
        sum_wall = round(sum(float(r["total_seconds"]) for r in rows), 2)
        sum_gpu = round(sum(float(r["rv3_forward"]["seconds"]) + float(r["rv3_reverse"]["seconds"]) for r in rows), 2)
        n_alerts = sum(1 for r in rows if r["comparison"]["hit_alert"])

        self.assertAlmostEqual(sum_cm2, float(totals["total_cm2"]), places=4)
        self.assertAlmostEqual(sum_wall, float(totals["sum_mesh_wall_seconds"]), places=2)
        self.assertAlmostEqual(sum_gpu, float(totals["sum_gpu_inference_seconds"]), places=2)
        self.assertEqual(n_alerts, int(totals["heuristic_alerts_triggered"]))

        for rel in (
            HERE / "evidence" / "colab_l4_rv3_batch_c1_receipt.json",
            HERE / "cloud_rv3_screen.py",
            HERE / "test_cloud_rv3_screen.py",
            REPO / "collab" / "gemini-021.md",
        ):
            text = rel.read_text(encoding="utf-8")
            leaks = LG.find_leaks(text)
            self.assertEqual(leaks, [], msg=f"Privacy leaks in {rel}: {leaks}")


if __name__ == "__main__":
    unittest.main()
