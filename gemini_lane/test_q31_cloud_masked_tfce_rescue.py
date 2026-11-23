# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for q31_cloud_masked_tfce_rescue.py (Exp T8 / Q31 + Q34)."""
from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import q31_cloud_masked_tfce_rescue as M  # noqa: E402
import tfce_stroke_calibration as tfce_cal  # noqa: E402

VOXEL_UM = 8.64


def _draw_greek_tile(tile_size: int = 128, voxel_um: float = 20.0, seed: int = 7) -> np.ndarray:
    """Create a synthetic tile with narrow Greek-like strokes on a ~0.18 papyrus floor."""
    rng = np.random.default_rng(seed)
    base = np.clip(0.18 + rng.normal(0.0, 0.015, (tile_size, tile_size)), 0.05, 0.35).astype(np.float32)
    mask = np.zeros((tile_size, tile_size), dtype=bool)
    yy, xx = np.mgrid[:tile_size, :tile_size].astype(np.float32)
    # Narrow stroke (~0.25-0.35 mm -> ~3.5 px half-width on 128x128 synthetic tile)
    stroke_half = max(2.0, min(4.0, 0.175 * (1000.0 / voxel_um)))
    # Ring (Omicron)
    mask |= np.abs(np.hypot(yy - tile_size * 0.35, xx - tile_size * 0.30) - tile_size * 0.14) <= stroke_half
    # Pi strokes
    mask |= (np.abs(xx - tile_size * 0.62) <= stroke_half) & (yy >= tile_size * 0.20) & (yy <= tile_size * 0.78)
    mask |= (np.abs(xx - tile_size * 0.82) <= stroke_half) & (yy >= tile_size * 0.20) & (yy <= tile_size * 0.78)
    mask |= (np.abs(yy - tile_size * 0.22) <= stroke_half) & (xx >= tile_size * 0.58) & (xx <= tile_size * 0.86)
    # Horizontal cross stroke
    mask |= (np.abs(yy - tile_size * 0.72) <= stroke_half) & (xx >= tile_size * 0.18) & (xx <= tile_size * 0.46)
    ink = ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    return np.clip(np.maximum(base, 0.18 + 0.66 * ink), 0.0, 1.0).astype(np.float32)


def _make_synthetic_surface(
    n_rows: int = 4,
    n_cols: int = 4,
    tile_size: int = 128,
    voxel_um: float = 20.0,
    n_text_tiles: int = 3,
    border_pad: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic scroll surface with 3 real-like text tiles, blank papyrus, and 0-padding border."""
    rng = np.random.default_rng(seed)
    h = n_rows * tile_size + 2 * border_pad
    w = n_cols * tile_size + 2 * border_pad
    raw = np.zeros((h, w), dtype=np.uint8)
    interior = np.clip(
        0.18 + rng.normal(0.0, 0.015, (n_rows * tile_size, n_cols * tile_size)), 0.06, 0.35
    ).astype(np.float32)

    text_coords = [(0, 1), (1, 2), (2, 1)][:n_text_tiles]
    for idx, (r, c) in enumerate(text_coords):
        y0, y1 = r * tile_size, (r + 1) * tile_size
        x0, x1 = c * tile_size, (c + 1) * tile_size
        interior[y0:y1, x0:x1] = _draw_greek_tile(tile_size, voxel_um=voxel_um, seed=seed + idx + 1)

    raw[border_pad : h - border_pad, border_pad : w - border_pad] = np.round(interior * 255.0).astype(
        np.uint8
    )
    prob = raw.astype(np.float32) / 255.0
    valid, _ = M.infer_valid_mask(raw)
    return prob, raw, valid


class TestQ31CloudMaskedTfceRescue(unittest.TestCase):
    """Offline tests covering decoding, valid masks, cloud injection, and A/B/C rescue."""

    def test_cpu_only_no_torch_imported(self) -> None:
        src = (HERE / "q31_cloud_masked_tfce_rescue.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(HERE)!r}); "
                "import q31_cloud_masked_tfce_rescue; assert 'torch' not in sys.modules",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_decode_lzw_uint8_tiff_and_infer_valid_mask(self) -> None:
        arr = np.full((80, 96), 45, dtype=np.uint8)
        arr[:10, :] = 0
        arr[-8:, :] = 0
        arr[:, :6] = 0
        # Interior isolated zero pixel must remain valid (not connected to border)
        arr[40, 48] = 0
        arr[25:35, 30:40] = 230

        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="TIFF", compression="tiff_lzw")
        prob, raw, meta = M.decode_ink_tiff_bytes(buf.getvalue())

        self.assertEqual(prob.shape, (80, 96))
        self.assertEqual(prob.dtype, np.float32)
        self.assertAlmostEqual(float(prob[25, 30]), 230.0 / 255.0, places=5)
        self.assertEqual(meta["dtype"], "uint8")

        valid, diag = M.infer_valid_mask(raw)
        self.assertFalse(bool(valid[0, 0]))
        self.assertFalse(bool(valid[79, 50]))
        self.assertTrue(bool(valid[40, 48]))  # interior zero preserved
        self.assertEqual(diag["interior_zero_px"], 1)
        self.assertGreater(diag["border_zero_frac"], 0.2)

    def test_valid_mask_without_padding_border(self) -> None:
        arr = np.full((32, 32), 100, dtype=np.uint8)
        arr[10, 10] = 0
        valid, diag = M.infer_valid_mask(arr)
        self.assertTrue(bool(valid.all()))
        self.assertEqual(diag["border_zero_px"], 0)
        self.assertEqual(diag["interior_zero_px"], 1)

    def test_margin_opposite_face_control_preserves_floor(self) -> None:
        prob, _, valid = _make_synthetic_surface(n_rows=4, n_cols=4, tile_size=128, border_pad=0)
        opp, prov = M.build_margin_opposite_face_control(prob, valid, tile_size=128, n_donor_tiles=4)
        self.assertEqual(opp.shape, prob.shape)
        self.assertEqual(opp.dtype, np.float32)
        self.assertAlmostEqual(prov["opp_valid_stats"]["median"], 0.18, delta=0.03)
        self.assertLess(prov["opp_valid_stats"]["frac_ge_045"], 0.01)
        self.assertIn("single-face", prov["assumption_disclosure"])

    def test_bilateral_cloud_injection_respects_protected_mask_and_area_fraction(self) -> None:
        prob, _, valid = _make_synthetic_surface(n_rows=4, n_cols=4, tile_size=128, border_pad=0)
        opp, _ = M.build_margin_opposite_face_control(prob, valid, tile_size=128)
        tiles = M.extract_valid_tiles(valid, tile_size=128)
        true_ids, prot_mask, _ = M.identify_protected_true_text_tiles(
            prob, valid, tiles, protected_frac=0.19, min_true_tiles=3, max_true_tiles=3
        )
        self.assertEqual(len(true_ids), 3)

        cdom, copp, cmask, diag = M.inject_bilateral_gaussian_clouds(
            prob,
            opp,
            valid,
            prot_mask,
            voxel_um=20.0,
            target_area_frac=0.20,
            seed=31,
            diameter_mm_range=(1.0, 1.8),
        )
        # Protected pixels are strictly untouched
        np.testing.assert_array_equal(cdom[prot_mask], prob[prot_mask])
        np.testing.assert_array_equal(copp[prot_mask], opp[prot_mask])
        self.assertFalse(bool((cmask & prot_mask).any()))
        # Target area fraction achieved within tight bounds
        self.assertGreaterEqual(diag["achieved_area_frac"], 0.18)
        self.assertLessEqual(diag["achieved_area_frac"], 0.28)
        # Clouds present bilaterally on BOTH faces
        self.assertGreater(float(np.minimum(cdom, copp)[cmask].mean()), 0.55)

    def test_q31_pipelines_rescue_under_20pct_bilateral_cloud(self) -> None:
        """Baseline (A) must degrade under 20% bilateral clouds while Rescues (B) & (C) recover text."""
        prob, _, valid = _make_synthetic_surface(
            n_rows=4, n_cols=4, tile_size=128, voxel_um=20.0, n_text_tiles=3, border_pad=0
        )
        opp, _ = M.build_margin_opposite_face_control(prob, valid, tile_size=128)
        tiles = M.extract_valid_tiles(valid, tile_size=128)
        true_ids, prot_mask, _ = M.identify_protected_true_text_tiles(
            prob, valid, tiles, protected_frac=0.19, min_true_tiles=3, max_true_tiles=3
        )

        clean_a = M.score_pipeline_a_global_z2(prob, valid, tiles)
        clean_b = M.score_pipeline_b_cloud_masked_mad(prob, opp, valid, tiles)
        clean_c = M.score_pipeline_c_cloud_masked_sb_tfce(
            prob, opp, valid, tiles, voxel_um=20.0, n_steps=16, eval_top_q34_k=1
        )
        # At 0% cloud, all three pipelines recover all 3 true-text tiles in Top-3
        self.assertEqual({r["tile_id"] for r in clean_a["ranked_tiles"][:3]}, true_ids)
        self.assertEqual({r["tile_id"] for r in clean_b["ranked_tiles"][:3]}, true_ids)
        self.assertEqual({r["tile_id"] for r in clean_c["ranked_tiles"][:3]}, true_ids)

        # Inject 20% bilateral clouds (1.2-2.0 mm -> 60-100 px at 20 um)
        cdom, copp, cmask, _ = M.inject_bilateral_gaussian_clouds(
            prob,
            opp,
            valid,
            prot_mask,
            voxel_um=20.0,
            target_area_frac=0.20,
            seed=99,
            diameter_mm_range=(1.2, 2.0),
        )
        cloud_ids = {
            t["tile_id"]
            for t in tiles
            if float(cmask[t["y0"] : t["y1"], t["x0"] : t["x1"]].mean()) >= 0.05
        }
        self.assertGreater(len(cloud_ids), 0)

        res_a = M.score_pipeline_a_global_z2(cdom, valid, tiles)
        res_b = M.score_pipeline_b_cloud_masked_mad(cdom, copp, valid, tiles)
        res_c = M.score_pipeline_c_cloud_masked_sb_tfce(
            cdom, copp, valid, tiles, voxel_um=20.0, n_steps=16, eval_top_q34_k=1
        )

        # Baseline (A) suffers substantial sigma inflation and cloud false alarms
        sigma_inf_a = res_a["global_std"] / max(clean_a["global_std"], 1e-6)
        self.assertGreater(sigma_inf_a, 1.25)
        eval_a = M.evaluate_tile_ranking(res_a["ranked_tiles"], true_ids, cloud_ids, true_ids)
        eval_b = M.evaluate_tile_ranking(res_b["ranked_tiles"], true_ids, cloud_ids, true_ids)
        eval_c = M.evaluate_tile_ranking(res_c["ranked_tiles"], true_ids, cloud_ids, true_ids)

        self.assertGreater(eval_a["cloud_top3_false_alarm_rate"], 0.5)
        self.assertEqual(eval_b["top1_hit"], 1.0)
        self.assertEqual(eval_b["top3_true_text_recall"], 1.0)
        self.assertEqual(eval_b["cloud_top3_false_alarm_rate"], 0.0)
        self.assertEqual(eval_c["top1_hit"], 1.0)
        self.assertEqual(eval_c["top3_true_text_recall"], 1.0)
        self.assertEqual(eval_c["cloud_top3_false_alarm_rate"], 0.0)

        # Effective robust MAD sigma stays near 1.0x after bilateral cloud masking
        sigma_inf_b = res_b["effective_mad_sigma"] / max(clean_b["effective_mad_sigma"], 1e-6)
        self.assertLess(abs(sigma_inf_b - 1.0), 0.15)

    def test_http_byte_cap_enforced_at_35_mib(self) -> None:
        self.assertEqual(M.GLOBAL_HTTP_BYTE_CAP, 35 * 1024 * 1024)
        budget = tfce_cal.HttpByteBudget(cap_bytes=M.GLOBAL_HTTP_BYTE_CAP)
        self.assertEqual(budget.remaining(), 35 * 1024 * 1024)
        self.assertEqual(len(M.PUBLIC_INK_MAPS), 3)

    def test_evidence_json_schema_and_sanitization(self) -> None:
        ev_path = HERE / "evidence" / "q31_cloud_masked_tfce_rescue.json"
        if not ev_path.exists():
            self.skipTest("Evidence JSON not yet generated")
        raw_text = ev_path.read_text(encoding="utf-8")
        self.assertNotIn("C:\\Users\\", raw_text)
        self.assertNotIn("C:/Users/", raw_text)
        data = json.loads(raw_text)
        self.assertEqual(data["device"], "cpu_only_numpy_scipy_pil")
        self.assertTrue(data["http_budget"]["within_cap"])
        self.assertLessEqual(data["http_budget"]["used_bytes"], 35 * 1024 * 1024)
        self.assertEqual(len(data["sources"]), 3)
        self.assertEqual(len(data["per_map_results"]), 3)
        self.assertEqual(len(data["q31_rescue_summary_table"]), 12)  # 4 fracs x 3 pipelines


if __name__ == "__main__":
    unittest.main()
