# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for tiled_q34_cloud_masked_ranker.py (Exp T13 / Q31 + Q34)."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import q22_radiometric_shift_audit as q22_mod  # noqa: E402
import tfce_stroke_calibration as tfce_cal  # noqa: E402
import tiled_q34_cloud_masked_ranker as M  # noqa: E402

VOXEL_UM = M.DEFAULT_VOXEL_UM  # 8.64 um/px
MM_PX = 1000.0 / VOXEL_UM
FORBIDDEN_GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "pycuda")


def _segment_mask(
    shape: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    width_px: float,
) -> np.ndarray:
    """Boolean mask of pixels within ``width_px / 2`` of segment ``p0 -> p1``."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = y1 - y0, x1 - x0
    denom = max(dy * dy + dx * dx, 1e-6)
    t = np.clip(((yy - y0) * dy + (xx - x0) * dx) / denom, 0.0, 1.0)
    return np.hypot(yy - (y0 + t * dy), xx - (x0 + t * dx)) <= (width_px / 2.0)


def _synthetic_looped_letter_line(
    shape: tuple[int, int] = (360, 720),
    stroke_mm: float = 0.35,
    voxel_um: float = VOXEL_UM,
    y_center: float = 180.0,
) -> np.ndarray:
    """Single line of looped Greek strokes (Ο, Δ, Π) with narrow width and b1 >= 2."""
    mm_px = 1000.0 / voxel_um
    w = stroke_mm * mm_px
    half_w = w / 2.0
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)

    m = np.zeros(shape, dtype=bool)
    # Omicron 'Ο'
    m |= np.abs(np.hypot(yy - y_center, xx - 140.0) - 0.75 * mm_px) <= half_w
    # Delta 'Δ'
    m |= _segment_mask(shape, (y_center - 75.0, 350.0), (y_center + 75.0, 275.0), w)
    m |= _segment_mask(shape, (y_center - 75.0, 350.0), (y_center + 75.0, 425.0), w)
    m |= _segment_mask(shape, (y_center + 75.0, 275.0), (y_center + 75.0, 425.0), w)
    # Pi 'Π'
    m |= _segment_mask(shape, (y_center - 70.0, 520.0), (y_center + 70.0, 520.0), w)
    m |= _segment_mask(shape, (y_center - 70.0, 620.0), (y_center + 70.0, 620.0), w)
    m |= _segment_mask(shape, (y_center - 70.0, 510.0), (y_center - 70.0, 630.0), w)
    return m


class TestTiledQ34CloudMaskedRanker(unittest.TestCase):
    """Offline CPU-only unit tests for Exp T13-TileQ34-Ranker."""

    def test_cpu_only_no_gpu_or_torch_imported(self) -> None:
        src = (HERE / "tiled_q34_cloud_masked_ranker.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for mod_name in FORBIDDEN_GPU_MODULES:
            self.assertNotIn(mod_name, imported)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(HERE)!r}); "
                "import tiled_q34_cloud_masked_ranker; "
                f"assert not any(m in sys.modules for m in {FORBIDDEN_GPU_MODULES!r})",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_area_capped_tfce_2d_multiline_stacking_invariance_vs_compact_blob(self) -> None:
        shape = (580, 640)
        valid = np.ones(shape, dtype=bool)
        stroke_w_px = 0.34 * MM_PX

        # 1) Porous 8-line text web connected via a central 1.10 mm vertical junction hub:
        # Euclidean-disk opening (0.90 mm) in stroke_bounded_tfce_2d removes the junction hub
        # and splits the web into individual half-line word clusters (~0.82 mm^2), whereas
        # unbounded tfce_2d merges all 8 lines into one giant super-cluster.
        web_mask = np.zeros(shape, dtype=bool)
        for row_y in range(55, 530, 88):
            for col_x in range(55, 595, 105):
                web_mask |= _segment_mask(
                    shape,
                    (float(row_y - 12), float(col_x)),
                    (float(row_y + 12), float(col_x)),
                    stroke_w_px,
                )
            web_mask |= _segment_mask(
                shape, (float(row_y), 45.0), (float(row_y), 585.0), stroke_w_px
            )
        web_mask |= _segment_mask(shape, (55.0, 315.0), (500.0, 315.0), 1.25 * MM_PX)

        web_score = np.clip(ndi.gaussian_filter(web_mask.astype(np.float32), 1.2) * 0.92, 0.0, 1.0)
        raw_web = tfce_cal.tfce_2d(web_score, valid)
        sb_web = tfce_cal.stroke_bounded_tfce_2d(web_score, valid, voxel_um=VOXEL_UM)
        ac_web = M.area_capped_tfce_2d(web_score, valid, voxel_um=VOXEL_UM)

        legacy_ret_web = float(sb_web.max()) / float(raw_web.max())
        area_matched_ret_web = float(min(1.0, float(sb_web.max()) / float(ac_web.max())))
        # Legacy unbounded denominator collapses retention < 0.45, while area_capped_tfce_2d
        # caps porous glyph networks at matched single-glyph scale and preserves retention >= 0.70.
        self.assertLess(legacy_ret_web, 0.45)
        self.assertGreaterEqual(area_matched_ret_web, 0.70)
        self.assertGreater(area_matched_ret_web, legacy_ret_web + 0.30)

        # 2) Compact fat 3.0 mm diffuse cloud blob keeps full max_component_area_mm2 denominator
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        cloud_m = np.hypot(yy - 240.0, xx - 320.0) <= (1.5 * MM_PX)
        cloud_score = np.clip(
            ndi.gaussian_filter(cloud_m.astype(np.float32), 4.0) * 0.88, 0.0, 1.0
        )
        sb_cloud = tfce_cal.stroke_bounded_tfce_2d(cloud_score, valid, voxel_um=VOXEL_UM)
        ac_cloud = M.area_capped_tfce_2d(cloud_score, valid, voxel_um=VOXEL_UM)
        area_matched_ret_cloud = float(sb_cloud.max()) / float(ac_cloud.max())
        self.assertLess(area_matched_ret_cloud, 0.25)

    def test_area_capped_tfce_2d_validation_and_edge_cases(self) -> None:
        zeros = M.area_capped_tfce_2d(np.ones((16, 16)), np.zeros((16, 16), dtype=bool))
        self.assertEqual(float(zeros.max()), 0.0)
        zeros_s = M.area_capped_tfce_2d(np.zeros((16, 16)), np.ones((16, 16), dtype=bool))
        self.assertEqual(float(zeros_s.max()), 0.0)

        small_cap = M.area_capped_tfce_2d(
            np.full((32, 32), 0.8, dtype=np.float32),
            np.ones((32, 32), dtype=bool),
            max_component_area_mm2=2.0,
        )
        self.assertGreater(float(small_cap.max()), 0.0)

        with self.assertRaises(ValueError):
            M.area_capped_tfce_2d(np.ones((8, 8)), np.ones((8, 8), dtype=bool), voxel_um=0.0)
        with self.assertRaises(ValueError):
            M.area_capped_tfce_2d(
                np.ones((8, 8)), np.ones((8, 8), dtype=bool), max_component_area_mm2=-1.0
            )

    def test_rim_eroded_valid_mask_boundary_distance_pinholes_and_fallback(self) -> None:
        valid = np.ones((240, 240), dtype=bool)
        # Outer invalid margin on top and right
        valid[:10, :] = False
        valid[:, 225:] = False
        # Isolated interior 3x3 pinhole (9 px < 64 px) must NOT trigger interior rim erosion
        valid[120:123, 120:123] = False

        eroded = M.rim_eroded_valid_mask(valid, voxel_um=VOXEL_UM, rim_erosion_mm=0.35)
        erosion_px = int(0.35 * 1000.0 / VOXEL_UM)  # ~40 px
        # Outer rim within ~40 px of top invalid border (row 10) is eroded out
        self.assertFalse(bool(eroded[10 + erosion_px - 2, 120]))
        self.assertTrue(bool(eroded[10 + erosion_px + 3, 120]))
        # Neighbor of the 3x3 interior pinhole remains valid!
        self.assertTrue(bool(eroded[119, 120]))
        self.assertTrue(bool(eroded[123, 120]))
        self.assertFalse(bool(eroded[121, 121]))  # pinhole itself stays invalid

        # Small crop fallback when min(H, W) <= 2.5 * erosion_px (~101 px)
        small_valid = np.ones((80, 160), dtype=bool)
        small_eroded = M.rim_eroded_valid_mask(small_valid, voxel_um=VOXEL_UM, rim_erosion_mm=0.35)
        np.testing.assert_array_equal(small_eroded, small_valid)

        # Zero erosion returns copy
        zero_eroded = M.rim_eroded_valid_mask(valid, voxel_um=VOXEL_UM, rim_erosion_mm=0.0)
        np.testing.assert_array_equal(zero_eroded, valid)

        with self.assertRaises(ValueError):
            M.rim_eroded_valid_mask(np.ones((8, 8, 2), dtype=bool))
        with self.assertRaises(ValueError):
            M.rim_eroded_valid_mask(np.ones((16, 16), dtype=bool), voxel_um=-5.0)
        with self.assertRaises(ValueError):
            M.rim_eroded_valid_mask(np.ones((16, 16), dtype=bool), rim_erosion_mm=-0.1)

    def test_suppress_rim_attached_blobs_removes_seam_blobs_preserves_interior(self) -> None:
        shape = (260, 320)
        valid = np.ones(shape, dtype=bool)
        valid[:8, :] = False
        eroded = M.rim_eroded_valid_mask(valid, voxel_um=VOXEL_UM, rim_erosion_mm=0.35)

        prob = np.full(shape, 0.12, dtype=np.float32)
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        # Rim-attached seam blob centered near row 28 (straddling rim_band and eroded_valid)
        rim_blob = np.hypot(yy - 28.0, xx - 160.0) <= (0.85 * MM_PX)
        prob[rim_blob] = 0.88
        # Genuine interior stroke at row 150
        interior_stroke = _segment_mask(shape, (150.0, 70.0), (150.0, 250.0), 0.35 * MM_PX)
        prob[interior_stroke] = 0.90

        cleaned, supp = M._suppress_rim_attached_blobs(
            prob, valid, eroded, voxel_um=VOXEL_UM, rim_erosion_mm=0.35
        )
        # Rim blob protruding into eroded_valid was suppressed down to background floor
        self.assertGreater(int(supp.sum()), 100)
        self.assertLessEqual(float(cleaned[rim_blob & eroded].max()), 0.12 + 1e-5)
        # Interior stroke inside eroded_valid is untouched
        np.testing.assert_array_equal(cleaned[interior_stroke], prob[interior_stroke])

    def test_bottleneck_b_opposite_face_border_speck_rescue(self) -> None:
        rng = np.random.default_rng(13)
        shape = (360, 720)
        valid = np.ones(shape, dtype=bool)
        loop_m = _synthetic_looped_letter_line(shape, stroke_mm=0.35, voxel_um=VOXEL_UM)
        prob_dom = np.clip(
            ndi.gaussian_filter(loop_m.astype(np.float32), 1.4) * 0.90
            + rng.normal(0.0, 0.03, shape).astype(np.float32),
            0.0,
            1.0,
        )
        prob_opp_clean = np.clip(
            ndi.gaussian_filter(rng.normal(0.0, 0.10, shape).astype(np.float32), 2.2), 0.0, 1.0
        )
        # Inject opposite-face border speck on the valid-mask rim
        prob_opp_speck = M.inject_opposite_face_border_speck(
            prob_opp_clean,
            valid,
            voxel_um=VOXEL_UM,
            speck_width_mm=0.45,
            speck_length_mm=2.20,
            peak=0.96,
        )

        legacy = tfce_cal.tfce_opposite_face_fwer(
            prob_dom, prob_opp_speck, valid, voxel_um=VOXEL_UM
        )
        self.assertFalse(legacy["q34_calibrated"]["passes_q34_bar"])
        self.assertLess(legacy["max_tfce_ratio"], 3.0)

        upgraded = M.evaluate_upgraded_q34_and_q31_ranker(
            prob_dom, prob_opp_speck, valid, voxel_um=VOXEL_UM
        )
        self.assertTrue(upgraded["global_upgraded_passes_q34_bar"])
        self.assertGreaterEqual(upgraded["cloud_masked_p995_tfce_ratio"], 3.0)
        self.assertGreater(upgraded["upgraded_q34_score"], 0.25)

    def test_step_stair_seam_blob_control_rejected(self) -> None:
        pa, pb, val = M.make_synthetic_step_stair_seam_blob_control(
            (512, 768), voxel_um=VOXEL_UM
        )
        res = M.evaluate_upgraded_q34_and_q31_ranker(pa, pb, val, voxel_um=VOXEL_UM)
        self.assertFalse(res["global_upgraded_passes_q34_bar"])
        self.assertFalse(res["top_tile_passes_q34_bar"])
        diag = res["rim_erosion_diagnostic"]
        self.assertGreater(
            diag["dom_rim_blob_suppressed_px"] + diag["opp_rim_blob_suppressed_px"], 0
        )

    def test_synthetic_bilateral_2_to_4mm_cloud_control_rejected(self) -> None:
        pa, pb, val = tfce_cal.make_synthetic_pherc1203_bilateral_clouds(
            (480, 768), voxel_um=VOXEL_UM, diameters_mm=(2.2, 2.8)
        )
        res = M.evaluate_upgraded_q34_and_q31_ranker(pa, pb, val, voxel_um=VOXEL_UM)
        self.assertFalse(res["global_upgraded_passes_q34_bar"])
        self.assertFalse(res["top_tile_passes_q34_bar"])
        self.assertFalse(res["upgraded_q34_gates"]["passes_width_retention"])
        self.assertEqual(res["upgraded_regime_label"], "bilateral_or_wide_cloud_rejected")

    def test_synthetic_single_face_dot_row_control_rejected(self) -> None:
        pa, pb, val = tfce_cal.make_synthetic_pherc0125_dot_row(
            (480, 768), voxel_um=VOXEL_UM, diameters_mm=(0.65, 0.85, 1.0, 1.15)
        )
        res = M.evaluate_upgraded_q34_and_q31_ranker(pa, pb, val, voxel_um=VOXEL_UM)
        self.assertFalse(res["global_upgraded_passes_q34_bar"])
        self.assertFalse(res["top_tile_passes_q34_bar"])
        self.assertFalse(res["upgraded_q34_gates"]["passes_stroke_morphology"])

    def test_sliding_window_tiles_and_empty_valid_mask_handling(self) -> None:
        valid = np.ones((600, 700), dtype=bool)
        tiles = M._build_sliding_window_tiles(valid, tile_size_px=512, tile_stride_px=256)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[-1]["y1"], 600)
        self.assertEqual(tiles[-1]["x1"], 700)

        empty_res = M.evaluate_upgraded_q34_and_q31_ranker(
            np.zeros((64, 64), dtype=np.float32),
            np.zeros((64, 64), dtype=np.float32),
            np.zeros((64, 64), dtype=bool),
        )
        self.assertEqual(empty_res["valid_in_px"], 0)
        self.assertFalse(empty_res["global_upgraded_passes_q34_bar"])
        self.assertEqual(empty_res["upgraded_regime_label"], "no_signal")
        json.dumps(empty_res)

    def test_http_byte_budget_guard_40_mib(self) -> None:
        self.assertEqual(M.GLOBAL_HTTP_BYTE_CAP, 40 * 1024 * 1024)
        budget = tfce_cal.HttpByteBudget(
            cap_bytes=M.GLOBAL_HTTP_BYTE_CAP, used_bytes=M.GLOBAL_HTTP_BYTE_CAP
        )
        self.assertEqual(budget.remaining(), 0)
        with self.assertRaises(RuntimeError):
            budget.fetch("https://example.com/exceed.bin")

    def test_privacy_invariant_validator(self) -> None:
        clean_json = json.dumps({"experiment": "Exp T13", "device": "cpu_only_numpy_scipy_pil"})
        M._assert_public_privacy_invariant(clean_json)

        forbidden_samples = (
            '{"path": "C:\\\\Users\\\\alice\\\\secret.json"}',
            '{"path": "C:/Users/alice/secret.json"}',
            '{"path": "/home/alice/secret.json"}',
            '{"token": "ghp_1234567890abcdef1234567890abcdef"}',
            '{"token": "hf_1234567890abcdef1234567890abcdef"}',
            '{"leak": "uv_crop"}',
            '{"leak": "top_hotspots"}',
            '{"leak": "r10-56"}',
            '{"leak": "c178-255"}',
            '{"leak": "z8736_w035"}',
        )
        for sample in forbidden_samples:
            with self.assertRaises(RuntimeError):
                M._assert_public_privacy_invariant(sample)

    def test_committed_live_receipt(self) -> None:
        ev_path = HERE / "evidence" / "tiled_q34_cloud_masked_calibration.json"
        self.assertTrue(ev_path.is_file(), f"Missing committed live receipt: {ev_path}")
        raw_text = ev_path.read_text(encoding="utf-8")
        M._assert_public_privacy_invariant(raw_text)
        data = json.loads(raw_text)
        q22_mod.assert_no_local_paths(data)

        self.assertEqual(data["schema_version"], "tiled_q34_cloud_masked_calibration_v1")
        self.assertEqual(data["device"], "cpu_only_numpy_scipy_pil")
        self.assertTrue(data["http_budget"]["within_cap"])
        self.assertLessEqual(data["http_budget"]["used_bytes"], M.GLOBAL_HTTP_BYTE_CAP)

        verdict = data["summary_verdict"]
        self.assertEqual(verdict["n_regimes"], 7)
        self.assertEqual(verdict["n_positive_regimes"], 5)
        self.assertEqual(verdict["n_negative_regimes"], 2)
        self.assertEqual(verdict["upgraded_positive_pass_count"], 5)
        self.assertEqual(verdict["upgraded_negative_pass_count"], 0)
        self.assertTrue(verdict["all_positive_regimes_pass_upgraded"])
        self.assertTrue(verdict["all_negative_controls_rejected_upgraded"])
        self.assertTrue(verdict["bottleneck_a_multiline_rescue_verified"])
        self.assertTrue(verdict["bottleneck_b_rim_speck_rescue_verified"])

        regimes = {r["regime_id"]: r for r in data["regime_benchmarks"]}
        self.assertEqual(len(regimes), 7)
        for r in data["regime_benchmarks"]:
            upg = r["upgraded_exp_t13_q34"]
            if r["expected_passes_q34_bar"]:
                self.assertTrue(
                    upg["global_upgraded_passes_q34_bar"],
                    msg=f"Positive regime {r['regime_id']} failed upgraded Q34 bar",
                )
            else:
                self.assertFalse(
                    upg["global_upgraded_passes_q34_bar"],
                    msg=f"Negative regime {r['regime_id']} passed global upgraded Q34 bar",
                )
                self.assertFalse(
                    upg["top_tile_passes_q34_bar"],
                    msg=f"Negative regime {r['regime_id']} passed top-tile upgraded Q34 bar",
                )

        # Verify all blank-margin subcontrols and synthetic subcontrols are rejected
        reg6 = regimes["6_real_blank_papyrus_margins_pherc0306b_0343_0841"]
        for sc in reg6["all_blank_margin_subcontrols"]:
            self.assertFalse(sc["upgraded_passes_q34_bar"], msg=f"Blank subcontrol {sc} passed")

        reg7 = regimes["7_synthetic_mesh_rim_seam_blob_and_bilateral_cloud_controls"]
        for name, sc in reg7["synthetic_subcontrols"].items():
            self.assertFalse(sc["upgraded_passes_q34_bar"], msg=f"Synthetic subcontrol {name} passed")


if __name__ == "__main__":
    unittest.main()
