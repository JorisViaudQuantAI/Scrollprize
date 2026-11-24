# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gpu_0211_and_0306b_calibration_matrix.py (Exp T17-GPU)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpu_0211_and_0306b_calibration_matrix as G17  # noqa: E402
import gpu_transverse_interventions as GTI  # noqa: E402
import screen_scroll_rtx4060 as SSRX  # noqa: E402
from test_screen_scroll_rtx4060 import _write_f32_tiff, _write_local_zarr_v2  # noqa: E402

RECEIPT_PATH = HERE / "evidence" / "gpu_0211_and_0306b_calibration_matrix.json"


class TestGpu0211And0306bCalibrationMatrix(unittest.TestCase):
    def test_01_layer_order_profile_card_asymmetry_and_reversal(self) -> None:
        rng = np.random.default_rng(7)
        h, w = 32, 48
        base = rng.normal(120.0, 20.0, size=(h, w))
        stack = np.zeros((31, h, w), dtype=np.uint8)
        for k in range(31):
            offset = k - 15
            # Asymmetric gradient across sheet thickness + subtle shifted spatial pattern
            rolled = np.roll(base, shift=offset // 3, axis=1)
            stack[k] = np.clip(np.round(rolled + 1.8 * offset), 1, 254).astype(np.uint8)
        valid = np.ones((h, w), dtype=bool)
        valid[:, :2] = False

        card = G17.compute_layer_order_profile_card(stack, valid, mesh_idx=15)
        self.assertEqual(card["num_layers"], 31)
        self.assertEqual(card["mesh_center_index"], 15)
        self.assertEqual(card["detector_slice_conventions"]["studio_desc_indices"][0], 23)
        self.assertEqual(card["detector_slice_conventions"]["studio_asc_indices"][0], 7)
        self.assertEqual(card["detector_slice_conventions"]["ahn_s0_asc_indices"][0], 3)
        self.assertEqual(card["detector_slice_conventions"]["ahn_s0_desc_indices"][0], 26)
        self.assertGreater(card["center_symmetry_summary"]["mean_diff_plus8_minus_minus8"], 25.0)
        self.assertLess(card["per_layer_profiles"][23]["pearson_r_vs_center_layer"], 0.99)

        # Reversing layer axis flips mean_diff sign exactly
        rev_card = G17.compute_layer_order_profile_card(stack[::-1], valid, mesh_idx=15)
        self.assertAlmostEqual(
            card["center_symmetry_summary"]["mean_diff_plus8_minus_minus8"],
            -rev_card["center_symmetry_summary"]["mean_diff_plus8_minus_minus8"],
            places=3,
        )

    def test_02_score_detector_pair_integrates_t13_and_legacy_q34(self) -> None:
        h, w = 360, 480
        valid = np.ones((h, w), dtype=bool)
        ct_mid = np.full((h, w), 120, dtype=np.uint8)
        asc = np.full((h, w), 0.05, dtype=np.float32)
        desc = np.full((h, w), 0.05, dtype=np.float32)
        # Synthesize thin (~0.26 mm = 28 px at 9.362 um) stroke-like glyphs well inside rim
        asc[90:118, 120:360] = 0.88
        asc[90:270, 226:254] = 0.88
        asc[242:270, 140:340] = 0.88
        res_pair = {
            "asc": {"prob": asc, "valid": valid, "seconds": 0.1, "peak_allocated_mib": 1200, "peak_reserved_mib": 1400},
            "desc": {"prob": desc, "valid": valid, "seconds": 0.1, "peak_allocated_mib": 1200, "peak_reserved_mib": 1400},
        }
        scored = G17.score_detector_pair(
            res_pair,
            family="studio",
            stack_mid=ct_mid,
            valid_ct=valid,
            voxel_um=9.362,
        )
        self.assertEqual(scored["dominant_face"], "asc")
        self.assertGreater(scored["dom_over_opp_gt_07_ratio"], 10.0)
        self.assertTrue(scored["upgraded_exp_t13_q34"]["global_upgraded_passes_q34_bar"])
        self.assertIn("legacy_exp_t6_q34", scored)
        self.assertIn("fp_diagnostics", scored)
        self.assertTrue(scored["recovers_signal"])

    def test_03_arm1_and_arm2_end_to_end_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="g17-e2e-test-") as tmp:
            root = Path(tmp)
            ct_zarr = root / "ct.zarr"
            m7_zarr = root / "m7.zarr"
            mesh_dir = root / "mesh_dir"
            work_root = root / "work"
            out_dir = root / "out_private"
            work_root.mkdir()
            out_dir.mkdir()

            zz, yy, xx = np.meshgrid(
                np.arange(48, dtype=np.float32),
                np.arange(160, dtype=np.float32),
                np.arange(160, dtype=np.float32),
                indexing="ij",
            )
            z_sheet = 24.0 + 0.004 * yy
            ct = np.clip(
                100.0 + 85.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.8) ** 2) + 8.0 * np.sin(xx * 0.25),
                20,
                220,
            ).astype(np.uint8)
            m7 = np.clip(
                220.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.4) ** 2),
                0,
                255,
            ).astype(np.uint8)
            _write_local_zarr_v2(ct_zarr, ct, chunks=(24, 80, 80))
            _write_local_zarr_v2(m7_zarr, m7, chunks=(24, 80, 80))

            gy, gx = np.meshgrid(
                np.linspace(20.0, 140.0, 14, dtype=np.float32),
                np.linspace(20.0, 140.0, 14, dtype=np.float32),
                indexing="ij",
            )
            gz = (24.0 + 0.004 * gy - 3.2).astype(np.float32)
            mesh_dir.mkdir()
            _write_f32_tiff(mesh_dir / "x.tif", gx)
            _write_f32_tiff(mesh_dir / "y.tif", gy)
            _write_f32_tiff(mesh_dir / "z.tif", gz)
            (mesh_dir / "meta.json").write_text(json.dumps({"scale": [0.1, 0.1]}), encoding="utf-8")

            def _fake_studio(
                stack_lhw,
                mesh_index,
                staged,
                *,
                stride,
                batch_size,
                batch_pause_ms,
                max_gpu_temp_c,
            ):
                mid = stack_lhw[mesh_index].astype(np.float32)
                norm = np.clip(mid / 200.0, 0.0, 1.0)
                desc = np.where(norm > 0.72, 0.88, 0.06).astype(np.float32)
                asc = np.full(mid.shape, 0.05, dtype=np.float32)
                val = stack_lhw[mesh_index] > 0
                return {
                    "asc": {"prob": asc, "valid": val, "seconds": 0.02, "peak_allocated_mib": 120, "peak_reserved_mib": 140},
                    "desc": {"prob": desc, "valid": val, "seconds": 0.02, "peak_allocated_mib": 120, "peak_reserved_mib": 140},
                }

            def _fake_ahn(
                stack_lhw,
                mesh_index,
                staged,
                *,
                batch_size,
                batch_pause_ms,
                max_gpu_temp_c,
            ):
                mid = stack_lhw[mesh_index].astype(np.float32)
                norm = np.clip(mid / 200.0, 0.0, 1.0)
                desc = np.where(norm > 0.72, 0.84, 0.05).astype(np.float32)
                asc = np.full(mid.shape, 0.04, dtype=np.float32)
                val = stack_lhw[mesh_index] > 0
                return {
                    "asc": {"prob": asc, "valid": val, "tiles": 4, "seconds": 0.02, "peak_allocated_mib": 90, "peak_reserved_mib": 110},
                    "desc": {"prob": desc, "valid": val, "tiles": 4, "seconds": 0.02, "peak_allocated_mib": 90, "peak_reserved_mib": 110},
                }

            arm1 = G17.run_arm1_pherc0211_matrix(
                out_private_dir=out_dir,
                work_root=work_root,
                staged={},
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                m7_zarr_url=str(m7_zarr),
                uv_crop=(0, 12, 0, 12),
                voxel_um=9.362,
                num_slices=31,
                cache_mb=32,
                run_studio_fn=_fake_studio,
                run_ahn_fn=_fake_ahn,
                guardrail_fn=None,
            )
            self.assertEqual(set(arm1["snap_modes"].keys()), {"snap0", "snap2"})
            self.assertGreater(
                arm1["snap_modes"]["snap2"]["geometry_snap_metrics"]["on_sheet_after_snap"],
                arm1["snap_modes"]["snap2"]["geometry_snap_metrics"]["on_sheet_raw"],
            )
            self.assertTrue(
                (out_dir / "gpu_0211_0306b_matrix" / "arm1_pherc0211_calibration_montage.png").is_file()
            )

            arm2 = G17.run_arm2_pherc0306b_layer_card(
                out_private_dir=out_dir,
                work_root=work_root,
                staged={},
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                uv_crop=(0, 12, 0, 12),
                voxel_um=8.64,
                num_slices=31,
                cache_mb=32,
                run_studio_fn=_fake_studio,
                run_ahn_fn=_fake_ahn,
                guardrail_fn=None,
            )
            self.assertEqual(len(arm2["layer_order_profile_card"]["per_layer_profiles"]), 31)
            self.assertTrue(arm2["reading_order_alignment"]["same_face_direction_between_families"])
            self.assertTrue(
                (out_dir / "gpu_0211_0306b_matrix" / "arm2_pherc0306b_layer_card_montage.png").is_file()
            )
            self.assertEqual(list(work_root.iterdir()), [])

    def test_04_privacy_and_committed_live_receipt(self) -> None:
        with self.assertRaises(ValueError):
            SSRX.ensure_outside_git_repo(SSRX.REPO_ROOT / "gemini_lane" / "evidence")
        if not RECEIPT_PATH.is_file():
            self.skipTest("Live Exp T17-GPU evidence receipt not yet generated")
        data = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        GTI.assert_no_local_paths_in_public_json(data)
        self.assertEqual(data["schema_version"], "gpu_0211_and_0306b_calibration_matrix_v1")
        self.assertFalse(data["hardware_safety"]["persistent_ct_on_disk"])
        self.assertLessEqual(data["hardware_safety"]["vram_fraction"], 0.50)

        arm1 = data["arm1_pherc0211_matrix"]
        self.assertEqual(set(arm1["snap_modes"].keys()), {"snap0", "snap2"})
        for sk in ("snap0", "snap2"):
            for fk in ("studio_s48", "studio_s96", "ahn_s0"):
                self.assertIn(fk, arm1["snap_modes"][sk])
                self.assertIn("upgraded_exp_t13_q34", arm1["snap_modes"][sk][fk])

        arm2 = data["arm2_pherc0306b_layer_card"]
        self.assertEqual(len(arm2["layer_order_profile_card"]["per_layer_profiles"]), 31)
        self.assertIn("ahn_s0", arm2)


if __name__ == "__main__":
    unittest.main()
