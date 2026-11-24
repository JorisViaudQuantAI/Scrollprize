# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gpu_shear_and_0211_control.py (Exp T16-GPU)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpu_shear_and_0211_control as G16  # noqa: E402
import gpu_transverse_interventions as GTI  # noqa: E402
import screen_scroll_rtx4060 as SSRX  # noqa: E402
from test_screen_scroll_rtx4060 import _write_f32_tiff, _write_local_zarr_v2  # noqa: E402

RECEIPT_PATH = HERE / "evidence" / "gpu_shear_and_0211_control.json"


class TestGpuShearAnd0211Control(unittest.TestCase):
    def test_01_shear_geometry_and_round_trip_identity(self) -> None:
        h, w = 80, 120
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        blob = np.exp(-(((yy - 40.0) ** 2) / 90.0 + ((xx - 60.0) ** 2) / 140.0)).astype(np.float32)
        valid = np.ones((h, w), dtype=bool)
        valid[:4, :] = False
        valid[-4:, :] = False
        valid[:, :4] = False
        valid[:, -4:] = False

        for angle in (-30.0, -15.0, 0.0, 15.0, 30.0):
            s_prob, s_valid = G16.apply_shear_transform(blob, valid, angle)
            geom = G16.compute_shear_geometry((h, w), angle)
            self.assertEqual(s_prob.shape, (h, geom["out_w"]))
            self.assertEqual(s_valid.shape, (h, geom["out_w"]))
            self.assertTrue(s_prob.flags["C_CONTIGUOUS"])
            self.assertTrue(s_valid.flags["C_CONTIGUOUS"])
            self.assertTrue(np.all(s_prob[~s_valid] == 0.0))

            back_prob, back_valid = G16.invert_shear_map(s_prob, s_valid, (h, w), angle)
            self.assertEqual(back_prob.shape, (h, w))
            self.assertTrue(back_valid.any())
            # Interior away from valid boundary recovers smoothly without clipping or shift
            interior = G16.erode_interior_mask(back_valid & valid, margin_px=6)
            self.assertGreater(interior.mean(), 0.55)
            mae = float(np.mean(np.abs(back_prob[interior] - blob[interior])))
            self.assertLess(mae, 0.015, f"angle={angle} mae={mae}")

    def test_02_shear_3d_uint8_preserves_nonzero_foreground_and_zero_padding(self) -> None:
        rng = np.random.default_rng(23)
        stack = rng.integers(35, 200, size=(5, 48, 64), dtype=np.uint8)
        valid = np.ones((48, 64), dtype=bool)
        valid[:, :3] = False
        stack[:, ~valid] = 0

        for angle in (-30.0, 15.0):
            sh_stack, sh_valid = G16.apply_shear_transform(stack, valid, angle)
            self.assertEqual(sh_stack.dtype, np.uint8)
            self.assertTrue(np.all(sh_stack[:, ~sh_valid] == 0))
            self.assertTrue(np.all(sh_stack[:, sh_valid] >= 1))

    def test_03_shear_angle_geometry_matches_exact_axis_skew(self) -> None:
        # Verify vertical line x = x0 tilts by tan(theta) across dy = 100 rows
        h, w = 120, 120
        img = np.zeros((h, w), dtype=np.float32)
        img[:, 60] = 1.0
        valid = np.ones((h, w), dtype=bool)
        for angle in (-30.0, 15.0):
            sh, sv = G16.apply_shear_transform(img, valid, angle)
            r_lo, r_hi = 20, 100
            x_lo = float(np.argmax(sh[r_lo]))
            x_hi = float(np.argmax(sh[r_hi]))
            measured_k = (x_hi - x_lo) / float(r_hi - r_lo)
            self.assertAlmostEqual(measured_k, np.tan(np.deg2rad(angle)), delta=0.03)

    def test_04_erode_interior_mask_removes_rim_and_has_small_crop_fallback(self) -> None:
        v = np.ones((100, 100), dtype=bool)
        er = G16.erode_interior_mask(v, margin_px=12)
        self.assertEqual(int(er.sum()), (100 - 24) * (100 - 24))

        tiny = np.ones((16, 16), dtype=bool)
        fb = G16.erode_interior_mask(tiny, margin_px=32)
        np.testing.assert_array_equal(fb, tiny)

    def test_05_arm_a_and_arm_b_end_to_end_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="g16-e2e-test-") as tmp:
            root = Path(tmp)
            ct_zarr = root / "ct.zarr"
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
            z_sheet = 24.0 + 0.005 * yy
            ct = np.clip(
                100.0 + 85.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.8) ** 2) + 8.0 * np.sin(xx * 0.25),
                20,
                220,
            ).astype(np.uint8)
            _write_local_zarr_v2(ct_zarr, ct, chunks=(24, 80, 80))

            gy, gx = np.meshgrid(
                np.linspace(20.0, 140.0, 14, dtype=np.float32),
                np.linspace(20.0, 140.0, 14, dtype=np.float32),
                indexing="ij",
            )
            gz = (24.0 + 0.005 * gy).astype(np.float32)
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
                asc = np.where(norm > 0.72, 0.88, 0.06).astype(np.float32)
                desc = np.full(mid.shape, 0.05, dtype=np.float32)
                val = stack_lhw[mesh_index] > 0
                return {
                    "asc": {"prob": asc, "valid": val, "seconds": 0.02, "peak_allocated_mib": 120, "peak_reserved_mib": 140},
                    "desc": {"prob": desc, "valid": val, "seconds": 0.02, "peak_allocated_mib": 120, "peak_reserved_mib": 140},
                }

            arm_a = G16.run_arm_a_shear_stress(
                out_private_dir=out_dir,
                work_root=work_root,
                staged={},
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                uv_crop=(0, 12, 0, 12),
                voxel_um=8.64,
                shear_angles_deg=(0.0, 15.0, -30.0),
                num_slices=31,
                interior_erosion_px=8,
                cache_mb=32,
                run_studio_fn=_fake_studio,
                guardrail_fn=None,
            )
            self.assertEqual(set(arm_a["variants"].keys()), {"shear_0", "shear_p15", "shear_m30"})
            self.assertGreater(arm_a["findings_summary"]["min_overall_interior_pearson_r"], 0.70)
            self.assertTrue((out_dir / "gpu_shear_0211" / "arm_a_shear_montage.png").is_file())

            arm_b = G16.run_arm_b_pherc0211_winner(
                out_private_dir=out_dir,
                work_root=work_root,
                staged={},
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                uv_crop=(0, 12, 0, 12),
                voxel_um=9.362,
                num_slices=31,
                cache_mb=32,
                run_studio_fn=_fake_studio,
                guardrail_fn=None,
            )
            self.assertEqual(arm_b["scroll"], "PHerc0211")
            self.assertIn("tfce_q34", arm_b)
            self.assertTrue((out_dir / "gpu_shear_0211" / "arm_b_pherc0211_winner_montage.png").is_file())
            self.assertEqual(list(work_root.iterdir()), [])

    def test_06_privacy_and_committed_live_receipt(self) -> None:
        with self.assertRaises(ValueError):
            SSRX.ensure_outside_git_repo(SSRX.REPO_ROOT / "gemini_lane" / "evidence")
        if not RECEIPT_PATH.is_file():
            self.skipTest("Live Exp T16-GPU evidence receipt not yet generated")
        data = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        GTI.assert_no_local_paths_in_public_json(data)
        self.assertEqual(data["schema_version"], "gpu_shear_and_0211_control_v1")
        self.assertFalse(data["hardware_safety"]["persistent_ct_on_disk"])
        self.assertLessEqual(data["hardware_safety"]["vram_fraction"], 0.50)

        arm_a = data["arm_a_shear_stress"]
        self.assertIsNotNone(arm_a)
        self.assertEqual(
            set(arm_a["variants"].keys()),
            {"shear_0", "shear_p15", "shear_m15", "shear_p30", "shear_m30"},
        )

        arm_b = data["arm_b_pherc0211_winner"]
        self.assertIsNotNone(arm_b)
        self.assertEqual(arm_b["scroll"], "PHerc0211")
        self.assertTrue(arm_b["tfce_q34"]["passes_stroke_morphology"])
        self.assertTrue(arm_b["validation_summary"]["winner_letters_recovered"])


if __name__ == "__main__":
    unittest.main()
