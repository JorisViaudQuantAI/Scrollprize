# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gpu_transverse_interventions.py."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import gpu_transverse_interventions as GTI
import screen_scroll_rtx4060 as SSRX
from test_screen_scroll_rtx4060 import _write_f32_tiff, _write_local_zarr_v2


class TestGpuTransverseInterventions(unittest.TestCase):
    def test_spatial_transforms_and_inversion_are_exact_involutions(self) -> None:
        rng = np.random.default_rng(42)
        stack = rng.integers(0, 256, size=(7, 24, 30), dtype=np.uint8)
        map2d = rng.random((24, 30), dtype=np.float32)

        for mode in GTI.VALID_SPATIAL_MODES:
            t_stack = GTI.apply_spatial_transform(stack, mode)
            self.assertEqual(t_stack.shape, stack.shape)
            self.assertTrue(t_stack.flags["C_CONTIGUOUS"])
            back_stack = GTI.apply_spatial_transform(t_stack, mode)
            np.testing.assert_array_equal(back_stack, stack)

            t_map = GTI.apply_spatial_transform(map2d, mode)
            back_map = GTI.invert_spatial_map(t_map, mode)
            np.testing.assert_array_equal(back_map, map2d)

        with self.assertRaises(ValueError):
            GTI.apply_spatial_transform(stack, "invalid_mode")

    def test_radiometric_shift_and_nyul_standardization_preserve_zeros_and_restore_landmarks(self) -> None:
        rng = np.random.default_rng(7)
        stack = rng.integers(40, 175, size=(11, 48, 48), dtype=np.uint8)
        stack[:, :6, :] = 0  # zero margin
        valid = np.ones((48, 48), dtype=bool)
        valid[:, :5] = False  # invalid column band

        pcts = (1.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.0)
        orig_stats = GTI.summarize_stack_radiometry(stack, valid, pcts)
        ref_lm = tuple(orig_stats["landmarks"][f"p{int(p):02d}"] for p in pcts)

        shifted = GTI.apply_radiometric_shift(stack, valid, delta_counts=24.0)
        self.assertEqual(shifted.dtype, np.uint8)
        self.assertTrue(np.all(shifted[:, :6, :] == 0))
        self.assertTrue(np.all(shifted[:, :, :5] == 0))

        fg = (stack > 0) & valid[None, :, :]
        np.testing.assert_array_equal(shifted[fg], np.clip(stack[fg].astype(np.int32) + 24, 1, 255).astype(np.uint8))

        rescued = GTI.apply_nyul_standardization_u8(shifted, valid, ref_landmarks=ref_lm, percentiles=pcts)
        self.assertEqual(rescued.dtype, np.uint8)
        self.assertTrue(np.all(rescued[~fg] == 0))
        self.assertTrue(np.all(rescued[fg] >= 1))
        rescued_stats = GTI.summarize_stack_radiometry(rescued, valid, pcts)
        for p in (10, 25, 50, 75, 90):
            key = f"p{p:02d}"
            self.assertAlmostEqual(rescued_stats["landmarks"][key], orig_stats["landmarks"][key], delta=1.5)

    def test_extract_depth_window_bounds_and_centering(self) -> None:
        stack = np.arange(43 * 8 * 8, dtype=np.uint8).reshape(43, 8, 8)
        for dz in (-6, -3, 0, 3, 6):
            sub, mid = GTI.extract_depth_window(stack, base_mesh_index=21, dz=dz, window_slices=31)
            self.assertEqual(sub.shape, (31, 8, 8))
            self.assertEqual(mid, 15)
            np.testing.assert_array_equal(sub[mid], stack[21 + dz])

        with self.assertRaises(ValueError):
            GTI.extract_depth_window(stack, base_mesh_index=21, dz=-7, window_slices=31)
        with self.assertRaises(ValueError):
            GTI.extract_depth_window(stack, base_mesh_index=21, dz=7, window_slices=31)

    def test_refuses_output_inside_git_repo_and_detects_path_leaks(self) -> None:
        inside = SSRX.REPO_ROOT / "gemini_lane" / "evidence" / "forbidden_dir"
        with self.assertRaises(ValueError):
            SSRX.ensure_outside_git_repo(inside)

        GTI.assert_no_local_paths_in_public_json({"scroll": "PHerc0306B", "ok": True})
        with self.assertRaises(ValueError):
            GTI.assert_no_local_paths_in_public_json({"leaked": r"C:\Users\joris\Documents\Scrollprize_private"})

    def test_run_arm1_control_and_arm2_adversarial_end_to_end_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gti-e2e-test-") as tmp:
            root = Path(tmp)
            ct_zarr = root / "ct.zarr"
            m7_zarr = root / "m7.zarr"
            mesh_dir = root / "mesh_dir"
            work_root = root / "work"
            out_dir = root / "out_private"
            work_root.mkdir()
            out_dir.mkdir()

            zz, yy, xx = np.meshgrid(
                np.arange(64, dtype=np.float32),
                np.arange(160, dtype=np.float32),
                np.arange(160, dtype=np.float32),
                indexing="ij",
            )
            z_sheet = 32.0 + 0.005 * yy
            ct = np.clip(95.0 + 90.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.8) ** 2) + 6.0 * np.sin(xx * 0.25), 25, 220).astype(
                np.uint8
            )
            m7 = np.clip(255.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.2) ** 2), 0, 255).astype(np.uint8)
            _write_local_zarr_v2(ct_zarr, ct, chunks=(32, 80, 80))
            _write_local_zarr_v2(m7_zarr, m7, chunks=(32, 80, 80))

            gy, gx = np.meshgrid(
                np.linspace(24.0, 136.0, 18, dtype=np.float32),
                np.linspace(24.0, 136.0, 18, dtype=np.float32),
                indexing="ij",
            )
            gz = (32.0 + 0.005 * gy).astype(np.float32)
            mesh_dir.mkdir()
            _write_f32_tiff(mesh_dir / "x.tif", gx)
            _write_f32_tiff(mesh_dir / "y.tif", gy)
            _write_f32_tiff(mesh_dir / "z.tif", gz)
            (mesh_dir / "meta.json").write_text(json.dumps({"scale": [0.2, 0.2]}), encoding="utf-8")

            # Canonical synthetic probability pattern (thin stroke on asc face) so flips test un-flipping
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
                h, w = stack_lhw.shape[1], stack_lhw.shape[2]
                # Make output depend deterministically on spatial coordinates of mid-plane CT
                mid = stack_lhw[mesh_index].astype(np.float32)
                norm = np.clip((mid - float(mid.min())) / max(float(np.ptp(mid)), 1.0), 0.0, 1.0)
                asc = (0.05 + 0.85 * (norm > 0.65).astype(np.float32)).astype(np.float32)
                desc = np.full((h, w), 0.06, dtype=np.float32)
                val = stack_lhw[mesh_index] > 0
                return {
                    "asc": {"prob": asc, "valid": val, "seconds": 0.02, "peak_allocated_mib": 120, "peak_reserved_mib": 140},
                    "desc": {"prob": desc, "valid": val, "seconds": 0.02, "peak_allocated_mib": 120, "peak_reserved_mib": 140},
                }

            arm1 = GTI.run_arm1_control(
                out_private_dir=out_dir,
                work_root=work_root,
                staged={},
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                uv_crop=(0, 16, 0, 16),
                voxel_um=8.64,
                num_slices=31,
                studio_stride=48,
                cache_mb=32,
                run_studio_fn=_fake_studio,
                guardrail_fn=None,
            )
            self.assertEqual(
                set(arm1["variants"].keys()),
                {"orig", "flip_lr", "flip_ud", "rot180", "nyul_udupa", "shift_plus24", "shift_plus24_nyul"},
            )
            # Because _fake_studio is equivariant with CT mid-plane, un-flipped orientations match orig 1.0
            self.assertAlmostEqual(arm1["findings_summary"]["min_orientation_iou_ge_05"], 1.0, places=5)
            self.assertTrue((out_dir / "arm1_pherc0306b_control_public.json").is_file())
            self.assertTrue((out_dir / "arm1_pherc0306b_montage.png").is_file())
            with Image.open(out_dir / "arm1_pherc0306b_montage.png") as im:
                self.assertGreater(im.size[0], 50)
                self.assertGreater(im.size[1], 100)

            arm2 = GTI.run_arm2_adversarial(
                out_private_dir=out_dir,
                work_root=work_root,
                staged={},
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                m7_zarr_url=str(m7_zarr),
                uv_crop=(0, 16, 0, 16),
                voxel_um=9.362,
                snap_passes=1,
                num_slices=43,
                window_slices=31,
                depth_shifts=(-6, 0, 6),
                cache_mb=32,
                run_studio_fn=_fake_studio,
                guardrail_fn=None,
            )
            self.assertIn("dz0_orig_s48", arm2["variants"])
            self.assertIn("dz_m6_s96", arm2["variants"])
            self.assertIn("dz_p6_s96", arm2["variants"])
            adv_dir = out_dir / "pherc0125_z8736_adversarial_check"
            self.assertTrue((adv_dir / "pherc0125_z8736_adversarial_summary.json").is_file())
            self.assertTrue((adv_dir / "pherc0125_z8736_adversarial_montage.png").is_file())
            self.assertEqual(list(work_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
