# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU unit tests for Exp T12-Villa1739 (villa1739_cross_resolution_bridge.py).

Strictly CPU-only, zero network I/O, zero torch/CUDA imports, synthetic data only.
Covers >= 8 tests:
1. AST + isolated subprocess verification of zero `torch`/GPU imports.
2. 2D cross-resolution block-area downsampling (3.90047x) + translational shift registration
   (dy, dx) recovery on a synthetic shifted letter grid (IoU improvement from < 0.30 to > 0.90).
3. 42D antisymmetric depth-spatial feature extractor + Ridge classifier recovery of a synthetic
   one-sided ink layer on held-out Fold B (on_sheet_auc > 0.90) while passing the +/-8-slice
   off-sheet kill-test (drop_vs_pm8 > 0.30).
4. Adversarial macro-curvature / depth-invariant artifact rejection by the +/-8-slice off-sheet
   kill-test (drop_vs_pm8 <= 0.0 -> CLOSED NO-GO).
5. Locked polarity `+` gate invariant (inverted signal at test time cannot flip sign to pass).
6. Spatial train/test split 32 px buffer-zone non-overlap invariant (modifying Fold B or buffer
   voxels changes Fold A features and learned Ridge weights by 0.0 bits).
7. Per-ROI robust median/IQR standardization invariance to global CT affine gain/offset shifts (Q22).
8. Strict <= 100 MiB HTTP byte budget enforcement.
9. Offline end-to-end 3-scroll pipeline + public-repo privacy verification.
10. Committed live receipt schema, gate consistency, budget <= 100 MiB, and privacy check.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_transverse_cross_scroll_replication import (  # noqa: E402
    _write_local_zarr3_sharded_u8,
    _write_local_zarr_v2,
)
import villa1739_cross_resolution_bridge as v1739  # noqa: E402


def _make_letter_grid_mask(h: int, w: int) -> np.ndarray:
    """Create a multi-bar synthetic Greek-letter-like stroke pattern."""
    mask = np.zeros((h, w), dtype=bool)
    for r0 in range(16, h - 20, 22):
        for c0 in range(14, w - 20, 24):
            mask[r0 : r0 + 4, c0 : c0 + 14] = True
            mask[r0 : r0 + 14, c0 : c0 + 4] = True
            mask[r0 : r0 + 14, c0 + 10 : c0 + 14] = True
    return mask


class TestVilla1739CrossResolutionBridge(unittest.TestCase):
    def test_01_no_torch_or_gpu_imports(self) -> None:
        src_path = Path(__file__).resolve().parent / "villa1739_cross_resolution_bridge.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        forbidden_prefixes = ("torch", "cupy", "jax", "tensorflow", "pycuda")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for pref in forbidden_prefixes:
                        self.assertFalse(
                            alias.name == pref or alias.name.startswith(f"{pref}."),
                            msg=f"Forbidden GPU/DL import: {alias.name}",
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for pref in forbidden_prefixes:
                    self.assertFalse(
                        node.module == pref or node.module.startswith(f"{pref}."),
                        msg=f"Forbidden GPU/DL import: {node.module}",
                    )

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        code = (
            f"import sys; sys.path.insert(0, {str(src_path.parent)!r}); "
            "import villa1739_cross_resolution_bridge; "
            "assert 'torch' not in sys.modules, 'torch unexpectedly loaded'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_02_cross_resolution_downsampling_and_2d_shift_registration(self) -> None:
        # Target 9.366 um canvas (120 x 140) and high-res 2.403 um canvas (~3.90x -> 468 x 546)
        h_lo, w_lo = 120, 140
        h_hi, w_hi = 468, 546
        true_dy, true_dx = -11, 14

        base_mask_lo = _make_letter_grid_mask(h_lo, w_lo)
        sup_mask_lo = np.zeros((h_lo, w_lo), dtype=bool)
        sup_mask_lo[12 : h_lo - 12, 12 : w_lo - 12] = True
        base_mask_lo &= sup_mask_lo

        # Reference 9.366 um map has the letter grid shifted by (true_dy, true_dx)
        ref_ink_lo = v1739.shift_2d_zero_pad(base_mask_lo, true_dy, true_dx)
        ref_u8 = np.where(ref_ink_lo, 220, 25).astype(np.uint8)

        # Upsample unshifted human mask & supervision to 2.403 um canvas (exact 3.9x ratio)
        yy_hi = np.clip((np.arange(h_hi) * h_lo) // h_hi, 0, h_lo - 1)
        xx_hi = np.clip((np.arange(w_hi) * w_lo) // w_hi, 0, w_lo - 1)
        human_ink_hi = np.where(base_mask_lo[yy_hi[:, None], xx_hi[None, :]], 255, 0).astype(np.uint8)
        human_sup_hi = np.where(sup_mask_lo[yy_hi[:, None], xx_hi[None, :]], 255, 0).astype(np.uint8)

        reg = v1739.register_2p4um_human_labels_to_9p366um(
            human_ink_hi, human_sup_hi, ref_u8, max_shift_px=24
        )
        info = reg["registration_info"]
        self.assertEqual(info["best_shift_dy_dx"], [true_dy, true_dx])
        self.assertLess(info["pre_alignment_iou"], 0.30)
        self.assertGreater(info["post_alignment_iou"], 0.90)
        self.assertAlmostEqual(info["scale_ratio_mean"], 3.9, places=2)

    def test_03_42d_ridge_recovers_onesided_ink_and_passes_pm8_killtest(self) -> None:
        rng = np.random.default_rng(1739)
        depth, h, w = 29, 160, 128
        ink_mask = _make_letter_grid_mask(h, w)
        blank_mask = ~ink_mask
        valid_mask = np.ones((h, w), dtype=bool)
        zz = np.arange(depth, dtype=np.float64)[:, None, None]

        # Symmetric papyrus sheet centered at z = 14 + one-sided ink layer at z = 14 + 1.8
        sheet = 135.0 * np.exp(-0.5 * ((zz - 14.0) / 2.2) ** 2)
        one_sided_ink = np.where(
            ink_mask[None, :, :],
            48.0 * np.exp(-0.5 * ((zz - 15.8) / 1.1) ** 2),
            0.0,
        )
        slab = np.clip(
            sheet + one_sided_ink + rng.normal(20.0, 3.5, (depth, h, w)), 1.0, 255.0
        ).astype(np.float32)

        sl_a, sl_b, _ = v1739.split_spatial_folds_with_buffer(h, buffer_px=32)
        std_a, val_a, _, _ = v1739.extract_and_standardize_42d_for_slab(
            slab[:, sl_a, :], valid_mask[sl_a, :], voxel_um=9.366
        )
        self.assertEqual(std_a.shape, (42, sl_a.stop, w))

        w_vec, bias, fit_info = v1739.fit_ridge_linear_classifier(
            std_a, ink_mask[sl_a, :].astype(np.float64), val_a, l2_lambda=100.0
        )
        self.assertEqual(fit_info["feature_dim"], 42)

        eval_b = v1739.evaluate_frozen_weights_on_and_off_sheet(
            slab[:, sl_b, :],
            valid_mask[sl_b, :],
            voxel_um=9.366,
            border_guard_px=8,
            off_sheet_shift_slices=8,
            models={"human_2p4um_bridged": (w_vec, bias)},
            label_targets={"human_2p4um_coregistered": (ink_mask[sl_b, :], blank_mask[sl_b, :])},
        )
        res = eval_b["models"]["human_2p4um_bridged"]["human_2p4um_coregistered"]
        self.assertGreater(res["on_sheet_auc"], 0.90)
        self.assertGreater(res["delta_null"], 0.20)
        self.assertGreater(res["drop_vs_pm8"], 0.30)
        self.assertTrue(res["passes_all_gates"])
        self.assertEqual(res["verdict"], "PASS")

    def test_04_macro_curvature_depth_invariant_artifact_rejected_by_pm8_killtest(self) -> None:
        rng = np.random.default_rng(404)
        depth, h, w = 29, 160, 128
        ink_mask = _make_letter_grid_mask(h, w)
        blank_mask = ~ink_mask
        valid_mask = np.ones((h, w), dtype=bool)
        zz = np.arange(depth, dtype=np.float64)[:, None, None]

        # Adversarial depth-invariant 2D artifact present across ALL depth slices equally
        sheet = 130.0 * np.exp(-0.5 * ((zz - 14.0) / 2.2) ** 2)
        artifact_2d = np.where(ink_mask, 35.0, 0.0)[None, :, :]
        slab = np.clip(
            sheet + artifact_2d + rng.normal(25.0, 3.0, (depth, h, w)), 1.0, 255.0
        ).astype(np.float32)

        sl_a, sl_b, _ = v1739.split_spatial_folds_with_buffer(h, buffer_px=32)
        std_a, val_a, _, _ = v1739.extract_and_standardize_42d_for_slab(
            slab[:, sl_a, :], valid_mask[sl_a, :], voxel_um=9.366
        )
        w_vec, bias, _ = v1739.fit_ridge_linear_classifier(
            std_a, ink_mask[sl_a, :].astype(np.float64), val_a, l2_lambda=100.0
        )

        eval_b = v1739.evaluate_frozen_weights_on_and_off_sheet(
            slab[:, sl_b, :],
            valid_mask[sl_b, :],
            voxel_um=9.366,
            border_guard_px=8,
            off_sheet_shift_slices=8,
            models={"human_2p4um_bridged": (w_vec, bias)},
            label_targets={"human_2p4um_coregistered": (ink_mask[sl_b, :], blank_mask[sl_b, :])},
        )
        res = eval_b["models"]["human_2p4um_bridged"]["human_2p4um_coregistered"]
        self.assertGreater(res["on_sheet_auc"], 0.85)
        self.assertGreater(res["off_sheet_pm8_mean_auc"], 0.85)
        self.assertLessEqual(res["drop_vs_pm8"], 0.0)
        self.assertFalse(res["passes_drop_vs_pm8"])
        self.assertFalse(res["passes_all_gates"])

    def test_05_locked_plus_polarity_rejects_inverted_test_signal(self) -> None:
        h, w = 80, 80
        ink = _make_letter_grid_mask(h, w)
        blank = ~ink
        vm = np.ones((h, w), dtype=bool)
        # Inverted score map where ink has strongly negative values
        score = np.where(ink, -5.0, 5.0).astype(np.float32)
        ev = v1739.evaluate_score_map_locked_plus(score, ink, blank, vm)
        self.assertLess(ev["auc_locked_plus"], 0.05)
        self.assertGreater(ev["unsigned_diagnostic_auc"], 0.95)
        self.assertEqual(ev["observed_polarity"], "-")

    def test_06_spatial_buffer_zone_zero_filter_footprint_leakage(self) -> None:
        rng = np.random.default_rng(999)
        depth, h, w = 25, 128, 96
        slab1 = rng.uniform(20.0, 220.0, size=(depth, h, w)).astype(np.float32)
        slab2 = slab1.copy()
        valid = np.ones((h, w), dtype=bool)

        sl_a, sl_b, split_info = v1739.split_spatial_folds_with_buffer(h, buffer_px=32)
        self.assertEqual(split_info["buffer_px"], 32)
        self.assertEqual(sl_b.start - sl_a.stop, 32)

        # Radically corrupt both the 32 px buffer zone AND all of Fold B in slab2
        slab2[:, sl_a.stop :, :] = 255.0

        std_a1, val_a1, _, _ = v1739.extract_and_standardize_42d_for_slab(
            slab1[:, sl_a, :], valid[sl_a, :], voxel_um=9.366
        )
        std_a2, val_a2, _, _ = v1739.extract_and_standardize_42d_for_slab(
            slab2[:, sl_a, :], valid[sl_a, :], voxel_um=9.366
        )
        # Bitwise identical Fold A features despite extreme corruption in buffer + Fold B
        np.testing.assert_array_equal(std_a1, std_a2)
        np.testing.assert_array_equal(val_a1, val_a2)

    def test_07_q22_global_gain_and_offset_invariance_of_standardization(self) -> None:
        rng = np.random.default_rng(22)
        feats_raw = rng.normal(10.0, 4.0, size=(42, 48, 48)).astype(np.float32)
        vm = np.ones((48, 48), dtype=bool)
        std1, _ = v1739.standardize_features_robust_per_roi(feats_raw, vm)
        # Apply positive affine transform: 3.7 * X - 42.5
        std2, _ = v1739.standardize_features_robust_per_roi(3.7 * feats_raw - 42.5, vm)
        np.testing.assert_allclose(std1, std2, atol=1e-5)

    def test_08_http_byte_budget_100mib_cap_enforcement(self) -> None:
        budget = v1739.MultiVolumeHttpByteBudget(
            {"pherc0841_9p366um": 30 * v1739.MIB, "pherc0343_8p64um": 30 * v1739.MIB, "pherc0306b_8p64um": 55 * v1739.MIB},
            global_limit_bytes=v1739.GLOBAL_HTTP_BUDGET_BYTES,
        )
        self.assertEqual(budget.global_limit, 100 * 1024 * 1024)
        budget.charge("pherc0841_9p366um", 25 * v1739.MIB, label="0841")
        budget.charge("pherc0343_8p64um", 25 * v1739.MIB, label="0343")
        with self.assertRaises(v1739.BudgetExceededError):
            budget.check_planned("pherc0306b_8p64um", 51 * v1739.MIB, label="exceed_100mib_global")

    def test_09_offline_end_to_end_pipeline_and_privacy_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2026)
            depth, h, w = 25, 128, 128
            ink_gt = _make_letter_grid_mask(h, w)
            zz = np.arange(depth, dtype=np.float64)[:, None, None]

            def _make_ink_slab() -> np.ndarray:
                sheet = 135.0 * np.exp(-0.5 * ((zz - 12.0) / 2.0) ** 2)
                ink_layer = np.where(ink_gt[None], 50.0 * np.exp(-0.5 * ((zz - 13.6) / 1.1) ** 2), 0.0)
                return np.clip(sheet + ink_layer + rng.normal(20.0, 3.0, (depth, h, w)), 1.0, 255.0).astype(
                    np.uint8
                )

            # 1) Synthetic PHerc. 0841 (9.366 um SV + 2.403 um Zarr v3 human labels + argus TIFF)
            z841 = root / "sv841.zarr"
            _write_local_zarr_v2(z841, _make_ink_slab(), chunks=(depth, 64, 64))
            h_hi, w_hi = h * 4, w * 4
            yy_hi = np.clip((np.arange(h_hi) * h) // h_hi, 0, h - 1)
            xx_hi = np.clip((np.arange(w_hi) * w) // w_hi, 0, w - 1)
            ink_z3 = root / "inklabels.zarr"
            sup_z3 = root / "supervision.zarr"
            _write_local_zarr3_sharded_u8(
                ink_z3, np.where(ink_gt[yy_hi[:, None], xx_hi[None, :]], 255, 0).astype(np.uint8), inner_chunks=(64, 64)
            )
            _write_local_zarr3_sharded_u8(
                sup_z3, np.full((h_hi, w_hi), 255, dtype=np.uint8), inner_chunks=(64, 64)
            )
            tif841 = root / "argus841.tif"
            tifffile.imwrite(str(tif841), np.where(ink_gt, 225, 25).astype(np.uint8))

            # 2) Synthetic PHerc. 0343 (8.64 um SV + argus TIFF)
            z343 = root / "sv343.zarr"
            _write_local_zarr_v2(z343, _make_ink_slab(), chunks=(depth, 64, 64))
            tif343 = root / "argus343.tif"
            tifffile.imwrite(str(tif343), np.where(ink_gt, 225, 25).astype(np.uint8))

            # 3) Synthetic PHerc. 0306B (8.64 um CT Zarr + tifxyz mesh + claim PNG/JSON)
            z306 = root / "ct306.zarr"
            # Build a small 3D CT volume (40, 140, 140) where z correlates with slice depth
            ct306 = np.transpose(_make_ink_slab(), (0, 1, 2))  # (25, 128, 128)
            _write_local_zarr_v2(z306, ct306, chunks=(25, 64, 64))
            mesh306 = root / "mesh306"
            mesh306.mkdir(parents=True, exist_ok=True)
            mg_v, mg_u = np.meshgrid(np.linspace(4, 120, 32), np.linspace(4, 120, 32), indexing="ij")
            tifffile.imwrite(str(mesh306 / "x.tif"), mg_u.astype(np.float32))
            tifffile.imwrite(str(mesh306 / "y.tif"), mg_v.astype(np.float32))
            tifffile.imwrite(str(mesh306 / "z.tif"), np.full((32, 32), 12.0, dtype=np.float32))
            (mesh306 / "meta.json").write_text(
                json.dumps({"scale": [1.0, 1.0], "carrier_origin_mm": [0.0, 0.0]}), encoding="utf-8"
            )
            page_rgb = np.array([243, 236, 222], dtype=np.uint8)
            ink_rgb = np.array([22, 20, 18], dtype=np.uint8)
            png_arr = np.where(ink_gt[..., None], ink_rgb[None, None, :], page_rgb[None, None, :])
            png306 = root / "claim306.png"
            Image.fromarray(png_arr, mode="RGB").save(str(png306))
            json306 = root / "claim306.json"
            json306.write_text(
                json.dumps(
                    {
                        "voxel_size_um": 8.64,
                        "px_per_mm": 1.0 / 0.00864,
                        "image_px": {"map_left": 0, "map_top": 0, "map_w": w, "map_h": h},
                        "context_mm": {"u0": 0.0, "u1": w * 0.00864, "v0": 0.0, "v1": h * 0.00864},
                        "claim_box_mm": {"u0": 0.0, "u1": w * 0.00864, "v0": 0.0, "v1": h * 0.00864},
                        "claim_box_image_px": {"x0": 0, "x1": w, "y0": 0, "y1": h},
                        "rows": {"centre_y_image_px": [h // 2]},
                        "display": {"page_rgb": [243, 236, 222], "ink_rgb": [22, 20, 18]},
                    }
                ),
                encoding="utf-8",
            )

            specs = {
                "pherc0841_9p366um": {
                    "sv_zarr_url": str(z841),
                    "human_2p4um_inklabels_zarr": str(ink_z3),
                    "human_2p4um_supervision_zarr": str(sup_z3),
                    "argus_ink_tif_url": str(tif841),
                    "border_guard_px": 6,
                    "registration_max_shift_px": 4,
                },
                "pherc0343_8p64um": {
                    "sv_zarr_url": str(z343),
                    "fallback_ink_tif_url": str(tif343),
                    "border_guard_px": 6,
                },
                "pherc0306b_8p64um": {
                    "ct_zarr_url": str(z306),
                    "mesh_url": str(mesh306),
                    "claim_png_url": str(png306),
                    "claim_json_url": str(json306),
                    "roi_override": (0, h, 0, w),
                    "num_slices": depth,
                    "border_guard_px": 6,
                },
            }

            receipt = v1739.run_villa1739_cross_resolution_bridge(
                specs=specs,
                roi_h=h,
                roi_w=w,
                spatial_buffer_px=16,
                off_sheet_shift_slices=8,
            )
            v1739.assert_no_local_paths(receipt)
            self.assertEqual(receipt["compute_mode"], "CPU_ONLY")
            self.assertTrue(receipt["http_byte_budget"]["within_limits"])
            fold_b_human = receipt["pherc0841_cross_resolution_bridge"]["fold_b_held_out_evaluation"][
                "models"
            ]["human_2p4um_bridged"]["human_2p4um_coregistered"]
            self.assertGreater(fold_b_human["on_sheet_auc"], 0.85)
            self.assertEqual(fold_b_human["verdict"], "PASS")

    def test_10_committed_live_receipt(self) -> None:
        receipt_path = (
            Path(__file__).resolve().parent
            / "evidence"
            / "villa1739_cross_resolution_bridge.json"
        )
        if not receipt_path.is_file():
            self.skipTest("Live Exp T12-Villa1739 evidence receipt not yet generated")
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        v1739.assert_no_local_paths(data)
        self.assertEqual(data["compute_mode"], "CPU_ONLY")
        b = data["http_byte_budget"]
        self.assertTrue(b["within_limits"])
        self.assertLessEqual(b["global_bytes"], 100 * 1024 * 1024)

        reg = data["pherc0841_cross_resolution_bridge"]["registration_2p4um_to_9p366um"]
        self.assertEqual(reg["source_shape"], [16460, 18560])
        self.assertEqual(reg["target_shape"], [4220, 4760])
        self.assertGreaterEqual(reg["post_alignment_iou"], reg["pre_alignment_iou"])

        split = data["pherc0841_cross_resolution_bridge"]["spatial_fold_split"]
        self.assertEqual(split["buffer_px"], 32)
        self.assertEqual(
            split["fold_b_row_range"][0] - split["fold_a_row_range"][1], 32
        )

        for mode in v1739.SUPERVISION_MODES:
            w_list = data["pherc0841_cross_resolution_bridge"]["fold_a_training"][mode]["weights"]
            self.assertEqual(len(w_list), 42)


if __name__ == "__main__":
    unittest.main()
