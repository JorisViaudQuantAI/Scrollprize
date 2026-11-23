# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline unit tests for Exp T3 (pherc0139_transverse_physics.py).

Strictly CPU-only, zero network I/O, synthetic data only.
Covers:
1. MSII spherical-volume occupancy invariants, curvature signs, and roughness.
2. 2D FFT fiber-axis detection & directional subtraction (FFT notch + StructN2V).
3. Exact Mann-Whitney ROC-AUC, Cohen's d, and 16-shift rolled-label null scoring.
4. Strict HTTP byte budget enforcement (per-part 60 MiB and global 120 MiB caps).
5. Offline end-to-end local Zarr + tifxyz pipeline, coordinate mapping, receipt
   schema, and privacy guardrails (zero local paths or cluster paths).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pherc0139_transverse_physics as t3  # noqa: E402
from stream_snap_render import write_tifxyz  # noqa: E402


def _write_local_zarr_v2(
    root: Path, arr: np.ndarray, *, chunks: tuple[int, int, int]
) -> None:
    level = root / "0"
    level.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 2,
        "shape": list(arr.shape),
        "chunks": list(chunks),
        "dtype": arr.dtype.str,
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
    }
    (level / ".zarray").write_text(json.dumps(meta), encoding="utf-8")
    cz, cy, cx = chunks
    nz = (arr.shape[0] + cz - 1) // cz
    ny = (arr.shape[1] + cy - 1) // cy
    nx = (arr.shape[2] + cx - 1) // cx
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                block = np.zeros(chunks, dtype=arr.dtype)
                slz = slice(iz * cz, min((iz + 1) * cz, arr.shape[0]))
                sly = slice(iy * cy, min((iy + 1) * cy, arr.shape[1]))
                slx = slice(ix * cx, min((ix + 1) * cx, arr.shape[2]))
                block[: slz.stop - slz.start, : sly.stop - sly.start, : slx.stop - slx.start] = arr[
                    slz, sly, slx
                ]
                cpath = level / str(iz) / str(iy) / str(ix)
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_bytes(block.tobytes(order="C"))


class TestMSIIAndHeightField(unittest.TestCase):
    def test_msii_flat_plane_and_shift_invariance(self) -> None:
        h, w = 64, 64
        vm = np.ones((h, w), dtype=bool)
        flat = np.zeros((h, w), dtype=np.float32)
        for r_um in (20.0, 60.0, 140.0):
            occ = t3.compute_msii_spherical_occupancy(flat, vm, voxel_um=2.4, radius_um=r_um)
            np.testing.assert_allclose(occ, 0.5, atol=1e-6)

        rng = np.random.default_rng(7)
        bumpy = rng.normal(0.0, 4.0, size=(h, w)).astype(np.float32)
        occ1 = t3.compute_msii_spherical_occupancy(bumpy, vm, voxel_um=2.4, radius_um=20.0)
        occ2 = t3.compute_msii_spherical_occupancy(bumpy + 123.45, vm, voxel_um=2.4, radius_um=20.0)
        np.testing.assert_allclose(occ1, occ2, atol=1e-5)

    def test_msii_convex_bump_vs_concave_pit_and_curvature(self) -> None:
        h, w = 96, 96
        voxel_um = 2.4
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        bump = 12.0 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2.0 * 6.0 ** 2))
        pit = -12.0 * np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2.0 * 6.0 ** 2))
        surf = (bump + pit).astype(np.float32)
        vm = np.ones((h, w), dtype=bool)

        feats = t3.compute_msii_and_curvature(
            surf, vm, voxel_um=voxel_um, radii_um=(20.0, 60.0, 140.0)
        )
        # Embossed convex bump has positive convexity (0.5 - V_r > 0) and positive -H.
        self.assertGreater(float(feats["msii_convexity_20um"][32, 32]), 0.05)
        self.assertLess(float(feats["msii_convexity_20um"][64, 64]), -0.05)
        self.assertGreater(float(feats["mean_curvature_convex"][32, 32]), 0.0)
        self.assertLess(float(feats["mean_curvature_convex"][64, 64]), 0.0)
        self.assertGreater(float(feats["gaussian_curvature"][32, 32]), 0.0)
        self.assertGreater(float(feats["relief_roughness_20um"][32, 32]), 0.5)

    def test_subvoxel_peak_extraction_recovers_embossed_height(self) -> None:
        depth, h, w = 41, 64, 64
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        z_true = 20.0 + 1.5 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2.0 * 5.0 ** 2))
        zz = np.arange(depth, dtype=np.float64)[:, None, None]
        slab = 180.0 * np.exp(-0.5 * ((zz - z_true[None, :, :]) / 2.2) ** 2)
        out = t3.extract_subvoxel_height_field(
            slab.astype(np.float32), voxel_um=2.399, detrend_sigma_um=200.0
        )
        self.assertEqual(out["z_center_slice"], 20)
        # Sub-voxel peak recovers the +1.5 voxel embossed bump accurately.
        delta_peak = float(out["n_peak"][32, 32] - out["n_peak"][4, 4])
        self.assertAlmostEqual(delta_peak, 1.5, delta=0.15)


class TestDirectionalFiberSubtraction(unittest.TestCase):
    def test_fft_axis_detection_and_fiber_suppression_preserves_crackle(self) -> None:
        h, w = 256, 256
        voxel_um = 2.4
        yy, xx = np.meshgrid(np.arange(h) * voxel_um, np.arange(w) * voxel_um, indexing="ij")

        # True orthogonal fiber directions: phi1 = 35 deg, phi2 = 125 deg.
        # A stripe running along phi has wavevector perpendicular to phi: theta = phi - 90.
        phi1_true = 35.0
        phi2_true = 125.0
        th1 = np.radians(phi1_true - 90.0)
        th2 = np.radians(phi2_true - 90.0)
        fiber1 = 30.0 * np.cos(2.0 * np.pi * (yy * np.sin(th1) + xx * np.cos(th1)) / 55.0)
        fiber2 = 22.0 * np.cos(2.0 * np.pi * (yy * np.sin(th2) + xx * np.cos(th2)) / 72.0)
        base = 120.0 + fiber1 + fiber2

        # Add localized isotropic micro-crackle (~6-10 um scale) inside an ink stroke region.
        from scipy import ndimage as ndi

        rng = np.random.default_rng(42)
        ink_mask = (np.abs(yy - h * voxel_um * 0.5) < 60.0) & (np.abs(xx - w * voxel_um * 0.5) < 160.0)
        blank_mask = ~ink_mask
        crackle_field = ndi.gaussian_filter(rng.normal(0.0, 20.0, size=(h, w)), sigma=1.5)
        crackle = np.where(ink_mask, crackle_field, 0.0)
        img = (base + crackle).astype(np.float32)
        vm = np.ones((h, w), dtype=bool)

        axes = t3.estimate_fiber_axes_fft(img, vm, voxel_um=voxel_um)
        est_phis = sorted([axes["fiber_axis_1_deg"], axes["fiber_axis_2_deg"]])
        true_phis = sorted([phi1_true, phi2_true])
        for est, tru in zip(est_phis, true_phis):
            ang_err = min(abs(est - tru), 180.0 - abs(est - tru))
            self.assertLessEqual(ang_err, 3.0)
        self.assertGreater(axes["anisotropy_ratio"], 5.0)

        fft_resid, removed = t3.directional_fft_notch(
            img,
            vm,
            voxel_um=voxel_um,
            freq_angle_1_deg=axes["freq_angle_1_deg"],
            freq_angle_2_deg=axes["freq_angle_2_deg"],
        )
        self.assertGreater(removed, 0.70)

        s2v_resid, r2 = t3.structn2v_fiber_subtract(
            img,
            vm,
            voxel_um=voxel_um,
            fiber_axis_1_deg=axes["fiber_axis_1_deg"],
            fiber_axis_2_deg=axes["fiber_axis_2_deg"],
        )
        self.assertGreater(r2, 0.70)
        # Blank papyrus variance in interior should drop strongly after both FFT notch and StructN2V.
        interior = np.zeros((h, w), dtype=bool)
        interior[32:-32, 32:-32] = True
        b_int = blank_mask & interior
        self.assertLess(float(np.var(fft_resid[b_int])), 0.10 * float(np.var(img[b_int])))
        self.assertLess(float(np.var(s2v_resid[b_int])), 0.10 * float(np.var(img[b_int])))

        # Crackle energy on fiber-subtracted residuals should clearly beat raw CT in interior.
        e_raw = t3.compute_crackle_energy(img, vm, voxel_um=voxel_um)
        e_fft = t3.compute_crackle_energy(fft_resid, vm, voxel_um=voxel_um)
        e_s2v = t3.compute_crackle_energy(s2v_resid, vm, voxel_um=voxel_um)
        auc_raw = t3.exact_roc_auc(e_raw[interior], ink_mask[interior])
        auc_fft = t3.exact_roc_auc(e_fft[interior], ink_mask[interior])
        auc_s2v = t3.exact_roc_auc(e_s2v[interior], ink_mask[interior])
        self.assertGreater(auc_fft, 0.95)
        self.assertGreater(auc_s2v, 0.95)
        self.assertGreater(auc_fft, auc_raw)
        self.assertGreater(auc_s2v, auc_raw)


class TestRolledNullAndBudget(unittest.TestCase):
    def test_16_shift_rolled_null_separates_true_signal_from_smooth_ramp(self) -> None:
        h, w = 80, 100
        shifts = t3.build_16_rolled_shifts((h, w))
        self.assertEqual(len(shifts), 16)
        self.assertEqual(len(set(shifts)), 16)
        self.assertTrue(all(dy > 0 and dx > 0 for dy, dx in shifts))

        rng = np.random.default_rng(123)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ink_mask = (yy >= 16) & (yy < 32) & (xx >= 25) & (xx < 75)
        blank_mask = ~ink_mask
        vm = np.ones((h, w), dtype=bool)

        # True localized signal passes gate cleanly.
        true_feat = rng.normal(0.0, 1.0, size=(h, w)) + np.where(ink_mask, 2.5, 0.0)
        res_true = t3.evaluate_feature_with_rolled_null(true_feat, ink_mask, blank_mask, vm)
        self.assertGreater(res_true["roc_auc"], 0.90)
        self.assertGreater(res_true["auc_above_null"], 0.30)
        self.assertTrue(res_true["passes_gate"])

        # Smooth linear spatial gradient gives spurious AUC > 0.70 yet is caught by rolled null.
        ramp = yy / float(h)
        res_ramp = t3.evaluate_feature_with_rolled_null(ramp, ink_mask, blank_mask, vm)
        self.assertGreater(res_ramp["roc_auc"], 0.70)
        self.assertGreater(res_ramp["rolled_null_mean"], 0.65)
        self.assertLess(res_ramp["auc_above_null"], 0.10)
        self.assertFalse(res_ramp["passes_gate"])

    def test_http_byte_budget_enforces_part_and_global_caps(self) -> None:
        budget = t3.HttpByteBudget(
            part_a_limit_bytes=1000, part_b_limit_bytes=1000, global_limit_bytes=1500
        )
        budget.check_planned("part_a", 900, label="ok_a")
        budget.charge("part_a", 900, label="ok_a")
        with self.assertRaises(t3.BudgetExceededError):
            budget.check_planned("part_a", 101, label="over_part_a")
        with self.assertRaises(t3.BudgetExceededError):
            budget.charge("part_a", 101, label="over_part_a_live")
        # Part B has 1000 remaining individually, but global only has 600 remaining!
        with self.assertRaises(t3.BudgetExceededError):
            budget.check_planned("part_b", 601, label="over_global")
        budget.charge("part_b", 600, label="ok_b")
        summary = budget.summary()
        self.assertEqual(summary["global_bytes"], 1500)
        self.assertTrue(summary["within_limits"])


class TestPrivacyAndOfflineEndToEnd(unittest.TestCase):
    def test_privacy_guardrail_rejects_local_and_cluster_paths(self) -> None:
        bad_payloads = [
            {"p": r"C:\Users\alice\data.json"},
            {"p": "D:/work/scroll.zarr"},
            {"p": "/Users/bob/secret"},
            {"p": "/home/carol/file"},
            {"p": "/gpfs/gb/data/visitor/file.h5"},
            {"p": "file:///tmp/test"},
        ]
        for bad in bad_payloads:
            with self.assertRaises(ValueError):
                t3.assert_no_local_paths(bad)

        good = {
            "source": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/a",
            "verdict": "CLOSED NO-GO",
        }
        t3.assert_no_local_paths(good)

    def test_offline_end_to_end_pipeline_and_receipt_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Synthetic 2.399 um surface volume (33, 64, 64) + ink map (64, 64)
            sv_h, sv_w, sv_d = 64, 64, 33
            yy, xx = np.meshgrid(np.arange(sv_h), np.arange(sv_w), indexing="ij")
            ink_gt = (yy >= 16) & (yy < 40) & (xx >= 16) & (xx < 48)
            ink_map = np.where(ink_gt, 230, 5).astype(np.uint8)
            ink_path = root / "ink.tif"
            tifffile.imwrite(str(ink_path), ink_map)

            # Emboss the synthetic sheet on ink pixels so MSII & height relief have true signal.
            z_center = 16.0 + np.where(ink_gt, 1.8, 0.0)
            zz = np.arange(sv_d, dtype=np.float64)[:, None, None]
            rng = np.random.default_rng(11)
            sv_vol = np.clip(
                170.0 * np.exp(-0.5 * ((zz - z_center[None, :, :]) / 2.5) ** 2)
                + rng.normal(0.0, 4.0, size=(sv_d, sv_h, sv_w)),
                1.0,
                255.0,
            ).astype(np.uint8)
            sv_zarr = root / "sv.zarr"
            _write_local_zarr_v2(sv_zarr, sv_vol, chunks=(33, 32, 32))

            # Synthetic aligned 9.362 um volumes (32, 32, 32) and planar meshes (16, 16)
            vb_shape = (32, 32, 32)
            gz = np.arange(32, dtype=np.float64)[:, None, None]
            # Plane sits at z = 16 in 9.362 um volume, with x,y covering [4..20)
            gy_m, gx_m = np.meshgrid(
                np.linspace(4.0, 20.0, 16, dtype=np.float32),
                np.linspace(4.0, 20.0, 16, dtype=np.float32),
                indexing="ij",
            )
            gz_m = np.full((16, 16), 16.0, dtype=np.float32)

            urls: dict[str, str] = {
                "surface_volume_2p4um": str(sv_zarr),
                "ink_reference_2p4um": str(ink_path),
            }
            for var, db, noise_sd in (("pag0", 0.0001, 6.0), ("pag50", 50.0, 3.0), ("pag1000", 1000.0, 2.0)):
                vdir = root / f"{var}.zarr"
                # Put a clear ink-correlated intensity contrast in pag50/pag1000
                ink_b = np.zeros((32, 32), dtype=bool)
                ink_b[8:14, 8:16] = True
                vol_b = np.clip(
                    150.0 * np.exp(-0.5 * ((gz - 16.0) / 1.8) ** 2)
                    + np.where(ink_b[None, :, :], 25.0, 0.0)
                    + rng.normal(0.0, noise_sd, size=vb_shape),
                    1.0,
                    255.0,
                ).astype(np.uint8)
                _write_local_zarr_v2(vdir, vol_b, chunks=(16, 16, 16))
                (vdir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "zarr_export": {
                                "target_window_f32_min": -0.03,
                                "target_window_f32_max": 0.145,
                            },
                            "scan": {
                                "tomo": {
                                    "acquisition": {"energy": 113.0, "sampleDetectorDistance": 1200.0},
                                    "processing": {
                                        "preprocessing": {
                                            "phase": {
                                                "method": "Paganin",
                                                "delta_beta": db,
                                                "unsharp_coeff": 0.0 if db < 1 else 4.0,
                                                "unsharp_sigma": 0.0 if db < 1 else 1.2,
                                            }
                                        },
                                        "postprocessing": {
                                            "32BitsConversion": {
                                                "dataset_used_min": -0.03,
                                                "dataset_used_max": 0.145,
                                            }
                                        },
                                    },
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                mdir = root / f"{var}.tifxyz"
                write_tifxyz(mdir, gx_m, gy_m, gz_m, meta={"scale": [0.05, 0.05]})
                urls[f"{var}_vol"] = str(vdir)
                urls[f"{var}_mesh"] = str(mdir)

            receipt = t3.run_pherc0139_ablation(
                urls=urls,
                roi_bounds_override=(0, 64, 0, 64),
                part_b_num_slices=11,
                stride_sv=4,
            )
            t3.assert_no_local_paths(receipt)
            self.assertEqual(receipt["compute_mode"], "CPU_ONLY")
            self.assertIn("part_a_2p4um", receipt)
            self.assertIn("part_b_9p362um_paganin_ablation", receipt)
            self.assertIn(receipt["verdict"]["overall"], ("PASS", "CLOSED NO-GO"))
            self.assertTrue(receipt["http_byte_budget"]["within_limits"])
            # Synthetic embossed height field should pass the predeclared gate in Part A!
            self.assertTrue(receipt["part_a_2p4um"]["features"]["height_detrended_um"]["passes_gate"])
            # All 3 Paganin variants must appear in the comparison table.
            comp = receipt["part_b_9p362um_paganin_ablation"]["comparison_table"]
            for var in ("pag0", "pag50", "pag1000"):
                self.assertIn(var, comp["raw_ct_peak"])
                self.assertIn(var, comp["hf_phase_edge_variance"])
                self.assertIn(var, comp["surface_height_relief_um"])

    def test_committed_live_receipt_if_present(self) -> None:
        receipt_path = (
            Path(__file__).resolve().parent
            / "evidence"
            / "pherc0139_transverse_physics_ablation.json"
        )
        if not receipt_path.is_file():
            self.skipTest("Live evidence receipt not yet generated")
        raw = receipt_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        t3.assert_no_local_paths(data)
        self.assertEqual(data["compute_mode"], "CPU_ONLY")
        self.assertEqual(data["scroll"], "PHerc0139")
        budget = data["http_byte_budget"]
        self.assertTrue(budget["within_limits"])
        self.assertLessEqual(budget["part_a_bytes"], 60 * 1024 * 1024)
        self.assertLessEqual(budget["part_b_bytes"], 60 * 1024 * 1024)
        self.assertLessEqual(budget["global_bytes"], 120 * 1024 * 1024)
        self.assertEqual(data["predeclared_gate"]["rolled_null_shifts"], 16)
        self.assertEqual(data["predeclared_gate"]["min_roc_auc"], 0.70)
        self.assertEqual(data["predeclared_gate"]["min_auc_above_rolled_null"], 0.10)
        # Honest verdict consistency check against predeclared gate (transverse hypotheses vs baselines).
        baselines = {"raw_ct_surface", "raw_crackle_energy"}
        part_a_trans = any(
            v["passes_gate"]
            for k, v in data["part_a_2p4um"]["features"].items()
            if k not in baselines
        )
        part_b_low_pag = any(
            data["part_b_9p362um_paganin_ablation"]["variants"][var]["features"][fn]["passes_gate"]
            for var in ("pag0", "pag50")
            for fn in data["part_b_9p362um_paganin_ablation"]["comparison_table"]
        )
        self.assertEqual(
            data["verdict"]["part_a_transverse"], "PASS" if part_a_trans else "CLOSED NO-GO"
        )
        self.assertEqual(
            data["verdict"]["part_b_low_paganin"], "PASS" if part_b_low_pag else "CLOSED NO-GO"
        )
        self.assertEqual(
            data["verdict"]["overall"],
            "PASS" if (part_a_trans or part_b_low_pag) else "CLOSED NO-GO",
        )


if __name__ == "__main__":
    unittest.main()
