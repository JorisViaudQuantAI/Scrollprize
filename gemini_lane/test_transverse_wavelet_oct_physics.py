# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline unit tests for Exp T5 (transverse_wavelet_oct_physics.py).

Strictly CPU-only, zero network I/O, zero torch imports, synthetic data only.
Covers:
1. Pure CPU / zero-torch import compliance.
2. OCT / Seismic 31D peak-aligned depth-waveform symmetry nulls, ink-shoulder
   asymmetry sensitivity, Local RX Mahalanobis, and Antisymmetric MNF vs PCA.
3. Astrophysics 2D Wavelet Scattering (S_1 & S_2) affine invariance, orientation
   anisotropy, and cross-orientation (90 deg) vs parallel (0 deg) modulation.
4. Kovesi Log-Gabor Phase Congruency (1999) strict contrast & offset invariance
   plus maximum/minimum covariance moments (edges M vs crackle corners m).
5. PHerc. 0306B claim-PNG tone inversion, off-sheet rejection, and analytical
   reading-orientation coordinate alignment verification.
6. Offline end-to-end dual-control pipeline, receipt schema, privacy guardrail
   (zero local C:\\Users\\... paths), and live receipt consistency check.
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
from scipy import ndimage as ndi
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transverse_wavelet_oct_physics as t5  # noqa: E402
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


class TestCpuOnlyCompliance(unittest.TestCase):
    def test_no_torch_imported(self) -> None:
        src_path = Path(__file__).resolve().parent / "transverse_wavelet_oct_physics.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name == "torch" or alias.name.startswith("torch."),
                        msg=f"Forbidden torch import: {alias.name}",
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(
                    node.module == "torch" or node.module.startswith("torch."),
                    msg=f"Forbidden torch import: {node.module}",
                )

        # Also verify in an isolated subprocess (immune to earlier test modules in discover).
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        code = (
            f"import sys; sys.path.insert(0, {str(src_path.parent)!r}); "
            "import transverse_wavelet_oct_physics; "
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


class TestOCT31DDepthWaveform(unittest.TestCase):
    def test_symmetric_profile_has_exact_zero_asymmetry(self) -> None:
        depth, h, w = 41, 32, 32
        zz = np.arange(depth, dtype=np.float64)[:, None, None]
        # Exact symmetric Gaussian centered at integer slice 20.0 -> zero asymmetry identically.
        slab = (180.0 * np.exp(-0.5 * ((zz - 20.0) / 3.0) ** 2)).astype(np.float32)
        slab = np.broadcast_to(slab, (depth, h, w)).copy()
        n_peak = np.full((h, w), 20.0, dtype=np.float32)
        vm = np.ones((h, w), dtype=bool)

        feats, diag = t5.compute_oct_depth_waveform_features(
            slab, n_peak, vm, voxel_um=2.4, half_span_vx=15.0, waveform_dim=31
        )
        self.assertEqual(diag["waveform_dim"], 31)
        self.assertEqual(diag["fully_in_bounds_fraction"], 1.0)
        np.testing.assert_allclose(feats["oct_hilbert_phase_asym"], 0.0, atol=1e-5)
        np.testing.assert_allclose(feats["oct_hilbert_env_asym"], 0.0, atol=1e-5)
        np.testing.assert_allclose(feats["oct_profile_abs_asym"], 0.0, atol=1e-5)
        np.testing.assert_allclose(feats["oct_depth_skewness"], 0.0, atol=1e-5)
        np.testing.assert_allclose(feats["oct_front_back_slope_ratio"], 0.0, atol=1e-5)
        np.testing.assert_allclose(feats["oct_antisym_energy_ratio"], 0.0, atol=1e-5)
        self.assertGreater(float(feats["oct_hilbert_env_peak"].mean()), 10.0)

    def test_one_sided_ink_shoulder_detected_by_oct_and_rx_and_mnf(self) -> None:
        depth, h, w = 45, 96, 96
        rng = np.random.default_rng(17)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ink_mask = (yy >= 34) & (yy < 62) & (xx >= 24) & (xx < 72)
        blank_mask = ~ink_mask
        vm = np.ones((h, w), dtype=bool)

        # Papyrus sheet with mild spatial waviness + one-sided ink crust shoulder at +3.5 slices.
        z0 = 22.0 + 0.8 * np.sin(2.0 * np.pi * xx / w)
        zz = np.arange(depth, dtype=np.float64)[:, None, None]
        base = 170.0 * np.exp(-0.5 * ((zz - z0[None]) / 2.8) ** 2)
        shoulder = np.where(
            ink_mask[None, :, :],
            38.0 * np.exp(-0.5 * ((zz - (z0[None] + 4.0)) / 1.4) ** 2),
            0.0,
        )
        slab = np.clip(base + shoulder + rng.normal(0.0, 2.0, size=(depth, h, w)), 1.0, 255.0).astype(
            np.float32
        )

        hf = t5.extract_subvoxel_height_field(slab, voxel_um=2.4, valid_mask=vm)
        feats, _ = t5.compute_oct_depth_waveform_features(
            slab, hf["n_peak"], hf["sheet_valid"], voxel_um=2.4, half_span_vx=15.0, waveform_dim=31
        )
        interior = np.zeros((h, w), dtype=bool)
        interior[8:-8, 8:-8] = True
        for key in (
            "oct_hilbert_phase_asym",
            "oct_hilbert_env_asym",
            "oct_profile_abs_asym",
            "oct_depth_skewness",
            "oct_antisym_energy_ratio",
            "oct_rx_mahalanobis_local",
            "oct_rx_mahalanobis_antisym",
            "oct_antisym_mnf_pc1",
            "oct_antisym_pca_pc1",
        ):
            res = t5.evaluate_feature_with_rolled_null(feats[key], ink_mask, blank_mask, interior)
            self.assertGreater(res["roc_auc"], 0.90, msg=f"{key} AUC too low: {res}")

    def test_antisymmetric_mnf_beats_pca_under_high_variance_spatial_noise(self) -> None:
        h, w, m_dim = 64, 64, 15
        rng = np.random.default_rng(101)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ink_mask = (yy >= 20) & (yy < 44) & (xx >= 16) & (xx < 48)
        vm = np.ones((h, w), dtype=bool)

        # Channel 0 carries spatially coherent ink signal (moderate variance).
        # Channel 1 carries 5x larger spatially white pixel noise (dominates PCA, suppressed by MNF).
        a_vec = rng.normal(0.0, 0.3, size=(m_dim, h, w))
        a_vec[0] += np.where(ink_mask, 3.0, 0.0)
        a_vec[1] += rng.normal(0.0, 6.0, size=(h, w))

        mnf_map, pca_map, diag = t5._antisymmetric_mnf_and_pca(a_vec, vm)
        auc_mnf = max(t5.exact_roc_auc(mnf_map.ravel(), ink_mask.ravel()), 1.0 - t5.exact_roc_auc(mnf_map.ravel(), ink_mask.ravel()))
        auc_pca = max(t5.exact_roc_auc(pca_map.ravel(), ink_mask.ravel()), 1.0 - t5.exact_roc_auc(pca_map.ravel(), ink_mask.ravel()))
        self.assertGreater(diag["mnf_top_snr"], 2.0)
        self.assertGreater(auc_mnf, 0.98)
        self.assertGreater(auc_mnf, auc_pca + 0.35)


class TestWaveletScattering2D(unittest.TestCase):
    def test_affine_invariance_and_anisotropy_and_s2_cross_modulation(self) -> None:
        h, w = 128, 128
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
        vm = np.ones((h, w), dtype=bool)

        # Left half: parallel-modulated horizontal grating (carrier along y, slow envelope along y -> Delta_theta = 0).
        # Right half: cross-modulated horizontal grating (carrier along y, slow envelope along x -> Delta_theta = 90).
        carrier = np.cos(2.0 * np.pi * yy / 4.0)
        env_par = 1.0 + 0.85 * np.cos(2.0 * np.pi * yy / 16.0)
        env_perp = 1.0 + 0.85 * np.cos(2.0 * np.pi * xx / 16.0)
        img = np.where(xx < w // 2, carrier * env_par, carrier * env_perp).astype(np.float32)

        f1, diag1 = t5.compute_wavelet_scattering_features(img, vm, voxel_um=8.64, avg_sigma_um=30.0)
        # Strict affine invariance: I -> 4.2 * I - 77.0.
        f2, _ = t5.compute_wavelet_scattering_features(4.2 * img - 77.0, vm, voxel_um=8.64, avg_sigma_um=30.0)
        for k in f1:
            np.testing.assert_allclose(f1[k], f2[k], atol=1e-5)
        self.assertEqual(diag1["s2_pairs_evaluated"], 18)

        # Cross-modulated half must have strictly higher S2 cross-to-parallel log-ratio than parallel half.
        left_ratio = float(f1["wst_s2_cross_par_ratio"][24:-24, 16 : w // 2 - 16].mean())
        right_ratio = float(f1["wst_s2_cross_par_ratio"][24:-24, w // 2 + 16 : -16].mean())
        self.assertGreater(right_ratio, left_ratio + 0.15)

        # Pure 1D grating has strictly higher S1 orientation anisotropy than isotropic white noise.
        rng = np.random.default_rng(7)
        iso_img = ndi.gaussian_filter(rng.normal(0.0, 1.0, size=(h, w)), sigma=1.0)
        f_stripe, _ = t5.compute_wavelet_scattering_features(carrier, vm, voxel_um=8.64, avg_sigma_um=30.0)
        f_iso, _ = t5.compute_wavelet_scattering_features(iso_img, vm, voxel_um=8.64, avg_sigma_um=30.0)
        self.assertGreater(
            float(f_stripe["wst_s1_anisotropy"][16:-16, 16:-16].mean()),
            float(f_iso["wst_s1_anisotropy"][16:-16, 16:-16].mean()) + 0.5,
        )


class TestKovesiPhaseCongruency(unittest.TestCase):
    def test_strict_contrast_and_offset_invariance(self) -> None:
        h, w = 96, 96
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rng = np.random.default_rng(5)
        img = np.where((yy >= 24) & (yy < 72) & (xx >= 24) & (xx < 72), 100.0, 20.0)
        img = img + rng.normal(0.0, 1.5, size=(h, w))
        vm = np.ones((h, w), dtype=bool)

        f1, _ = t5.compute_kovesi_phase_congruency_features(img, vm, voxel_um=8.64)
        f2, _ = t5.compute_kovesi_phase_congruency_features(7.25 * img - 135.0, vm, voxel_um=8.64)
        for k in f1:
            np.testing.assert_allclose(f1[k], f2[k], atol=1e-4, err_msg=f"Contrast invariance failed for {k}")

    def test_edges_vs_crackle_corners_covariance_moments(self) -> None:
        h, w = 128, 128
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        vm = np.ones((h, w), dtype=bool)

        # Clean 1D vertical step edge at x = 64 -> high max moment M, near-zero min moment m.
        edge_img = np.where(xx >= 64, 1.0, 0.0).astype(np.float64)
        # 2D square corner / checkerboard junctions -> high min moment m at corners.
        checker = (((yy // 16) + (xx // 16)) % 2).astype(np.float64)

        fe, _ = t5.compute_kovesi_phase_congruency_features(edge_img, vm, voxel_um=8.64)
        fc, _ = t5.compute_kovesi_phase_congruency_features(checker, vm, voxel_um=8.64)

        edge_m_max = float(fe["pc_max_moment_edges"][32:96, 62:66].max())
        edge_m_min = float(fe["pc_min_moment_corners"][32:96, 62:66].max())
        corner_m_min = float(fc["pc_min_moment_corners"][32:96, 32:96].max())

        self.assertGreater(edge_m_max, 0.25)
        self.assertLess(edge_m_min, 0.05 * edge_m_max)
        self.assertGreater(corner_m_min, 5.0 * max(edge_m_min, 1e-4))


class TestPHerc0306BGeometryAndOfflineEndToEnd(unittest.TestCase):
    def test_claim_png_tone_inversion_and_offsheet_rejection(self) -> None:
        claim_meta = {
            "display": {
                "page_rgb": [243.0, 236.0, 222.0],
                "ink_rgb": [22.0, 20.0, 18.0],
            }
        }
        page = np.array([243.0, 236.0, 222.0])
        ink = np.array([22.0, 20.0, 18.0])
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[0, :] = np.rint(page).astype(np.uint8)                            # d = 0.0
        rgb[1, :] = np.rint(0.5 * page + 0.5 * ink).astype(np.uint8)          # d = 0.5
        rgb[2, :] = np.rint(0.1 * page + 0.9 * ink).astype(np.uint8)          # d = 0.9
        rgb[3, :] = np.array([120, 150, 200], dtype=np.uint8)                 # off-sheet blue-grey

        d_map, on_sheet, _ = t5.decode_claim_png_tone(rgb, claim_meta)
        self.assertTrue(on_sheet[:3, :].all())
        self.assertFalse(on_sheet[3, :].any())
        np.testing.assert_allclose(d_map[0], 0.0, atol=0.01)
        np.testing.assert_allclose(d_map[1], 0.5, atol=0.01)
        np.testing.assert_allclose(d_map[2], 0.9, atol=0.01)

    def test_offline_end_to_end_dual_control_pipeline_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2026)

            # ------------------------------------------------------------------
            # 1. Synthetic PHerc. 0306B (8.64 um) local Zarr + tifxyz + PNG + JSON
            # ------------------------------------------------------------------
            # Build a compact synthetic claim PNG (160 x 192) with map_left=16, map_top=16,
            # map_w=160, map_h=128, px_per_mm=10.0 -> u in [0, 16] mm, v in [0, 12.8] mm.
            # Mesh cell_mm = 0.8 mm (scale=0.108 -> voxel_um=8.64 -> 1/scale*0.00864? Wait:
            # use voxel_size_um=100.0, px_per_mm=10.0, scale=0.1 -> cell_mm = 1.0 mm exact!).
            img_h, img_w = 160, 192
            map_left, map_top, map_w, map_h = 16, 16, 160, 128
            claim_meta = {
                "voxel_size_um": 100.0,
                "px_per_mm": 10.0,
                "context_mm": {"u0": 0.0, "u1": 16.0, "v0": 0.0, "v1": 12.8},
                "claim_box_mm": {"u0": 1.0, "u1": 15.0, "v0": 1.0, "v1": 11.8},
                "claim_box_image_px": {"x0": 26, "x1": 166, "y0": 26, "y1": 134},
                "image_px": {
                    "width": img_w,
                    "height": img_h,
                    "map_left": map_left,
                    "map_top": map_top,
                    "map_w": map_w,
                    "map_h": map_h,
                },
                "display": {
                    "page_rgb": [243.0, 236.0, 222.0],
                    "ink_rgb": [22.0, 20.0, 18.0],
                },
                "rows": {"centre_y_image_px": [64, 112]},
            }
            claim_json_path = root / "claim.json"
            claim_json_path.write_text(json.dumps(claim_meta), encoding="utf-8")

            # Mesh carrier: node (i, j) <-> (u, v) = (0 + 1.0*j, 0 + 1.0*i) mm; shape (14, 17).
            # Make upper-right corner (high v -> low y_img in flipped frame!) invalid so
            # flipped IoU strictly exceeds unflipped IoU!
            mh, mw = 14, 17
            gi, gj = np.meshgrid(np.arange(mh, dtype=np.float32), np.arange(mw, dtype=np.float32), indexing="ij")
            mx = 4.0 + gj * 3.0
            my = 4.0 + gi * 3.0
            mz = np.full((mh, mw), 16.0, dtype=np.float32)
            # Invalidate top-right in v-high (i >= 11, j >= 13) -> maps to y_img near map_top!
            invalid_corner = (gi >= 11) & (gj >= 13)
            mx[invalid_corner] = -1.0
            my[invalid_corner] = -1.0
            mz[invalid_corner] = -1.0
            mesh_dir = root / "mesh306"
            write_tifxyz(
                mesh_dir, mx, my, mz,
                meta={"scale": [0.1, 0.1], "carrier_origin_mm": [0.0, 0.0]},
            )

            # Build PNG matching mesh validity and placing a clear Line 1 ink band at y_img in [54..74], x_img in [46..110].
            page = np.array([243.0, 236.0, 222.0], dtype=np.float32)
            ink = np.array([22.0, 20.0, 18.0], dtype=np.float32)
            offsheet = np.array([130.0, 150.0, 185.0], dtype=np.float32)
            png = np.broadcast_to(page[None, None, :], (img_h, img_w, 3)).copy()
            yy_p, xx_p = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing="ij")
            gv_p, gu_p, _, _ = t5.map_0306b_image_px_to_mesh_uv(
                yy_p, xx_p, claim_meta, {"scale": [0.1, 0.1], "carrier_origin_mm": [0.0, 0.0]}, flip_v=True
            )
            iv_p = np.rint(gv_p).astype(np.int64)
            iu_p = np.rint(gu_p).astype(np.int64)
            inb_p = (iv_p >= 0) & (iv_p < mh) & (iu_p >= 0) & (iu_p < mw)
            off_p = np.ones((img_h, img_w), dtype=bool)
            off_p[inb_p] = invalid_corner[iv_p[inb_p], iu_p[inb_p]]
            png[off_p] = offsheet

            ink_band_306 = (yy_p >= 54) & (yy_p < 74) & (xx_p >= 46) & (xx_p < 110) & (~off_p)
            png[ink_band_306] = 0.2 * page + 0.8 * ink
            png_path = root / "claim.png"
            Image.fromarray(np.rint(png).astype(np.uint8), mode="RGB").save(str(png_path))

            # Build 3D CT Zarr (32, 64, 64) with papyrus peak at z=16 and asymmetric ink shoulder
            # in the 3D region corresponding to ink_band_306!
            # Compute voxel coords of ink_band_306 in (z, y, x):
            # y_img 54..74 -> v_mm = 12.8 - (54..74 - 16)/10 = 7.0..9.0 -> gi = 7..9 -> my = 25..31
            # x_img 46..110 -> u_mm = (46..110 - 16)/10 = 3.0..9.4 -> gj = 3..9.4 -> mx = 13..32.2
            vz, vy, vx = 32, 64, 64
            gz = np.arange(vz, dtype=np.float64)[:, None, None]
            gy = np.arange(vy, dtype=np.float64)[None, :, None]
            gx = np.arange(vx, dtype=np.float64)[None, None, :]
            ink_3d = (gy >= 25.0) & (gy <= 31.5) & (gx >= 13.0) & (gx <= 32.5)
            vol_306 = 160.0 * np.exp(-0.5 * ((gz - 16.0) / 2.2) ** 2)
            vol_306 = vol_306 + np.where(ink_3d, 45.0 * np.exp(-0.5 * ((gz - 19.5) / 1.2) ** 2), 0.0)
            vol_306 = np.clip(vol_306 + rng.normal(0.0, 2.0, size=(vz, vy, vx)), 1.0, 255.0).astype(np.uint8)
            zarr_306 = root / "ct306.zarr"
            _write_local_zarr_v2(zarr_306, vol_306, chunks=(16, 32, 32))

            # ------------------------------------------------------------------
            # 2. Synthetic PHerc. 0139 (2.399 um) surface volume Zarr + ink TIFF
            # ------------------------------------------------------------------
            sv_d, sv_h, sv_w = 41, 96, 96
            yy139, xx139 = np.meshgrid(np.arange(sv_h), np.arange(sv_w), indexing="ij")
            ink_gt_139 = (yy139 >= 32) & (yy139 < 64) & (xx139 >= 24) & (xx139 < 72)
            ink_u8_139 = np.where(ink_gt_139, 230, 32).astype(np.uint8)
            ink_139_path = root / "ink139.tif"
            tifffile.imwrite(str(ink_139_path), ink_u8_139)

            gz139 = np.arange(sv_d, dtype=np.float64)[:, None, None]
            sv_139 = 165.0 * np.exp(-0.5 * ((gz139 - 20.0) / 2.5) ** 2)
            sv_139 = sv_139 + np.where(
                ink_gt_139[None], 40.0 * np.exp(-0.5 * ((gz139 - 24.0) / 1.3) ** 2), 0.0
            )
            sv_139 = np.clip(sv_139 + rng.normal(0.0, 2.5, size=(sv_d, sv_h, sv_w)), 1.0, 255.0).astype(np.uint8)
            zarr_139 = root / "sv139.zarr"
            _write_local_zarr_v2(zarr_139, sv_139, chunks=(41, 48, 48))

            receipt = t5.run_wavelet_oct_ablation(
                urls={
                    "pherc0306b_ct_zarr": str(zarr_306),
                    "pherc0306b_mesh": str(mesh_dir),
                    "pherc0306b_claim_png": str(png_path),
                    "pherc0306b_claim_json": str(claim_json_path),
                    "pherc0139_sv_zarr": str(zarr_139),
                    "pherc0139_ink_tif": str(ink_139_path),
                },
                pherc0306b_roi_override=(36, 92, 36, 120),
                pherc0306b_num_slices=17,
                pherc0139_roi_bounds=(0, 96, 0, 96),
            )
            t5.assert_no_local_paths(receipt)
            self.assertEqual(receipt["compute_mode"], "CPU_ONLY")
            self.assertTrue(receipt["http_byte_budget"]["within_limits"])
            self.assertTrue(receipt["pherc0306b_8p64um"]["alignment_verification"]["orientation_verified_flipped"])
            self.assertGreater(
                receipt["pherc0306b_8p64um"]["alignment_verification"]["on_sheet_iou_reading_flip"],
                receipt["pherc0306b_8p64um"]["alignment_verification"]["on_sheet_iou_unflipped"],
            )

            expected_features = {f for fams in t5.FEATURE_FAMILIES.values() for f in fams}
            self.assertEqual(set(receipt["pherc0306b_8p64um"]["features"].keys()), expected_features)
            self.assertEqual(set(receipt["pherc0139_2p399um"]["features"].keys()), expected_features)
            # Both synthetic controls have a strong asymmetric ink shoulder -> PASS!
            self.assertEqual(receipt["verdict"]["pherc0306b_transverse"], "PASS")
            self.assertEqual(receipt["verdict"]["pherc0139_transverse"], "PASS")
            self.assertEqual(receipt["verdict"]["overall"], "PASS")

    def test_committed_live_receipt_if_present(self) -> None:
        receipt_path = (
            Path(__file__).resolve().parent
            / "evidence"
            / "transverse_wavelet_oct_ablation.json"
        )
        if not receipt_path.is_file():
            self.skipTest("Live Exp T5 evidence receipt not yet generated")
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        t5.assert_no_local_paths(data)
        self.assertEqual(data["compute_mode"], "CPU_ONLY")
        b = data["http_byte_budget"]
        self.assertTrue(b["within_limits"])
        self.assertLessEqual(b["pherc0306b_bytes"], 60 * 1024 * 1024)
        self.assertLessEqual(b["pherc0139_bytes"], 60 * 1024 * 1024)
        self.assertLessEqual(b["global_bytes"], 120 * 1024 * 1024)

        # Verify predeclared gate and honest verdict consistency.
        g = data["predeclared_gate"]
        self.assertEqual(g["pherc0306b_min_roc_auc"], 0.70)
        self.assertEqual(g["pherc0139_min_roc_auc"], 0.765)
        self.assertEqual(g["rolled_null_shifts"], 16)

        pass_306 = any(
            v["passes_gate"]
            for k, v in data["pherc0306b_8p64um"]["features"].items()
            if k not in t5.BASELINE_FEATURE_NAMES
        )
        pass_139 = any(
            v["passes_gate"]
            for k, v in data["pherc0139_2p399um"]["features"].items()
            if k not in t5.BASELINE_FEATURE_NAMES
        )
        self.assertEqual(data["verdict"]["pherc0306b_transverse"], "PASS" if pass_306 else "CLOSED NO-GO")
        self.assertEqual(data["verdict"]["pherc0139_transverse"], "PASS" if pass_139 else "CLOSED NO-GO")
        self.assertEqual(data["verdict"]["overall"], "PASS" if (pass_306 or pass_139) else "CLOSED NO-GO")

        # Verify PHerc0139 raw_ct_surface matches Exp T3 (0.7148) within 1e-4!
        self.assertAlmostEqual(
            data["pherc0139_2p399um"]["features"]["raw_ct_surface"]["roc_auc"],
            0.71484,
            places=4,
        )
        self.assertTrue(data["pherc0306b_8p64um"]["alignment_verification"]["orientation_verified_flipped"])


if __name__ == "__main__":
    unittest.main()
