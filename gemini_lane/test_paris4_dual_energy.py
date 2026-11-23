# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gemini_lane/paris4_dual_energy.py."""
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

import paris4_dual_energy as p4  # noqa: E402
import stream_snap_render as ssr  # noqa: E402


def _write_synthetic_zarr_v2(
    root: Path,
    arr_u8: np.ndarray,
    *,
    f32_min: float,
    f32_max: float,
    chunks: tuple[int, int, int] = (32, 32, 32),
    level: str = "0",
) -> Path:
    """Create a local OME-Zarr v2 store with metadata.json window_f32."""
    data = np.ascontiguousarray(arr_u8, dtype=np.uint8)
    arr_dir = root / level
    arr_dir.mkdir(parents=True, exist_ok=True)
    (root / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "zarr_export": {
                    "target_window_f32_min": float(f32_min),
                    "target_window_f32_max": float(f32_max),
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    zarray = {
        "zarr_format": 2,
        "shape": list(data.shape),
        "chunks": list(chunks),
        "dtype": "|u1",
        "compressor": None,
        "filters": None,
        "fill_value": 0,
        "order": "C",
        "dimension_separator": "/",
    }
    (arr_dir / ".zarray").write_text(json.dumps(zarray, indent=2) + "\n", encoding="utf-8")

    sz, sy, sx = data.shape
    cz, cy, cx = chunks
    for iz in range((sz + cz - 1) // cz):
        for iy in range((sy + cy - 1) // cy):
            for ix in range((sx + cx - 1) // cx):
                blk = np.zeros(chunks, dtype=np.uint8)
                z0, z1 = iz * cz, min((iz + 1) * cz, sz)
                y0, y1 = iy * cy, min((iy + 1) * cy, sy)
                x0, x1 = ix * cx, min((ix + 1) * cx, sx)
                blk[: z1 - z0, : y1 - y0, : x1 - x0] = data[z0:z1, y0:y1, x0:x1]
                cpath = arr_dir / str(iz) / str(iy) / str(ix)
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_bytes(blk.tobytes(order="C"))
    return root


def _mu_to_u8(mu: np.ndarray, f32_min: float, f32_max: float) -> np.ndarray:
    scaled = (mu - f32_min) / (f32_max - f32_min) * 255.0
    return np.clip(np.rint(scaled), 1, 255).astype(np.uint8)


def _build_synthetic_dual_energy_pair(
    shape: tuple[int, int, int] = (48, 80, 80),
    shift_zyx: tuple[int, int, int] = (3, -3, 3),
    ink_delta_mu: float = 0.035,
    seed: int = 42,
):
    """Build synthetic 78 keV & 137 keV volumes with textured papyrus sheet, injected 3-vx shift, and high-Z ink."""
    rng = np.random.default_rng(seed)
    sz, sy, sx = shape
    zz, yy, xx = np.meshgrid(
        np.arange(sz, dtype=np.float64),
        np.arange(sy, dtype=np.float64),
        np.arange(sx, dtype=np.float64),
        indexing="ij",
    )

    # Textured papyrus sheet centered at z = 24 with multi-frequency fibers so 3D NCC has a crisp peak.
    sheet_profile = np.exp(-0.5 * ((zz - 24.0) / 3.2) ** 2)
    fibers = (
        0.018 * np.sin(2.0 * np.pi * xx / 7.0 + 0.5 * np.cos(yy / 4.0))
        + 0.015 * np.cos(2.0 * np.pi * yy / 6.5)
        + 0.012 * np.sin(2.0 * np.pi * (xx + yy) / 9.0)
        + 0.010 * rng.normal(0.0, 1.0, size=shape)
    )
    from scipy import ndimage as ndi

    fibers = ndi.gaussian_filter(fibers, sigma=0.9)
    mu_78 = 0.03 + 0.085 * sheet_profile + fibers * (0.4 + 0.6 * sheet_profile)
    mu_78 += rng.normal(0.0, 0.0012, size=shape)

    # True linear papyrus relation: mu_137 = alpha * mu_78 + beta + noise.
    alpha_true = 0.62
    beta_true = 0.025
    mu_137_aligned = alpha_true * mu_78 + beta_true + rng.normal(0.0, 0.0008, size=shape)

    # Inject a high-Z surface ink stroke ONLY on the sheet surface (|z - 24| <= 1.2) in a localized box.
    ink_mask = (np.abs(zz - 24.0) <= 1.2) & (yy >= 30) & (yy < 50) & (xx >= 32) & (xx < 48)
    mu_137_aligned = mu_137_aligned + np.where(ink_mask, ink_delta_mu, 0.0)

    # Inject known 3-voxel translation: mov[z + dz, y + dy, x + dx] = aligned[z, y, x].
    dz, dy, dx = shift_zyx
    mu_137_shifted = np.roll(np.roll(np.roll(mu_137_aligned, dz, axis=0), dy, axis=1), dx, axis=2)

    w78 = (-0.04, 0.22)
    w137 = (0.01, 0.16)
    u8_78 = _mu_to_u8(mu_78, *w78)
    u8_137 = _mu_to_u8(mu_137_shifted, *w137)
    return u8_78, u8_137, w78, w137, alpha_true, beta_true, ink_mask


class TestParis4DualEnergy(unittest.TestCase):
    def test_a_window_f32_linear_conversion_and_metadata_load(self) -> None:
        """(a) Verify exact window_f32 linear attenuation conversion for 78, 137, and 54 keV."""
        for key, (lo, hi) in p4.DEFAULT_WINDOW_F32.items():
            u8 = np.array([0, 127, 255], dtype=np.uint8)
            mu = p4.uint8_to_linear_attenuation(u8, lo, hi)
            self.assertEqual(mu.dtype, np.float32)
            self.assertAlmostEqual(float(mu[0]), lo, places=6, msg=key)
            self.assertAlmostEqual(float(mu[1]), lo + (127.0 / 255.0) * (hi - lo), places=6, msg=key)
            self.assertAlmostEqual(float(mu[2]), hi, places=6, msg=key)

        with tempfile.TemporaryDirectory() as td:
            zroot = _write_synthetic_zarr_v2(
                Path(td) / "vol137.zarr",
                np.full((8, 8, 8), 128, dtype=np.uint8),
                f32_min=0.01,
                f32_max=0.16,
                chunks=(8, 8, 8),
            )
            lo, hi, _ = p4.load_volume_window_f32(zroot)
            self.assertAlmostEqual(lo, 0.01, places=6)
            self.assertAlmostEqual(hi, 0.16, places=6)

    def test_b_local_3d_ncc_recovers_3_voxel_shift_and_beats_6_nulls(self) -> None:
        """(b) Verify local 3D band-pass NCC recovers an injected 3-voxel shift and beats all 6 +/-20 um nulls."""
        shift_zyx = (3, -3, 3)
        u8_78, u8_137, w78, w137, _, _, _ = _build_synthetic_dual_energy_pair(
            shape=(44, 72, 72),
            shift_zyx=shift_zyx,
            ink_delta_mu=0.0,  # Pure papyrus texture lock test.
        )
        with tempfile.TemporaryDirectory() as td:
            r78_dir = _write_synthetic_zarr_v2(Path(td) / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1])
            r137_dir = _write_synthetic_zarr_v2(Path(td) / "v137.zarr", u8_137, f32_min=w137[0], f32_max=w137[1])

            with (
                ssr.RemoteZarrV2(r78_dir, level="0", cache_mb=16) as z78,
                ssr.RemoteZarrV2(r137_dir, level="0", cache_mb=16) as z137,
            ):
                b78, _ = z78.read_box([6, 8, 8], [38, 64, 64])
                b137, _ = z137.read_box([6, 8, 8], [38, 64, 64])

        mu_78 = p4.uint8_to_linear_attenuation(b78, *w78)
        mu_137 = p4.uint8_to_linear_attenuation(b137, *w137)
        mask = np.ones_like(b78, dtype=bool)

        # Spacing = 8.0 um/voxel -> 3 voxels = 24 um, null offset = 20 um (2.5 voxels).
        lock = p4.local_ncc_3d_lock(
            mu_78,
            mask,
            mu_137,
            mask,
            spacing_nvu_um=(8.0, 8.0, 8.0),
            max_shift_um=40.0,
            null_offset_um=20.0,
            min_ncc=0.30,
        )
        self.assertTrue(lock.passed, f"Expected NCC lock to pass, got {lock.to_dict()}")
        self.assertAlmostEqual(lock.shift_nvu_vx[0], float(shift_zyx[0]), delta=0.35)
        self.assertAlmostEqual(lock.shift_nvu_vx[1], float(shift_zyx[1]), delta=0.35)
        self.assertAlmostEqual(lock.shift_nvu_vx[2], float(shift_zyx[2]), delta=0.35)
        self.assertGreater(lock.ncc_peak, 0.75)
        self.assertEqual(len(lock.ncc_6_nulls), 6)
        for label, null_val in lock.ncc_6_nulls.items():
            self.assertGreater(
                lock.ncc_peak,
                null_val + 0.15,
                f"Peak NCC ({lock.ncc_peak:.4f}) failed to clearly beat null {label} ({null_val:.4f})",
            )

    def test_c_high_z_surface_ink_spike_and_ghosting_exclusion(self) -> None:
        """(c) Verify high-Z surface ink spike stands out cleanly above blank noise & ghosting rejects wrap-behind."""
        shift_zyx = (2, -2, 3)
        u8_78, u8_137, w78, w137, alpha_true, beta_true, _ = _build_synthetic_dual_energy_pair(
            shape=(48, 80, 80),
            shift_zyx=shift_zyx,
            ink_delta_mu=0.032,
            seed=123,
        )

        # Build local tifxyz plane at z = 24.0 spanning y in [10..70), x in [10..70).
        yy, xx = np.meshgrid(np.arange(10, 70, dtype=np.float32), np.arange(10, 70, dtype=np.float32), indexing="ij")
        zz = np.full_like(xx, 24.0)

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            r78_dir = _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1])
            r137_dir = _write_synthetic_zarr_v2(td_p / "v137.zarr", u8_137, f32_min=w137[0], f32_max=w137[1])
            mesh_dir = td_p / "mesh.tifxyz"
            ssr.write_tifxyz(mesh_dir, xx, yy, zz, meta={"scale": [1.0, 1.0]})

            eye4 = np.eye(4, dtype=np.float64)
            # Spacing 8.0 um/voxel -> n_offsets from -40 to +40 um in 8 um steps.
            n_offs = np.arange(-5, 6, dtype=np.float64) * 8.0

            with (
                p4.TifxyzPatchReader(mesh_dir) as mr,
                ssr.RemoteZarrV2(r78_dir, level="0", cache_mb=16) as z78,
                ssr.RemoteZarrV2(r137_dir, level="0", cache_mb=16) as z137,
            ):
                # Control B = left/top blank region (y in [12..26], x in [12..64] -> mesh rows [2..16]).
                tile_b = p4.evaluate_paired_tile(
                    mr,
                    z78,
                    z137,
                    eye4,
                    v0=2,
                    v1=18,
                    u0=4,
                    u1=56,
                    ref_voxel_um=8.0,
                    tgt_voxel_um=8.0,
                    ref_level=0,
                    tgt_level=0,
                    ref_window=w78,
                    tgt_window=w137,
                    uv_spacing_um=8.0,
                    n_offsets_um=n_offs,
                    max_shift_um=32.0,
                    null_offset_um=20.0,
                )
                # Test / Control A = central region containing the high-Z ink box (y in [22..58], x in [22..58]).
                tile_a = p4.evaluate_paired_tile(
                    mr,
                    z78,
                    z137,
                    eye4,
                    v0=12,
                    v1=48,
                    u0=12,
                    u1=48,
                    ref_voxel_um=8.0,
                    tgt_voxel_um=8.0,
                    ref_level=0,
                    tgt_level=0,
                    ref_window=w78,
                    tgt_window=w137,
                    uv_spacing_um=8.0,
                    n_offsets_um=n_offs,
                    max_shift_um=32.0,
                    null_offset_um=20.0,
                )

        self.assertTrue(tile_b["lock"].passed)
        self.assertTrue(tile_a["lock"].passed)

        # Fit papyrus background strictly on blank Control B.
        bg = p4.fit_papyrus_background(tile_b["mu_ref"], tile_b["mu_tgt"], tile_b["joint_valid"])
        self.assertAlmostEqual(float(bg["alpha"]), alpha_true, delta=0.08)
        self.assertAlmostEqual(float(bg["beta"]), beta_true, delta=0.01)

        res_b = p4.compute_dual_energy_residual(tile_b["mu_ref"], tile_b["mu_tgt"], float(bg["alpha"]), float(bg["beta"]))
        res_a = p4.compute_dual_energy_residual(tile_a["mu_ref"], tile_a["mu_tgt"], float(bg["alpha"]), float(bg["beta"]))

        th_b, _, _, v_b = p4.compute_depth_tophat(res_b, tile_b["joint_valid"], n_offs)
        th_a, _, _, v_a = p4.compute_depth_tophat(res_a, tile_a["joint_valid"], n_offs)
        blank_stats = p4.compute_blank_fpr_stats(th_b, v_b)
        z_a = (th_a - float(blank_stats["mean_blank"])) / float(blank_stats["sigma_blank"])

        # High-Z ink sits in the center of tile_a -> must spike far above blank noise (> 6 sigma).
        ink_peak_z = float(np.max(z_a[v_a]))
        self.assertGreater(ink_peak_z, 6.0, f"Expected high-Z surface ink peak > 6 sigma, got {ink_peak_z:.2f}")
        self.assertLess(float(blank_stats["fpr_4sigma"]), 0.01)

        # Verify ghosting exclusion flags a 150 um wrap-behind feature and passes a surface feature.
        g_offs = np.array([-200.0, -150.0, -100.0, 0.0, 100.0, 150.0, 200.0])
        slab_surf = np.zeros((len(g_offs), 20, 20), dtype=np.float32)
        slab_surf[3, 5:15, 5:15] = 0.03
        g_surf = p4.evaluate_ghosting_exclusion(slab_surf, np.ones_like(slab_surf, dtype=bool), g_offs)
        self.assertFalse(g_surf["is_ghost_dominated"])

        slab_ghost = np.zeros((len(g_offs), 20, 20), dtype=np.float32)
        slab_ghost[5, 5:15, 5:15] = 0.05  # Peaks at +150 um wrap-behind!
        slab_ghost[3, 5:15, 5:15] = 0.01
        g_wrap = p4.evaluate_ghosting_exclusion(slab_ghost, np.ones_like(slab_ghost, dtype=bool), g_offs)
        self.assertTrue(g_wrap["is_ghost_dominated"])

    def test_d_zero_local_paths_and_privacy_boundary(self) -> None:
        """(d) Verify emitted Step 0 JSON has zero local filesystem paths and refuses private writes in git repo."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            u8_78, u8_137, w78, w137, _, _, _ = _build_synthetic_dual_energy_pair(
                shape=(44, 72, 72), shift_zyx=(2, -1, 2), ink_delta_mu=0.025, seed=7
            )
            r78_dir = _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1])
            r137_dir = _write_synthetic_zarr_v2(td_p / "v137.zarr", u8_137, f32_min=w137[0], f32_max=w137[1])
            t78_path = td_p / "t78.json"
            t137_path = td_p / "t137.json"
            eye_json = json.dumps({"transformation_matrix": np.eye(4)[:3].tolist()})
            t78_path.write_text(eye_json, encoding="utf-8")
            t137_path.write_text(eye_json, encoding="utf-8")

            yy, xx = np.meshgrid(np.arange(6, 66, dtype=np.float32), np.arange(6, 66, dtype=np.float32), indexing="ij")
            zz = np.full_like(xx, 22.0)
            mesh_dir = td_p / "mesh.tifxyz"
            ssr.write_tifxyz(mesh_dir, xx, yy, zz, meta={"scale": [0.25, 0.25]})

            out_json = td_p / "preflight.json"
            p4.run_step0_preflight(
                vol_78_url=str(r78_dir),
                vol_137_url=str(r137_dir),
                transform_78_url=str(t78_path),
                transform_137_url=str(t137_path),
                mesh_78_url=str(mesh_dir),
                level=0,
                cache_mb=16.0,
                workers=2,
                out_json=out_json,
                tile_a_vu=(16, 44, 16, 44),
                tile_b_vu=(2, 18, 4, 48),
                tile_c_vu=(2, 18, 4, 48),
                uv_spacing_um=2.4,
            )
            raw_text = out_json.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
            p4.assert_no_local_paths(parsed)
            for pat in p4.FORBIDDEN_PATH_PATTERNS:
                self.assertIsNone(pat.search(raw_text), f"Forbidden path pattern matched in JSON: {pat.pattern}")

            # Verify public HTTPS URLs pass path sanitization while Windows/POSIX local paths fail.
            p4.assert_no_local_paths({"url": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/foo"})
            with self.assertRaises(ValueError):
                p4.assert_no_local_paths({"bad": "C:/Users/example/file.png"})
            with self.assertRaises(ValueError):
                p4.assert_no_local_paths({"bad": r"C:\Users\example\file.png"})

            # Verify privacy guard rejects writing private maps inside the git repository.
            with self.assertRaises(ValueError):
                p4._assert_outside_git_repo(HERE / "should_never_exist")

    def test_e_54_vs_78_psf_match_and_affine_composition(self) -> None:
        """(e) Verify 78<->DLS<->137 affine composition and 54 vs 78 keV PSF-matched residual R_{54/78}."""
        # Non-trivial rigid + scale affines simulating 78->DLS and 137->DLS.
        m78 = np.array(
            [
                [0.30, -0.05, 0.0, 100.0],
                [0.05, 0.30, 0.0, 200.0],
                [0.0, 0.0, 0.30, -50.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        m137 = np.array(
            [
                [0.29, -0.06, 0.0, 40.0],
                [0.06, 0.29, 0.0, 180.0],
                [0.0, 0.0, 0.30, 120.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        m_78_to_137 = p4.compose_affine_via_dls(m78, m137)
        pts = np.array([[1000.0, 2000.0, 45000.0], [1500.0, 2500.0, 46000.0]])
        p_dls_direct = p4.apply_affine_points(pts, m78)
        p_dls_via_137 = p4.apply_affine_points(p4.apply_affine_points(pts, m_78_to_137), m137)
        np.testing.assert_allclose(p_dls_direct, p_dls_via_137, atol=1e-9)

        # Verify PSF-matched 54 vs 78 keV residual R_{54/78} suppresses high-freq mismatch and recovers spike.
        rng = np.random.default_rng(99)
        vol_78 = 0.08 + 0.02 * rng.normal(0.0, 1.0, size=(11, 32, 32)).astype(np.float32)
        mask = np.ones_like(vol_78, dtype=bool)
        vol_78_psf = p4.psf_match_volume(
            vol_78, mask, src_fwhm_um=2.4, dst_fwhm_um=7.91, spacing_nvu_um=(2.4, 2.4, 2.4)
        )
        self.assertLess(float(vol_78_psf.std()), float(vol_78.std()))
        vol_54 = 0.85 * vol_78_psf + 0.015
        vol_54[5, 12:20, 12:20] += 0.04  # High-Z photoelectric boost at 54 keV.
        bg_mask = mask.copy()
        bg_mask[:, 10:22, 10:22] = False
        bg = p4.fit_papyrus_background(vol_78_psf, vol_54, bg_mask)
        r_54_78 = p4.compute_dual_energy_residual(vol_78_psf, vol_54, float(bg["alpha"]), float(bg["beta"]))
        self.assertAlmostEqual(float(r_54_78[5, 16, 16]), 0.04, delta=1e-4)
        self.assertLess(float(np.abs(r_54_78[bg_mask]).max()), 1e-4)

    def test_f_minimal_trilinear_parity_and_chunk_efficiency(self) -> None:
        """(f) Verify sample_trilinear_exact_minimal matches map_coordinates and fetches only touched chunks."""
        rng = np.random.default_rng(7)
        vol = rng.integers(10, 240, size=(32, 32, 32), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            zroot = _write_synthetic_zarr_v2(
                Path(td) / "v.zarr", vol, f32_min=0.0, f32_max=1.0, chunks=(16, 16, 16)
            )
            # Sample strictly inside chunk (0, 0, 0) at [0.25 .. 14.75]: should touch ONLY 1 chunk!
            pts = rng.uniform(0.25, 14.75, size=(200, 3))
            with ssr.RemoteZarrV2(zroot, level="0", cache_mb=16) as r_min:
                v_min = p4.sample_trilinear_exact_minimal(r_min, pts)
                chunks_min = r_min.chunks_fetched
            with ssr.RemoteZarrV2(zroot, level="0", cache_mb=16) as r_ref:
                v_ref = r_ref.sample_trilinear(pts)
            np.testing.assert_allclose(v_min, v_ref, atol=1e-4)
            self.assertEqual(chunks_min, 1, f"Expected exactly 1 chunk fetched, got {chunks_min}")

    def test_g_tps_deformation_and_transform_bundle(self) -> None:
        """(g) Verify 3D thin-plate-spline (TPS) transform bundle loading, batching, and mapping."""
        aff = np.array(
            [
                [0.5, 0.0, 0.0, 10.0],
                [0.0, 0.5, 0.0, 20.0],
                [0.0, 0.0, 0.5, 30.0],
            ],
            dtype=np.float64,
        )
        tps_inv = {
            "centre": [100.0, 200.0, 300.0],
            "scale": 50.0,
            "control_points": [[100.0, 200.0, 300.0], [150.0, 200.0, 300.0]],
            "weights": [[1.0, -0.5, 0.25], [-1.0, 0.5, -0.25]],
            "affine": [[0.5, 0.0, -0.5], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
        }
        payload = {
            "schema_version": "1.3.0",
            "transformation_matrix": aff.tolist(),
            "deformation": {"model": "thin-plate-spline", "kernel": "-r", "inverse": tps_inv},
        }
        with tempfile.TemporaryDirectory() as td:
            tpath = Path(td) / "r2_transform.json"
            tpath.write_text(json.dumps(payload), encoding="utf-8")
            bundle = p4.load_transform_bundle(tpath)

        self.assertEqual(bundle["matrix"].shape, (4, 4))
        self.assertIsNotNone(bundle["inverse_deformation"])
        self.assertEqual(bundle["inverse_deformation"]["kernel"], "-r")

        pts = np.array([[100.0, 200.0, 300.0], [125.0, 210.0, 290.0], [150.0, 200.0, 300.0]], dtype=np.float64)
        # Verify exact analytic value at control point 0 (x_norm = [0, 0, 0], r = [0, 1]).
        disp_single = p4.eval_tps_displacement(pts, bundle["inverse_deformation"], batch_size=1)
        disp_full = p4.eval_tps_displacement(pts, bundle["inverse_deformation"], batch_size=1024)
        np.testing.assert_allclose(disp_single, disp_full, atol=1e-12)
        expected_p0 = np.array([1.0 + 0.5, -0.5 + 0.0, 0.25 - 0.5], dtype=np.float64)
        np.testing.assert_allclose(disp_full[0], expected_p0, atol=1e-12)

        mapped = p4.map_points_affine_and_tps(pts, bundle["matrix"], tps_on_input=bundle["inverse_deformation"])
        expected_m0 = p4.apply_affine_points(pts[:1], bundle["matrix"])[0] + expected_p0
        np.testing.assert_allclose(mapped[0], expected_m0, atol=1e-12)

    def test_h_symmetric_psf_matching_in_evaluate_paired_tile(self) -> None:
        """(h) Verify symmetric PSF matching blurs finer target (e.g. 54 keV L1 vs 78 keV L3) and keeps native."""
        rng = np.random.default_rng(11)
        u8_78, u8_54, w78, w54, _, _, _ = _build_synthetic_dual_energy_pair(
            shape=(44, 64, 64), shift_zyx=(1, -1, 1), ink_delta_mu=0.0, seed=11
        )
        # Add high-frequency noise strictly to target to verify symmetric PSF smoothing acts on target.
        u8_54_noisy = np.clip(u8_54.astype(np.int32) + rng.integers(-12, 13, size=u8_54.shape), 1, 255).astype(np.uint8)
        yy, xx = np.meshgrid(np.arange(8, 56, dtype=np.float32), np.arange(8, 56, dtype=np.float32), indexing="ij")
        zz = np.full_like(xx, 22.0)

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            r78_dir = _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1])
            r54_dir = _write_synthetic_zarr_v2(td_p / "v54.zarr", u8_54_noisy, f32_min=w54[0], f32_max=w54[1])
            mesh_dir = td_p / "mesh.tifxyz"
            ssr.write_tifxyz(mesh_dir, xx, yy, zz, meta={"scale": [1.0, 1.0]})

            n_offs = np.arange(-6, 7, dtype=np.float64) * 8.0
            with (
                p4.TifxyzPatchReader(mesh_dir) as mr,
                ssr.RemoteZarrV2(r78_dir, level="0", cache_mb=16) as z78,
                ssr.RemoteZarrV2(r54_dir, level="0", cache_mb=16) as z54,
            ):
                # ref voxel = 9.6 um (coarser), tgt voxel = 5.0 um (finer) -> tgt must be PSF-matched to 9.6 um!
                tile = p4.evaluate_paired_tile(
                    mr,
                    z78,
                    z54,
                    np.eye(4, dtype=np.float64),
                    v0=4,
                    v1=44,
                    u0=4,
                    u1=44,
                    ref_voxel_um=9.6,
                    tgt_voxel_um=5.0,
                    ref_level=0,
                    tgt_level=0,
                    ref_window=w78,
                    tgt_window=w54,
                    uv_spacing_um=9.6,
                    n_offsets_um=n_offs,
                    max_shift_um=30.0,
                    null_offset_um=20.0,
                    psf_match_to_um=9.6,
                )
        self.assertAlmostEqual(float(tile["psf_matched_fwhm_um"]), 9.6, places=3)
        jv = tile["joint_valid"]
        self.assertLess(float(tile["mu_tgt"][jv].std()), float(tile["mu_tgt_native"][jv].std()))

    def test_i_crust_metrics_and_unread_coordinate_privacy_guard(self) -> None:
        """(i) Verify 1.129 um crust metrics detect surface crust and unread-coordinate guard blocks leaks."""
        n_offs = np.arange(-10, 11, dtype=np.float64) * 3.0  # -30..+30 um
        rng = np.random.default_rng(19)
        blank = 0.05 + rng.normal(0.0, 0.002, size=(len(n_offs), 32, 32)).astype(np.float32)
        crust = blank.copy()
        # Add thin surface crust at |n| <= 3 um with high-frequency spatial texture.
        surf_idx = np.abs(n_offs) <= 3.5
        checker = ((np.arange(32)[:, None] + np.arange(32)[None, :]) % 2).astype(np.float32)
        crust[surf_idx] += 0.025 + 0.015 * checker[None, :, :]
        valid = np.ones_like(blank, dtype=bool)

        m_b, th_b, _, v_b = p4.compute_crust_metrics(blank, valid, n_offs)
        m_a, th_a, _, v_a = p4.compute_crust_metrics(crust, valid, n_offs)
        sep = p4.compute_control_separation(th_a[v_a], th_b[v_b])
        self.assertGreater(float(m_a["tophat_mean"]), float(m_b["tophat_mean"]) + 0.02)
        self.assertGreater(float(m_a["surface_hf_roughness_rms"]), float(m_b["surface_hf_roughness_rms"]) * 2.0)
        self.assertGreater(float(sep["roc_auc"]), 0.99)

        # Unread coordinate privacy guard: aggregate counts pass, coordinate keys fail.
        p4.assert_no_unread_coordinates(
            {
                "controls_54_78": {"tile_a": {"tile_bounds_vu": [1, 2, 3, 4]}},
                "agraphon_strip_54_78": {"tiles_evaluated": 10, "any_stopping_rule_trigger": False},
                "outer_wraps_w130_w135": {"tiles_evaluated": 6, "ncc_peak_min": 0.85},
            }
        )
        for bad_key in ("u_range", "tile_bounds_vu", "z78_range", "wraps"):
            with self.assertRaises(ValueError):
                p4.assert_no_unread_coordinates({"agraphon_strip_54_78": {bad_key: [1, 2]}})
            with self.assertRaises(ValueError):
                p4.assert_no_unread_coordinates({"outer_wraps_w130_w135": {bad_key: [1, 2]}})

    def test_j_phase2_end_to_end_offline_and_budget_caps(self) -> None:
        """(j) Verify full Phase 2 pipeline offline (54/78, outer wraps, extended height, 1.129 crust) + guards."""
        with self.assertRaises(ValueError):
            p4.run_phase2_scans(cache_mb=300.0)
        with self.assertRaises(ValueError):
            p4.run_phase2_scans(max_total_mb=245.0)

        # Build synthetic 78 keV (level 2 = 9.6 um) and 54 keV (level 0 = 7.91 um) sharing physical space via m78_to_dls.
        u8_78, u8_54, w78, w54, _, _, _ = _build_synthetic_dual_energy_pair(
            shape=(56, 72, 72), shift_zyx=(2, -1, 2), ink_delta_mu=0.025, seed=29
        )
        # Map 78 keV L0 (2.4 um) -> 54 keV L0 (7.91 um): factor = 2.4 / 7.91 = 0.303413...
        # In our synthetic test, put u8_78 at level "2" (scale=4 -> 9.6 um) and u8_54 at level "0" (7.91 um)
        # so level-2 indices [0..56) correspond to L0 [0..224). Using m78_to_dls = diag(0.25) aligns L2(78) == L0(54)!
        s78_to_54 = 0.25
        m78_to_dls = np.diag([s78_to_54, s78_to_54, s78_to_54, 1.0])
        # Put u8_1129 at level "0" with m_1129_to_78 = diag(2.0) -> m_78_to_1129 = diag(0.5) aligns L1(78) == L0(1129).
        m1129_to_78 = np.diag([2.0, 2.0, 2.0, 1.0])

        # Mesh in 78 keV L0 coords: vertices spaced by 4 L0 voxels (= 1 L2 voxel = 9.6 um), z_L0 = 28*4 = 112.
        yy78, xx78 = np.meshgrid(
            np.arange(6, 66, dtype=np.float32) * 4.0,
            np.arange(6, 66, dtype=np.float32) * 4.0,
            indexing="ij",
        )
        zz78 = np.full_like(xx78, 28.0 * 4.0)
        # Outer mesh in 54 keV L0 coords: z = 28.0, spacing 1 voxel.
        yy54, xx54 = np.meshgrid(
            np.arange(6, 66, dtype=np.float32),
            np.arange(6, 66, dtype=np.float32),
            indexing="ij",
        )
        zz54 = np.full_like(xx54, 28.0)

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            r78_dir = _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1], level="2")
            # Also populate level "1" for crust stage (same synthetic array, L1 scale=2).
            _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1], level="1")
            r54_dir = _write_synthetic_zarr_v2(td_p / "v54.zarr", u8_54, f32_min=w54[0], f32_max=w54[1], level="0")
            r1129_dir = _write_synthetic_zarr_v2(td_p / "v1129.zarr", u8_54, f32_min=-0.03, f32_max=0.145, level="0")
            mesh_78 = td_p / "mesh78.tifxyz"
            mesh_outer = td_p / "mesh_outer.tifxyz"
            ssr.write_tifxyz(mesh_78, xx78, yy78, zz78, meta={"scale": [0.25, 0.25]})
            ssr.write_tifxyz(mesh_outer, xx54, yy54, zz54, meta={"scale": [1.0, 1.0]})

            t78_p = td_p / "t78.json"
            t1129_p = td_p / "t1129.json"
            t78_p.write_text(json.dumps({"transformation_matrix": m78_to_dls[:3].tolist()}), encoding="utf-8")
            t1129_p.write_text(json.dumps({"transformation_matrix": m1129_to_78[:3].tolist()}), encoding="utf-8")

            # Mesh for crust stage in L1 coords (L0 = 2 * L1 -> vertices spaced by 2 L0 voxels, z_L0 = 28*2 = 56).
            out_json = td_p / "phase2_public.json"
            priv_dir = td_p / "private_out"
            res = p4.run_phase2_scans(
                vol_78_url=str(r78_dir),
                vol_54_url=str(r54_dir),
                vol_1129_url=str(r1129_dir),
                transform_78_url=str(t78_p),
                transform_1129_url=str(t1129_p),
                mesh_78_url=str(mesh_78),
                mesh_outer_url=str(mesh_outer),
                preflight_level_78=2,
                preflight_level_54=0,
                strip_level_78=2,
                strip_level_54=0,
                crust_level_1129=0,
                crust_level_78=2,
                cache_mb=32.0,
                workers=2,
                max_total_mb=100.0,
                out_json=out_json,
                private_out_dir=priv_dir,
                preflight_tile_a_vu=(10, 50, 10, 50),
                preflight_tile_b_vu=(4, 30, 4, 52),
                strip_v_range=(4, 52),
                strip_wraps=(
                    ("wrap_08_blank_agraphon", 4, 28, "control_blank"),
                    ("wrap_09_agraphon", 28, 52, "test_agraphon"),
                ),
                outer_v_range=(6, 48),
                outer_wraps=(("w135", 6, 30), ("w134", 28, 52)),
                upper_ext_vu=(8, 44, 8, 44),
                lower_ext_vu=(10, 46, 10, 46),
                crust_tile_a_vu=(10, 36, 10, 36),
                crust_tile_b_vu=(4, 28, 4, 32),
                run_crust_1129=False,  # Crust tested separately with dedicated L0/L1 geometry below.
            )
            raw_text = out_json.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
            p4.assert_no_local_paths(parsed)
            p4.assert_no_unread_coordinates(parsed)
            self.assertTrue(res["preflight_54_78_passed"], f"Preflight failed: {res['controls_54_78']}")
            self.assertEqual(res["agraphon_strip_54_78"]["tiles_evaluated"], 2)
            self.assertEqual(res["outer_wraps_w130_w135"]["tiles_evaluated"], 2)
            self.assertTrue(res["extended_height_20260701"]["executed"])
            self.assertTrue((priv_dir / "phase2_private_summary.json").is_file())

            # Dedicated offline test of Stage 2B-3 (1.129 um native crust with TPS bundle).
            yy_c, xx_c = np.meshgrid(
                np.arange(6, 66, dtype=np.float32) * 2.0,
                np.arange(6, 66, dtype=np.float32) * 2.0,
                indexing="ij",
            )
            zz_c = np.full_like(xx_c, 28.0 * 2.0)
            mesh_c = td_p / "mesh_crust.tifxyz"
            ssr.write_tifxyz(mesh_c, xx_c, yy_c, zz_c, meta={"scale": [0.5, 0.5]})
            t78_c = td_p / "t78_c.json"
            t78_c.write_text(json.dumps({"transformation_matrix": np.diag([0.5, 0.5, 0.5, 1.0])[:3].tolist()}), encoding="utf-8")
            out_json_c = td_p / "phase2_crust_public.json"
            res_c = p4.run_phase2_scans(
                vol_78_url=str(r78_dir),
                vol_54_url=str(r54_dir),
                vol_1129_url=str(r1129_dir),
                transform_78_url=str(t78_c),
                transform_1129_url=str(t1129_p),
                mesh_78_url=str(mesh_c),
                mesh_outer_url=str(mesh_outer),
                preflight_level_78=1,
                preflight_level_54=0,
                crust_level_1129=0,
                crust_level_78=1,
                cache_mb=32.0,
                workers=2,
                max_total_mb=100.0,
                out_json=out_json_c,
                private_out_dir=priv_dir,
                preflight_tile_a_vu=(10, 50, 10, 50),
                preflight_tile_b_vu=(4, 30, 4, 52),
                crust_tile_a_vu=(10, 46, 10, 46),
                crust_tile_b_vu=(4, 34, 4, 46),
                run_strip=False,
                run_outer_wraps=False,
                run_extended_height=False,
                run_crust_1129=True,
            )
            p4.assert_no_local_paths(res_c)
            p4.assert_no_unread_coordinates(res_c)
            self.assertTrue(res_c["native_crust_1129_preflight"]["executed"])
            self.assertTrue(res_c["native_crust_1129_preflight"]["ncc_passed_both_controls"])


if __name__ == "__main__":
    unittest.main()
