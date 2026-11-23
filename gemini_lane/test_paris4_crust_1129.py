# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gemini_lane/paris4_crust_1129.py."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paris4_crust_1129 as c1129  # noqa: E402
import paris4_dual_energy as p4  # noqa: E402
import stream_snap_render as ssr  # noqa: E402
from test_paris4_dual_energy import _mu_to_u8, _write_synthetic_zarr_v2  # noqa: E402


def _build_shifted_papyrus_crust_volumes(
    *,
    add_crackle_a: bool = True,
    seed: int = 101,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float], np.ndarray]:
    """Build physically consistent synthetic 78 keV L1 (4.8 um/vx) and 1.129 um L0 (1.129 um/vx) volumes.

    Physical domain: Z in [0, 144 um], Y in [0, 144 um], X in [0, 144 um].
    Mesh sits at Z_um = 72.0 um.
    Right half (X_um >= 72.0) = Control A (papyrus peak shifted by +12.0 um to Z_um = 84.0 um + optional 2D crackle).
    Left half  (X_um <  72.0) = Control B (papyrus peak centered at Z_um = 72.0 um, blank 1D fibers only).
    """
    rng = np.random.default_rng(seed)
    v78_l1_um = 4.8
    v1129_um = 1.129

    def _eval_phys(z_um: np.ndarray, y_um: np.ndarray, x_um: np.ndarray, *, with_crackle: bool) -> np.ndarray:
        z_peak = np.where(x_um >= 72.0, 84.0, 72.0)
        sheet = np.exp(-0.5 * ((z_um - z_peak) / 8.5) ** 2)
        # Multi-frequency physical papyrus fibers shared across both volumes for crisp 3D NCC lock.
        fibers = (
            0.015 * np.sin(2.0 * np.pi * y_um / 22.0 + 0.5 * np.cos(x_um / 14.0))
            + 0.013 * np.cos(2.0 * np.pi * x_um / 26.0)
            + 0.010 * np.sin(2.0 * np.pi * (x_um + y_um) / 19.0)
        )
        mu = 0.01 + 0.065 * sheet + fibers * (0.35 + 0.65 * sheet)
        if with_crackle:
            on_a_surf = (x_um >= 72.0) & (np.abs(z_um - 84.0) <= 3.0)
            crackle = 0.018 * np.sin(2.0 * np.pi * x_um / 5.0) * np.sin(2.0 * np.pi * y_um / 5.0)
            mu = mu + np.where(on_a_surf, 0.014 + crackle, 0.0)
        return mu

    # 78 keV L1 grid: 30x30x30 voxels at 4.8 um/vx -> 144 um cube.
    s78 = (30, 30, 30)
    z78, y78, x78 = np.meshgrid(
        (np.arange(s78[0]) + 0.5) * v78_l1_um,
        (np.arange(s78[1]) + 0.5) * v78_l1_um,
        (np.arange(s78[2]) + 0.5) * v78_l1_um,
        indexing="ij",
    )
    mu_78 = _eval_phys(z78, y78, x78, with_crackle=False) + rng.normal(0.0, 0.0008, size=s78)

    # 1.129 um L0 grid: 128x128x128 voxels at 1.129 um/vx -> 144.5 um cube.
    s11 = (128, 128, 128)
    z11, y11, x11 = np.meshgrid(
        (np.arange(s11[0]) + 0.5) * v1129_um,
        (np.arange(s11[1]) + 0.5) * v1129_um,
        (np.arange(s11[2]) + 0.5) * v1129_um,
        indexing="ij",
    )
    mu_1129 = _eval_phys(z11, y11, x11, with_crackle=add_crackle_a) + rng.normal(0.0, 0.0012, size=s11)
    if add_crackle_a:
        # Add fine 1.129 um granular crust noise on shifted Control A surface peak.
        on_a = (x11 >= 72.0) & (np.abs(z11 - 84.0) <= 3.0)
        mu_1129 = mu_1129 + np.where(on_a, rng.normal(0.0, 0.010, size=s11), 0.0)

    # Exact physical affine mapping 1.129 um L0 coords -> 78 keV L0 (2.4 um) coords.
    r_scale = v1129_um / 2.4
    m_1129_to_78 = np.diag([r_scale, r_scale, r_scale, 1.0])
    w78 = (-0.04, 0.22)
    w1129 = (-0.03, 0.145)
    return _mu_to_u8(mu_78, *w78), _mu_to_u8(mu_1129, *w1129), w78, w1129, m_1129_to_78


class TestParis4Crust1129(unittest.TestCase):
    def test_1_papyrus_peak_recentering_recovers_normal_offset_and_blocks_adjacent_wrap(self) -> None:
        """Verify argmax_n mu_1129 recovers a +18.6 um normal offset while ignoring a +42 um neighbor wrap."""
        n_offs = np.arange(-24, 25, dtype=np.float64) * 2.0  # -48..+48 um
        out_offs = np.arange(-12, 13, dtype=np.float64) * 2.0  # -24..+24 um
        h, w = 40, 40
        true_offset_um = 18.6

        # Primary papyrus sheet shifted to +18.6 um; stronger neighboring wrap at -44 um (outside search=36 um).
        prof_main = 0.065 * np.exp(-0.5 * ((n_offs - true_offset_um) / 6.0) ** 2)
        prof_neighbor = 0.095 * np.exp(-0.5 * ((n_offs + 44.0) / 5.0) ** 2)
        slab = np.broadcast_to((0.005 + prof_main + prof_neighbor)[:, None, None], (len(n_offs), h, w)).astype(
            np.float32
        )
        valid = np.ones_like(slab, dtype=bool)

        rec = c1129.recenter_papyrus_peak_slab(
            slab,
            valid,
            n_offs,
            uv_spacing_um=1.5,
            search_half_width_um=36.0,
            out_n_offsets_um=out_offs,
        )
        # Must lock onto +18.6 um (NOT the stronger -44 um adjacent wrap outside 36 um!).
        self.assertAlmostEqual(float(rec["stats"]["n_peak_median_um"]), true_offset_um, delta=0.35)
        self.assertGreater(
            float(rec["stats"]["post_recenter_surface_mu_mean"]),
            float(rec["stats"]["pre_recenter_surface_mu_mean"]) + 0.04,
        )
        # Recentered peak must sit at index n' = 0 (mid index = 12).
        mid_k = len(out_offs) // 2
        mean_prof = rec["mu_recentered"].mean(axis=(1, 2))
        self.assertEqual(int(np.argmax(mean_prof)), mid_k)

    def test_1b_two_stage_recentering_prevents_bimodal_layer_jump_cliffs(self) -> None:
        """Verify two-stage recentering locks onto one continuous sheet without jumping across bimodal layers."""
        n_offs = np.arange(-24, 25, dtype=np.float64) * 2.0  # -48..+48 um
        out_offs = np.arange(-12, 13, dtype=np.float64) * 2.0
        h, w = 48, 48
        rng = np.random.default_rng(55)
        # Main sheet near -6 um, competing second layer at +32 um with tiny random amplitude fluctuations.
        s1 = 0.062 * np.exp(-0.5 * ((n_offs + 6.0) / 5.5) ** 2)[:, None, None]
        s2 = 0.060 * np.exp(-0.5 * ((n_offs - 32.0) / 5.5) ** 2)[:, None, None]
        # Slight spatially varying bump on s2 that would make naive argmax jump by 38 um on half the tile.
        bump = np.where(np.arange(w)[None, None, :] >= w // 2, 1.08, 0.92)
        slab = (0.005 + s1 + s2 * bump + rng.normal(0.0, 0.001, (len(n_offs), h, w))).astype(np.float32)
        valid = np.ones_like(slab, dtype=bool)

        rec = c1129.recenter_papyrus_peak_slab(
            slab,
            valid,
            n_offs,
            uv_spacing_um=1.5,
            search_half_width_um=36.0,
            local_track_half_width_um=10.0,
            out_n_offsets_um=out_offs,
        )
        # Must stay continuously on the sheet nearest n = 0 (-6 um) with tight std (< 2.5 um), zero 38 um cliff!
        self.assertAlmostEqual(float(rec["stats"]["n_tile_anchor_um"]), -6.0, delta=2.1)
        self.assertAlmostEqual(float(rec["stats"]["n_peak_median_um"]), -6.0, delta=2.0)
        self.assertLess(float(rec["stats"]["n_peak_std_um"]), 2.5)

    def test_2_crust_crackle_features_separate_2d_crackle_from_1d_fibers(self) -> None:
        """Verify isotropic crackle sqrt(lambda_2), DoG, and HF roughness separate 2D crackle from 1D fibers."""
        n_offs = np.arange(-12, 13, dtype=np.float64) * 2.0
        h, w = 64, 64
        uv_um = 1.5
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
        rng = np.random.default_rng(77)

        sheet = np.exp(-0.5 * (n_offs / 7.0) ** 2)[:, None, None]
        # Blank papyrus with strong 1D unidirectional fibers along x (variation only in y).
        fiber_1d = 0.012 * np.sin(2.0 * np.pi * yy / 6.0)
        blank = (0.01 + 0.055 * sheet + fiber_1d[None, :, :] * sheet + rng.normal(0.0, 0.001, (len(n_offs), h, w))).astype(
            np.float32
        )
        # Inked crust: same papyrus + 2D isotropic polygonal crackle at |n'| <= 4 um.
        surf_band = (np.abs(n_offs) <= 4.0)[:, None, None]
        crackle_2d = (
            0.015 * np.sin(2.0 * np.pi * xx / 3.5) * np.sin(2.0 * np.pi * yy / 3.5)
            + 0.010 * rng.normal(0.0, 1.0, (h, w))
        )
        inked = blank + np.where(surf_band, (0.012 + crackle_2d)[None, :, :], 0.0).astype(np.float32)
        valid = np.ones_like(blank, dtype=bool)

        fb = c1129.compute_crust_crackle_maps(blank, valid, n_offs, uv_spacing_um=uv_um)
        fa = c1129.compute_crust_crackle_maps(inked, valid, n_offs, uv_spacing_um=uv_um)

        sep_th = p4.compute_control_separation(
            fa["maps"]["tophat_recentered"][fa["valid_2d"]],
            fb["maps"]["tophat_recentered"][fb["valid_2d"]],
        )
        self.assertGreater(float(sep_th["roc_auc"]), 0.75, f"Expected tophat AUC > 0.75, got {sep_th}")
        self.assertGreater(float(sep_th["cohens_d"]), 0.9, f"Expected tophat d > 0.9, got {sep_th}")

        for key in ("hf_roughness_local", "hf_roughness_excess", "crackle_dog_rms", "crackle_isotropic"):
            sep = p4.compute_control_separation(fa["maps"][key][fa["interior_2d"]], fb["maps"][key][fb["interior_2d"]])
            self.assertGreater(float(sep["roc_auc"]), 0.95, f"Expected AUC > 0.95 for {key}, got {sep}")
            self.assertGreater(float(sep["cohens_d"]), 1.5, f"Expected Cohen's d > 1.5 for {key}, got {sep}")

    def test_3_ghosting_exclusion_flags_wrap_behind_hf_crust_and_passes_surface_crust(self) -> None:
        """Verify wrap-behind ghosting exclusion flags 110-150 um HF ghost and passes surface crust."""
        g_offs = np.array([-150.0, -110.0, 0.0, 110.0, 150.0], dtype=np.float64)
        h, w = 48, 48
        rng = np.random.default_rng(13)
        checker = ((np.arange(h)[:, None] + np.arange(w)[None, :]) % 2).astype(np.float32) * 0.03

        # Surface-localized crust at n = 0.
        slab_surf = (0.04 + rng.normal(0.0, 0.001, (len(g_offs), h, w))).astype(np.float32)
        slab_surf[2, 10:38, 10:38] += checker[10:38, 10:38]
        g_s = c1129.evaluate_crust_ghosting_exclusion(slab_surf, np.ones_like(slab_surf, dtype=bool), g_offs, uv_spacing_um=1.5)
        self.assertTrue(g_s["checked"])
        self.assertFalse(g_s["is_ghost_dominated"], f"Surface crust misflagged as ghost: {g_s}")

        # Wrap-behind ghost peaking at +150 um with bleed onto n = 0.
        slab_ghost = (0.04 + rng.normal(0.0, 0.001, (len(g_offs), h, w))).astype(np.float32)
        slab_ghost[4, 10:38, 10:38] += 1.6 * checker[10:38, 10:38]
        slab_ghost[2, 10:38, 10:38] += 0.6 * checker[10:38, 10:38]
        g_w = c1129.evaluate_crust_ghosting_exclusion(slab_ghost, np.ones_like(slab_ghost, dtype=bool), g_offs, uv_spacing_um=1.5)
        self.assertTrue(g_w["checked"])
        self.assertTrue(g_w["is_ghost_dominated"], f"Wrap-behind ghost missed: {g_w}")

    def test_4_end_to_end_gate_pass_and_gate_fail_plus_privacy_and_budget_guards(self) -> None:
        """Verify end-to-end offline preflight: ROC-AUC >= 0.70 unlocks scan, < 0.70 skips scan, zero leaks."""
        with self.assertRaises(ValueError):
            c1129.run_crust_1129_preflight_and_scan(cache_mb=300.0)
        with self.assertRaises(ValueError):
            c1129.run_crust_1129_preflight_and_scan(max_total_mb=245.0)

        # Privacy guard must block unread coordinates & local paths.
        with self.assertRaises(ValueError):
            c1129.assert_no_unread_crust_coordinates({"targeted_agraphon_scan": {"tile_bounds_vu": [1, 2, 3, 4]}})
        with self.assertRaises(ValueError):
            c1129.assert_no_unread_crust_coordinates({"targeted_agraphon_scan": {"tiles": [{"tile_id": "x"}]}})
        with self.assertRaises(ValueError):
            c1129.assert_no_unread_crust_coordinates({"bad_path": r"C:\Users\example\secret.png"})

        # Build mesh in 78 keV L0 space (2.4 um/vx): center-aligned Z_um = (z_L0 + 0.5)*2.4 = 72.0 -> z_L0 = 29.5.
        yy_c, xx_c = np.meshgrid(
            np.arange(1, 29, dtype=np.float32) * 2.0,
            np.arange(1, 29, dtype=np.float32) * 2.0,
            indexing="ij",
        )
        zz_c = np.full_like(xx_c, 72.0 / 2.4 - 0.5)

        # Case 1: Strong crackle on Control A (shifted +12.0 um along n!) -> gate PASSES and scan executes.
        u8_78, u8_1129, w78, w1129, m_1129_to_78 = _build_shifted_papyrus_crust_volumes(add_crackle_a=True, seed=201)
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            r78_dir = _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78, f32_min=w78[0], f32_max=w78[1], level="1", chunks=(32, 32, 32))
            r1129_dir = _write_synthetic_zarr_v2(td_p / "v1129.zarr", u8_1129, f32_min=w1129[0], f32_max=w1129[1], level="0", chunks=(64, 64, 64))
            mesh_dir = td_p / "mesh.tifxyz"
            ssr.write_tifxyz(mesh_dir, xx_c, yy_c, zz_c, meta={"scale": [0.5, 0.5]})
            t1129_p = td_p / "t1129.json"
            t1129_p.write_text(json.dumps({"transformation_matrix": m_1129_to_78[:3].tolist()}), encoding="utf-8")

            out_pass = td_p / "public_pass.json"
            priv_pass = td_p / "priv_pass"
            n_offs = np.arange(-20, 21, dtype=np.float64) * 2.0  # -40..+40 um
            out_offs = np.arange(-10, 11, dtype=np.float64) * 2.0  # -20..+20 um
            res_pass = c1129.run_crust_1129_preflight_and_scan(
                vol_78_url=str(r78_dir),
                vol_1129_url=str(r1129_dir),
                transform_1129_url=str(t1129_p),
                mesh_78_url=str(mesh_dir),
                ref_level_78=1,
                tgt_level_1129=0,
                uv_spacing_um=1.5,
                cache_mb=32.0,
                workers=2,
                max_total_mb=100.0,
                roc_auc_gate=0.70,
                control_pairs=(("pair_0", (2, 26, 15, 26), (2, 26, 1, 12)),),
                agraphon_tiles=(("agraph_0", (3, 25, 2, 11)),),
                n_offsets_um=n_offs,
                out_n_offsets_um=out_offs,
                ghost_offsets_um=None,
                out_json=out_pass,
                private_out_dir=priv_pass,
            )
            c1129.assert_no_unread_crust_coordinates(json.loads(out_pass.read_text(encoding="utf-8")))
            self.assertTrue(res_pass["ncc_passed_all_controls"])
            self.assertTrue(res_pass["control_gate_passed"], f"Expected gate pass: {res_pass['pooled_control_a_vs_b']}")
            self.assertGreaterEqual(res_pass["best_crust_roc_auc"], 0.70)
            self.assertTrue(res_pass["targeted_agraphon_scan"]["executed"])
            self.assertEqual(res_pass["targeted_agraphon_scan"]["tiles_evaluated"], 1)
            self.assertTrue((priv_pass / "crust_1129_private_summary.json").is_file())
            self.assertTrue((priv_pass / "scan_agraph_0_zmap.png").is_file())

            # Verify Control A's +12.0 um normal offset was actually recentered!
            ctrl_a_rec = res_pass["control_pairs"][0]["control_a"]["recentering"]
            self.assertAlmostEqual(float(ctrl_a_rec["n_peak_median_um"]), 12.0, delta=1.5)

        # Case 2: No crackle on Control A (identical blank papyrus shifted +12.0 um along n) -> gate FAILS & skips scan!
        u8_78_n, u8_1129_n, w78_n, w1129_n, m_1129_to_78_n = _build_shifted_papyrus_crust_volumes(
            add_crackle_a=False, seed=202
        )
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            r78_dir = _write_synthetic_zarr_v2(td_p / "v78.zarr", u8_78_n, f32_min=w78_n[0], f32_max=w78_n[1], level="1", chunks=(32, 32, 32))
            r1129_dir = _write_synthetic_zarr_v2(td_p / "v1129.zarr", u8_1129_n, f32_min=w1129_n[0], f32_max=w1129_n[1], level="0", chunks=(64, 64, 64))
            mesh_dir = td_p / "mesh.tifxyz"
            ssr.write_tifxyz(mesh_dir, xx_c, yy_c, zz_c, meta={"scale": [0.5, 0.5]})
            t1129_p = td_p / "t1129.json"
            t1129_p.write_text(json.dumps({"transformation_matrix": m_1129_to_78_n[:3].tolist()}), encoding="utf-8")

            out_fail = td_p / "public_fail.json"
            res_fail = c1129.run_crust_1129_preflight_and_scan(
                vol_78_url=str(r78_dir),
                vol_1129_url=str(r1129_dir),
                transform_1129_url=str(t1129_p),
                mesh_78_url=str(mesh_dir),
                ref_level_78=1,
                tgt_level_1129=0,
                uv_spacing_um=1.5,
                cache_mb=32.0,
                workers=2,
                max_total_mb=100.0,
                roc_auc_gate=0.70,
                control_pairs=(("pair_0", (2, 26, 15, 26), (2, 26, 1, 12)),),
                agraphon_tiles=(("agraph_0", (3, 25, 2, 11)),),
                n_offsets_um=np.arange(-20, 21, dtype=np.float64) * 2.0,
                out_n_offsets_um=np.arange(-10, 11, dtype=np.float64) * 2.0,
                ghost_offsets_um=None,
                out_json=out_fail,
            )
            c1129.assert_no_unread_crust_coordinates(json.loads(out_fail.read_text(encoding="utf-8")))
            self.assertFalse(res_fail["control_gate_passed"])
            self.assertFalse(res_fail["targeted_agraphon_scan"]["executed"])
            self.assertEqual(res_fail["targeted_agraphon_scan"]["tiles_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
