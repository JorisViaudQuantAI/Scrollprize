# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline unit tests for Exp T7 (transverse_cross_scroll_replication.py).

Strictly CPU-only, zero network I/O, zero torch imports, synthetic data only.
Covers:
1. Pure CPU / zero-torch import compliance (AST + isolated subprocess).
2. Cryo-ET / CryoSamba-inspired `cryo_slice_coherence` exact identities (+1 identical,
   -1 negated, ~0 independent noise), strict affine invariance, and scalar vs 2D center_idx.
3. Exact numerical parity of `_compute_oct_hilbert_env_peak` with Exp T5.
4. Unsupervised equal-weight rank-average fusion (`compute_rank_average_fusion`) polarity
   handling and monotonic invariance.
5. Adversarial Off-Sheet Kill-Test (`claude-016` safeguard):
   - True on-sheet ink passes both +/-8 and boundary kill-tests -> PASS.
   - Depth-invariant 2D geometry leakage is strictly caught & killed -> CLOSED NO-GO.
   - Flipped polarity fails `fixed_t5_polarity` gate even when unsigned AUC is high -> CLOSED NO-GO.
6. Multi-volume HTTP byte budget (`<= 40 MiB/vol`, `<= 120 MiB` global).
7. Zarr v3 sharded human-label loader & deterministic 1:1 / 2x / 4x label alignment.
8. Offline end-to-end 3-volume pipeline + privacy guardrail + committed live receipt check.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transverse_cross_scroll_replication as t7  # noqa: E402
import transverse_wavelet_oct_physics as t5  # noqa: E402


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


def _write_local_zarr3_sharded_u8(
    root: Path,
    arr: np.ndarray,
    *,
    inner_chunks: tuple[int, int] = (32, 32),
) -> None:
    """Write a minimal valid uncompressed Zarr v3 sharded 2D uint8 array for offline tests."""
    h, w = arr.shape
    ih, iw = inner_chunks
    ny = (h + ih - 1) // ih
    nx = (w + iw - 1) // iw
    shard_h, shard_w = ny * ih, nx * iw

    level = root / "0"
    shard_dir = level / "c" / "0"
    shard_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [int(h), int(w)],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [int(shard_h), int(shard_w)]},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [int(ih), int(iw)],
                    "codecs": [{"name": "bytes"}],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ],
    }
    (level / "zarr.json").write_text(json.dumps(meta), encoding="utf-8")

    payload = bytearray()
    empty = (1 << 64) - 1
    index = np.full((ny, nx, 2), empty, dtype="<u8")
    for iy in range(ny):
        for ix in range(nx):
            tile = np.zeros((ih, iw), dtype=np.uint8)
            y0, y1 = iy * ih, min(h, (iy + 1) * ih)
            x0, x1 = ix * iw, min(w, (ix + 1) * iw)
            tile[: y1 - y0, : x1 - x0] = arr[y0:y1, x0:x1]
            if tile.any():
                raw = tile.tobytes(order="C")
                index[iy, ix, 0] = len(payload)
                index[iy, ix, 1] = len(raw)
                payload.extend(raw)
    payload.extend(index.tobytes(order="C"))
    payload.extend(struct.pack("<I", 0))  # 4-byte crc32c trailer
    (shard_dir / "0").write_bytes(bytes(payload))


class TestCpuOnlyCompliance(unittest.TestCase):
    def test_no_torch_imported(self) -> None:
        src_path = Path(__file__).resolve().parent / "transverse_cross_scroll_replication.py"
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

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        code = (
            f"import sys; sys.path.insert(0, {str(src_path.parent)!r}); "
            "import transverse_cross_scroll_replication; "
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


class TestCryoSliceCoherence(unittest.TestCase):
    def test_identities_and_affine_invariance(self) -> None:
        rng = np.random.default_rng(42)
        h, w = 64, 64
        vm = np.ones((h, w), dtype=bool)
        base = rng.normal(100.0, 15.0, size=(h, w)).astype(np.float32)

        # Identical slices -> +1.0 everywhere
        stack_same = np.stack([base, base, base, base, base], axis=0)
        c_same = t7.cryo_slice_coherence(stack_same, vm, center_idx=2, patch_sigma_px=2.5)
        np.testing.assert_allclose(c_same, 1.0, atol=1e-5)

        # Exact affine transform per slice (independent positive gain + offset) -> still +1.0!
        stack_aff = np.stack(
            [base, 2.5 * base + 40.0, 0.4 * base - 12.0, 5.0 * base + 99.0, base], axis=0
        )
        c_aff = t7.cryo_slice_coherence(stack_aff, vm, center_idx=2, patch_sigma_px=2.5)
        np.testing.assert_allclose(c_aff, 1.0, atol=1e-5)

        # Alternating sign slices -> -1.0 everywhere
        stack_anti = np.stack([base, -base, base, -base, base], axis=0)
        c_anti = t7.cryo_slice_coherence(stack_anti, vm, center_idx=2, patch_sigma_px=2.5)
        np.testing.assert_allclose(c_anti, -1.0, atol=1e-5)

        # Independent white noise across slices -> mean near 0.0
        stack_noise = rng.normal(100.0, 20.0, size=(5, h, w)).astype(np.float32)
        c_noise = t7.cryo_slice_coherence(stack_noise, vm, center_idx=2, patch_sigma_px=3.0)
        self.assertLess(abs(float(c_noise.mean())), 0.08)

        # Scalar center_idx matches constant 2D center_idx identically
        c_2d = t7.cryo_slice_coherence(
            stack_noise, vm, center_idx=np.full((h, w), 2.0, dtype=np.float32), patch_sigma_px=3.0
        )
        np.testing.assert_allclose(c_noise, c_2d, atol=1e-6)


class TestOctParityAndRankFusion(unittest.TestCase):
    def test_oct_hilbert_env_peak_matches_exp_t5_exactly(self) -> None:
        rng = np.random.default_rng(13)
        depth, h, w = 31, 48, 48
        zz = np.arange(depth, dtype=np.float64)[:, None, None]
        slab = (160.0 * np.exp(-0.5 * ((zz - 15.0) / 3.0) ** 2) + rng.normal(0.0, 5.0, (depth, h, w))).astype(
            np.float32
        )
        n_peak = np.full((h, w), 15.0, dtype=np.float32)
        vm = np.ones((h, w), dtype=bool)

        f_t5, _ = t5.compute_oct_depth_waveform_features(
            slab, n_peak, vm, voxel_um=8.64, half_span_vx=7.2, waveform_dim=31
        )
        f_t7, _ = t7._compute_oct_hilbert_env_peak(
            slab, n_peak, vm, half_span_vx=7.2, waveform_dim=31
        )
        np.testing.assert_allclose(f_t7, f_t5["oct_hilbert_env_peak"], atol=1e-6)

    def test_rank_fusion_respects_fixed_polarities_and_monotonic_invariance(self) -> None:
        rng = np.random.default_rng(77)
        h, w = 64, 64
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ink = (yy >= 16) & (yy < 48) & (xx >= 16) & (xx < 48)
        vm = np.ones((h, w), dtype=bool)

        # Build 3 noisy features matching high_res_2p4um polarities: (+, -, -)
        f_oct = np.where(ink, 1.2, 0.0) + rng.normal(0.0, 1.0, (h, w))
        f_wst = np.where(ink, -1.2, 0.0) + rng.normal(0.0, 1.0, (h, w))
        f_pc = np.where(ink, -1.2, 0.0) + rng.normal(0.0, 1.0, (h, w))
        fmap = {
            "oct_hilbert_env_peak": f_oct.astype(np.float32),
            "wst_s1_isotropic": f_wst.astype(np.float32),
            "pc_min_moment_corners_pooled": f_pc.astype(np.float32),
        }
        pols = t7.FIXED_T5_POLARITIES["high_res_2p4um"]
        fused = t7.compute_rank_average_fusion(fmap, vm, pols)

        # Strict invariance under positive monotonic transforms (exp, affine, sinh) in float64
        fmap_mono = {
            "oct_hilbert_env_peak": np.exp(0.3 * fmap["oct_hilbert_env_peak"].astype(np.float64)),
            "wst_s1_isotropic": 3.0 * fmap["wst_s1_isotropic"].astype(np.float64) + 10.0,
            "pc_min_moment_corners_pooled": np.sinh(0.5 * fmap["pc_min_moment_corners_pooled"].astype(np.float64)),
        }
        fused_mono = t7.compute_rank_average_fusion(fmap_mono, vm, pols)
        np.testing.assert_allclose(fused, fused_mono, atol=1e-5)

        # Fusion strictly beats every individual noisy constituent
        auc_fused = t7.exact_roc_auc(fused.ravel(), ink.ravel())
        auc_oct = t7.exact_roc_auc(f_oct.ravel(), ink.ravel())
        auc_wst = t7.exact_roc_auc((-f_wst).ravel(), ink.ravel())
        auc_pc = t7.exact_roc_auc((-f_pc).ravel(), ink.ravel())
        self.assertGreater(auc_fused, max(auc_oct, auc_wst, auc_pc) + 0.03)


class TestAdversarialOffSheetKillTest(unittest.TestCase):
    def _make_base_masks(self, h: int = 96, w: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ink = ((yy >= 24) & (yy < 44) & (xx >= 20) & (xx < 76)) | (
            (yy >= 54) & (yy < 72) & (xx >= 28) & (xx < 68)
        )
        blank = ~ink
        vm = np.ones((h, w), dtype=bool)
        return ink, blank, vm

    def test_true_onsheet_ink_passes_killtest(self) -> None:
        rng = np.random.default_rng(101)
        depth, h, w = 31, 96, 96
        ink, blank, vm = self._make_base_masks(h, w)
        zz = np.arange(depth, dtype=np.float64)[:, None, None]

        # Papyrus sheet at z = 15 with localized ink attenuation boost ONLY on-sheet (z = 15 +/- 2).
        # Off-sheet (z <= 7 or z >= 23) is pure i.i.d. noise -> off-sheet AUC ~ 0.50!
        sheet = 140.0 * np.exp(-0.5 * ((zz - 15.0) / 2.0) ** 2)
        ink_boost = np.where(ink[None], 55.0 * np.exp(-0.5 * ((zz - 15.0) / 1.5) ** 2), 0.0)
        slab = np.clip(sheet + ink_boost + rng.normal(20.0, 5.0, (depth, h, w)), 1.0, 255.0).astype(
            np.float32
        )

        res = t7.evaluate_volume_on_and_off_sheet(
            slab, vm, ink, blank, voxel_um=2.403, regime="high_res_2p4um", border_guard_px=8
        )
        oct_r = res["features"]["oct_hilbert_env_peak"]
        self.assertGreater(oct_r["on_sheet_auc_fixed"], 0.90)
        self.assertGreater(oct_r["delta_null_fixed"], 0.20)
        self.assertGreater(oct_r["drop_vs_pm8_fixed"], 0.30)
        self.assertGreater(oct_r["drop_vs_boundary_fixed"], 0.30)
        self.assertTrue(oct_r["passes_gate_strict_both"])
        self.assertEqual(res["summary"]["verdict"], "PASS")

    def test_geometry_leakage_killed_by_offsheet_test(self) -> None:
        rng = np.random.default_rng(202)
        depth, h, w = 31, 96, 96
        ink, blank, vm = self._make_base_masks(h, w)
        zz = np.arange(depth, dtype=np.float64)[:, None, None]

        # Adversarial depth-invariant 2D texture leak across ALL slices (same pattern on-sheet & off-sheet).
        # Must fail drop_vs_pm8 and drop_vs_boundary -> CLOSED NO-GO!
        sheet = 120.0 * np.exp(-0.5 * ((zz - 15.0) / 2.2) ** 2)
        leak_2d = np.where(ink, 0.0, rng.normal(0.0, 22.0, (h, w)))
        slab = np.clip(sheet + leak_2d[None] + 40.0 + rng.normal(0.0, 1.5, (depth, h, w)), 1.0, 255.0).astype(
            np.float32
        )

        res = t7.evaluate_volume_on_and_off_sheet(
            slab, vm, ink, blank, voxel_um=8.64, regime="faint_8_9um", border_guard_px=8
        )
        wst_r = res["features"]["wst_s1_isotropic"]
        # High on-sheet AND high off-sheet -> drop near 0 -> strictly killed!
        self.assertGreater(wst_r["on_sheet_auc_fixed"], 0.85)
        self.assertGreater(wst_r["off_sheet_pm8_mean_auc_fixed"], 0.80)
        self.assertLess(wst_r["drop_vs_pm8_fixed"], 0.08)
        self.assertFalse(wst_r["passes_gate_pm8"])
        self.assertFalse(wst_r["passes_gate_boundary"])

    def test_flipped_polarity_cannot_claim_pass(self) -> None:
        rng = np.random.default_rng(303)
        depth, h, w = 31, 96, 96
        ink, blank, vm = self._make_base_masks(h, w)
        zz = np.arange(depth, dtype=np.float64)[:, None, None]

        # Ink has LOWER peak amplitude at 2.4 um (OPPOSITE of Exp T5 high_res_2p4um polarity '+'!).
        sheet = 180.0 * np.exp(-0.5 * ((zz - 15.0) / 2.0) ** 2)
        ink_drop = np.where(ink[None], -60.0 * np.exp(-0.5 * ((zz - 15.0) / 1.5) ** 2), 0.0)
        slab = np.clip(sheet + ink_drop + rng.normal(25.0, 4.0, (depth, h, w)), 1.0, 255.0).astype(
            np.float32
        )

        res = t7.evaluate_volume_on_and_off_sheet(
            slab, vm, ink, blank, voxel_um=2.403, regime="high_res_2p4um", border_guard_px=8
        )
        oct_r = res["features"]["oct_hilbert_env_peak"]
        # Unsigned diagnostic sees strong inverted signal, but fixed_t5_polarity is < 0.2 -> FAIL!
        self.assertGreater(oct_r["unsigned_best_polarity_diagnostic"]["on_sheet_unsigned_auc"], 0.90)
        self.assertLess(oct_r["on_sheet_auc_fixed"], 0.20)
        self.assertFalse(oct_r["polarity_matches_t5"])
        self.assertFalse(oct_r["passes_gate_pm8"])
        self.assertFalse(oct_r["passes_gate_boundary"])


class TestBudgetZarr3AlignmentAndEndToEnd(unittest.TestCase):
    def test_multi_volume_budget_enforcement(self) -> None:
        b = t7.MultiVolumeHttpByteBudget(
            {"v1": 1000, "v2": 1000, "v3": 1000}, global_limit_bytes=2200
        )
        b.charge("v1", 900, label="v1:a")
        with self.assertRaises(t7.BudgetExceededError):
            b.check_planned("v1", 200, label="v1:over")
        b.charge("v2", 900, label="v2:a")
        with self.assertRaises(t7.BudgetExceededError):
            b.check_planned("v3", 500, label="v3:global_over")
        s = b.summary()
        self.assertTrue(s["within_limits"])
        self.assertEqual(s["global_bytes"], 1800)

    def test_align_label_map_and_zarr3_sharded_loader(self) -> None:
        arr = np.zeros((32, 48), dtype=np.uint8)
        arr[8:20, 12:36] = 255
        up, info_up = t7.align_label_map_to_sv(arr, (64, 96), is_binary_mask=True)
        self.assertEqual(up.shape, (64, 96))
        self.assertEqual(info_up["mode"], "exact_integer_upsample_2x")
        dn, info_dn = t7.align_label_map_to_sv(up, (32, 48), is_binary_mask=True)
        np.testing.assert_array_equal(dn, arr)
        self.assertEqual(info_dn["mode"], "exact_integer_downsample_2x")
        with self.assertRaises(ValueError):
            t7.align_label_map_to_sv(arr, (50, 70))

        with tempfile.TemporaryDirectory() as tmp:
            z3_dir = Path(tmp) / "inklabels.zarr"
            _write_local_zarr3_sharded_u8(z3_dir, arr, inner_chunks=(16, 16))
            loaded, meta = t7.load_zarr3_sharded_2d_u8(z3_dir)
            np.testing.assert_array_equal(loaded, arr)
            self.assertEqual(meta["shape"], [32, 48])

    def test_offline_end_to_end_3_volumes_and_privacy_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2027)
            depth, h, w = 29, 96, 96
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            ink_gt = ((yy >= 24) & (yy < 44) & (xx >= 20) & (xx < 76)) | (
                (yy >= 54) & (yy < 72) & (xx >= 28) & (xx < 68)
            )
            zz = np.arange(depth, dtype=np.float64)[:, None, None]

            # Volume 1 (high_res_2p4um) with local Zarr v3 human labels + true on-sheet ink peak boost
            v1 = np.clip(
                140.0 * np.exp(-0.5 * ((zz - 14.0) / 2.0) ** 2)
                + np.where(ink_gt[None], 55.0 * np.exp(-0.5 * ((zz - 14.0) / 1.4) ** 2), 0.0)
                + rng.normal(20.0, 4.0, (depth, h, w)),
                1.0,
                255.0,
            ).astype(np.uint8)
            z1 = root / "v1.zarr"
            _write_local_zarr_v2(z1, v1, chunks=(depth, 32, 32))
            ink_z3 = root / "inklabels.zarr"
            sup_z3 = root / "supervision.zarr"
            _write_local_zarr3_sharded_u8(ink_z3, np.where(ink_gt, 255, 0).astype(np.uint8))
            _write_local_zarr3_sharded_u8(sup_z3, np.full((h, w), 255, dtype=np.uint8))

            # Volume 2 (faint_8_9um) with TIFF model-derived reference + true on-sheet negative peak drop
            v2 = np.clip(
                170.0 * np.exp(-0.5 * ((zz - 14.0) / 2.0) ** 2)
                + np.where(ink_gt[None], -55.0 * np.exp(-0.5 * ((zz - 14.0) / 1.4) ** 2), 0.0)
                + rng.normal(25.0, 4.0, (depth, h, w)),
                1.0,
                255.0,
            ).astype(np.uint8)
            z2 = root / "v2.zarr"
            _write_local_zarr_v2(z2, v2, chunks=(depth, 32, 32))
            tif2 = root / "ink2.tif"
            tifffile.imwrite(str(tif2), np.where(ink_gt, 230, 30).astype(np.uint8))

            # Volume 3 (faint_8_9um) pure noise null -> CLOSED NO-GO
            v3 = np.clip(
                150.0 * np.exp(-0.5 * ((zz - 14.0) / 2.2) ** 2) + rng.normal(20.0, 8.0, (depth, h, w)),
                1.0,
                255.0,
            ).astype(np.uint8)
            z3 = root / "v3.zarr"
            _write_local_zarr_v2(z3, v3, chunks=(depth, 32, 32))
            tif3 = root / "ink3.tif"
            tifffile.imwrite(str(tif3), np.where(ink_gt, 230, 30).astype(np.uint8))

            specs = {
                "pherc0841_2p403um": {
                    "scroll": "PHerc0841",
                    "segment": "synth_w00",
                    "voxel_um": 2.403,
                    "regime": "high_res_2p4um",
                    "sv_zarr_url": str(z1),
                    "human_inklabels_zarr": str(ink_z3),
                    "human_supervision_zarr": str(sup_z3),
                    "border_guard_px": 8,
                    "z_smooth_sigma": 0.8,
                },
                "pherc0841_9p366um": {
                    "scroll": "PHerc0841",
                    "segment": "synth_w00",
                    "voxel_um": 9.366,
                    "regime": "faint_8_9um",
                    "sv_zarr_url": str(z2),
                    "fallback_ink_tif_url": str(tif2),
                    "fallback_provenance": "model_derived_argus_v3_synth",
                    "border_guard_px": 8,
                    "z_smooth_sigma": 0.8,
                },
                "pherc0343_8p64um": {
                    "scroll": "PHerc0343",
                    "segment": "synth_w047_w048",
                    "voxel_um": 8.64,
                    "regime": "faint_8_9um",
                    "sv_zarr_url": str(z3),
                    "fallback_ink_tif_url": str(tif3),
                    "fallback_provenance": "model_derived_argus_v3_synth",
                    "border_guard_px": 8,
                    "z_smooth_sigma": 0.8,
                },
            }

            receipt = t7.run_cross_scroll_replication(
                volume_specs=specs,
                roi_h=96,
                roi_w=96,
                off_sheet_shift_slices=8,
            )
            t7.assert_no_local_paths(receipt)
            self.assertEqual(receipt["compute_mode"], "CPU_ONLY")
            self.assertTrue(receipt["http_byte_budget"]["within_limits"])
            self.assertEqual(
                receipt["volumes"]["pherc0841_2p403um"]["reference_label_provenance"]["label_type"],
                "human_ink_labels_zarr_v3",
            )
            self.assertFalse(
                receipt["volumes"]["pherc0841_2p403um"]["reference_label_provenance"]["is_model_derived"]
            )
            self.assertTrue(
                receipt["volumes"]["pherc0343_8p64um"]["reference_label_provenance"]["is_model_derived"]
            )
            self.assertEqual(receipt["verdict"]["per_volume"]["pherc0841_2p403um"], "PASS")
            self.assertEqual(receipt["verdict"]["per_volume"]["pherc0841_9p366um"], "PASS")
            self.assertEqual(receipt["verdict"]["per_volume"]["pherc0343_8p64um"], "CLOSED NO-GO")
            self.assertEqual(receipt["verdict"]["overall"], "PARTIAL")

    def test_committed_live_receipt_if_present(self) -> None:
        receipt_path = (
            Path(__file__).resolve().parent
            / "evidence"
            / "transverse_cross_scroll_replication.json"
        )
        if not receipt_path.is_file():
            self.skipTest("Live Exp T7 evidence receipt not yet generated")
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        t7.assert_no_local_paths(data)
        self.assertEqual(data["compute_mode"], "CPU_ONLY")
        b = data["http_byte_budget"]
        self.assertTrue(b["within_limits"])
        self.assertLessEqual(b["global_bytes"], 120 * 1024 * 1024)
        for vkey in t7.VOLUME_ORDER:
            self.assertIn(vkey, b["per_volume"])
            self.assertLessEqual(b["per_volume"][vkey]["bytes"], 40 * 1024 * 1024)
            self.assertIn(vkey, data["volumes"])
            vol = data["volumes"][vkey]
            self.assertEqual(set(vol["features"].keys()), set(t7.ALL_FEATURE_NAMES))
            # Verify honest verdict consistency from fixed_t5_polarity gated features only
            p_both = [
                k for k in t7.GATED_TRANSVERSE_FEATURES if vol["features"][k]["passes_gate_strict_both"]
            ]
            p_pm8 = [
                k for k in t7.GATED_TRANSVERSE_FEATURES if vol["features"][k]["passes_gate_pm8"]
            ]
            p_bnd = [
                k for k in t7.GATED_TRANSVERSE_FEATURES if vol["features"][k]["passes_gate_boundary"]
            ]
            expected_v = "PASS" if p_both else ("PARTIAL" if (p_pm8 or p_bnd) else "CLOSED NO-GO")
            self.assertEqual(vol["summary"]["verdict"], expected_v)
            self.assertEqual(data["verdict"]["per_volume"][vkey], expected_v)


if __name__ == "__main__":
    unittest.main()
