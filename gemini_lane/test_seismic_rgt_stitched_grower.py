# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for ``seismic_rgt_stitched_grower`` (Exp T11-RGT-Scale, Q46)."""
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
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import seismic_rgt_stitched_grower as SRG  # noqa: E402
import seismic_rgt_unwrapper as RGT  # noqa: E402
import stream_snap_render as SSR  # noqa: E402

MODULE_PATH = HERE / "seismic_rgt_stitched_grower.py"
EVIDENCE_PATH = HERE / "evidence" / "seismic_rgt_stitched_patches.json"
GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba.cuda", "pycuda")


def make_synthetic_stitched_slab(
    shape=(48, 160, 160),
    n_wraps=5,
    spacing=6.0,
    sigma=1.0,
    noise=3.5,
    seed=42,
    *,
    fold=2.8,
    pinch=1.4,
):
    """Synthetic ``48x160x160`` 5-wrap folded/pinched CT + m7 slab with a delamination gap on wrap 1."""
    rng = np.random.default_rng(seed)
    d, a, b = shape
    aa, bb = np.meshgrid(np.arange(a, dtype=float), np.arange(b, dtype=float), indexing="ij")
    fold_f = fold * np.sin(2.0 * np.pi * aa / a) * np.cos(np.pi * bb / b)
    local_sp = spacing + 0.8 * np.sin(2.0 * np.pi * bb / b)
    pinch_f = pinch * np.exp(-((aa - 0.62 * a) ** 2 + (bb - 0.38 * b) ** 2) / (2.0 * 12.0 ** 2))
    c = (n_wraps - 1) / 2.0
    heights = []
    for m in range(n_wraps):
        h = d / 2.0 + fold_f + (m - c) * local_sp
        if m == 2:
            h = h + pinch_f
        heights.append(h)

    gap_mask = ((aa - 0.28 * a) ** 2 + (bb - 0.72 * b) ** 2) < (11.0 ** 2)
    z = np.arange(d, dtype=float)[:, None, None]
    vol = np.zeros(shape, dtype=np.float64)
    clean = np.zeros(shape, dtype=np.float64)
    for m, h in enumerate(heights):
        texture = 1.0 + 0.35 * ndi.gaussian_filter(rng.normal(0.0, 1.0, (a, b)), 1.3) / 0.25
        mask_factor = np.where(gap_mask, 0.0, 1.0) if m == 1 else 1.0
        g = np.exp(-((z - h[None]) ** 2) / (2.0 * sigma ** 2)) * mask_factor
        vol += np.clip(texture, 0.4, 1.6) * g
        clean += g

    ct = np.clip(40.0 + 150.0 * vol + rng.normal(0.0, noise, shape), 1, 255).astype(np.uint8)
    # Continuous m7 surface response peaked at wrap centers; wrap 1 is pinched near wrap 2 so
    # 3x3x3 binary connectivity bridges across wraps and challenges single-sheet column tracking.
    m7 = np.clip(255.0 * np.exp(-0.5 * ((clean - 1.0) / 0.45) ** 2) * (clean > 0.22), 0, 255).astype(np.uint8)
    # Add local inter-wrap bridge threads in the binary m7 volume (realistic crushed-scroll m7 bridges)
    bridge = ((np.abs(aa - 0.5 * a) < 18) & (np.abs(bb - 0.5 * b) < 18))
    z_bridge = (z >= heights[1][None] - 0.5) & (z <= heights[2][None] + 0.5) & bridge[None]
    m7 = np.where(z_bridge, np.maximum(m7, np.uint8(200)), m7)
    return ct, m7, np.stack(heights), gap_mask


def match_sheets_to_truth(sheets, true_h):
    """Return ``(extracted_idx, true_idx, rmse_vx)`` for each extracted sheet."""
    out = []
    for i, s in enumerate(sheets):
        v = s["valid"]
        dists = [float(np.median(np.abs(s["height"][v] - t[v]))) for t in true_h]
        m = int(np.argmin(dists))
        err = s["height"][v] - true_h[m][v]
        out.append((i, m, float(np.sqrt(np.mean(err ** 2)))))
    return out


_SYNTHETIC_FIXTURE: dict[str, object] = {}


def _get_synthetic_stitched_fixture() -> dict[str, object]:
    """Build and cache the synthetic stitched slab fixture once per test process."""
    if not _SYNTHETIC_FIXTURE:
        ct, m7, true_h, gap_mask = make_synthetic_stitched_slab()
        stitched = SRG.stitch_overlapping_rgt_tiles_3d(
            ct, m7, tile_lateral=96, overlap_lateral=32, min_spacing_vx=4.0, max_spacing_vx=18.0
        )
        extracted = SRG.extract_and_snap_stitched_rgt_sheets(
            stitched["phi"],
            ct,
            m7,
            normal_axis=stitched["normal_axis"],
            voxel_um=9.362,
            snap_passes=2,
            num_slices=9,
        )
        _SYNTHETIC_FIXTURE.update(
            {
                "ct": ct,
                "m7": m7,
                "true_h": true_h,
                "gap_mask": gap_mask,
                "stitched": stitched,
                "extracted": extracted,
            }
        )
    return _SYNTHETIC_FIXTURE


class StitchedSyntheticSlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fx = _get_synthetic_stitched_fixture()
        cls.ct = fx["ct"]
        cls.m7 = fx["m7"]
        cls.true_h = fx["true_h"]
        cls.gap_mask = fx["gap_mask"]
        cls.stitched = fx["stitched"]
        cls.extracted = fx["extracted"]

    def test_2x2_tile_stitching_monotone_and_seam_continuity(self):
        st = self.stitched
        self.assertEqual(st["normal_axis"], 0)
        self.assertEqual(st["tile_grid"], [2, 2])
        self.assertEqual(st["tile_lateral_vx"], [96, 96])
        self.assertEqual(st["overlap_lateral_vx"], [32, 32])
        self.assertEqual(st["phi"].shape, self.ct.shape)
        self.assertTrue(np.all(np.diff(st["phi"], axis=0) > 0), "stitched RGT phi must be strictly increasing")
        np.testing.assert_allclose(st["phasor_cos"] ** 2 + st["phasor_sin"] ** 2, 1.0, atol=1e-4)
        self.assertLess(st["seam_phase_rmse_rad"], 0.25)
        self.assertGreater(st["phase_normal_coherence"], 0.85)
        for key in ("0_0", "0_1", "1_0", "1_1"):
            g = st["tile_gauges"][key]
            self.assertGreaterEqual(g["a"], 0.8)
            self.assertLessEqual(g["a"], 1.25)

    def test_stitched_multi_sheet_recovery_and_area_gain_vs_baseline(self):
        m = self.extracted["metrics"]
        self.assertGreaterEqual(m["sheet_count_rgt"], 4)
        self.assertEqual(m["crossing_fraction_rgt"], 0.0)
        self.assertGreater(m["usable_cm2_single_sheet_baseline"], 0.0)
        self.assertGreaterEqual(m["area_gain_ratio_vs_baseline"], 3.0)
        self.assertGreater(m["usable_cm2_rgt_total"], m["usable_cm2_best_sheet_rgt"])
        self.assertIsNotNone(m["median_f_parallel_rgt"])
        self.assertIsNotNone(m["max_f_parallel_rgt"])
        self.assertTrue(set(m) <= SRG.PUBLIC_STITCHED_RGT_METRIC_KEYS)

        sheets = self.extracted["sheets"]
        matches = match_sheets_to_truth(sheets, self.true_h)
        self.assertEqual(len({t for _, t, _ in matches}), len(matches), "duplicate sheets on same wrap")
        for _, _, rmse in matches:
            self.assertLess(rmse, 0.75)

    def test_order_preserving_m7_snap_refinement(self):
        m = self.extracted["metrics"]
        sheets = self.extracted["sheets"]
        self.assertEqual(m["snap_passes"], 2)
        self.assertEqual(m["crossing_fraction_rgt"], 0.0)
        self.assertIsNotNone(m["mean_on_sheet_after_snap_rgt"])
        self.assertGreaterEqual(m["mean_on_sheet_after_snap_rgt"], m["mean_on_sheet_raw_rgt"] - 1e-6)
        self.assertGreater(m["mean_on_sheet_after_snap_rgt"], 0.80)
        for lo, hi in zip(sheets[:-1], sheets[1:]):
            jv = lo["valid"] & hi["valid"]
            self.assertTrue(jv.any())
            self.assertTrue(np.all(hi["height"][jv] > lo["height"][jv] + 0.1))

    def test_direct_order_clamp_helper_prevents_sheet_crossings(self):
        h0 = np.full((16, 16), 10.0)
        h1 = np.full((16, 16), 9.5)  # inverted relative to h0
        h2 = np.full((16, 16), 9.6)  # inverted relative to h0/h1
        v = [np.ones((16, 16), dtype=bool) for _ in range(3)]
        clamped = SRG._enforce_strict_sheet_order([h0, h1, h2], v, min_gap_vx=0.12, depth_max=31.0)
        self.assertTrue(np.all(clamped[1] > clamped[0] + 0.1))
        self.assertTrue(np.all(clamped[2] > clamped[1] + 0.1))


class PrivateTifxyzExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fx = _get_synthetic_stitched_fixture()
        cls.stitched = fx["stitched"]
        cls.extracted = fx["extracted"]

    def test_export_private_rgt_tifxyz_meshes_round_trip(self):
        st = self.stitched
        ext = self.extracted
        origin_zyx = (120, 240, 360)
        with tempfile.TemporaryDirectory(prefix="rgt_export_test_") as tmp:
            out_dir = Path(tmp) / "private_meshes"
            written = SRG.export_private_rgt_tifxyz_meshes(
                ext["sheets"][:2], origin_zyx, out_dir, "PHerc1545", scale=(1.0, 1.0)
            )
            self.assertEqual(len(written), 2)
            for idx, mesh_dir in enumerate(written):
                self.assertTrue((mesh_dir / "x.tif").is_file())
                self.assertTrue((mesh_dir / "y.tif").is_file())
                self.assertTrue((mesh_dir / "z.tif").is_file())
                self.assertTrue((mesh_dir / "meta.json").is_file())
                gx, gy, gz, gvalid, meta = SSR.read_tifxyz(mesh_dir)
                src = ext["sheets"][idx]
                np.testing.assert_array_equal(gvalid, src["valid"])
                # Verify box_origin_zyx[::-1] = (x0=360, y0=240, z0=120) translation on valid vertices.
                np.testing.assert_allclose(gx[gvalid], src["x"][gvalid] + 360.0, atol=1e-4)
                np.testing.assert_allclose(gy[gvalid], src["y"][gvalid] + 240.0, atol=1e-4)
                np.testing.assert_allclose(gz[gvalid], src["z"][gvalid] + 120.0, atol=1e-4)
                # Verify -1.0 sentinels on invalid vertices.
                if (~gvalid).any():
                    self.assertTrue(np.all(gx[~gvalid] == -1.0))
                    self.assertTrue(np.all(gy[~gvalid] == -1.0))
                    self.assertTrue(np.all(gz[~gvalid] == -1.0))
                self.assertEqual(meta["format"], "tifxyz")
                self.assertEqual(meta["scroll"], "PHerc1545")
                self.assertEqual(meta["scale"], [1.0, 1.0])
        self.assertIsNotNone(st)

    def test_export_private_rgt_tifxyz_rejects_in_repo_path(self):
        in_repo_dir = HERE / "forbidden_in_repo_private_out"
        with self.assertRaises(ValueError):
            SRG.export_private_rgt_tifxyz_meshes(
                self.extracted["sheets"][:1],
                (0, 0, 0),
                in_repo_dir,
                "PHerc1545",
            )
        self.assertFalse(in_repo_dir.exists())


class BudgetAndComplianceTests(unittest.TestCase):
    def test_byte_budget_guard_and_four_chunk_alignment(self):
        b = SRG.ByteBudget(SRG.GLOBAL_HTTP_CAP_BYTES)
        self.assertEqual(b.cap, 125_829_120)
        b.charge(120 * 1024 * 1024 - 8)
        with self.assertRaises(SRG.BudgetExceeded):
            b.precheck(9)
        with self.assertRaises(SRG.BudgetExceeded):
            b.charge(9)

        lo = SRG._align_slab_to_four_chunks((505, 710, 845), (64, 192, 192), (2048, 2048, 2048), (128, 128, 128))
        hi = tuple(l + s for l, s in zip(lo, (64, 192, 192)))
        n_chunks = 1
        for l_i, h_i in zip(lo, hi):
            n_chunks *= (h_i - 1) // 128 - (l_i // 128) + 1
        self.assertEqual(n_chunks, 4, f"aligned 64x192x192 slab lo={lo} hi={hi} spanned {n_chunks} chunks != 4")

    def test_ast_has_no_gpu_imports(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        for n in names:
            self.assertFalse(any(n == g or n.startswith(g + ".") for g in GPU_MODULES), n)

    def test_subprocess_import_loads_no_gpu_library(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import seismic_rgt_stitched_grower; "
            f"print([m for m in {GPU_MODULES!r} if m in sys.modules])"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        out = subprocess.run(
            [sys.executable, "-c", code, str(HERE)],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
            check=True,
        )
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")


class PrivacyAndReceiptTests(unittest.TestCase):
    def _unread_receipt(self, scroll="PHerc1545", **extra):
        return {
            "boxes": [
                {
                    "scroll": scroll,
                    "metrics": {
                        "sheet_count_rgt": 5,
                        "seam_phase_rmse_rad": 0.08,
                        "mean_on_sheet_after_snap_rgt": 0.91,
                    },
                    **extra,
                }
            ]
        }

    def test_privacy_validator_rejects_coordinate_and_path_leaks(self):
        SRG.validate_public_receipt_privacy(self._unread_receipt("PHerc1545"))
        SRG.validate_public_receipt_privacy(self._unread_receipt("PHerc0846A"))
        bad_receipts = [
            {"note": "C:\\Users\\someone\\data\\x.zarr"},
            {"note": "/Users/someone/scroll"},
            {"note": "/home/someone/scroll"},
            {"note": "file:///tmp/x"},
            {"note": "token ghp_" + "a" * 20},
            {"boxes": [{"scroll": "PHerc0306B", "bbox": [0, 0, 0, 1, 1, 1]}]},
            self._unread_receipt("PHerc1545", origin_zyx=[1, 2, 3]),
            self._unread_receipt("PHerc0846A", chunk_indices=[[1, 2, 3]]),
            self._unread_receipt("PHerc1545", nested={"z0": 5}),
            self._unread_receipt("PHerc0846A", note="fetched 12.30.4"),
            self._unread_receipt("PHerc1545", url="https://host/vol.zarr/0/12/30/4"),
            {"boxes": [{"scroll": "PHerc0846A", "metrics": {"seed_zyx_hint": 1}}]},
        ]
        for receipt in bad_receipts:
            with self.assertRaises(ValueError, msg=json.dumps(receipt)):
                SRG.validate_public_receipt_privacy(receipt)

        clean = SRG.sanitize_public_metrics(
            {
                "sheet_count_rgt": np.int64(5),
                "seam_phase_rmse_rad": np.float64(0.1123456),
                "origin_zyx": [10, 20, 30],
                "private_path": "C:\\Users\\test",
            }
        )
        self.assertEqual(clean, {"sheet_count_rgt": 5, "seam_phase_rmse_rad": 0.112346})

    def test_committed_live_receipt(self):
        self.assertTrue(EVIDENCE_PATH.is_file(), "live stitched preflight receipt missing")
        r = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        SRG.validate_public_receipt_privacy(r)
        self.assertEqual(r["schema"], "seismic_rgt_stitched_patches/v1")
        self.assertTrue(r["cpu_only"])
        self.assertFalse(r["torch_imported"])
        self.assertIn(r["cuda_visible_devices"], ("", "-1"))
        http = r["http"]
        self.assertEqual(http["budget_bytes"], SRG.GLOBAL_HTTP_CAP_BYTES)
        self.assertLessEqual(http["http_bytes"], SRG.GLOBAL_HTTP_CAP_BYTES)
        self.assertFalse(http["persistent_ct_on_disk"])
        scrolls = [b["scroll"] for b in r["boxes"]]
        self.assertEqual(scrolls, ["PHerc0306B", "PHerc1545", "PHerc0846A"])
        for b in r["boxes"]:
            m = b["metrics"]
            self.assertTrue(set(m) <= SRG.PUBLIC_STITCHED_RGT_METRIC_KEYS)
            self.assertEqual(m["box_shape_zyx"], list(SRG.STITCHED_BOX_SHAPE_ZYX))
            self.assertEqual(m["tile_grid"], [2, 2])
            self.assertEqual(m["crossing_fraction_rgt"], 0.0)
            self.assertGreaterEqual(m["sheet_count_rgt"], 2)
            self.assertLess(m["seam_phase_rmse_rad"], 0.5)
            if b["scroll"] in RGT.UNREAD_SCROLLS:
                self.assertEqual(m["snap_passes"], 2)
                self.assertIsNotNone(m["mean_on_sheet_after_snap_rgt"])
        self.assertEqual(sum(b["metrics"]["http_bytes"] for b in r["boxes"]), http["http_bytes"])


if __name__ == "__main__":
    unittest.main()
