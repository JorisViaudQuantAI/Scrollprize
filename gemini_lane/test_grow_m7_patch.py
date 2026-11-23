# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline synthetic-Zarr tests for grow_m7_patch.py (CPU-only, no network, no scroll data)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grow_m7_patch as G  # noqa: E402
import stream_snap_render as SSR  # noqa: E402
from test_stream_snap_render import _write_local_zarr_v2  # noqa: E402

SHAPE = (320, 320, 48)  # zyx; sheets are roughly x = const (normal ~ x)


def sheet_height(zz, yy, base=16.0):
    return base + 0.04 * yy + 2.5 * np.sin(zz / 45.0)


def make_m7(shape=SHAPE, bases=(16.0, 30.0), half_width=1.0):
    zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=np.float64) for s in shape), indexing="ij")
    m = np.zeros(shape, dtype=np.uint8)
    for b in bases:
        m[np.abs(xx - sheet_height(zz, yy, b)) <= half_width] = 255
    return m


def make_ct(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 256, size=shape, dtype=np.uint8)


class GrowBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.m7 = make_m7()
        cls.m7_root = root / "m7.zarr"
        _write_local_zarr_v2(cls.m7_root, cls.m7, chunks=(64, 64, 64), level="0")
        _write_local_zarr_v2(cls.m7_root, cls.m7[::2, ::2, ::2], chunks=(64, 64, 64), level="1")
        cls.ct_root = _write_local_zarr_v2(root / "ct.zarr", make_ct(), chunks=(64, 64, 64), level="0")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def extract(self, level="1", factor=2, **kw):
        with SSR.RemoteZarrV2(self.m7_root, level, workers=1) as r:
            hi = [s for s in r.shape]
            return G.extract_patch(r, box_lo_zyx=(0, 0, 0), box_hi_zyx=hi, extract_factor=factor,
                                   voxel_um=10.0, height_axis=2, **kw)


class TestReadersAndBudget(GrowBase):
    def test_scaled_reader_maps_target_frame_to_coarse_level(self):
        ramp = np.broadcast_to(np.arange(48, dtype=np.uint8)[None, None, :], (16, 16, 48)).copy()
        p = _write_local_zarr_v2(self.root / "ramp.zarr", ramp, chunks=(8, 8, 16), level="0")
        _write_local_zarr_v2(p, ramp[::2, ::2, ::2], chunks=(8, 8, 16), level="1")
        with SSR.RemoteZarrV2(p, "1", workers=1) as r1:
            sr = G.ScaledReader(r1, 2)
            self.assertEqual(sr.shape, (15, 15, 47))
            v, ok = sr.sample_trilinear(np.array([[10.0, 4.0, 4.0], [11.0, 4.0, 4.0]]), return_support=True)
        np.testing.assert_allclose(v, [10.0, 11.0])  # coarse voxel j holds fine value 2j; linear between
        self.assertTrue(ok.all())

    def test_budget_counts_metadata_and_refuses_uncompressed_overrun(self):
        budget = G.ByteBudget(64 * 64 * 64 * 2 + 4096)
        r = G.BudgetedRemoteZarrV2(self.ct_root, "0", budget=budget, workers=1)
        try:
            self.assertGreater(budget.used, 0)  # .zarray charged
            r.read_box((0, 0, 0), (128, 64, 48))  # two chunks fit
            with self.assertRaises(G.BudgetExceeded):
                r.read_box((128, 0, 0), (192, 64, 48))
            self.assertLessEqual(budget.used, budget.cap)
            self.assertEqual(r.chunks_fetched, 2)
        finally:
            r.close()

    def test_chunk_estimate_covers_render_fetches(self):
        (x, y, z), _ = self.extract()
        with SSR.RemoteZarrV2(self.ct_root, "0", workers=1, cache_mb=512) as ct:
            exact = G.ScaledReader(ct, 1)
            est = G.estimate_render_chunks(x, y, z, exact, num_slices=9, slice_step=1.0)
            SSR.render_surface_stack(x, y, z, exact, num_slices=9, scale=0.05)
            fetched = set(ct.cache.keys())
        self.assertTrue(fetched)
        self.assertTrue(fetched <= est, fetched - est)
        self.assertLessEqual(len(est), len(fetched) + 4)

    def test_precise_sampler_matches_reference_and_skips_zero_weight_chunks(self):
        vol = make_ct(shape=(16, 16, 16), seed=3)
        p = _write_local_zarr_v2(self.root / "small.zarr", vol, chunks=(8, 8, 8), level="0")
        (p / "0" / "1" / "1" / "1").unlink()  # one missing chunk
        rng = np.random.default_rng(1)
        pts = rng.uniform(-1.0, 16.5, size=(400, 3))
        pts[:5] = [[7.0, 7.0, 7.0], [8.0, 3.0, 3.0], [7.5, 7.5, 7.5], [15.0, 15.0, 15.0], [np.nan, 1, 1]]
        with SSR.RemoteZarrV2(p, "0", workers=1) as a, SSR.RemoteZarrV2(p, "0", workers=1) as b:
            v_ref, s_ref = a.sample_trilinear(pts, return_support=True)
            v, s = G.ScaledReader(b, 1).sample_trilinear(pts, return_support=True)
            np.testing.assert_array_equal(s, s_ref)
            np.testing.assert_allclose(v, v_ref, atol=1e-3)
        with SSR.RemoteZarrV2(p, "0", workers=1) as c:
            G.ScaledReader(c, 1).sample_trilinear(np.array([[7.0, 7.0, 7.0]]))
            self.assertEqual(set(c.cache.keys()), {(0, 0, 0)})  # exact integer: neighbours carry zero weight


class TestExtraction(GrowBase):
    def test_height_field_tracks_one_sheet_without_mixing_neighbours(self):
        h, axis, m = G.extract_height_field(self.m7 >= 128, height_axis=2)
        self.assertEqual(axis, 2)
        self.assertEqual(m["height_axis"], "x")
        zz, yy = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
        ok = np.isfinite(h)
        self.assertGreater(ok.mean(), 0.9)
        err = [np.abs(h[ok] - sheet_height(zz, yy, b)[ok]) for b in (16.0, 30.0)]
        best = min(err, key=lambda e: float(np.median(e)))
        self.assertLess(float(np.percentile(best, 99)), 0.75)

    def test_auto_axis_finds_x_normal(self):
        self.assertEqual(G.dominant_normal_axis(self.m7[::4, ::4, :] >= 128), 2)

    def test_merged_contact_columns_are_excluded(self):
        m = make_m7(shape=(96, 96, 40), bases=(12.0,))
        m[30:60, 40:60, 10:24] = 255   # thick contact blob: runs longer than max_run
        h, _, meta = G.extract_height_field(m >= 128, height_axis=2, max_run=4)
        self.assertTrue(np.isnan(h[32:58, 42:58]).all())
        self.assertLess(meta["single_run_column_fraction"], 1.0)
        self.assertGreater(np.isfinite(h).mean(), 0.7)

    def test_tracker_never_hops_to_adjacent_wrap_across_a_gap(self):
        m = make_m7(shape=(96, 96, 40), bases=(12.0, 16.5), half_width=0.6)
        zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=np.float64) for s in m.shape), indexing="ij")
        hole = (np.abs(zz - 48) < 12) & (np.abs(yy - 48) < 12) & (np.abs(xx - sheet_height(zz, yy, 12.0)) <= 1.0)
        m[hole] = 0
        c, l = G.column_runs(np.moveaxis(m >= 128, 2, 0))
        k = int(np.nanargmin(np.abs(c[10, 10] - sheet_height(10, 10, 12.0))))
        h = G.track_sheet(c, l, (10, 10), k, max_run=4, max_step=1.0, min_separation=3.0)
        zz2, yy2 = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
        ok = np.isfinite(h)
        self.assertGreater(ok.mean(), 0.8)
        self.assertLess(float(np.abs(h[ok] - sheet_height(zz2, yy2, 12.0)[ok]).max()), 1.0)
        self.assertTrue(np.isnan(h[40:56, 40:56]).all())

    def test_column_runs_lists_centres_and_lengths(self):
        m = np.zeros((10, 1, 2), bool)
        m[1:3, 0, 0] = True
        m[6:9, 0, 0] = True
        c, l = G.column_runs(m)
        np.testing.assert_allclose(c[0, 0, :2], [1.5, 7.0])
        np.testing.assert_array_equal(l[0, 0, :2], [2, 3])
        self.assertTrue(np.isnan(c[0, 1]).all())

    def test_grid_drops_nodes_across_a_slit_but_bridges_level_gaps(self):
        aa, bb = np.meshgrid(np.arange(61.0), np.arange(61.0), indexing="ij")
        level = 10.0 + 0.3 * bb
        level[:, 35] = np.nan                 # one-column gap between node columns 30 and 40
        _, _, _, v = G.height_field_to_tifxyz(level, 2, (0, 0, 0), grid_step=10)
        self.assertTrue(v.all())              # consistent slope across the gap: bridged
        slit = level.copy()
        slit[:, 36:] += 6.0                   # same gap, but a wrap jump across it
        x, _, _, v = G.height_field_to_tifxyz(slit, 2, (0, 0, 0), grid_step=10)
        self.assertFalse(v[:, 3:5].any())     # both nodes next to the slit dropped
        self.assertTrue(v[:, :3].all())       # largest side kept, one connected sheet
        self.assertTrue((x[~v] == -1.0).all())

    def test_empty_block_returns_no_patch(self):
        h, _, meta = G.extract_height_field(np.zeros((8, 8, 8), bool), height_axis=0)
        self.assertTrue(np.isnan(h).all())
        self.assertEqual(meta["patch_columns"], 0)

    def test_tifxyz_is_connected_sentinel_clean_and_private(self):
        (x, y, z), m = self.extract()
        self.assertEqual(m["connected_components_in_grid"], 1)
        cx, cy, cz, valid = SSR.validate_tifxyz_channels(x, y, z)
        self.assertTrue(valid.any())
        out = self.root / "private_patch"
        hashes = G.write_private_patch(out, x, y, z, {"uuid": "synthetic"})
        self.assertEqual(set(hashes), {"x.tif", "y.tif", "z.tif", "meta.json"})
        rx, ry, rz, rv, meta = SSR.read_tifxyz(out)
        self.assertEqual(meta["scale"], [0.05, 0.05])
        self.assertEqual(meta["format"], "tifxyz")
        for ch in (rx, ry, rz):
            self.assertTrue((ch[~rv] == -1.0).all())
        np.testing.assert_array_equal(rv, valid)
        # Grid spacing is 1 / scale = 20 target voxels along z (rows).
        rows = np.diff(rz, axis=0)[rv[1:] & rv[:-1]]
        np.testing.assert_allclose(rows, 20.0)

    def test_private_output_refused_inside_git_worktree(self):
        (x, y, z), _ = self.extract()
        with self.assertRaises(ValueError):
            G.write_private_patch(HERE / "_should_not_exist_patch", x, y, z, {})
        self.assertFalse((HERE / "_should_not_exist_patch").exists())


class TestSnapRenderScore(GrowBase):
    def test_snap_pulls_offset_patch_back_onto_sheet(self):
        (x, y, z), _ = self.extract()
        valid = x >= 0
        xs = np.where(valid, x + 3.0, -1.0).astype(np.float32)
        with SSR.RemoteZarrV2(self.m7_root, "0", workers=1) as m7:
            _, _, _, sm = SSR.snap_tifxyz_to_m7(xs, y, z, m7, passes=2, threshold=128)
        self.assertLess(sm["on_sheet_raw"], 0.2)
        self.assertGreater(sm["on_sheet_after_snap"], 0.95)
        self.assertGreater(sm["median_disp_vx"], 1.5)

    def test_end_to_end_grow_passes_gates_on_layer_independent_ct(self):
        with SSR.RemoteZarrV2(self.m7_root, "1", workers=1) as ext, \
                SSR.RemoteZarrV2(self.m7_root, "0", workers=1) as fine, \
                SSR.RemoteZarrV2(self.ct_root, "0", workers=2) as ct:
            budget = G.ByteBudget(10 ** 9)
            (x, y, z), m = G.grow_m7_patch(
                ext, box_lo_zyx=(0, 0, 0), box_hi_zyx=ext.shape, extract_factor=2,
                snap_reader=G.ScaledReader(fine, 1), ct_reader=G.ScaledReader(ct, 1),
                voxel_um=40.0, height_axis=2, threshold=128, num_slices=9,
                ct_budget=budget, ct_chunk_bytes=64 ** 3,
            )
        self.assertGreater(m["on_sheet_after_snap"], 0.95)
        self.assertGreater(m["tiles"], 0)
        self.assertAlmostEqual(m["valid_ct_cm2"], m["valid_ct_px"] * (40e-4) ** 2)
        self.assertGreaterEqual(m["f_parallel"], 0.70)
        self.assertTrue(m["gates"]["passed"], m["gates"])
        self.assertIn(m["backend"], ("onsheet_score", "stream_snap_render_fallback"))

    def test_ct_budget_precheck_refuses_render(self):
        (x, y, z), _ = self.extract()
        with SSR.RemoteZarrV2(self.ct_root, "0", workers=1) as ct, \
                SSR.RemoteZarrV2(self.m7_root, "0", workers=1) as fine:
            with self.assertRaises(G.BudgetExceeded):
                G.snap_render_score(x, y, z, snap_reader=fine, ct_reader=ct, voxel_um=10.0,
                                    snap_passes=0, ct_budget=G.ByteBudget(1000), ct_chunk_bytes=64 ** 3)
            self.assertEqual(ct.chunks_fetched, 0)


class TestCliOffline(unittest.TestCase):
    def test_cli_on_local_mirror_writes_private_patch_and_sanitized_receipt(self):
        spec = G.SCROLL_SPECS["PHerc0846A"]
        with tempfile.TemporaryDirectory() as tmp:
            bucket = Path(tmp) / "bucket"
            m7 = make_m7()
            ct = make_ct()
            mroot = bucket.joinpath(*spec["m7"].split("/"))
            _write_local_zarr_v2(mroot, m7, chunks=(64, 64, 64), level="0")
            _write_local_zarr_v2(mroot, m7[::2, ::2, ::2], chunks=(64, 64, 64), level="1")
            croot = bucket.joinpath(*spec["ct"].split("/"))
            _write_local_zarr_v2(croot, ct, chunks=(64, 64, 64), level="2")
            _write_local_zarr_v2(croot, ct[::2, ::2, ::2], chunks=(32, 32, 32), level="3")
            evidence = Path(tmp) / "receipt.json"
            old = G.S3_BUCKET
            G.S3_BUCKET = str(bucket)
            try:
                rc = G.main(["--scroll", "PHerc0846A", "--budget-mb", "50", "--snap-level", "0",
                             "--box-target", "0", "0", "0", *map(str, SHAPE), "--height-axis", "x",
                             "--num-slices", "9", "--private-out", str(Path(tmp) / "patch"),
                             "--evidence-out", str(evidence)])
            finally:
                G.S3_BUCKET = old
            self.assertEqual(rc, 0)
            text = evidence.read_text(encoding="utf-8")
            data = json.loads(text)
            self.assertNotIn(tmp.replace("\\", "/"), text.replace("\\", "/"))
            self.assertNotIn("bbox", text)
            self.assertEqual(data["levels"]["snap_m7"], 0)
            self.assertEqual(data["levels"]["render_ct"], 3)
            self.assertGreater(data["metrics"]["on_sheet_after_snap"], 0.9)
            self.assertIn("passed", data["gates"])
            self.assertLessEqual(data["stream"]["total_bytes"], data["stream"]["budget_bytes"])
            self.assertEqual(set(data["s3_metadata_inspection"]), set(G.SCROLL_SPECS))
            rx, _, _, rv, meta = SSR.read_tifxyz(Path(tmp) / "patch")
            self.assertEqual(meta["scale"], [0.05, 0.05])
            self.assertTrue(rv.any())


class TestGatesAndReceipt(unittest.TestCase):
    def test_gate_thresholds(self):
        ok = G.evaluate_gates({"f_parallel": 0.70, "valid_ct_cm2": 1.0, "tiles": 4})
        self.assertTrue(ok["passed"])
        self.assertFalse(G.evaluate_gates({"f_parallel": 0.699, "valid_ct_cm2": 5.0, "tiles": 4})["passed"])
        self.assertFalse(G.evaluate_gates({"f_parallel": 0.9, "valid_ct_cm2": 0.99, "tiles": 4})["passed"])
        self.assertFalse(G.evaluate_gates({"f_parallel": 1.0, "valid_ct_cm2": 3.0, "tiles": 0})["passed"])

    def test_sanitize_drops_coordinates_paths_and_arrays(self):
        m = {"f_parallel": np.float64(0.81), "valid_ct_cm2": 1.234567891, "bbox": [[1, 2, 3], [4, 5, 6]],
             "origin": (1, 2, 3), "private_dir": "C:\\Users\\someone\\x", "grid_shape": (12, 30),
             "patch_scores": [{"row": 1}], "tiles": np.int64(7), "backend": "onsheet_score",
             "median_disp_vx": float("nan")}
        s = G.sanitize_metrics(m)
        self.assertEqual(set(s), {"f_parallel", "valid_ct_cm2", "grid_shape", "tiles", "backend", "median_disp_vx"})
        self.assertIsNone(s["median_disp_vx"])
        self.assertEqual(s["grid_shape"], [12, 30])
        json.dumps(s)

    def test_inspect_public_metadata_offline(self):
        def fake_fetch(key, budget):
            budget.charge(10)
            if key.endswith(".zarray"):
                return {"shape": [10, 20, 20], "chunks": [192, 192, 192], "dtype": "|u1",
                        "compressor": {"id": "blosc", "cname": "zstd"}}
            if key.endswith("normal-grids/metadata.json"):
                return {"grid-step": 64, "input-level": 0, "io-threads": 128}
            if key.endswith(".lasagna.json"):
                return {"version": 2, "base_shape_zyx": [1, 2, 3], "groups": {"cos": {"scaledown": 1, "channels": ["cos"], "zarr": "x"}}}
            return None
        b = G.ByteBudget(10 ** 6)
        a = G.inspect_public_metadata("PHerc1545", b, levels=(0,), fetch=fake_fetch)
        self.assertEqual(a["m7_levels"]["0"]["compressor"], "zstd")
        self.assertEqual(a["normal_grids"], {"grid-step": 64, "input-level": 0})
        self.assertEqual(a["lasagna"]["groups"], {"cos": {"scaledown": 1, "channels": ["cos"]}})
        self.assertNotIn("lasagna", G.inspect_public_metadata("PHerc0846A", b, levels=(0,), fetch=fake_fetch))
        self.assertGreater(b.used, 0)

    def test_select_interior_box_picks_radial_x_direction(self):
        o = np.zeros((20, 64, 64), bool)
        yy, xx = np.mgrid[:64, :64]
        r = np.hypot(yy - 32, xx - 32)
        rings = (np.mod(r, 4) < 1) & (r < 28)
        dense = (np.mod(r, 2) < 1) & (r < 28) & (xx > 40)  # extra sheets on the +x side
        o[:] = rings | dense
        lo, hi, sel = G.select_interior_box(o, 16, box_depth=256, box_lateral=512, depth_align=256)
        self.assertEqual(sel["direction"], "+x")
        self.assertEqual([h - l for l, h in zip(lo, hi)], [512, 512, 256])
        self.assertEqual(lo[2] % 256, 0)

    def test_public_receipt_has_no_local_paths(self):
        receipt_path = HERE / "evidence" / "virgin_scroll_patch_preflight.json"
        if not receipt_path.is_file():
            self.skipTest("receipt not generated yet")
        text = receipt_path.read_text(encoding="utf-8")
        for bad in ("C:\\", "C:/", "/Users/", "/home/", "Scrollprize_private", "bbox"):
            self.assertNotIn(bad, text)
        data = json.loads(text)
        self.assertEqual(data["cuda_visible_devices"], "")
        self.assertLess(data["stream"]["total_bytes"], 120 * 10 ** 6)
        self.assertFalse(data["stream"]["persistent_ct_on_disk"])
        self.assertIn("passed", data["gates"])


if __name__ == "__main__":
    unittest.main()
