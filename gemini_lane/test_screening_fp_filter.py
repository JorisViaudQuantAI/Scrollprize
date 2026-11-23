# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Deterministic offline synthetic tests + benchmark for screening_fp_filter.py (CPU-only)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import screening_fp_filter as F  # noqa: E402

VOXEL_UM = F.DEFAULT_VOXEL_UM
MM_PX = 1000.0 / VOXEL_UM  # pixels per millimetre (~106.8)


def _woven_ct(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Intact flat woven papyrus mid-slice: crossed-fibre weave plus smoothed noise."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    weave = 0.05 * np.sin(2 * np.pi * xx / 36.0) + 0.05 * np.sin(2 * np.pi * yy / 36.0)
    noise = ndi.gaussian_filter(rng.normal(0.0, 0.05, shape), 1.0)
    return (0.6 + weave + noise).astype(np.float32)


def _segment_mask(shape, p0, p1, width) -> np.ndarray:
    """Pixels within ``width / 2`` of the segment ``p0 -> p1`` (y, x)."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = y1 - y0, x1 - x0
    t = np.clip(((yy - y0) * dy + (xx - x0) * dx) / (dy * dy + dx * dx), 0.0, 1.0)
    return np.hypot(yy - (y0 + t * dy), xx - (x0 + t * dx)) <= width / 2.0


def _disk_mask(shape, cy, cx, r, wobble=0.0, phase=0.0) -> np.ndarray:
    """Filled disk of radius ``r`` with optional low-order radial wobble (amorphous blob)."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    th = np.arctan2(yy - cy, xx - cx)
    rr = r * (1.0 + wobble * np.sin(2 * th + phase) + 0.5 * wobble * np.sin(3 * th - phase))
    return np.hypot(yy - cy, xx - cx) <= rr


def _prob(mask: np.ndarray, rng: np.random.Generator, peak=0.92, bg=0.05) -> np.ndarray:
    """Soft probability map from a binary ink-like mask plus weak background noise."""
    p = ndi.gaussian_filter(mask.astype(np.float32), 1.5) * (peak - bg) + bg
    return np.clip(p + rng.normal(0.0, 0.01, mask.shape), 0.0, 1.0).astype(np.float32)


def _greek_strokes(shape) -> np.ndarray:
    """Crisp thin (~0.3 mm) Greek-like strokes: an omicron ring plus a lambda."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    rr = np.hypot(yy - 200, xx - 200)
    ring = (rr <= 140) & (rr >= 110)
    lam = _segment_mask(shape, (560, 380), (140, 480), 30) | _segment_mask(shape, (140, 480), (560, 600), 30)
    return ring | lam


def case_intact_strokes(seed: int = 0) -> dict:
    """(i) Thin one-sided Greek-like strokes on intact woven CT with agreeing A-hn."""
    rng = np.random.default_rng(seed)
    shape = (640, 640)
    ct, valid = _woven_ct(shape, rng), np.ones(shape, bool)
    ink = _greek_strokes(shape)
    halo = F.build_crack_void_halo_mask(ct, valid)
    asc, desc = _prob(ink, rng), _prob(np.zeros(shape, bool), rng)
    ahn_asc, ahn_desc = _prob(ndi.binary_dilation(ink, iterations=2), rng, peak=0.85), _prob(np.zeros(shape, bool), rng)
    diag = F.score_probability_map_diagnostics(asc, desc, valid, halo, ahn_asc, ahn_desc, VOXEL_UM)
    stroke_hits = (asc > 0.7) & valid
    retained = int((stroke_hits & ~halo).sum()) / max(int(stroke_hits.sum()), 1)
    return {"diag": diag, "stroke_hit_retention": retained}


def case_crack_specks(seed: int = 1) -> dict:
    """(ii) Studio > 0.7 specks on delamination crack walls and the mesh-border erosion halo."""
    rng = np.random.default_rng(seed)
    shape = (640, 640)
    ct = _woven_ct(shape, rng)
    valid = np.ones(shape, bool)
    valid[:, :40] = False  # unrendered mesh border
    ct[~valid] = 0.0
    xs = np.arange(shape[1])
    crack_y = (320 + 40 * np.sin(2 * np.pi * xs / 320.0)).astype(int)
    half_w = 4
    for x, y in zip(xs, crack_y):
        ct[y - half_w : y + half_w + 1, x] = 0.1  # air gap between delaminated layers
    specks = np.zeros(shape, bool)
    for x in range(60, 620, 20):  # specks hugging both crack walls
        side = 1 if (x // 20) % 2 else -1
        specks |= _disk_mask(shape, crack_y[x] + side * (half_w + 4), x, 3)
    for y in range(40, 620, 60):  # specks on the eroded mesh border
        specks |= _disk_mask(shape, y, 44, 3)
    halo = F.build_crack_void_halo_mask(ct, valid)
    diag = F.score_probability_map_diagnostics(_prob(specks, rng), _prob(np.zeros(shape, bool), rng), valid, halo)
    return {"diag": diag, "fp_rejection": diag["faces"][diag["dominant_face"]]["halo_share_gt07"]}


def case_bilateral_mottle(seed: int = 2) -> dict:
    """(iii) Amorphous 2-4 mm clouds firing in both depth orders."""
    rng = np.random.default_rng(seed)
    shape = (960, 960)
    valid = np.ones(shape, bool)
    blobs = [(230, 230, 2.0), (690, 300, 1.5), (520, 760, 1.0)]  # (cy, cx, radius_mm) -> 2/3/4 mm wide
    asc_m = np.zeros(shape, bool)
    desc_m = np.zeros(shape, bool)
    for k, (cy, cx, r_mm) in enumerate(blobs):
        r = r_mm * MM_PX
        asc_m |= _disk_mask(shape, cy, cx, r, wobble=0.06, phase=k)
        desc_m |= _disk_mask(shape, cy + 12, cx - 8, r * 0.95, wobble=0.06, phase=k + 1)
    diag = F.score_probability_map_diagnostics(_prob(asc_m, rng), _prob(desc_m, rng, peak=0.88), valid, None)
    return {"diag": diag}


def case_dot_rows(seed: int = 3) -> dict:
    """(iv) Single-face 1 mm dot rows where A-hn fires somewhere else."""
    rng = np.random.default_rng(seed)
    shape = (360, 1100)
    valid = np.ones(shape, bool)
    r = 0.5 * MM_PX
    dots = np.zeros(shape, bool)
    for cx in np.arange(90, 1050, 1.6 * MM_PX):
        dots |= _disk_mask(shape, 180, cx, r)
    ahn_ink = _segment_mask(shape, (40, 100), (40, 1000), 20)
    none = np.zeros(shape, bool)
    diag = F.score_probability_map_diagnostics(
        _prob(dots, rng), _prob(none, rng), valid, None, _prob(ahn_ink, rng), _prob(none, rng), VOXEL_UM
    )
    return {"diag": diag}


CASES = {
    "intact_strokes": case_intact_strokes,
    "crack_specks": case_crack_specks,
    "bilateral_mottle": case_bilateral_mottle,
    "dot_rows": case_dot_rows,
}


def run_benchmark() -> dict:
    """Run all synthetic cases and return compact metrics with wall-clock timings."""
    out = {}
    for name, fn in CASES.items():
        t0 = time.perf_counter()
        res = fn()
        d = res.pop("diag")
        out[name] = {
            **{k: round(v, 4) for k, v in res.items()},
            "seconds": round(time.perf_counter() - t0, 3),
            "clean_papyrus_area_frac": d["clean_papyrus_area_frac"],
            "asymmetry_ratio": d["asymmetry_ratio"],
            "bilateral_overlap_frac": d["bilateral_overlap_frac"],
            "strokeness": d["strokeness"]["weighted_mean"],
            "two_family_iou": None if d["two_family"] is None else d["two_family"]["iou"],
            "flags": [k for k, v in d["flags"].items() if v],
            "ink_like_candidate": d["ink_like_candidate"],
        }
    return out


class CrackHaloMaskTests(unittest.TestCase):
    def test_intact_woven_sheet_keeps_95pct_clean(self):
        rng = np.random.default_rng(10)
        ct = _woven_ct((512, 512), rng)
        halo = F.build_crack_void_halo_mask(ct, np.ones(ct.shape, bool))
        self.assertLessEqual(halo.mean(), 0.05)

    def test_invalid_pixels_never_flagged_and_border_halo_present(self):
        rng = np.random.default_rng(11)
        ct = _woven_ct((256, 256), rng)
        valid = np.ones(ct.shape, bool)
        valid[:, :32] = False
        ct[~valid] = 0.0
        halo = F.build_crack_void_halo_mask(ct, valid, border_halo_radius=5)
        self.assertFalse((halo & ~valid).any())
        self.assertTrue(halo[:, 32:37].all())
        self.assertFalse(halo[:, 60:].any())

    def test_explicit_threshold_and_percentile_modes(self):
        ct = np.full((128, 128), 0.6, np.float32)
        ct[60:68, :] = 0.1
        valid = np.ones(ct.shape, bool)
        for kw in ({"dark_threshold": 0.3}, {"dark_percentile": 5.0}):
            halo = F.build_crack_void_halo_mask(ct, valid, halo_radius=3, **kw)
            self.assertTrue(halo[59:69, :].all(), kw)  # 1-px smoothing softens the band edge
            self.assertFalse(halo[:50].any(), kw)

    def test_shape_mismatch_and_empty_valid(self):
        with self.assertRaises(ValueError):
            F.build_crack_void_halo_mask(np.zeros((4, 4)), np.ones((4, 5), bool))
        self.assertFalse(F.build_crack_void_halo_mask(np.zeros((4, 4)), np.zeros((4, 4), bool)).any())


class StrokenessTests(unittest.TestCase):
    def test_line_vs_disk(self):
        shape = (300, 300)
        line = F.component_strokeness(_segment_mask(shape, (150, 20), (150, 280), 12))
        disk = F.component_strokeness(_disk_mask(shape, 150, 150, 100))
        self.assertGreater(line["weighted_mean"], 2.0)
        self.assertLess(disk["weighted_mean"], 0.5)

    def test_empty(self):
        self.assertIsNone(F.component_strokeness(np.zeros((10, 10), bool))["weighted_mean"])


class SyntheticCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = {name: fn() for name, fn in CASES.items()}

    def test_i_intact_strokes_retained_and_ink_like(self):
        r = self.res["intact_strokes"]
        d = r["diag"]
        self.assertGreaterEqual(d["clean_papyrus_area_frac"], 0.95)
        self.assertGreaterEqual(r["stroke_hit_retention"], 0.95)
        self.assertEqual(d["dominant_face"], "asc")
        self.assertGreaterEqual(d["asymmetry_ratio"], 3.0)
        self.assertGreaterEqual(d["strokeness"]["weighted_mean"], 1.5)
        self.assertLess(d["bilateral_overlap_frac"], 0.05)
        self.assertGreaterEqual(d["two_family"]["iou"], 0.5)
        self.assertTrue(d["ink_like_candidate"], d["flags"])

    def test_ii_crack_specks_rejected(self):
        r = self.res["crack_specks"]
        self.assertGreaterEqual(r["fp_rejection"], 0.8)
        self.assertTrue(r["diag"]["flags"]["crack_halo_dominated"])
        self.assertGreaterEqual(r["diag"]["clean_papyrus_area_frac"], 0.8)
        self.assertFalse(r["diag"]["ink_like_candidate"])

    def test_iii_bilateral_mottle_flagged(self):
        d = self.res["bilateral_mottle"]["diag"]
        self.assertGreaterEqual(d["bilateral_overlap_frac"], 0.5)
        self.assertLess(d["strokeness"]["weighted_mean"], 1.5)
        self.assertTrue(d["flags"]["bilateral_mottle"])
        self.assertTrue(d["flags"]["low_asymmetry"])
        self.assertFalse(d["ink_like_candidate"])
        widths = sorted(c["extent_mm"] for c in d["strokeness"]["top_components"])
        self.assertGreater(widths[0], 1.8)
        self.assertLess(widths[-1], 4.6)

    def test_iv_dot_rows_flagged(self):
        d = self.res["dot_rows"]["diag"]
        self.assertGreaterEqual(d["asymmetry_ratio"], 3.0)
        self.assertLess(d["strokeness"]["weighted_mean"], 1.5)
        self.assertTrue(d["flags"]["low_strokeness"])
        self.assertTrue(d["flags"]["two_family_disagree"])
        self.assertLess(d["two_family"]["iou"], 0.1)
        self.assertFalse(d["ink_like_candidate"])

    def test_json_safe(self):
        for name, r in self.res.items():
            json.dumps(r["diag"], allow_nan=False)

    def test_input_validation(self):
        z = np.zeros((8, 8), np.float32)
        v = np.ones((8, 8), bool)
        with self.assertRaises(ValueError):
            F.score_probability_map_diagnostics(z, z, v, ahn_asc=z)
        with self.assertRaises(ValueError):
            F.score_probability_map_diagnostics(z, np.zeros((8, 9)), v)
        d = F.score_probability_map_diagnostics(z, z, v)
        self.assertTrue(d["flags"]["no_hits"])
        self.assertFalse(d["ink_like_candidate"])

    def test_module_is_cpu_only(self):
        self.assertNotIn("torch", sys.modules.get("screening_fp_filter").__dict__)


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_runs_and_reports(self):
        bench = run_benchmark()
        sys.stderr.write("\nscreening_fp_filter benchmark: " + json.dumps(bench, indent=1) + "\n")
        self.assertEqual(set(bench), set(CASES))
        self.assertTrue(bench["intact_strokes"]["ink_like_candidate"])
        self.assertFalse(any(bench[k]["ink_like_candidate"] for k in ("crack_specks", "bilateral_mottle", "dot_rows")))


class RealCTBenchmarkTests(unittest.TestCase):
    """Offline checks of ``benchmark_fp_filter_real_ct`` helpers and its committed public receipt."""

    RECEIPT = HERE / "evidence" / "fp_filter_real_ct_benchmark.json"

    @classmethod
    def setUpClass(cls):
        import benchmark_fp_filter_real_ct as B

        cls.B = B

    def test_receipt_schema_and_budget(self):
        data = json.loads(self.RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], self.B.SCHEMA_VERSION)
        self.assertEqual({s["role"] for s in data["strips"]}, {"intact_clean_papyrus", "delaminated_crack_void"})
        total = round(sum(s["streamed_mib"]["total"] for s in data["strips"]), 2)
        self.assertEqual(total, data["total_streamed_mib"])
        self.assertLess(total, self.B.MIB_BUDGET)
        self.assertTrue(data["within_budget"])
        for s in data["strips"]:
            h = s["halo"]
            self.assertLessEqual(s["subwindow"]["dry_run_ct_chunks"], self.B.CHUNKS_PER_STRIP)
            self.assertLessEqual(s["streamed_mib"]["ct"], self.B.CT_MIB_CAP_PER_STRIP)
            self.assertAlmostEqual(h["clean_papyrus_retention_frac"] + h["crack_halo_frac"], 1.0, places=5)
            self.assertAlmostEqual(h["void_halo_frac"] + h["border_halo_only_frac"], h["crack_halo_frac"], places=5)
            self.assertEqual(h["void_detection_fired"], h["void_halo_frac"] > 0)
        v = data["verdict"]
        clean = next(s for s in data["strips"] if s["role"] == "intact_clean_papyrus")["halo"]
        self.assertEqual(v["clean_retention_observed"], clean["clean_papyrus_retention_frac"])
        self.assertEqual(v["clean_retention_meets_target"], clean["clean_papyrus_retention_frac"] >= v["clean_retention_target"])

    def test_receipt_privacy(self):
        raw = self.RECEIPT.read_text(encoding="utf-8")
        for forbidden in ("C:\\\\", "C:/", "/Users/", "/home/", "Scrollprize_private", ".png", "top_hotspots",
                          "prob", "studio", "ahn", "detectors", "fp_diagnostics"):
            self.assertNotIn(forbidden, raw)
        data = json.loads(raw)
        unread = next(s for s in data["strips"] if s["role"] == "delaminated_crack_void")
        self.assertIsNone(unread["subwindow"]["window"])
        self.assertTrue(all(s["ct_volume"].startswith("https://") for s in data["strips"]))

    def test_centered_crop_and_subwindow_selection(self):
        self.assertEqual(self.B.centered_crop((56, 245)), (3, 53, 97, 147))
        with self.assertRaises(ValueError):
            self.B.centered_crop((40, 245))
        grid = np.zeros((40, 40), dtype=np.float32)
        valid = np.ones((40, 40), dtype=bool)
        # Footprint grows with area: only windows of <= 120 vertices fit the per-strip chunk budget.
        footprint = lambda x, y, z: int(np.ceil(x.size / 120 * self.B.CHUNKS_PER_STRIP))  # noqa: E731
        window, factor, n = self.B.select_subwindow(grid, grid, grid, valid, (0, 20, 0, 20), footprint)
        self.assertEqual(factor, 0.5)
        self.assertEqual((window[1] - window[0], window[3] - window[2]), (10, 10))
        self.assertLessEqual(n, self.B.CHUNKS_PER_STRIP)
        with self.assertRaises(RuntimeError):
            self.B.select_subwindow(grid, grid, grid, valid, (0, 20, 0, 20), lambda *_: 10 ** 6)

    def test_summarize_halo_and_receipt_on_synthetic_ct(self):
        rng = np.random.default_rng(7)
        woven = _woven_ct((240, 240), rng)
        cracked = woven.copy()
        cracked[110:130, 20:220] = 0.05
        valid = np.ones((240, 240), dtype=bool)
        clean_stats = self.B.summarize_halo(woven, valid, VOXEL_UM)
        crack_stats = self.B.summarize_halo(cracked, valid, VOXEL_UM)
        self.assertFalse(clean_stats["void_detection_fired"])
        self.assertTrue(crack_stats["void_detection_fired"])
        self.assertGreater(crack_stats["crack_halo_frac"], clean_stats["crack_halo_frac"])
        mk = lambda role, h: {"role": role, "halo": h, "streamed_mib": {"total": 1.0}}  # noqa: E731
        receipt = self.B.build_receipt([mk("intact_clean_papyrus", clean_stats), mk("delaminated_crack_void", crack_stats)])
        self.assertTrue(receipt["verdict"]["clean_retention_meets_target"])
        self.assertGreater(receipt["verdict"]["delaminated_minus_clean_halo_frac"], 0)
        with self.assertRaises(ValueError):
            self.B.summarize_halo(woven, np.zeros_like(valid), VOXEL_UM)

    def test_budgeted_reader_refuses_fetch_past_cap(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="fpbench-zarr-test-") as tmp:
            level = Path(tmp) / "0"
            level.mkdir()
            meta = {"zarr_format": 2, "shape": [8, 8, 16], "chunks": [8, 8, 8], "dtype": "|u1",
                    "compressor": None, "fill_value": 0, "order": "C", "dimension_separator": "."}
            (level / ".zarray").write_text(json.dumps(meta), encoding="utf-8")
            for cx in (0, 1):
                (level / f"0.0.{cx}").write_bytes(np.full(512, 100 + cx, dtype=np.uint8).tobytes())
            cap_mib = 512 * 1.5 / 1024 ** 2
            reader = self.B.BudgetedZarrV2(tmp, cache_mb=1, workers=1, max_mib=cap_mib)
            block, _ = reader.read_box((0, 0, 0), (8, 8, 8))
            self.assertEqual(int(block[0, 0, 0]), 100)
            with self.assertRaises(self.B.BudgetExceeded):
                reader.read_box((0, 0, 8), (8, 8, 16))
            self.assertEqual(reader.chunks_fetched, 1)


if __name__ == "__main__":
    unittest.main()
