# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for tfce_stroke_calibration.py (Exp T6 / Q34)."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tfce_stroke_calibration as M  # noqa: E402

VOXEL_UM = M.DEFAULT_VOXEL_UM  # 8.64 um/px
MM_PX = 1000.0 / VOXEL_UM


def _segment_mask(
    shape: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    width_px: float,
) -> np.ndarray:
    """Boolean mask of pixels within ``width_px / 2`` of segment ``p0 -> p1``."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = y1 - y0, x1 - x0
    denom = max(dy * dy + dx * dx, 1e-6)
    t = np.clip(((yy - y0) * dy + (xx - x0) * dx) / denom, 0.0, 1.0)
    return np.hypot(yy - (y0 + t * dy), xx - (x0 + t * dx)) <= (width_px / 2.0)


def _synthetic_open_letter_line(
    shape: tuple[int, int] = (360, 1000),
    stroke_mm: float = 0.35,
    voxel_um: float = VOXEL_UM,
) -> np.ndarray:
    """Single line of open Greek strokes (Π, Τ, Ν, Ε) with narrow width and b1 == 0."""
    mm_px = 1000.0 / voxel_um
    w = stroke_mm * mm_px
    m = np.zeros(shape, dtype=bool)
    # Pi (Π)
    m |= _segment_mask(shape, (100.0, 100.0), (260.0, 100.0), w)
    m |= _segment_mask(shape, (100.0, 210.0), (260.0, 210.0), w)
    m |= _segment_mask(shape, (100.0, 90.0), (100.0, 220.0), w)
    # Tau (Τ)
    m |= _segment_mask(shape, (100.0, 310.0), (100.0, 440.0), w)
    m |= _segment_mask(shape, (100.0, 375.0), (260.0, 375.0), w)
    # Nu (Ν)
    m |= _segment_mask(shape, (260.0, 530.0), (100.0, 530.0), w)
    m |= _segment_mask(shape, (100.0, 530.0), (260.0, 650.0), w)
    m |= _segment_mask(shape, (260.0, 650.0), (100.0, 650.0), w)
    # Epsilon-like open E (Ε)
    m |= _segment_mask(shape, (100.0, 760.0), (260.0, 760.0), w)
    m |= _segment_mask(shape, (100.0, 760.0), (100.0, 880.0), w)
    m |= _segment_mask(shape, (180.0, 760.0), (180.0, 860.0), w)
    m |= _segment_mask(shape, (260.0, 760.0), (260.0, 880.0), w)
    return m


def _synthetic_looped_letter_line(
    shape: tuple[int, int] = (360, 1000),
    stroke_mm: float = 0.35,
    voxel_um: float = VOXEL_UM,
) -> np.ndarray:
    """Single line of looped Greek strokes (Ο, Δ, Π) with narrow width and b1 >= 2."""
    mm_px = 1000.0 / voxel_um
    w = stroke_mm * mm_px
    half_w = w / 2.0
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)

    m = np.zeros(shape, dtype=bool)
    # Omicron 'Ο'
    m |= np.abs(np.hypot(yy - 180.0, xx - 200.0) - 0.85 * mm_px) <= half_w
    # Delta 'Δ'
    m |= _segment_mask(shape, (95.0, 500.0), (265.0, 410.0), w)
    m |= _segment_mask(shape, (95.0, 500.0), (265.0, 590.0), w)
    m |= _segment_mask(shape, (265.0, 410.0), (265.0, 590.0), w)
    # Pi 'Π'
    m |= _segment_mask(shape, (100.0, 720.0), (260.0, 720.0), w)
    m |= _segment_mask(shape, (100.0, 840.0), (260.0, 840.0), w)
    m |= _segment_mask(shape, (100.0, 710.0), (100.0, 850.0), w)
    return m


class TestTfceStrokeCalibration(unittest.TestCase):
    """Verify TFCE, stroke-bounded TFCE, empirical-null FWER, and display inversion."""

    def test_cpu_only_no_torch_imported(self) -> None:
        src = (HERE / "tfce_stroke_calibration.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(HERE)!r}); "
                "import tfce_stroke_calibration; assert 'torch' not in sys.modules",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_tfce_2d_exact_analytical_disk_and_connectivity(self) -> None:
        shape = (64, 64)
        valid = np.ones(shape, dtype=bool)
        score = np.zeros(shape, dtype=np.float32)
        score[20:30, 20:30] = 0.75  # exact 100-pixel square at constant height 0.75
        area = 100.0
        n_steps = 32
        H, E = 2.0, 0.5
        dh = 1.0 / n_steps
        expected = sum(((k + 0.5) * dh) ** H * dh * (area ** E) for k in range(n_steps) if (k + 0.5) * dh <= 0.75)

        res = M.tfce_2d(score, valid, H=H, E=E, n_steps=n_steps, connectivity=8)
        self.assertEqual(res.dtype, np.float32)
        self.assertEqual(res.shape, shape)
        self.assertAlmostEqual(float(res[25, 25]), expected, places=4)
        self.assertEqual(float(res[0, 0]), 0.0)

        # Diagonal pixels connect under connectivity=8 (extent=2) but stay separate under 4 (extent=1)
        diag = np.zeros((16, 16), dtype=np.float32)
        diag[5, 5] = 0.8
        diag[6, 6] = 0.8
        t8 = M.tfce_2d(diag, np.ones((16, 16), dtype=bool), n_steps=16, connectivity=8)
        t4 = M.tfce_2d(diag, np.ones((16, 16), dtype=bool), n_steps=16, connectivity=4)
        self.assertAlmostEqual(float(t8[5, 5]) / float(t4[5, 5]), float(np.sqrt(2.0)), places=4)

    def test_tfce_2d_edge_cases_and_validation(self) -> None:
        # NaN/Inf/out-of-range/invalid pixels handled safely
        score = np.array([[np.nan, np.inf, -0.5, 1.5], [0.0, 0.5, 0.9, 0.2]], dtype=np.float64)
        valid = np.array([[True, True, True, True], [False, True, True, False]], dtype=bool)
        out = M.tfce_2d(score, valid, n_steps=16)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(float(out[0, 0]), 0.0)
        self.assertEqual(float(out[0, 1]), 0.0)
        self.assertEqual(float(out[0, 2]), 0.0)
        self.assertEqual(float(out[1, 0]), 0.0)
        self.assertGreater(float(out[0, 3]), float(out[1, 1]))

        # Empty valid mask -> zeros
        zeros = M.tfce_2d(np.ones((8, 8)), np.zeros((8, 8), dtype=bool))
        self.assertEqual(float(zeros.max()), 0.0)

        # Invalid arguments raise ValueError
        with self.assertRaises(ValueError):
            M.tfce_2d(np.zeros((8, 8)), np.ones((8, 9), dtype=bool))
        with self.assertRaises(ValueError):
            M.tfce_2d(np.zeros((8, 8)), np.ones((8, 8), dtype=bool), connectivity=6)
        with self.assertRaises(ValueError):
            M.tfce_2d(np.zeros((8, 8)), np.ones((8, 8), dtype=bool), H=-1.0)
        with self.assertRaises(ValueError):
            M.tfce_2d(np.zeros((8, 8)), np.ones((8, 8), dtype=bool), n_steps=0)

    def test_stroke_bounded_tfce_suppresses_wide_clouds_over_thin_strokes(self) -> None:
        shape = (420, 420)
        valid = np.ones(shape, dtype=bool)
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)

        # Thin stroke (~0.35 mm wide, ~2.8 mm long)
        stroke_m = _segment_mask(shape, (60.0, 60.0), (360.0, 360.0), 0.35 * MM_PX)
        stroke_p = np.clip(ndi.gaussian_filter(stroke_m.astype(np.float32), 1.5) * 0.90, 0, 1)

        # Wide 3.0 mm diffuse cloud
        cloud_m = np.hypot(yy - 210.0, xx - 210.0) <= (1.5 * MM_PX)
        cloud_p = np.clip(ndi.gaussian_filter(cloud_m.astype(np.float32), 5.0) * 0.88, 0, 1)

        raw_stroke = M.tfce_2d(stroke_p, valid)
        raw_cloud = M.tfce_2d(cloud_p, valid)
        # Unbounded TFCE is fooled by the huge 3 mm diffuse cloud
        self.assertGreater(float(raw_cloud.max()), float(raw_stroke.max()))

        sb_stroke = M.stroke_bounded_tfce_2d(stroke_p, valid, voxel_um=VOXEL_UM)
        sb_cloud = M.stroke_bounded_tfce_2d(cloud_p, valid, voxel_um=VOXEL_UM)

        ret_stroke = float(sb_stroke.max()) / float(raw_stroke.max())
        ret_cloud = float(sb_cloud.max()) / float(raw_cloud.max())
        self.assertGreater(ret_stroke, 0.75)
        self.assertLess(ret_cloud, 0.25)
        # After stroke bounding, the thin stroke decisively beats the 3 mm cloud
        self.assertGreater(float(sb_stroke.max()), 2.0 * float(sb_cloud.max()))

    def test_tfce_opposite_face_fwer_regimes_and_zero_clip_resilience(self) -> None:
        rng = np.random.default_rng(77)
        shape = (360, 1000)
        valid = np.ones(shape, dtype=bool)

        # 1) Looped single-face Greek line with zero-clipped background (like PHerc. 0306B display)
        loop_m = _synthetic_looped_letter_line(shape, stroke_mm=0.35, voxel_um=VOXEL_UM)
        loop_asc = np.clip(
            ndi.gaussian_filter(loop_m.astype(np.float32), 1.5) * 0.92
            + rng.normal(0.0, 0.04, shape).astype(np.float32),
            0.0,
            1.0,
        )
        # Zero-clipped blank opposite-face null (~50% exact zeros, small papyrus ripples)
        opp_raw = ndi.gaussian_filter(rng.normal(0.0, 0.14, shape).astype(np.float32), 2.5)
        opp_clipped = np.clip(opp_raw, 0.0, 1.0)
        res_loop = M.tfce_opposite_face_fwer(loop_asc, opp_clipped, valid, voxel_um=VOXEL_UM)
        self.assertTrue(res_loop["q34_calibrated"]["passes_q34_bar"])
        self.assertTrue(res_loop["q34_calibrated"]["has_enclosed_loops"])
        self.assertGreater(res_loop["max_tfce_ratio"], 5.0)
        self.assertGreater(res_loop["p995_tfce_ratio"], 5.0)
        self.assertGreater(res_loop["diff_tfce_p99"], 12.0)
        self.assertGreater(res_loop["stroke_bounded_tfce"]["retention_max"], 0.70)
        self.assertGreaterEqual(res_loop["transverse_stroke_topology"]["betti"]["dominant"]["max_b1"], 2)
        self.assertTrue(res_loop["logit_zero_clip_diagnostic"]["zero_clip_logit_collapse_detected"])

        # 2) Open-letter single-face line (b1 == 0) STILL passes calibrated Q34 bar
        open_m = _synthetic_open_letter_line(shape, stroke_mm=0.35, voxel_um=VOXEL_UM)
        open_asc = np.clip(
            ndi.gaussian_filter(open_m.astype(np.float32), 1.5) * 0.90 + 0.04, 0.0, 1.0
        )
        open_desc = np.clip(0.04 + rng.normal(0.0, 0.01, shape).astype(np.float32), 0.0, 1.0)
        res_open = M.tfce_opposite_face_fwer(open_asc, open_desc, valid, voxel_um=VOXEL_UM)
        self.assertEqual(res_open["transverse_stroke_topology"]["betti"]["dominant"]["max_b1"], 0)
        self.assertTrue(res_open["q34_calibrated"]["passes_q34_bar"])
        self.assertEqual(res_open["q34_calibrated"]["regime_label"], "single_face_open_stroke_pass")

        # 3) Blank noise vs blank noise FAILS
        blank_a = np.clip(
            ndi.gaussian_filter(rng.normal(0.0, 0.14, shape).astype(np.float32), 2.5), 0.0, 1.0
        )
        res_blank = M.tfce_opposite_face_fwer(blank_a, opp_clipped, valid, voxel_um=VOXEL_UM)
        self.assertFalse(res_blank["q34_calibrated"]["passes_q34_bar"])
        self.assertLess(res_blank["max_tfce_ratio"], 2.5)
        self.assertLess(res_blank["diff_tfce_p99"], 2.0)

        # 4) Synthetic PHerc. 0125 single-face dot row (0.6-1.2 mm) FAILS
        dot_a, dot_b, dot_v = M.make_synthetic_pherc0125_dot_row(
            shape, voxel_um=VOXEL_UM, diameters_mm=(0.65, 0.85, 1.0, 1.15)
        )
        res_dots = M.tfce_opposite_face_fwer(dot_a, dot_b, dot_v, voxel_um=VOXEL_UM)
        self.assertFalse(res_dots["q34_calibrated"]["passes_q34_bar"])
        self.assertFalse(res_dots["q34_calibrated"]["passes_stroke_morphology"])
        self.assertEqual(res_dots["transverse_stroke_topology"]["betti"]["dominant"]["max_b1"], 0)

        # 5) Synthetic PHerc. 1203 2-4 mm bilateral clouds FAIL
        cl_a, cl_b, cl_v = M.make_synthetic_pherc1203_bilateral_clouds(
            shape, voxel_um=VOXEL_UM, diameters_mm=(2.2, 2.8)
        )
        res_cloud = M.tfce_opposite_face_fwer(cl_a, cl_b, cl_v, voxel_um=VOXEL_UM)
        self.assertFalse(res_cloud["q34_calibrated"]["passes_q34_bar"])
        self.assertFalse(res_cloud["q34_calibrated"]["passes_width_retention"])
        self.assertLess(res_cloud["stroke_bounded_tfce"]["retention_max"], 0.30)

        # 6) Empty valid mask -> JSON-safe zero result
        res_empty = M.tfce_opposite_face_fwer(loop_asc, opp_clipped, np.zeros(shape, dtype=bool))
        self.assertEqual(res_empty["valid_px"], 0)
        self.assertFalse(res_empty["q34_calibrated"]["passes_q34_bar"])
        json.dumps(res_empty)

    def test_invert_display_tone_map(self) -> None:
        h, w = 80, 120
        page = np.array([243.0, 236.0, 222.0], dtype=np.float32)
        ink = np.array([22.0, 20.0, 18.0], dtype=np.float32)
        offsheet = np.array([150.0, 160.0, 175.0], dtype=np.float32)
        border_blue = np.array([30.0, 90.0, 220.0], dtype=np.float32)

        gt_d = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
        rgb = np.round(page + gt_d[..., None] * (ink - page)).astype(np.uint8)
        # Put off-sheet blue-grey on left strip, blue claim-box border on row 30, outside-map margin
        rgb[:, 10:25] = offsheet.astype(np.uint8)
        rgb[30, 30:100] = border_blue.astype(np.uint8)

        meta = {
            "display": {"page_rgb": page.tolist(), "ink_rgb": ink.tolist()},
            "image_px": {"map_left": 10, "map_top": 10, "map_w": 100, "map_h": 60},
            "claim_box_image_px": {"x0": 30, "x1": 100, "y0": 20, "y1": 50},
            "half_tone_share_box": 0.5,
            "p99_tone_box": 0.85,
        }
        d_inv, valid, diag = M.invert_display_tone_map(rgb, meta)
        # Outside map excluded
        self.assertFalse(valid[5, 50])
        self.assertFalse(valid[75, 50])
        # Off-sheet blue-grey excluded
        self.assertFalse(valid[40, 15])
        # Blue claim-box border excluded
        self.assertFalse(valid[30, 50])
        # Valid interior matches ground-truth tone accurately
        self.assertTrue(valid[40, 60])
        err = float(np.abs(d_inv[valid] - gt_d[valid]).max())
        self.assertLess(err, 0.01)
        self.assertIn("claim_box_inversion_check", diag)

    def test_http_byte_budget_guard(self) -> None:
        budget = M.HttpByteBudget(cap_bytes=1024, used_bytes=1024)
        self.assertEqual(budget.remaining(), 0)
        with self.assertRaises(RuntimeError):
            budget.fetch("https://example.com/file.bin")
        with self.assertRaises(ValueError):
            M.HttpByteBudget(cap_bytes=1024).fetch("file:///etc/passwd")

    def test_evidence_json_integrity_and_no_local_paths(self) -> None:
        ev_path = HERE / "evidence" / "tfce_q34_calibration.json"
        if not ev_path.exists():
            self.skipTest("tfce_q34_calibration.json not yet generated")
        raw_text = ev_path.read_text(encoding="utf-8")
        self.assertNotIn("C:\\Users\\", raw_text)
        self.assertNotIn("C:/Users/", raw_text)
        data = json.loads(raw_text)
        self.assertLessEqual(data["sources"]["total_http_bytes"], M.GLOBAL_HTTP_BYTE_CAP)
        self.assertTrue(data["sources"]["png_sha256_matches_meta"])
        rows = {r["case"]: r for r in data["calibration_table"]}
        self.assertEqual(
            set(rows.keys()),
            {
                "a_real_pherc0306b_line1",
                "b_real_pherc0306b_blank_margin",
                "c_synthetic_pherc0125_dot_row",
                "d_synthetic_pherc1203_bilateral_clouds",
            },
        )
        self.assertTrue(rows["a_real_pherc0306b_line1"]["passes_q34_bar"])
        self.assertFalse(rows["b_real_pherc0306b_blank_margin"]["passes_q34_bar"])
        self.assertFalse(rows["c_synthetic_pherc0125_dot_row"]["passes_q34_bar"])
        self.assertFalse(rows["d_synthetic_pherc1203_bilateral_clouds"]["passes_q34_bar"])


if __name__ == "__main__":
    unittest.main()
