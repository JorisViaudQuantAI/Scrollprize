# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline synthetic CPU-only unit tests for transverse_stroke_topology.py."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import transverse_stroke_topology as T  # noqa: E402

VOXEL_UM = T.DEFAULT_VOXEL_UM  # 9.362 um/px
MM_PX = 1000.0 / VOXEL_UM  # ~106.81 px/mm


def _segment_mask(
    shape: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    width_px: float,
) -> np.ndarray:
    """Pixels within ``width_px / 2`` of line segment ``p0 -> p1`` (y, x)."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = y1 - y0, x1 - x0
    denom = max(dy * dy + dx * dx, 1e-6)
    t = np.clip(((yy - y0) * dy + (xx - x0) * dx) / denom, 0.0, 1.0)
    return np.hypot(yy - (y0 + t * dy), xx - (x0 + t * dx)) <= (width_px / 2.0)


def _disk_mask(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    radius_px: float,
    wobble: float = 0.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Filled disk of radius ``radius_px`` with optional radial wobble."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    th = np.arctan2(yy - cy, xx - cx)
    rr = radius_px * (1.0 + wobble * np.sin(2.0 * th + phase) + 0.5 * wobble * np.sin(3.0 * th - phase))
    return np.hypot(yy - cy, xx - cx) <= rr


def _ellipse_mask(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    ry: float,
    rx: float,
) -> np.ndarray:
    """Filled axis-aligned ellipse (for elongated broken dot segments)."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2 <= 1.0


def _soft_prob(
    mask: np.ndarray,
    rng: np.random.Generator,
    peak: float = 0.92,
    bg: float = 0.05,
    sigma: float = 1.5,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Smooth probability map from a binary mask plus weak background noise."""
    p = ndi.gaussian_filter(mask.astype(np.float32), sigma) * (peak - bg) + bg
    if noise_std > 0.0:
        p = p + rng.normal(0.0, noise_std, mask.shape).astype(np.float32)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def _synthetic_greek_glyphs(
    shape: tuple[int, int] = (720, 720),
    stroke_mm: float = 0.35,
    voxel_um: float = VOXEL_UM,
) -> np.ndarray:
    """Three Greek-like narrow-stroke glyphs (~0.35 mm): Omicron (1 loop), Theta (2 loops), Delta (1 loop)."""
    mm_px = 1000.0 / voxel_um
    w = stroke_mm * mm_px  # ~37.4 px at 9.362 um/px
    half_w = w / 2.0
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)

    # Omicron 'O' at (180, 180), mean radius 1.15 mm (~123 px)
    r_o = np.hypot(yy - 180.0, xx - 180.0)
    omicron = np.abs(r_o - 1.15 * mm_px) <= half_w

    # Theta 'Θ' at (190, 510): ring + horizontal crossbar -> 2 enclosed b1 loops
    r_th = np.hypot(yy - 190.0, xx - 510.0)
    rad_th = 1.25 * mm_px
    theta = (np.abs(r_th - rad_th) <= half_w) | _segment_mask(
        shape, (190.0, 510.0 - rad_th), (190.0, 510.0 + rad_th), w
    )

    # Delta 'Δ' at bottom center -> 1 enclosed triangular b1 loop
    p_top = (390.0, 360.0)
    p_bl = (620.0, 220.0)
    p_br = (620.0, 500.0)
    delta = (
        _segment_mask(shape, p_top, p_bl, w)
        | _segment_mask(shape, p_top, p_br, w)
        | _segment_mask(shape, p_bl, p_br, w)
    )
    return omicron | theta | delta


def _synthetic_bilateral_clouds(
    shape: tuple[int, int] = (768, 768),
    voxel_um: float = VOXEL_UM,
    shift_px: tuple[int, int] = (10, -8),
) -> tuple[np.ndarray, np.ndarray]:
    """Amorphous 2-4 mm diffuse clouds firing on both asc and desc faces."""
    mm_px = 1000.0 / voxel_um
    blobs = [
        (210.0, 220.0, 1.4),  # 2.8 mm diameter
        (540.0, 240.0, 1.1),  # 2.2 mm diameter
        (390.0, 540.0, 1.7),  # 3.4 mm diameter
    ]
    asc_m = np.zeros(shape, dtype=bool)
    desc_m = np.zeros(shape, dtype=bool)
    dy, dx = shift_px
    for k, (cy, cx, r_mm) in enumerate(blobs):
        r = r_mm * mm_px
        asc_m |= _disk_mask(shape, cy, cx, r, wobble=0.06, phase=float(k))
        desc_m |= _disk_mask(shape, cy + dy, cx + dx, r * 0.96, wobble=0.06, phase=float(k) + 0.7)
    return asc_m, desc_m


def _synthetic_dot_row(
    shape: tuple[int, int] = (720, 720),
    voxel_um: float = VOXEL_UM,
    elongated: bool = False,
) -> np.ndarray:
    """Single-face broken dot row (~0.11/cm^2-like isolated dots, b1 == 0)."""
    mm_px = 1000.0 / voxel_um
    mask = np.zeros(shape, dtype=bool)
    centers = [(360.0, 110.0 + i * 100.0) for i in range(6)]
    for cy, cx in centers:
        if elongated:
            # Slightly elongated broken dot segments (~0.35 mm x 0.55 mm)
            mask |= _ellipse_mask(shape, cy, cx, 0.175 * mm_px, 0.275 * mm_px)
        else:
            # Circular dots (~0.35 mm diameter)
            mask |= _disk_mask(shape, cy, cx, 0.175 * mm_px)
    return mask


class TestTransverseStrokeTopology(unittest.TestCase):
    """Verify differential, stroke-width, Betti filtration, and composite behavior."""

    def test_single_face_greek_glyphs_signal_and_topology(self) -> None:
        rng = np.random.default_rng(101)
        shape = (720, 720)
        valid = np.ones(shape, dtype=bool)
        ink = _synthetic_greek_glyphs(shape, stroke_mm=0.35, voxel_um=VOXEL_UM)
        prob_asc = _soft_prob(ink, rng, peak=0.93, bg=0.05)
        prob_desc = _soft_prob(np.zeros(shape, dtype=bool), rng, peak=0.93, bg=0.05)

        diff = T.opposite_face_differential_map(prob_asc, prob_desc, valid)
        self.assertEqual(diff["delta_map"].shape, shape)
        self.assertEqual(diff["z_map"].shape, shape)
        self.assertLess(diff["bilateral_suppression_frac"], 0.05)
        self.assertGreater(diff["diff_p995"], 4.0)
        self.assertGreater(diff["diff_excess_ratio"], 3.5)
        self.assertGreater(diff["z_p995"], 8.0)
        self.assertGreater(diff["tail_energy_asymmetry"], 0.95)

        sw = T.stroke_width_consistency((prob_asc >= 0.7) & valid, voxel_um=VOXEL_UM)
        self.assertGreater(sw["n_skeleton_px"], 500)
        self.assertIsNotNone(sw["width_median_mm"])
        self.assertGreaterEqual(sw["width_median_mm"], 0.25)
        self.assertLessEqual(sw["width_median_mm"], 0.48)
        self.assertLess(sw["width_cv"], 0.28)
        self.assertGreater(sw["narrow_stroke_frac"], 0.80)
        self.assertLess(sw["wide_blob_frac"], 0.05)

        betti = T.betti_filtration_signature(prob_asc, valid)
        # Omicron (1) + Theta (2) + Delta (1) = 4 enclosed loops
        self.assertGreaterEqual(betti["max_b1"], 4)
        self.assertGreater(betti["integrated_b1"], 1.0)
        self.assertEqual(betti["b1_persistence_frac"], 1.0)
        self.assertGreater(betti["loop_persistence_score"], 0.85)

        res = T.evaluate_transverse_fp_features(prob_asc, prob_desc, valid, VOXEL_UM)
        self.assertEqual(res["dominant_face"], "asc")
        self.assertEqual(res["opposite_face"], "desc")
        self.assertFalse(res["composite"]["gating"])
        self.assertGreater(res["composite"]["single_face_score"], 0.90)
        self.assertGreater(res["composite"]["stroke_consistency_score"], 0.75)
        self.assertGreater(res["composite"]["topology_score"], 0.85)
        self.assertGreater(res["composite"]["discriminability_score"], 0.80)
        self.assertEqual(res["composite"]["regime_hint"], "single_face_looped_stroke_like")

    def test_face_symmetry_desc_dominant(self) -> None:
        rng = np.random.default_rng(102)
        shape = (512, 512)
        valid = np.ones(shape, dtype=bool)
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        r = np.hypot(yy - 256.0, xx - 256.0)
        ring = np.abs(r - 1.1 * MM_PX) <= (0.35 * MM_PX / 2.0)
        prob_asc = _soft_prob(np.zeros(shape, dtype=bool), rng)
        prob_desc = _soft_prob(ring, rng)

        res = T.evaluate_transverse_fp_features(prob_asc, prob_desc, valid, VOXEL_UM)
        self.assertEqual(res["dominant_face"], "desc")
        self.assertEqual(res["opposite_face"], "asc")
        self.assertGreaterEqual(res["betti"]["dominant"]["max_b1"], 1)
        self.assertEqual(res["betti"]["by_face"]["asc"]["max_b1"], 0)
        self.assertGreater(res["composite"]["discriminability_score"], 0.75)

    def test_bilateral_mottle_clouds_suppressed(self) -> None:
        rng = np.random.default_rng(202)
        shape = (768, 768)
        valid = np.ones(shape, dtype=bool)
        asc_m, desc_m = _synthetic_bilateral_clouds(shape, voxel_um=VOXEL_UM, shift_px=(10, -8))
        prob_asc = _soft_prob(asc_m, rng, peak=0.91, bg=0.05, sigma=3.0)
        prob_desc = _soft_prob(desc_m, rng, peak=0.89, bg=0.05, sigma=3.0)

        diff = T.opposite_face_differential_map(prob_asc, prob_desc, valid)
        self.assertGreater(diff["bilateral_suppression_frac"], 0.75)
        self.assertLess(diff["diff_excess_ratio"], 1.8)
        self.assertLess(diff["tail_energy_asymmetry"], 0.88)

        # Matched-radius shifted clouds are nearly symmetric across opposite faces
        asc_eq = _disk_mask(shape, 384.0, 384.0, 1.4 * MM_PX, wobble=0.05)
        desc_eq = _disk_mask(shape, 394.0, 376.0, 1.4 * MM_PX, wobble=0.05)
        diff_eq = T.opposite_face_differential_map(
            _soft_prob(asc_eq, rng, peak=0.90, bg=0.05, sigma=3.0),
            _soft_prob(desc_eq, rng, peak=0.90, bg=0.05, sigma=3.0),
            valid,
        )
        self.assertGreater(diff_eq["bilateral_suppression_frac"], 0.85)
        self.assertLess(abs(diff_eq["tail_energy_asymmetry"] - 0.5), 0.15)

        sw = T.stroke_width_consistency((prob_asc >= 0.7) & valid, voxel_um=VOXEL_UM)
        self.assertIsNotNone(sw["width_median_mm"])
        self.assertGreater(sw["width_median_mm"], 1.8)
        self.assertLess(sw["narrow_stroke_frac"], 0.10)
        self.assertGreater(sw["wide_blob_frac"], 0.85)

        betti = T.betti_filtration_signature(prob_asc, valid)
        self.assertEqual(betti["max_b1"], 0)
        self.assertEqual(betti["integrated_b1"], 0.0)
        self.assertEqual(betti["loop_persistence_score"], 0.0)

        res = T.evaluate_transverse_fp_features(prob_asc, prob_desc, valid, VOXEL_UM)
        self.assertLess(res["composite"]["single_face_score"], 0.15)
        self.assertLess(res["composite"]["stroke_consistency_score"], 0.10)
        self.assertEqual(res["composite"]["topology_score"], 0.0)
        self.assertLess(res["composite"]["discriminability_score"], 0.05)
        self.assertEqual(res["composite"]["regime_hint"], "bilateral_cloud_like")

    def test_single_face_dot_rows_separated_from_glyphs(self) -> None:
        for elongated in (False, True):
            with self.subTest(elongated=elongated):
                rng = np.random.default_rng(303 if not elongated else 304)
                shape = (720, 720)
                valid = np.ones(shape, dtype=bool)
                dots = _synthetic_dot_row(shape, voxel_um=VOXEL_UM, elongated=elongated)
                prob_asc = _soft_prob(dots, rng, peak=0.92, bg=0.05)
                prob_desc = _soft_prob(np.zeros(shape, dtype=bool), rng, peak=0.92, bg=0.05)

                diff = T.opposite_face_differential_map(prob_asc, prob_desc, valid)
                # Dot rows are genuinely single-face asymmetric:
                self.assertLess(diff["bilateral_suppression_frac"], 0.05)
                self.assertGreater(diff["tail_energy_asymmetry"], 0.90)

                betti = T.betti_filtration_signature(prob_asc, valid)
                self.assertGreaterEqual(betti["max_b0"], 6)
                self.assertEqual(betti["max_b1"], 0)
                self.assertEqual(betti["integrated_b1"], 0.0)
                self.assertEqual(betti["loop_persistence_score"], 0.0)

                res = T.evaluate_transverse_fp_features(prob_asc, prob_desc, valid, VOXEL_UM)
                self.assertEqual(res["dominant_face"], "asc")
                self.assertLess(res["stroke_width"]["hi"]["weighted_component_skeleton_mm"], 0.45)
                self.assertEqual(res["composite"]["topology_score"], 0.0)
                self.assertLess(res["composite"]["stroke_consistency_score"], 0.30)
                self.assertLess(res["composite"]["discriminability_score"], 0.20)
                self.assertEqual(res["composite"]["regime_hint"], "single_face_dot_or_wide_like")

    def test_clear_three_way_separation(self) -> None:
        rng = np.random.default_rng(404)
        valid_720 = np.ones((720, 720), dtype=bool)
        valid_768 = np.ones((768, 768), dtype=bool)

        greek = T.evaluate_transverse_fp_features(
            _soft_prob(_synthetic_greek_glyphs((720, 720)), rng),
            _soft_prob(np.zeros((720, 720), dtype=bool), rng),
            valid_720,
            VOXEL_UM,
        )
        dots = T.evaluate_transverse_fp_features(
            _soft_prob(_synthetic_dot_row((720, 720), elongated=True), rng),
            _soft_prob(np.zeros((720, 720), dtype=bool), rng),
            valid_720,
            VOXEL_UM,
        )
        asc_c, desc_c = _synthetic_bilateral_clouds((768, 768))
        clouds = T.evaluate_transverse_fp_features(
            _soft_prob(asc_c, rng, peak=0.91, sigma=3.0),
            _soft_prob(desc_c, rng, peak=0.89, sigma=3.0),
            valid_768,
            VOXEL_UM,
        )

        g_score = greek["composite"]["discriminability_score"]
        d_score = dots["composite"]["discriminability_score"]
        c_score = clouds["composite"]["discriminability_score"]
        self.assertGreater(g_score - d_score, 0.55)
        self.assertGreater(g_score - c_score, 0.70)

        # Orthogonal axis checks:
        self.assertGreater(
            clouds["differential"]["dominant"]["bilateral_suppression_frac"]
            - greek["differential"]["dominant"]["bilateral_suppression_frac"],
            0.70,
        )
        self.assertGreater(
            clouds["stroke_width"]["hi"]["width_median_mm"]
            - greek["stroke_width"]["hi"]["width_median_mm"],
            1.2,
        )
        self.assertGreater(
            greek["betti"]["dominant"]["loop_persistence_score"]
            - dots["betti"]["dominant"]["loop_persistence_score"],
            0.75,
        )

    def test_border_and_invalid_mask_loops_excluded_from_b1(self) -> None:
        shape = (200, 200)
        yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        r = np.hypot(yy - 100.0, xx - 100.0)
        ring = ((r >= 40.0) & (r <= 58.0)).astype(np.float32) * 0.9

        # Intact valid_mask -> enclosed interior hole gives b1 == 1
        valid_full = np.ones(shape, dtype=bool)
        b_full = T.betti_filtration_signature(ring, valid_full)
        self.assertEqual(b_full["max_b1"], 1)

        # Invalid pixel cutout inside the ring -> hole touches ~valid_mask -> b1 == 0
        valid_cut = valid_full.copy()
        valid_cut[95:105, 95:105] = False
        b_cut = T.betti_filtration_signature(ring, valid_cut)
        self.assertEqual(b_cut["max_b1"], 0)

        # Ring touching image border (open semicircle against top border) -> b1 == 0
        r_edge = np.hypot(yy - 5.0, xx - 100.0)
        arc = ((r_edge >= 40.0) & (r_edge <= 58.0)).astype(np.float32) * 0.9
        b_edge = T.betti_filtration_signature(arc, valid_full)
        self.assertEqual(b_edge["max_b1"], 0)

    def test_edge_cases_empty_tiny_zeros_nans_infs_and_json_roundtrip(self) -> None:
        shape = (64, 64)
        zeros = np.zeros(shape, dtype=np.float32)
        empty_valid = np.zeros(shape, dtype=bool)
        full_valid = np.ones(shape, dtype=bool)

        # 1. Empty valid_mask
        d_empty = T.opposite_face_differential_map(zeros, zeros, empty_valid)
        self.assertEqual(d_empty["valid_px"], 0)
        self.assertEqual(d_empty["bilateral_suppression_frac"], 0.0)
        self.assertTrue(np.all(d_empty["delta_map"] == 0.0))
        self.assertTrue(np.all(d_empty["z_map"] == 0.0))

        sw_empty = T.stroke_width_consistency(empty_valid, VOXEL_UM)
        self.assertEqual(sw_empty["n_skeleton_px"], 0)
        self.assertIsNone(sw_empty["width_median_mm"])
        self.assertIsNone(sw_empty["width_cv"])
        self.assertEqual(sw_empty["narrow_stroke_frac"], 0.0)

        b_empty = T.betti_filtration_signature(zeros, empty_valid)
        self.assertEqual(b_empty["max_b0"], 0)
        self.assertEqual(b_empty["max_b1"], 0)
        self.assertEqual(b_empty["loop_persistence_score"], 0.0)

        ev_empty = T.evaluate_transverse_fp_features(zeros, zeros, empty_valid, VOXEL_UM)
        self.assertEqual(ev_empty["valid_px"], 0)
        self.assertEqual(ev_empty["composite"]["discriminability_score"], 0.0)
        self.assertEqual(ev_empty["composite"]["regime_hint"], "no_signal")

        # 2. All-zero probabilities on non-empty valid_mask
        ev_zero = T.evaluate_transverse_fp_features(zeros, zeros, full_valid, VOXEL_UM)
        self.assertEqual(ev_zero["valid_px"], shape[0] * shape[1])
        self.assertEqual(ev_zero["composite"]["discriminability_score"], 0.0)
        self.assertEqual(ev_zero["composite"]["regime_hint"], "no_signal")

        # 3. Tiny valid_mask (1 pixel and 5 pixels)
        tiny_valid = np.zeros(shape, dtype=bool)
        tiny_valid[10, 10] = True
        pa_tiny = zeros.copy()
        pa_tiny[10, 10] = 0.9
        d_tiny = T.opposite_face_differential_map(pa_tiny, zeros, tiny_valid)
        self.assertEqual(d_tiny["valid_px"], 1)
        self.assertTrue(np.all(np.isfinite(d_tiny["delta_map"])))
        self.assertTrue(np.all(np.isfinite(d_tiny["z_map"])))

        # 4. NaNs and Infs masked out safely
        pa_bad = np.full(shape, 0.05, dtype=np.float32)
        pb_bad = np.full(shape, 0.05, dtype=np.float32)
        pa_bad[0:8, 0:8] = np.nan
        pa_bad[8:16, 0:8] = np.inf
        pb_bad[16:24, 0:8] = -np.inf
        pa_bad[30:40, 30:55] = 0.9
        d_bad = T.opposite_face_differential_map(pa_bad, pb_bad, full_valid)
        self.assertEqual(d_bad["valid_px"], shape[0] * shape[1] - 3 * 64)
        self.assertTrue(np.all(np.isfinite(d_bad["delta_map"])))
        self.assertTrue(np.all(np.isfinite(d_bad["z_map"])))
        self.assertTrue(np.all(d_bad["delta_map"][0:24, 0:8] == 0.0))

        ev_bad = T.evaluate_transverse_fp_features(pa_bad, pb_bad, full_valid, VOXEL_UM)
        # 5. Strict JSON round-trip (allow_nan=False rejects NaN/Inf anywhere in payload)
        for payload in (ev_empty, ev_zero, ev_bad, sw_empty, b_empty):
            encoded = json.dumps(payload, allow_nan=False)
            decoded = json.loads(encoded)
            self.assertEqual(decoded, payload)

    def test_2x2_block_preserves_skeleton_and_input_validation(self) -> None:
        block = np.zeros((16, 16), dtype=bool)
        block[7:9, 7:9] = True
        sw = T.stroke_width_consistency(block, voxel_um=100.0)
        self.assertGreaterEqual(sw["n_skeleton_px"], 1)
        self.assertIsNotNone(sw["width_median_mm"])

        zeros = np.zeros((8, 8), dtype=np.float32)
        valid = np.ones((8, 8), dtype=bool)
        with self.assertRaises(ValueError):
            T.opposite_face_differential_map(zeros, zeros, valid, eps=0.0)
        with self.assertRaises(ValueError):
            T.opposite_face_differential_map(zeros, np.zeros((8, 7)), valid)
        with self.assertRaises(ValueError):
            T.stroke_width_consistency(valid, voxel_um=-1.0)
        with self.assertRaises(ValueError):
            T.betti_filtration_signature(zeros, valid, thresholds=())
        with self.assertRaises(ValueError):
            T.betti_filtration_signature(zeros, valid, min_area_px=0)


if __name__ == "__main__":
    unittest.main()
