"""Coordinate, interpolation and coverage regression tests; no network/model."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampling import NativeMosaic, build_mosaic, context_support, map_prediction, sample_input

ARMS = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0)]


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(90210)
        cls.raw = cls.rng.integers(0, 256, (420, 440, 16), dtype=np.uint8)
        cls.mosaic = NativeMosaic(cls.raw, np.ones(cls.raw.shape[:2], dtype=bool))

    def test_identity_matches_existing_raw_crop_across_tile_boundaries(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio_control"))
        from run_public_control import raw_crop
        tiles = self.rng.integers(0, 256, (4, 512, 512, 16), dtype=np.uint8)
        origins = np.array([[0, 0], [0, 512], [512, 0], [512, 512]], dtype=np.int32)
        mosaic = build_mosaic(tiles, origins)
        for y, x in [(0, 0), (450, 470), (896, 896)]:
            got = sample_input(mosaic, y, x)
            self.assertTrue(np.array_equal(got, raw_crop(tiles, origins, y, x).astype(np.float32)))
            self.assertEqual(got.dtype, np.float32)
            self.assertTrue(got.flags.c_contiguous)

    def test_all_arms_match_independent_scipy_reference(self):
        y, x = 135, 145
        for sx, sy in ARMS:
            with self.subTest(sx=sx, sy=sy):
                reference = self.raw.astype(np.float64)
                if sy < 1:
                    reference = gaussian_filter1d(reference, 0.5, axis=0, truncate=4)
                if sx < 1:
                    reference = gaussian_filter1d(reference, 0.5, axis=1, truncate=4)
                gy, gx = np.meshgrid(y + 63.5 + (np.arange(128) - 63.5) / sy,
                                     x + 63.5 + (np.arange(128) - 63.5) / sx, indexing="ij")
                expected = np.stack([map_coordinates(reference[..., d], [gy, gx], order=1,
                                                     mode="constant", cval=np.nan, prefilter=False)
                                     for d in range(16)]).astype(np.float32)
                got = sample_input(self.mosaic, y, x, sx=sx, sy=sy)
                np.testing.assert_allclose(got, expected, rtol=0, atol=2e-5)
                self.assertEqual(got.shape, (16, 128, 128))
                self.assertTrue(got.flags.c_contiguous)

    def test_constant_fields_and_depth_order_preserved(self):
        raw = np.broadcast_to(np.arange(16, dtype=np.uint8) * 15, (400, 400, 16)).copy()
        mosaic = NativeMosaic(raw, np.ones((400, 400), dtype=bool))
        for sx, sy in ARMS:
            got = sample_input(mosaic, 136, 136, sx=sx, sy=sy)
            expected = np.broadcast_to(np.arange(16, dtype=np.float32)[:, None, None] * 15, got.shape)
            self.assertTrue(np.array_equal(got, expected))
        self.assertEqual(float(got.max()), 225.0)  # Sampling has not clipped to200 or normalized.

    def test_half_scale_antialiases_period_four_input(self):
        raw = np.broadcast_to(np.array([0, 0, 255, 255] * 110, np.uint8)[None, :, None], (420, 440, 16)).copy()
        mosaic = NativeMosaic(raw, np.ones(raw.shape[:2], dtype=bool))
        got = sample_input(mosaic, 135, 145, sx=0.5)[0]
        self.assertGreater(float(got.min()), 20.0)
        self.assertLess(float(got.max()), 235.0)
        np.testing.assert_allclose(got.mean(), 127.5, rtol=0, atol=1e-5)

    def test_explicit_gaussian_halo_and_json_support(self):
        support = context_support(self.mosaic, 135, 145, sx=0.5)
        self.assertTrue(support["complete"])
        self.assertEqual(support["source_bounds_yx"], [[135, 263], [79, 339]])
        self.assertEqual(support["aa_sigma_yx"], [0.0, 0.5])
        self.assertEqual(support["aa_radius_yx"], [0, 2])
        self.assertEqual(support["required_source_pixels"], 128 * 260)
        json.dumps(support, allow_nan=False)
        available = self.mosaic.available_yx.copy()
        available[135, 79] = False  # Gaussian halo, outside unfiltered bilinear support.
        mosaic = NativeMosaic(self.raw, available)
        self.assertTrue(context_support(mosaic, 135, 145)["complete"])
        self.assertEqual(context_support(mosaic, 135, 145, sx=0.5)["missing_source_pixels"], 1)
        with self.assertRaisesRegex(ValueError, "Incomplete native context"):
            sample_input(mosaic, 135, 145, sx=0.5)

    def test_missing_tiles_are_distinct_from_real_zeros(self):
        tiles = np.zeros((2, 128, 128, 16), dtype=np.uint8)
        mosaic = build_mosaic(tiles, np.array([[0, 0], [0, 256]]))
        self.assertTrue(context_support(mosaic, 0, 0)["complete"])
        self.assertEqual(context_support(mosaic, 0, 128)["missing_source_pixels"], 128 * 128)
        with self.assertRaisesRegex(ValueError, "Incomplete native context"):
            sample_input(mosaic, 0, 128)
        self.assertTrue(np.array_equal(sample_input(mosaic, 0, 256), np.zeros((16, 128, 128), np.float32)))

    def test_mosaic_boundary_is_not_padded(self):
        self.assertTrue(context_support(self.mosaic, 292, 312)["complete"])
        for y, x, sx, sy in [(0, 0, 0.5, 1), (0, 0, 1, 0.5), (-1, 0, 1, 1), (293, 312, 1, 1)]:
            self.assertFalse(context_support(self.mosaic, y, x, sx=sx, sy=sy)["complete"])
            with self.assertRaises(ValueError):
                sample_input(self.mosaic, y, x, sx=sx, sy=sy)

    def test_prediction_identity_is_exact_and_contiguous(self):
        p = self.rng.random((128, 128), dtype=np.float32)
        got = map_prediction(p)
        self.assertTrue(np.array_equal(got, p[48:80, 48:80]))
        self.assertTrue(got.flags.c_contiguous)

    def test_prediction_inverse_mapping_preserves_physical_affine_field(self):
        physical_y, physical_x = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
        expected = 0.5 + physical_y * 0.001 + physical_x * 0.002
        yy, xx = np.meshgrid(np.arange(128) - 63.5, np.arange(128) - 63.5, indexing="ij")
        for sx, sy in ARMS:
            probability = 0.5 + (yy / sy) * 0.001 + (xx / sx) * 0.002
            got = map_prediction(probability, sx=sx, sy=sy)
            np.testing.assert_allclose(got, expected, rtol=0, atol=3e-8)
            self.assertEqual(got.dtype, np.float32)

    def test_probability_support_checks_full_positive_weight_stencil(self):
        p = np.full((128, 128), 0.5, dtype=np.float32)
        available = np.ones((128, 128), dtype=bool)
        available[48, 55] = False  # x_half first query55.75 needs55 and56.
        with self.assertRaisesRegex(ValueError, "Incomplete probability"):
            map_prediction(p, sx=0.5, available_hw=available)
        available[:] = False
        available[48:80, 48:80] = True
        p[80, 80] = np.nan  # Zero-weight outside identity stencil is not read.
        self.assertTrue(np.isfinite(map_prediction(p, available_hw=available)).all())
        p[48, 48] = np.nan
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            map_prediction(p, available_hw=available)

    def test_invalid_inputs_fail_explicitly(self):
        tiles = np.zeros((2, 128, 128, 16), dtype=np.uint8)
        for origins in [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, -1]]), np.array([[0., 0.], [128., 0.]])]:
            with self.assertRaises(ValueError):
                build_mosaic(tiles, origins)
        for scale in [0, -1, 1.5, np.nan, np.inf, True]:
            with self.assertRaises(ValueError):
                context_support(self.mosaic, 135, 145, sx=scale)
        with self.assertRaises(ValueError):
            sample_input(self.mosaic, 135.0, 145)
        with self.assertRaises(ValueError):
            map_prediction(np.zeros((127, 128)))
        with self.assertRaises(ValueError):
            map_prediction(np.full((128, 128), 2.0))


if __name__ == "__main__":
    unittest.main()
