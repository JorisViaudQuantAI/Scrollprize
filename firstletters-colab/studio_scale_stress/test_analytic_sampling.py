"""Independent analytic checks of the fixed public-control scale sampler.

Uses coordinate ramps, constants and a period-four signal with a closed-form
Gaussian response. No learned model, source dataset or sampler helper is used
to calculate the expected values.
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = Path(os.environ.get("SCROLL_SAMPLING_PATH", Path(__file__).with_name("sampling.py")))
SPEC = importlib.util.spec_from_file_location("independently_checked_sampling", PATH)
sampling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampling
SPEC.loader.exec_module(sampling)
ARMS = ((1, 1), (.5, 1), (2, 1), (1, .5), (1, 2))


class AnalyticSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yy, xx = np.mgrid[:512, :512]
        cls.tile = np.empty((512, 512, 16), np.uint8)
        cls.tile[..., 0] = np.clip(xx - 128, 0, 255)
        cls.tile[..., 1] = np.clip(yy - 128, 0, 255)
        cls.tile[..., 2] = (xx % 4 >= 2) * 255
        cls.tile[..., 3] = (yy % 4 >= 2) * 255
        for d in range(4, 16):
            cls.tile[..., d] = (d - 4) * 23
        cls.mosaic = sampling.build_mosaic(cls.tile[None], np.array([[0, 0]], np.int64))

    def test_identity_exactly_preserves_every_native_value(self):
        got = sampling.sample_input(self.mosaic, 192, 192, sx=1, sy=1)
        expected = self.tile[192:320, 192:320].transpose(2, 0, 1).astype(np.float32)
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_source_pixel_centers_on_independent_coordinate_ramps(self):
        u = np.arange(128, dtype=np.float64)
        for sx, sy in ARMS:
            with self.subTest(sx=sx, sy=sy):
                got = sampling.sample_input(self.mosaic, 192, 192, sx=sx, sy=sy)
                x = 192 + 63.5 + (u - 63.5) / sx - 128
                y = 192 + 63.5 + (u - 63.5) / sy - 128
                # Stay away from the deliberately clipped ends of the uint8 ramp.
                np.testing.assert_allclose(got[0, 4:-4, 4:-4], np.broadcast_to(x, (128, 128))[4:-4, 4:-4], atol=2e-5, rtol=0)
                np.testing.assert_allclose(got[1, 4:-4, 4:-4], np.broadcast_to(y[:, None], (128, 128))[4:-4, 4:-4], atol=2e-5, rtol=0)
                for d in range(4, 16):
                    np.testing.assert_allclose(got[d], (d-4)*23, atol=2e-5, rtol=0)

    def test_halfscale_uses_declared_gaussian_not_only_subsampling(self):
        offsets = np.arange(-2, 3, dtype=np.float64)
        weights = np.exp(-(offsets**2)/(2*.5**2))
        weights /= weights.sum()
        low = 255 * (weights[1] + 2*weights[0])
        expected = np.where(np.arange(128) % 2 == 0, low, 255-low)
        xhalf = sampling.sample_input(self.mosaic, 192, 192, sx=.5, sy=1)
        yhalf = sampling.sample_input(self.mosaic, 192, 192, sx=1, sy=.5)
        np.testing.assert_allclose(xhalf[2], np.broadcast_to(expected, (128, 128)), atol=3e-5, rtol=0)
        np.testing.assert_allclose(yhalf[3], np.broadcast_to(expected[:, None], (128, 128)), atol=3e-5, rtol=0)

    def test_readback_preserves_fixed_physical_centers(self):
        yy, xx = np.mgrid[:128, :128]
        probability = ((10 + 3*yy + 5*xx)/2000).astype(np.float32)
        core = np.arange(32, dtype=np.float64) - 15.5
        for sx, sy in ARMS:
            got = sampling.map_prediction(probability, sx=sx, sy=sy)
            expected = (10 + 3*(63.5+sy*core[:, None]) + 5*(63.5+sx*core[None, :]))/2000
            np.testing.assert_allclose(got, expected, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(sampling.map_prediction(probability, sx=1, sy=1), probability[48:80, 48:80])

    def test_missing_antialias_halo_rejected_before_value_sampling(self):
        # Without the radius-2 Gaussian halo, halfscale queries only rows256.5..510.5.
        self.assertTrue(sampling.context_support(self.mosaic, 320, 192, sx=1, sy=1)["complete"])
        self.assertFalse(sampling.context_support(self.mosaic, 320, 192, sx=1, sy=.5)["complete"])
        with self.assertRaises(ValueError):
            sampling.sample_input(self.mosaic, 320, 192, sx=1, sy=.5)

    def test_readback_checks_only_positive_weight_corners(self):
        p = np.ones((128, 128), np.float32) * .7
        available = np.ones_like(p, dtype=bool)
        available[80, 80] = False  # Outside all identity readback stencils.
        sampling.map_prediction(p, sx=1, sy=1, available_hw=available)
        available[48, 95] = False  # Positive-weight corner of the rightmost x-double cell.
        with self.assertRaises(ValueError):
            sampling.map_prediction(p, sx=2, sy=1, available_hw=available)


if __name__ == "__main__":
    unittest.main()
