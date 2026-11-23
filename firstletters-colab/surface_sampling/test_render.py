"""Generated-only checks of conservative footprint rendering.

No manuscript, scan download, optical label, or real data array is used here.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

import render as renderer


class Plane:
    """Analytical surface oracle with optional continuous rectangular slit."""

    def __init__(self, base=(4.0, 4.0, 4.0), slit=None, warp=0.0, normal=(0, 0, 1)):
        self.base = np.asarray(base, dtype=np.float64)
        self.slit = slit
        self.warp = warp
        self.normal = np.asarray(normal, dtype=np.float64)
        self.sample_shapes = []

    def footprint_support(self, origin_uv, step_uv, shape_yx, query_mask=None):
        height, width = shape_yx
        origin = np.asarray(origin_uv)
        step = np.asarray(step_uv)
        u = origin[0] + np.arange(width) * step[0]
        valid = np.ones((height, width), dtype=bool)
        if self.slit is not None:
            lo, hi = self.slit
            crosses = (u - abs(step[0]) / 2 < hi) & (u + abs(step[0]) / 2 > lo)
            valid[:, crosses] = False
        if query_mask is not None:
            valid &= query_mask
        return {"valid": valid}

    def sample_grid(self, origin_uv, step_uv, shape_yx, query_mask=None):
        height, width = shape_yx
        self.sample_shapes.append(shape_yx)
        origin = np.asarray(origin_uv)
        step = np.asarray(step_uv)
        u, v = np.meshgrid(
            origin[0] + np.arange(width) * step[0],
            origin[1] + np.arange(height) * step[1],
        )
        xyz = np.stack((u + self.warp * u * u, v, np.zeros_like(u)), axis=-1) + self.base
        valid = np.ones(shape_yx, dtype=bool)
        if self.slit is not None:
            valid &= ~((u > self.slit[0]) & (u < self.slit[1]))
        if query_mask is not None:
            valid &= query_mask
        normals = np.broadcast_to(self.normal, xyz.shape).copy()
        xyz[~valid] = np.nan
        normals[~valid] = np.nan
        return {"xyz": xyz, "normals": normals, "valid": valid}


class FootprintRendererTests(unittest.TestCase):
    def test_original_triangles_through_footprint_to_affine_ct(self):
        from surface import OriginalSurface

        volume_origin = np.asarray([-10.0, 20.0, 30.0])
        uv = np.asarray([[-0.5, -0.5], [3.5, -0.5], [3.5, 2.5], [-0.5, 2.5]])
        xyz = np.column_stack((4 + uv[:, 0], 4 + uv[:, 1], 4 + 0.2 * uv[:, 0] + 0.1 * uv[:, 1]))
        xyz += volume_origin
        surface = OriginalSurface(uv, xyz, np.asarray([[0, 1, 2], [0, 2, 3]]))
        normal = np.asarray([-0.2, -0.1, 1.0]) / np.sqrt(1.05)
        z, y, x = np.indices((12, 12, 12), dtype=np.float64)
        volume = (x + volume_origin[0]) + 2 * (y + volume_origin[1]) + 3 * (z + volume_origin[2])
        u, v = np.meshgrid(np.arange(4), np.arange(3))
        centres = np.stack((4 + u, 4 + v, 4 + 0.2 * u + 0.1 * v), axis=-1) + volume_origin
        expected = np.stack([
            np.sum((centres + depth * normal) * [1, 2, 3], axis=-1)
            for depth in [-0.75, 0.0, 0.75]
        ])
        for n in (1, 4, 8):
            stack, valid, _ = renderer.render(
                volume, volume_origin, surface, (0, 0), (1, 1), (3, 4), [-0.75, 0.0, 0.75], n,
            )
            self.assertTrue(valid.all())
            np.testing.assert_allclose(stack, expected, rtol=0, atol=8e-6)

    def test_constant_and_affine_global_volume_on_plane(self):
        volume_origin = np.asarray([-17.0, 23.0, 101.0])
        z, y, x = np.indices((16, 16, 16), dtype=np.float64)
        x, y, z = x + volume_origin[0], y + volume_origin[1], z + volume_origin[2]
        origin_uv, step_uv, shape = (-0.4, 0.2), (0.8, 0.6), (3, 4)
        u, v = np.meshgrid(
            origin_uv[0] + np.arange(shape[1]) * step_uv[0],
            origin_uv[1] + np.arange(shape[0]) * step_uv[1],
        )
        base = volume_origin + np.asarray([5.0, 6.0, 7.0])
        depths = [-1.25, 0.0, 1.75]
        for dtype in (np.float32, np.float64):
            for n in (1, 4, 8):
                for coefficients in ((0, 0, 0, 13.25), (2.0, -3.0, 0.5, 7.0)):
                    with self.subTest(dtype=dtype, n=n, coefficients=coefficients):
                        a, b, c, offset = coefficients
                        volume = (a * x + b * y + c * z + offset).astype(dtype)
                        expected = np.stack([
                            a * (base[0] + u) + b * (base[1] + v)
                            + c * (base[2] + d) + offset
                            for d in depths
                        ])
                        stack, valid, diagnostics = renderer.render(
                            volume, volume_origin, Plane(base), origin_uv,
                            step_uv, shape, depths, n,
                        )
                        self.assertTrue(valid.all())
                        self.assertEqual(stack.dtype, np.dtype("float32"))
                        np.testing.assert_allclose(stack, expected, rtol=1e-7, atol=3e-6)
                        self.assertEqual(diagnostics["valid_output_pixels"], 12)
                        json.dumps(diagnostics, allow_nan=False)

    def test_depths_remain_separate_and_ordered(self):
        z, _, _ = np.indices((10, 10, 10), dtype=np.float64)
        volume = z * z
        for n in (1, 4, 8):
            stack, valid, _ = renderer.render(
                volume, (0, 0, 0), Plane(), (0, 0), (1, 1), (1, 1),
                [1, -1, 0, 1], n,
            )
            self.assertTrue(valid[0, 0])
            np.testing.assert_array_equal(stack[:, 0, 0], [25, 9, 16, 25])

    def test_continuous_slit_missed_by_midpoints_rejects_every_arm(self):
        # The slit misses N=1, N=4, and N=8 midpoint abscissae. The exact
        # footprint oracle, not a lucky subquery, must catch it.
        surface = Plane(slit=(0.19, 0.20))
        for n in (1, 4, 8):
            mids = (np.arange(n) + 0.5) / n - 0.5
            self.assertFalse(((mids > 0.19) & (mids < 0.20)).any())
            stack, valid, diagnostics = renderer.render(
                np.ones((10, 10, 10), dtype=np.float32), (0, 0, 0), surface,
                (0, 0), (1, 1), (1, 2), [-1, 0, 1], n,
            )
            np.testing.assert_array_equal(valid, [[False, True]])
            np.testing.assert_array_equal(stack[:, 0, 0], [0, 0, 0])
            np.testing.assert_array_equal(stack[:, 0, 1], [1, 1, 1])
            self.assertEqual(diagnostics["rejected_footprint_pixels"], 1)

    def test_geometry_subquery_failure_rejects_whole_pixel(self):
        class UnreliablePlane(Plane):
            def sample_grid(self, origin_uv, step_uv, shape_yx, query_mask=None):
                result = super().sample_grid(origin_uv, step_uv, shape_yx, query_mask)
                # A provider failure despite complete coverage, at one
                # subquery of the first output pixel only.
                result["valid"][0, 0] = False
                result["xyz"][0, 0] = np.nan
                return result

        stack, valid, diagnostics = renderer.render(
            np.full((10, 10, 10), 5.0, dtype=np.float32), (0, 0, 0), UnreliablePlane(),
            (0, 0), (1, 1), (1, 2), [-1, 0, 1], 4,
        )
        np.testing.assert_array_equal(valid, [[False, True]])
        np.testing.assert_array_equal(stack[:, 0, 0], [0, 0, 0])
        np.testing.assert_array_equal(stack[:, 0, 1], [5, 5, 5])
        self.assertEqual(diagnostics["invalid_geometry_subqueries"], 1)

    def test_full_ct_support_no_clamping_and_common_depth_mask(self):
        volume = np.ones((6, 6, 6), dtype=np.float32)
        original = renderer._trilinear_valid
        seen = []

        def checked_interpolation(source, coords):
            seen.append(coords.copy())
            self.assertTrue(np.isfinite(coords).all())
            self.assertTrue((coords >= 0).all())
            self.assertTrue((coords < 5).all())
            return original(source, coords)

        with patch.object(renderer, "_trilinear_valid", side_effect=checked_interpolation):
            # The centre is in bounds; some N=4 footprint subqueries are not.
            _, valid, _ = renderer.render(
                volume, (0, 0, 0), Plane(base=(0.1, 2, 2)),
                (0, 0), (1, 1), (1, 1), [0], 1,
            )
            self.assertTrue(valid[0, 0])
            stack, valid, _ = renderer.render(
                volume, (0, 0, 0), Plane(base=(0.1, 2, 2)),
                (0, 0), (1, 1), (1, 1), [0], 4,
            )
            self.assertFalse(valid[0, 0])
            self.assertEqual(float(stack[0, 0, 0]), 0)
            # A final-voxel query is unsupported, even with zero +1 weight.
            _, valid, _ = renderer.render(
                volume, (0, 0, 0), Plane(base=(5, 2, 2)),
                (0, 0), (1, 1), (1, 1), [0], 1,
            )
            self.assertFalse(valid[0, 0])
            # Only the third depth leaves the CT array: reject *all* depths.
            stack, valid, _ = renderer.render(
                volume, (0, 0, 0), Plane(base=(2, 2, 4)),
                (0, 0), (1, 1), (1, 1), [-1, 0, 1], 4,
            )
            self.assertFalse(valid[0, 0])
            np.testing.assert_array_equal(stack, np.zeros((3, 1, 1)))
        self.assertEqual(len(seen), 1, "unsupported pixels must never reach interpolation")
        with self.assertRaises(ValueError):
            original(volume, np.asarray([[5.0, 2.0, 2.0]]))

    def test_warped_parameterisation_integrates_mapped_uv_footprint(self):
        _, _, x = np.indices((12, 12, 12), dtype=np.float64)
        alpha, u0, du = 0.2, 0.3, 1.6
        exact_integral = 4 + u0 + alpha * (u0 * u0 + du * du / 12)
        errors = []
        for n in (1, 4, 8):
            stack, valid, _ = renderer.render(
                x, (0, 0, 0), Plane(warp=alpha),
                (u0, 0), (du, 1), (1, 1), [0], n,
            )
            expected = 4 + u0 + alpha * (
                u0 * u0 + du * du * (n * n - 1) / (12 * n * n)
            )
            self.assertTrue(valid[0, 0])
            self.assertAlmostEqual(float(stack[0, 0, 0]), expected, places=6)
            errors.append(abs(float(stack[0, 0, 0]) - exact_integral))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_tiling_stays_bounded_and_has_no_seams(self):
        # 65x65 N=8 spans nine tiles, including unequal last rows/columns.
        z, y, x = np.indices((8, 80, 80), dtype=np.float64)
        surface = Plane(base=(2, 2, 3))
        stack, valid, diagnostics = renderer.render(
            x + 2 * y + 3 * z, (0, 0, 0), surface,
            (0, 0), (1, 1), (65, 65), [-0.25, 0.25], 8,
        )
        u, v = np.meshgrid(np.arange(65), np.arange(65))
        expected = np.stack([2 + u + 2 * (2 + v) + 3 * (3 + d) for d in [-0.25, 0.25]])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(stack, expected, rtol=0, atol=1e-5)
        self.assertEqual(diagnostics["tile_count"], 9)
        self.assertLessEqual(diagnostics["max_tile_subqueries"], 65_536)
        self.assertLessEqual(diagnostics["max_vertex_reads_per_interpolation_call"], 1_000_000)
        self.assertTrue(all(h * w <= 65_536 for h, w in surface.sample_shapes))

    def test_missing_volume_and_nonunit_normals_are_not_filled(self):
        volume = np.ones((10, 10, 10), dtype=np.float64)
        volume[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            renderer.render(volume, (0, 0, 0), Plane(), (0, 0), (1, 1), (1, 1), [0])
        volume[0, 0, 0] = 1
        stack, valid, diagnostics = renderer.render(
            volume, (0, 0, 0), Plane(normal=(0, 0, 2)),
            (0, 0), (1, 1), (1, 1), [0], 4,
        )
        self.assertFalse(valid[0, 0])
        self.assertEqual(float(stack[0, 0, 0]), 0)
        self.assertEqual(diagnostics["nonunit_normal_subqueries"], 16)

    def test_contract_rejects_unsupported_quadrature_and_output_overflow(self):
        volume = np.ones((10, 10, 10), dtype=np.float64)
        for n in (0, 2, 3, 5, 16, True, 4.0):
            with self.subTest(n=n), self.assertRaises(ValueError):
                renderer.render(volume, (0, 0, 0), Plane(), (0, 0), (1, 1), (1, 1), [0], n)
        class PointOnlySurface:
            sample_grid = Plane().sample_grid

        with self.assertRaisesRegex(ValueError, "footprint_support"):
            renderer.render(volume, (0, 0, 0), PointOnlySurface(), (0, 0), (1, 1), (1, 1), [0])
        with self.assertRaises(FloatingPointError):
            renderer.render(
                volume * 1e100, (0, 0, 0), Plane(), (0, 0), (1, 1), (1, 1), [0]
            )


if __name__ == "__main__":
    unittest.main()
