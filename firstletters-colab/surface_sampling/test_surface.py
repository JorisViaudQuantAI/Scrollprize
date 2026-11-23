"""Generated source-geometry fixtures only; no manuscript data or downloads."""
import unittest

import numpy as np

from surface import (
    AMBIGUOUS_OVERLAP,
    DEGENERATE_UV_COVERAGE,
    DEGENERATE_XYZ_COVERAGE,
    INCONSISTENT_SHARED_BOUNDARY,
    NEGATIVE_UV_COVERAGE,
    NOT_REQUESTED,
    NO_SOURCE_COVERAGE,
    OriginalSurface,
    _topology,
)


class OriginalSurfaceTests(unittest.TestCase):
    J = np.array([[2., 1.], [0., 3.], [.5, -1.]])
    translation = np.array([1., -2., 7.])

    @classmethod
    def physical(cls, uv):
        return np.asarray(uv) @ cls.J.T + cls.translation

    @classmethod
    def square(cls, size=4.):
        uv = np.array([[0., 0.], [size, 0.], [size, size], [0., size]])
        return uv, cls.physical(uv), np.array([[0, 1, 2], [0, 2, 3]])

    @classmethod
    def slit(cls):
        uv = np.array([[0., 0.], [1.2, 0.], [1.2, 3.], [0., 3.],
                       [1.3, 0.], [3., 0.], [3., 3.], [1.3, 3.]])
        return uv, cls.physical(uv), np.array([[0, 1, 2], [0, 2, 3],
                                               [4, 5, 6], [4, 6, 7]])

    def assert_invalid_sentinels(self, result):
        invalid = ~result["valid"]
        np.testing.assert_array_equal(result["source_triangle_id"][invalid], -1)
        for key in ("xyz", "normals", "jacobian", "barycentric_weights"):
            self.assertTrue(np.isnan(result[key][invalid]).all(), key)

    def test_fractional_anisotropic_affine_plane_normal_and_original_uv_jacobian(self):
        uv, xyz, faces = self.square()
        result = OriginalSurface(uv, xyz, faces).sample_grid(
            (.125, .25), (.375, .625), (6, 10))
        yy, xx = np.mgrid[:6, :10]
        queries = np.stack((.125 + .375 * xx, .25 + .625 * yy), axis=-1)
        expected = self.physical(queries)
        self.assertTrue(result["valid"].all())
        np.testing.assert_allclose(result["xyz"], expected, atol=1e-12, rtol=0)
        normal = np.cross(self.J[:, 0], self.J[:, 1])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(result["normals"], np.broadcast_to(normal, (6, 10, 3)), atol=1e-14)
        np.testing.assert_allclose(result["jacobian"], np.broadcast_to(self.J, (6, 10, 3, 2)), atol=1e-14)
        owners, weights = result["source_triangle_id"], result["barycentric_weights"]
        recovered = (xyz[faces[owners]] * weights[..., None]).sum(axis=-2)
        np.testing.assert_allclose(recovered, result["xyz"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(weights.sum(axis=-1), 1., atol=1e-15)

    def test_bent_shared_edge_uses_smallest_global_face_after_bbox_crop(self):
        uv, xyz, faces = self.square()
        xyz[3, 2] += 2.
        remote_uv = np.array([[20., 20.], [21., 20.], [20., 21.]])
        uv = np.vstack((remote_uv, uv))
        xyz = np.vstack((self.physical(remote_uv), xyz))
        faces = np.vstack(([0, 1, 2], faces + 3))
        result = OriginalSurface(uv, xyz, faces).sample_grid((.25, .25), (.5, .5), (8, 8))
        self.assertTrue(result["valid"].all())
        self.assertFalse(result["ambiguity_mask"].any())
        np.testing.assert_array_equal(np.diag(result["coverage_count"]), 2)
        np.testing.assert_array_equal(np.diag(result["source_triangle_id"]), 1)
        self.assertEqual(result["counts"]["candidate_source_triangles"], 2)
        first = xyz[faces[1]]
        expected_normal = np.cross(first[1] - first[0], first[2] - first[0])
        expected_normal /= np.linalg.norm(expected_normal)
        np.testing.assert_allclose(result["normals"][0, 0], expected_normal, atol=1e-14)
        self.assertFalse(np.allclose(result["normals"][0, 0], result["normals"][7, 0]))

    def test_fractional_slit_is_not_bridged(self):
        surface = OriginalSurface(*self.slit())
        result = surface.sample_grid((.025, .125), (.05, .25), (12, 60))
        u = .025 + .05 * np.arange(60)
        gap = (u > 1.2) & (u < 1.3)
        self.assertEqual(int(gap.sum()), 2)
        self.assertTrue(result["valid"][:, ~gap].all())
        self.assertFalse(result["valid"][:, gap].any())
        np.testing.assert_array_equal(result["coverage_count"][:, gap], 0)
        np.testing.assert_array_equal(result["reason_bits"][:, gap], NO_SOURCE_COVERAGE)
        self.assert_invalid_sentinels(result)
        # A coarser set of valid POINTS says nothing about the intervening slit.
        coarse = surface.sample_grid((.5, .5), (1., 1.), (3, 3))
        self.assertTrue(coarse["valid"].all())

    def test_fractional_internal_hole_in_connected_original_mesh(self):
        uv = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.],
                       [1.375, 1.375], [1.625, 1.375],
                       [1.625, 1.625], [1.375, 1.625]])
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            faces.extend(([i, j, j + 4], [i, j + 4, i + 4]))
        result = OriginalSurface(uv, self.physical(uv), np.array(faces)).sample_grid(
            (.0625, .0625), (.125, .125), (32, 32))
        yy, xx = np.mgrid[:32, :32]
        u, v = .0625 + .125 * xx, .0625 + .125 * yy
        hole = (u > 1.375) & (u < 1.625) & (v > 1.375) & (v < 1.625)
        self.assertEqual(int(hole.sum()), 4)
        np.testing.assert_array_equal(result["valid"], ~hole)
        np.testing.assert_array_equal(result["coverage_count"][hole], 0)
        self.assert_invalid_sentinels(result)

    def test_negative_face_poisons_positive_coverage(self):
        uv, xyz, faces = self.square()
        faces = np.vstack((faces, [0, 2, 1]))
        result = OriginalSurface(uv, xyz, faces).sample_grid((.25, .25), (.5, .5), (8, 8))
        yy, xx = np.mgrid[:8, :8]
        poisoned = xx >= yy
        np.testing.assert_array_equal(result["valid"], ~poisoned)
        self.assertTrue(((result["reason_bits"][poisoned] & NEGATIVE_UV_COVERAGE) != 0).all())
        self.assertFalse(result["ambiguity_mask"].any())
        self.assert_invalid_sentinels(result)

    def test_coincident_nonadjacent_positive_faces_remain_ambiguous(self):
        uv, xyz, faces = self.square()
        result = OriginalSurface(np.vstack((uv, uv)), np.vstack((xyz, xyz)),
                                 np.vstack((faces, faces + 4))).sample_grid(
                                     (.25, .25), (.5, .5), (8, 8))
        self.assertFalse(result["valid"].any())
        self.assertTrue(result["ambiguity_mask"].all())
        self.assertTrue(((result["reason_bits"] & AMBIGUOUS_OVERLAP) != 0).all())
        self.assert_invalid_sentinels(result)

    def test_degenerate_distance_tolerance_stays_in_original_uv_units(self):
        uv, xyz, faces = self.square()
        line = np.array([[.5, 0.], [.5, 2.], [.5, 1.]])
        # UV degeneracy need not imply physical degeneracy.
        line_xyz = np.array([[0., 0., 0.], [0., 2., 0.], [1., 1., 1.]])
        surface = OriginalSurface(np.vstack((uv, line)), np.vstack((xyz, line_xyz)),
                                  np.vstack((faces, [4, 5, 6])))
        result = surface.sample_grid((.5 + .25e-9, .75), (.5e-9, 2.), (1, 4))
        np.testing.assert_array_equal(result["valid"], [[False, False, True, True]])
        np.testing.assert_array_equal(result["reason_bits"][0, :2], DEGENERATE_UV_COVERAGE)
        np.testing.assert_array_equal(result["coverage_count"][0, :2], 2)
        self.assert_invalid_sentinels(result)

    def test_uv_area_epsilon_stays_in_original_uv_units(self):
        uv = np.array([[.25, .25], [.25 + 5e-13, .25], [.25, .75]])
        xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        result = OriginalSurface(uv, xyz, np.array([[0, 1, 2]])).sample_grid(
            (.25 + 1e-13, .375), (1e-13, .125), (2, 2))
        self.assertFalse(result["valid"].any())
        np.testing.assert_array_equal(result["reason_bits"], DEGENERATE_UV_COVERAGE)
        self.assert_invalid_sentinels(result)

    def test_zero_physical_area_poisons_only_covered_queries(self):
        uv, xyz, faces = self.square()
        zero_uv = np.array([[0., 0.], [4., 0.], [0., 4.]])
        zero_xyz = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
        result = OriginalSurface(np.vstack((uv, zero_uv)), np.vstack((xyz, zero_xyz)),
                                 np.vstack((faces, [4, 5, 6]))).sample_grid(
                                     (.25, .25), (.5, .5), (8, 8))
        yy, xx = np.mgrid[:8, :8]
        poison = (.25 + .5 * xx) + (.25 + .5 * yy) <= 4.
        np.testing.assert_array_equal(result["valid"], ~poison)
        np.testing.assert_array_equal(result["reason_bits"][poison], DEGENERATE_XYZ_COVERAGE)
        self.assert_invalid_sentinels(result)

    def test_inconsistent_tolerance_shared_boundary_is_rejected(self):
        uv = np.array([[1e-10, 0.], [1e-10, 2.], [-1., 0.], [1., 0.]])
        xyz = np.column_stack((uv, [0., 0., 1e6, 0.]))
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        result = OriginalSurface(uv, xyz, faces).sample_grid((0., .5), (.25, .5), (3, 2))
        self.assertFalse(result["valid"][:, 0].any())
        self.assertTrue(((result["reason_bits"][:, 0] & INCONSISTENT_SHARED_BOUNDARY) != 0).all())
        self.assert_invalid_sentinels(result)

    def test_unit_grid_matches_frozen_rasterizer(self):
        uv, xyz, faces = self.square()
        faces = np.vstack((faces, [0, 2, 1]))
        mask = (np.indices((6, 7)).sum(axis=0) % 3) != 0
        expected = _topology.rasterize_triangles(
            uv, xyz, faces, (6, 7), origin_xy=(-.25, -.5), query_mask=mask)
        actual = OriginalSurface(uv, xyz, faces).sample_grid(
            (-.25, -.5), (1., 1.), (6, 7), query_mask=mask)
        for key in ("valid", "source_triangle_id", "reason_bits", "coverage_count", "ambiguity_mask"):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for key in ("xyz", "barycentric_weights"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-14, rtol=0, err_msg=key)
        np.testing.assert_array_equal(actual["reason_bits"][~mask], NOT_REQUESTED)
        np.testing.assert_array_equal(actual["coverage_count"][~mask], 0)

    def test_tiles_and_working_blocks_match_whole_arbitrary_grid(self):
        uv, xyz, faces = self.slit()
        extra_uv = np.array([[2.3, .4], [2.9, .4], [2.6, 1.2]])
        uv, xyz = np.vstack((uv, extra_uv)), np.vstack((xyz, self.physical(extra_uv)))
        faces = np.vstack((faces, [8, 10, 9]))
        surface = OriginalSurface(uv, xyz, faces)
        origin, step, shape = np.array([-.125, -.125]), np.array([.125, .25]), (14, 27)
        mask = np.indices(shape).sum(axis=0) % 7 != 0
        whole = surface.sample_grid(origin, step, shape, query_mask=mask)
        small_blocks = OriginalSurface(uv, xyz, faces, max_working_points=7).sample_grid(
            origin, step, shape, query_mask=mask)
        discrete = ("valid", "reason_bits", "coverage_count", "source_triangle_id", "ambiguity_mask")
        floating = ("xyz", "normals", "jacobian", "barycentric_weights")
        for key in discrete:
            np.testing.assert_array_equal(whole[key], small_blocks[key], err_msg=key)
        for key in floating:
            np.testing.assert_allclose(whole[key], small_blocks[key], atol=1e-13, rtol=0, err_msg=key)
        total_incidence = np.zeros(len(faces), dtype=np.int64)
        for y in range(0, shape[0], 5):
            for x in range(0, shape[1], 11):
                height, width = min(5, shape[0] - y), min(11, shape[1] - x)
                tile = surface.sample_grid(origin + step * (x, y), step, (height, width),
                                           query_mask=mask[y:y + height, x:x + width])
                total_incidence += tile["source_triangle_covered_query_count"]
                for key in discrete:
                    np.testing.assert_array_equal(tile[key], whole[key][y:y + height, x:x + width], err_msg=key)
                for key in floating:
                    np.testing.assert_allclose(tile[key], whole[key][y:y + height, x:x + width],
                                               atol=1e-13, rtol=0, err_msg=key)
        np.testing.assert_array_equal(total_incidence, whole["source_triangle_covered_query_count"])

    def test_dense_eight_by_eight_subpixel_grid(self):
        # 64×64 output cells, each with 8×8 point queries: 262,144 queries.
        result = OriginalSurface(*self.square(size=64.)).sample_grid(
            (1 / 16, 1 / 16), (1 / 8, 1 / 8), (512, 512))
        self.assertTrue(result["valid"].all())
        self.assertEqual(result["counts"]["requested_pixels"], 262144)
        for row, col in ((0, 0), (123, 318), (511, 511)):
            uv = np.array([(col + .5) / 8, (row + .5) / 8])
            np.testing.assert_allclose(result["xyz"][row, col], self.physical(uv), atol=1e-12)
            np.testing.assert_allclose(result["jacobian"][row, col], self.J, atol=1e-14)

    def test_empty_geometry_and_all_masked_queries(self):
        empty = OriginalSurface(np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3), dtype=int))
        result = empty.sample_grid((0., 0.), (.25, 4.), (2, 3))
        np.testing.assert_array_equal(result["reason_bits"], NO_SOURCE_COVERAGE)
        self.assert_invalid_sentinels(result)
        masked = OriginalSurface(*self.square()).sample_grid(
            (0., 0.), (.25, 4.), (2, 3), query_mask=np.zeros((2, 3), dtype=bool))
        np.testing.assert_array_equal(masked["reason_bits"], NOT_REQUESTED)
        self.assertEqual(masked["counts"]["candidate_source_triangles"], 0)
        self.assert_invalid_sentinels(masked)

    def test_geometry_is_copied_and_invalid_inputs_fail(self):
        uv, xyz, faces = self.square()
        surface = OriginalSurface(uv, xyz, faces)
        uv[:] = 100.
        xyz[:] = 0.
        faces[:] = 0
        self.assertTrue(surface.sample_grid((1., 1.), (.5, .5), (2, 2))["valid"].all())
        for array in (surface.uv, surface.xyz, surface.triangles):
            self.assertFalse(array.flags.writeable)
        for step in ((0., 1.), (-1., 1.), (np.nan, 1.), (np.inf, 1.)):
            with self.assertRaises(ValueError):
                surface.sample_grid((0., 0.), step, (2, 2))
        for shape in ((0, 2), (True, 2), (2.1, 3), (2,)):
            with self.assertRaises(ValueError):
                surface.sample_grid((0., 0.), (1., 1.), shape)
        with self.assertRaises(ValueError):
            surface.sample_grid((0., np.inf), (1., 1.), (2, 2))
        with self.assertRaises(ValueError):
            surface.sample_grid((0., 0.), (1., 1.), (2, 2), query_mask=np.ones((3, 3)))
        with self.assertRaises(MemoryError):
            OriginalSurface(*self.square(), max_output_bytes=100).sample_grid(
                (0., 0.), (1., 1.), (2, 2))


if __name__ == "__main__":
    unittest.main()
