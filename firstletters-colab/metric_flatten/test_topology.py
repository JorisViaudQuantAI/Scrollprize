"""Generated geometry checks; no manuscript arrays, inference, or downloads."""
import unittest

import numpy as np

from topology import (
    AMBIGUOUS_OVERLAP,
    DEGENERATE_UV_COVERAGE,
    INCONSISTENT_SHARED_BOUNDARY,
    NEGATIVE_UV_COVERAGE,
    NOT_REQUESTED,
    rasterize_triangles,
    triangle_metrics,
)


class TopologyTests(unittest.TestCase):
    @staticmethod
    def square():
        uv = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])
        xyz = np.column_stack((2 * uv[:, 0] + uv[:, 1],
                               3 * uv[:, 1], 0.5 * uv[:, 0] - uv[:, 1]))
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return uv, xyz, faces

    def test_affine_plane_and_original_face_reconstruction(self):
        uv, xyz, faces = self.square()
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        yy, xx = np.mgrid[:5, :5]
        expected = np.stack((2 * xx + yy, 3 * yy, .5 * xx - yy), axis=-1)
        self.assertTrue(out["valid"].all())
        np.testing.assert_allclose(out["xyz"], expected, atol=1e-12, rtol=0)
        owners = out["source_triangle_id"]
        recovered = (xyz[faces[owners]] * out["barycentric_weights"][..., None]).sum(axis=-2)
        np.testing.assert_array_equal(recovered, out["xyz"])
        self.assertEqual(out["counts"]["source_triangles"], 2)

    def test_shared_original_edge_and_vertices_are_not_ambiguous(self):
        uv, xyz, faces = self.square()
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        self.assertFalse(out["ambiguity_mask"].any())
        np.testing.assert_array_equal(np.diag(out["coverage_count"]), np.full(5, 2))
        np.testing.assert_array_equal(np.diag(out["source_triangle_id"]), np.zeros(5))

    def test_shared_fan_vertex_is_valid_without_all_pairs_sharing_edges(self):
        uv = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.], [2., 2.]])
        xyz = np.column_stack((uv, np.zeros(5)))
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        self.assertTrue(out["valid"].all())
        self.assertEqual(out["coverage_count"][2, 2], 4)
        self.assertFalse(out["ambiguity_mask"].any())

    def test_slit_is_not_filled_by_new_triangulation(self):
        uv = np.array([[0., 0.], [1., 0.], [1., 4.], [0., 4.],
                       [3., 0.], [4., 0.], [4., 4.], [3., 4.]])
        xyz = np.column_stack((uv, np.zeros(8)))
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        self.assertFalse(out["valid"][:, 2].any())
        self.assertTrue(out["valid"][:, [0, 1, 3, 4]].all())
        self.assertTrue(np.isnan(out["xyz"][:, 2]).all())
        self.assertEqual(out["counts"]["source_components"], 2)
        np.testing.assert_array_equal(out["source_component_output_pixel_count"], [10, 10])

    def test_nonadjacent_overlapping_faces_are_rejected_even_if_xyz_agree(self):
        uv = np.array([[0., 0.], [4., 0.], [0., 4.]] * 2)
        xyz = np.column_stack((uv, np.zeros(6)))
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        self.assertFalse(out["valid"].any())
        self.assertTrue(out["ambiguity_mask"][1, 1])
        self.assertTrue(out["ambiguity_mask"][0, 0])
        self.assertTrue(out["reason_bits"][1, 1] & AMBIGUOUS_OVERLAP)
        self.assertEqual(out["counts"]["source_components_without_output"], 2)

    def test_internal_hole_in_connected_surface_remains_empty(self):
        uv = np.array([[0., 0.], [6., 0.], [6., 6.], [0., 6.],
                       [2., 2.], [4., 2.], [4., 4.], [2., 4.]])
        xyz = np.column_stack((uv, np.zeros(8)))
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            faces.extend(([i, j, j + 4], [i, j + 4, i + 4]))
        out = rasterize_triangles(uv, xyz, np.array(faces), (7, 7))
        expected = np.ones((7, 7), dtype=bool)
        expected[3, 3] = False
        np.testing.assert_array_equal(out["valid"], expected)
        self.assertEqual(out["coverage_count"][3, 3], 0)
        self.assertEqual(out["counts"]["source_components"], 1)

    def test_shared_boundary_tolerance_requires_xyz_consistency(self):
        uv = np.array([[1e-10, 0.], [1e-10, 2.], [-1., 0.], [1., 0.]])
        xyz = np.column_stack((uv, [0., 0., 1e6, 0.]))
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        out = rasterize_triangles(uv, xyz, faces, (3, 2))
        # This query falls within the declared barycentric edge tolerance, but
        # the two interpolants disagree by 1e-4 XYZ units, above 1e-6.
        self.assertFalse(out["valid"][1, 0])
        self.assertTrue(out["reason_bits"][1, 0] & INCONSISTENT_SHARED_BOUNDARY)
        xyz[2, 2] = 1e3
        consistent = rasterize_triangles(uv, xyz, faces, (3, 2))
        self.assertTrue(consistent["valid"][1, 0])

    def test_duplicate_face_interior_is_not_a_shared_boundary(self):
        uv = np.array([[0., 0.], [4., 0.], [0., 4.]])
        xyz = np.column_stack((uv, np.zeros(3)))
        out = rasterize_triangles(uv, xyz, np.array([[0, 1, 2], [0, 1, 2]]), (5, 5))
        self.assertFalse(out["valid"][1, 1])
        self.assertTrue(out["ambiguity_mask"][1, 1])
        self.assertTrue(out["valid"][0, 0])

    def test_flipped_face_poisoning_is_independent_of_face_order(self):
        uv = np.array([[0., 0.], [4., 0.], [0., 4.]])
        xyz = np.column_stack((uv, np.zeros(3)))
        for faces in (np.array([[0, 1, 2], [0, 2, 1]]), np.array([[0, 2, 1], [0, 1, 2]])):
            out = rasterize_triangles(uv, xyz, faces, (5, 5))
            self.assertFalse(out["valid"].any())
            self.assertTrue(out["reason_bits"][1, 1] & NEGATIVE_UV_COVERAGE)
            self.assertEqual(out["counts"]["negative_uv_triangles"], 1)
            self.assertEqual(out["triangle_metrics"]["singular_values"].shape, (2, 2))

    def test_degenerate_line_poisoning_keeps_other_pixels(self):
        uv, xyz, faces = self.square()
        extra = np.array([[0., 1.], [2., 1.], [4., 1.]])
        extra_xyz = np.column_stack((2 * extra[:, 0] + extra[:, 1],
                                    3 * extra[:, 1], .5 * extra[:, 0] - extra[:, 1]))
        uv, xyz = np.concatenate((uv, extra)), np.concatenate((xyz, extra_xyz))
        faces = np.concatenate((faces, [[4, 5, 6]]))
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        self.assertFalse(out["valid"][1].any())
        self.assertTrue(out["valid"][[0, 2, 3, 4]].all())
        self.assertTrue(((out["reason_bits"][1] & DEGENERATE_UV_COVERAGE) != 0).all())
        self.assertEqual(out["counts"]["degenerate_uv_triangles"], 1)

    def test_metrics_preserve_negative_and_degenerate_source_faces(self):
        uv = np.array([[0., 0.], [1., 0.], [0., 1.], [2., 0.]])
        xyz = np.column_stack((3 * uv[:, 0], 2 * uv[:, 1], np.zeros(4)))
        faces = np.array([[0, 1, 2], [0, 2, 1], [0, 1, 3]])
        met = triangle_metrics(uv, xyz, faces)
        np.testing.assert_allclose(met["singular_values"][:2], [[3., 2.], [3., 2.]])
        np.testing.assert_allclose(met["signed_uv_area"], [.5, -.5, 0.])
        np.testing.assert_allclose(met["source_3d_area"], [3., 3., 0.])
        self.assertTrue(np.isnan(met["singular_values"][2]).all())
        self.assertEqual(sum(met["counts"][k] for k in
                             ["positive_uv_triangles", "negative_uv_triangles", "degenerate_uv_triangles"]), 3)

    def test_epsilon_degenerate_interior_poisons_positive_coverage(self):
        uv, xyz, faces = self.square()
        tiny = np.array([[-1., -1.], [2., -1.], [-1., 2.]]) * 1e-7
        tiny_xyz = np.column_stack((tiny, np.zeros(3)))
        uv = np.concatenate((uv, tiny))
        xyz = np.concatenate((xyz, tiny_xyz))
        faces = np.concatenate((faces, [[4, 5, 6]]))
        out = rasterize_triangles(uv, xyz, faces, (5, 5))
        self.assertFalse(out["valid"][0, 0])
        self.assertTrue(out["reason_bits"][0, 0] & DEGENERATE_UV_COVERAGE)
        self.assertEqual(out["counts"]["degenerate_uv_triangles"], 1)
        self.assertEqual(out["counts"]["valid_pixels"], 24)

    def test_masked_queries_and_absent_source_components_are_explicit(self):
        uv, xyz, faces = self.square()
        requested = np.zeros((5, 5), dtype=bool)
        out = rasterize_triangles(uv, xyz, faces, (5, 5), query_mask=requested)
        self.assertFalse(out["valid"].any())
        self.assertTrue((out["reason_bits"] == NOT_REQUESTED).all())
        self.assertEqual(out["counts"]["source_components_without_output"], 1)
        self.assertEqual(out["counts"]["source_triangles_without_requested_grid_coverage"], 2)

    def test_block_partition_and_origin_do_not_change_geometry(self):
        uv, xyz, faces = self.square()
        first = rasterize_triangles(uv, xyz, faces, (5, 5), max_working_points=3)
        second = rasterize_triangles(uv + [100, -200], xyz, faces, (5, 5),
                                     origin_xy=(100, -200), max_working_points=17)
        for key in ("xyz", "valid", "source_triangle_id", "barycentric_weights", "reason_bits"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_invalid_sources_and_unbounded_grid_fail_before_rasterization(self):
        uv, xyz, faces = self.square()
        with self.assertRaises(MemoryError):
            rasterize_triangles(uv, xyz, faces, (100000, 100000))
        bad = uv.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            rasterize_triangles(bad, xyz, faces, (5, 5))
        with self.assertRaises(ValueError):
            rasterize_triangles(uv, xyz, faces.astype(float), (5, 5))


if __name__ == "__main__":
    unittest.main()
