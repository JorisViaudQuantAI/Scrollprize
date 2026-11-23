"""Whole-cell failures which finite sample-point checks can miss."""
import unittest
import numpy as np
from footprint import certify, OVERLAP, INCONSISTENT_BOUNDARY, NEGATIVE


def rectangle(x0, x1, z=0.):
    uv = np.array([[x0, 0.], [x1, 0.], [x1, 1.], [x0, 1.]])
    xyz = np.column_stack((uv, np.full(4, z)))
    return uv, xyz, np.array([[0, 1, 2], [0, 2, 3]])


def combine(*meshes):
    vertices, xyz, triangles, count = [], [], [], 0
    for u, x, t in meshes:
        vertices.append(u); xyz.append(x); triangles.append(t + count); count += len(u)
    return np.concatenate(vertices), np.concatenate(xyz), np.concatenate(triangles)


class FootprintTests(unittest.TestCase):
    def test_huge_finite_negative_bbox_cannot_overflow_and_disappear(self):
        u, x, t = rectangle(0, 1)
        large = np.array([[-1e20, -1e20], [0., 1e20], [1e20, -1e20]])
        u = np.vstack((u, large)); x = np.vstack((x, np.column_stack((large, np.zeros(3)))))
        t = np.vstack((t, [4, 5, 6]))
        with np.errstate(all='raise'):
            out = certify(u, x, t, (.5, .5), (1, 1), (1, 1))
        self.assertTrue(out['reason_bits'][0, 0] & NEGATIVE)

    def test_empty_or_unrepresentable_pixel_cannot_pass_absolute_tolerance(self):
        with self.assertRaises(ValueError):
            certify(np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3), int),
                    (0, 0), (1e-7, 1e-7), (1, 1))
        out = certify(np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3), int),
                      (0, 0), (1, 1), (1, 1))
        self.assertFalse(out['valid'][0, 0])

    def test_closed_edge_contact_from_unrelated_surface_is_rejected(self):
        u, x, t = combine(rectangle(0, 1), rectangle(1, 2, 3))
        out = certify(u, x, t, (.5, .5), (1, 1), (1, 1))
        self.assertTrue(out['reason_bits'][0, 0] & INCONSISTENT_BOUNDARY)

    def test_outward_bbox_enumeration_keeps_touching_negative_face(self):
        u = np.array([[.95, .95], [1.05, .95], [1.05, 1.05], [.95, 1.05],
                      [1.05, .98], [1.05, 1.02], [1.1, 1.]])
        x = np.column_stack((u, np.zeros(len(u))))
        t = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
        out = certify(u, x, t, (1, 1), (.1, .1), (1, 1))
        self.assertTrue(out['reason_bits'][0, 0] & NEGATIVE)

    def test_complete_square_split_at_shared_original_edge(self):
        u, x, t = rectangle(0, 1)
        out = certify(u, x, t, (.5, .5), (1., 1.), (1, 1))
        self.assertTrue(out['valid'][0, 0])
        self.assertAlmostEqual(out['coverage_area'][0, 0], 1.)
        self.assertAlmostEqual(out['union_area_lower_bound'][0, 0], 1.)

    def test_narrow_hole_missed_by_all_four_midpoints(self):
        u, x, t = combine(rectangle(0, .49), rectangle(.51, 1))
        samples = (np.arange(4) + .5) / 4
        self.assertTrue(((samples < .49) | (samples > .51)).all())
        out = certify(u, x, t, (.5, .5), (1., 1.), (1, 1))
        self.assertFalse(out['valid'][0, 0])
        self.assertAlmostEqual(out['coverage_area'][0, 0], .98)

    def test_overlap_cannot_compensate_for_equal_area_hole(self):
        u, x, t = combine(rectangle(0, .4), rectangle(.5, 1), rectangle(.1, .2))
        out = certify(u, x, t, (.5, .5), (1., 1.), (1, 1))
        self.assertAlmostEqual(out['coverage_area'][0, 0], 1.)
        self.assertAlmostEqual(out['pairwise_overlap_area'][0, 0], .1)
        self.assertAlmostEqual(out['union_area_lower_bound'][0, 0], .9)
        self.assertTrue(out['reason_bits'][0, 0] & OVERLAP)
        self.assertFalse(out['valid'][0, 0])

    def test_abutting_unrelated_surfaces_are_not_one_pixel(self):
        u, x, t = combine(rectangle(0, .5, 0), rectangle(.5, 1, 3))
        out = certify(u, x, t, (.5, .5), (1., 1.), (1, 1))
        self.assertTrue(out['reason_bits'][0, 0] & INCONSISTENT_BOUNDARY)

    def test_tiny_negative_face_inside_otherwise_valid_pixel_poisons_it(self):
        u, x, t = rectangle(0, 1)
        small = np.array([[.48, .48], [.50, .52], [.52, .48]])
        u = np.vstack((u, small)); x = np.vstack((x, np.column_stack((small, np.zeros(3)))))
        t = np.vstack((t, [4, 5, 6]))
        out = certify(u, x, t, (.5, .5), (1., 1.), (1, 1))
        self.assertTrue(out['reason_bits'][0, 0] & NEGATIVE)

    def test_grid_tiling_and_query_mask(self):
        u, x, t = rectangle(0, 1)
        whole = certify(u, x, t, (.125, .125), (.25, .25), (4, 4))
        left = certify(u, x, t, (.125, .125), (.25, .25), (4, 2))
        right = certify(u, x, t, (.625, .125), (.25, .25), (4, 2))
        np.testing.assert_array_equal(whole['valid'], np.column_stack((left['valid'], right['valid'])))
        np.testing.assert_array_equal(whole['coverage_area'], np.column_stack((left['coverage_area'], right['coverage_area'])))
        out = certify(u, x, t, (.5, .5), (1, 1), (1, 1), query_mask=np.zeros((1, 1), bool))
        self.assertEqual(int(out['reason_bits'][0, 0]), 1)


if __name__ == '__main__':
    unittest.main()
