"""Generated geometry only; no file, network, manuscript or model reads."""

from dataclasses import replace
import unittest

import numpy as np

try:
    from .transport import GeometryError, Settings, transport_normals
except ImportError:
    from transport import GeometryError, Settings, transport_normals


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_degrees(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def surface_tangents(normals):
    """Build an oriented source tangent basis independently of any Jacobian."""
    axes = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    u = unit(np.cross(axes, normals))
    v = np.cross(normals, u)
    return u, v


def mapped_surface_normal(point_map, points, normals, epsilon=1e-5):
    """Differentiate the parametrised SURFACE, not coordinate-axis Jacobians."""
    u, v = surface_tangents(normals)
    du = (point_map(points + epsilon * u) - point_map(points - epsilon * u)) / (2 * epsilon)
    dv = (point_map(points + epsilon * v) - point_map(points - epsilon * v)) / (2 * epsilon)
    return unit(np.cross(du, dv))


def rotation():
    c, s = np.cos(0.37), np.sin(0.37)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    c, s = np.cos(-0.23), np.sin(-0.23)
    rx = np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
    return rz @ rx


def smooth_example():
    a = np.array([[1.1, 0.15, 0.2], [0, 0.9, 0.1], [0, 0, 1.2]])
    b = np.array([0.8, -1., 0.1])

    def displacement(x):
        return np.column_stack((0.07 * np.sin(x[:, 1]), 0.03 * x[:, 0] * x[:, 2], 0.04 * x[:, 0] ** 2 + 0.02 * x[:, 1] ** 2))

    def point_map(x):
        return x @ a.T + b + displacement(x)

    return a, b, displacement, point_map


class TransportTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261123)
        self.points = rng.uniform(-2, 2, (24, 3))
        self.normals = unit(rng.normal(size=(24, 3)))
        self.settings = Settings(step_um=0.002, batch_size=7)

    def test_rigid_and_uniform_scale_match_oriented_surface(self):
        r = rotation()
        for scale in (1., 1.7):
            f = lambda x: scale * (x @ r.T) + np.array([2., -3., 0.4])
            out = transport_normals(f, self.points, self.normals, settings=self.settings)
            analytic = self.normals @ r.T
            self.assertLess(float(angle_degrees(out.normals, analytic).max()), 1e-7)
            self.assertLess(float(angle_degrees(out.normals, mapped_surface_normal(f, self.points, self.normals)).max()), 1e-7)
            np.testing.assert_allclose(out.determinant, scale ** 3, rtol=1e-10)

    def test_anisotropic_scale_matches_transformed_tangent_cross_product(self):
        a = np.diag([2., 3., 0.5])
        f = lambda x: x @ a.T
        out = transport_normals(f, self.points, self.normals, settings=self.settings)
        u, v = surface_tangents(self.normals)
        expected = unit(np.cross(u @ a.T, v @ a.T))
        self.assertLess(float(angle_degrees(out.normals, expected).max()), 1e-7)
        self.assertGreater(float(np.median(angle_degrees(unit(self.normals @ a.T), expected))), 20.)

    def test_shear_demonstrates_old_vector_rule_error(self):
        a = np.array([[2., 0.5, 1.], [0., 1., 0.], [0., 0., 0.5]])
        normals = np.tile([0., 0., 1.], (len(self.points), 1))
        f = lambda x: x @ a.T
        out = transport_normals(f, self.points, normals, settings=self.settings)
        expected = mapped_surface_normal(f, self.points, normals)
        self.assertLess(float(angle_degrees(out.normals, expected).max()), 1e-8)
        old_error = angle_degrees(unit(normals @ a.T), expected)
        np.testing.assert_allclose(old_error, np.degrees(np.arctan(2.)), atol=1e-8)

    def test_complete_smooth_map_matches_independent_surface_and_analytic_jacobian(self):
        a, _, _, f = smooth_example()
        out = transport_normals(f, self.points, self.normals, settings=self.settings)
        expected = mapped_surface_normal(f, self.points, self.normals)
        self.assertLess(float(angle_degrees(out.normals, expected).max()), 2e-6)
        analytic = np.broadcast_to(a, (len(self.points), 3, 3)).copy()
        analytic[:, 0, 1] += 0.07 * np.cos(self.points[:, 1])
        analytic[:, 1, 0] += 0.03 * self.points[:, 2]
        analytic[:, 1, 2] += 0.03 * self.points[:, 0]
        analytic[:, 2, 0] += 0.08 * self.points[:, 0]
        analytic[:, 2, 1] += 0.04 * self.points[:, 1]
        np.testing.assert_allclose(out.jacobian, analytic, atol=4e-9, rtol=0)

    def test_point_map_composition_is_not_assumed(self):
        a, b, d, actual = smooth_example()
        alternate = lambda x: (x + d(x)) @ a.T + b
        actual_out = transport_normals(actual, self.points, self.normals, settings=self.settings)
        alternate_out = transport_normals(alternate, self.points, self.normals, settings=self.settings)
        expected = mapped_surface_normal(alternate, self.points, self.normals)
        self.assertLess(float(angle_degrees(alternate_out.normals, expected).max()), 2e-6)
        self.assertGreater(float(np.median(angle_degrees(actual_out.normals, alternate_out.normals))), 0.1)

    def test_anisotropic_voxel_map_requires_explicit_physical_wrapper(self):
        source_pitch = np.array([2., 3., 5.])
        target_pitch = np.array([0.5, 1., 4.])
        r = rotation()
        voxel_map = lambda x: ((x * source_pitch) @ r.T + np.array([0.2, 0.3, -0.4])) / target_pitch
        physical_map = lambda x_um: voxel_map(x_um / source_pitch) * target_pitch
        out = transport_normals(physical_map, self.points, self.normals, settings=self.settings)
        self.assertLess(float(angle_degrees(out.normals, self.normals @ r.T).max()), 1e-7)

    def test_bad_step_fails_and_resolved_step_passes(self):
        def f(x):
            y = x.copy()
            y[:, 0] += 0.01 * np.sin(50 * x[:, 0])
            return y
        point, normal = np.zeros((1, 3)), np.array([[1., 1., 1.]])
        with self.assertRaisesRegex(GeometryError, "step convergence"):
            transport_normals(f, point, normal, settings=Settings(step_um=0.2))
        out = transport_normals(f, point, normal, settings=Settings(step_um=0.0001))
        np.testing.assert_allclose(out.jacobian[0], np.diag([1.5, 1., 1.]), atol=2e-7)

    def test_cusp_with_stable_central_difference_is_rejected(self):
        def f(x):
            y = x.copy()
            y[:, 0] += 0.05 * np.abs(x[:, 0])
            return y
        with self.assertRaisesRegex(GeometryError, "one-sided"):
            transport_normals(f, np.zeros((1, 3)), np.array([[0., 0., 1.]]), settings=self.settings)

    def test_normal_angle_check_catches_amplified_jacobian_error(self):
        def f(x):
            return np.column_stack((x[:, 0] + 0.01 * x[:, 2] ** 3, x[:, 1], 1e-4 * x[:, 2]))
        with self.assertRaisesRegex(GeometryError, "normal step convergence"):
            transport_normals(f, np.zeros((1, 3)), np.array([[1., 0., 0.]]), settings=Settings(step_um=0.01))

    def test_singular_reflected_and_ill_conditioned_maps_fail(self):
        for scales, message in (([1., 1., 0.], "singular"), ([1., 1., -1.], "nonpositive"), ([1., 1., 1e-7], "ill-conditioned"), ([1e-12] * 3, "vanishing")):
            with self.subTest(scales=scales), self.assertRaisesRegex(GeometryError, message):
                transport_normals(lambda x: x * scales, self.points, self.normals, settings=self.settings)

    def test_one_folded_member_rejects_entire_batch(self):
        def f(x):
            return np.column_stack((x[:, 0] ** 2, x[:, 1], x[:, 2]))
        with self.assertRaisesRegex(GeometryError, "nonpositive determinant.*\\[1\\]"):
            transport_normals(f, np.array([[1., 0., 0.], [-1., 0., 0.]]), np.tile([0., 0., 1.], (2, 1)), settings=self.settings)

    def test_nonfinite_and_nonphysical_data_fail(self):
        for bad in (np.nan, np.inf):
            points = self.points.copy()
            points[0, 0] = bad
            with self.assertRaisesRegex(GeometryError, "nonfinite points"):
                transport_normals(lambda x: x, points, self.normals, settings=self.settings)
            normals = self.normals.copy()
            normals[0, 0] = bad
            with self.assertRaisesRegex(GeometryError, "nonfinite normals"):
                transport_normals(lambda x: x, self.points, normals, settings=self.settings)
            with self.assertRaisesRegex(GeometryError, "nonfinite mapped"):
                transport_normals(lambda x: np.full_like(x, bad), self.points, self.normals, settings=self.settings)
        with self.assertRaisesRegex(GeometryError, "zero normal"):
            transport_normals(lambda x: x, self.points, np.zeros_like(self.points), settings=self.settings)
        with self.assertRaisesRegex(GeometryError, "real numeric"):
            transport_normals(lambda x: x, self.points.astype(complex), self.normals, settings=self.settings)
        with self.assertRaisesRegex(GeometryError, "float64"):
            transport_normals(lambda x: x.astype(np.float32), self.points, self.normals, settings=self.settings)

    def test_shape_step_and_resource_contracts(self):
        with self.assertRaisesRegex(GeometryError, "shape"):
            transport_normals(lambda x: x, self.points, self.normals[:1], settings=self.settings)
        with self.assertRaisesRegex(GeometryError, "point limit"):
            transport_normals(lambda x: x, self.points, self.normals, settings=replace(self.settings, max_points=2))
        with self.assertRaisesRegex(GeometryError, "runtime limit"):
            transport_normals(lambda x: x, self.points, self.normals, settings=replace(self.settings, max_seconds=1e-12))
        with self.assertRaisesRegex(GeometryError, "not representable"):
            transport_normals(lambda x: x, np.full((1, 3), 1e20), np.ones((1, 3)), settings=self.settings)
        with self.assertRaisesRegex(GeometryError, "positive"):
            transport_normals(lambda x: x, self.points, self.normals, settings=Settings(step_um=0.))

    def test_rounded_nonzero_stencil_cannot_false_accept_identity(self):
        coordinate = 1.5 * 2 ** 20
        step = 5 * np.spacing(coordinate)
        # Without an actual-spacing guard, h/2 and h/4 both give J_xx=0.8,
        # pass the central convergence check and rotate this normal by 6.34 deg.
        with self.assertRaisesRegex(GeometryError, "spacing differs"):
            transport_normals(lambda x: x, np.array([[coordinate, 0., 0.]]), np.array([[1., 1., 0.]]), settings=Settings(step_um=step))

    def test_batching_and_normal_magnitude_do_not_change_geometry(self):
        _, _, _, f = smooth_example()
        a = transport_normals(f, self.points, self.normals * 1e200, settings=replace(self.settings, batch_size=1))
        b = transport_normals(f, self.points, self.normals, settings=replace(self.settings, batch_size=24))
        # BLAS may round the supplied map differently for B=1 and B>1;
        # require angular agreement, not unjustified bitwise invariance.
        self.assertLess(float(angle_degrees(a.normals, b.normals).max()), 1e-7)
        self.assertEqual(a.mapped_point_evaluations, 19 * len(self.points))
        self.assertEqual(a.map_calls, 19 * len(self.points))
        self.assertEqual(b.map_calls, 19)
        self.assertFalse(a.normals.flags.writeable)


if __name__ == "__main__":
    unittest.main()
