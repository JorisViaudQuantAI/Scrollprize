"""Independent analytic oracles for the anchor/ray 2x2 sampler.

These fixtures use planes and a cylinder with closed-form inverse coordinates.
They do not read CT, weights, annotations, or experiment predictions. Legacy
geometry is used only for the explicitly required byte-compatibility check.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "studio_oblique_stress"))
import geometry as legacy
import factor_geometry as fg


PITCH = 9.36
CT_UM = 9.366
OFFSETS = np.arange(-8, 8, dtype=np.float64) * CT_UM
FAMILIES = ("identity", "normal_only", "anchor_only", "rigid")
ANGLES = (-20, -5, 5, 20)


class SurfaceFixture:
    """Independent Cartesian bilinear evaluator with the author conventions."""

    def __init__(self, points, mesh_pitch, ct_um=CT_UM, sign=1):
        self.a = np.asarray(points, np.float32)
        self.valid = np.ones(self.a.shape[:2], bool)
        self.pitch = np.asarray(mesh_pitch, np.float64)
        self.ct_um = float(ct_um)
        self.sign = sign

    def points(self, ys, xs, pitch_um):
        yy, xx = np.meshgrid(np.asarray(ys, np.float32), np.asarray(xs, np.float32), indexing="ij")
        yy = (yy + np.float32(.5)) * np.float32(pitch_um / self.pitch[1])
        xx = (xx + np.float32(.5)) * np.float32(pitch_um / self.pitch[0])
        iy, ix = np.floor(yy).astype(int), np.floor(xx).astype(int)
        valid = ((iy >= 0) & (ix >= 0) & (iy < self.a.shape[0]-1) & (ix < self.a.shape[1]-1))
        iy, ix = np.clip(iy, 0, self.a.shape[0]-2), np.clip(ix, 0, self.a.shape[1]-2)
        v, u = yy-iy.astype(np.float32), xx-ix.astype(np.float32)
        aa, bb = self.a[iy, ix], self.a[iy, ix+1]
        cc, dd = self.a[iy+1, ix], self.a[iy+1, ix+1]
        xyz = ((1-u)[..., None]*(1-v)[..., None]*aa + u[..., None]*(1-v)[..., None]*bb
               + (1-u)[..., None]*v[..., None]*cc + u[..., None]*v[..., None]*dd)
        col = (1-v)[..., None]*(bb-aa) + v[..., None]*(dd-cc)
        row = (1-u)[..., None]*(cc-aa) + u[..., None]*(dd-bb)
        normal = np.cross(row, col)
        length = np.linalg.norm(normal, axis=-1)
        valid &= (self.valid[iy, ix] & self.valid[iy, ix+1]
                  & self.valid[iy+1, ix] & self.valid[iy+1, ix+1] & (length > 1e-6))
        normal *= self.sign / np.maximum(length, 1e-6)[..., None]
        xyz[~valid], normal[~valid] = -1, 0
        return xyz, normal, valid


def plane(sign=1, large_offset=False):
    yy, xx = np.mgrid[:84, :84]
    # Unequal chart scales prevent an inverse from silently assuming an
    # isometric parameterization or confusing label pitch and native depth.
    kx, ky = 1.25 * PITCH / CT_UM, .8 * PITCH / CT_UM
    origin = np.array([4200, 2800, 14000] if large_offset else [100, 200, 300])
    points = np.stack([origin[0] + 2*kx*xx, origin[1] + 2*ky*yy,
                       np.full(xx.shape, origin[2])], axis=-1)
    return SurfaceFixture(points, [2*PITCH, 2*PITCH], sign=sign), (ky, kx)


def cylinder():
    radius = 300.0
    yy, xx = np.mgrid[:70, :1100]
    u, v = xx*.125-64, yy*2.0
    points = np.stack([200 + radius*np.sin(u/radius), 100 + v,
                       200 + radius*(np.cos(u/radius)-1)], axis=-1)
    return SurfaceFixture(points, [.125, 2.0], ct_um=1.0), radius


def grid():
    return np.stack(np.meshgrid(np.arange(48, 80), np.arange(48, 80), indexing="ij"), axis=-1).astype(float)


def patch(surface, family, axis, angle, y=0, x=0, pitch=PITCH, offsets=OFFSETS):
    return fg.build_patch(surface, y, x, angle, axis=axis, family=family,
                          pitch_um=pitch, offsets_um=offsets)


class FactorGeometryOracles(unittest.TestCase):
    def test_identity_and_rigid_preserve_legacy_bytes_at_large_native_origin(self):
        surface, _ = plane(large_offset=True)
        for axis in ("x", "y"):
            for angle in ANGLES:
                for family in ("identity", "rigid"):
                    with self.subTest(axis=axis, angle=angle, family=family):
                        expected = legacy.rotate_patch(surface, 6, 10, 0 if family=="identity" else angle,
                                                       axis=axis, pitch_um=PITCH, offsets_um=OFFSETS)
                        actual = patch(surface, family, axis, angle, y=6, x=10)
                        self.assertEqual(actual["coords"].dtype, np.float32)
                        np.testing.assert_array_equal(actual["coords"], expected["coords"])
                        np.testing.assert_array_equal(actual["mesh_valid"], expected["mesh_valid"])
                        np.testing.assert_array_equal(actual["rotation"], expected["rotation"])

    def test_zero_angle_all_four_families_have_exact_identity_readback(self):
        surface, _ = plane()
        expected = legacy.rotate_patch(surface, 0, 0, 0, offsets_um=OFFSETS)
        for family in FAMILIES:
            arm = patch(surface, family, "x", 0)
            inv = fg.inverse_readback(surface, 0, 0, arm, pitch_um=PITCH)
            np.testing.assert_array_equal(arm["coords"], expected["coords"])
            np.testing.assert_array_equal(inv["query_yx"], grid())
            np.testing.assert_array_equal(inv["depth_vox"], np.zeros((32, 32)))
            self.assertTrue(inv["valid"].all())

    def test_planar_anchor_inverse_is_secant_and_rigid_inverse_is_cosine(self):
        original, relative = grid(), grid()-63.5
        for normal_sign in (1, -1):
            surface, metric_yx = plane(sign=normal_sign)
            for axis in ("x", "y"):
                changing = 0 if axis == "x" else 1
                depth_sign = normal_sign * (1 if axis == "x" else -1)
                for angle in ANGLES:
                    theta = np.deg2rad(angle)
                    for family in ("anchor_only", "rigid"):
                        with self.subTest(axis=axis, angle=angle, family=family, normal_sign=normal_sign):
                            arm = patch(surface, family, axis, angle, y=6, x=10)
                            inv = fg.inverse_readback(surface, 6, 10, arm, pitch_um=PITCH)
                            factor = 1/np.cos(theta) if family=="anchor_only" else np.cos(theta)
                            depth_factor = np.tan(theta) if family=="anchor_only" else np.sin(theta)
                            expected = original.copy()
                            expected[..., changing] = 63.5 + relative[..., changing]*factor
                            depth = depth_sign * relative[..., changing]*metric_yx[changing]*depth_factor
                            expected_valid = (depth >= -8) & (depth <= 7)
                            np.testing.assert_array_equal(inv["valid"], expected_valid)
                            np.testing.assert_allclose(inv["query_yx"], expected, rtol=0, atol=3e-4)
                            np.testing.assert_allclose(inv["depth_vox"], depth, rtol=0, atol=3e-4)
                            self.assertLessEqual(float(inv["residual_vox"].max()), .01)

    def test_ray_only_keeps_exact_reference_points_on_a_curved_surface(self):
        surface, _ = cylinder()
        reference = surface.points(np.arange(128), np.arange(128), 1.0)[0]
        for axis in ("x", "y"):
            for angle in ANGLES:
                with self.subTest(axis=axis, angle=angle):
                    arm = patch(surface, "normal_only", axis, angle, pitch=1.0, offsets=np.arange(-8, 8))
                    inv = fg.inverse_readback(surface, 0, 0, arm, pitch_um=1.0)
                    np.testing.assert_array_equal(arm["coords"][8], reference)
                    np.testing.assert_array_equal(inv["query_yx"], grid())
                    np.testing.assert_array_equal(inv["depth_vox"], np.zeros((32, 32)))
                    self.assertTrue(inv["valid"].all())

    def test_tangent_hybrid_rays_are_rejected_as_singular(self):
        surface, _ = plane()
        for axis in ("x", "y"):
            for family in ("normal_only", "anchor_only"):
                with self.subTest(axis=axis, family=family):
                    arm = patch(surface, family, axis, 90)
                    inv = fg.inverse_readback(surface, 0, 0, arm, pitch_um=PITCH)
                    self.assertFalse(inv["valid"].any())

    def test_factorial_coordinate_interaction_is_only_fp32_roundoff(self):
        surface, _ = plane(large_offset=True)
        for axis in ("x", "y"):
            for angle in ANGLES:
                arms = {f:patch(surface, f, axis, angle)["coords"] for f in FAMILIES}
                interaction = (arms["rigid"].astype(float)-arms["anchor_only"]
                               -arms["normal_only"]+arms["identity"])
                largest = np.float32(max(float(np.max(np.abs(a))) for a in arms.values()))
                tolerance = 4 * float(np.spacing(largest))
                self.assertLessEqual(float(np.abs(interaction).max()), tolerance)

    def test_anchor_only_cylinder_inverse_has_an_independent_closed_form(self):
        surface, radius = cylinder()
        original = grid()
        t0 = (original[..., 1]-63.5)/radius
        for angle in (-20, 20):
            with self.subTest(angle=angle):
                theta = np.deg2rad(angle)
                # In the Z+iX section, B = R exp(i*t0)-R+R exp(i*theta).
                # B = exp(i*t) * (R exp(i*theta)-d) for the inward source normal.
                b = radius * (np.exp(1j*t0)-1+np.exp(1j*theta))
                radial = np.sqrt(np.abs(b)**2 - (radius*np.sin(theta))**2)
                depth = radius*np.cos(theta)-radial
                t = np.angle(b)-np.arctan2(radius*np.sin(theta), radial)
                expected = original.copy()
                expected[..., 1] = 63.5 + radius*t
                arm = patch(surface, "anchor_only", "y", angle, pitch=1.0, offsets=np.arange(-8, 8))
                inv = fg.inverse_readback(surface, 0, 0, arm, pitch_um=1.0)
                self.assertTrue(inv["valid"].all())
                np.testing.assert_allclose(inv["query_yx"], expected, rtol=0, atol=.005)
                np.testing.assert_allclose(inv["depth_vox"], depth, rtol=0, atol=.005)

    def test_missing_source_geometry_is_not_restored_by_rotating_rays(self):
        surface, _ = plane()
        surface.valid[27, 27] = False
        reference_valid = surface.points(np.arange(48, 80), np.arange(48, 80), PITCH)[2]
        self.assertFalse(reference_valid.all())
        arm = patch(surface, "normal_only", "y", 20)
        inv = fg.inverse_readback(surface, 0, 0, arm, pitch_um=PITCH)
        np.testing.assert_array_equal(inv["valid"], reference_valid)


if __name__ == "__main__":
    unittest.main()
