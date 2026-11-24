"""Independent analytic controls for the rigid native-volume geometry.

No network, model, or real holdout pixels are needed. The synthetic mesh adapter
uses the author's pixel-center and signed-normal conventions. Expected inverse
coordinates come from closed-form planes/cylinders, not producer inversions.
"""
import unittest

import numpy as np

import geometry as g


PITCH = 9.36
CT_UM = 9.366
OFFSETS = np.arange(-8, 8, dtype=np.float64) * CT_UM


class SyntheticSurface:
    """Minimal upstream Surface interface, evaluated independently on a grid."""

    def __init__(self, points, mesh_pitch, ct_um=CT_UM, sign=1):
        self.a = np.asarray(points, dtype=np.float32)
        self.valid = np.ones(self.a.shape[:2], dtype=bool)
        self.pitch = np.asarray(mesh_pitch, dtype=np.float64)
        self.ct_um = float(ct_um)
        self.sign = sign

    def points(self, ys, xs, pitch_um):
        yy, xx = np.meshgrid(np.asarray(ys, np.float32),
                             np.asarray(xs, np.float32), indexing="ij")
        yy = (yy + np.float32(.5)) * np.float32(pitch_um / self.pitch[1])
        xx = (xx + np.float32(.5)) * np.float32(pitch_um / self.pitch[0])
        iy, ix = np.floor(yy).astype(int), np.floor(xx).astype(int)
        valid = ((iy >= 0) & (ix >= 0) & (iy < self.a.shape[0] - 1)
                 & (ix < self.a.shape[1] - 1))
        iy = np.clip(iy, 0, self.a.shape[0] - 2)
        ix = np.clip(ix, 0, self.a.shape[1] - 2)
        v, u = yy - iy.astype(np.float32), xx - ix.astype(np.float32)
        aa, bb = self.a[iy, ix], self.a[iy, ix + 1]
        cc, dd = self.a[iy + 1, ix], self.a[iy + 1, ix + 1]
        xyz = ((1-u)[..., None]*(1-v)[..., None]*aa
               + u[..., None]*(1-v)[..., None]*bb
               + (1-u)[..., None]*v[..., None]*cc
               + u[..., None]*v[..., None]*dd)
        col = (1-v)[..., None]*(bb-aa) + v[..., None]*(dd-cc)
        row = (1-u)[..., None]*(cc-aa) + u[..., None]*(dd-bb)
        normal = np.cross(row, col)
        length = np.linalg.norm(normal, axis=-1)
        valid &= (self.valid[iy, ix] & self.valid[iy, ix + 1]
                  & self.valid[iy + 1, ix] & self.valid[iy + 1, ix + 1]
                  & (length > 1e-6))
        normal *= self.sign / np.maximum(length, 1e-6)[..., None]
        xyz[~valid], normal[~valid] = -1, 0
        return xyz, normal, valid


def plane(sign=1):
    yy, xx = np.mgrid[:80, :80]
    step = 2 * PITCH / CT_UM
    a = np.stack((100 + step*xx, 200 + step*yy,
                  np.full(xx.shape, 300.0)), axis=-1)
    return SyntheticSurface(a, [2*PITCH, 2*PITCH], sign=sign)


def rotation(axis, degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def core_grid():
    return np.stack(np.meshgrid(np.arange(48, 80), np.arange(48, 80),
                               indexing="ij"), axis=-1).astype(float)


class GeometryOracles(unittest.TestCase):
    def test_identity_preserves_source_bytes_and_zero_depth_anchor(self):
        surface = plane()
        xyz, normal, valid = surface.points(np.arange(128), np.arange(128), PITCH)
        native_offsets = np.asarray(OFFSETS, np.float32) / surface.ct_um
        expected = xyz[None] + normal[None]*native_offsets[:, None, None, None]
        for axis in ("x", "y"):
            with self.subTest(axis=axis):
                arm = g.rotate_patch(surface, 0, 0, 0, axis=axis,
                                     pitch_um=PITCH, offsets_um=OFFSETS)
                self.assertEqual(arm["coords"].dtype, np.float32)
                np.testing.assert_array_equal(arm["coords"], expected)
                np.testing.assert_array_equal(arm["coords"][8], xyz)
                np.testing.assert_array_equal(arm["mesh_valid"], valid)
                readback = g.inverse_readback(surface, 0, 0, arm["rotation"],
                                              arm["pivot_xyz"], pitch_um=PITCH)
                np.testing.assert_array_equal(readback["query_yx"], core_grid())
                np.testing.assert_array_equal(readback["depth_vox"], np.zeros((32, 32)))
                self.assertTrue(np.all(readback["valid"]))

    def test_world_rotation_is_rigid_about_the_named_axis(self):
        surface = plane()
        xyz, normal, _ = surface.points(np.arange(128), np.arange(128), PITCH)
        base = xyz[None] + normal[None] * (
            np.asarray(OFFSETS, np.float32) / CT_UM)[:, None, None, None]
        center = surface.points([63.5], [63.5], PITCH)[0][0, 0]
        for axis in ("x", "y"):
            for angle in (-20, 5, 20):
                with self.subTest(axis=axis, angle=angle):
                    arm = g.rotate_patch(surface, 0, 0, angle, axis=axis,
                                         pitch_um=PITCH, offsets_um=OFFSETS)
                    matrix = rotation(axis, angle)
                    expected = center + (base.astype(float)-center) @ matrix.T
                    np.testing.assert_allclose(arm["rotation"], matrix, atol=1e-12)
                    np.testing.assert_allclose(arm["coords"], expected, rtol=0, atol=5e-5)
                    self.assertAlmostEqual(np.linalg.det(arm["rotation"]), 1, places=12)

    def test_planar_inverse_uses_cosine_and_physical_depth_units(self):
        surface, original = plane(), core_grid()
        relative = original - 63.5
        for axis in ("x", "y"):
            for angle in (-20, -10, -5, 5, 10, 20):
                with self.subTest(axis=axis, angle=angle):
                    arm = g.rotate_patch(surface, 0, 0, angle, axis=axis,
                                         pitch_um=PITCH, offsets_um=OFFSETS)
                    result = g.inverse_readback(surface, 0, 0, arm["rotation"],
                                                arm["pivot_xyz"], pitch_um=PITCH,
                                                residual_tol=.01)
                    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
                    expected = original.copy()
                    changing = 0 if axis == "x" else 1
                    expected[..., changing] = 63.5 + relative[..., changing]*c
                    # The author's cross(drow,dcol) normal is -world-Z here.
                    sign = 1 if axis == "x" else -1
                    depth = sign * relative[..., changing]*(PITCH/CT_UM)*s
                    self.assertTrue(np.all(result["valid"]))
                    np.testing.assert_allclose(result["query_yx"], expected, atol=2e-4, rtol=0)
                    np.testing.assert_allclose(result["depth_vox"], depth, atol=2e-4, rtol=0)
                    self.assertLessEqual(float(np.max(result["residual_vox"])), .01)

    def test_reversing_normal_changes_depth_not_fixed_point_uv(self):
        results = []
        for sign in (1, -1):
            surface = plane(sign)
            arm = g.rotate_patch(surface, 0, 0, 20, axis="y", offsets_um=OFFSETS)
            results.append(g.inverse_readback(surface, 0, 0, arm["rotation"],
                                              arm["pivot_xyz"], residual_tol=.01))
        np.testing.assert_allclose(results[0]["query_yx"], results[1]["query_yx"], atol=1e-9)
        np.testing.assert_allclose(results[0]["depth_vox"], -results[1]["depth_vox"], atol=1e-9)
        self.assertTrue(np.all(results[0]["valid"] & results[1]["valid"]))

    def test_curved_cylinder_has_independent_closed_form_inverse(self):
        radius = 300.0
        yy, xx = np.mgrid[:70, :1100]
        u, v = xx*.125-64, yy*2.0
        mesh = np.stack((200 + radius*np.sin(u/radius), 100 + v,
                         200 + radius*(np.cos(u/radius)-1)), axis=-1)
        surface = SyntheticSurface(mesh, [.125, 2.0], ct_um=1.0)
        original = core_grid()
        u0, v0 = original[..., 1]-63.5, original[..., 0]+.5
        targets = np.stack((200 + radius*np.sin(u0/radius), 100 + v0,
                            200 + radius*(np.cos(u0/radius)-1)), axis=-1)
        pivot = np.array([200.0, 164.0, 200.0])
        for angle in (-20, 20):
            with self.subTest(angle=angle):
                arm = g.rotate_patch(surface, 0, 0, angle, axis="y",
                                     pitch_um=1.0, offsets_um=np.arange(-8, 8))
                result = g.inverse_readback(surface, 0, 0, arm["rotation"],
                                            arm["pivot_xyz"], pitch_um=1.0,
                                            residual_tol=.01)
                # Closed-form inverse of the continuous cylinder; mesh chord and
                # FP32 errors are bounded by a conservative .003-voxel tolerance.
                y = pivot + (targets-pivot) @ rotation("y", angle)
                dx, dz = y[..., 0]-200, y[..., 2]-200+radius
                expected = np.stack((y[..., 1]-100-.5,
                                     63.5+radius*np.arctan2(dx, dz)), axis=-1)
                depth = radius-np.hypot(dx, dz)
                self.assertTrue(np.all(result["valid"]))
                np.testing.assert_allclose(result["query_yx"], expected, atol=.003, rtol=0)
                np.testing.assert_allclose(result["depth_vox"], depth, atol=.003, rtol=0)

    def test_missing_mesh_corner_is_not_valid_identity_geometry(self):
        surface = plane()
        surface.valid[27, 27] = False
        arm = g.rotate_patch(surface, 0, 0, 0, offsets_um=OFFSETS)
        # The missing corner affects the scored core but not the rotation pivot.
        source_valid = surface.points(np.arange(48, 80), np.arange(48, 80), PITCH)[2]
        self.assertFalse(np.all(source_valid))
        readback = g.inverse_readback(surface, 0, 0, arm["rotation"],
                                      arm["pivot_xyz"], pitch_um=PITCH)
        np.testing.assert_array_equal(readback["valid"], source_valid)
        np.testing.assert_array_equal(arm["mesh_valid"][48:80, 48:80], source_valid)

    def test_normal_seam_gap_abstains_instead_of_inventing_a_root(self):
        # V-shaped surface z=|x| with its downward normals. The point (0,-1)
        # in the XZ section belongs to neither half-plane's normal tube: each
        # algebraic solution lies on the wrong half-plane. A normal seam is a
        # real inverse-support gap even though every mesh vertex is valid.
        yy, xx = np.mgrid[:80, :80]
        u = 2*xx-64
        mesh = np.stack((200+u, 100+2*yy, 300+np.abs(u)), axis=-1)
        surface = SyntheticSurface(mesh, [2.0, 2.0], ct_um=1.0)
        original = surface.points([63], [63], 1.0)[0][0, 0].astype(float)
        inverse_target = np.array([200.0, original[1], 299.0])
        # A half-turn about this finite pivot maps the reference point to the
        # prescribed gap. This is an inverse-support fixture, not a CNN arm.
        pivot = (original + inverse_target)/2
        result = g.inverse_readback(surface, 0, 0, rotation("y", 180), pivot,
                                    targetcore=(63, 64), pitch_um=1.0,
                                    residual_tol=.01)
        self.assertFalse(bool(result["valid"][0, 0]))
        self.assertGreater(float(result["residual_vox"][0, 0]), .01)


if __name__ == "__main__":
    unittest.main()
