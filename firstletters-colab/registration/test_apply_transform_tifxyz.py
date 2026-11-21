#!/usr/bin/env python3
"""Self-test for apply_transform_tifxyz.py (CPU, a second, no data download).

Reproduces the review case: a 2 x 2 mesh with an identity transform where only one vertex has three
finite, non-negative coordinates (the others have NaN in x, a negative y, and the tifxyz -1 marker).
Checks that exactly one vertex is valid, that the other three are written as -1 in every channel,
that the bbox is finite and equals the valid vertex, and that the output names the destination volume.

usage: python test_apply_transform_tifxyz.py [-v]
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apply_transform_tifxyz as T  # noqa: E402

GOOD = (10.0, 20.0, 30.0)


def write_fixture(root, matrix, fixed="FIXED.zarr", moving=None, meta_extra=None):
    src = os.path.join(root, "src")
    os.makedirs(src)
    X = np.array([[GOOD[0], np.nan], [5.0, -1.0]], np.float32)
    Y = np.array([[GOOD[1], 7.0], [-3.0, -1.0]], np.float32)
    Z = np.array([[GOOD[2], 8.0], [9.0, -1.0]], np.float32)
    for c, a in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(os.path.join(src, f"{c}.tif"), a)
    meta = {"format": "tifxyz", "type": "seg", "scale": [0.05, 0.05], "bbox": [[0, 0, 0], [1, 1, 1]],
            "uuid": "src", "volume": "SOURCE.zarr", "target_volume": "SOURCE-surface.zarr"}
    meta.update(meta_extra or {})
    json.dump(meta, open(os.path.join(src, "meta.json"), "w"))
    tr = {"schema_version": "1.0.0", "fixed_volume": fixed, "transformation_matrix": matrix}
    if moving:
        tr["moving_volume"] = moving
    tp = os.path.join(root, "transform.json")
    json.dump(tr, open(tp, "w"))
    return tp, src


def read_out(dst):
    X, Y, Z = [tifffile.imread(os.path.join(dst, f"{c}.tif")) for c in "xyz"]
    return X, Y, Z, json.load(open(os.path.join(dst, "meta.json")))


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]


class TestApplyTransform(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_review_four_vertex_case(self):
        tp, src = write_fixture(self.root, IDENTITY)
        dst = os.path.join(self.root, "dst")
        n = T.main([tp, src, dst])
        self.assertEqual(n, 1)
        X, Y, Z, m = read_out(dst)
        valid = (X != -1) & (Y != -1) & (Z != -1)
        self.assertEqual(int(valid.sum()), 1)
        self.assertTrue(valid[0, 0])
        for a in (X, Y, Z):  # invalid vertices are -1 in every channel, never NaN
            self.assertTrue(np.all(a[~valid] == -1))
            self.assertTrue(np.all(np.isfinite(a)))
        bbox = np.array(m["bbox"])
        self.assertTrue(np.all(np.isfinite(bbox)))
        np.testing.assert_allclose(bbox, [GOOD, GOOD])
        self.assertEqual(m["volume"], "FIXED.zarr")
        self.assertNotIn("target_volume", m)
        self.assertEqual(m["source_frame"]["volume"], "SOURCE.zarr")

    def test_scaled_transform_and_bbox(self):
        S = [[0.25, 0, 0, 1], [0, 0.25, 0, 2], [0, 0, 0.25, 3]]
        tp, src = write_fixture(self.root, S)
        dst = os.path.join(self.root, "dst")
        T.main([tp, src, dst, "--level", "2"])  # level 2: coordinates x4 first, then the matrix
        X, Y, Z, m = read_out(dst)
        np.testing.assert_allclose([X[0, 0], Y[0, 0], Z[0, 0]], [GOOD[0] + 1, GOOD[1] + 2, GOOD[2] + 3], rtol=1e-6)
        np.testing.assert_allclose(m["bbox"][0], m["bbox"][1])

    def test_inverse_frame_from_provenance(self):
        fit = os.path.join(self.root, "pair", "fit")
        os.makedirs(fit)
        tp, src = write_fixture(self.root, IDENTITY)
        os.replace(tp, os.path.join(fit, "transform.json"))
        tp = os.path.join(fit, "transform.json")
        json.dump({"moving_volume": "MOVING.zarr", "fixed_volume": "FIXED.zarr"},
                  open(os.path.join(self.root, "pair", "provenance.json"), "w"))
        dst = os.path.join(self.root, "dst")
        T.main([tp, src, dst, "--inverse"])
        self.assertEqual(read_out(dst)[3]["volume"], "MOVING.zarr")

    def test_inverse_without_destination_is_refused(self):
        tp, src = write_fixture(self.root, IDENTITY)
        with self.assertRaises(SystemExit):
            T.main([tp, src, os.path.join(self.root, "dst"), "--inverse"])
        dst = os.path.join(self.root, "dst2")
        T.main([tp, src, dst, "--inverse", "--dst-volume-id", "GIVEN.zarr"])
        self.assertEqual(read_out(dst)[3]["volume"], "GIVEN.zarr")

    def test_zrange_uses_input_z(self):
        tp, src = write_fixture(self.root, IDENTITY)
        dst = os.path.join(self.root, "dst")
        self.assertEqual(T.main([tp, src, dst, "--zrange", "31", "40"]), 0)
        self.assertNotIn("bbox", read_out(dst)[3])


if __name__ == "__main__":
    unittest.main()
