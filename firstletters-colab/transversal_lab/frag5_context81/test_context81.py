#!/usr/bin/env python3
"""Generated sampling/overlap checks plus metadata-only plan verification."""
import argparse
import json
from pathlib import Path
import unittest

import numpy as np
import run_context81 as impl

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "protocol.json").read_bytes())
ORIGINAL = impl.load_module(HERE.parent / "frag5_native_pair/frag5_native_pair_stage1.py", PLAN["stage1_runner_sha256"], "test_context_original")
LOCAL = impl.load_module(HERE.parent / "frag5_local_structure/analyze_local_structure.py", PLAN["local_structure_runner_sha256"], "test_context_local")


class Checks(unittest.TestCase):
    def test_constant_variance_semantics(self):
        constant = np.full((65,) * 3, .1)
        varying = np.arange(65 ** 3, dtype=float).reshape((65,) * 3)
        self.assertIsNone(impl.variance_retention(constant, constant))
        self.assertEqual(impl.variance_retention(constant, varying), 0)

    def test_geometry_and_chunk_union(self):
        base = json.loads((HERE.parent / "frag5_native_pair/frag5_native_pair_stage1_protocol.json").read_bytes())
        impl.verify_plan(PLAN, base, ORIGINAL)
        self.assertEqual(sum(x["cached"] is not None for x in PLAN["source_objects"]), 6)
        cap = PLAN["caps"]
        self.assertLess(cap["prior_source_body_bytes"] + 43 * cap["new_object_body_bytes"], cap["cumulative_source_body_bytes"])

    def test_affine_linear_field_point_and_midpoint_integral(self):
        z, y, x = np.indices((64, 64, 64), dtype=float)
        region = ORIGINAL.Region(2 * x + 3 * y + 5 * z + 7, [0, 0, 0])
        a = np.array([[.5, .02, 0, 1], [.01, .51, .01, -1], [0, -.01, .49, 2], [0, 0, 0, 1]])
        center = np.array([15, 16, 14])
        point, footprint = impl.sample_grid(region, center, a, side=7)
        zz, yy, xx = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), np.arange(-3, 4), indexing="ij")
        q = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())) + center
        xyz = (q - a[:3, 3]) @ np.linalg.inv(a[:3, :3]).T
        expected = (xyz @ np.array([2, 3, 5]) + 7).reshape((7,) * 3)
        np.testing.assert_allclose(point, expected, atol=1e-11, rtol=0)
        np.testing.assert_allclose(footprint, expected, atol=1e-11, rtol=0)
        small_p, small_f = impl.sample_grid(region, center, a, side=3)
        np.testing.assert_allclose(LOCAL.crop(point, 3), small_p, atol=1e-11, rtol=0)
        np.testing.assert_allclose(LOCAL.crop(footprint, 3), small_f, atol=1e-11, rtol=0)

    def test_overlap_gate_catches_fine_and_native_change(self):
        fine = np.arange(81 ** 3, dtype=float).reshape((81,) * 3)
        coarse = np.arange(83 ** 3, dtype=float).reshape((83,) * 3)
        new = {"fine_point": fine.copy(), "fine_nominal_coarse_footprint": fine.copy() / 10, "coarse_neighborhood": coarse.copy()}
        old = {k: LOCAL.crop(a, 35 if k == "coarse_neighborhood" else 33).copy() for k, a in new.items()}
        self.assertTrue(impl.overlap_gate(new, old, LOCAL, 1e-8)["passed"])
        new["fine_point"][40, 40, 40] += 2e-8
        self.assertFalse(impl.overlap_gate(new, old, LOCAL, 1e-8)["passed"])
        new["fine_point"][40, 40, 40] = fine[40, 40, 40]
        new["coarse_neighborhood"][41, 41, 41] += 1
        self.assertFalse(impl.overlap_gate(new, old, LOCAL, 1e-8)["passed"])

    def test_native_halo_and_shell_size(self):
        a = np.ones((83,) * 3)
        self.assertEqual(LOCAL.crop(a, 81, (1, 1, 1)).shape, (81,) * 3)
        with self.assertRaises(ValueError):
            LOCAL.crop(a, 81, (2, 0, 0))
        shell = np.ones((81,) * 3, dtype=bool)
        shell[24:57, 24:57, 24:57] = False
        self.assertEqual(int(shell.sum()), 81 ** 3 - 33 ** 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    impl.write(args.out, {"generated_or_metadata_only": True, "real_ct_opened": False,
                         "runner_sha256": impl.sha(impl.__file__), "protocol_sha256": impl.sha(HERE / "protocol.json"),
                         "test_sha256": impl.sha(__file__), "tests": result.testsRun,
                         "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
