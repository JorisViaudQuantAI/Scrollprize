#!/usr/bin/env python3
"""Generated checks only: no CT file, source download, label or model access."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import analyze_local_structure as impl


def translated_texture():
    rng = np.random.default_rng(20261122)
    base = impl.smooth_valid(rng.normal(size=(61, 61, 61)), 2)
    # A 57³ field leaves complete support for both crops and the known shift.
    truth = np.array([2, -1, 3])
    fine = impl.crop(base, 33)
    coarse = 2.3 * impl.crop(base, 33, -truth) + 17
    return fine, coarse, truth


def full_field(fine, coarse, passes=0, mask=None):
    a, b = impl.smooth_valid(fine, passes), impl.smooth_valid(coarse, passes)
    return impl.correlation_field(impl.crop(a, 17), impl.crop(b, 25),
                                  impl.masks()[0] if mask is None else mask)


def generated_report():
    fine, coarse, truth = translated_texture()
    textured = [dict(binomial_passes=p, **impl.peak_summary(full_field(fine, coarse, p))) for p in impl.PASSES]
    z, y, x = np.meshgrid(np.arange(-16, 17), np.arange(-16, 17), np.arange(-16, 17), indexing="ij")
    slab_a = np.exp(-y.astype(float) ** 2 / 9)
    slab_b = 4 * np.exp(-(y - 2).astype(float) ** 2 / 9) + 6
    slab = impl.peak_summary(full_field(slab_a, slab_b))
    periodic = np.cos(2 * np.pi * x / 4)
    bands = impl.peak_summary(full_field(periodic, periodic))
    rng = np.random.default_rng(20261122)
    a, b = rng.normal(size=(33, 33, 33)), rng.normal(size=(33, 33, 33))
    noise = []
    for p in impl.PASSES:
        sa, sb = impl.smooth_valid(a, p), impl.smooth_valid(b, p)
        qa, qb = impl.structure_metrics(a, sa, 0), impl.structure_metrics(b, sb, 0)
        noise.append({"binomial_passes": p, "independent_fields": True,
                      "a_lag_pearson": qa["lag_pearson_positive_xyz"],
                      "b_lag_pearson": qb["lag_pearson_positive_xyz"],
                      "a_tensor_eigenvalue_fractions": qa["tensor_eigenvalue_fractions_descending"],
                      "b_tensor_eigenvalue_fractions": qb["tensor_eigenvalue_fractions_descending"],
                      "cross_scan": impl.peak_summary(full_field(a, b, p))})
    return {"seed": 20261122, "generated_only": True, "known_shift_xyz": truth.tolist(),
            "translated_texture": textured, "planar_slab": slab, "periodic_bands": bands,
            "independent_noise": noise,
            "interpretation": "High lag coherence can result from filtering independent noise; high NCC can leave translation directions or periodic alternatives ambiguous. These are implementation/adversarial checks, not anatomical calibration."}


class Checks(unittest.TestCase):
    def test_constant_is_undefined(self):
        rng = np.random.default_rng(71)
        for value in (1., .1, .7, 1e-12):
            for mask in impl.masks():
                out = full_field(np.full((33,) * 3, value), rng.normal(size=(33,) * 3), mask=mask)
                self.assertTrue(np.all(np.isnan(out)))
                self.assertIsNone(impl.peak_summary(out)["zero_shift"])
            self.assertIsNone(impl.pearson(np.full((17,) * 3, value), rng.normal(size=(17,) * 3)))
            self.assertIsNone(impl.pearson(rng.normal(size=(17,) * 3), np.full((17,) * 3, value)))
            metrics = impl.structure_metrics(np.full((33,) * 3, value), np.full((33,) * 3, value), 0)
            self.assertEqual(metrics["sd"], 0)
            self.assertIsNone(metrics["variance_retention"])

    def test_known_shift_and_positive_intensity_affine(self):
        a, b, truth = translated_texture()
        for p in impl.PASSES:
            r = impl.peak_summary(full_field(a, b, p))
            self.assertEqual(r["argmax_shift_xyz"], truth.tolist())
            self.assertAlmostEqual(r["maximum"], 1, places=12)

    def test_fft_against_direct_selected_dot_products(self):
        rng = np.random.default_rng(73)
        a, b = rng.normal(size=(33,) * 3), rng.normal(size=(33,) * 3)
        for p in impl.PASSES:
            sa, sb = impl.smooth_valid(a, p), impl.smooth_valid(b, p)
            for mask in impl.masks():
                field = full_field(a, b, p, mask)
                for shift in ((0, 0, 0), (4, -4, 1), (-3, 2, -4), (4, 4, 4)):
                    x, y = impl.crop(sa, 17)[mask.astype(bool)], impl.crop(sb, 17, shift)[mask.astype(bool)]
                    direct = np.corrcoef(x, y)[0, 1]
                    idx = tuple(np.asarray(shift)[::-1] + 4)
                    self.assertAlmostEqual(field[idx], direct, places=12)

    def test_finite_filter_has_expected_coefficients(self):
        a = np.zeros((17,) * 3)
        a[8, 8, 8] = 1
        b = impl.smooth_valid(a, 1)
        kernel = np.array([.25, .5, .25])
        expected = np.einsum("i,j,k->ijk", kernel, kernel, kernel)
        np.testing.assert_array_equal(b[6:9, 6:9, 6:9], expected)
        self.assertEqual(float(b.sum()), 1)

    def test_planar_aperture_ambiguity(self):
        r = generated_report()["planar_slab"]
        self.assertAlmostEqual(r["maximum"], 1, places=12)
        self.assertGreaterEqual(r["within_0_01_of_maximum"], 81)

    def test_periodic_ambiguity(self):
        r = generated_report()["periodic_bands"]
        self.assertGreaterEqual(r["within_0_01_of_maximum"], 243)

    def test_smoothing_independent_noise_creates_coherence(self):
        r = generated_report()["independent_noise"]
        self.assertLess(abs(r[0]["a_lag_pearson"]["x_1"]), .1)
        self.assertGreater(r[-1]["a_lag_pearson"]["x_1"], .7)
        self.assertGreater(r[-1]["b_lag_pearson"]["x_1"], .7)

    def test_no_padding_and_core_support(self):
        a = np.ones((33,) * 3)
        b = impl.smooth_valid(a, 4)
        self.assertEqual(b.shape, (25,) * 3)
        self.assertEqual(impl.crop(b, 17, (4, 4, 4)).size, 17 ** 3)
        with self.assertRaises(ValueError):
            impl.crop(b, 17, (5, 0, 0))

    def test_slab_masks_are_disjoint_and_declared(self):
        full, left, right = impl.masks()
        self.assertEqual(full.sum(), 17 ** 3)
        self.assertEqual(left.sum(), 4 * 17 ** 2)
        self.assertEqual(right.sum(), 4 * 17 ** 2)
        self.assertEqual((left * right).sum(), 0)
        # Central raw supports of a fixed shift: [-12,-1] and [1,12].
        self.assertLess((-5 + 4), (5 - 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = generated_report()
    report["tests"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful()}
    report["runner_sha256"] = hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest()
    report["test_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out:
        impl.write_json(args.out, report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
