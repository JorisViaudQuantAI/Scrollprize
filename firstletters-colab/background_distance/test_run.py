"""Generated geometry/accounting tests; no retained predictions or CT opened."""
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("background_distance_under_test", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class GeometryAndAccountingTests(unittest.TestCase):
    def test_original_centres_match_fraction_mapping(self):
        origin = (17, 31)
        shape = (7, 11)
        points = runner.common_points(origin, shape)
        original = (np.arange(100 * 120, dtype=np.int64).reshape(100, 120) % 3).astype(np.uint8)
        mapped = runner.project_categories(original, origin, shape)
        expected = np.empty(shape, np.uint8)
        for y in range(shape[0]):
            for x in range(shape[1]):
                y_um = (Fraction(origin[0] + y) + Fraction(1, 2)) * Fraction(48, 5)
                x_um = (Fraction(origin[1] + x) + Fraction(1, 2)) * Fraction(48, 5)
                iy, ix = int(y_um / Fraction(234, 25)), int(x_um / Fraction(234, 25))
                expected[y, x] = original[iy, ix]
                self.assertEqual(Fraction(int(points[y * shape[1] + x, 0]), 25), y_um)
                self.assertEqual(Fraction(int(points[y * shape[1] + x, 1]), 25), x_um)
        np.testing.assert_array_equal(mapped, expected)

    def test_halo_contains_every_test_radius_without_clipping(self):
        origin = (100, 200)
        bounds = runner.halo_bounds(origin, (1000, 1200))
        query = runner.common_points(origin)
        for axis, (low, high) in enumerate(bounds):
            last_excluded_low_centre = (2 * (low - 1) + 1) * 117
            first_excluded_high_centre = (2 * high + 1) * 117
            self.assertGreater(int(query[:, axis].min()) - last_excluded_low_centre, 1920)
            self.assertGreater(first_excluded_high_centre - int(query[:, axis].max()), 1920)
        with self.assertRaises(runner.ContractError):
            runner.halo_bounds((0, 0), (1000, 1200))
        with self.assertRaises(runner.ContractError):
            runner.halo_bounds(origin, (300, 400))

    def test_nearest_centres_match_brute_force_and_int64(self):
        rng = np.random.default_rng(703)
        query = rng.integers(100_000, 1_000_000, (30, 2), dtype=np.int64)
        refs = rng.integers(100_000, 1_000_000, (13, 2), dtype=np.int64)
        expected = np.array([min(int(y - ry) ** 2 + int(x - rx) ** 2 for ry, rx in refs)
                             for y, x in query], dtype=np.int64)
        actual = runner.squared_distance_to_centres(query, refs)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue((actual > np.iinfo(np.uint32).max).any())
        self.assertEqual(actual.dtype, np.int64)

    def test_band_edges_are_inclusive_and_missing_category_is_beyond_radius(self):
        distances, expected = [], []
        for index, edge in enumerate((240, 480, 960, 1920)):
            distances.extend((edge * edge - 1, edge * edge, edge * edge + 1))
            expected.extend((index, index, index + 1))
        distances += [-1, 0]
        expected += [4, 0]
        np.testing.assert_array_equal(runner.distance_bins(np.array(distances, np.int64)), expected)
        no_reference = runner.squared_distance_to_centres(np.array([[120, 120]], np.int64), np.empty((0, 2), np.int64))
        self.assertEqual(int(no_reference[0]), -1)
        self.assertEqual(int(runner.distance_bins(no_reference)[0]), 4)

    def test_source_category_points_use_original_pitch_and_crop_offset(self):
        crop = np.array([[1, 2], [0, 1]], np.uint8)
        ink = runner.label_points(crop, (100, 200), 2)
        unknown = runner.label_points(crop, (100, 200), 0)
        np.testing.assert_array_equal(ink, [[201 * 117, 403 * 117]])
        np.testing.assert_array_equal(unknown, [[203 * 117, 401 * 117]])

    def test_support_boundary_is_flagged_without_becoming_a_hole(self):
        censored = runner.support_boundary_censored()
        self.assertEqual(int(censored.sum()), 256 * 256 - 240 * 240)
        self.assertTrue(censored[:8].all())
        self.assertFalse(censored[8:-8, 8:-8].any())
        original = np.ones((400, 400), np.uint8)
        original[75:90, 80:95] = 2
        original[100:110, 120:130] = 0
        support = np.ones((256, 256), bool)
        partitions, receipt = runner.label_geometry(original, (32, 32), support)
        self.assertTrue((partitions["observed_support_hole_distance"] == 4).all())
        self.assertEqual(receipt["unsupported_core_reference_centres"], 0)
        self.assertFalse(receipt["support_beyond_scoring_core_observed"])
        self.assertTrue(receipt["no_cells_removed_for_censorship"])
        np.testing.assert_array_equal(partitions["support_distance_censorship"], censored)

    def test_unchanged_strict_cutoffs_and_exact_export_saturation(self):
        f = np.full((256, 256), .5, np.float32)
        q = np.full_like(f, .5)
        f[0, :4] = [0., .99, 1., .75]
        q[0, :4] = [0., 1., 1., .75]
        own = {"float": {"threshold_strict_greater": .5}, "q8": {"threshold_strict_greater": 1.0}}
        cross = {"float": {"threshold_strict_greater": .75}, "q8": {"threshold_strict_greater": .5}}
        masks = runner.endpoint_masks(f, q, own, cross)
        self.assertEqual(int(masks["own_float"].sum()), 3)
        self.assertEqual(int(masks["own_q8"].sum()), 0)
        self.assertEqual(int(masks["cross_float"].sum()), 2)
        self.assertEqual(int(masks["cross_q8"].sum()), 3)
        self.assertEqual(int(masks["float_ge_0_99"].sum()), 2)
        self.assertEqual(int(masks["q8_equal_one"].sum()), 2)

    def test_every_original_positive_reconciles_in_each_partition(self):
        yy, xx = np.indices((256, 256))
        background = ((yy + 2 * xx) % 3 == 0)
        endpoints = {"many": xx % 2 == 0, "few": (yy == 0) & (xx < 17), "none": np.zeros((256, 256), bool)}
        assignments = (yy % 5).astype(np.uint8)
        rows = runner.partition_summary(background, endpoints, assignments, [str(i) for i in range(6)])
        self.assertEqual(sum(row["background_denominator"] for row in rows), int(background.sum()))
        for endpoint, positive in endpoints.items():
            self.assertEqual(sum(row["endpoints"][endpoint]["positive_count"] for row in rows), int((background & positive).sum()))
        self.assertEqual(rows[-1]["background_denominator"], 0)
        self.assertIsNone(rows[-1]["endpoints"]["many"]["positive_rate"])
        self.assertIsNone(rows[0]["endpoints"]["none"]["share_of_all_endpoint_positives"])
        self.assertGreater(rows[0]["endpoints"]["few"]["positive_count"], 0)
        # An invalid assignment must fail rather than silently erase a row.
        assignments[0, 0] = 99
        with self.assertRaises(runner.ContractError):
            runner.partition_summary(background, endpoints, assignments, [str(i) for i in range(6)])

    def test_consumed_hash_rejects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="background-distance-generated-") as temporary:
            path = Path(temporary) / "generated.txt"
            path.write_bytes(b"generated only")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            inputs = runner.Inputs(SimpleNamespace(check=lambda: None))
            inputs.check(path, digest, "generated_fixture")
            inputs.final_check()
            path.write_bytes(b"changed only")
            with self.assertRaises(runner.ContractError):
                inputs.final_check()

    def test_freeze_failure_precedes_all_inherited_array_access(self):
        args = SimpleNamespace(freeze_reference="0" * 40)
        with mock.patch.object(runner, "verify_freeze", side_effect=runner.ContractError("generated freeze refusal")), \
                mock.patch.object(runner, "read_array") as read_array, \
                mock.patch.object(runner.Image, "open") as open_image:
            with self.assertRaises(runner.ContractError):
                runner.run(args)
            read_array.assert_not_called()
            open_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
