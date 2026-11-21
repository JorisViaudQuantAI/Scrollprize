"""Input-position and provenance checks; synthetic data only, no ink claims."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare import prepare_file, select_planes
from verify_prepared import verify_prepared_file


def stack(depth):
    # Each plane has a unique nonzero value so misplaced or reversed planes are
    # detectable independently of the adapter's own metadata.
    return np.broadcast_to(np.arange(1, depth + 1, dtype=np.uint8)[:, None, None],
                           (depth, 3, 4)).copy()


def options(mesh=20, **kwargs):
    result = dict(mesh_plane_index=mesh, xy_spacing_um=(9.36, 9.36),
                  depth_spacing_um=9.36, index_direction=1, order="release-descending")
    result.update(kwargs)
    return result


class SelectionTests(unittest.TestCase):
    def test_published_31_plane_mapping_and_41_plane_translation(self):
        a, ma = select_planes(stack(31), **options(mesh=15))
        b, mb = select_planes(stack(41), **options())
        np.testing.assert_array_equal(a[:, 0, 0], np.arange(24, 8, -1))
        np.testing.assert_array_equal(b[:, 0, 0], np.arange(29, 13, -1))
        self.assertEqual(ma["source_plane_indices"], list(range(23, 7, -1)))
        self.assertEqual(mb["source_plane_indices"], list(range(28, 12, -1)))
        self.assertEqual(ma["sample_offsets_um_along_declared_mesh_normal"],
                         mb["sample_offsets_um_along_declared_mesh_normal"])
        self.assertAlmostEqual(mb["sample_band_midpoint_offset_um"], 4.68)
        self.assertAlmostEqual(mb["sample_centers_span_um"], 140.4)

    def test_same_band_reversal_is_not_whole_stack_reextraction(self):
        source = stack(41)
        forward, _ = select_planes(source, **options())
        reverse, _ = select_planes(source, **options(order="same-band-reversed"))
        whole_stack, meta = select_planes(source[::-1], **options(index_direction=-1))
        np.testing.assert_array_equal(reverse, forward[::-1])
        np.testing.assert_array_equal(reverse[:, 0, 0], np.arange(14, 30))
        np.testing.assert_array_equal(whole_stack[:, 0, 0], np.arange(13, 29))
        self.assertFalse(np.array_equal(whole_stack, reverse))
        self.assertAlmostEqual(meta["sample_band_midpoint_offset_um"], -4.68)

    def test_normal_sign_changes_physical_offsets_not_source_plane_choice(self):
        positive, p = select_planes(stack(41), **options())
        negative, n = select_planes(stack(41), **options(index_direction=-1))
        np.testing.assert_array_equal(positive, negative)
        np.testing.assert_allclose(n["sample_offsets_um_along_declared_mesh_normal"],
                                   -np.array(p["sample_offsets_um_along_declared_mesh_normal"]))

    def test_actual_anisotropic_spacing_is_preserved(self):
        _, meta = select_planes(stack(41), **options(xy_spacing_um=(9.36, 9.37),
                                                     depth_spacing_um=8.64))
        self.assertEqual(meta["xy_spacing_um_yx"], [9.36, 9.37])
        self.assertAlmostEqual(meta["sample_offsets_um_along_declared_mesh_normal"][0], 69.12)

    def test_intensities_above_upstream_clip_threshold_are_unchanged(self):
        value = stack(41)
        value[28, 1, 2] = 255
        output, meta = select_planes(value, **options())
        self.assertEqual(int(output[0, 1, 2]), 255)
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(meta["adapter_intensity_transform"], "none")

    def test_wrong_dtype_or_shape_fails(self):
        for value in (stack(41).astype(np.uint16), stack(41).astype(float),
                      np.ones((41, 3), np.uint8), np.ones((41, 0, 3), np.uint8)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    select_planes(value, **options())

    def test_bounds_never_pad_or_assume_stack_midpoint(self):
        for depth, mesh in ((28, 20), (41, 6), (41, 33), (41, -1), (41, 41)):
            with self.subTest(depth=depth, mesh=mesh):
                with self.assertRaises(ValueError):
                    select_planes(stack(depth), **options(mesh=mesh))
        # A caller-proven mesh location in a 28-plane stack is accepted, but is
        # not the centered 6:22 prepared-source special case from Studio's data.
        result, _ = select_planes(stack(28), **options(mesh=13))
        np.testing.assert_array_equal(result[:, 0, 0], np.arange(22, 6, -1))

    def test_zero_required_plane_rejected_without_rejecting_individual_zeros(self):
        value = stack(41)
        value[0] = 0  # Outside the requested band.
        value[20, 0, 0] = 0  # A measured zero need not mean missing data.
        select_planes(value, **options())
        value[20] = 0
        with self.assertRaisesRegex(ValueError, "entirely zero"):
            select_planes(value, **options())

    def test_coverage_mask_requires_boolean_full_shape_and_selected_coverage(self):
        value = stack(41)
        for mask in (np.ones_like(value, np.uint8), np.ones(value.shape[1:], bool),
                     np.ones((40, 3, 4), bool), np.zeros_like(value, bool)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    select_planes(value, coverage_mask=mask, **options())
        mask = np.ones_like(value, bool)
        mask[0] = False  # Outside selected band is irrelevant to this output.
        select_planes(value, coverage_mask=mask, **options())
        mask[13, 1, 2] = False
        with self.assertRaisesRegex(ValueError, "uncovered"):
            select_planes(value, coverage_mask=mask, **options())

    def test_missing_coverage_is_reported_and_not_faked(self):
        value = stack(41)
        value[:, 0, 0] = 0
        _, meta = select_planes(value, **options())
        self.assertEqual(meta["coverage_check"], "unverified: no acquisition-coverage mask supplied")
        self.assertEqual(meta["all_zero_depth_column_count"], 1)

    def test_invalid_geometry_metadata_rejected(self):
        invalid = [dict(mesh_plane_index=20.0), dict(mesh_plane_index=True),
                   dict(xy_spacing_um=(9.36,)), dict(xy_spacing_um=(9.36, 0)),
                   dict(xy_spacing_um=(9.36, float("nan"))), dict(depth_spacing_um=-1),
                   dict(depth_spacing_um=float("inf")), dict(depth_spacing_um=True),
                   dict(index_direction=0), dict(index_direction=True), dict(order="forward")]
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    select_planes(stack(41), **options(**override))


class FileTests(unittest.TestCase):
    def test_hashes_metadata_and_values_round_trip_without_recording_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "do-not-copy-source-name.npy"
            coverage = folder / "coverage.npy"
            output = folder / "planes16.npy"
            np.save(source, stack(41))
            np.save(coverage, np.ones((41, 3, 4), bool))
            record = prepare_file(source, output, coverage_mask_path=coverage, **options())
            disk = json.loads(output.with_suffix(".npy.json").read_text())
            self.assertEqual(record, disk)
            for path, key in ((source, "input_file_sha256"), (output, "output_file_sha256"),
                              (coverage, "coverage_mask_file_sha256")):
                self.assertEqual(record[key], hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_array_equal(np.load(output), stack(41)[28:12:-1])
            self.assertNotIn(str(folder), json.dumps(record))
            self.assertNotIn(source.name, json.dumps(record))
            self.assertEqual(record["adapter_intensity_transform"], "none")
            self.assertEqual(len(record["adapter_file_sha256"]), 64)

    def test_invalid_input_produces_neither_output_nor_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.npy"
            output = Path(directory) / "output.npy"
            np.save(source, stack(41).astype(np.uint16))
            with self.assertRaises(ValueError):
                prepare_file(source, output, **options())
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".npy.json").exists())

    def test_existing_output_or_sidecar_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.npy"
            output = Path(directory) / "output.npy"
            np.save(source, stack(41))
            for existing in (output, output.with_suffix(".npy.json")):
                existing.write_bytes(b"untouched")
                with self.assertRaises(FileExistsError):
                    prepare_file(source, output, **options())
                self.assertEqual(existing.read_bytes(), b"untouched")
                existing.unlink()

    def test_prepared_verifier_checks_shape_and_does_not_invent_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "released_input.npy"
            np.save(source, stack(16))
            record = verify_prepared_file(source)
            self.assertEqual(record["input_file_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIsNone(record["mesh_plane_index"])
            self.assertIsNone(record["sample_offsets_um_along_declared_mesh_normal"])
            self.assertIn("unverified", record["depth_order_check"])
            self.assertIn("unverified", record["coverage_check"])
            self.assertEqual(list(Path(directory).iterdir()), [source])
            np.save(source, stack(15))
            with self.assertRaises(ValueError):
                verify_prepared_file(source)

    def test_prepared_verifier_applies_dtype_zero_and_mask_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "released_input.npy"
            mask = Path(directory) / "coverage.npy"
            for invalid in (stack(16).astype(np.uint16), np.zeros((16, 3, 4), np.uint8)):
                np.save(source, invalid)
                with self.assertRaises(ValueError):
                    verify_prepared_file(source)
            np.save(source, stack(16))
            for invalid in (np.ones((16, 3, 4), np.uint8), np.ones((3, 4), bool),
                            np.zeros((16, 3, 4), bool)):
                np.save(mask, invalid)
                with self.assertRaises(ValueError):
                    verify_prepared_file(source, coverage_mask_path=mask)
            np.save(mask, np.ones((16, 3, 4), bool))
            record = verify_prepared_file(source, coverage_mask_path=mask)
            self.assertEqual(record["coverage_mask_file_sha256"], hashlib.sha256(mask.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
