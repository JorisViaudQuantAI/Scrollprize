import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import support as s
import prepare_studio as p


class TileSupportTests(unittest.TestCase):
    def test_valid_zero_values_do_not_define_support(self):
        # Source values may all be zero: this API uses separate availability.
        available = np.ones((2, 6, 6), bool)
        tile = s.TileContribution((0, 0, 6, 6), (0, 0, 6, 6), np.ones((6, 6)))
        result = s.tile_context_support(available, np.ones((6, 6), bool), [tile])
        self.assertTrue(result.context_support_yx.all())
        self.assertEqual(result.counts()["source_all_required_pixels"], 36)

    def test_missing_context_voxel_invalidates_valid_centre(self):
        available = np.ones((2, 6, 6), bool)
        available[1, 3, 3] = False
        tile = s.TileContribution((0, 0, 4, 4), (0, 0, 4, 4), np.ones((4, 4)))
        result = s.tile_context_support(available, np.ones((6, 6), bool), [tile])
        self.assertTrue(result.source_all_required_yx[0, 0])
        self.assertTrue(result.contribution_support_yx[0, 0])
        self.assertFalse(result.context_support_yx[0, 0])
        self.assertEqual(result.counts()["context_supported_pixels"], 0)

    def test_zero_blend_weights_do_not_contaminate_good_contributor(self):
        available = np.ones((1, 5, 5), bool)
        available[0, 0, 0] = False
        good = s.TileContribution((1, 1, 3, 3), (1, 1, 3, 3), np.ones((2, 2)))
        weights = np.zeros((2, 2), dtype=np.float32)
        weights[0, 1] = 0.1
        bad = s.TileContribution((0, 0, 4, 4), (1, 1, 3, 3), weights)
        result = s.tile_context_support(available, np.ones((5, 5), bool), [good, bad])
        np.testing.assert_array_equal(result.context_support_yx[1:3, 1:3], [[True, False], [True, True]])
        self.assertEqual(result.contributor_count_yx[1, 1], 1)
        self.assertEqual(result.contributor_count_yx[1, 2], 2)

    def test_hann_zeros_are_uncovered_and_reflect_padding_is_not_real_context(self):
        window = np.outer(np.hanning(4), np.hanning(4))
        tile = s.TileContribution((-1, -1, 5, 5), (0, 0, 4, 4), window)
        result = s.tile_context_support(np.ones((1, 4, 4), bool), np.ones((4, 4), bool), [tile])
        self.assertEqual(result.counts()["contribution_supported_pixels"], 4)
        self.assertEqual(result.counts()["context_supported_pixels"], 0)

    def test_mesh_hole_and_source_hole_have_distinct_counts(self):
        mesh = np.ones((4, 4), bool); mesh[0, 0] = False
        tile = s.TileContribution((0, 0, 4, 4), (0, 0, 4, 4), np.ones((4, 4)))
        result = s.tile_context_support(np.ones((1, 4, 4), bool), mesh, [tile])
        self.assertEqual(result.counts()["source_all_required_pixels"], 16)
        self.assertEqual(result.counts()["mesh_valid_pixels"], 15)
        self.assertEqual(result.counts()["context_supported_pixels"], 0)

    def test_roi_retains_full_external_context(self):
        available = np.ones((1, 6, 6), bool); available[0, 0, 0] = False
        tile = s.TileContribution((0, 0, 6, 6), (0, 0, 6, 6), np.ones((6, 6)))
        result = s.tile_context_support(available, np.ones((6, 6), bool), [tile], output_roi_yxyx=np.array([2, 2, 4, 4]))
        self.assertTrue(result.source_all_required_yx.all())
        self.assertFalse(result.context_support_yx.any())

    def test_prediction_coverage_remains_independent(self):
        context = np.ones((3, 3), bool)
        predicted = np.ones((3, 3), bool); predicted[1, 1] = False
        score = np.zeros((3, 3), np.float32); score[1, 1] = np.nan
        actual = s.review_support(context, predicted, score)
        self.assertFalse(actual[1, 1])
        predicted[1, 1] = True
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            s.review_support(context, predicted, score)

    def test_depth_weights_partial_mass_reversal_and_zero_weight(self):
        available = np.ones((3, 2, 2), bool); available[1, 0, 0] = False
        contributors = [
            [{"source_layer": 0, "weight": 1.0}, {"source_layer": 99, "weight": 0.0}],
            [{"source_layer": 0, "weight": 0.5}, {"source_layer": 1, "weight": 0.5}],
            [{"source_layer": 2, "weight": 0.25}],
        ]
        planes = s.select_depth_support(available, contributors)
        self.assertTrue(planes[0].all())
        self.assertFalse(planes[1, 0, 0])
        self.assertFalse(planes[2].any())
        np.testing.assert_array_equal(s.select_depth_support(available, contributors[::-1]), planes[::-1])

    def test_bad_dtypes_shapes_weights_fail(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            s.tile_context_support(np.ones((1, 4, 4), np.uint8), np.ones((4, 4), bool), [])
        with self.assertRaisesRegex(ValueError, "grids differ"):
            s.tile_context_support(np.ones((1, 4, 4), bool), np.ones((3, 4), bool), [])
        tile = s.TileContribution((0, 0, 4, 4), (0, 0, 4, 4), np.full((4, 4), -1.0))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            s.tile_context_support(np.ones((1, 4, 4), bool), np.ones((4, 4), bool), [tile])
        complex_tile = s.TileContribution((0, 0, 4, 4), (0, 0, 4, 4), np.ones((4, 4), complex))
        with self.assertRaisesRegex(ValueError, "Blend weights"):
            s.tile_context_support(np.ones((1, 4, 4), bool), np.ones((4, 4), bool), [complex_tile])

    def test_chart_area_is_not_unique_surface_area(self):
        result = s.chart_areas({"chart_pixels": 100}, [10.0, 20.0])
        self.assertAlmostEqual(result["chart_areas"]["chart_cm2"], 0.0002)
        self.assertIsNone(result["unique_papyrus_cm2"])
        self.assertIsNone(s.chart_areas({"chart_pixels": 100})["chart_areas"]["chart_cm2"])


class StudioPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = np.ones((31, 256, 256), np.uint8)
        self.stack[:, 128, 128] = 0  # Valid encoded zero, not missing source.
        self.path = self.root / "render.npy"; np.save(self.path, self.stack)
        self.available = np.ones(self.stack.shape, bool)
        self.mesh = np.ones(self.stack.shape[1:], bool)
        self.render = p.studio_module().Render(self.path)

    def prepare(self, order="desc", stride=48):
        return p.prepare_studio_context(self.render, self.available, self.mesh,
                                        mesh_index=15, order=order, stride=stride)

    def test_exact_studio_schedule_and_no_torch_or_forwards(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            result, plan = self.prepare()
        self.assertEqual(plan["planned_windows"], 49)
        self.assertEqual(plan["planes"]["source_layers"], list(range(23, 7, -1)))
        self.assertEqual(result.counts()["complete_context_tiles"], 9)
        self.assertEqual(result.counts()["context_supported_pixels"], 64 * 64)
        self.assertTrue(result.context_support_yx[128, 128])
        self.assertEqual(plan["legacy_intensity_proxy_pixels"], 256 * 256 - 1)
        self.assertEqual(plan["model_forward_calls"], 0)
        self.assertFalse(plan["prediction_completed"])

    def test_only_selected_planes_matter_and_orders_are_explicit(self):
        before, _ = self.prepare()
        self.available[0] = False
        unchanged, _ = self.prepare()
        np.testing.assert_array_equal(before.context_support_yx, unchanged.context_support_yx)
        self.available[23] = False
        desc, _ = self.prepare("desc")
        asc, plan = self.prepare("asc")
        self.assertFalse(desc.context_support_yx.any())
        self.assertTrue(asc.context_support_yx.any())
        self.assertEqual(plan["planes"]["source_layers"], list(range(7, 23)))

    def test_missing_selected_context_voxel_keeps_planned_coverage(self):
        self.available[10, 128, 128] = False
        result, _ = self.prepare()
        self.assertTrue(result.contribution_support_yx[128, 128])
        self.assertFalse(result.context_support_yx[128, 128])

    def test_empty_tile_skip_matches_actual_helper(self):
        self.stack[:, :, :96] = 0
        np.save(self.path, self.stack)
        self.render = p.studio_module().Render(self.path)
        result, plan = self.prepare(stride=96)
        self.assertEqual(plan["planned_windows"], 6)
        self.assertFalse(result.contribution_support_yx[:, :96].any())

    def test_private_cli_and_public_projection(self):
        support_path = self.root / "support.npz"
        np.savez(support_path, source_available_lhw=self.available, mesh_valid_yx=self.mesh)
        out = self.root / "preflight"
        with mock.patch("builtins.print"):
            p.main(["--input", str(self.path), "--support-npz", str(support_path),
                    "--mesh-index", "15", "--order", "desc", "--spacing-um", "8.64", "8.64",
                    "--output-dir", str(out)])
        receipt = json.loads((out / "preflight.json").read_text())
        self.assertFalse(receipt["handoff_ready"])
        self.assertFalse((out / "context.json").exists())
        self.assertEqual(receipt["plan"]["input"]["sha256"], s.sha256_file(self.path))
        receipt["private_candidate_location"] = "PRIVATE_TEST_MARKER"
        summary = p.public_aggregate([receipt], scroll="PHerc0125", surface_set="spiral_fits")
        encoded = json.dumps(summary)
        self.assertNotIn("PRIVATE_TEST_MARKER", encoded)
        self.assertNotIn("source_layers", encoded)
        self.assertNotIn("sha256", encoded)
        self.assertIsNone(summary["unique_papyrus_cm2"])
        with self.assertRaises(FileExistsError):
            p.write_preflight(out, *self.prepare(), support_path=support_path)


class HandoffDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = {}
        for key in ("source_content_sha256", "mesh_sha256", "normal_sha256"):
            path = self.root / (key + ".npy")
            np.save(path, np.ones((3, 4, 4), np.float32))
            self.assets[key] = path
        self.geometry = {k: s.sha256_file(v) for k, v in self.assets.items()}
        self.geometry.update(normal_convention="Synthetic +Z", plane_order="Synthetic increasing source layer",
                             frame={"acquisition": "synthetic-fixture", "axes": "XYZ", "units": "native_voxels",
                                    "level": 0, "voxel_um_xyz": [8.64] * 3, "origin_native_xyz": [0, 0, 0]})
        self.context_record = {"tile_record_kind": "executed", "acquisition_semantics_confirmed": True,
                               "acquisition_definition": "Synthetic fixture: all required voxels explicitly present"}

    def write(self, target=None):
        return s.write_handoff_declaration(
            target or self.root / "handoff", np.ones((4, 4), bool), geometry=self.geometry,
            source_path=self.assets["source_content_sha256"], mesh_path=self.assets["mesh_sha256"],
            normal_path=self.assets["normal_sha256"], producer="synthetic CPU no-op fixture",
            definition="Every contributing full input tile has explicit synthetic acquisition support",
            shift_to_normal_sign=1, context_record=self.context_record)

    def test_existing_consumer_accepts_exact_sidecar(self):
        result = self.write()
        self.assertEqual(result["status"], "complete")
        spec = importlib.util.spec_from_file_location("test_handoff_contract", HERE.parent / "component_export/prepare_handoff.py")
        consumer = importlib.util.module_from_spec(spec); spec.loader.exec_module(consumer)
        declaration = json.loads((self.root / "handoff/context.json").read_text())
        consumer.validate_producer(declaration, self.root / "handoff/acquisition_context.npy")
        self.assertEqual({k: declaration[k] for k in s.GEOMETRY_KEYS}, self.geometry)

    def test_planned_or_unconfirmed_acquisition_remains_blocked(self):
        self.context_record["tile_record_kind"] = "planned"
        with self.assertRaisesRegex(ValueError, "executed"):
            self.write()
        self.context_record["tile_record_kind"] = "executed"
        self.context_record["acquisition_semantics_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "acquisition semantics"):
            self.write()
        self.assertFalse((self.root / "handoff").exists())

    def test_contradictory_preflight_cannot_be_relabelled(self):
        self.context_record.update(prediction_completed=False, weights_loaded=False, model_forward_calls=0)
        with self.assertRaisesRegex(ValueError, "cannot be relabelled"):
            self.write()

    def test_changed_asset_leaves_no_consumable_partial_declaration(self):
        real_sha = s.sha256_file
        source = self.assets["source_content_sha256"]
        reads = 0
        def altered_sha(path):
            nonlocal reads
            if Path(path) == source:
                reads += 1
                if reads > 1:
                    return "e" * 64
            return real_sha(path)
        with mock.patch.object(s, "sha256_file", side_effect=altered_sha):
            with self.assertRaisesRegex(ValueError, "changed before completion"):
                self.write()
        self.assertFalse((self.root / "handoff").exists())
        self.assertEqual(list(self.root.glob(".handoff-*")), [])

    def test_wrong_source_hash_and_unknown_geometry_are_blocked(self):
        self.geometry["source_content_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Actual asset bytes"):
            self.write()
        self.geometry["normal_sha256"] = None
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.write()

    def test_output_outside_git_and_no_overwrite(self):
        git_root = self.root / "worktree"; git_root.mkdir(); (git_root / ".git").write_text("gitdir: fixture")
        with self.assertRaisesRegex(ValueError, "outside Git"):
            self.write(git_root / "private")
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()


if __name__ == "__main__":
    unittest.main()
