"""Small synthetic regressions for the historical campaign importer.

No private bundle, model, network, CT data or GPU is needed. The 4x5 arrays
exercise identity/accounting contracts; they are not text-positive controls.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import adapt_visual_campaign as adapter


class CampaignAdapterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.data = self.root / "data"
        self.data.mkdir()
        self.prepared = self.root / "render-audit/0125-w025-v2"
        self.prepared.mkdir(parents=True)
        self.selection = {"render_node_bounds_rc": [7, 48, 130, 184]}
        self.preparation = {
            "renderer_sha256": "a" * 64,
            "arms": {"raw": {"stack31_sha256": "b" * 64},
                     "snapped": {"stack31_sha256": "c" * 64}},
        }
        self.write_preparation()

    def write_preparation(self):
        (self.prepared / "selection_before_ct.json").write_text(json.dumps(self.selection))
        (self.prepared / "preparation_receipt.json").write_text(json.dumps(self.preparation))

    def save_array(self, filename, array):
        path = self.data / filename
        np.save(path, array, allow_pickle=False)
        return path

    def fixture(self, name="0800_C_left_context96_k08_to_k23_s96", *,
                reverse=False, implicit=False):
        source = np.arange(1, 321, dtype=np.uint16).reshape(16, 4, 5).astype(np.uint8)
        if reverse:
            source = source[::-1].copy()
        ct = self.save_array(name + ".input.npy", source)
        depth_mask = self.save_array(name + ".source.npy", np.ones(source.shape, bool))
        prob = self.save_array(name + ".prob.npy", np.full((4, 5), 0.25, np.float32))
        valid = self.save_array(name + ".valid.npy", np.ones((4, 5), bool))
        context = self.save_array(name + ".context.npy", np.zeros((4, 5), bool))
        depths = list(range(8, 24))
        if reverse:
            depths.reverse()
        if implicit:
            depths = "unchanged synthetic prepared order"
        job = {"name": name, "input": ct.relative_to(self.root).as_posix(),
               "stride": 96, "depth_order": depths}
        if not implicit:
            job["source_support"] = depth_mask.relative_to(self.root).as_posix()
        if name.startswith("0125_"):
            job["full_canvas_roi_y0_y1_x0_x1"] = [80, 84, 432, 437]
        if "left_context96" in name:
            job["input_global_y0_y1_x0_x1"] = [650, 654, 804, 809]
        receipt = {
            "job": job, "input_sha256": adapter.sha(ct), "input_shape": list(source.shape),
            "source_support_sha256": None if implicit else adapter.sha(depth_mask),
            "outputs": {"prob": {"file": prob.name, "sha256": adapter.sha(prob)},
                        "valid": {"file": valid.name, "sha256": adapter.sha(valid)}},
            "extra_masks": {
                "source_valid": {"file": valid.relative_to(self.root).as_posix(), "sha256": adapter.sha(valid)},
                "context_valid": {"file": context.relative_to(self.root).as_posix(), "sha256": adapter.sha(context)},
            },
            "depth_order": depths, "stride": 96, "blend": "none",
            "weights_sha256": "1" * 64, "pipeline_sha256": "2" * 64,
            "runner_sha256": "3" * 64, "model_source": {"model_py_sha256": "4" * 64},
            "normalization": "synthetic uint8 / 255", "statistics": {"windows": 1, "seconds": 0.25},
            "new_model_forwards_including_sentinel": 2, "sentinel_bitwise_equal": True,
        }
        path = self.data / (name + ".receipt.json")
        path.write_text(json.dumps(receipt))
        record = {"name": name, "receipt": path.relative_to(self.root).as_posix(),
                  "receipt_sha256": adapter.sha(path), "shape": list(source.shape),
                  "stride": 96, "blend": "none", "windows": 1, "sentinel": 1, "forwards": 2}
        return job, adapter.manifest_for(record, self.root), record

    def test_0800_conflicting_full_extents_rejected(self):
        job, manifest, _ = self.fixture()
        adapter.verify_declaration(job, manifest, self.root)
        for index in range(4):
            with self.subTest(bound=index):
                changed = copy.deepcopy(job)
                changed["input_global_y0_y1_x0_x1"][index] += 1
                with self.assertRaisesRegex(ValueError, "crop extent"):
                    adapter.verify_declaration(changed, manifest, self.root)

    def test_0125_end_extent_is_checked_not_only_origin(self):
        job, manifest, _ = self.fixture("0125_raw_reverse_roi_s48")
        adapter.verify_declaration(job, manifest, self.root)
        for index in (1, 3):
            with self.subTest(bound=index):
                changed = copy.deepcopy(job)
                changed["full_canvas_roi_y0_y1_x0_x1"][index] += 1
                with self.assertRaisesRegex(ValueError, "crop extent"):
                    adapter.verify_declaration(changed, manifest, self.root)

    def test_depth_conflict_rejected_despite_identical_file_bytes(self):
        job, manifest, _ = self.fixture()
        changed = copy.deepcopy(job)
        changed["depth_order"] = changed["depth_order"][::-1]
        with self.assertRaisesRegex(ValueError, "depth declaration"):
            adapter.verify_declaration(changed, manifest, self.root)

    def test_repeated_input_or_support_change_rejected(self):
        job, manifest, _ = self.fixture()
        for key, dtype in (("input", np.uint8), ("source_support", bool)):
            with self.subTest(role=key):
                changed = copy.deepcopy(job)
                path = self.save_array("different_" + key + ".npy", np.zeros((16, 4, 5), dtype))
                changed[key] = path.relative_to(self.root).as_posix()
                with self.assertRaisesRegex(ValueError, "Conflicting repeated"):
                    adapter.verify_declaration(changed, manifest, self.root)

    def test_legacy_control_preserves_implicit_prepared_support(self):
        job, manifest, _ = self.fixture("author841_positive_s96", implicit=True)
        self.assertNotIn("source_support", job)
        self.assertEqual(manifest["support_basis"], "IMPLICIT_PREPARED_PRESENCE")
        self.assertEqual(manifest["arrays"]["source_depth"]["derivation"], "all_true_as_executed")
        self.assertIsNone(manifest["depth_samples"])
        self.assertEqual(manifest["calibration"]["semantic_control_status"], "NOT_MATCHED")
        adapter.verify_declaration(job, manifest, self.root)

    def test_legacy_proof_location_and_hash_do_not_change_computation_key(self):
        _, manifest, _ = self.fixture("author841_positive_s96", implicit=True)
        changed = copy.deepcopy(manifest)
        changed["arrays"]["source_depth"]["evidence_sha256"] = "f" * 64
        changed["arrays"]["source_depth"]["evidence_path"] = "relocated-proof.json"
        self.assertEqual(adapter.execution_key(manifest), adapter.execution_key(changed))

    def test_new_origin_or_recipe_requires_a_new_computation_key(self):
        _, manifest, _ = self.fixture()
        origin = copy.deepcopy(manifest)
        origin["frame"]["origin_yx"][1] += 1
        origin["recipe"]["origin_yx"] = list(origin["frame"]["origin_yx"])
        origin["recipe"]["phase_yx"] = [v % 96 for v in origin["frame"]["origin_yx"]]
        recipe = copy.deepcopy(manifest)
        recipe["recipe"].update(stride=48, blend="hann")
        recipe["recipe"]["phase_yx"] = [v % 48 for v in recipe["frame"]["origin_yx"]]
        for changed in (origin, recipe):
            self.assertNotEqual(adapter.execution_key(manifest), adapter.execution_key(changed))

    def test_two_orders_share_chart_pixels_but_not_execution_identity(self):
        _, forward, _ = self.fixture("0800_C_left512_k08_to_k23_s96")
        _, reverse, _ = self.fixture("0800_C_left512_k23_to_k08_s96", reverse=True)
        self.assertNotEqual(adapter.execution_key(forward), adapter.execution_key(reverse))
        rows = adapter.chart_accounting([forward, reverse], self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["maps"], 2)
        self.assertEqual(rows[0]["distinct_input_recipes"], 2)
        self.assertEqual(rows[0]["source_valid"], {
            "gross_map_pixels": 40, "union_chart_pixels": 20, "repeated_chart_pixels": 20})
        self.assertIsNone(rows[0]["physical_area_cm2"])
        self.assertEqual(rows[0]["physical_sheet_identity"], "NOT_ESTABLISHED")

    def test_raw_and_snapped_are_separate_even_if_values_happen_to_match(self):
        _, raw, _ = self.fixture("0125_raw_reverse_roi_s48")
        _, snapped, _ = self.fixture("0125_snapped_reverse_roi_s48")
        self.assertEqual(raw["arrays"]["input"]["sha256"], snapped["arrays"]["input"]["sha256"])
        rows = adapter.chart_accounting([raw, snapped], self.root)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["source_valid"]["union_chart_pixels"] == 20 for r in rows))
        self.assertTrue(all(r["source_valid"]["repeated_chart_pixels"] == 0 for r in rows))

    def test_different_prepared_selection_cannot_share_chart_accounting(self):
        job, original, _ = self.fixture("0125_raw_reverse_roi_s48")
        self.selection["render_node_bounds_rc"] = [8, 49, 130, 184]
        self.write_preparation()
        changed = copy.deepcopy(original)
        changed["frame"] = adapter.frame_for(job["name"], job, self.root)
        self.assertNotEqual(original["frame"]["chart_id"], changed["frame"]["chart_id"])
        self.assertEqual(len(adapter.chart_accounting([original, changed], self.root)), 2)

    def test_stack_or_renderer_change_creates_new_prepared_chart(self):
        job, original, _ = self.fixture("0125_raw_reverse_roi_s48")
        baseline = copy.deepcopy(self.preparation)
        for change in ("stack", "renderer"):
            with self.subTest(change=change):
                self.preparation = copy.deepcopy(baseline)
                if change == "stack":
                    self.preparation["arms"]["raw"]["stack31_sha256"] = "d" * 64
                else:
                    self.preparation["renderer_sha256"] = "e" * 64
                self.write_preparation()
                new_frame = adapter.frame_for(job["name"], job, self.root)
                self.assertNotEqual(original["frame"]["chart_id"], new_frame["chart_id"])

    def test_translated_chart_union_counts_overlap_once(self):
        _, first, _ = self.fixture("0800_C_left512_k08_to_k23_s96")
        second = copy.deepcopy(first)
        second["frame"]["origin_yx"][1] += 3
        second["recipe"]["origin_yx"] = list(second["frame"]["origin_yx"])
        second["recipe"]["phase_yx"] = [v % 96 for v in second["frame"]["origin_yx"]]
        row, = adapter.chart_accounting([first, second], self.root)
        # Two 4x5 masks, shifted three columns: an independently countable 4x8 union.
        self.assertEqual(row["source_valid"], {
            "gross_map_pixels": 40, "union_chart_pixels": 32, "repeated_chart_pixels": 8})

    def test_duplicate_completed_names_fail_before_artifact_import(self):
        folder = self.root / "deliverables"
        folder.mkdir()
        (folder / "campaign_summary.json").write_text(json.dumps({
            "records": [{"name": "same"}, {"name": "same"}]}))
        with patch.object(adapter, "manifest_for", side_effect=AssertionError("must not load files")) as mocked:
            with self.assertRaisesRegex(ValueError, "Duplicate completed-map"):
                adapter.audit(self.root)
            mocked.assert_not_called()

    def test_inconsistent_completion_counts_fail_before_artifact_import(self):
        folder = self.root / "deliverables"
        folder.mkdir()
        (folder / "campaign_summary.json").write_text(json.dumps({
            "records": [{"name": "one"}], "unique_jobs": 1,
            "completed_maps": 2, "pending_jobs": []}))
        with patch.object(adapter, "manifest_for", side_effect=AssertionError("must not load files")) as mocked:
            with self.assertRaisesRegex(ValueError, "completion counts"):
                adapter.audit(self.root)
            mocked.assert_not_called()

    def test_campaign_forward_count_cannot_override_original_receipt(self):
        _, _, record = self.fixture()
        changed = copy.deepcopy(record)
        changed["forwards"] += 1
        with self.assertRaisesRegex(ValueError, "counts/recipe"):
            adapter.manifest_for(changed, self.root)


if __name__ == "__main__":
    unittest.main()
