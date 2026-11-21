import copy
import unittest

from audit import audit_point, audit_points, audit_interval, validate_pair


def point(**changes):
    row = dict(material_fraction=0.95, n_peaks=3, raw_nearest=4.0,
               snapped_nearest=1.0, after_same_peak=1.0, unique_raw=True,
               unique_snapped=True, nearest_peak_changed=False)
    row.update(changes)
    return row


def manifest():
    frame = dict(acquisition="synthetic-control", volume_metadata_sha256="1" * 64,
                 axes="XYZ", units="native_voxels", level=0,
                 voxel_um_xyz=[9.362] * 3, origin_native_xyz=[0.0] * 3)
    arm = dict(frame=frame, chart=dict(reference_sha256="2" * 64, shape_yx=[128, 128],
                                     pitch_um_yx=[9.362, 9.362], pixel_center_convention="index+0.5"),
               mesh_sha256="3" * 64, model_sha256="4" * 64, normal_field_sha256="5" * 64,
               input_sha256="6" * 64, validity_mask_sha256="7" * 64,
               depth_offsets_um=[i * 9.362 for i in range(8, -8, -1)],
               normal_convention="cross(dy,dx) in native XYZ", preprocess=dict(clip_max=200, divide=255),
               inference=dict(precision="fp32", stride=48, blend="hann", threshold=None))
    return dict(raw=copy.deepcopy(arm), adjusted=copy.deepcopy(arm),
                common_valid_mask_sha256="8" * 64, normal_policy="fixed_original_field",
                comparison_basis="common_chart", claims_same_physical_positions=False)


class CrestAuditTests(unittest.TestCase):
    def test_closer_to_any_but_different_crest_is_held(self):
        reasons = audit_point(point(snapped_nearest=2, after_same_peak=8, nearest_peak_changed=True))
        self.assertIn("detected_nearest_crest_changed", reasons)
        self.assertIn("same_crest_distance_increased", reasons)

    def test_unchanged_crest_nonregression_passes_only_audit(self):
        self.assertEqual(audit_point(point()), [])
        result = audit_points([point()])
        self.assertEqual(result["audit_pass"], 1)
        self.assertIn("does not identify", result["meaning"])

    def test_existing_quarter_voxel_tie_boundary_is_explicit(self):
        self.assertEqual(audit_point(point(snapped_nearest=4.25, after_same_peak=4.25)), [])
        self.assertIn("same_crest_distance_increased", audit_point(point(snapped_nearest=4.2501, after_same_peak=4.2501)))

    def test_missing_ambiguous_or_low_support_is_never_a_pass(self):
        self.assertTrue(audit_point({}))
        self.assertIn("ambiguous_nearest_crest", audit_point(point(unique_raw=False)))
        self.assertIn("insufficient_material_support", audit_point(point(material_fraction=0.79)))
        self.assertIn("no_detected_crest", audit_point(point(n_peaks=0)))

    def test_invalid_or_inconsistent_numbers_are_held(self):
        self.assertTrue(audit_point(point(after_same_peak=float("nan"))))
        self.assertTrue(audit_point(point(raw_nearest=-1)))
        self.assertTrue(audit_point(point(unique_raw="true")))
        self.assertTrue(audit_point(point(n_peaks=True)))
        self.assertIn("inconsistent_stored_distances", audit_point(point(snapped_nearest=2)))

    def test_aggregate_and_empty_list_do_not_invent_acceptance(self):
        result = audit_points(dict(metrics=dict(changed_nearest_peak=9)))
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["audit_pass"])
        self.assertEqual(audit_points([])["status"], "insufficient_evidence")

    def test_input_coordinates_and_identifiers_are_not_emitted(self):
        row = point(id="private-id", xyz=[1, 2, 3])
        result = str(audit_points([row]))
        self.assertNotIn("private-id", result)
        self.assertNotIn("xyz", result)


class PairManifestTests(unittest.TestCase):
    def test_consistent_pair_is_only_declared_metadata(self):
        result = validate_pair(manifest())
        self.assertEqual(result["status"], "declared_metadata_consistent")
        self.assertFalse(result["verified_file_contents"])
        self.assertFalse(result["anatomical_alignment_certified"])

    def test_same_uv_never_certifies_same_physical_positions(self):
        doc = manifest(); doc["claims_same_physical_positions"] = True
        self.assertIn("common_chart_cannot_claim_same_physical_positions", validate_pair(doc)["errors"])

    def test_native_frame_and_physical_spacing_must_match(self):
        for field, bad in (("acquisition", "another-scan"), ("axes", "ZYX"),
                           ("level", 2), ("units", "micrometres"), ("voxel_um_xyz", [9.6]*3)):
            doc = manifest(); doc["adjusted"]["frame"][field] = bad
            with self.subTest(field=field):
                self.assertEqual(validate_pair(doc)["status"], "invalid_manifest")

    def test_model_sampling_order_and_threshold_changes_are_separate_experiments(self):
        for field, bad in (("model_sha256", "a" * 64),
                           ("depth_offsets_um", list(reversed(manifest()["raw"]["depth_offsets_um"]))),
                           ("inference", dict(precision="fp16", stride=64, threshold=0.5)),
                           ("normal_convention", "cross(dx,dy) in native XYZ")):
            doc = manifest(); doc["adjusted"][field] = bad
            with self.subTest(field=field):
                self.assertIn("mismatched_" + field, validate_pair(doc)["errors"])

    def test_normal_change_is_rejected_or_explicitly_broadens_scope(self):
        doc = manifest(); doc["adjusted"]["normal_field_sha256"] = "b" * 64
        self.assertIn("fixed_original_normal_fields_differ", validate_pair(doc)["errors"])
        doc["normal_policy"] = "recomputed_per_mesh"
        result = validate_pair(doc)
        self.assertEqual(result["status"], "declared_metadata_consistent")
        self.assertIn("geometry_and_normal_change", result["comparison_scope"])

    def test_physical_anchor_comparison_requires_both_mappings(self):
        doc = manifest(); doc["comparison_basis"] = "fixed_native_anchors"
        self.assertEqual(validate_pair(doc)["status"], "invalid_manifest")
        doc["native_anchors"] = dict(sha256="c" * 64, frame=copy.deepcopy(doc["raw"]["frame"]))
        doc["raw"]["anchor_to_chart_map_sha256"] = "d" * 64
        doc["adjusted"]["anchor_to_chart_map_sha256"] = "e" * 64
        result = validate_pair(doc)
        self.assertEqual(result["status"], "declared_metadata_consistent")
        self.assertFalse(result["anatomical_alignment_certified"])

    def test_incomplete_arrays_booleans_and_missing_mask_are_rejected(self):
        changes = (("frame", "voxel_um_xyz", [True, 9, 9]),
                   ("chart", "shape_yx", [128]), ("chart", "pitch_um_yx", [0, 9]))
        for group, key, value in changes:
            doc = manifest(); doc["raw"][group][key] = value
            self.assertEqual(validate_pair(doc)["status"], "invalid_manifest")
        doc = manifest(); del doc["common_valid_mask_sha256"]
        self.assertEqual(validate_pair(doc)["status"], "invalid_manifest")


class IntervalTests(unittest.TestCase):
    def example(self):
        return dict(peaks_vox=[-8.0, 2.0, 12.0], location_error_bounds_vox=[0.5]*3,
                    proposed_displacement_vox=6.0, error_bound_basis="synthetic known bound only")

    def test_conditional_midpoint_bounds_and_strict_boundary(self):
        doc = self.example(); result = audit_interval(doc)
        self.assertEqual(result["admissible_interval_vox"], [-2.5, 6.5])
        self.assertEqual(result["status"], "conditional_interval_pass")
        self.assertFalse(result["anatomical_sheet_identity_certified"])
        doc["proposed_displacement_vox"] = 6.5
        self.assertEqual(audit_interval(doc)["status"], "hold_for_review")

    def test_no_uncertainty_bound_is_fabricated(self):
        doc = self.example(); del doc["location_error_bounds_vox"]
        self.assertEqual(audit_interval(doc)["status"], "insufficient_evidence")

    def test_uncertain_initial_association_and_missing_neighbor_abstain(self):
        doc = self.example(); doc["peaks_vox"] = [-2.5, 2, 12]
        self.assertEqual(audit_interval(doc)["status"], "hold_for_review")
        doc = self.example(); doc["peaks_vox"] = [2, 12, 22]
        self.assertEqual(audit_interval(doc)["status"], "hold_for_review")

    def test_overlapping_crest_intervals_and_unsorted_inputs_abstain(self):
        doc = self.example(); doc["location_error_bounds_vox"] = [6, 6, 6]
        self.assertEqual(audit_interval(doc)["status"], "hold_for_review")
        doc = self.example(); doc["peaks_vox"] = [2, -8, 12]
        self.assertEqual(audit_interval(doc)["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
