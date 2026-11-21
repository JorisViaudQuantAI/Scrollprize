import unittest

from audit import audit_point
from replay_published_pair import adapt_record, grouped_geometry, nearest


def record(peaks, displacement=0):
    return dict(rays=dict(raw=dict(peak_offsets_vox=peaks, material_center_fraction=1)),
                along_raw_normal_vox=displacement)


class PublishedPairAdapterTests(unittest.TestCase):
    def test_nearest_uniqueness_is_not_quarter_voxel_change_tolerance(self):
        self.assertEqual(nearest([1, 1 + 5e-10])[1], [0, 1])
        self.assertEqual(nearest([1, 1 + 2e-9])[1], [0])
        row, group = adapt_record(record([-1, 1.1]))
        self.assertTrue(row["unique_raw"])
        self.assertEqual(group, "unchanged")

    def test_switch_uses_fixed_raw_peak_list_and_keeps_distance_to_old_crest(self):
        row, group = adapt_record(record([-2, 4], 3))
        self.assertEqual(group, "switched")
        self.assertEqual(row["snapped_nearest"], 1)
        self.assertEqual(row["after_same_peak"], 5)
        self.assertIn("detected_nearest_crest_changed", audit_point(row))

    def test_initial_tie_never_gets_invented_landmark_or_changed_flag(self):
        row, group = adapt_record(record([-2, 2], 1))
        self.assertEqual(group, "ambiguous")
        self.assertNotIn("after_same_peak", row)
        self.assertNotIn("nearest_peak_changed", row)
        self.assertTrue(audit_point(row))

    def test_contradictory_published_identity_is_rejected(self):
        value = record([-2, 4], 3)
        value["legacy_nearest_peak_changed"] = False
        with self.assertRaises(ValueError):
            adapt_record(value)

    def test_missing_schema_stays_unknown_and_aggregate_omits_coordinates(self):
        self.assertEqual(adapt_record({}), ({}, "unknown"))
        result = grouped_geometry([dict(id="site-secret", public_uv_yx=[1, 2])], ["ambiguous"], 9.362)
        self.assertEqual(result["ambiguous"]["fixed_reference_transverse_miss_same_uv_um"]["unknown"], 1)
        self.assertNotIn("site-secret", str(result))
        self.assertNotIn("public_uv", str(result))


if __name__ == "__main__":
    unittest.main()
