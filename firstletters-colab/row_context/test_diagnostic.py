import copy
import unittest

import cv2
import numpy as np

from diagnostic import local_targets, reference_case, row_metrics, validate_manifest


def band(x0, y0, x1, y1):
    return {"band_polygon_xy": [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]}


class RowGeometryTests(unittest.TestCase):
    def test_native_subfield_translates_both_source_centres(self):
        reference = reference_case()
        targets = local_targets(reference, [1268,3356,2618,4514])
        np.testing.assert_allclose([(t["box_xyxy"][1]+t["box_xyxy"][3])/2 for t in targets], [406,943], rtol=0, atol=1e-12)
        self.assertEqual(targets[0]["box_xyxy"][0], -1612)
        self.assertEqual(targets[0]["box_xyxy"][2], 2770)

    def test_polygon_rounding_occurs_before_odd_crop_translation(self):
        valid = np.ones((11,13), bool)
        valid[4,4] = False
        target = {"id":"row", "box_xyxy":[1,1,10,10]}
        proposal = band(2.5,2.5,6.5,6.5)
        result = row_metrics([proposal,proposal], [target], valid)[0]
        union = np.zeros(valid.shape, np.uint8)
        cv2.fillPoly(union, [np.rint(proposal["band_polygon_xy"]).astype(np.int32)], 1)
        covered = (union > 0) & valid
        expected = covered[1:10,1:10].sum()/valid[1:10,1:10].sum()
        self.assertEqual(result["top_k"]["1"]["box_coverage"], expected)
        self.assertEqual(result["top_k"]["3"]["box_coverage"], expected)
        cy=round((1+10)/2)
        self.assertEqual(result["top_k"]["1"]["centerline_coverage"],covered[cy,1:10].sum()/valid[cy,1:10].sum())

    def test_outside_centreline_is_unknown_not_clipped_to_image_edge(self):
        valid=np.ones((8,10),bool)
        target={"id":"row", "box_xyxy":[0,-8,10,3]}
        result=row_metrics([band(0,0,9,2)], [target], valid)[0]
        self.assertGreater(result["source_box_support_fraction"],0)
        self.assertEqual(result["source_centreline_support_fraction"],0)
        self.assertIsNone(result["top_k"]["1"]["centerline_coverage"])
        self.assertIsNone(result["top_k"]["1"]["omitted_observed_centreline"])

    def test_missing_target_remains_unassessable(self):
        valid=np.zeros((12,14),bool)
        target={"id":"row", "box_xyxy":[2,2,12,10]}
        result=row_metrics([band(0,0,13,11)], [target], valid)[0]
        self.assertEqual(result["status"], "source_target_unobserved")
        self.assertIsNone(result["top_k"]["10"]["box_coverage"])
        self.assertIsNone(result["first_centreline_overlap_rank"])

    def test_full_source_support_denominator_survives_field_cropping(self):
        valid=np.ones((10,10),bool)
        target={"id":"row", "box_xyxy":[-5,1,15,9]}
        result=row_metrics([band(0,0,9,9)], [target], valid)[0]
        self.assertEqual(result["source_box_support_fraction"], .5)
        self.assertEqual(result["source_centreline_support_fraction"], .5)
        self.assertEqual(result["top_k"]["1"]["centerline_coverage"], 1.0)

    def test_lower_ranked_row_is_reported_even_when_top_five_omit_it(self):
        valid=np.ones((30,40),bool)
        target={"id":"row2", "box_xyxy":[5,18,35,24]}
        proposals=[band(5,3,34,7) for _ in range(5)]+[band(5,18,34,23)]
        result=row_metrics(proposals, [target], valid)[0]
        self.assertTrue(result["top_k"]["5"]["omitted_observed_centreline"])
        self.assertFalse(result["top_k"]["10"]["omitted_observed_centreline"])
        self.assertEqual(result["first_centreline_overlap_rank"],6)
        self.assertEqual(result["first_half_centreline_rank"],6)
        self.assertEqual(result["greatest_centreline_overlap_rank"],6)


class PairContractTests(unittest.TestCase):
    def setUp(self):
        self.reference=reference_case()
        signature=dict(model={"checkpoint_sha256":"1"*64},source_frame="published306b native raster",
                       depth_order=list(range(22,7,-1)),preprocess={"kind":"declared"},fusion={"kind":"gaussian"},precision="fp32",tta=False,
                       tile_shape_yx_px=[64,64],stride_yx_px=[21,21],coverage_policy="common complete contexts")
        self.manifest=dict(reference_case=self.reference["id"],experiment_kind="tile_grid_pair",
                           field_box_yxyx=[1268,3356,2618,4514],maps=[
                               dict(map="a.npy",mask="am.npy",map_sha256="a"*64,mask_sha256="b"*64,run_signature=signature,tile_origin_yx_px=[0,0]),
                               dict(map="b.npy",mask="bm.npy",map_sha256="c"*64,mask_sha256="d"*64,run_signature=copy.deepcopy(signature),tile_origin_yx_px=[10,10])])

    def test_only_declared_origin_changes_are_accepted(self):
        validate_manifest(self.manifest,self.reference)
        changed=copy.deepcopy(self.manifest)
        changed["maps"][1]["run_signature"]["fusion"]={"kind":"hann"}
        with self.assertRaisesRegex(ValueError,"match all shared"):
            validate_manifest(changed,self.reference)

    def test_same_phase_modulo_stride_is_not_a_second_placement(self):
        changed=copy.deepcopy(self.manifest)
        changed["maps"][1]["tile_origin_yx_px"]=[21,42]
        with self.assertRaisesRegex(ValueError,"identical residues"):
            validate_manifest(changed,self.reference)

    def test_one_map_cannot_claim_a_phase_comparison(self):
        changed=copy.deepcopy(self.manifest);changed["maps"]=changed["maps"][:1]
        with self.assertRaisesRegex(ValueError,"never a phase comparison"):
            validate_manifest(changed,self.reference)


if __name__ == "__main__":
    unittest.main()
