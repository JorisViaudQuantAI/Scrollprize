"""Tests of source support and stitching, not evidence that an ink model works."""
import unittest
import numpy as np
from common import PROTOCOL, compare, field_plan, gaussian, positions
from run import add_tile


class Geometry(unittest.TestCase):
    def test_all_covering_tiles_match_full_source_grid(self):
        roi=PROTOCOL["field"]["native_render_box_yxyx"]
        for phase in ((0,0),(10,10)):
            full=[(y,x) for y in range(115+phase[0],4166-63,21)
                  for x in range(694+phase[1],7638-63,21)
                  if y+64>roi[0] and y<roi[2] and x+64>roi[1] and x<roi[3]]
            np.testing.assert_array_equal(positions(roi,phase),full)

    def test_every_physical_pixel_has_all_source_support(self):
        plan=field_plan()
        self.assertEqual([p["tiles_before_CT_admission"] for p in plan["phases"]],[3886,3944])
        self.assertEqual([p["contributors_min_before_CT_admission"] for p in plan["phases"]],[9,9])
        self.assertEqual(plan["required_native_source_box_yxyx"],[1490,3991,2961,5262])
        # Source patch is a real acquisition slice, wider than the scored field.
        source=plan["required_native_source_box_yxyx"]
        for phase in PROTOCOL["phases_yx"]:
            pts=positions(plan["native_roi_yxyx"],phase)
            self.assertTrue(np.all(pts>=source[:2]))
            self.assertTrue(np.all(pts+64<=source[2:]))

    def test_two_micro_inputs_share_exact_physical_samples(self):
        yy,xx=np.indices((169,169))
        location=yy*169+xx
        first=location[0:64,0:64]
        second=location[10:74,10:74]
        np.testing.assert_array_equal(first[10:64,10:64],second[:54,:54])
        # A naive same-index comparison would compare different material.
        self.assertFalse(np.array_equal(first[:54,:54],second[:54,:54]))

    def test_weighted_stitch_preserves_physical_signal(self):
        # Feed a source-coordinate ramp as deterministic numerical tile data.
        # This tests clipping/accumulation only; it is not substitute inference.
        roi=[1549,4050,1624,4140]
        shape=(roi[2]-roi[0],roi[3]-roi[1])
        yy,xx=np.indices(shape)
        expected=((yy+roi[0])*.0001+(xx+roi[1])*.00001).astype(np.float32)
        outputs=[]
        for phase in PROTOCOL["phases_yx"]:
            num=np.zeros(shape,np.float32);den=np.zeros(shape,np.float32);count=np.zeros(shape,np.uint16)
            for y,x in positions(roi,phase):
                ty,tx=np.indices((64,64))
                q=((ty+y)*.0001+(tx+x)*.00001).astype(np.float32)
                add_tile(num,den,count,q,gaussian(),(y,x),roi)
            self.assertGreaterEqual(int(count.min()),9)
            self.assertTrue((den>0).all())
            outputs.append(num/den)
            np.testing.assert_allclose(outputs[-1],expected,atol=1e-7,rtol=0)
        np.testing.assert_allclose(*outputs,atol=1e-7,rtol=0)

    def test_unknown_pixels_are_excluded_not_cast_as_background(self):
        a=np.array([[.1,.9],[np.nan,.3]],np.float32)
        b=np.array([[.2,.6],[np.nan,.3]],np.float32)
        valid=np.array([[True,True],[False,True]])
        result=compare(a,b,valid)
        self.assertEqual(result["valid_pixels"],3)
        self.assertEqual(result["binary_disagreement"]["0.75"]["pixels"],1)
        self.assertAlmostEqual(result["mean_absolute_difference"],.4/3,places=6)


if __name__=="__main__":unittest.main()
