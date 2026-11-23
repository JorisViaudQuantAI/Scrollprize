import unittest
from coalesced_control import spans_for_box, coalesce, reconstruct

class CoalescedTest(unittest.TestCase):
    def test_frozen_plan(self):
        s=spans_for_box([20,20,20,36,84,84],[65,128,128])
        self.assertEqual(len(s),1024)
        self.assertEqual((len(coalesce(s,64)),sum(b-a+1 for a,b in coalesce(s,64))),(16,130048))
        self.assertEqual((len(coalesce(s,8256)),sum(b-a+1 for a,b in coalesce(s,8256))),(1,253888))
    def test_real_bytes_after_overfetch(self):
        shape=[3,4,8]; box=[0,1,2,3,3,6]
        source=bytes(range(96))
        original=spans_for_box(box,shape)
        expected=b''.join(source[a:b+1] for a,b in original)
        for gap in (0,3,4,63):
            groups=coalesce(original,gap)
            payloads=[source[a:b+1] for a,b in groups]
            self.assertEqual(reconstruct(groups,payloads,original),expected)
    def test_negative_and_overlapping(self):
        for gap in (-1,1.0,True):
            with self.assertRaises(ValueError): coalesce([(0,1)],gap)
        with self.assertRaises(ValueError): coalesce([(0,3),(3,4)],0)
        with self.assertRaises(ValueError): spans_for_box([0,0,0,4,3,1],[3,4,8])
        with self.assertRaises(ValueError): reconstruct([(0,3)],[b'xx'],[(0,1)])
        with self.assertRaises(ValueError): reconstruct([(0,3)],[b'xxxx'],[(0,5)])
if __name__=='__main__':unittest.main()
