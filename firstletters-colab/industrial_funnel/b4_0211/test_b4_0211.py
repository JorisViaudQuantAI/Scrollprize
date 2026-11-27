"""B4 controls: synthetic adversarial tests, no CT, GPU or hidden data."""
from __future__ import annotations
import copy
from fractions import Fraction
from math import comb
import unittest
import numpy as np
from audit_0211 import (p_no_flagged,max_missed_after_zero,minimum_zero_audit,
                        validate_and_summarize,RANDOM_REVIEW_RANKS)
from order_pair_0211 import (region_stats,pair_report)

class FiniteTest(unittest.TestCase):
    def test_hypergeometric_exact(self):
        self.assertEqual(p_no_flagged(10,3,2),Fraction(comb(8,3),comb(10,3)))
        self.assertEqual(p_no_flagged(10,0,10),1)
        self.assertEqual(p_no_flagged(10,3,8),0)
        with self.assertRaises(ValueError):p_no_flagged(10,11,1)
        with self.assertRaises(TypeError):p_no_flagged(10.0,1,1)

    def test_published_bound_and_power(self):
        self.assertEqual(max_missed_after_zero(124,15),21)
        self.assertGreaterEqual(p_no_flagged(124,15,21),Fraction(1,20))
        self.assertLess(p_no_flagged(124,15,22),Fraction(1,20))
        for miss,n in ((1,118),(3,78),(6,48),(12,27),(21,16)):
            self.assertEqual(minimum_zero_audit(124,miss),n)
            self.assertGreaterEqual(p_no_flagged(124,n-1,miss),Fraction(1,20))

    def test_exhaustive_small_cases(self):
        for pop in range(1,13):
            for n in range(pop+1):
                bound=max_missed_after_zero(pop,n)
                self.assertTrue(all(p_no_flagged(pop,n,m)>=Fraction(1,20)
                                    for m in range(bound+1)))
                if bound<pop:self.assertLess(p_no_flagged(pop,n,bound+1),Fraction(1,20))

    def make_fixture(self):
        w=[dict(window=f"mock{k:03d}_x0",winding=f"mock{k:03d}",k=0,x0=0,
                x1=150,valid_share=.75,valid_cm2=293.369/149,det="st96desc",
                line=149-k,lr=149-k,row=1.0,hot=.01,glyph=0.,rank=k)
           for k in range(1,150)]
        for z in w[95:]:z["line"]=None;z["lr"]=-99
        ranks=list(range(1,26))+list(RANDOM_REVIEW_RANKS)
        changes={"A19":2,"A16":6,"A02":7,"A17":10,"A36":23,"A06":43}
        for label,desired in changes.items():
            idx=int(label[1:])-1;old=ranks[idx]
            other=ranks.index(desired)
            ranks[other]=old;ranks[idx]=desired
        for rank,name in {2:"fw033_x1",6:"fw030_x1",7:"fw035_x1",
                          10:"fw057_x3",23:"fw055_x1",43:"fw047_x0"}.items():
            w[rank-1]["window"]=name
        key={f"A{i:02d}":w[rank-1]["window"] for i,rank in enumerate(ranks,1)}
        return w,key

    def test_sealed_review_and_abstention(self):
        w,key=self.make_fixture()
        d=validate_and_summarize(w,key)
        self.assertEqual(d["no_line_score"]["windows"],54)
        self.assertEqual(d["no_line_score"]["randomly_reviewed"],6)
        self.assertEqual(d["audit"]["max_additional_visual_flags_not_excluded_95pct"],21)
        self.assertFalse(d["production_skip_allowed"])

    def test_corrupt_selection_and_lines_fail(self):
        w,key=self.make_fixture()
        bad=copy.deepcopy(w);bad[0]["rank"]=2
        with self.assertRaises(ValueError):validate_and_summarize(bad,key)
        badkey=dict(key);badkey["A10"]=badkey["A11"]
        with self.assertRaises(ValueError):validate_and_summarize(w,badkey)
        bad=copy.deepcopy(w);bad[110]["line"]=None;bad[110]["lr"]=0
        with self.assertRaises(ValueError):validate_and_summarize(bad,key)

class OrderTest(unittest.TestCase):
    def synthetic(self):
        desc=np.zeros((20,60),dtype=np.uint8)
        asc=np.zeros_like(desc)
        mask=np.ones_like(desc,dtype=bool)
        desc[:,30:]=240;asc[:,:30]=240
        return desc,asc,mask

    def test_global_cancellation_is_not_local_agreement(self):
        d,a,m=self.synthetic()
        q=pair_report(d,a,m)
        self.assertTrue(q["regional_order_disagreement"])
        self.assertEqual(q["regions"]["whole"]["dominance_desc_minus_asc"],0)
        self.assertEqual(q["regions"]["whole"]["activation_jaccard"],0)
        self.assertFalse(q["production_skip_allowed"])

    def test_one_order_or_unobserved_half_does_not_trigger(self):
        d,a,m=self.synthetic();a[:]=0
        self.assertFalse(pair_report(d,a,m)["regional_order_disagreement"])
        d,a,m=self.synthetic();m[:,:30]=False
        self.assertFalse(pair_report(d,a,m)["regional_order_disagreement"])
        self.assertIsNone(pair_report(d,a,m)["regions"]["left"]["activation_jaccard"])

    def test_shape_and_type_fail_closed(self):
        d,a,m=self.synthetic()
        with self.assertRaises(ValueError):pair_report(d[:,:10],a,m)
        with self.assertRaises(ValueError):region_stats(d.astype(float),a,m)
        with self.assertRaises(ValueError):region_stats(d,a,m.astype(np.uint8))

if __name__=="__main__":unittest.main()
