"""Counterexamples to old maximum-ratio and global-direction summaries."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'pinned/gemini_lane'))
import cloud_rv3_screen as old
from postscreen import regional_review
class T(unittest.TestCase):
    def test_max_ratio_masks_qualified_tile(self):
        a=np.zeros((512,1024),np.float32);b=a.copy();m=np.ones(a.shape,bool)
        a[:20,:30]=.9;a[:40,512:572]=.9;b[100:110,512:532]=.9
        # 600/0 has greater ratio than 2400/200. Global fraction below 1.5%.
        self.assertFalse(old.compute_direction_comparison(a,b,m)['hit_alert'])
        self.assertEqual(regional_review(a,b,m)['by_direction'],{'forward':1,'reverse':0})
    def test_local_reverse_despite_global_forward(self):
        a=np.zeros((512,1536),np.float32);b=a.copy();m=np.ones(a.shape,bool)
        a[:50,:60]=b[:50,:60]=.9;a[:10,60:160]=.9;b[:40,512:562]=.9
        # Global forward dominates; bilateral overlap globally obscures reverse-only tile.
        self.assertFalse(old.compute_direction_comparison(a,b,m)['hit_alert'])
        self.assertEqual(regional_review(a,b,m)['by_direction']['reverse'],1)
    def test_unknown_and_invalid(self):
        a=np.zeros((10,10),np.float32);m=np.zeros_like(a,bool)
        self.assertEqual(regional_review(a,a,m)['status'],'unresolved');m[:]=True
        self.assertEqual(regional_review(a,a,m)['status'],'routine_review');a[0,0]=np.nan
        with self.assertRaises(ValueError):regional_review(a,a,m)
if __name__=='__main__':unittest.main()
