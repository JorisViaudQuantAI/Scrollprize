"""Offline staged format and exact capture tests; no weights or GPU."""
import importlib.util,sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'patched/gemini_lane'))
from staged_rv3 import stage_arrays,load_stage,native_ratio,outside_repo
class TestStage(unittest.TestCase):
    def test_stage_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'stage';a=np.arange(17*7*9,dtype=np.uint8).reshape(17,7,9);b=np.ones((7,9),bool);b[0]=False
            stage_arrays(dest,a,b,{'fixture':'synthetic'});x,y,r=load_stage(dest);self.assertTrue(np.array_equal(x,a));self.assertTrue(np.array_equal(y,b));self.assertIsInstance(x,np.memmap)
            with self.assertRaises(FileExistsError):stage_arrays(dest,a,b,{'fixture':'synthetic'})
            with (dest/'stack.npy').open('r+b') as f:f.seek(-1,2);v=f.read(1);f.seek(-1,2);f.write(bytes([v[0]^1]))
            with self.assertRaises(ValueError):load_stage(dest)
    def test_no_empty_or_inside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=np.zeros((17,4,5),np.uint8)
            with self.assertRaises(ValueError):stage_arrays(p/'bad',a,np.zeros((4,5),bool),{'test':True})
            (p/'.git').mkdir()
            with self.assertRaises(ValueError):outside_repo(p/'data')
    def test_ratio(self):
        rng=np.random.default_rng(1203);w=rng.random((511,389),dtype=np.float32);w[::3]=0;mask=rng.random(w.shape)>.2;p=rng.random(w.shape,dtype=np.float32)*w
        a,b=native_ratio(p,w,mask,(64,128));v=(w>1e-6)&mask;ref=np.where(v,p/np.maximum(w,1e-6),0).astype(np.float32)
        self.assertTrue(np.array_equal(a,ref));self.assertTrue(np.array_equal(b,v));p[0,0]=np.nan
        self.assertTrue(np.array_equal(native_ratio(p,w,mask)[0],ref))
        with self.assertRaises(ValueError):native_ratio(p,np.zeros_like(w),mask)
        p[v]=np.nan
        with self.assertRaises(ValueError):native_ratio(p,w,mask)
if __name__=='__main__':unittest.main()
