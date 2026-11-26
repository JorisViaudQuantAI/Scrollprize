"""P12 audit: exact scalar/batch parity, real texture proxies, synthetic geometry."""
import argparse, hashlib, json, sys, time, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'pinned/gemini_lane'))
import stream_snap_render as ssr
from ridge_batch import pick_nearest_ridge_offsets as bat
from test_stream_snap_render import TestStreamSnapRender
REF=ssr._pick_nearest_ridge_offset

def scalar(p,o,t,s):return np.array([REF(x,o,t,s) for x in p],dtype=np.float64)
def check(p,o,t,s):
    a=scalar(p,o,t,s);b=bat(p,o,t,s,REF)
    assert np.array_equal(a.view(np.uint64),b.view(np.uint64)), (np.max(np.abs(a-b)),np.flatnonzero(a!=b)[:10])
    return len(p)

def main(ct_path):
    rng=np.random.default_rng(12021);o=np.arange(-12,13,dtype=np.float64);count=0;rows=[]
    p=rng.uniform(0,255,(4096,25)).astype(np.float32);p[rng.random(p.shape)<.15]=-np.inf
    discrete=rng.choice(np.array([-np.inf,0,50,51,52,80,200,255],np.float32),(4096,25))
    plateau=np.full((4096,25),-np.inf,np.float32)
    for j in range(len(plateau)):
        a,b=sorted(rng.integers(0,25,2));plateau[j,a:b+1]=rng.choice([51,80,255])
    # Near ties, threshold exactly, multiple equidistant runs, edge peaks and missing corners.
    cases=np.array([[0]*25,[51]*25,[-np.inf]*25,[0]*12+[51]+[0]*12,[255]+[0]*24,[0]*24+[255],[0]*8+[51]+[0]*7+[51]+[0]*8],np.float32)
    count+=check(cases,o,51.,1.)
    for k in (1,2,3,7,13,25,51):
        a=rng.uniform(0,255,(512,k)).astype(np.float32);a[rng.random(a.shape)<.2]=-np.inf
        for step in (.5,1.,1.7):
            oo=(np.arange(k,dtype=np.float64)-k//2)*step
            for thresh in (0.,51.,255.):count+=check(a,oo,thresh,step)
    # Exact float32 CT-derived numeric profiles; not M7 probabilities or anatomy.
    ct=np.load(ct_path,mmap_mode='r',allow_pickle=False)
    assert hashlib.file_digest(open(ct_path,'rb'),'sha256').hexdigest()=='0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33'
    texture=np.stack([t[192:224,192:224] for t in ct]).reshape(-1,16).astype(np.float32)
    texture=np.pad(texture,((0,0),(4,5)),constant_values=-np.inf)
    for name,q in [('continuous',p),('discrete',discrete),('plateau',plateau),('ct_texture_proxy',texture)]:
        ts=[];tb=[];checks=0
        for rep in range(3):
            for fn,times in ((scalar,ts),(lambda a,b,c,d:bat(a,b,c,d,REF),tb)) if rep%2==0 else ((lambda a,b,c,d:bat(a,b,c,d,REF),tb),(scalar,ts)):
                t=time.perf_counter();res=fn(q,o,51.,1.);times.append(time.perf_counter()-t)
            checks+=check(q,o,51.,1.)
        count+=checks;rows.append(dict(group=name,profiles=len(q),scalar_seconds=float(np.median(ts)),batch_seconds=float(np.median(tb)),speedup=float(np.median(ts)/np.median(tb))))
    # Reference uses real pinned source. Patch its scalar site at batch level, without modifying pinned fixture.
    print(json.dumps(dict(checks=count,rows=rows),indent=2))
    return dict(checks=count,rows=rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ct',type=Path);p.add_argument('--output',type=Path,default=Path('ridge_result.json'));a=p.parse_args();v=main(a.ct);a.output.write_text(json.dumps(v,indent=2)+'\n')
