"""P13: cached M7 center values must replay baseline bit-for-bit on immutable source.

The precise no-download test uses public-license synthetic volume fixtures, not
eligible-scroll coordinates, CT or model predictions.
"""
from __future__ import annotations
import argparse,json,sys,time,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import stream_snap_render as ssr
from test_stream_snap_render import _build_curved_sheet_fixture,_write_local_zarr_v2

class Meter(ssr.RemoteZarrV2):
    def __init__(self,*args,**kwargs):
        self.calls=0;self.points=0
        super().__init__(*args,**kwargs)
    def sample_trilinear(self,coords,**kwargs):
        self.calls+=1;self.points+=int(np.prod(np.asarray(coords).shape[:-1]))
        return super().sample_trilinear(coords,**kwargs)

def run():
    rng=np.random.default_rng(13077)
    m,x,y,z=_build_curved_sheet_fixture()
    x=x.copy();y=y.copy();z=z.copy()
    variants={'curved':(m,x,y,z),'partial':(m,x,y,z)}
    X,Y,Z=x.copy(),y.copy(),z.copy();X[:8,:10]=Y[:8,:10]=Z[:8,:10]=-1
    variants['holes']=(m,X,Y,Z)
    zz,yy,xx=np.indices(m.shape)
    variants['double']=(np.maximum(m,(255*np.exp(-.5*((zz-25)/.9)**2)).astype(np.uint8)),x,y,z)
    variants['switch']=((255*np.exp(-.5*((zz-(20+np.where(xx<30,8,-6)))/.7)**2)).astype(np.uint8),x,y,z)
    variants['flat']=(np.full(m.shape,60,np.uint8),x,y,z)
    variants['no_sheet']=(np.zeros(m.shape,np.uint8),x,y,z)
    nchecks=0;rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for label,(vol,a,b,c) in variants.items():
            p=_write_local_zarr_v2(Path(tmp)/label,vol,chunks=(16,16,16))
            if label=='partial':(p/'0/2/2/2').unlink()
            for passes in (0,1,2,3):
                samples=[]
                for reuse in (False,True):
                    with Meter(p,'0',workers=2,cache_mb=1) as reader:
                        t=time.perf_counter();result=ssr.snap_tifxyz_to_m7(a,b,c,reader,passes=passes,reuse_center_samples=reuse)
                        samples.append((result,reader.calls,reader.points,reader.bytes_fetched,time.perf_counter()-t))
                f,g=samples
                for u,v in zip(f[0][:3],g[0][:3]):
                    assert np.array_equal(u.view(np.uint32),v.view(np.uint32)),(label,passes,'coords')
                    nchecks+=u.size
                assert f[0][3]==g[0][3],(label,passes,'telemetry')
                nchecks+=len(f[0][3]);rows.append(dict(scene=label,passes=passes,baseline_calls=f[1],reused_calls=g[1],baseline_points=f[2],reused_points=g[2],baseline_seconds=f[4],reused_seconds=g[4],baseline_bytes=f[3],reused_bytes=g[3]))
    assert len(rows)==28 and nchecks>100000
    assert sum(r['reused_calls'] for r in rows)<sum(r['baseline_calls'] for r in rows)
    return dict(schema='p13.snap_reuse.v1',checks=nchecks,rows=rows,summary=dict(calls_baseline=sum(r['baseline_calls'] for r in rows),calls_reused=sum(r['reused_calls'] for r in rows),seconds_baseline=sum(r['baseline_seconds'] for r in rows),seconds_reused=sum(r['reused_seconds'] for r in rows)),scope='Synthetic immutable M7 only, no remote network/true anatomical ground truth.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();r=run();args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
