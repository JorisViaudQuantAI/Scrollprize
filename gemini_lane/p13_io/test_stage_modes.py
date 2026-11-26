"""End-to-end synthetic staged M7/CT strict byte parity across reader options."""
import json,sys,time,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from staged_rv3 import stage_meshes,load_stage
from stream_snap_render import write_tifxyz
from test_stream_snap_render import _build_curved_sheet_fixture,_write_local_zarr_v2

def run():
    v,x,y,z=_build_curved_sheet_fixture();zz,yy,xx=np.indices(v.shape)
    ct=np.clip(50+v.astype(float)*.5+zz+xx*.1+yy*.2,0,255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t);m7=_write_local_zarr_v2(p/'m7',v);ctz=_write_local_zarr_v2(p/'ct',ct)
        meshes=[]
        for j in range(2):
            mes=p/f'mesh{j}';write_tifxyz(mes,x[j:30+j,j:30+j],y[j:30+j,j:30+j],z[j:30+j,j:30+j],{'scale':[1.,1.]});meshes.append(mes)
        arms=[('legacy',False),('stencil',False),('stencil',True)]
        raw=[];rows=[];checks=0
        for idx,(reader_mode,reuse) in enumerate(arms):
            out=p/f'staged{idx}';start=time.perf_counter()
            stage_meshes(meshes,ctz,m7,out,workers=2,ct_mb=8,m7_mb=8,reader_mode=reader_mode,reuse_center_samples=reuse,prefetch_mb=2)
            rows.append(dict(mode=reader_mode,reuse_center_samples=reuse,seconds=time.perf_counter()-start,items=[]))
            vals=[]
            for i in range(len(meshes)):
                a,b,receipt=load_stage(out/f'{i:04d}');vals.append((np.asarray(a).copy(),np.asarray(b).copy()))
                rows[-1]['items'].append({'values':int(a.size),'valid':int(b.sum()),'delta_counters':receipt['provenance']['delta_counters'],'reader_mode':receipt['provenance']['reader_mode']})
                if idx:
                    assert np.array_equal(a.view(np.uint8),raw[i][0].view(np.uint8))
                    assert np.array_equal(b,raw[i][1])
                    assert receipt['provenance']['geometry']==baselines[i]
                    checks+=int(a.size)+int(b.size)+len(baselines[i])
            if idx==0:raw=vals;baselines=[load_stage(out/f'{i:04d}')[2]['provenance']['geometry'] for i in range(len(meshes))]
    return dict(schema='p13.stage_engines.v1',checks=checks,rows=rows,scope='Generated small local M7/CT only; no actual target CT, no GPU.')

if __name__=='__main__':
    res=run();(Path(sys.argv[1]).write_text(json.dumps(res,indent=2)+'\n') if len(sys.argv)>1 else None);print(json.dumps(res,indent=2))
