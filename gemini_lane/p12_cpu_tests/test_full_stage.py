"""Stage meshes offline on synthetic chunks; compare with old full renderer."""
from pathlib import Path
import sys,tempfile,json,importlib.util
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'patched/gemini_lane'))
spec=importlib.util.spec_from_file_location('ssr_original',root/'pinned/gemini_lane/stream_snap_render.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
from test_stream_snap_render import _write_local_zarr_v2,_build_curved_sheet_fixture,_curved_surface_z_and_grad
from staged_rv3 import stage_meshes,load_stage

def main():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);v,x,y,z=_build_curved_sheet_fixture();zz,yy,xx=np.indices(v.shape);ct=np.clip(50+v.astype(float)*.5+zz+xx*.1+yy*.2,0,255).astype(np.uint8)
        m7=_write_local_zarr_v2(p/'m7',v);cp=_write_local_zarr_v2(p/'ct',ct);meshes=[]
        for j in range(2):
            m=p/f'mesh{j}';old.write_tifxyz(m,x[j:30+j,j:30+j],y[j:30+j,j:30+j],z[j:30+j,j:30+j],{'scale':[1.,1.]});meshes.append(m)
        stage_meshes(meshes,cp,m7,p/'stage',workers=2,ct_mb=8,m7_mb=8)
        checks=0;rows=[]
        for j,m in enumerate(meshes):
            a,b,c,_,meta=old.read_tifxyz(m)
            with old.RemoteZarrV2(m7,'0',workers=2,cache_mb=8) as r:a,b,c,g=old.snap_tifxyz_to_m7(a,b,c,r,passes=2)
            with old.RemoteZarrV2(cp,'0',workers=2,cache_mb=8) as r:stack,mask=old.render_surface_stack(a,b,c,r,num_slices=17,scale=meta['scale'])
            s,t,rec=load_stage(p/'stage'/f'{j:04d}');assert np.array_equal(s,stack) and np.array_equal(t,mask);checks+=2;rows.append({'values':int(s.size),'valid':int(t.sum()),'counters':rec['provenance']['delta_counters']})
        # high raw support, missing chunk: 2-pass result must be checked independently.
        x=x.copy();y=y.copy();z=z.copy();ztrue,*_= _curved_surface_z_and_grad(x,y);z[:24]=ztrue[:24].astype(np.float32);(m7/'0/2/2/2').unlink()
        out=[]
        for passes in (1,2):
            with old.RemoteZarrV2(m7,'0',workers=2,cache_mb=8) as r:out.append(old.snap_tifxyz_to_m7(x,y,z,r,passes=passes))
        a,b=[k[3] for k in out];changed=int(np.any(np.stack(out[0][:3])!=np.stack(out[1][:3]),axis=0).sum());result={'checks':checks,'stage_rows':rows,'high_raw_case':{'raw':a['on_sheet_raw'],'one':a['on_sheet_after_snap'],'two':b['on_sheet_after_snap'],'changed_vertices':changed}}
        Path('full_stage_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
