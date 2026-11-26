"""Exact end-to-end SSR geometry parity and controlled cache/pass stress (synthetic only)."""
import importlib.util,json,sys,tempfile,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'pinned/gemini_lane'))
import stream_snap_render as ref
from test_stream_snap_render import _write_local_zarr_v2,_build_curved_sheet_fixture
spec=importlib.util.spec_from_file_location('ssr_candidate',root/'patched/gemini_lane/stream_snap_render.py');new=importlib.util.module_from_spec(spec);sys.modules[spec.name]=new;spec.loader.exec_module(new)

def main():
    rng=np.random.default_rng(12022);checks=0;rows=[];m,x,y,z=_build_curved_sheet_fixture();h,w=x.shape
    scenarios=[('curved',m,x,y,z)]
    yy,xx=np.indices((64,64));xx=xx.astype(np.float32);yy=yy.astype(np.float32)
    # Stable single sheet, multi-ridge competition, truncated/holes, wrong normal starts.
    zz,vy,vx=np.indices((64,64,64));plane=(255*np.exp(-.5*((zz-(30+1.4*np.sin(vx/8)))/.9)**2)).astype(np.uint8)
    dual=np.maximum(plane,(245*np.exp(-.5*((zz-(40+1.2*np.cos(vy/7)))/.9)**2)).astype(np.uint8))
    for name,vol,dep in [('single',plane,29.),('dual',dual,35.),('dual_offset',dual,32.),('split',np.where(vx<32,plane,np.roll(plane,10,axis=0)),35.)]:
        u=xx[12:52,12:52].copy();v=yy[12:52,12:52].copy();d=np.full_like(u,dep)+rng.uniform(-.4,.4,u.shape).astype(np.float32);scenarios.append((name,vol,u,v,d))
    x2=x.copy();y2=y.copy();z2=z.copy();x2[:5,:5]=y2[:5,:5]=z2[:5,:5]=-1;scenarios.append(('missing',m,x2,y2,z2))
    with tempfile.TemporaryDirectory() as tmp:
        for name,vol,x,y,z in scenarios:
            p=_write_local_zarr_v2(Path(tmp)/name,vol,chunks=(16,16,16))
            if name=='missing':(p/'0/2/2/2').unlink()
            rec={}
            for passes in (1,2):
                for mod,label in [(ref,'scalar'),(new,'batch')]:
                    with mod.RemoteZarrV2(p,'0',workers=2,cache_mb=8) as r:
                        t=time.perf_counter();res=mod.snap_tifxyz_to_m7(x,y,z,r,passes=passes);dt=time.perf_counter()-t
                        rec[label,passes]=(res,dt,r.bytes_fetched)
                a,b=rec['scalar',passes][0],rec['batch',passes][0]
                assert all(np.array_equal(v,q) for v,q in zip(a[:3],b[:3])) and a[3]==b[3];checks+=4
            a,b=rec['batch',1],rec['batch',2];m1,m2=a[0][3],b[0][3]
            rows.append(dict(scene=name,vertices=m1['valid_vertex_count'],on_raw=m1['on_sheet_raw'],on_1=m1['on_sheet_after_snap'],on_2=m2['on_sheet_after_snap'],step_1=m1['adjacent_signed_step_ge_6vx_frac'],step_2=m2['adjacent_signed_step_ge_6vx_frac'],scalar2_s=rec['scalar',2][1],batch2_s=b[1],one_s=a[1],two_s=b[1],bytes_1=a[2],bytes_2=b[2],changed_vertices=int(np.any(np.stack(a[0][:3])!=np.stack(b[0][:3]),axis=0).sum())))
        # Three overlapping synthetic meshes, exact geometry and render support on fresh vs shared caches.
        p=_write_local_zarr_v2(Path(tmp)/'cache',plane,chunks=(16,16,16));xs=xx[16:40,16:40];ys=yy[16:40,16:40];zs=np.full_like(xs,29.)
        qs=[(xs+i,ys,zs) for i in (0,1,2)];out={};receipts={}
        for policy in ('fresh','shared'):
            shared=new.RemoteZarrV2(p,'0',workers=2,cache_mb=8) if policy=='shared' else None
            total=0;hits=0;vals=[];t=time.perf_counter()
            try:
                for a,b,c in qs:
                    r=shared or new.RemoteZarrV2(p,'0',workers=2,cache_mb=8);start=r.bytes_fetched;ch=r.cache_hits
                    try:vals.append(new.snap_tifxyz_to_m7(a,b,c,r,passes=2));total+=r.bytes_fetched-start;hits+=r.cache_hits-ch
                    finally:
                        if shared is None:r.close()
            finally:
                if shared:shared.close()
            out[policy]=vals;receipts[policy]=dict(bytes=total,hits=hits,seconds=time.perf_counter()-t)
        for a,b in zip(out['fresh'],out['shared']):assert all(np.array_equal(v,q) for v,q in zip(a[:3],b[:3])) and a[3]==b[3];checks+=4
    r=dict(checks=checks,rows=rows,cache=receipts,scope='synthetic sheet volumes, not measured remote cache or real anatomy');Path('ssr_result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
