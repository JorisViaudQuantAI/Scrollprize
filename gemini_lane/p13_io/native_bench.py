"""P13 bounded native CT benchmark on the public awarded 0846B mesh only."""
import argparse,hashlib,itertools,json,sys,time,threading,resource
from pathlib import Path
import numpy as np
from stream_snap_render import RemoteZarrV2,read_tifxyz,render_surface_stack,compute_grid_normals
from stencil_reader import StencilReader
PINS={'x.tif':'010fea74c444119e2e15928ab1c1c570a201b079','y.tif':'f9476664a61c8b52e3ab121694fda1722ae851a5','z.tif':'d86564e2ebea49d0f6f8be227271a50fbc98aca5','meta.json':'548ed7c9ece10496101ffc17721b8e4fbbfdc36a'}
URL='https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0846B/volumes/20250804142305-9.362um-1.2m-113keV-masked.zarr'
LIMIT=512*1024**2
def blob(p):
    b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def instrument(cls,known):
    class Metered(cls):
        def __init__(self,*args,**kw):
            self.receipt_lock=threading.Lock();self.wire=0;self.calls=0;self.requested=set();self.needed=0;self.responses=0
            super().__init__(*args,**kw)
        def _get_bytes(self,key,*,allow_404=True):
            with self.receipt_lock:
                if self.wire>=LIMIT:raise RuntimeError('native byte budget exhausted')
            raw=super()._get_bytes(key,allow_404=allow_404)
            h=None if raw is None else hashlib.sha256(raw).hexdigest()
            with self.receipt_lock:
                self.responses+=1;self.wire+=0 if raw is None else len(raw)
                if key in known and known[key]!=h:raise RuntimeError('source body changed between arms')
                known[key]=h
                if self.wire>LIMIT:raise RuntimeError('native post-response byte budget exhausted')
            return raw
        def get_chunks(self,idxs):
            keys=list(idxs);self.calls+=1;self.needed+=len(keys);self.requested.update(keys)
            return super().get_chunks(keys)
    return Metered
def choose(x,y,z,valid):
    # Two disjoint public surface contexts, deterministic geometry-only selection.
    n=16;slots=[]
    for a in range(0,x.shape[0]-n+1,n):
        for b in range(0,x.shape[1]-n+1,n):
            if valid[a:a+n,b:b+n].all():slots.append((a,b))
    if len(slots)<2:raise ValueError('not enough full-valid contexts')
    first=slots[0];second=max(slots[1:],key=lambda p:((p[0]-first[0])**2+(p[1]-first[1])**2,-p[0],-p[1]))
    contexts=[tuple(v[a:a+n,b:b+n] for v in (x,y,z)) for a,b in (first,second)]
    normal,ok=compute_grid_normals(x,y,z,valid=valid)
    points=np.stack([x,y,z],-1);keep=np.flatnonzero((valid&ok).ravel())
    # 24 maximin distributed geometry points, not prediction-selected.
    coords=points.reshape(-1,3)[keep].astype(float);chosen=[0];dist=np.sum((coords-coords[0])**2,axis=1)
    for _ in range(min(24,len(coords))-1):
        j=int(np.argmax(dist));chosen.append(j);dist=np.minimum(dist,np.sum((coords-coords[j])**2,axis=1))
    idx=keep[np.array(chosen)]
    rays=points.reshape(-1,3)[idx,None,:].astype(float)+normal.reshape(-1,3)[idx,None,:].astype(float)*np.arange(-12,13,dtype=float)[None,:,None]
    return contexts,rays
def main():
    p=argparse.ArgumentParser();p.add_argument('mesh',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for n,h in PINS.items():assert blob(a.mesh/n)==h,n
    x,y,z,valid,meta=read_tifxyz(a.mesh)
    if x.ndim!=2 or y.shape!=x.shape or z.shape!=x.shape or min(x.shape)<16 or not np.allclose(meta.get('scale',[]),[.05,.05],rtol=1e-7,atol=0):
        raise ValueError('unexpected pinned mesh geometry or physical scale')
    print('P13_PREFLIGHT',json.dumps(dict(shape=list(x.shape),scale=meta['scale'],valid_count=int(valid.sum()))),flush=True)
    contexts,rays=choose(x,y,z,valid);source_hashes={};rows=[];reference={};checks=0
    for name,cls in [('legacy',RemoteZarrV2),('stencil',StencilReader)]:
        opts=dict(cache_mb=128,workers=8,timeout=45,max_retries=3)
        if name=='stencil':opts['prefetch_mb']=64
        with instrument(cls,source_hashes)(URL,'0',**opts) as r:
            assert r.dtype==np.dtype(np.uint8) and r.fill_value==0
            for phase in ('first','repeat'):
                for k,ctx in enumerate(contexts+[None]):
                    before=(r.bytes_fetched,r.chunks_fetched,r.cache_hits,r.calls,r.needed,r.wire,r.responses);t=time.perf_counter()
                    if ctx is None:
                        v,s=r.sample_trilinear(rays,return_support=True);out=(v,s)
                    else:
                        v,s,m=render_surface_stack(*ctx,r,num_slices=17,scale=meta['scale'],tile_size=256,return_support=True,strict_uint8=True);out=(v,s,*m.values())
                    elapsed=time.perf_counter()-t
                    after=(r.bytes_fetched,r.chunks_fetched,r.cache_hits,r.calls,r.needed,r.wire,r.responses)
                    hashes=[sha(q) for q in out]
                    if k not in reference:reference[k]=tuple(np.array(q,copy=True) for q in out)
                    else:
                        assert len(reference[k])==len(out)
                        for aa,bb in zip(reference[k],out):
                            assert aa.dtype==bb.dtype and aa.shape==bb.shape and np.array_equal(aa.view(np.uint8),bb.view(np.uint8)),(k,name,phase)
                            checks+=1
                    v,s=out[:2]
                    row=dict(method=name,phase=phase,context=k,kind='rays' if ctx is None else 'render',values=int(v.size),supported=int(s.sum()),seconds=elapsed,body_bytes=after[0]-before[0],decoded=after[1]-before[1],cache_hits=after[2]-before[2],get_chunks_calls=after[3]-before[3],requested_chunk_refs=after[4]-before[4],received_body_bytes=after[5]-before[5],responses=after[6]-before[6],hashes=hashes)
                    if name=='stencil':row['sampler_stats']=dict(r.sampler_stats)
                    rows.append(row);print('P13_NATIVE_ROW',json.dumps(row),flush=True)
                    (a.output/'partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
            print('P13_NATIVE_ARM',json.dumps(dict(method=name,bytes=r.wire,unique_requested=len(r.requested))),flush=True)
    result=dict(schema='p13.native_0846b.v1',source_mesh_commit='daabe53b90654f9aa46e1512bf4bef251fd3975b',source_mesh_blobs=PINS,ct_url=URL,rows=rows,bitwise_checks=checks,unique_source_responses=len(source_hashes),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Awarded public control; SSR orientation not winner-recipe reproduction, CT only, no Reader or M7. One ordered remote A/B, post-response limit not exact wire cap.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('P13_NATIVE_DONE',json.dumps(dict(checks=checks,rows=len(rows),unique_source_responses=len(source_hashes),peak_rss_kib=result['peak_rss_kib'])),flush=True)
if __name__=='__main__':main()
