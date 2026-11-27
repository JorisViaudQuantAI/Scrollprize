"""Exact paired depth-order preparation on pinned public byte-valued CT."""
import hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
from fast_mad import reference, fast_guarded, _lut

PIN='0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33'


def padded(raw, depth, height, width, y, x, reverse=False):
    if raw.ndim!=3 or not 0<raw.shape[2]<=depth: raise ValueError('bad depth')
    h,w,d=raw.shape
    if not (0<=y<=height-h and 0<=x<=width-w): raise ValueError('bad placement')
    start=(depth-d)//2
    a=np.zeros((height,width,depth),dtype=np.float32)
    a[y:y+h,x:x+w,start:start+d]=raw[:,:,::-1] if reverse else raw
    return a.transpose(2,0,1)


def separate(raw, depth, height, width, y, x, normalize=reference):
    return tuple(normalize(padded(raw,depth,height,width,y,x,reverse)) for reverse in (False,True))


def paired(raw, depth, height, width, y=0, x=0):
    """Reuse exact integer LUT; independently normalize unsupported/degenerate input."""
    if raw.ndim!=3 or not 0<raw.shape[2]<=depth: raise ValueError('bad depth')
    h,w,d=raw.shape
    if not (0<=y<=height-h and 0<=x<=width-w): raise ValueError('bad placement')
    if raw.dtype!=np.uint8 or np.__version__!='2.3.5':
        return separate(raw,depth,height,width,y,x),False
    start=(depth-d)//2
    a=np.zeros((depth,height,width),dtype=np.uint8)
    a[start:start+d,y:y+h,x:x+w]=raw.transpose(2,0,1)
    lut=_lut(a)
    if lut is None:
        return separate(raw,depth,height,width,y,x),False
    f=lut[a]
    r=f.copy();r[start:start+d]=f[start:start+d][::-1]
    return (f,r),True


def equal(a,b):
    assert a.dtype==b.dtype==np.float32 and a.shape==b.shape and a.flags.c_contiguous and b.flags.c_contiguous
    assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),float(np.max(np.abs(a-b)))


def tests():
    rng=np.random.default_rng(1800);n=0;fallback=0
    for depth in range(1,18):
        for d in range(1,depth+1):
            for kind in range(5):
                a=rng.integers(0,256,(7,9,d),dtype=np.uint8)
                if kind==1:a.fill(42)
                if kind==2:a=np.where(rng.random(a.shape)<.96,40,a).astype(np.uint8)
                if kind==3:a=a.astype(np.float32)/3
                if kind==4:a=a[::-1,::-1,::-1]
                p=(depth,10,12,1,2);b,used=paired(a,*p);fallback+=not used
                for v,w in zip(b,separate(a,*p)):equal(v,w);n+=1
    for a in [np.zeros((2,3,2),np.uint8),np.full((2,3,2),255,np.uint8),np.array([[[0.,-0.,np.nan]]],np.float32)]:
        with np.errstate(all='ignore'):
            b,used=paired(a,5,3,4);fallback+=not used
            for v,w in zip(b,separate(a,5,3,4,0,0)):equal(v,w);n+=1
    # A 16-plane band in a 17-plane output leaves the trailing pad fixed.
    a=np.broadcast_to(np.arange(16,dtype=np.uint8),(2,3,16));r=paired(a,17,2,3)[0][1]
    f=paired(a,17,2,3)[0][0];assert np.array_equal(f[-1],r[-1]) and np.array_equal(f[:16][::-1],r[:16]);n+=1
    return n,fallback


def main():
    p=Path(sys.argv[1])/'ct-tiles.npy';out=Path(sys.argv[2]);t=time.perf_counter()
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PIN
    a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(28,512,512,16) and a.dtype==np.uint8
    n,fb=tests();rows=[];timings=[]
    for i,tile in enumerate(a):
        for y,x in [(0,0),(64,64),(128,128)]:
            raw=tile[y:y+384,x:x+384];args=(17,384,384,0,0)
            ref=separate(raw,*args);old=separate(raw,*args,normalize=fast_guarded);new,used=paired(raw,*args)
            for k in range(2):equal(ref[k],old[k]);equal(ref[k],new[k])
            rows.append(dict(tile=i,y=y,x=x,shared=used,sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in new]))
            if (y,x)!=(64,64):continue
            funcs=[lambda:separate(raw,*args),lambda:separate(raw,*args,normalize=fast_guarded),lambda:paired(raw,*args)[0]]
            for rep in range(3):
                for idx in np.roll(np.arange(3),i+rep):
                    st=time.perf_counter_ns();v=funcs[idx]();dt=time.perf_counter_ns()-st
                    for k in range(2):equal(ref[k],v[k])
                    timings.append([i,rep,('reference','b15','paired')[idx],int(dt)])
    med={m:float(np.median([v[3] for v in timings if v[2]==m])/1e6) for m in ('reference','b15','paired')}
    s=dict(schema='b18.paired_depth_preparation.v1',source_sha256=PIN,python=sys.version.split()[0],numpy=np.__version__,synthetic_checks=n,synthetic_fallback_cases=fb,real_pairs=len(rows),real_shared=sum(r['shared'] for r in rows),real_bitwise_checks=4*len(rows),timed_bitwise_checks=2*len(timings),timings_count=len(timings),median_ms=med,speedup_vs_reference=med['reference']/med['paired'],speedup_vs_b15=med['b15']/med['paired'],wall_seconds=time.perf_counter()-t,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows,timings=timings,scope='Prepared 16-plane CT padded to 17; exact preprocessing only; no new Reader model forward.')
    out.write_text(json.dumps(s,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in s.items() if k not in ('rows','timings')},indent=2))

if __name__=='__main__':main()
