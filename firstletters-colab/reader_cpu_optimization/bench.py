"""Run real prepared CT normalization benchmark, no network forwards."""
import hashlib, json, time, resource, sys, warnings
from pathlib import Path
import numpy as np
from fast_mad import reference,fast_uint8,fast_guarded,_lut

PIN='0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33'
METHODS=('reference','uint8','guarded')


def assert_equal(a,b):
    assert a.dtype==b.dtype==np.float32 and a.shape==b.shape
    if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
        raise AssertionError((np.max(np.abs(a-b)),int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))))


def make_patch(tile,y,x,reverse):
    raw=tile[y:y+384,x:x+384]
    out=np.zeros((384,384,17),np.uint8);out[:,:,:16]=raw[:,:,::-1] if reverse else raw
    return out


def validate():
    rng=np.random.default_rng(1501);checks=0
    for n in [1,2,3,4,5,13,49,50,99,100,101,103,997,1000,1001,10000]:
        for kind in range(16):
            if kind<6:a=rng.integers(0,256,n,dtype=np.uint8)
            elif kind<12:a=rng.integers(0,kind-4,n,dtype=np.uint8)
            else:a=np.where(rng.random(n)<.9,kind,rng.integers(0,256,n)).astype(np.uint8)
            for v in (a,a[::2],a[::-1]):
                r=reference(v);assert_equal(r,fast_uint8(v));assert_equal(r,fast_guarded(v.astype(np.float32)));checks+=2
    cases=[np.zeros((1,5),np.uint8),np.full((2,3),255,np.uint8),np.array([],np.uint8),np.array([-.0,0.,1.,2.],np.float32),np.array([1.2,2.,3.,4.],np.float32),np.array([-1.,0.,1.],np.float32),np.array([0.,256.,12.],np.float32),np.array([1.,np.nan,3.],np.float32),np.array([1.,np.inf,3.],np.float32),np.arange(100,dtype=np.uint16)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for a in cases:
            r=reference(a);assert_equal(r,fast_uint8(a));assert_equal(r,fast_guarded(a));checks+=2
    return checks


def main():
    d=Path(sys.argv[1]);out=Path(sys.argv[2]);started=time.perf_counter()
    p=d/'ct-tiles.npy';assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==PIN
    tiles=np.load(p,mmap_mode='r',allow_pickle=False);assert tiles.shape==(28,512,512,16) and tiles.dtype==np.uint8
    checks=validate();real=[];times=[];warm=False
    for i,tile in enumerate(tiles):
        for y,x in [(0,0),(64,64),(128,128)]:
            for reverse in (False,True):
                patch=make_patch(tile,y,x,reverse);f=patch.astype(np.float32);a=patch.transpose(2,0,1);b=f.transpose(2,0,1)
                ref=reference(b);u=fast_uint8(a);g=fast_guarded(b);assert_equal(ref,u);assert_equal(ref,g)
                real.append(dict(tile=i,y=y,x=x,reverse=reverse,fallback=_lut(a) is None,sha256=hashlib.sha256(ref.tobytes()).hexdigest()))
                if (y,x)!=(64,64):continue
                funs=[lambda:reference(b),lambda:fast_uint8(a),lambda:fast_guarded(b)]
                ends=[lambda:reference(make_patch(tile,y,x,reverse).astype(np.float32).transpose(2,0,1)),lambda:fast_uint8(make_patch(tile,y,x,reverse).transpose(2,0,1)),lambda:fast_guarded(make_patch(tile,y,x,reverse).astype(np.float32).transpose(2,0,1))]
                if not warm:
                    for fun in funs+ends:fun()
                    warm=True
                for kind,fs in [('normalize',funs),('construct_normalize',ends)]:
                    for rep in range(3):
                        for idx in np.roll(np.arange(3),i+rep+int(reverse)):
                            t=time.perf_counter_ns();v=fs[idx]();dt=time.perf_counter_ns()-t
                            assert v.flags.c_contiguous;times.append([i,reverse,kind,rep,METHODS[idx],dt])
    summary={}
    for kind in ('normalize','construct_normalize'):
        med={m:float(np.median([r[-1] for r in times if r[2]==kind and r[4]==m])/1e6) for m in METHODS}
        summary[kind]=dict(median_ms=med,speedup_uint8=med['reference']/med['uint8'],speedup_guarded=med['reference']/med['guarded'])
    result=dict(source_sha256=PIN,python=sys.version.split()[0],numpy=np.__version__,synthetic_bitwise_checks=checks,real_cases=len(real),real_bitwise_checks=2*len(real),real_fallbacks=sum(x['fallback'] for x in real),summary=summary,wall_seconds=time.perf_counter()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,real=real,timings=times)
    out.write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('real','timings')},indent=2))

if __name__=='__main__':main()
