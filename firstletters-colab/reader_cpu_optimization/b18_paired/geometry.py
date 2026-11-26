"""B20: fixed multi-scale bidirectional relative depth matching on real CT."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
from registration import simulate,scores,SHIFTS,PIN

METHODS=('m5','m15','cycle5','cycle15','adaptive')


def confidence(c):
    idx=c.argmax(0);best=np.take_along_axis(c,idx[None],0)[0];second=np.partition(c,-2,axis=0)[-2]
    with np.errstate(invalid='ignore'):gap=best-second
    return SHIFTS[idx],best,gap


def gate(c,r,margin):
    return np.isfinite(c[1])&(c[1]>=r)&(c[2]>=margin)


def combine(f5,f15,b5,b15):
    a5=gate(f5,.6,.1);a15=gate(f15,.6,.1)
    cyc5=a5&gate(b5,.6,.1)&(f5[0]==-b5[0]);cyc15=a15&gate(b15,.6,.1)&(f15[0]==-b15[0])
    high=cyc5&gate(f5,.85,.2)&gate(b5,.85,.2)
    return [(f5[0],a5),(f15[0],a15),(f5[0],cyc5),(f15[0],cyc15),(np.where(high,f5[0],f15[0]),high|cyc15)]


def pred(a,b):
    return combine(confidence(scores(a,b,5)[1]),confidence(scores(a,b,15)[1]),confidence(scores(b,a,5)[1]),confidence(scores(b,a,15)[1]))


def count(p,ok,mask,truth):
    m=mask&ok;v={'attempted':int(mask.sum()),'admitted':int(m.sum())}
    if truth is not None:
        e=np.abs(p-truth);v.update(exact=int((m&(e==0)).sum()),within1=int((m&(e<=1)).sum()),abs_error=int(e[m].sum()))
    return v


def tests():
    rng=np.random.default_rng(20);a=rng.integers(20,160,(32,32,16),dtype=np.uint8);truth=np.full((32,32),2);b=simulate(a,truth);m=np.s_[8:-8,8:-8]
    r=pred(a,b);assert all((p[m][ok[m]]==2).all() and ok[m].any() for p,ok in r)
    z=np.zeros_like(a);assert all(not ok.any() for p,ok in pred(z,z))
    shape=(3,3);f=(np.full(shape,2),np.ones(shape),np.ones(shape));b=(np.full(shape,2),np.ones(shape),np.ones(shape));r=combine(f,f,b,b);assert all(not r[i][1].any() for i in (2,3,4))
    mask=np.ones((3,3),bool);before=mask.copy();count(r[0][0],r[0][1],mask,np.zeros((3,3),int));assert np.array_equal(mask,before)
    return 4


def main():
    t=time.perf_counter();d=Path(sys.argv[1]);out=Path(sys.argv[2]);p=d/'ct-tiles.npy'
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PIN
    tiles=np.load(p,mmap_mode='r',allow_pickle=False);assert tiles.shape==(28,512,512,16);checks=tests();yy,xx=np.indices((128,128));interior=(yy>=7)&(yy<121)&(xx>=7)&(xx<121);ramp=np.rint(-3+6*(xx+yy)/254).astype(int);step=np.where(xx<yy,-2,2);seam=np.abs(xx-yy)<8;occ=(yy>=58)&(yy<70)&(xx>=58)&(xx<70);rows=[]
    for i,tile in enumerate(tiles):
        a=np.asarray(tile[192:320,192:320]);mask=interior&(a[:,:,4:12].std(2)>=4);rng=np.random.default_rng(2000+i)
        def noisy(truth):
            b=simulate(a,truth);return np.clip(np.rint(1.15*b+8+rng.normal(0,3,b.shape)),0,255).astype(np.uint8)
        nr=noisy(ramp);ns=noisy(step);oc=ns.copy();oc[occ]=rng.integers(0,256,oc[occ].shape,dtype=np.uint8)
        for scene,b,truth,extra in [('noisy_ramp',nr,ramp,None),('noisy_step',ns,step,seam),('occluded_step',oc,step,occ),('decoy_xy',np.asarray(tile[:128,:128]),None,None),('decoy_tile',np.asarray(tiles[(i+1)%28,192:320,192:320]),None,None)]:
            before=mask.copy()
            for method,(p,ok) in zip(METHODS,pred(a,b)):
                v=count(p,ok,mask,truth);rows.append(dict(tile=i,scene=scene,method=method,**v,extra=count(p,ok,mask&extra,truth) if extra is not None else None))
            assert np.array_equal(mask,before)
    summary=[]
    for scene in ('noisy_ramp','noisy_step','occluded_step','decoy_xy','decoy_tile'):
        for method in METHODS:
            r=[v for v in rows if v['scene']==scene and v['method']==method];keys=['attempted','admitted']+([] if scene.startswith('decoy') else ['exact','within1','abs_error']);v={k:sum(x[k] for x in r) for k in keys}
            v['extra']={k:sum(x['extra'][k] for x in r) for k in keys} if r[0]['extra'] is not None else None
            summary.append(dict(scene=scene,method=method,**v))
    idx={(r['scene'],r['method']):r for r in summary};a=idx['noisy_step','adaptive']['extra'];b=idx['noisy_step','m15']['extra'];seam_delta=100*(a['exact']/a['attempted']-b['exact']/b['attempted']);a=idx['noisy_ramp','adaptive'];precision=a['exact']/a['admitted'];decoy={}
    for s in ('decoy_xy','decoy_tile'):
        a,b=idx[s,'adaptive'],idx[s,'m15'];decoy[s]=100*(a['admitted']/a['attempted']-b['admitted']/b['attempted'])
    gates={'seam_delta_pp':seam_delta,'ramp_exact_precision':precision,'decoy_delta_pp':decoy,'pass':bool(seam_delta>=5 and precision>=.99 and all(x<=.25 for x in decoy.values()))}
    res=dict(schema='b20.bidirectional_multiscale.v1',source_sha256=PIN,tests=checks,wall_seconds=time.perf_counter()-t,rows=rows,summary=summary,gates=gates,scope='Real texture, synthetic relative shifts and decoys; not actual sheet identity or mesh correction.')
    out.write_text(json.dumps(res,separators=(',',':'),allow_nan=False)+'\n');print(json.dumps({k:v for k,v in res.items() if k not in ('rows','summary')},indent=2))
    for r in summary: print(r['scene'],r['method'],r['admitted'],r.get('exact'),r['attempted'],r['extra'])

if __name__=='__main__':main()
