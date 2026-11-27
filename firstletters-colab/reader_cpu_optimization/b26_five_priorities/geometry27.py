"""B27: directional support for relative depth matching, real CT + simulated shifts."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter,gaussian_filter
from registration import simulate,scores,SHIFTS,PIN
from base_geometry import confidence,gate,count

def quadrant_scores(a,b,origin):
    def avg(v):return uniform_filter(v.mean(2),size=8,origin=origin,mode='nearest')
    u=a[:,:,4:12].astype(np.float64);mu=avg(u);vu=np.maximum(avg(u*u)-mu*mu,0)
    rr=[]
    for d in SHIFTS:
        v=b[:,:,4+d:12+d].astype(np.float64);mv=avg(v);vv=np.maximum(avg(v*v)-mv*mv,0);den=np.sqrt(vu*vv);q=np.full(den.shape,-np.inf)
        np.divide(avg(u*v)-mu*mv,den,out=q,where=den>1e-8);rr.append(np.clip(q,-1,1,where=np.isfinite(q),out=q))
    return confidence(np.stack(rr))

def predict(a,b):
    f15=confidence(scores(a,b,15)[1]);base=gate(f15,.6,.1)
    f5=confidence(scores(a,b,5)[1]);b5=confidence(scores(b,a,5)[1]);valid5=gate(f5,.6,.1)&gate(b5,.6,.1)&(f5[0]==-b5[0])
    votes=np.zeros((7,*base.shape),np.uint8)
    for oy in (-4,3):
        for ox in (-4,3):
            f=quadrant_scores(a,b,(oy,ox));r=quadrant_scores(b,a,(oy,ox));valid=gate(f,.8,.12)&gate(r,.8,.12)&(f[0]==-r[0])
            for k,d in enumerate(SHIFTS):votes[k]+=valid&(f[0]==d)
    candidate=votes.argmax(0);winner=np.take_along_axis(votes,candidate[None],0)[0];d=SHIFTS[candidate]
    rescue=(~base)&(winner>=2)&((votes>0).sum(0)==1)&valid5&(f5[0]==d)
    return [(f15[0],base),(d,rescue),(np.where(base,f15[0],d),base|rescue)]

def normal_test():
    z,y,x=np.indices((24,24,24),dtype=np.float64);coords=np.stack([z-11.5,y-11.5,x-11.5]);rng=np.random.default_rng(2700);rows=[]
    for n in ([1,0,0],[1,.3,.6],[.5,-1,1],[1,1,1]):
        n=np.array(n,dtype=float);n/=np.linalg.norm(n);d=np.einsum('i,ijkl->jkl',n,coords);mask=(np.abs(d)<2)&(np.max(np.abs(coords),axis=0)<7)
        for sigma in (0.,.02,.1):
            im=np.tanh(d/1.5)+rng.normal(0,sigma,d.shape);g=np.gradient(gaussian_filter(im,.7));J=np.empty((*im.shape,3,3))
            for i in range(3):
                for j in range(3):J[...,i,j]=gaussian_filter(g[i]*g[j],1.5)
            val,vec=np.linalg.eigh(J[mask]);ang=np.rad2deg(np.arccos(np.clip(np.abs(vec[:,:,-1]@n),0,1)));aniso=(val[:,-1]-val[:,-2])/np.maximum(val[:,-1],1e-12)
            rows.append(dict(normal=n.tolist(),noise=sigma,pixels=int(mask.sum()),median_angle_deg=float(np.median(ang)),p95_angle_deg=float(np.percentile(ang,95)),median_anisotropy=float(np.median(aniso))))
    assert max(r['p95_angle_deg'] for r in rows if not r['noise'])<5
    return rows

def test():
    rng=np.random.default_rng(27);a=rng.integers(20,160,(32,32,16),dtype=np.uint8);truth=np.full((32,32),2);b=simulate(a,truth);m=np.s_[8:-8,8:-8]
    for o in [(-4,-4),(-4,3),(3,-4),(3,3)]:assert (quadrant_scores(a,b,o)[0][m]==2).all()
    r=predict(a,b);assert (r[2][0][m]==2).all() and r[2][1][m].all();z=np.zeros_like(a);assert not predict(z,z)[2][1].any()
    return 6

def main():
    tic=time.perf_counter();p=Path(sys.argv[1])/'ct-tiles.npy';out=Path(sys.argv[2]);assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==PIN
    tiles=np.load(p,mmap_mode='r',allow_pickle=False);assert tiles.shape==(28,512,512,16);checks=test();normals=normal_test();yy,xx=np.indices((128,128));interior=(yy>=7)&(yy<121)&(xx>=7)&(xx<121);ramp=np.rint(-3+6*(xx+yy)/254).astype(int);step=np.where(xx<yy,-2,2);seam=np.abs(xx-yy)<8;occ=(yy>=58)&(yy<70)&(xx>=58)&(xx<70);rows=[]
    for i,tile in enumerate(tiles):
        a=np.asarray(tile[192:320,192:320]);mask=interior&(a[:,:,4:12].std(2)>=4);rng=np.random.default_rng(2000+i)
        def noisy(truth):
            b=simulate(a,truth);return np.clip(np.rint(1.15*b+8+rng.normal(0,3,b.shape)),0,255).astype(np.uint8)
        nr=noisy(ramp);ns=noisy(step);oc=ns.copy();oc[occ]=rng.integers(0,256,oc[occ].shape,dtype=np.uint8)
        for scene,b,truth,extra in [('noisy_ramp',nr,ramp,None),('noisy_step',ns,step,seam),('occluded_step',oc,step,occ),('decoy_xy',np.asarray(tile[:128,:128]),None,None),('decoy_tile',np.asarray(tiles[(i+1)%28,192:320,192:320]),None,None)]:
            r=predict(a,b);assert not (r[0][1]&r[1][1]).any()
            for method,(q,ok) in zip(('m15','rescue','combined'),r):rows.append(dict(tile=i,scene=scene,method=method,**count(q,ok,mask,truth),extra=count(q,ok,mask&extra,truth) if extra is not None else None))
    summ=[]
    for scene in ('noisy_ramp','noisy_step','occluded_step','decoy_xy','decoy_tile'):
        for method in ('m15','rescue','combined'):
            rr=[r for r in rows if r['scene']==scene and r['method']==method];keys=[k for k in rr[0] if k in ('attempted','admitted','exact','within1','abs_error')];summ.append(dict(scene=scene,method=method,**{k:sum(r[k] for r in rr) for k in keys},extra={k:sum(r['extra'][k] for r in rr) for k in keys} if rr[0]['extra'] is not None else None))
    idx={(r['scene'],r['method']):r for r in summ};a=idx['noisy_step','combined']['extra'];b=idx['noisy_step','m15']['extra'];seam_pp=100*(a['exact']-b['exact'])/a['attempted'];a=idx['noisy_ramp','combined'];precision=a['exact']/a['admitted'];decoy={s:100*(idx[s,'combined']['admitted']-idx[s,'m15']['admitted'])/idx[s,'m15']['attempted'] for s in ('decoy_xy','decoy_tile')};gates=dict(seam_delta_pp=seam_pp,ramp_exact_precision=precision,decoy_delta_pp=decoy,pass_=bool(seam_pp>=5 and precision>=.99 and all(v<=.25 for v in decoy.values())))
    result=dict(schema='b27.directional_support.v1',source_sha256=PIN,checks=checks,wall_seconds=time.perf_counter()-tic,rows=rows,summary=summ,gates=gates,normal_synthetic=normals,scope='Real CT texture, simulated shifts, synthetic planar normal truth; not actual sheet identity or mesh correction.')
    out.write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
