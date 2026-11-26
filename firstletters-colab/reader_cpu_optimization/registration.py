"""B17: controlled relative depth registration on actual CT profiles."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter

PIN='0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33'
SHIFTS=np.arange(-3,4)


def simulate(a,d):
    z=np.arange(16)[None,None,:]-d[:,:,None]
    b=np.take_along_axis(a,np.clip(z,0,15),axis=2)
    return np.where((z>=0)&(z<16),b,0)


def scores(a,b,size):
    def avg(v):return uniform_filter(v.mean(2),size=size,mode='nearest')
    a=a[:,:,4:12].astype(np.float64);ma=avg(a);aa=avg(a*a);va=np.maximum(aa-ma*ma,0)
    ssd=[];corr=[]
    for d in SHIFTS:
        x=b[:,:,4+d:12+d].astype(np.float64);mx=avg(x);xx=avg(x*x);ax=avg(a*x);vx=np.maximum(xx-mx*mx,0)
        ssd.append(avg((a-x)**2));den=np.sqrt(va*vx)
        q=np.full(den.shape,-np.inf);np.divide(ax-ma*mx,den,out=q,where=den>1e-8);corr.append(np.clip(q,-1,1,where=np.isfinite(q),out=q))
    return np.stack(ssd),np.stack(corr)


def evaluate(score,mask,truth,decoy,boundary,method):
    best_idx=score.argmax(0);best=np.take_along_axis(score,best_idx[None],0)[0];second=np.partition(score,-2,axis=0)[-2];pred=SHIFTS[best_idx]
    with np.errstate(invalid='ignore'):gap=best-second
    admitted=mask&np.isfinite(best)
    if method.startswith('corr_') and method!='corr_raw':admitted&=(best>=.6)&(gap>=float(method.split('_')[1]))
    def counts(m):
        n=int(m.sum());good=m&admitted;out=dict(attempted=n,accepted=int(good.sum()))
        if not decoy:
            err=np.abs(pred-truth);out.update(exact=int((good&(err==0)).sum()),within1=int((good&(err<=1)).sum()),sum_abs_error=int(err[good].sum()))
        return out
    return counts(mask),counts(mask&boundary)


def tests():
    r=np.random.default_rng(1);a=r.integers(30,140,(32,32,16),dtype=np.uint8);d=np.full((32,32),2);b=simulate(a,d)
    s,c=scores(a,b,5);m=np.ones((32,32),bool);m[:3]=m[-3:]=False;m[:,:3]=m[:,-3:]=False
    assert np.all(SHIFTS[s.argmin(0)][m]==2);assert np.all(SHIFTS[c.argmax(0)][m]==2)
    _,c=scores(a,1.2*b+7,5);assert np.all(SHIFTS[c.argmax(0)][m]==2)
    _,c=scores(np.ones_like(a),np.ones_like(a),5);assert np.all(~np.isfinite(c))
    return 4


def main():
    start=time.perf_counter();p=Path(sys.argv[1])/'ct-tiles.npy';out=Path(sys.argv[2]);assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==PIN
    tiles=np.load(p,mmap_mode='r',allow_pickle=False);assert tiles.shape==(28,512,512,16) and tiles.dtype==np.uint8
    ntest=tests();yy,xx=np.indices((128,128));interior=(yy>=7)&(yy<121)&(xx>=7)&(xx<121);ramp=np.rint(-3+6*(xx+yy)/254).astype(int);step=np.where(xx<64,-2,2);zero=np.zeros_like(xx);boundary=np.abs(xx-63.5)<7.5
    rows=[]
    for i,tile in enumerate(tiles):
        a=np.asarray(tile[192:320,192:320,:]);mask=interior&(a[:,:,4:12].std(2)>=4)
        rng=np.random.default_rng(1700+i);b=simulate(a,ramp);noisy=np.clip(np.rint(1.15*b+8+rng.normal(0,3,b.shape)),0,255).astype(np.uint8)
        for scene,truth,b in [('constant',zero+2,simulate(a,zero+2)),('ramp',ramp,b),('step',step,simulate(a,step)),('noisy_ramp',ramp,noisy),('decoy',zero,np.asarray(tile[:128,:128,:]))]:
            for size in (1,5,15):
                s,c=scores(a,b,size)
                for method,val in [('ssd',-s),('corr_raw',c),('corr_0',c),('corr_0.02',c),('corr_0.1',c)]:
                    v,w=evaluate(val,mask,truth,scene=='decoy',boundary,method);rows.append(dict(tile=i,scene=scene,size=size,method=method,interior=int(interior.sum()),**v,boundary=w))
    summary=[]
    for scene in ('constant','ramp','step','noisy_ramp','decoy'):
        for size in (1,5,15):
            for method in ('ssd','corr_raw','corr_0','corr_0.02','corr_0.1'):
                r=[v for v in rows if (v['scene'],v['size'],v['method'])==(scene,size,method)];keys=['interior','attempted','accepted']+([] if scene=='decoy' else ['exact','within1','sum_abs_error']);v={k:sum(t[k] for t in r) for k in keys};v['coverage']=v['accepted']/v['attempted'] if v['attempted'] else None
                if scene!='decoy':v.update(exact_rate=v['exact']/v['accepted'] if v['accepted'] else None,within1_rate=v['within1']/v['accepted'] if v['accepted'] else None,mae=v['sum_abs_error']/v['accepted'] if v['accepted'] else None)
                v['boundary']={k:sum(t['boundary'][k] for t in r) for k in r[0]['boundary']};summary.append(dict(scene=scene,size=size,method=method,**v))
    result=dict(schema='b17.relative_depth_registration.v1',source_sha256=PIN,tests=ntest,wall_seconds=round(time.perf_counter()-start,3),rows=rows,summary=summary,scope='Actual prepared CT, artificial axial shifts and decoy; not native geometric repair.')
    out.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n');print('tests',ntest,'rows',len(rows),'seconds',result['wall_seconds']);
    for r in summary:
        if r['method'] in ('ssd','corr_raw','corr_0.1'):print(r['scene'],r['size'],r['method'],'coverage',round(r['coverage'],4),'exact',round(r.get('exact_rate',0),4),'mae',round(r.get('mae',0),4))

if __name__=='__main__':main()
