"""B34: held-whole-fibre surface and normal validation, public 0846A traces."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay,cKDTree
from scipy.linalg import lstsq
PLY=9.;SPACING=12.;SCALE=.05;METHODS=('affine','tps','robust_tps')
def gitblob(p):
    v=p.read_bytes();return hashlib.sha1(b'blob '+str(len(v)).encode()+b'\0'+v).hexdigest()
def load(d):
    tree={Path(l.split('\t')[1]).name:l.split()[2] for l in (d/'tree.txt').read_text().splitlines()}
    assert tree['layers.json']=='4508f845cc4d396c07fc882d879773387a2439d0' and tree['crossings.csv']=='b043287a6532929a833aaeddf7e12d31077b7879'
    for n in ['layers.json','crossings.csv']+[p.name for p in d.glob('*.path.npy')]:assert gitblob(d/n)==tree[n],n
    layers=json.loads((d/'layers.json').read_text());assert layers['spacing_vx']==SPACING and layers['ply_offset_vx']==PLY
    paths={p.name[:-9]:np.load(p,allow_pickle=False).astype(float) for p in sorted(d.glob('*.path.npy'))}
    assert len(paths)==116 and set(paths)==set(layers['layers']['0']) and all(v==0 for v in layers['fibre_layer'].values())
    for p in paths.values():assert p.ndim==2 and p.shape[1]==5 and np.isfinite(p).all() and (np.diff(p[:,4])>=0).all() and ((p[:,3]>=0)&(p[:,3]<=1)).all()
    cross=list(csv.DictReader((d/'crossings.csv').open()));return paths,cross,tree

def samples(a,step,cap):
    arc=a[:,4];s=np.arange(50,arc[-1]-50,step)
    if len(s)>cap:s=s[np.rint(np.linspace(0,len(s)-1,cap)).astype(int)]
    p=np.stack([np.interp(s,arc,a[:,j]) for j in range(3)],1);return p[np.interp(s,arc,a[:,3])>=.5]
def fit(p,fam,method,extra=None):
    x=p[:,[0,2]];y=p[:,1]
    if method=='affine':
        origin=x.mean(0);coef=lstsq(np.column_stack([np.ones(len(x)),(x-origin)*SCALE]),y)[0]
        return lambda z:coef[0]+(np.asarray(z)-origin)@coef[1:]*SCALE
    smooth=np.where(fam=='h',5.,50.)
    if extra is not None:smooth*=extra
    model=RBFInterpolator(x*SCALE,y,kernel='thin_plate_spline',smoothing=smooth)
    return lambda z:model(np.asarray(z)*SCALE)
def normal(f,x,h=2.):
    fx=(f(x+[h,0])-f(x-[h,0]))/(2*h);fz=(f(x+[0,h])-f(x-[0,h]))/(2*h);n=np.column_stack([fx,-np.ones(len(x)),fz]);return n/np.linalg.norm(n,axis=1)[:,None]
def signed(f,p):
    x=p[:,[0,2]];return (p[:,1]-f(x))*normal(f,x)[:,1]
def train(p,fam,method):
    f=fit(p,fam,method);side=1.
    for it in range(3):
        d=signed(f,p);side=float(np.sign(np.median(d[fam=='h'])-np.median(d[fam=='v'])) or 1.)
        shift=np.where(fam=='h',0.,side*PLY);target=p+shift[:,None]*normal(f,p[:,[0,2]])
        f=fit(target,fam,method)
        if method=='robust_tps':
            for _ in range(2):
                err=np.abs(signed(f,target));factor=np.clip((err/3.)**2,1.,25.);f=fit(target,fam,method,factor)
    return f,side

def tangent(a,at):
    k=np.argmin(np.linalg.norm(a[:,:3]-at,axis=1));s=a[k,4];lo,hi=s-40,s+40
    if lo<a[0,4] or hi>a[-1,4] or np.linalg.norm(a[k,:3]-at)>100:return None
    v=np.array([np.interp(hi,a[:,4],a[:,i])-np.interp(lo,a[:,4],a[:,i]) for i in range(3)]);r=np.linalg.norm(v)
    return v/r if r>1e-8 else None

def cross_eval(paths,cross,held):
    pts=[];norms=[]
    for r in cross:
        h,v=r['horizontal'],r['vertical']
        if h not in held or v not in held or r['verdict']=='no overlap' or abs(float(r['normal_offset_vx'])-PLY)>=SPACING/2:continue
        at=np.fromstring(r['at'].strip('[]'),sep=' ');a,b=tangent(paths[h],at),tangent(paths[v],at)
        if a is None or b is None:continue
        n=np.cross(a,b);l=np.linalg.norm(n)
        if l<.5:continue
        pts.append(at[[0,2]]);norms.append(n/l)
    return np.asarray(pts).reshape(-1,2),np.asarray(norms).reshape(-1,3)
def stats(e):
    return dict(n=len(e),mae=float(np.mean(e)) if len(e) else None,median=float(np.median(e)) if len(e) else None,p95=float(np.quantile(e,.95)) if len(e) else None,near=int((e<=3).sum()),far=int((e>=6).sum()))
def test():
    x,z=np.meshgrid(np.arange(6)*100.,np.arange(6)*100.);p=np.column_stack([x.ravel(),(2+x*.1+z*.2).ravel(),z.ravel()]);f=fit(p,np.array(['h']*len(p)),'affine');assert np.max(np.abs(signed(f,p)))<1e-9
    n=normal(f,p[:,[0,2]]);assert np.allclose(n,np.array([.1,-1.,.2])/np.linalg.norm([.1,-1.,.2]));assert len(samples(np.column_stack([np.arange(200),np.zeros((200,2)),np.ones(200),np.arange(200)]),20,16))==5
    return 3

def run(d):
    t=time.perf_counter();paths,cross,tree=load(d);tests=test();tr={n:samples(a,80,16) for n,a in paths.items()};te={n:samples(a,20,256) for n,a in paths.items()};rows=[];per=[];pool={};angles={}
    for split in ('interleave','spatial'):
        foldsets=[set() for _ in range(5)]
        for axis in ('h','v'):
            names=[n for n in paths if n[0]==axis];names=sorted(names) if split=='interleave' else sorted(names,key=lambda n:(np.median(paths[n][:,2 if axis=='h' else 0]),n))
            for i,fold in enumerate([names[j::5] for j in range(5)] if split=='interleave' else np.array_split(names,5)):foldsets[i].update(fold)
        assert sum(map(len,foldsets))==116 and len(set.union(*foldsets))==116
        for j,held in enumerate(foldsets):
            trainnames=[n for n in paths if n not in held];pn=np.concatenate([tr[n] for n in trainnames]);fam=np.concatenate([np.repeat(n[0],len(tr[n])) for n in trainnames]);names=sorted(held);pt=np.concatenate([te[n] for n in names]);fn=np.concatenate([np.repeat(n,len(te[n])) for n in names]);family=np.array([n[0] for n in fn]);xz=pt[:,[0,2]];hull=Delaunay(pn[:,[0,2]]);tree0=cKDTree(pn[:,[0,2]]);support=(hull.find_simplex(xz)>=0)&(tree0.query(xz)[0]<=400);cx,cn=cross_eval(paths,cross,held);cs=(hull.find_simplex(cx)>=0)&(tree0.query(cx)[0]<=400) if len(cx) else np.zeros(0,bool)
            for method in METHODS:
                f,side=train(pn,fam,method);adjusted=pt+np.where(family=='h',0.,side*PLY)[:,None]*normal(f,xz);err=np.abs(signed(f,adjusted));assert np.isfinite(err).all();ae=np.rad2deg(np.arccos(np.clip(np.abs((normal(f,cx[cs])*cn[cs]).sum(1)),0,1))) if cs.any() else np.array([])
                row=dict(split=split,fold=j,method=method,train_fibres=len(trainnames),held_fibres=len(held),train_samples=len(pn),attempted=len(pt),support=int(support.sum()),side=side,**stats(err[support]),all_points=stats(err),crossings=int(cs.sum()),cross_median_deg=float(np.median(ae)) if len(ae) else None,cross_p95_deg=float(np.quantile(ae,.95)) if len(ae) else None);rows.append(row);pool.setdefault((split,method),[]).append(err[support]);angles.setdefault((split,method),[]).append(ae)
                for n in names:
                    m=(fn==n)&support;per.append(dict(split=split,fold=j,method=method,family=n[0],fiber=n,attempted=int((fn==n).sum()),**stats(err[m])))
            print('fold',split,j,'support',int(support.sum()),len(pt),flush=True)
    summary=[]
    for (split,method),v in pool.items():
        err=np.concatenate(v);a=np.concatenate(angles[split,method]);r=[x for x in rows if x['split']==split and x['method']==method];f=[x for x in per if x['split']==split and x['method']==method and x['n']];summary.append(dict(split=split,method=method,attempted=sum(x['attempted'] for x in r),**stats(err),fiber_macro_mae=float(np.mean([x['mae'] for x in f])),worst_fiber_mae=max(x['mae'] for x in f),crossings=len(a),cross_median_deg=float(np.median(a)) if len(a) else None,cross_p95_deg=float(np.quantile(a,.95)) if len(a) else None))
    by={(x['split'],x['method']):x for x in summary};gates={}
    for split in ('interleave','spatial'):
        a,b=by[split,'tps'],by[split,'robust_tps'];gates[split]=dict(p95_gain=a['p95']-b['p95'],far_delta_pp=100*(b['far']/b['n']-a['far']/a['n']),coverage=b['n']/b['attempted'],pass_=bool(a['p95']-b['p95']>=.5 and b['far']<=a['far'] and b['n']/b['attempted']>=.9))
    return dict(schema='b34.fibre_holdout.v1',source_commit='e1f05571b095c41dded6121a784b794ba308c510',verified_files=118,tests=tests,rows=rows,per_fiber=per,summary=summary,gates=gates,wall_seconds=time.perf_counter()-t,scope='Curated winner traces, known layer and ply; held whole fibres, not independent anatomical truth or sheet identity.')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('out',type=Path);v=a.parse_args();r=run(v.data);v.out.write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');print(json.dumps({k:x for k,x in r.items() if k not in ('rows','per_fiber')},indent=2))
