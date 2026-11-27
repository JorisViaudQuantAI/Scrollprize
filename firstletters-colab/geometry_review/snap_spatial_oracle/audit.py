#!/usr/bin/env python3
"""B12: exact reference-derived fitted/snap spatial-selection ceiling."""
import argparse, base64, hashlib, itertools, json, time, zlib, resource
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from scipy.ndimage import label

SOURCE_BLOB='2e4a8b220b6f9c090d8cb55fa9cd0c6e6ea413ef'
PINS={
'w033':(158,'722b3d45101bd581ccb8b8300f9279ebebc0a32e389c68aab46e7fac00aa7488','79ae96d0bbec9951229fc37f47193b34c6bd521c0a27bb49ecfa07ab6f97c5ab',11735),
'w034':(163,'8b5bb6a8ffa6006d04981c32cd3e7f9835eec02e1a604f385cf1a5d34d2cb48b','d5ddb444e20f68d53b5bb7b2ebd19de34d4a718045e45655688464629165bae3',12083),
'w035':(167,'2a06faddf98bf0d755c73451f924824b22d559fe304498dbdb8aa1bd12066565','da766bb875b7a5dc44637e16894820b67467e5a078a323c67a67d17444911189',12360),
'w036':(172,'a9ab7225f7aaeb17e98be8ce6a3415a9ceadcd757fce615f8013c0ed200bbe8c','0b728d6735bc64093f5c1f34ac9b93e264f2d76fbdb66e92c3250ef4813dcd6f',12731)}
TRANS={
'w033':[[1256,0,0,0,0,15],[0,4564,1,38,5,5],[4,3,45,1,0,3],[0,81,0,1163,39,6],[0,93,3,194,205,100],[159,96,6,6,138,3506]],
'w034':[[898,3,0,0,1,39],[1,5509,13,178,90,34],[2,4,60,1,0,6],[0,1,1,586,281,63],[0,1,2,996,100,104],[45,1,12,992,649,1410]],
'w035':[[1143,0,1,0,0,59],[0,5386,1,25,3,4],[2,1,58,1,1,9],[0,72,2,2577,175,8],[0,15,0,239,726,102],[13,7,6,26,95,1603]],
'w036':[[2009,0,6,0,0,13],[1,2011,3,2,24,910],[7,3,54,1,1,4],[0,23,5,410,432,259],[0,115,3,1111,121,567],[292,392,41,761,477,2673]]}
LAM=(0,1,2,4,8,16,32,64,128)


def decode(s,width,sha):
    b=zlib.decompress(base64.b64decode(s.strip(),validate=True))
    if hashlib.sha256(b).hexdigest()!=sha or len(b)!=91*width:
        raise ValueError('source byte pin mismatch')
    a=np.frombuffer(b,np.uint8).reshape(91,width)
    if not np.all(a<=6): raise ValueError('invalid code')
    return a


def load(source,maps_dir):
    if source:
        b=Path(source).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=SOURCE_BLOB:
            raise ValueError('source Git blob mismatch')
        j=json.loads(b)
    for w,(width,hf,hs,n) in PINS.items():
        arms=[]
        for state,sha in [('fitted',hf),('snap',hs)]:
            s=j['states'][state]['windings'][w]['reciprocal_code_map']['data'] if source else (Path(maps_dir)/f'{w}_{state}.b64').read_text()
            arms.append(decode(s,width,sha))
        a,b=arms;mask=(a!=0)&(b!=0)
        tab=np.bincount(a[mask].astype(int)*7+b[mask],minlength=49).reshape(7,7)
        if mask.sum()!=n or not np.array_equal(tab[1:,1:],TRANS[w]) or tab[0,:].sum() or tab[:,0].sum():
            raise ValueError('source support or transition mismatch')
        yield w,a,b,mask


def edges(mask):
    ids=np.full(mask.shape,-1,dtype=np.int64);ids[mask]=np.arange(mask.sum())
    x=np.concatenate([ids[:,:-1][mask[:,:-1]&mask[:,1:]],ids[:-1,:][mask[:-1,:]&mask[1:,:]]])
    y=np.concatenate([ids[:,1:][mask[:,:-1]&mask[:,1:]],ids[1:,:][mask[:-1,:]&mask[1:,:]]])
    return x,y


def energy(c0,c1,x,y,sel,lam):
    unary=int(np.where(sel,c1,c0).sum());sw=int((sel[x]!=sel[y]).sum())
    return unary+lam*sw,unary,sw


def solve(c0,c1,x,y,lam):
    # Source-side label fitted (0): cut i->sink costs c0. Sink-side label snap (1): cut source->i costs c1.
    n=len(c0);idx=np.arange(n,dtype=np.int64);source=n;sink=n+1
    r=np.concatenate([np.full(n,source),idx,x,y]);c=np.concatenate([idx,np.full(n,sink),y,x])
    v=np.concatenate([c1,c0,np.full(len(x)*2,lam)]).astype(np.int64)
    cap=coo_matrix((v,(r,c)),shape=(n+2,n+2)).tocsr();cap.eliminate_zeros()
    f=maximum_flow(cap,source,sink,method='dinic')
    residual=cap-f.flow
    if residual.data.min(initial=0)<0: raise AssertionError('negative residual')
    residual.eliminate_zeros();reach=breadth_first_order(residual,source,directed=True,return_predecessors=False)
    if sink in reach: raise AssertionError('augmenting path remains')
    sel=np.ones(n,bool);sel[reach[reach<n]]=False
    e,u,s=energy(c0,c1,x,y,sel,lam)
    if e!=f.flow_value: raise AssertionError('cut/flow mismatch')
    return sel,e,u,s


def stats(a,b,mask,sel,c0,c1,x,y,lam):
    selected=np.where(sel,b[mask],a[mask]);counts=np.bincount(selected,minlength=7)
    e,u,s=energy(c0,c1,x,y,sel,lam)
    grid=np.zeros(mask.shape,np.uint8);grid[mask]=1+sel.astype(np.uint8)
    comps=[]
    for k in (1,2):
        lab,n=label(grid==k);size=np.bincount(lab.ravel())[1:];comps.append([int(n),int(size.max(initial=0))])
    return dict(near=int(counts[4]),intermediate=int(counts[5]),far=int(counts[6]),unresolved=int(counts[1:4].sum()),snap=int(sel.sum()),switches=s,unary=u,objective=e,components=comps,selection_sha256=hashlib.sha256(grid.tobytes()).hexdigest())


def test():
    rng=np.random.default_rng(12067);checks=0
    for i in range(128):
        mask=rng.random((3,3))>.2
        if not mask.any(): mask[0,0]=True
        n=int(mask.sum());x,y=edges(mask);c0=rng.integers(0,4,n);c1=rng.integers(0,4,n)
        candidates=np.array(list(itertools.product([False,True],repeat=n)))
        # Literal grid-neighbour traversal, independent of edges() and energy().
        pts=list(zip(*np.nonzero(mask)));indices={p:j for j,p in enumerate(pts)}
        pairs=[(j,indices[q]) for j,(r,c) in enumerate(pts) for q in [(r+1,c),(r,c+1)] if q in indices]
        assert sorted(zip(x.tolist(),y.tolist()))==sorted(pairs)
        for lam in (0,1,2,7):
            _,e,_,_=solve(c0,c1,x,y,lam)
            un=np.where(candidates,c1,c0).sum(axis=1)
            sw=sum((candidates[:,i]!=candidates[:,j]).astype(int) for i,j in pairs)
            assert e==int(np.min(un+lam*sw));checks+=1
    return checks


def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--source');g.add_argument('--maps-dir');ap.add_argument('--output',required=True);args=ap.parse_args()
    started=time.perf_counter();checks=test();data=list(load(args.source,args.maps_dir));rows=[]
    for w,a,b,m in data:
        x,y=edges(m);n=int(m.sum())
        for uc in (2,3):
            costs=np.array([0,uc,uc,uc,0,1,3]);c0=costs[a[m]];c1=costs[b[m]]
            base={k:stats(a,b,m,sel,c0,c1,x,y,0) for k,sel in [('fitted',np.zeros(n,bool)),('snap',np.ones(n,bool))]}
            prev=None
            for lam in LAM:
                sel,e,u,s=solve(c0,c1,x,y,lam);d=stats(a,b,m,sel,c0,c1,x,y,lam)
                assert (e,u,s)==(d['objective'],d['unary'],d['switches'])
                assert e<=min(v['unary'] for v in base.values())
                if prev is not None: assert s<=prev['switches'] and u>=prev['unary']
                prev=d;rows.append(dict(winding=w,unresolved_cost=uc,lambda_=lam,vertices=n,edges=len(x),**d))
            rows.append(dict(winding=w,unresolved_cost=uc,vertices=n,edges=len(x),baseline=base))
    pooled=[]
    keys=('vertices','edges','near','intermediate','far','unresolved','snap','switches','unary','objective')
    for uc in (2,3):
        rr=[r for r in rows if r['unresolved_cost']==uc]
        for lam in LAM:
            rs=[r for r in rr if r.get('lambda_')==lam];pooled.append(dict(unresolved_cost=uc,lambda_=lam,**{k:sum(r[k] for r in rs) for k in keys}))
        for arm in ('fitted','snap'):
            rs=[r for r in rr if 'baseline' in r];pooled.append(dict(unresolved_cost=uc,baseline=arm,**{k:sum((r[k] if k in ('vertices','edges') else r['baseline'][arm][k]) for r in rs) for k in keys}))
    result=dict(schema='b12.snap_spatial_oracle.v1',source_blob=SOURCE_BLOB,execution=dict(python=__import__('sys').version.split()[0],numpy=np.__version__,scipy=scipy.__version__,wall_seconds=round(time.perf_counter()-started,3),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,small_grid_exhaustive=checks,source_maps_verified=8,transition_tables_verified=4,real_min_cuts=72),pooled=pooled,rows=rows,scope='Public 0211 reference oracle only. Vertex-index counts, not physical area; no CT, new labels, mesh welding or deployment.')
    Path(args.output).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(dict(execution=result['execution'],pooled=pooled),indent=2))

if __name__=='__main__':main()
