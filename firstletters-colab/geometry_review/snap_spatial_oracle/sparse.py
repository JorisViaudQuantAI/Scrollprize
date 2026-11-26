#!/usr/bin/env python3
"""B12b: simulated sparse paired reference labels, held-out spatial stress."""
import argparse, json, time, hashlib, resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from audit import load, edges, solve

BUDGETS=(16,64,256);NOISE=(0,10,25);METHODS=('snap','global','nearest','idw8','cut')
COST=np.array([0,2,2,2,0,1,3])


def interp(points, anchor_idx, d):
    # cKDTree is used only on the sampled reference coordinates.
    dist,idx=cKDTree(points[anchor_idx]).query(points,k=8,workers=1)
    weights=1/np.maximum(dist,1e-12)**2
    pred=(d[idx]*weights).sum(axis=1)/weights.sum(axis=1)
    pred[anchor_idx]=d # exact interpolation at unique anchor coordinates
    return d[idx[:,0]],pred,dist[:,0]>2


def policies(points,ids,d,x,y):
    nn,est,held=interp(points,ids,d)
    q=np.rint(est*100).astype(np.int64)
    sel,_,_,_=solve(np.maximum(q,0),np.maximum(-q,0),x,y,100)
    return (np.ones(len(points),bool),np.full(len(points),d.mean()>=0),nn>=0,est>=0,sel),held


def evaluate(a,b,held,sel,x,y):
    c=np.where(sel,b,a)[held];counts=np.bincount(c,minlength=7)
    return [int(counts[4]),int(counts[6]),int(counts[1:4].sum()),int(COST[c].sum()),int((sel[x]!=sel[y]).sum())]


def test():
    rng=np.random.default_rng(443);p=np.column_stack(np.unravel_index(np.arange(100),(10,10)))
    ids=np.arange(0,100,10);d=np.full(len(ids),2)
    nn,est,held=interp(p,ids,d);assert np.all(nn==2) and np.allclose(est,2) and not held[ids].any()
    a=rng.integers(1,7,100);b=rng.integers(1,7,100);delta=COST[a]-COST[b];d=delta[ids].copy()
    _,est,_=interp(p,ids,d);assert np.array_equal(est[ids],d)
    x,y=edges(np.ones((10,10),bool));sel,held=policies(p,ids,d,x,y)
    delta2=delta.copy();delta2[held]=-delta2[held];sel2,held2=policies(p,ids,delta2[ids],x,y)
    assert np.array_equal(held,held2) and all(np.array_equal(s,t) for s,t in zip(sel,sel2))
    return 3


def run_winding(item):
    wi,w,a,b,m=item;p=np.column_stack(np.nonzero(m));a=a[m];b=b[m];x,y=edges(m);d=COST[a]-COST[b];rows=[]
    for seed in range(32):
        rng=np.random.default_rng(120670+wi*100+seed);ids=rng.choice(len(p),256,replace=False);corrupt_order=rng.permutation(256)
        for n in BUDGETS:
            anchors=ids[:n];order=corrupt_order[corrupt_order<n]
            for noise in NOISE:
                lab=d[anchors].copy();j=order[:n*noise//100];lab[j]*=-1
                sels,held=policies(p,anchors,lab,x,y);h=int(held.sum());assert h>0 and not held[anchors].any()
                scores=[evaluate(a,b,held,sel,x,y) for sel in sels]
                rows.append([w,seed,n,noise,h,int((lab!=d[anchors]).sum()),*scores])
    return rows


def summarize(rows):
    out=[]
    for w in sorted(set(r[0] for r in rows))+['pooled']:
        for n in BUDGETS:
            for noise in NOISE:
                rs=[r for r in rows if (w=='pooled' or r[0]==w) and r[2:4]==[n,noise]]
                for i,method in enumerate(METHODS[1:],1):
                    dd=np.array([[(r[6+i][k]-r[6][k])/r[4] for k in range(4)] for r in rs]);sw=np.array([r[6+i][4] for r in rs]);harm=int((dd[:,3]>0).sum());win=int((dd[:,3]<0).sum());worst=int(np.argmax(dd[:,3]));r=rs[worst]
                    out.append([w,n,noise,method,len(rs),*np.round(dd.mean(axis=0),8).tolist(),win,harm,int(np.count_nonzero(dd[:,3]==0)),round(float(sw.mean()),3),round(float(dd[worst,3]),8),r[0],r[1]])
    return out


def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--source');g.add_argument('--maps-dir');ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    t=time.perf_counter();tests=test();data=[(i,*d) for i,d in enumerate(load(args.source,args.maps_dir))]
    if args.workers==1: groups=list(map(run_winding,data))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:groups=list(pool.map(run_winding,data))
    rows=[r for g in groups for r in g];assert len(rows)==4*32*3*3
    summary=summarize(rows);out=dict(schema='b12b.sparse_reference_stress.v1',tests=tests,wall_seconds=round(time.perf_counter()-t,3),workers=args.workers,parent_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,methods=METHODS,score_columns=['near','far','unresolved','unary','switches'],row_prefix=['winding','seed','anchors','noise_pct','held','actually_changed'],rows=rows,summary_columns=['winding','anchors','noise_pct','method','runs','mean_delta_near','mean_delta_far','mean_delta_unresolved','mean_delta_unary','wins','harms','ties','mean_switches','worst_delta_unary','worst_winding','worst_seed'],summary=summary)
    Path(args.output).write_text(json.dumps(out,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},separators=(',',':')))

if __name__=='__main__':main()
