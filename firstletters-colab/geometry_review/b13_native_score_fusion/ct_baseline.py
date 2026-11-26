#!/usr/bin/env python3
"""B14: new lightweight CPU models on spatially separated exposed 0841 CT."""
import argparse, json, time, hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from replay import PINS, threshold, count, pack, cohort

CUT=(800,1696); GUARD=16; NAMES=('studio','logistic','histgb')


def quadrant(y,x):
    return 2*(y>=CUT[0]).astype(np.int8)+(x>=CUT[1]).astype(np.int8)


def prepare(data):
    for name,pin in PINS.items():
        with (data/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==pin,name
    labels=np.load(data/'labels.npy',allow_pickle=False)
    ct=np.load(data/'ct-tiles.npy',mmap_mode='r',allow_pickle=False)
    tiles=np.load(data/'tiles.npy',allow_pickle=False)
    coords=np.load(data/'coordinates.npy',allow_pickle=False)
    with np.load(data/'scale_stress.npz',allow_pickle=False) as z:
        ids=z['author_indices']; origins=z['core_origins_yx']; states=z['states']; studio=z['probabilities'][0].astype(np.float64)
    assert ct.shape==(28,512,512,16) and np.array_equal(origins,coords[ids]+48)
    assert np.array_equal(states,np.stack([labels[y:y+32,x:x+32] for y,x in origins])) and np.isin(labels,[0,1,2]).all()
    ys,xs=np.nonzero(labels);y0,x0=ys.min()-2,xs.min()-2;y1,x1=ys.max()+3,xs.max()+3
    h,w=y1-y0,x1-x0; mosaic=np.zeros((h,w,16),np.uint8);support=np.zeros((h,w),np.uint8)
    for tile,(ty,tx) in zip(ct,tiles):
        a,b=max(y0,ty),min(y1,ty+512);c,d=max(x0,tx),min(x1,tx+512)
        if a<b and c<d:
            mosaic[a-y0:b-y0,c-x0:d-x0]=tile[a-ty:b-ty,c-tx:d-tx];support[a-y0:b-y0,c-x0:d-x0]+=1
    yy,xx=ys-y0,xs-x0;assert np.all(minimum_filter((support==1).astype(np.uint8),size=5,mode='constant',cval=0)[yy,xx]==1)
    p=mosaic.astype(np.float32)/255;raw=p[yy,xx];local=uniform_filter(p,size=(5,5,1),mode='constant')[yy,xx]
    feature=np.concatenate([raw,local,raw-local,np.diff(raw,axis=1),raw.mean(axis=1,keepdims=True),raw.std(axis=1,keepdims=True)],axis=1)
    assert feature.shape==(len(ys),65) and np.isfinite(feature).all()
    truth=labels[ys,xs]==2;quads=quadrant(ys,xs);inside=(np.abs(ys-CUT[0])>=GUARD)&(np.abs(xs-CUT[1])>=GUARD)
    core_q=quadrant(origins[:,0]+16,origins[:,1]+16);core_indices=[];linear=ys.astype(np.int64)*labels.shape[1]+xs
    for k,(y,x) in enumerate(origins):
        r,c=np.meshgrid(np.arange(y,y+32),np.arange(x,x+32),indexing='ij')
        assert np.all(quadrant(r,c)==core_q[k]) and (np.abs(r-CUT[0])>=GUARD).all() and (np.abs(c-CUT[1])>=GUARD).all(),int(ids[k])
        known=states[k]>0;ix=np.searchsorted(linear,(r*labels.shape[1]+c)[known]);assert np.array_equal(linear[ix],(r*labels.shape[1]+c)[known])
        assert np.array_equal(truth[ix],states[k][known]==2);core_indices.append(ix)
    return dict(raw=raw,feature=feature,truth=truth,quads=quads,inside=inside,ids=ids,states=states,studio=studio,core_q=core_q,core_indices=core_indices,known_pixels=len(ys),ct_crop=[int(y0),int(y1),int(x0),int(x1)],known_support=int((support[yy,xx]==1).sum()))


def ap(v,y):
    return float(average_precision_score(y,v)) if y.any() and not y.all() else None


def summarize(rows):
    out={}
    for group,rr in [('all',rows),('mixed',[r for r in rows if r['mixed']])]:
        out[group]={}
        for m in NAMES:
            a=[r['metrics'][m] for r in rr];v=[sum(r[k] for r in a) for k in ('ink','bg','tp','fp')];aa=[r['ap'] for r in a if r['ap'] is not None]
            out[group][m]=dict(pack(v),cores=len(rr),macro_ap=float(np.mean(aa)) if aa else None,ap_cores=len(aa))
    return out


def run_fold(job):
    q,d=job;cal=(q+1)%4;test=np.flatnonzero(d['core_q']==q);cal_ids=np.flatnonzero(d['core_q']==cal)
    train=d['inside']&~np.isin(d['quads'],[q,cal]);train_idx=np.flatnonzero(train);y=d['truth'];s=d['states'];studio=d['studio'];tic=time.perf_counter();cpu=time.process_time()
    for i in np.r_[test,cal_ids]: assert not train[d['core_indices'][i]].any()
    assert (d['quads'][train_idx]!=q).all() and (d['quads'][train_idx]!=cal).all() and y[train_idx].any() and (~y[train_idx]).any()
    neg=s[cal_ids]==1;budget=int((studio[cal_ids][neg]>=.5).sum());assert neg.sum()>=512
    predictions={};times={};ths={'studio':.5};cal_fp={'studio':budget};out=np.full((2,len(test),32,32),np.nan,np.float64)
    with threadpool_limits(limits=1):
        specs=[('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500)),d['raw']),('histgb',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1,early_stopping=False,random_state=130),d['feature'])]
        for mi,(name,model,f) in enumerate(specs):
            t=time.perf_counter();model.fit(f[train_idx],y[train_idx]);times[name]=round(time.perf_counter()-t,3)
            calneg=np.concatenate([model.predict_proba(f[d['core_indices'][i]])[:,1][s[i][s[i]>0]==1] for i in cal_ids if (s[i]==1).any()]);assert len(calneg)==neg.sum()
            ths[name]=threshold(calneg,budget);cal_fp[name]=int((calneg>=ths[name]).sum());assert cal_fp[name]<=budget
            preds=[]
            for j,i in enumerate(test):
                v=model.predict_proba(f[d['core_indices'][i]])[:,1];assert np.isfinite(v).all();preds.append(v);out[mi,j][s[i]>0]=v
            predictions[name]=preds
    mixed,_=cohort(s);rows=[]
    for j,i in enumerate(test):
        label=s[i][s[i]>0];r={'id':int(d['ids'][i]),'mixed':bool(i in mixed),'metrics':{}}
        vals={'studio':studio[i][s[i]>0],**{m:predictions[m][j] for m in predictions}}
        for name,v in vals.items():r['metrics'][name]=dict(pack(count(v>=ths[name],label)),ap=ap(v,label==2))
        rows.append(r)
    fold={'test_q':q,'calibration_q':cal,'train_pixels':int(train.sum()),'train_ink':int(y[train].sum()),'train_bg':int((~y[train]).sum()),'calibration_bg':int(neg.sum()),'calibration_fp_budget':budget,'calibration_fp':cal_fp,'thresholds':ths,'fits_wall_seconds':times,'wall_seconds':round(time.perf_counter()-tic,3),'cpu_seconds':round(time.process_time()-cpu,3),'rows':rows,'summary':summarize(rows)}
    return fold,test,out


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=4);args=p.parse_args();t=time.perf_counter();d=prepare(args.data)
    if args.workers==1:results=list(map(run_fold,[(q,d) for q in range(4)]))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run_fold,[(q,d) for q in range(4)]))
    folds=[];out=np.full((2,53,32,32),np.nan,np.float64);seen=[]
    for fold,idx,pred in results:folds.append(fold);out[:,idx]=pred;seen.extend(idx.tolist())
    assert sorted(seen)==list(range(53)) and np.array_equal(np.isfinite(out),np.broadcast_to(d['states']>0,out.shape))
    rows=[r for f in folds for r in f['rows']];summary=summarize(rows);receipt=dict(schema='b14.ct_baseline.v1',cuts=CUT,guard=GUARD,known_pixels=d['known_pixels'],known_support=d['known_support'],ct_crop=d['ct_crop'],features=65,workers=args.workers,wall_seconds=round(time.perf_counter()-t,3),cpu_seconds=sum(f['cpu_seconds'] for f in folds),summary=summary,folds=folds,scope='New light model fits, one exposed donor, no independent scroll, no Studio checkpoint inference or deployable claim.')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'ct_results.json').write_text(json.dumps(receipt,separators=(',',':'),allow_nan=False)+'\n');np.savez_compressed(args.output/'ct_predictions.npz',probabilities=out,states=d['states'],author_indices=d['ids'],model_names=np.array(NAMES[1:]),core_q=d['core_q']);print(json.dumps({k:v for k,v in receipt.items() if k!='folds'},indent=2))

if __name__=='__main__':main()
