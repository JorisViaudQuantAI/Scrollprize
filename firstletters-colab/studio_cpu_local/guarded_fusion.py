#!/usr/bin/env python3
"""B9 follow-up: real archived predictions, spatially separated calibration."""
import argparse, importlib.util, hashlib, json, time
from pathlib import Path
import numpy as np
from train_ct import PINS

B9_BLOB='f7212ef5430c718799c8f72926e3932d91ca18a0'
METHODS=('baseline','max','mean','zmean','product')

def b9_module(path):
    b=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==B9_BLOB
    spec=importlib.util.spec_from_file_location('b9',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fit_scores(q,arms,base,train,valid,states,b9):
    fit=q[arms][:,train][:,valid[train]];mu=fit.mean(axis=1);sd=fit.std(axis=1)
    assert (sd>=1e-8).all()
    values=[q[base],q[arms].max(axis=0),q[arms].mean(axis=0),((q[arms]-mu[:,None,None,None])/sd[:,None,None,None]).mean(axis=0),q[arms].prod(axis=0)]
    neg=states[train]==1;budget=int((q[base,train][neg]>=.5).sum())
    assert neg.any() and (states[train]==2).any()
    ts=[.5]+[b9.threshold(v[train][neg],budget) for v in values[1:]]
    assert all(int((v[train][neg]>=t).sum())<=budget for v,t in zip(values,ts))
    return values,ts,int(neg.sum()),budget

def main(root,output,b9_path):
    start=time.perf_counter();b9=b9_module(b9_path)
    for name in ('pose_factors.npz','coordinates.npy'):
        assert sha(root/name)==PINS[name]
    z=np.load(root/'pose_factors.npz',allow_pickle=False);c=np.load(root/'coordinates.npy',allow_pickle=False)
    p=z['probabilities'].astype(np.float64);s=z['states'];orig=z['original_states'];valid=z['common_masks'];ids=z['author_indices'];xy=c[ids]
    assert z['block_names'].tolist()==b9.BLOCKS and z['families'].tolist()==b9.FAMILIES and ids.tolist()==b9.IDS
    assert np.array_equal(s,np.where(valid,orig[None],0)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    ink=(orig==2).sum((1,2));bg=(orig==1).sum((1,2));known=ink+bg;mixed=np.flatnonzero((known>=512)&(10*ink>=known)&(10*ink<=9*known));assert len(mixed)==11
    D=np.max(np.abs(xy[:,None,:]-xy[None,:,:]),axis=2)
    rows=[];raw=[];checks=0;leak_checks=0
    for bi,block in enumerate(b9.BLOCKS):
      q=p[bi];state=s[bi];v=valid[bi]
      for base,arms in [(0,[0,1,2,3]),(3,[3,1,2])]:
        # Verify actual OR novelty vs the exact sharp cardinality bounds, separately for both labels.
        raw_row={'block':block,'base':b9.FAMILIES[base]}
        for cls,name in [(2,'ink'),(1,'background')]:
            lo=hi=actual=N=0
            for held in mixed:
                m=state[held]==cls;n=int(m.sum());b=int((q[base,held][m]>=.5).sum());counts=[int((q[a,held][m]>=.5).sum()) for a in arms if a!=base]
                x=int(((q[arms,held][:,m]>=.5).any(axis=0)&(q[base,held][m]<.5)).sum());L=max([0]+[k-b for k in counts]);H=min(n-b,sum(counts))
                assert L<=x<=H;checks+=1;lo+=L;hi+=H;actual+=x;N+=n
            raw_row[name]=[N,lo,actual,hi]
        raw.append(raw_row)
        for guard in (128,192):
          pred=np.zeros((len(METHODS),*state.shape),bool);folds=[]
          for held in mixed:
            train=mixed[D[held,mixed]>=guard];assert len(train)>=3 and held not in train
            scores,ts,nneg,budget=fit_scores(q,arms,base,train,v,state,b9)
            # One independent no-leak perturbation per block/baseline/guard.
            if held==mixed[0]:
                perturbed=q.copy();not_train=np.setdiff1d(np.arange(len(ids)),train);perturbed[:,not_train]=1-perturbed[:,not_train]
                _,t2,n2,b2=fit_scores(perturbed,arms,base,train,v,state,b9)
                assert ts==t2 and nneg==n2 and budget==b2;leak_checks+=1
            for k,(score,t) in enumerate(zip(scores,ts)):pred[k,held]=score[held]>=t
            folds.append(dict(held=int(ids[held]),train_ids=ids[train].tolist(),calib_bg=nneg,calib_fp=budget,thresholds=ts,counts={m:b9.counts(a[held],state[held]) for m,a in zip(METHODS,pred)}))
          agg={m:b9.counts(a[mixed],state[mixed]) for m,a in zip(METHODS,pred)}
          rows.append(dict(block=block,base=b9.FAMILIES[base],guard=guard,held_out=agg,folds=folds))
    assert checks==352 and leak_checks==32 and len(rows)==32
    d=dict(schema='b9.guarded_fusion.v1',source_sha256={k:PINS[k] for k in ('pose_factors.npz','coordinates.npy')},b9_blob=B9_BLOB,wall_seconds=round(time.perf_counter()-start,3),bound_checks=checks,leak_checks=leak_checks,folds=sum(len(r['folds']) for r in rows),raw_union=raw,rows=rows,scope='Same exposed donor; prepared-chart origin guard not native physical separation; oracle geometry hybrids, no deployment.')
    output.write_text(json.dumps(d,allow_nan=False,indent=2)+'\n')
    for guard in (128,192):
        for base in ('identity','rigid'):
            rs=[r for r in rows if r['guard']==guard and r['base']==base]
            print(guard,base,[(m,sum(r['held_out'][m]['tp']>r['held_out']['baseline']['tp'] for r in rs),sum(r['held_out'][m]['fp']<=r['held_out']['baseline']['fp'] for r in rs)) for m in METHODS[1:]])
    print('PASS',d['folds'],checks,leak_checks,d['wall_seconds'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--b9-script',type=Path,required=True);args=a.parse_args();main(args.input,args.output,args.b9_script)
