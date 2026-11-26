"""B25: depth-score combinations fitted only on fixed B21 development crops."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from cascade import PRED,sha,threshold,metric

NAMES=('D','DO')

def features(c,m,p,opp,name):
    stack=np.stack([c,m,p]);v=np.median(stack,axis=0);lo=stack.min(0);hi=stack.max(0)
    q=[c,v,lo,hi,hi-lo]
    if name=='DO':q.extend([opp,c-opp])
    elif name!='D':raise ValueError(name)
    return np.stack(q,axis=-1).astype(np.float64)

def fit(X,s,ids):
    known=s[ids]>0;truth=s[ids][known]==2;data=X[ids][known];n=known.sum((1,2));assert (n>0).all() and truth.any() and (~truth).any()
    weights=np.broadcast_to((known.sum()/len(ids)/n)[:,None,None],known.shape)[known]
    sc=StandardScaler().fit(data,sample_weight=weights);model=LogisticRegression(C=1,max_iter=500).fit(sc.transform(data),truth,sample_weight=weights)
    assert model.n_iter_.max()<500
    return dict(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]),train_ids=ids.tolist(),n_known=int(known.sum()))

def score(X,m):
    return expit(((X-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coef'])+m['intercept'])

def crossfit(X,s,source):
    oof=np.zeros(s[source].shape,np.float64);folds=[]
    for j,i in enumerate(source):
        train=source[source!=i];assert i not in train
        m=fit(X,s,train);oof[j]=score(X[i],m);folds.append(dict(held=int(i),train=m['train_ids']))
    t=threshold(oof,s[source]);model=fit(X,s,source);full=score(X[source],model)
    model.update(threshold=t,source_ids=source.tolist(),folds=folds,oof_recall=float((oof[s[source]==2]>=t).mean()),refit_recall=float((full[s[source]==2]>=t).mean()),oof_ap=float(average_precision_score(s[source][s[source]>0]==2,oof[s[source]>0])))
    return model,oof

def test():
    a=np.array([.3,.8]);b=np.array([.1,.5]);c=np.array([.2,.6]);o=np.array([.6,.1]);q=features(a,b,c,o,'DO');assert np.allclose(q[0],[.3,.2,.1,.3,.2,.6,-.3]);assert np.array_equal(features(a,b,c,o,'D'),q[:,:5]);assert np.array_equal(features(a,c,b,o,'DO'),q)
    assert threshold(np.array([.5,.5,.5,.1,.1]),np.full(5,2))==.1
    rng=np.random.default_rng(25);X=rng.random((3,4,4,5));s=np.ones((3,4,4),np.uint8);s[:,:2]=2;ids=np.array([0,1]);m,_=crossfit(X,s,ids);s[2]=0;X[2]=rng.random((4,4,5));n,_=crossfit(X,s,ids);assert m==n
    return 3

def run(d):
    t=time.perf_counter();checks=test();assert sha(d/'predictions.npz')==PRED
    with np.load(d/'predictions.npz',allow_pickle=False) as z:p,s,w,blank=[z[k] for k in ('probabilities','states','windings','blank')];assert z['arms'].tolist()==['reverse','forward','minus2','plus2','bf16']
    center,opp,minus,plus,bf=p;med=np.median(p[[0,2,3]],axis=0);rows=[];models={};fit_count=0
    for name in NAMES:
        X=features(center,minus,plus,opp,name);models[name]={}
        for target in ('w058','w060','future0841'):
            tr=np.flatnonzero(w!=target);te=np.flatnonzero(w==target);m,oof=crossfit(X,s,tr);fit_count+=len(tr)+1;models[name][target]=m
            if len(te):
                q=score(X[te],m);out=dict(method=name,target=target,threshold=m['threshold'],source_oof_recall=m['oof_recall'],source_refit_recall=m['refit_recall'],source_oof_ap=m['oof_ap'],held=metric(q,s[te],m['threshold'],blank[te]));rows.append(out)
    for target in ('w058','w060'):
        tr=w!=target;te=w==target
        for name,q in [('center',center),('median3',med)]:
            th=threshold(q[tr],s[tr]);rows.append(dict(method=name,target=target,threshold=th,held=metric(q[te],s[te],th,blank[te])))
    gates={}
    for name in NAMES:
        diff=[]
        for target in ('w058','w060'):
            a,b,c=[next(r['held'] for r in rows if r['target']==target and r['method']==m) for m in ('center','median3',name)]
            diff.append(dict(target=target,delta_recall_pp=100*(c['recall']-a['recall']),delta_fpr_pp=100*(c['fpr']-a['fpr']),delta_median_recall_pp=100*(c['recall']-b['recall'])))
        gates[name]=dict(deltas=diff,pass_=all(r['delta_recall_pp']>=-2 and r['delta_fpr_pp']<=-2 and r['delta_median_recall_pp']>=-2 for r in diff))
    return dict(schema='b25.learned_depth.v1',source_sha256=PRED,checks=checks,fits=fit_count,rows=rows,gates=gates,models=models,wall_seconds=time.perf_counter()-t,scope='Same 1447 development parent, spatially separated crops; OOF threshold vs full refit; no independent held manuscript score used.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('output',type=Path);v=a.parse_args();r=run(v.data);v.output.write_text(json.dumps(r,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in r.items() if k!='models'},indent=2))
