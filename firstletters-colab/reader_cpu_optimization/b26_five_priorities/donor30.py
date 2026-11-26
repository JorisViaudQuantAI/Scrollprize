"""B30: unlabeled raw/normalized distribution donor selection on staged native panels."""
import json,sys,time
from pathlib import Path
import numpy as np
from cascade import PRED,sha,threshold,metric
from fast_mad import reference
from student28 import PINS,read_panel

def distributions(raw):
    h=[];q=[]
    for x in raw[:,2:19]:
        v=np.bincount(x.ravel(),minlength=256).astype(np.float64);h.append(v/v.sum());q.append(np.quantile(reference(x),np.linspace(0,1,101)))
    return np.array(h),np.array(q)

def main():
    start=time.perf_counter();d=Path(sys.argv[1]);p=Path(sys.argv[2]);o=Path(sys.argv[3]);assert sha(p)==PRED and sha(d/'pred841/predictions.npz')==PINS['pred841/predictions.npz']
    x,s,orig,w=read_panel(d,'panel1447');y,t,orig2,n=read_panel(d,'panel841');a=np.load(p,allow_pickle=False);b=np.load(d/'pred841/predictions.npz',allow_pickle=False);assert np.array_equal(a['states'],s) and np.array_equal(a['windings'],w) and np.array_equal(a['origins_yx'],orig);assert np.array_equal(b['states'],t) and np.array_equal(b['origins_yx'],orig2)
    hh,qq=distributions(x);h2,q2=distributions(y);groups={'w058':(hh[:6],qq[:6]),'w060':(hh[6:],qq[6:]),'0841':(h2,q2)};dist=[]
    for target,(th,tq) in groups.items():
        for source,(sh,sq) in groups.items():
            raw=float(np.abs(th.mean(0).cumsum()-sh.mean(0).cumsum()).sum());norm=float(np.abs(tq.mean(0)-sq.mean(0)).mean());dist.append(dict(target=target,source=source,raw_wasserstein=raw,normalized_quantile_distance=norm))
    choices={k:min([v for v in dist if v['target']=='0841' and v['source']!='0841'],key=lambda v:(v[k],v['source']))['source'] for k in ('raw_wasserstein','normalized_quantile_distance')};rows=[];teacher=a['probabilities'][0];q=b['probabilities'][:,0];source_thresholds={wi:threshold(teacher[w==wi],s[w==wi]) for wi in ('w058','w060')};source_thresholds['pooled1447']=threshold(teacher,s)
    for i,sign in enumerate(b['directions']):
        for wi,th in source_thresholds.items():rows.append(dict(target='0841_'+str(sign),source=wi,threshold=th,**metric(q[i],t,th,b['blank'])))
    for target in ('w058','w060'):
        source='w060' if target=='w058' else 'w058';m=w==target;rows.append(dict(target=target,source=source,threshold=source_thresholds[source],**metric(teacher[m],s[m],source_thresholds[source],a['blank'][m])))
    assert all(v['raw_wasserstein']==0 and v['normalized_quantile_distance']==0 for v in dist if v['source']==v['target']);assert len(choices)==2
    r=dict(schema='b30.donor_diagnostic.v1',pins=PINS,teacher_sha256=PRED,checks=2,distances=dist,nearest=choices,thresholds=source_thresholds,rows=rows,wall_seconds=time.perf_counter()-start,scope='Two donor windings share one parent; distribution distance selected without held labels, not generalized donor learning.')
    o.write_text(json.dumps(r,separators=(',',':'))+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
