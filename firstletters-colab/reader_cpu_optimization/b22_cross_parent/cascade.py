"""B23: exact replay and whole-crop cascade limits on real B21 predictions."""
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

PRED='baf48b9301c9697c0dd8f4efbb7ffde9c2f891305f08d39194323eb860bd2d81'
RESULT='e701fb6cb47faa35b83d78d39073330a4423289be9850786c07d0cba6b82edd1'
SEL='f83f9de701a9c3b6806a64e1c986ff47d6159a7840465e4cd781b545d25f0376'
MARGINS=(0.,.001,.002,.005,.01)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def threshold(p,y):
    v=p[y==2];assert len(v)>0
    return float(np.sort(v)[-((4*len(v)+4)//5)])


def metric(p,y,t,blank):
    pos=y==2;neg=y==1;z=p>=t;k=pos|neg;bg=neg&blank[:,None,None]
    n,b=int(pos.sum()),int(neg.sum());tp,fp=int((z&pos).sum()),int((z&neg).sum());bb,bf=int(bg.sum()),int((z&bg).sum())
    return dict(ink=n,bg=b,tp=tp,fp=fp,recall=tp/n,fpr=fp/b,blank_bg=bb,blank_fp=bf,blank_fpr=bf/bb if bb else None,ap=float(average_precision_score(pos[k],p[k])))


def mismatch(a,b,y):
    d=a!=b
    return dict(all=int(d.sum()),known=int((d&(y>0)).sum()),ink_lost=int((a&~b&(y==2)).sum()),ink_added=int((~a&b&(y==2)).sum()),bg_added=int((~a&b&(y==1)).sum()),bg_removed=int((a&~b&(y==1)).sum()))


def test():
    checks=0
    for t in (0.,.5,1.):
        for z in itertools.product((0.,.5,1.),repeat=3):
            q=np.array(z);a=q>=t;ref=np.median(q)>=t
            assert ref==(a.sum()>=2)
            if a[0]==a[1]:assert ref==a[0]
            checks+=1
    return checks


def run(d):
    start=time.perf_counter();checks=test()
    for n,h in [('predictions.npz',PRED),('results.json',RESULT),('selection.json',SEL)]:assert sha(d/n)==h,n
    old=json.loads((d/'results.json').read_text());selection=json.loads((d/'selection.json').read_text())
    assert old['predictions_sha256']==PRED and old['selection_sha256']==SEL
    with np.load(d/'predictions.npz',allow_pickle=False) as z:
        p,s,w,blank,orig=[z[k] for k in ('probabilities','states','windings','blank','origins_yx')]
        assert z['arms'].tolist()==['reverse','forward','minus2','plus2','bf16']
    assert p.shape==(5,12,192,192) and p.dtype==np.float32 and s.shape==(12,192,192) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.isin(s,[0,1,2]).all()
    # Cross-check source panel metadata against saved arrays.
    j=0
    for a in selection['selection']:
        for r in a['patches']:
            assert w[j]==a['winding'] and orig[j].tolist()==[r['y'],r['x']] and blank[j]==(r['group']=='blank')
            assert int((s[j]==2).sum())==r['ink'] and int((s[j]==1).sum())==r['bg'];j+=1
    assert j==12
    r,f,m,z,bf=p;median=np.median(p[[0,2,3]],axis=0);scores=dict(reverse=r,contrast=r-f,product=r*(1-f),median3=median,bf16=bf)
    metric_checks=0;max_metric_error=0.
    for row in old['rows']:
        q=scores[row['method']];tr=w==row['source'];te=w==row['target'];t=threshold(q[tr],s[tr]);assert t==row['threshold']
        jobs=[('calibration',tr,t),('held',te,t),('within_target_oracle',te,threshold(q[te],s[te]))]
        if row['fixed_0_5'] is not None:jobs.append(('fixed_0_5',te,.5))
        for name,k,u in jobs:
            obs=metric(q[k],s[k],u,blank[k]);ref=row[name]
            for key,v in obs.items():
                if v is None:assert ref[key] is None
                else:
                    err=abs(v-ref[key]);assert err<=1e-12,(name,key,v,ref[key]);max_metric_error=max(max_metric_error,err)
                metric_checks+=1
    # Measured arm medians from original runner, no actual cascade execution time.
    seconds={name:float(np.median([x['seconds'] for x in old['times'] if x['arm']==name])) for name in ('reverse','minus2','plus2','bf16')}
    rows=[];per_crop=[]
    for setting in ('fixed_0_5','other_winding'):
        t0=np.array([.5 if setting=='fixed_0_5' else threshold(r[w!=wi],s[w!=wi]) for wi in w],np.float64)
        tm=np.array([.5 if setting=='fixed_0_5' else threshold(median[w!=wi],s[w!=wi]) for wi in w],np.float64)
        # Thresholds are float32 representable; compare in FP64 to avoid implicit scalar casts.
        center=r.astype(np.float64)>=t0[:,None,None];mid=median.astype(np.float64)>=tm[:,None,None];fast=bf.astype(np.float64)>=t0[:,None,None]
        for name,other,third in [('minus2_first',m,z),('plus2_first',z,m)]:
            a=r.astype(np.float64)>=tm[:,None,None];c=other.astype(np.float64)>=tm[:,None,None];e=third.astype(np.float64)>=tm[:,None,None]
            disputed=a!=c;need=disputed.any(axis=(1,2));pred=np.where(need[:,None,None],(a.astype(np.uint8)+c+e)>=2,a)
            assert np.array_equal(pred,mid)
            first,second=('minus2','plus2') if name=='minus2_first' else ('plus2','minus2')
            estimated=len(w)*(seconds['reverse']+seconds[first])+int(need.sum())*seconds[second]
            rows.append(dict(setting=setting,method=name,third_required=int(need.sum()),crops=len(w),disputed_pixels=int(disputed.sum()),mismatches=mismatch(mid,pred,s),estimated_seconds=estimated,full_seconds=len(w)*(seconds['reverse']+seconds[first]+seconds[second])))
            per_crop.extend(dict(setting=setting,method=name,winding=str(w[i]),y=int(orig[i,0]),x=int(orig[i,1]),threshold=float(tm[i]),defer=bool(need[i]),disputed_pixels=int(disputed[i].sum())) for i in range(len(w)))
        for margin in MARGINS:
            near=np.abs(bf.astype(np.float64)-t0[:,None,None])<=margin;need=near.any(axis=(1,2));pred=np.where(need[:,None,None],center,fast)
            estimated=len(w)*seconds['bf16']+int(need.sum())*seconds['reverse']
            rows.append(dict(setting=setting,method='bf16_fallback',margin=margin,fp32_required=int(need.sum()),crops=len(w),near_pixels=int(near.sum()),mismatches=mismatch(center,pred,s),estimated_seconds=estimated,full_seconds=len(w)*seconds['reverse']))
            per_crop.extend(dict(setting=setting,method='bf16_fallback',margin=margin,winding=str(w[i]),y=int(orig[i,0]),x=int(orig[i,1]),threshold=float(t0[i]),defer=bool(need[i]),near_pixels=int(near[i].sum()),mismatches=int((center[i]!=pred[i]).sum())) for i in range(len(w)))
    return dict(schema='b23.whole_crop_cascade.v1',source_sha256=PRED,result_sha256=RESULT,selection_sha256=SEL,majority_oracle_checks=checks,metric_checks=metric_checks,max_metric_error=max_metric_error,arm_median_seconds=seconds,rows=rows,per_crop=per_crop,wall_seconds=time.perf_counter()-start,scope='Replay of real B21 predictions, not new forwards or measured cascade speed. Whole crop fallback, empirical BF16 margins not certified error bounds. No production gate.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('output',type=Path);v=a.parse_args();r=run(v.data);v.output.write_text(json.dumps(r,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in r.items() if k!='per_crop'},indent=2))
