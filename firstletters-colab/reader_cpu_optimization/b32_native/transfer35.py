"""B35: trusted-negative anchor transport and topology-preserving hysteresis."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_propagation
from sklearn.metrics import average_precision_score
from panels import load
METHODS=('source','source_hysteresis','blank_transport','blank_hysteresis')
def pos_threshold(v,s,num,den):
    x=v[s==2];assert len(x);return float(np.sort(x)[-((num*len(x)+den-1)//den)])
def transport(v,s,source_v,source_s,t):
    src=source_v[source_s==1];dst=v[s==1];assert len(src) and len(dst)
    k=int((src>=t).sum())*len(dst)//len(src)
    dst=dst.astype(np.float64);th=float(np.nextafter(dst.min(),-np.inf) if k==len(dst) else np.nextafter(np.sort(dst)[::-1][k],np.inf));assert (dst>=th).sum()<=k
    return th,dict(source_n=len(src),source_fp=int((src>=t).sum()),target_n=len(dst),budget=k,actual=int((dst>=th).sum()))
def hysteresis(v,lo,hi):
    v=np.asarray(v,np.float64);assert hi>=lo
    low=v>=lo;high=v>=hi
    out=np.stack([binary_propagation(a,structure=np.ones((3,3),bool),mask=b) for a,b in zip(high,low)])
    assert not (out&~low).any() and not (high&~out).any();return out
def counts(q,s):
    a=s==2;b=s==1;return dict(ink=int(a.sum()),bg=int(b.sum()),tp=int((a&q).sum()),fp=int((b&q).sum()),recall=float((a&q).sum()/a.sum()) if a.any() else None,fpr=float((b&q).sum()/b.sum()) if b.any() else None)
def test():
    v=np.zeros((1,32,32));v[0,4,4:20]=.4;v[0,4,4]=.8;v[0,18,2:12]=.4;q=hysteresis(v,.3,.7);assert q[0,4,4:20].all() and not q[0,18,2:12].any() and q.sum()==16
    v.fill(0);v[0,5:20,5]=.4;v[0,5,5:15]=.4;v[0,19,5:15]=.4;v[0,5,5]=.8;q=hysteresis(v,.3,.7);assert q.sum()==33 and not q[0,6:19,6:15].any()
    dst=np.array([.1,.3,.3,.9]);s=np.ones(4,np.uint8);t,r=transport(dst,s,np.array([.1,.2,.7,.8]),s,.5);assert r['budget']==2 and (dst>=t).sum()==1
    return 3
def run(d):
    t=time.perf_counter();checks=test();a,b,p,q=load(d);src=p['probabilities'][0];s=p['states'];w=p['windings'];blank=p['blank'];targets=[]
    for name in ('w058','w060'):
        tr=w!=name;te=w==name;targets.append((name,src[tr],s[tr],blank[tr],src[te],s[te],blank[te]))
    for j,name in enumerate(q['directions']):targets.append(('0841_'+str(name),src,s,blank,q['probabilities'][j,0],q['states'],q['blank']))
    rows=[];per=[];receipts=[];gates={}
    for name,v,ss,bb,z,ts,tb in targets:
        lo=pos_threshold(v,ss,4,5);hi=pos_threshold(v,ss,1,2);cal=int(np.flatnonzero(tb)[0]);te=np.arange(len(tb))!=cal;assert (ts[cal]==2).sum()==0
        a0,r0=transport(z[cal],ts[cal],v[bb],ss[bb],lo);a1,r1=transport(z[cal],ts[cal],v[bb],ss[bb],hi);assert a1>=a0
        receipts.append(dict(target=name,calibration_blank=cal,excluded_known_bg=int((ts[cal]==1).sum()),source_lo=lo,source_hi=hi,transport_lo=a0,transport_hi=a1,lo_receipt=r0,hi_receipt=r1))
        v=z[te].astype(np.float64);lab=ts[te];bl=tb[te];ids=np.flatnonzero(te);base=v>=lo
        methods=dict(source=base,source_hysteresis=hysteresis(v,lo,hi),blank_transport=v>=a0,blank_hysteresis=hysteresis(v,a0,a1))
        ref=counts(base,lab);base_crop=[counts(x,y) for x,y in zip(base,lab)];ap=float(average_precision_score(lab[lab>0]==2,v[lab>0]))
        for m,x in methods.items():
            r=counts(x,lab);r.update(target=name,method=m,center_ap=ap,blank=counts(x[bl],lab[bl]),unknown=int((lab==0).sum()),removed_from_low=int(((v>=(a0 if m.startswith('blank') else lo))&~x).sum()),recall_delta_pp=100*(r['recall']-ref['recall']),fpr_delta_pp=100*(r['fpr']-ref['fpr']))
            drops=[]
            for j,(mapp,truth) in enumerate(zip(x,lab)):
                c=counts(mapp,truth);dr=None if c['recall'] is None else 100*(c['recall']-base_crop[j]['recall']);per.append(dict(target=name,method=m,crop=int(ids[j]),**c,delta_recall_pp=dr))
                if dr is not None:drops.append(dr)
            r['worst_crop_recall_delta_pp']=min(drops);r['pass']=bool(r['recall_delta_pp']>=-2 and r['fpr_delta_pp']<=-2 and r['worst_crop_recall_delta_pp']>=-5 and (not name.startswith('0841') or r['recall']>=.6));rows.append(r)
    for m in METHODS:gates[m]=all(r['pass'] for r in rows if r['method']==m)
    return dict(schema='b35.negative_anchor_hysteresis.v1',tests=checks,rows=rows,per_crop=per,calibration=receipts,gates=gates,wall_seconds=time.perf_counter()-t,scope='Target trusted blank supervision, excluded from evaluation; no target mixed labels fitted, no independent blind validation or real fine-stroke certification.')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('out',type=Path);v=a.parse_args();r=run(v.data);v.out.write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');print(json.dumps({k:x for k,x in r.items() if k not in ('per_crop','calibration')},indent=2))
