"""B31: chart-texture coherence diagnostic on actual native Reader panels."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from cascade import PRED,sha,threshold,metric
from student28 import PINS,read_panel

def coherence(x):
    a=x.astype(np.float32)/255;gx=gaussian_filter(a,(0,1,1),order=(0,0,1));gy=gaussian_filter(a,(0,1,1),order=(0,1,0));xx=gaussian_filter((gx*gx).mean(0),2);yy=gaussian_filter((gy*gy).mean(0),2);xy=gaussian_filter((gx*gy).mean(0),2)
    return np.clip(np.sqrt((xx-yy)**2+4*xy*xy)/(xx+yy+1e-8),0,1)

def tests():
    y,x=np.indices((64,64));a=np.broadcast_to(np.round(128+100*np.sin(x/3)).astype(np.uint8),(17,64,64));v=coherence(a);w=coherence(a.transpose(0,2,1));assert np.mean(v[8:-8,8:-8])>.95 and np.mean(w[8:-8,8:-8])>.95;assert np.allclose(v,w.T,atol=1e-6);assert not coherence(np.full(a.shape,100,np.uint8)).any();return 3

def transform(center,med,c):return {'center':center,'median3':med,'center_minus':center-.1*c,'center_product':center*(1-.5*c),'median_minus':med-.1*c}

def main():
    start=time.perf_counter();d=Path(sys.argv[1]);p=Path(sys.argv[2]);o=Path(sys.argv[3]);assert sha(p)==PRED and sha(d/'pred841/predictions.npz')==PINS['pred841/predictions.npz'];checks=tests()
    raw,s,orig,w=read_panel(d,'panel1447');raw2,s2,orig2,_=read_panel(d,'panel841');a=np.load(p,allow_pickle=False);b=np.load(d/'pred841/predictions.npz',allow_pickle=False);assert np.array_equal(a['states'],s) and np.array_equal(a['origins_yx'],orig) and np.array_equal(a['windings'],w);assert np.array_equal(b['states'],s2) and np.array_equal(b['origins_yx'],orig2)
    c=np.stack([coherence(x[2:19])[96:288,96:288] for x in raw]);c2=np.stack([coherence(x[2:19])[96:288,96:288] for x in raw2]);v=a['probabilities'];z=b['probabilities'];q=transform(v[0],np.median(v[[0,2,3]],axis=0),c);stats=[];rows=[]
    for name,mask in [('w058',w=='w058'),('w060',w=='w060'),('0841',None)]:
        t,h=(s2,c2) if mask is None else (s[mask],c[mask]);k=t>0;stats.append(dict(group=name,ink_coherence=float(h[t==2].mean()),bg_coherence=float(h[t==1].mean()),auc_neg_coherence=float(roc_auc_score(t[k]==2,-h[k]))))
    for name in ('w058','w060'):
        te=w==name;tr=~te
        for m,pred in q.items():
            th=threshold(pred[tr],s[tr]);rows.append(dict(target=name,method=m,threshold=th,**metric(pred[te],s[te],th,a['blank'][te])))
    for i,sign in enumerate(b['directions']):
        t=transform(z[i,0],np.median(z[i,:3],axis=0),c2)
        for m,pred in t.items():
            th=threshold(q[m],s);rows.append(dict(target='0841_'+str(sign),method=m,threshold=th,**metric(pred,s2,th,b['blank'])))
    gates={}
    for m in q:
        deltas=[]
        for target in ('w058','w060','0841_ascending','0841_descending'):
            b0=next(r for r in rows if r['target']==target and r['method']=='center');v=next(r for r in rows if r['target']==target and r['method']==m);dr=100*(v['recall']-b0['recall']);df=100*(v['fpr']-b0['fpr']);deltas.append(dict(target=target,recall_delta_pp=dr,fpr_delta_pp=df,pass_=bool(dr>=-2 and df<=-2 and (not target.startswith('0841') or v['recall']>=.6))))
        gates[m]=dict(rows=deltas,same_parent=all(r['pass_'] for r in deltas[:2]),cross_parent=all(r['pass_'] for r in deltas[2:]))
    r=dict(schema='b31.native_coherence.v1',pins=PINS,teacher_sha256=PRED,checks=checks,stats=stats,rows=rows,gates=gates,wall_seconds=time.perf_counter()-start,scope='Actual native chart texture, not anatomical fibre labels; no new detector forwards or blind validation.')
    o.write_text(json.dumps(r,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
