"""B24: component and crop retention audit on the real B21 Reader archive."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import label,binary_dilation
from cascade import PRED,sha,threshold

STRUCT=np.ones((3,3),bool)

def components(s):
    ids,n=label(s==2,structure=STRUCT)
    sizes=np.bincount(ids.ravel(),minlength=n+1)
    edge=np.zeros(s.shape,bool);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
    touching=np.unique(ids[(edge|binary_dilation(s==0,structure=STRUCT))&(ids>0)])
    return ids,sizes,np.isin(np.arange(n+1),touching)

def test():
    s=np.ones((8,8),np.uint8);s[1:3,1:3]=2;s[3,3]=2;s[6,6]=2;s[6,5]=0
    ids,a,b=components(s);assert a[1:].tolist()==[5,1] and b[1:].tolist()==[False,True]
    s[0,0]=2;_,a,b=components(s);assert a[1:].tolist()==[6,1] and b[1:].tolist()==[True,True]
    return 2

def run(d):
    start=time.perf_counter();checks=test();assert sha(d/'predictions.npz')==PRED
    with np.load(d/'predictions.npz',allow_pickle=False) as z:
        p,s,w,orig=[z[k] for k in ('probabilities','states','windings','origins_yx')]
        assert z['arms'].tolist()==['reverse','forward','minus2','plus2','bf16']
    center=p[0];med=np.median(p[[0,2,3]],axis=0);crops=[];rows=[]
    for setting in ('fixed_0_5','other_winding'):
        for j,wi in enumerate(w):
            tr=w!=wi;t0=.5 if setting=='fixed_0_5' else threshold(center[tr],s[tr]);t1=.5 if setting=='fixed_0_5' else threshold(med[tr],s[tr]);a=center[j]>=t0;b=med[j]>=t1;pos=s[j]==2;neg=s[j]==1
            k=int(pos.sum());n=int(neg.sum());tpa=int((a&pos).sum());tpb=int((b&pos).sum());fpa=int((a&neg).sum());fpb=int((b&neg).sum())
            crops.append(dict(setting=setting,winding=str(wi),crop=j,y=int(orig[j,0]),x=int(orig[j,1]),ink=k,bg=n,center_tp=tpa,median_tp=tpb,center_fp=fpa,median_fp=fpb,delta_recall_pp=100*(tpb-tpa)/k if k else None,delta_fpr_pp=100*(fpb-fpa)/n if n else None))
            ids,sizes,edge=components(s[j]);ta=np.bincount(ids[a&pos],minlength=len(sizes));tb=np.bincount(ids[b&pos],minlength=len(sizes));assert ta.sum()==tpa and tb.sum()==tpb and sizes[1:].sum()==k
            for i in range(1,len(sizes)):
                n=int(sizes[i]);u=int(ta[i]);v=int(tb[i]);assert 0<=u<=n and 0<=v<=n
                rows.append(dict(setting=setting,winding=str(wi),crop=j,component=i,pixels=n,eligible=n>=16,boundary=bool(edge[i]),size_bin='lt16' if n<16 else '16-63' if n<64 else '64-255' if n<256 else '256-1023' if n<1024 else '1024+',center_tp=u,median_tp=v,delta_recall_pp=100*(v-u)/n))
    summaries=[]
    for setting in ('fixed_0_5','other_winding'):
        for wi in ('ALL','w058','w060'):
            rr=[r for r in rows if r['setting']==setting and (wi=='ALL' or r['winding']==wi) and r['eligible']]
            cc=[r for r in crops if r['setting']==setting and (wi=='ALL' or r['winding']==wi)]
            by=[]
            for subset in ('all','16-63','64-255','256-1023','1024+','interior','boundary'):
                v=[r for r in rr if subset=='all' or r['size_bin']==subset or (subset=='interior' and not r['boundary']) or (subset=='boundary' and r['boundary'])]
                crossings={}
                for t in (10,50,80):
                    a=[100*r['center_tp']>=t*r['pixels'] for r in v];b=[100*r['median_tp']>=t*r['pixels'] for r in v]
                    crossings[str(t)]=dict(center=sum(a),median=sum(b),lost=sum(x and not y for x,y in zip(a,b)),gained=sum(y and not x for x,y in zip(a,b)))
                by.append(dict(subset=subset,n=len(v),pixels=sum(r['pixels'] for r in v),crossings=crossings,loss_all_from_half=sum(2*r['center_tp']>=r['pixels'] and r['median_tp']==0 for r in v),gain_half_from_zero=sum(r['center_tp']==0 and 2*r['median_tp']>=r['pixels'] for r in v),loss_all_any=sum(r['center_tp']>0 and r['median_tp']==0 for r in v),gain_any_from_zero=sum(r['center_tp']==0 and r['median_tp']>0 for r in v)))
            summaries.append(dict(setting=setting,winding=wi,crops=len(cc),crops_with_ink=sum(r['ink']>0 for r in cc),recall_drop_gt5=[r for r in cc if r['delta_recall_pp'] is not None and r['delta_recall_pp']<-5],components=by))
    return dict(schema='b24.component_retention.v1',source_sha256=PRED,tests=checks,rows=rows,crops=crops,summaries=summaries,wall_seconds=time.perf_counter()-start,scope='Partial annotation components, not individual letters; same exposed manuscript, no new inference.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('output',type=Path);v=a.parse_args();r=run(v.data);v.output.write_text(json.dumps(r,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('rows','crops')},indent=2))
