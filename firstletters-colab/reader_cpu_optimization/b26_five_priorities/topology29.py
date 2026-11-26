"""B29: fixed topology filters on actual Reader maps + artificial open strokes."""
import hashlib,json,time,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter,uniform_filter,minimum_filter,convolve,label
from cascade import PRED,threshold,metric,sha
METHODS=('center','median3','median_filter','local_mean_product','erosion','line_average')

def transforms(center,med):
    # All maps are [N,H,W]; no filtering across crops.
    m=uniform_filter(med,size=(1,3,3),mode='nearest');ker=[]
    for k in range(4):
        a=np.zeros((5,5),np.float32)
        if k==0:a[2,:]=1/5
        elif k==1:a[:,2]=1/5
        elif k==2:np.fill_diagonal(a,1/5)
        else:np.fill_diagonal(a[:,::-1],1/5)
        ker.append(a[None])
    line=np.maximum.reduce([convolve(med,v,mode='nearest') for v in ker])
    return dict(center=center,median3=med,median_filter=median_filter(med,size=(1,3,3),mode='nearest'),local_mean_product=med*np.clip(m/.65,0,1),erosion=minimum_filter(med,size=(1,3,3),mode='nearest'),line_average=(med+line)*.5)

def artificial(bg):
    yy,xx=np.indices(bg.shape[-2:]);dy=yy-96;dx=xx-96;rad=np.sqrt(dy*dy+dx*dx);rows=[]
    for k in range(len(bg)):
        for width in (1,2,3):
            for shape in ('open_c','line'):
                if shape=='open_c':
                    truth=(np.abs(rad-20)<width*.5)&~((dx>0)&(np.abs(dy)<=5));aperture=(dx>=16)&(dx<=24)&(np.abs(dy)<=3)
                else:truth=(np.abs(dy)<width*.5)&(np.abs(dx)<=24);aperture=np.zeros_like(truth)
                p=np.minimum(bg[k],.30).copy();p[truth]=.65;assert not (aperture&truth).any()
                for name,s in transforms(p[None],p[None]).items():
                    det=s[0]>=.5;rows.append(dict(crop=k,width=width,shape=shape,method=name,stroke=int(truth.sum()),tp=int((det&truth).sum()),aperture=int(aperture.sum()),gap_fp=int((det&aperture).sum()),fp=int((det&~truth).sum()),components=int(label(det,np.ones((3,3)))[1])))
    return rows

def main():
    start=time.perf_counter();path=Path(sys.argv[1]);out=Path(sys.argv[2]);assert sha(path)==PRED
    with np.load(path,allow_pickle=False) as a:p,s,w,b=[a[k] for k in ('probabilities','states','windings','blank')];assert a['arms'].tolist()==['reverse','forward','minus2','plus2','bf16']
    med=np.median(p[[0,2,3]],axis=0);q=transforms(p[0],med);rows=[]
    for wi in ('w058','w060'):
        tr=w!=wi;te=~tr
        for name,v in q.items():
            t=threshold(v[tr],s[tr]);rows.append(dict(winding=wi,method=name,threshold=t,**metric(v[te],s[te],t,b[te])))
    syn=artificial(med[b]);gates={}
    for name in METHODS:
        delta=[]
        for wi in ('w058','w060'):
            x=next(r for r in rows if r['winding']==wi and r['method']=='center');y=next(r for r in rows if r['winding']==wi and r['method']==name);delta.append([100*(y['recall']-x['recall']),100*(y['fpr']-x['fpr'])])
        rr=[r for r in syn if r['method']==name];retention=[sum(r['tp'] for r in rr if r['width']==k)/sum(r['stroke'] for r in rr if r['width']==k) for k in (1,2,3)]
        gates[name]=dict(delta_pp=delta,stroke_recall_by_width=retention,worst_stroke_recall=min(r['tp']/r['stroke'] for r in rr),gap_fp=sum(r['gap_fp'] for r in rr),pass_=bool(all(a>=-2 and b<=-2 for a,b in delta) and min(r['tp']/r['stroke'] for r in rr)>=.95 and not any(r['gap_fp'] for r in rr)))
    assert all(r['tp']==r['stroke'] and r['gap_fp']==0 for r in syn if r['method']=='center')
    assert np.array_equal(q['center'],p[0]) and np.array_equal(q['median3'],med)
    result=dict(schema='b29.topology.v1',source_sha256=PRED,checks=3,wall_seconds=time.perf_counter()-start,rows=rows,synthetic=syn,gates=gates,scope='Real scores, same-parent label transfer; artificial strokes on clipped blank scores, not human letters or actual 3D fiber identification.')
    out.write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in result.items() if k!='synthetic'},indent=2))
if __name__=='__main__':main()
