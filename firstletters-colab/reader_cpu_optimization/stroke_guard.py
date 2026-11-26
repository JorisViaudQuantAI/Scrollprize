"""B16: guarded stroke support and threshold transfer across rendered geometries."""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter,uniform_filter,binary_erosion
from sklearn.metrics import average_precision_score

PINS={'scale_stress.npz':'04b107ef27c8a17cec25fbc90939d4ccb1b36190240a74f238fa75cf5a860e08','pose_factors.npz':'ba7e0f6a1032f8eb00cdb83d75b07613ce5c8cd496880943aaaf5652e3c372c2','coordinates.npy':'4a93ec41845b451ce0652f64a9e89b50521f1484b5b5440cf7d27dda471ab1dd','labels.npy':'ff445929d549e081133a27141e96e30ff22136c441c882055215bace58bcf548'}
METHODS=('raw','median3','mean3','line','blend')


def scores(p):
    # Input [core,y,x]. Only score interior, fully supported neighborhoods.
    a=np.asarray(p,np.float64)
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    opp=[np.minimum(v[:,1+dy:33+dy,1+dx:33+dx],v[:,1-dy:33-dy,1-dx:33-dx]) for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]]
    line=np.minimum(a,np.maximum.reduce(opp))
    return np.stack([a,median_filter(a,size=(1,3,3),mode='nearest'),uniform_filter(a,size=(1,3,3),mode='nearest'),line,.5*(a+line)])


def recall_threshold(positive):
    k=(4*len(positive)+4)//5
    return float(np.partition(positive,len(positive)-k)[len(positive)-k])


def counts(p,y,t):
    known=y>0;ink=y==2;bg=y==1;v=p>=t
    assert np.isfinite(p[known]).all()
    return np.array([ink.sum(),bg.sum(),(v&ink).sum(),(v&bg).sum()],dtype=np.int64)


def run_job(job,probs,labels,original,valid,coords,ids,arms,guard):
    mask=binary_erosion(valid,structure=np.ones((1,3,3),bool),border_value=0);y=np.where(mask,labels,0)
    ink=(original==2).sum((1,2));bg=(original==1).sum((1,2));n=ink+bg;pool=np.flatnonzero(n>=512);mixed=(10*ink>=n)&(10*ink<=9*n)&(n>=512)
    maps=np.stack([scores(p) for p in probs]);dist=np.abs(coords[:,None,:]-coords[None,:,:]).max(2)
    results=[];sums=np.zeros((2,len(arms),len(METHODS),4),np.int64);fixed=sums.copy();ap_sums={};abstain=[]
    for i in pool:
        cal=pool[(pool!=i)&(dist[i,pool]>=guard)];positive=y[cal]==2;negative=y[cal]==1
        if len(cal)<2 or positive.sum()<512 or negative.sum()<512:
            abstain.append(dict(id=int(ids[i]),cal=ids[cal].tolist(),cal_ink=int(positive.sum()),cal_bg=int(negative.sum()),ink=int((y[i]==2).sum()),bg=int((y[i]==1).sum())));continue
        assert i not in cal and np.all(dist[i,cal]>=guard)
        th=[recall_threshold(v[cal][positive]) for v in maps[0]]
        for m,t in enumerate(th):assert np.count_nonzero(maps[0,m,cal][positive]>=t)*5>=4*positive.sum()
        for ai,arm in enumerate(arms):
            for m,name in enumerate(METHODS):
                c=counts(maps[ai,m,i],y[i],th[m]);f=counts(maps[ai,m,i],y[i],.5)
                k=y[i]>0;truth=y[i][k]==2;ap=float(average_precision_score(truth,maps[ai,m,i][k])) if truth.any() and not truth.all() else None
                results.append(dict(id=int(ids[i]),arm=arm,method=name,threshold=th[m],counts=c.tolist(),fixed=f.tolist(),ap=ap))
                for g in [0]+([1] if mixed[i] else []):
                    sums[g,ai,m]+=c;fixed[g,ai,m]+=f
                    if ap is not None:ap_sums.setdefault((g,ai,m),[]).append(ap)
    summary=[]
    for g,name in enumerate(('all','mixed')):
        for ai,arm in enumerate(arms):
            for m,method in enumerate(METHODS):
                c=sums[g,ai,m];base=sums[g,ai,0];f=fixed[g,ai,m];assert np.array_equal(c[:2],base[:2]);a=ap_sums.get((g,ai,m),[])
                summary.append(dict(group=name,arm=arm,method=method,ink=int(c[0]),bg=int(c[1]),tp=int(c[2]),fp=int(c[3]),recall=float(c[2]/c[0]) if c[0] else None,fpr=float(c[3]/c[1]) if c[1] else None,recall_delta=float((c[2]-base[2])/c[0]) if c[0] else None,fpr_delta=float((c[3]-base[3])/c[1]) if c[1] else None,fixed_tp=int(f[2]),fixed_fp=int(f[3]),macro_ap=float(np.mean(a)) if a else None,ap_cores=len(a)))
    return dict(job=job,guard=guard,attempted=len(pool),admitted=len(pool)-len(abstain),abstained=abstain,summary=summary,folds=results)


def tests():
    a=np.zeros((1,32,32));a[0,16,16]=1;s=scores(a);assert s[3,0,16,16]==0
    a[0,16,15:18]=1;s=scores(a);assert s[3,0,16,16]==1 and s[3,0,16,15]==0
    a[:]=.25;assert np.all(scores(a)==.25)
    assert recall_threshold(np.array([.1,.2,.3,.4,.5]))==.2
    assert recall_threshold(np.array([.1,.2,.2,.2,.5]))==.2
    return 5


def main():
    started=time.perf_counter();d=Path(sys.argv[1]);out=Path(sys.argv[2]);receipts={}
    for k,h in PINS.items():
        p=d/k
        with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,k
        receipts[k]=h
    checks=tests();coords=np.load(d/'coordinates.npy',allow_pickle=False);labels=np.load(d/'labels.npy',allow_pickle=False);jobs=[]
    with np.load(d/'scale_stress.npz',allow_pickle=False) as z:
        ids=z['author_indices'];origin=z['states'];assert np.array_equal(origin,np.stack([labels[y:y+32,x:x+32] for y,x in coords[ids]+48]));assert z['probabilities'].shape==(6,53,32,32)
        jobs.append(run_job('scale',z['probabilities'][:5],origin,origin,np.ones_like(origin,bool),coords[ids],ids,z['condition_names'][:5].tolist(),272))
    with np.load(d/'pose_factors.npz',allow_pickle=False) as z:
        ids=z['author_indices'];origin=z['original_states'];assert np.array_equal(origin,np.stack([labels[y:y+32,x:x+32] for y,x in coords[ids]]));assert z['probabilities'].shape==(8,4,17,32,32)
        for i,block in enumerate(z['block_names'].tolist()):
            valid=z['common_masks'][i];assert np.array_equal(z['states'][i],np.where(valid,origin,0))
            jobs.append(run_job(block,z['probabilities'][i,[0,3]],z['states'][i],origin,valid,coords[ids],ids,['identity','rigid'],192))
    result=dict(schema='b16.stroke_guard.v1',pins=receipts,tests=checks,wall_seconds=round(time.perf_counter()-started,3),jobs=jobs,scope='Actual archived Studio scores, no Reader weights or new forward. Exposed same-donor chart guards; no transfer between manuscripts.')
    out.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n');print('jobs',[(j['job'],j['admitted'],j['attempted']) for j in jobs], 'tests',checks,'seconds',result['wall_seconds'])
    for j in jobs:
        print(j['job'],[(r['arm'],r['method'],round(100*r['recall_delta'],3),round(100*r['fpr_delta'],3)) for r in j['summary'] if r['group']=='mixed'])

if __name__=='__main__':main()
