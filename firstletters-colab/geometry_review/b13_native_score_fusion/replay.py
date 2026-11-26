#!/usr/bin/env python3
"""B13: actual pixelwise fusion and guarded calibration on public 0841 archives."""
import argparse, hashlib, json, time, resource, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

PINS = {
'pose_factors.npz':'ba7e0f6a1032f8eb00cdb83d75b07613ce5c8cd496880943aaaf5652e3c372c2',
'oblique_stress.npz':'9558cfd463a3d3450e0b7499321df8fa49b64fa83286c984666da2dfac5fe6ad',
'scale_stress.npz':'04b107ef27c8a17cec25fbc90939d4ccb1b36190240a74f238fa75cf5a860e08',
'coordinates.npy':'4a93ec41845b451ce0652f64a9e89b50521f1484b5b5440cf7d27dda471ab1dd',
'ct-tiles.npy':'0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33',
'labels.npy':'ff445929d549e081133a27141e96e30ff22136c441c882055215bace58bcf548',
 'tiles.npy':'fb8d65ff2a4c41e063a26c5cbc2312c34bd906f3927c5721653211286dfb16be'}
METHODS=('baseline','max','mean','zmean')
BLOCKS=['x_minus20','x_minus5','x_plus5','x_plus20','y_minus20','y_minus5','y_plus5','y_plus20']
FAMILIES=['identity','normal_only','anchor_only','rigid']
OBLIQUE=['identity','x_minus20','x_minus10','x_minus5','x_plus5','x_plus10','x_plus20','y_minus20','y_minus10','y_minus5','y_plus5','y_plus10','y_plus20']
SCALE=['identity','x_half','x_double','y_half','y_double','uniform_mean5']


def threshold(neg,budget):
    neg=np.asarray(neg,np.float64)
    if not len(neg) or not 0<=budget<=len(neg) or not np.isfinite(neg).all(): raise ValueError('invalid calibration')
    # B9 convention: retain all ties or exclude all ties conservatively.
    if budget==len(neg): return float(np.nextafter(neg.min(),-np.inf))
    return float(np.nextafter(np.sort(neg)[::-1][budget],np.inf))


def count(pred,s):
    ink=s==2;bg=s==1
    return [int(ink.sum()),int(bg.sum()),int((pred&ink).sum()),int((pred&bg).sum())]


def pack(c):
    i,b,t,f=map(int,c)
    return dict(ink=i,bg=b,tp=t,fp=f,recall=t/i if i else None,fpr=f/b if b else None)


def load(d):
    receipts={}
    for name,h in PINS.items():
        path=d/name
        with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=h: raise ValueError('pin mismatch: '+name)
        receipts[name]={'sha256':actual,'bytes':path.stat().st_size}
    coords=np.load(d/'coordinates.npy',allow_pickle=False);labels=np.load(d/'labels.npy',allow_pickle=False)
    arrays={}
    for name in ['pose_factors','oblique_stress','scale_stress']:
        with np.load(d/(name+'.npz'),allow_pickle=False) as z: a={k:z[k] for k in z.files}
        ids=a['author_indices']; p=a['probabilities']; orig=a.get('original_states',a['states'])
        if not np.array_equal(orig,np.stack([labels[y+48:y+80,x+48:x+80] for y,x in coords[ids]])): raise ValueError('label frame mismatch')
        if p.dtype!=np.float32 or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() or not np.isin(orig,[0,1,2]).all(): raise ValueError('invalid scores/labels')
        if name=='pose_factors':
            assert p.shape==(8,4,17,32,32) and a['block_names'].tolist()==BLOCKS and a['families'].tolist()==FAMILIES
            assert np.array_equal(a['states'],np.where(a['common_masks'],orig[None],0))
            assert np.array_equal(a['global_common'],a['common_masks'].all(axis=0))
        elif name=='oblique_stress':
            assert p.shape==(13,17,32,32) and a['condition_names'].tolist()==OBLIQUE
            assert np.array_equal(a['states'],np.where(a['common_mask'],orig,0))
        else:
            assert p.shape==(6,53,32,32) and a['condition_names'].tolist()==SCALE
            assert np.array_equal(a['core_origins_yx'],coords[ids]+48)
            assert np.array_equal(p[5],p[:5].mean(axis=0,dtype=np.float32))
        a['orig']=orig;a['coords']=coords[ids];arrays[name]=a
    a=arrays['pose_factors'];b=arrays['oblique_stress'];assert np.array_equal(a['author_indices'],b['author_indices'])
    for i,k in enumerate(BLOCKS):
        assert np.array_equal(a['probabilities'][i,0],b['probabilities'][0]);assert np.array_equal(a['probabilities'][i,3],b['probabilities'][OBLIQUE.index(k)])
    return arrays,receipts


def cohort(orig):
    a=(orig==2).sum((1,2));b=(orig==1).sum((1,2));n=a+b
    return np.flatnonzero((n>=512)&(10*a>=n)&(10*a<=9*n)),np.flatnonzero(n>=512)


def bound_check(q,s,test):
    # Exact B9 OR increments and stratified sharp marginal bounds.
    base=q[0,test]>=.5;others=q[1:,test]>=.5;union=base|others.any(axis=0)
    out={}
    for name,state in [('ink',2),('bg',1)]:
        truth=s[test]==state;N=truth.sum((1,2));b=(truth&base).sum((1,2));a=(truth[None]&others).sum((2,3));actual=(truth&union&~base).sum((1,2))
        lo=np.maximum(0,a.max(axis=0)-b);hi=np.minimum(N-b,a.sum(axis=0))
        assert np.all((actual>=lo)&(actual<=hi))
        out[name]=dict(n=int(N.sum()),baseline=int(b.sum()),lower=int(lo.sum()),actual=int(actual.sum()),upper=int(hi.sum()))
    return out


def evaluate(q,s,orig,valid,coords,ids,test,pool,guard):
    q=np.asarray(q,np.float64);dist=np.abs(coords[:,None,:]-coords[None,:,:]).max(axis=2)
    raw={'baseline':q[0],'max':q.max(axis=0),'mean':q.mean(axis=0)}
    folds=[];sums={m:np.zeros(4,np.int64) for m in METHODS};aps={m:[] for m in METHODS};cal_sum={m:0 for m in METHODS};cal_budget=0
    fixed={m:pack(count(v[test]>=.5,s[test])) for m,v in raw.items()}
    for i in test:
        train=pool[(pool!=i)&(dist[i,pool]>=guard)]
        negative=s[train]==1
        f={'held':int(ids[i]),'train':ids[train].tolist(),'ncal_bg':int(negative.sum()),'ncal_ink':int((s[train]==2).sum()),'held_counts':pack(count(raw['baseline'][i]>=.5,s[i]))}
        if len(train)<2 or negative.sum()<512:
            f['abstain']='insufficient calibration support';folds.append(f);continue
        fit=q[:,train][:,valid[train]];mu=fit.mean(axis=1);sd=fit.std(axis=1)
        if np.any(sd<1e-8): f['abstain']='degenerate normalization';folds.append(f);continue
        scores=dict(raw,zmean=((q-mu[:,None,None,None])/sd[:,None,None,None]).mean(axis=0))
        budget=int((raw['baseline'][train][negative]>=.5).sum());cal_budget+=budget;f['budget']=budget;f['methods']={}
        assert i not in train and np.all(dist[i,train]>=guard)
        for m,v in scores.items():
            t=.5 if m=='baseline' else threshold(v[train][negative],budget)
            fp=int((v[train][negative]>=t).sum());assert fp<=budget
            c=count(v[i]>=t,s[i]);sums[m]+=c;cal_sum[m]+=fp
            known=s[i]>0;y=s[i][known]==2
            ap=float(average_precision_score(y,v[i][known])) if y.any() and not y.all() else None
            if ap is not None: aps[m].append(ap)
            f['methods'][m]=dict(threshold=t,cal_fp=fp,ap=ap,**pack(c))
        folds.append(f)
    admitted=sum('abstain' not in f for f in folds);total=count(np.zeros_like(s[test],bool),s[test]);b=pack(sums['baseline'])
    pooled={m:dict(pack(c),macro_ap=float(np.mean(aps[m])) if aps[m] else None,cal_fp=cal_sum[m]) for m,c in sums.items()}
    for m,a in pooled.items():
        assert a['ink']==b['ink'] and a['bg']==b['bg']
        a['recall_delta']=a['recall']-b['recall'] if b['recall'] is not None else None
        a['fpr_delta']=a['fpr']-b['fpr'] if b['fpr'] is not None else None
    return dict(guard=guard,admitted=admitted,attempted=len(test),total_ink=total[0],total_bg=total[1],cal_budget=cal_budget,fixed=fixed,pooled=pooled,folds=folds)


def test():
    n=np.array([.9,.8,.8,.1]);assert (n>=threshold(n,2)).sum()==1;assert (n>=threshold(n,0)).sum()==0;assert (n>=threshold(n,4)).sum()==4
    rng=np.random.default_rng(130);orig=np.zeros((5,32,32),np.uint8);orig[:,:16]=2;orig[:,16:]=1
    q=rng.random((3,5,32,32));ids=np.arange(5);co=np.column_stack((ids*192,ids*0));valid=np.ones_like(orig,bool)
    a=evaluate(q,orig,orig,valid,co,ids,ids,ids,192);assert a['admitted']==5
    q2=q.copy();q2[:,0]=rng.random(q2[:,0].shape);b=evaluate(q2,orig,orig,valid,co,ids,ids,ids,192)
    assert all(a['folds'][0]['methods'][m]['threshold']==b['folds'][0]['methods'][m]['threshold'] for m in METHODS)
    b=evaluate(q,orig,orig,valid,co,ids,ids,ids,10000);assert b['admitted']==0 and all(v['ink']==0 for v in b['pooled'].values())
    bound_check(q,orig,ids);return 7


def run(d):
    t=time.perf_counter();tests=test();a,receipts=load(d);rows=[];overlaps=[]
    for name,z in a.items():
        mixed,eligible=cohort(z['orig']);assert len(mixed)==11
        if name=='pose_factors':
            jobs=[(block+'_'+base,z['probabilities'][i,arms],z['states'][i],z['common_masks'][i]) for i,block in enumerate(BLOCKS) for base,arms in [('identity',[0,1,2,3]),('rigid',[3,1,2])]]
        elif name=='oblique_stress':
            jobs=[('mild5',z['probabilities'][[0,3,4,9,10]],z['states'],z['common_mask']),('all13',z['probabilities'],z['states'],z['common_mask'])]
        else: jobs=[('all5',z['probabilities'][:5],z['states'],np.ones_like(z['states'],bool))]
        for job,q,s,valid in jobs:
            overlaps.append(dict(archive=name,job=job,**bound_check(q,s,mixed)))
            plans=[('mixed_loo',mixed,0)]
            if name=='scale_stress':plans += [('known_loo',eligible,0),('known_guard272',eligible,272)]
            else:plans += [('mixed_guard192',mixed,192)]
            for plan,pool,guard in plans:
                rows.append(dict(archive=name,job=job,plan=plan,**evaluate(q,s,z['orig'],valid,z['coords'],z['author_indices'],mixed,pool,guard)))
    return dict(schema='b13.real_fusion.v1',assets=receipts,tests=tests,wall_seconds=round(time.perf_counter()-t,3),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,python=sys.version.split()[0],numpy=np.__version__,rows=rows,overlaps=overlaps,scope='Same exposed 0841 panel; chart guard not native CT disjointness; no new network forwards or deployment.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);args=p.parse_args();r=run(args.data);args.output.write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('rows','overlaps')}));print('rows',len(r['rows']),'folds',sum(len(x['folds']) for x in r['rows']),'admitted',sum(x['admitted'] for x in r['rows']))
