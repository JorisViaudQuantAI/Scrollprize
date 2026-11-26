#!/usr/bin/env python3
"""B13: spatially guarded, same-donor CT-only supervised CPU cross-fits."""
import argparse, hashlib, json, time, resource, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage import uniform_filter, distance_transform_cdt, minimum_filter
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
PINS={
'coordinates.npy':'4a93ec41845b451ce0652f64a9e89b50521f1484b5b5440cf7d27dda471ab1dd',
'ct-tiles.npy':'0c2950497da0dc362ba572c2fb1c08f88e3accb51c1e741d0a85633fb46dde33',
'labels.npy':'ff445929d549e081133a27141e96e30ff22136c441c882055215bace58bcf548',
'oblique_stress.npz':'9558cfd463a3d3450e0b7499321df8fa49b64fa83286c984666da2dfac5fe6ad',
'pose_factors.npz':'ba7e0f6a1032f8eb00cdb83d75b07613ce5c8cd496880943aaaf5652e3c372c2',
'scale_stress.npz':'04b107ef27c8a17cec25fbc90939d4ccb1b36190240a74f238fa75cf5a860e08',
'tiles.npy':'fb8d65ff2a4c41e063a26c5cbc2312c34bd906f3927c5721653211286dfb16be'}
METHODS=('studio','logistic_profile','hgb_profile','hgb_context','mean_studio_context','product_studio_context')
def load(root):
    for name,expected in PINS.items():
        with (root/name).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected: raise ValueError(f'wrong SHA-256: {name}')
    arr={name:np.load(root/name,allow_pickle=False,mmap_mode='r') for name in PINS}
    l=arr['labels.npy'];c=arr['coordinates.npy'];t=arr['tiles.npy'];ct=arr['ct-tiles.npy']
    s=arr['scale_stress.npz'];p=arr['pose_factors.npz'];o=arr['oblique_stress.npz']
    assert l.shape==(2773,2609) and l.dtype==np.uint8 and np.isin(l,[0,1,2]).all()
    assert c.shape==(1334,2) and t.shape==(28,2) and ct.shape==(28,512,512,16) and ct.dtype==np.uint8
    origins=s['core_origins_yx'];states=s['states'];ids=s['author_indices'];probs=s['probabilities']
    assert len(ids)==53 and len(np.unique(ids))==53 and np.array_equal(origins,c[ids]+48)
    assert probs.shape==(6,53,32,32) and s['condition_names'].tolist()[0]=='identity'
    assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
    for (y,x),a in zip(origins,states): assert np.array_equal(l[y:y+32,x:x+32],a)
    ix=np.searchsorted(ids,p['author_indices']);assert np.array_equal(ids[ix],p['author_indices'])
    assert np.array_equal(states[ix],p['original_states']) and np.array_equal(p['original_states'],o['original_states'])
    assert np.array_equal(o['author_indices'],p['author_indices'])
    for j in range(8):
        v=p['common_masks'][j];assert np.array_equal(probs[0,ix][v],p['probabilities'][j,0][v])
        k=o['condition_names'].tolist().index(p['block_names'][j]);m=v&o['common_mask'];assert np.array_equal(p['probabilities'][j,3][m],o['probabilities'][k][m])
    assert np.array_equal(probs[0,ix][o['common_mask']],o['probabilities'][0][o['common_mask']])
    return l,t,ct,origins,states,ids,probs[0]
def make_features(l,tiles,ct):
    yy,xx=np.nonzero(l);lo=np.array([yy.min(),xx.min()])-4;hi=np.array([yy.max(),xx.max()])+5
    assert (lo>=0).all() and (hi<=l.shape).all()
    h,w=hi-lo;volume=np.zeros((h,w,16),np.uint8);support=np.zeros((h,w),np.uint8)
    for (y,x),tile in zip(tiles,ct):
        a,b=np.maximum([y,x],lo),np.minimum([y+512,x+512],hi)
        if (a>=b).any():continue
        a0,b0=a-lo,b-lo;at,bt=a-[y,x],b-[y,x]
        volume[a0[0]:b0[0],a0[1]:b0[1]]=tile[at[0]:bt[0],at[1]:bt[1]]
        support[a0[0]:b0[0],a0[1]:b0[1]]+=1
    ry,rx=yy-lo[0],xx-lo[1]
    assert np.all(minimum_filter((support==1).astype(np.uint8),size=9,mode='constant',cval=0)[ry,rx])
    q=volume.astype(np.float32)/255.;raw=q[ry,rx];profile=np.column_stack([raw,np.diff(raw,axis=1)])
    parts=[profile];
    for k in (3,9):
        mean=uniform_filter(q,size=(k,k,1));second=uniform_filter(q*q,size=(k,k,1));sd=np.sqrt(np.maximum(second-mean*mean,0.))
        parts.extend([mean[ry,rx],sd[ry,rx]])
        if k==9:parts.append(raw-mean[ry,rx])
    f=np.column_stack(parts).astype(np.float32);assert f.shape==(len(yy),111) and np.isfinite(f).all()
    for i in np.linspace(0,len(yy)-1,32,dtype=int):
        y,x=ry[i],rx[i];z=int(i%16)
        assert abs(float(f[i,z])-float(volume[y,x,z])/255)<1e-7
        a=q[y-4:y+5,x-4:x+5,z].astype(np.float64)
        assert abs(float(f[i,63+z])-a.mean())<2e-6
        assert abs(float(f[i,79+z])-a.std())<3e-5
    return f,l[yy,xx]==2,yy,xx,lo,hi
def metrics(y,p):
    assert len(y)>0 and y.any() and not y.all() and np.isfinite(p).all()
    detected=p>=.5;tp=int((detected&y).sum());fp=int((detected&~y).sum());n=int(y.sum());b=int((~y).sum())
    a=float(average_precision_score(y,p));auc=float(roc_auc_score(y,p))
    ix=np.argsort(-p,kind='stable');score=p[ix];truth=y[ix];ends=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(y)-1];hits=np.cumsum(truth)[ends]
    assert abs(a-float(np.sum(np.diff(np.r_[0,hits])*hits/(ends+1))/n))<1e-12
    return dict(ap=a,auc=auc,recall=tp/n,fpr=fp/b,ink=n,background=b,tp=tp,fp=fp)
def new_hgb():
    return HistGradientBoostingClassifier(max_iter=80,learning_rate=.1,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,early_stopping=False,random_state=20261126)
def run(root,output,workers):
    start=time.perf_counter();l,t,ct,origins,states,ids,studio=load(root);f,y,yy,xx,lo,hi=make_features(l,t,ct)
    indexes=np.full(l.shape,-1,np.int32);indexes[yy,xx]=np.arange(len(y),dtype=np.int32)
    test_idx=[];test_studio=[];test_groups=[];all_cores=np.zeros(l.shape,bool)
    for i,((a,b),s) in enumerate(zip(origins,states)):
        assert not all_cores[a:a+32,b:b+32].any();all_cores[a:a+32,b:b+32]=True
        ix=indexes[a:a+32,b:b+32][s!=0];assert (ix>=0).all() and np.array_equal(y[ix],s[s!=0]==2)
        test_idx.append(ix);test_studio.append(studio[i][s!=0].astype(np.float64));test_groups.append(np.full(len(ix),i,dtype=np.int32))
    ix_all=np.concatenate(test_idx);studio_all=np.concatenate(test_studio);group_all=np.concatenate(test_groups)
    assert len(np.unique(ix_all))==len(ix_all)==45349
    out=[];preds={};fold_count=0
    with threadpool_limits(limits=workers):
      for axis,name in [(0,'row'),(1,'col')]:
        cuts=np.quantile(origins[:,axis],[1/3,2/3]);groups=np.digitize(origins[:,axis],cuts)
        assert set(groups)=={0,1,2}
        dists=[]
        for fold in range(3):
            test_mask=np.zeros(l.shape,bool)
            for a,b in origins[groups==fold]:test_mask[a:a+32,b:b+32]=True
            dist=distance_transform_cdt(~test_mask,metric='chessboard')[yy,xx];dists.append(dist)
        for guard in (128,192):
          prediction=np.full((len(METHODS),len(ix_all)),np.nan,np.float64);prediction[0]=studio_all
          fr=[]
          for fold,dist in enumerate(dists):
            started=time.perf_counter();is_test=groups[group_all]==fold;te=ix_all[is_test];tr=np.flatnonzero(dist>guard)
            assert len(te)>0 and np.all(dist[te]==0) and np.min(dist[tr])>guard and len(np.intersect1d(tr,te))==0
            for a,b in origins[groups==fold]:
                dd=np.maximum(np.maximum(np.maximum(a-yy[tr],yy[tr]-(a+31)),np.maximum(b-xx[tr],xx[tr]-(b+31))),0)
                assert dd.min()>guard
            assert y[tr].any() and (~y[tr]).any()
            models=[make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=200,random_state=20261126)),new_hgb(),new_hgb()]
            for k,(model,nfeat) in enumerate(zip(models,(31,31,111)),1):
                model.fit(f[tr,:nfeat],y[tr]);prediction[k,is_test]=model.predict_proba(f[te,:nfeat])[:,1]
            prediction[4,is_test]=(prediction[0,is_test]+prediction[3,is_test])/2
            prediction[5,is_test]=prediction[0,is_test]*prediction[3,is_test]
            rec=dict(fold=fold,core_ids=ids[groups==fold].tolist(),train_ink=int(y[tr].sum()),train_bg=int((~y[tr]).sum()),train_min_guard=int(dist[tr].min()),test_ink=int(y[te].sum()),test_bg=int((~y[te]).sum()),methods={m:metrics(y[te],prediction[k,is_test]) for k,m in enumerate(METHODS)},seconds=round(time.perf_counter()-started,3));fr.append(rec);fold_count+=1
            print(name,guard,fold,'train',len(tr),'test',len(te),'AP',[round(rec['methods'][m]['ap'],4) for m in METHODS],f"{rec['seconds']}s",flush=True)
          assert np.isfinite(prediction).all();preds[f'{name}_{guard}']=prediction
          macro={m:{metric:float(np.mean([r['methods'][m][metric] for r in fr])) for metric in ('ap','auc','recall','fpr')} for m in METHODS}
          out.append(dict(partition=name,guard=guard,cuts=cuts.tolist(),folds=fr,macro=macro,pooled={m:metrics(y[ix_all],prediction[k]) for k,m in enumerate(METHODS)}))
    assert fold_count==12
    np.savez_compressed(output/'predictions.npz',**preds,truth=y[ix_all],core_ids=ids[group_all],yx=np.stack([yy[ix_all],xx[ix_all]],axis=1),method_names=np.array(METHODS))
    with (output/'predictions.npz').open('rb') as af: pred_sha=hashlib.file_digest(af,'sha256').hexdigest()
    result=dict(schema='b13.ct_classical_spatial.v1',source_sha256=PINS,execution=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,threads=workers,wall_seconds=round(time.perf_counter()-start,3),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,folds=fold_count,model_fits=3*fold_count,annotation_pixels=len(y),tested_pixels=len(ix_all),features=f.shape[1],source_rectangle=[lo.tolist(),hi.tolist()],literal_feature_checks=32,all_guard_checks_passed=True),predictions_sha256=pred_sha,results=out,scope='Same exposed donor, prepared CT, chart-distance guard only; not independent papyrus or new Studio inference.')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--threads',type=int,default=4);args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(args.input,args.output,args.threads)
