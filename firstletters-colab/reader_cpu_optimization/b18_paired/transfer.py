"""B19: operating-point transfer on pinned, real archived Studio scores."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np

PINS={'pose_factors.npz':'ba7e0f6a1032f8eb00cdb83d75b07613ce5c8cd496880943aaaf5652e3c372c2','scale_stress.npz':'04b107ef27c8a17cec25fbc90939d4ccb1b36190240a74f238fa75cf5a860e08','coordinates.npy':'4a93ec41845b451ce0652f64a9e89b50521f1484b5b5440cf7d27dda471ab1dd'}
METHODS=('fixed','identity','robust','quantile')


def positive_threshold(p):
    if len(p)==0:raise ValueError('no positive labels')
    # At least ceil(4/5*N) positives, entire ties retained.
    k=(4*len(p)+4)//5
    return float(np.sort(np.asarray(p,dtype=np.float64))[-k])


def quantile_threshold(p,k):
    if not 0<=k<=len(p) or not len(p):raise ValueError('invalid activated count')
    p=np.asarray(p,dtype=np.float64)
    if k==len(p):return float(np.nextafter(p.min(),-np.inf))
    return float(np.nextafter(np.sort(p)[::-1][k],np.inf))


def fit(scores,states,valid,train,target):
    # The target labels are never consulted: states encode source labels on known aligned pixels.
    source_pos=(states[train]==2)&valid[train]
    th=positive_threshold(scores[0,train][source_pos])
    other=[positive_threshold(scores[j,train][source_pos]) for j in range(len(scores)) if j!=target]
    assert all(np.count_nonzero(scores[j,train][source_pos]>=min(other))*5>=4*source_pos.sum() for j in range(len(scores)) if j!=target)
    p=scores[0,train][valid[train]];q=scores[target,train][valid[train]]
    k=int((p>=th).sum());qt=quantile_threshold(q,k)
    realized=int((q>=qt).sum());assert realized<=k
    return dict(fixed=.5,identity=th,robust=min(other),quantile=qt),[int(len(p)),k,realized]


def metric(p,labels,t):
    i=labels==2;b=labels==1;z=p>=t
    return [int(i.sum()),int(b.sum()),int((i&z).sum()),int((b&z).sum())]


def test():
    assert positive_threshold([.1,.2,.3,.4,.5])==.2
    assert positive_threshold([.2,.2,.2,.8,.9])==.2
    p=np.array([.2,.3,.3,.7,.9]);assert (p>=quantile_threshold(p,3)).sum()==2
    assert (p>=quantile_threshold(p,0)).sum()==0 and (p>=quantile_threshold(p,5)).sum()==5
    rng=np.random.default_rng(1900);s=rng.random((4,4,10,10));v=np.ones((4,10,10),bool);lab=np.ones(v.shape,np.uint8);lab[:,:5]=2;train=np.array([1,2,3]);a,_=fit(s,lab,v,train,2)
    ss=s.copy();ll=lab.copy();ss[:,0]=rng.random(ss[:,0].shape);ll[0]=0;assert fit(ss,ll,v,train,2)[0]==a
    # Quantile decisions under strictly increasing transformation of target scores are invariant.
    ss=s.copy();ss[2]=.1+.7*s[2];b,_=fit(ss,lab,v,train,2)
    assert np.array_equal(s[2]>=a['quantile'],ss[2]>=b['quantile'])
    return 6


def load(d):
    for n,h in PINS.items():
        with (d/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    coords=np.load(d/'coordinates.npy',allow_pickle=False)
    with np.load(d/'pose_factors.npz',allow_pickle=False) as z:
        p=z['probabilities'];orig=z['original_states'];valid=z['global_common'];ids=z['author_indices'];names=['identity']+z['block_names'].tolist()
        assert np.array_equal(p[:,0],np.broadcast_to(p[0,0],p[:,0].shape));scores=np.concatenate((p[:1,0],p[:,3]),axis=0).astype(np.float64)
        assert np.array_equal(valid,z['common_masks'].all(axis=0));yield 'pose',scores,orig,valid,coords[ids],ids,names,192
    with np.load(d/'scale_stress.npz',allow_pickle=False) as z:
        ids=z['author_indices'];orig=z['states'];scores=z['probabilities'][:5].astype(np.float64);names=z['condition_names'][:5].tolist();valid=np.ones_like(orig,bool)
        assert np.array_equal(coords[ids]+48,z['core_origins_yx']);yield 'scale',scores,orig,valid,coords[ids],ids,names,272


def run(d):
    tic=time.perf_counter();tests=test();rows=[];summary=[];abstain=[]
    for family,s,orig,valid,co,ids,names,guard in load(d):
        known=(orig>0).sum((1,2));ink=(orig==2).sum((1,2));mixed=np.flatnonzero((known>=512)&(10*ink>=known)&(10*ink<=9*known));assert len(mixed)==11
        states=np.where(valid,orig,0);dist=np.abs(co[:,None]-co[None]).max(2)
        for t in range(1,len(s)):
            group=[]
            for i in mixed:
                train=mixed[(mixed!=i)&(dist[i,mixed]>=guard)]
                if len(train)<2 or (states[train]==2).sum()<512 or (states[train]==1).sum()<512:
                    abstain.append([family,names[t],int(ids[i]),int((states[i]==2).sum()),int((states[i]==1).sum())]);continue
                th,activated=fit(s,states,valid,train,t)
                row=dict(family=family,arm=names[t],held=int(ids[i]),train=ids[train].tolist(),thresholds=th,activated=activated,counts={m:metric(s[t,i],states[i],th[m]) for m in METHODS});rows.append(row);group.append(row)
            counts={m:np.sum([r['counts'][m] for r in group],axis=0).tolist() for m in METHODS};rates={m:[v[2]/v[0],v[3]/v[1]] for m,v in counts.items()};deltas={m:[100*(a-b) for a,b in zip(rates[m],rates['identity'])] for m in METHODS}
            summary.append(dict(family=family,arm=names[t],attempted=11,admitted=len(group),counts=counts,rates=rates,delta_pp=deltas))
    gates={}
    for family in ['pose','scale']:
        a=[r for r in summary if r['family']==family];gates[family]={}
        for m in METHODS:
            q=np.array([r['delta_pp'][m] for r in a]);gates[family][m]={'mean_delta_pp':q.mean(axis=0).tolist(),'worst_recall_delta_pp':float(q[:,0].min()),'pass':bool(q[:,0].mean()>=-2 and q[:,1].mean()<=-2 and q[:,0].min()>=-5)}
    return dict(schema='b19.operating_point_transfer.v1',tests=tests,wall_seconds=time.perf_counter()-tic,source_sha256=PINS,rows=rows,abstain=abstain,summary=summary,gates=gates,scope='Archived Studio same donor scores; spatial guard not native independence, transductive quantile target scores, no Reader inference or cross-scroll accuracy.')

if __name__=='__main__':
    a=run(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(a,separators=(',',':'),allow_nan=False)+'\n');print(json.dumps({k:v for k,v in a.items() if k not in ('rows','abstain','summary')},indent=2));print('attempted',len(a['rows'])+len(a['abstain']),'admitted',len(a['rows']))
    for r in a['summary']:print(r['family'],r['arm'],r['admitted'],{k:[round(x,2) for x in v] for k,v in r['delta_pp'].items()})
