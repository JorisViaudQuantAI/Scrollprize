#!/usr/bin/env python3
"""Retrospective pixelwise rescue replay; no model forward, no deployment gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARCHIVE_SHA = 'ba7e0f6a1032f8eb00cdb83d75b07613ce5c8cd496880943aaaf5652e3c372c2'
BLOCKS = ['x_minus20','x_minus5','x_plus5','x_plus20','y_minus20','y_minus5','y_plus5','y_plus20']
FAMILIES = ['identity','normal_only','anchor_only','rigid']
IDS = [250,288,326,357,358,359,360,361,362,395,399,430,432,469,471,472,509]


def threshold(neg, budget):
    """Largest allowed count at >= threshold; tied scores never split."""
    neg = np.asarray(neg, np.float64)
    if not len(neg) or not 0 <= budget <= len(neg) or not np.isfinite(neg).all():
        raise ValueError('invalid calibration background')
    if budget == len(neg):
        return float(np.nextafter(neg.min(), -np.inf))
    return float(np.nextafter(np.sort(neg)[::-1][budget], np.inf))


def ap(scores, positive):
    # Tie-aware non-interpolated AP.
    idx = np.argsort(-scores, kind='stable')
    s, y = scores[idx], positive[idx]
    if not y.any() or y.all():
        return None
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/tp[-1])


def counts(pred, states):
    p, n = states == 2, states == 1
    tp, fp = int((pred & p).sum()), int((pred & n).sum())
    return dict(ink=int(p.sum()), background=int(n.sum()), tp=tp, fp=fp,
                recall=tp/int(p.sum()) if p.any() else None,
                fpr=fp/int(n.sum()) if n.any() else None)


def run(p, s, original, support):
    # Mixed-core membership determined by original annotations, not predictions.
    ink, bg = (original==2).sum(axis=(1,2)), (original==1).sum(axis=(1,2))
    known = ink+bg
    mixed = np.flatnonzero((known>=512)&(10*ink>=known)&(10*ink<=9*known))
    if len(mixed)!=11:
        raise ValueError('unexpected mixed group')
    rows=[]
    for bi, block in enumerate(BLOCKS):
        q = p[bi].astype(np.float64)
        state = s[bi]
        valid = support[bi]
        if not np.isfinite(q[:,valid]).all() or not ((q[:,valid]>=0)&(q[:,valid]<=1)).all():
            raise ValueError('invalid probability on common support')
        for base, arms in [(0,[0,1,2,3]),(3,[3,1,2])]:
            out = {method:np.zeros_like(state,bool) for method in ('baseline','max','mean','zmean')}
            fold=[]
            fixed = {m:counts(v[mixed]>=.5,state[mixed]) for m,v in [('baseline',q[base]),('max',q[arms].max(axis=0)),('mean',q[arms].mean(axis=0))]}
            for held in mixed:
                train=mixed[mixed!=held]
                # No held-core values or labels for calibration or z normalization.
                fit=q[arms][:,train][:,valid[train]]
                mu=fit.mean(axis=1); sd=fit.std(axis=1)
                if np.any(sd<1e-8):
                    raise ValueError('degenerate z normalization; abstain')
                scores={'baseline':q[base], 'max':q[arms].max(axis=0), 'mean':q[arms].mean(axis=0), 'zmean':((q[arms]-mu[:,None,None,None])/sd[:,None,None,None]).mean(axis=0)}
                negative=state[train]==1
                budget=int((scores['baseline'][train][negative]>=.5).sum())
                row={'held':int(IDS[held]),'calibration_negative':int(negative.sum()),'calibration_fp_budget':budget,'methods':{}}
                for method,v in scores.items():
                    t=.5 if method=='baseline' else threshold(v[train][negative],budget)
                    out[method][held]=v[held]>=t
                    calibration_fp=int((v[train][negative]>=t).sum())
                    if calibration_fp>budget: raise AssertionError('calibration budget exceeded')
                    row['methods'][method]=dict(threshold=t,calibration_fp=calibration_fp,ap=ap(v[held][state[held]>0],state[held][state[held]>0]==2),**counts(out[method][held],state[held]))
                fold.append(row)
            rows.append(dict(block=block,baseline=FAMILIES[base],arms=[FAMILIES[a] for a in arms],fixed_0_5=fixed,held_out={m:counts(v[mixed],state[mixed]) for m,v in out.items()},folds=fold))
    return rows


def selftest():
    # Ties cannot silently exceed the FPR budget.
    n=np.array([.9,.8,.8,.1]); t=threshold(n,2)
    assert (n>=t).sum()==1
    assert (n>=threshold(n,0)).sum()==0
    assert (n>=threshold(n,4)).sum()==4
    assert abs(ap(np.array([.8,.8,.1]),np.array([True,False,True]))-7/12)<1e-12
    # Spatially grouped synthetic archive, deliberately no claims about ink.
    rng=np.random.default_rng(721); original=np.zeros((17,32,32),np.uint8)
    original[:11,:16]=2;original[:11,16:]=1;original[11:14]=2;original[14:]=1
    s=np.broadcast_to(original,(8,17,32,32)).copy(); valid=np.ones_like(s,bool)
    p=rng.random((8,4,17,32,32),dtype=np.float32)
    r=run(p,s,original,valid); assert len(r)==16 and all(len(x['folds'])==11 for x in r)
    for row in r:
        assert row['held_out']['baseline']==row['fixed_0_5']['baseline']
        for f in row['folds']:
            assert all(m['calibration_fp']<=f['calibration_fp_budget'] for m in f['methods'].values())
    # Mutate held-out probabilities: calibration thresholds for that fold must not change.
    before=r[0]['folds'][0]['methods'];p[:,:,0]=rng.random((8,4,32,32),dtype=np.float32)
    after=run(p,s,original,valid)[0]['folds'][0]['methods']
    assert all(before[k]['threshold']==after[k]['threshold'] for k in before)
    print('PASS: ties, edge budgets, AP, 176 held-out folds, calibration budget, no held-core leakage')


def main():
    a=argparse.ArgumentParser();a.add_argument('archive',nargs='?');a.add_argument('--selftest',action='store_true');args=a.parse_args()
    if args.selftest: selftest(); return
    if not args.archive: a.error('archive required')
    path=Path(args.archive)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE_SHA: raise ValueError('wrong archive hash')
    with np.load(path,allow_pickle=False) as z:
        if z['block_names'].tolist()!=BLOCKS or z['families'].tolist()!=FAMILIES or z['author_indices'].tolist()!=IDS: raise ValueError('wrong lineage')
        p,s,o,c=[z[k] for k in ('probabilities','states','original_states','common_masks')]
        if p.shape!=(8,4,17,32,32) or p.dtype!=np.float32 or s.shape!=(8,17,32,32) or o.shape!=(17,32,32) or c.shape!=s.shape or c.dtype!=bool: raise ValueError('wrong shape/dtype')
        if not np.isin(o,[0,1,2]).all() or not np.array_equal(s,np.where(c,o[None],0)): raise ValueError('bad support or state')
        result=run(p,s,o,c)
    print(json.dumps(dict(archive_sha256=ARCHIVE_SHA,scope='Exploratory same-donor replay; leave-one-core-out calibration is not independent scroll validation. Rigid hybrids use oracle reference geometry. Not a deployed rule.',rows=result),indent=2,allow_nan=False))

if __name__=='__main__':main()
