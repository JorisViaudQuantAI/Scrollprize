"""Q43, digital bright-deposit feasibility on public native CT; not chemical calibration."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
KEYS = ('raw_max','local_max','local_mass')
CORE = np.s_[16:112,16:112]

def mass(a):
    a = a.astype(np.float64)
    gap = np.median(np.concatenate((a[:4],a[24:]),axis=0),axis=0)
    return np.maximum(a[4:24]-gap,0).sum(axis=0)

def scores(a):
    mx = a[4:24].max(axis=0).astype(np.float64)
    m = mass(a)
    return (mx,mx-gaussian_filter(mx,32,mode='reflect'),m-gaussian_filter(m,32,mode='reflect'))

def stroke(i):
    y,x=np.mgrid[:128,:128]
    theta=np.deg2rad((i%4)*45)
    u=(x-64)*np.cos(theta)+(y-64)*np.sin(theta)
    v=-(x-64)*np.sin(theta)+(y-64)*np.cos(theta)
    return (np.abs(u)<36)&(np.abs(v)<9)

def inject(a,mask,dose):
    original=mass(a)
    out=a.copy()
    increment=dose*original*mask/3
    for i in (13,14,15):
        out[i]=np.clip(np.rint(out[i].astype(float)+increment),0,255).astype(np.uint8)
    req=float(dose*original[mask].sum())
    gain=float((mass(out)-original)[mask].sum())
    return out, (gain/req if req>0 else 0.0), req, gain

def auc(a,b):
    n,m=a.size,b.size
    return float((rankdata(np.r_[a,b],method='average')[:n].sum()-n*(n+1)/2)/(n*m))

def summarize(vals):
    return dict(median=float(np.median(vals)),minimum=float(np.min(vals)),maximum=float(np.max(vals)))

def selftest():
    a=np.full((28,128,128),40,np.uint8)
    a[4:24]=80
    mask=stroke(0)
    b,ratio,req,gain=inject(a,mask,.30)
    assert np.array_equal(b[:,~mask],a[:,~mask])
    assert .98<ratio<1.02 and req>0 and gain>0
    assert auc(np.array([3,4]),np.array([1,2]))==1.0
    assert auc(np.ones(3),np.ones(4))==.5
    assert np.allclose(scores(a)[1],0) and np.allclose(scores(a)[2],0)
    assert all(s.shape==(128,128) for s in scores(b))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    selftest()
    if args.self_test:
        print('Q43 self-tests passed')
        return
    if args.data is None or args.output is None:
        parser.error('data and output required')
    old=json.loads((HERE.parent.parent/'depth_recentering/results/source_inventory.json').read_text())
    a=np.load(args.data,allow_pickle=False)
    digest=hashlib.sha256(a.tobytes()).hexdigest()
    if a.shape!=(28,768,768) or a.dtype!=np.uint8 or digest!=old['array_bytes_sha256']:
        raise RuntimeError('native public CT source mismatch')
    def cell(r,c):return a[:,r*128:(r+1)*128,c*128:(c+1)*128]
    calibration=[scores(cell(r,c)) for r in (0,1) for c in range(6)]
    thresholds={k:float(np.quantile(np.concatenate([s[j][CORE].ravel() for s in calibration]),.99,method='linear')) for j,k in enumerate(KEYS)}
    cases=[]
    for r in (4,5):
        for c in range(6):
            i=(r-4)*6+c
            original=cell(r,c)
            old_scores=scores(original)
            mask=stroke(i)
            m=mask[CORE]
            assert mask.sum()==m.sum()
            for dose in (.10,.30):
                modified,ratio,req,gain=inject(original,mask,dose)
                sc=scores(modified)
                entry={'cell':f'r{r}c{c}','dose':dose,'stroke_pixels':int(m.sum()),'realized_fraction_of_requested':ratio,'requested_digital_mass':req,'realized_digital_mass':gain,'scores':{}}
                for j,k in enumerate(KEYS):
                    v=sc[j][CORE]; orig=old_scores[j][CORE]; th=thresholds[k]
                    entry['scores'][k]={'auc':auc(v[m],v[~m]),'sensitivity':float((v[m]>th).mean()),'offstroke_flagged_fraction':float((v[~m]>th).mean()),'original_flagged_fraction':float((orig>th).mean())}
                cases.append(entry)
    summary={}
    for dose in (.10,.30):
        rows=[e for e in cases if e['dose']==dose]
        summary[str(dose)]={'realized_fraction_of_requested':summarize([e['realized_fraction_of_requested'] for e in rows]),'cells_with_80pct_dose':sum(e['realized_fraction_of_requested']>=.8 for e in rows),'scores':{k:{m:summarize([e['scores'][k][m] for e in rows]) for m in rows[0]['scores'][k]} for k in KEYS}}
    r=summary['0.3']; c=r['scores']['local_mass']; b=r['scores']['local_max']
    gate={'dose_retained':r['cells_with_80pct_dose']>=10,'sensitivity':c['sensitivity']['median']>=.5,'original_flagged':c['original_flagged_fraction']['median']<=.02,'offstroke_flagged':c['offstroke_flagged_fraction']['median']<=.02,'gain_over_local_max':c['sensitivity']['median']-b['sensitivity']['median']>=.10}
    result={'experiment':'Q43_digital_bright_deposit_v1','source_array_sha256':digest,'source_inventory_sha256':hashlib.sha256((HERE.parent.parent/'depth_recentering/results/source_inventory.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest(),'thresholds':thresholds,'summary':summary,'gate':gate,'passed':all(gate.values()),'cases':cases,'limits':'Synthetic digital strokes on real public CT; not true lead, confirmed blank, or ink recognition.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Q43_RECEIPT_BEGIN')
    print(json.dumps(result,separators=(',',':'),allow_nan=False))
    print('Q43_RECEIPT_END')

if __name__=='__main__':
    main()
