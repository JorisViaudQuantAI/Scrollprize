"""B30: actual Reader radiometric stress and unlabeled histogram transport."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'b18_paired'))
from reader_1447 import load_model,predict,metrics,threshold,sha
PIN841='926a284be12043aff107107c97b04a0af472e4e37c27a5277131891bad7d6357'
PIN1447='0f529972da360305aaad80d3400f209230427aa4ad9ad45b8df0f5807e4c9fb4'
PINPRED='2fb79c53bc83c4d008a9bcffaad7fd788691140e24c7012440b61553ffb8c691'

def lut(raw,source):
    h=np.bincount(raw.ravel(),minlength=256);q=(h.cumsum()-.5*h)/h.sum()
    cdf=source.cumsum()/source.sum();v=np.minimum(np.searchsorted(cdf,q,side='left'),255).astype(np.uint8)
    assert np.all(np.diff(v.astype(int))>=0)
    return v

def main():
    root=Path(sys.argv[1]);ck=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(exist_ok=False,parents=True)
    for n,h in [('841/panel.npz',PIN841),('1447/panel.npz',PIN1447),('pred/predictions.npz',PINPRED)]:assert sha(root/n)==h
    p=np.load(root/'841/panel.npz',allow_pickle=False);source=np.load(root/'1447/panel.npz',allow_pickle=False);pred=np.load(root/'pred/predictions.npz',allow_pickle=False)
    raw=p['raw'][:,2:19];s=p['states'];assert np.array_equal(s,pred['states']) and np.array_equal(p['origins_yx'],pred['origins_yx'])
    hist=np.bincount(source['raw'][:,2:19].ravel(),minlength=256);v=lut(np.full((10,),100,np.uint8),hist);assert len(v)==256 and v[100] in range(256)
    target_hist=np.bincount(raw.ravel(),minlength=256);lut_pooled=lut(raw,hist)
    bands={'original':raw,'cdf_per_crop':np.stack([lut(x,hist)[x] for x in raw]),'gain_bias':np.clip(np.rint(raw.astype(np.float32)*1.15+8),0,255).astype(np.uint8),'gamma08':np.clip(np.rint(255*(raw.astype(np.float32)/255)**.8),0,255).astype(np.uint8)}
    cal=json.loads((HERE.parent/'b22_cross_parent'/'calibration.json').read_text())['thresholds']['center'];blank=pred['blank']
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high');model,cfg,weights=load_model(ck)
    rows=[];times=[];qall=[]
    for i,sign in enumerate(('ascending','descending')):
        qq=[]
        for mode,x in bands.items():
            q=[]
            for n,v in enumerate(x):
                st=time.perf_counter();z=predict(model,cfg,v if i==0 else v[::-1]);dt=time.perf_counter()-st;q.append(z);times.append(dict(direction=sign,mode=mode,crop=n,seconds=dt));print('B30_FORWARD',json.dumps(times[-1]),flush=True)
            q=np.stack(q);qq.append(q);base=pred['probabilities'][i,0];err=np.abs(q-base);r=dict(direction=sign,mode=mode,held=metrics(q,s,cal,blank),oracle=metrics(q,s,threshold(q,s),blank),fixed=metrics(q,s,.5,blank),mae_vs_B26=float(err.mean()),max_vs_B26=float(err.max()),changed_vs_B26=int(((q>=cal)!=(base>=cal)).sum()),bitwise_vs_B26=bool(np.array_equal(q,base)));rows.append(r);print('B30_ROW',json.dumps(r),flush=True)
        qall.append(qq)
    np.savez_compressed(out/'predictions.npz',probabilities=np.array(qall),states=s,modes=np.array(list(bands)),directions=np.array(['ascending','descending']))
    result=dict(schema='b30.reader_radiometric.v1',source_sha256=PIN1447,target_sha256=PIN841,baseline_sha256=PINPRED,forwards=len(times),rows=rows,times=times,source_hist=hist.tolist(),target_hist=target_hist.tolist(),prediction_sha256=sha(out/'predictions.npz'),scope='Actual Reader fixed native chart, unlabeled radiometric stress/transport; both orientations; no target tuning or independent blind claim.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('B30_DONE',json.dumps({'forwards':len(times),'prediction_sha256':result['prediction_sha256']}),flush=True)
if __name__=='__main__':main()
