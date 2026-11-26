"""B21: hash-verified native 1447 annotated panel, actual Reader v3 CPU."""
import argparse, hashlib, json, resource, time, urllib.request
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import tifffile
from PIL import Image
from sklearn.metrics import average_precision_score
from fetch_fixture import fetch
from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.image_proc.intensity.normalization import normalize_robust

PIN='ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a'
REV='7e4d918712a8d8642a6fdeee45b80217c1e98317'
ROOT='https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/'+REV+'/'
WINDS=('w058','w060')
ARMS=('reverse','forward','minus2','plus2','bf16')
METHODS=('reverse','contrast','product','median3','bf16')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def select(ink,mask):
    rows={'mixed':[],'blank':[]}
    for y in range(0,ink.shape[0]-383,384):
        for x in range(0,ink.shape[1]-383,384):
            sl=np.s_[y+96:y+288,x+96:x+288];p=int((ink[sl]&mask[sl]).sum());n=int((~ink[sl]&mask[sl]).sum())
            if p>=512 and n>=512:rows['mixed'].append((y,x,p,n))
            elif p==0 and n>=4096:rows['blank'].append((y,x,p,n))
    out=[]
    for group,cap in [('mixed',4),('blank',2)]:
        pool=rows[group];selected=[]
        while pool and len(selected)<cap:
            idx=0 if not selected else max(range(len(pool)),key=lambda i:(min((pool[i][0]-r[0])**2+(pool[i][1]-r[1])**2 for r in selected),-i))
            selected.append(pool.pop(idx))
        for y,x,p,n in selected:out.append(dict(y=y,x=x,ink=p,bg=n,group=group))
    if not any(r['group']=='mixed' for r in out):raise ValueError('no mixed control')
    return out


def gray(path):
    with Image.open(path) as im:a=np.asarray(im)
    if a.ndim==3:
        assert a.shape[2] in (3,4) and np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])
        a=a[:,:,0]
    assert a.ndim==2 and a.dtype==np.uint8
    return a


def acquire(cache):
    cache.mkdir(parents=True,exist_ok=True);manifest=cache/'SHA256SUMS'
    if not manifest.exists():
        with urllib.request.urlopen(ROOT+'SHA256SUMS',timeout=90) as f:manifest.write_bytes(f.read())
    pins={}
    for l in manifest.read_text().splitlines():
        f=l.split(maxsplit=1)
        if len(f)==2 and len(f[0])==64:pins[f[1].lstrip('*').removeprefix('./')]=f[0]
    used={};data=[]
    def get(name):
        p=fetch(ROOT+name,pins[name],cache/name)
        used[name]={'sha256':sha(p),'bytes':p.stat().st_size};return p
    for w in WINDS:
        i=gray(get(w+'/labels/inklabels.png'))>127;m=gray(get(w+'/labels/mask.png'))>0
        assert i.shape==m.shape and i.shape=={'w058':(4512,8184),'w060':(4536,8488)}[w]
        rows=select(i,m);stack=np.empty((len(rows),21,384,384),np.uint8);labels=[]
        for r in rows:
            y,x=r['y'],r['x'];sl=np.s_[y+96:y+288,x+96:x+288];s=np.where(m[sl],np.where(i[sl],2,1),0).astype(np.uint8)
            assert int((s==2).sum())==r['ink'] and int((s==1).sum())==r['bg'];labels.append(s)
        for j,k in enumerate(range(2,23)):
            a=tifffile.imread(get(w+f'/layers/{k:02d}.tif'),maxworkers=1);assert a.shape==i.shape and a.dtype==np.uint8
            for n,r in enumerate(rows):stack[n,j]=a[r['y']:r['y']+384,r['x']:r['x']+384]
            del a
        print('B21_SELECTED',w,json.dumps(rows),flush=True)
        data.append((w,rows,stack,np.stack(labels)))
    return data,{'revision':REV,'manifest_sha256':sha(manifest),'files':used}


def load_model(checkpoint):
    assert sha(checkpoint)==PIN
    payload=load_checkpoint(checkpoint);cfg=InkConfig.from_mapping(payload['config']);assert cfg.data.mode=='flat' and tuple(cfg.model.crop_size)==(17,384,384)
    selected,state=select_inference_weights(payload);model=make_model(cfg).eval()
    if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);return model,cfg,selected


def predict(model,cfg,raw,amp=False):
    # Same HWZ float32 reader preparation as pinned Villa.
    hwz=np.asarray(np.moveaxis(raw,0,2),dtype=np.float32).copy(order='C');a=np.ascontiguousarray(normalize_robust(np.moveaxis(hwz,2,0)));x=torch.from_numpy(a[None,None])
    with torch.inference_mode(),(torch.autocast('cpu',dtype=torch.bfloat16) if amp else nullcontext()):
        q=model(center_pad_input_depth(x,cfg.model.input_pad_depth_to))['ink'];q=q[0] if isinstance(q,(tuple,list)) else q
        q=q.float().sigmoid()
        if q.shape[-2:]!=(384,384):q=torch.nn.functional.interpolate(q,size=(384,384),mode='bilinear',align_corners=False)
    q=q[0,0].cpu().numpy();assert q.shape==(384,384) and np.isfinite(q).all();return q[96:288,96:288].copy()


def scores(p):
    r,f,a,b,bf=p
    return dict(reverse=r,contrast=r-f,product=r*(1-f),median3=np.median(p[[0,2,3]],axis=0),bf16=bf)


def threshold(p,y):
    x=p[y==2];assert len(x)>0
    return float(np.sort(x)[-((4*len(x)+4)//5)])


def metrics(p,y,t,blank):
    ink=y==2;bg=y==1;q=p>=t;bm=blank[:,None,None]&bg;k=y>0
    tp=int((q&ink).sum());fp=int((q&bg).sum());n=int(ink.sum());b=int(bg.sum());bb=int(bm.sum());bf=int((q&bm).sum())
    return dict(ink=n,bg=b,tp=tp,fp=fp,recall=tp/n if n else None,fpr=fp/b if b else None,blank_bg=bb,blank_fp=bf,blank_fpr=bf/bb if bb else None,ap=float(average_precision_score(ink[k],p[k])) if n and b else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high')
    data,receipt=acquire(a.cache);protocol={'sources':receipt,'checkpoint_sha256':PIN,'selection':[{ 'winding':w,'patches':r,'raw_sha256':hashlib.sha256(s.tobytes()).hexdigest(),'label_sha256':hashlib.sha256(l.tobytes()).hexdigest()} for w,r,s,l in data]}
    (a.output/'selection.json').write_text(json.dumps(protocol,indent=2)+'\n');print('B21_FROZEN',sha(a.output/'selection.json'),flush=True)
    model,cfg,selected=load_model(a.checkpoint);all_p=[];all_s=[];all_w=[];all_blank=[];all_orig=[];times=[];bferrors=[]
    for w,rows,stack,lab in data:
        for r,raw,s in zip(rows,stack,lab):
            p=[]
            for name,band,amp in [('reverse',raw[2:19][::-1],False),('forward',raw[2:19],False),('minus2',raw[:17][::-1],False),('plus2',raw[4:21][::-1],False),('bf16',raw[2:19][::-1],True)]:
                t=time.perf_counter();v=predict(model,cfg,band,amp);dt=time.perf_counter()-t;p.append(v);times.append(dict(winding=w,y=r['y'],x=r['x'],arm=name,seconds=dt))
            pp=np.stack(p);err=np.abs(pp[0]-pp[4]);bferrors.append(dict(winding=w,y=r['y'],x=r['x'],max=float(err.max()),mean=float(err.mean()),changed=int(((pp[0]>=.5)!=(pp[4]>=.5)).sum()),n=int(err.size)))
            all_p.append(pp);all_s.append(s);all_w.append(w);all_blank.append(r['group']=='blank');all_orig.append([r['y'],r['x']])
    p=np.stack(all_p,axis=1);s=np.stack(all_s);w=np.array(all_w);blank=np.array(all_blank);ss=scores(p)
    np.savez_compressed(a.output/'predictions.npz',probabilities=p,states=s,windings=w,blank=blank,origins_yx=np.array(all_orig),arms=np.array(ARMS))
    rows=[]
    for target,source in [('w058','w060'),('w060','w058')]:
        tr=w==source;te=w==target
        for method in METHODS:
            q=ss[method];t=threshold(q[tr],s[tr]);oracle=threshold(q[te],s[te])
            r=dict(target=target,source=source,method=method,threshold=t,calibration=metrics(q[tr],s[tr],t,blank[tr]),held=metrics(q[te],s[te],t,blank[te]),within_target_oracle=metrics(q[te],s[te],oracle,blank[te]),fixed_0_5=metrics(q[te],s[te],.5,blank[te]) if method in ('reverse','bf16') else None)
            rows.append(r);print('B21_ROW',json.dumps(r),flush=True)
    gates={}
    for method in METHODS[1:]:
        dif=[]
        for target in WINDS:
            base=next(r['held'] for r in rows if r['target']==target and r['method']=='reverse');v=next(r['held'] for r in rows if r['target']==target and r['method']==method);dif.append([100*(v['recall']-base['recall']),100*(v['fpr']-base['fpr'])])
        gates[method]={'delta_pp_w058_w060':dif,'pass':all(x>=-2 and y<=-2 for x,y in dif)}
    result=dict(schema='b21.reader_1447.v1',selection_sha256=sha(a.output/'selection.json'),checkpoint_sha256=PIN,weights=selected,forwards=len(times),rows=rows,gates=gates,times=times,bf16_errors=bferrors,predictions_sha256=sha(a.output/'predictions.npz'),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Two windings of same 1447 parent; no independent cross-scroll validation or native geometric correction.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('B21_DONE',json.dumps({'forwards':len(times),'gates':gates,'timing_medians':{arm:float(np.median([x['seconds'] for x in times if x['arm']==arm])) for arm in ARMS},'bf16_max':max(r['max'] for r in bferrors),'bf16_changed':sum(r['changed'] for r in bferrors),'bf16_n':sum(r['n'] for r in bferrors),'prediction_sha256':result['predictions_sha256']}),flush=True)

if __name__=='__main__':main()
