"""Network-enabled runner: public native Reader v3 benchmark, no unread data."""
import argparse, hashlib, json, resource, time, os
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import zarr
from paired import paired
from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.image_proc.intensity.normalization import normalize_robust

PIN='ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a'
POSITIONS=((0,0),(192,192))


def main():
    p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    with a.checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PIN
    st=time.perf_counter();payload=load_checkpoint(a.checkpoint);cfg=InkConfig.from_mapping(payload['config']);assert cfg.data.mode=='flat' and tuple(cfg.model.crop_size)==(17,384,384)
    selected,state=select_inference_weights(payload);model=make_model(cfg).eval()
    if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);del state,payload
    load_seconds=time.perf_counter()-st
    v=zarr.open_group(str(a.fixture),mode='r')['0'];assert v.shape==(28,768,768) and v.dtype==np.uint8
    start=v.shape[0]//2-17//2;assert start==6
    def prepare(y,x,shared):
        if shared:
            raw=np.moveaxis(np.asarray(v[start:start+17,y:y+384,x:x+384]),0,2)
            return paired(raw,17,384,384)
        out=[]
        for rev in (False,True):
            raw=np.asarray(v[start:start+17,y:y+384,x:x+384])
            if rev:raw=raw[::-1]
            # Match upstream HWZ float32 reader then moveaxis.
            hwz=np.asarray(np.moveaxis(raw,0,2),dtype=np.float32).copy(order='C')
            out.append(np.ascontiguousarray(normalize_robust(np.moveaxis(hwz,2,0))))
        return tuple(out),False
    def forward(inp,batch,bf16):
        arr=np.stack(inp)[:,None]
        out=[]
        with torch.inference_mode(),(torch.autocast('cpu',dtype=torch.bfloat16) if bf16 else nullcontext()):
            for j in range(0,len(inp),batch):
                x=torch.from_numpy(arr[j:j+batch])
                logits=model(center_pad_input_depth(x,cfg.model.input_pad_depth_to))['ink']
                if isinstance(logits,(tuple,list)):logits=logits[0]
                prob=logits.float().sigmoid()
                if prob.shape[-2:]!=(384,384):prob=torch.nn.functional.interpolate(prob,size=(384,384),mode='bilinear',align_corners=False)
                assert prob.shape[1:]==(1,384,384)
                out.append(prob[:,0].cpu().numpy())
        q=np.concatenate(out);assert np.isfinite(q).all()
        return q
    specs=[('baseline',False,1,False),('paired_fp32',True,1,False),('paired_batch2',True,2,False),('paired_bf16',True,1,True)]
    rows=[];refs={};hashes={}
    for y,x in POSITIONS:
        old,_=prepare(y,x,False);new,used=prepare(y,x,True)
        assert all(np.array_equal(o.view(np.uint32),n.view(np.uint32)) for o,n in zip(old,new))
        hashes[str((y,x))]={'shared':used,'input_sha256':[hashlib.sha256(o.tobytes()).hexdigest() for o in old]}
        refs[y,x]=forward(old,1,False)
    failed={}
    for rep in range(2):
        for i in np.roll(np.arange(len(specs)),rep):
            name,shared,batch,bf16=specs[i]
            if name in failed:continue
            for y,x in POSITIONS:
                t=time.perf_counter()
                try:
                    inp,used=prepare(y,x,shared);pre=time.perf_counter()-t;t2=time.perf_counter();q=forward(inp,batch,bf16);fwd=time.perf_counter()-t2
                    base=refs[y,x];err=np.abs(q-base);changed=int(((q>=.5)!=(base>=.5)).sum());u8=int((np.clip(q*255,0,255).astype(np.uint8)!=np.clip(base*255,0,255).astype(np.uint8)).sum())
                    rec=dict(method=name,rep=rep,y=y,x=x,shared=used,pre_seconds=pre,forward_seconds=fwd,total_seconds=pre+fwd,max_error=float(err.max()),mean_error=float(err.mean()),changed_threshold=changed,changed_u8=u8,n=int(q.size),bitwise=bool(np.array_equal(q.view(np.uint32),base.view(np.uint32))),output_sha256=hashlib.sha256(q.tobytes()).hexdigest())
                    if name=='paired_fp32':assert rec['bitwise']
                    rows.append(rec);print('B18_READER_ROW',json.dumps(rec),flush=True)
                except Exception as e:
                    if name in ('baseline','paired_fp32'):raise
                    failed[name]=str(e);break
    med={}
    for name,*_ in specs:
        r=[t for t in rows if t['method']==name]
        if len(r)==4:
            med[name]={'median_seconds':float(np.median([t['total_seconds'] for t in r])),'max_error':max(t['max_error'] for t in r),'changed_threshold_fraction':sum(t['changed_threshold'] for t in r)/sum(t['n'] for t in r)}
    result=dict(schema='b18.reader_network_benchmark.v1',checkpoint_sha256=PIN,weights=selected,torch=torch.__version__,numpy=np.__version__,cpu_threads=4,load_seconds=load_seconds,input_hashes=hashes,rows=rows,failed=failed,summary=med,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Native W00 unlabelled fixture; preparation+forward, no full CLI, no ink accuracy or transfer claim.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('B18_READER_DONE',json.dumps(med),flush=True)

if __name__=='__main__':main()
