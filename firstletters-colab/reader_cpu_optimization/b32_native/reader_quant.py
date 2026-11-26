"""Actual Reader v3 selective quantized kernels and memory-layout CPU comparison."""
import argparse,copy,hashlib,json,sys,time,resource
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'b18_paired'))
from reader_1447 import load_model,metrics,threshold,sha
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.image_proc.intensity.normalization import normalize_robust
PINS={'1447':'0f529972da360305aaad80d3400f209230427aa4ad9ad45b8df0f5807e4c9fb4','841':'926a284be12043aff107107c97b04a0af472e4e37c27a5277131891bad7d6357'}
def prepare(raw,reverse):
    r=raw[2:19];r=r[::-1] if reverse else r
    hwz=np.moveaxis(r,0,2).astype(np.float32,order='C')
    a=np.ascontiguousarray(normalize_robust(np.moveaxis(hwz,2,0)))
    return torch.from_numpy(a[None,None])
def forward(model,cfg,x):
    q=model(center_pad_input_depth(x,cfg.model.input_pad_depth_to))['ink']
    if isinstance(q,(tuple,list)):q=q[0]
    q=q.float().sigmoid()
    if q.shape[-2:]!=(384,384):q=F.interpolate(q,size=(384,384),mode='bilinear',align_corners=False)
    assert q.shape==(1,1,384,384)
    return q[0,0,96:288,96:288].contiguous().cpu().numpy()
def affine(lo,hi):
    lo=min(float(lo),0.);hi=max(float(hi),0.);scale=max((hi-lo)/255,1e-8)
    return scale,max(0,min(255,int(round(-lo/scale))))
class QConv(nn.Module):
    def __init__(self,m,stats):
        super().__init__()
        self.s,self.z=affine(*stats[:2]);os,oz=affine(*stats[2:])
        self.q=nn.quantized.Conv2d(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,padding_mode='zeros')
        w=m.weight.detach();sc=w.abs().flatten(1).amax(1).clamp_min(1e-8)/127
        qw=torch.quantize_per_channel(w,sc.double(),torch.zeros(len(sc),dtype=torch.int64),0,torch.qint8)
        self.q.set_weight_bias(qw,m.bias.detach() if m.bias is not None else None)
        self.q.scale=os;self.q.zero_point=oz
    def forward(self,x):
        return self.q(torch.quantize_per_tensor(x,self.s,self.z,torch.quint8)).dequantize()
def supported(m):
    return type(m) is nn.Conv2d and m.padding_mode=='zeros' and not isinstance(m.padding,str)
def quantize(model,cfg,xs):
    mods={n:m for n,m in model.named_modules() if supported(m)};stats={n:[float('inf'),-float('inf'),float('inf'),-float('inf')] for n in mods}
    def hook(name):
        def h(m,x,y):
            lo,hi=torch.aminmax(x[0]);ol,oh=torch.aminmax(y)
            v=stats[name];v[0]=min(v[0],float(lo));v[1]=max(v[1],float(hi));v[2]=min(v[2],float(ol));v[3]=max(v[3],float(oh))
        return h
    hooks=[m.register_forward_hook(hook(n)) for n,m in mods.items()]
    try:
        for x in xs:forward(model,cfg,x)
    finally:
        for h in hooks:h.remove()
    new=copy.deepcopy(model);replaced=[];skipped=[]
    for n,m in mods.items():
        if not np.isfinite(stats[n]).all():skipped.append([n,'unused']);continue
        parent,key=n.rsplit('.',1);sub=new.get_submodule(parent)
        try:setattr(sub,key,QConv(m,stats[n]));replaced.append([n,int(m.weight.numel())])
        except Exception as e:skipped.append([n,str(e)])
    assert replaced
    return new.eval(),dict(replaced=replaced,skipped=skipped,eligible=len(mods),quantized_weights=sum(x[1] for x in replaced),float_parameters=sum(p.numel() for p in model.parameters()),engine=torch.backends.quantized.engine,activation_stats=stats)
def main():
    a=argparse.ArgumentParser();a.add_argument('p1447',type=Path);a.add_argument('p841',type=Path);a.add_argument('checkpoint',type=Path);a.add_argument('out',type=Path);a=a.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high');torch.backends.quantized.engine='x86'
    for f,k in [(a.p1447,'1447'),(a.p841,'841')]:assert sha(f)==PINS[k]
    with np.load(a.p1447,allow_pickle=False) as z:raw=z['raw'];s=z['states'];names=z['names'];assert raw.shape==(12,21,384,384)
    with np.load(a.p841,allow_pickle=False) as z:r2=z['raw'];s2=z['states'];assert r2.shape==(3,21,384,384)
    tr=np.flatnonzero(names=='w058');te=np.flatnonzero(names=='w060');assert len(tr)==len(te)==6
    train=[prepare(raw[i],True) for i in tr];groups=[('w060',[prepare(raw[i],True) for i in te],s[te]),('0841_ascending',[prepare(x,False) for x in r2],s2),('0841_descending',[prepare(x,True) for x in r2],s2)]
    print('B32_LOAD',flush=True);base,cfg,weights=load_model(a.checkpoint);print('B32_MODEL_LOADED',weights,flush=True)
    rows=[];errors={};preds={};times=[];qreceipt=None
    with torch.inference_mode():
        trainp=np.stack([forward(base,cfg,x) for x in train]);th=threshold(trainp,s[tr])
        print('B32_CAL',json.dumps(dict(threshold=th,source='w058',n=len(train))),flush=True)
        models={'fp32':base}
        try:
            cl=copy.deepcopy(base)
            nn.utils.convert_conv2d_weight_memory_format(cl,torch.channels_last)
            nn.utils.convert_conv3d_weight_memory_format(cl,torch.channels_last_3d)
            models['channels_last']=cl.eval()
        except Exception as e:errors['channels_last']=repr(e)
        try:
            models['int8_conv2d'],qreceipt=quantize(base,cfg,train)
            print('B32_QUANT',json.dumps({k:v for k,v in qreceipt.items() if k not in ('activation_stats','replaced')}),flush=True)
        except Exception as e:errors['int8_conv2d']=repr(e);print('B32_QUANT_FAILURE',repr(e),flush=True)
        def layout(name,x):return x.contiguous(memory_format=torch.channels_last_3d) if name=='channels_last' else x
        for name,m in list(models.items()):
            try:
                for _ in range(2):forward(m,cfg,layout(name,train[0]))
            except Exception as e:
                if name=='fp32':raise
                errors[name]=repr(e);del models[name];print('B32_WARM_FAILURE',name,repr(e),flush=True)
        for g,xx,ss in groups:
            out={name:[] for name in models};rep0={}
            for i,x in enumerate(xx):
                ref=None
                for rep in range(2):
                    for name in np.roll(list(models),i+rep):
                        name=str(name);inp=layout(name,x);t=time.perf_counter();q=forward(models[name],cfg,inp);dt=time.perf_counter()-t
                        assert np.isfinite(q).all()
                        if rep==0:rep0[name]=q;out[name].append(q)
                        else:assert np.array_equal(q,rep0[name]),(name,'repeat mismatch')
                        times.append(dict(group=g,crop=i,rep=rep,method=name,seconds=dt))
                print('B32_CROP',g,i,flush=True)
            z={k:np.stack(v) for k,v in out.items()};blank=(ss==2).sum((1,2))==0
            for name,p in z.items():
                diff=np.abs(p-z['fp32'])
                row=dict(group=g,method=name,held=metrics(p,ss,th,blank),fixed=metrics(p,ss,.5,blank),max_error=float(diff.max()),mean_error=float(diff.mean()),bitwise=bool(np.array_equal(p,z['fp32'])),changed_at_transfer=int(((p>=th)!=(z['fp32']>=th)).sum()),changed_at_half=int(((p>=.5)!=(z['fp32']>=.5)).sum()),n=int(p.size))
                rows.append(row);print('B32_ROW',json.dumps(row),flush=True);preds[g+'_'+name]=p
            np.savez_compressed(a.out/(g+'.npz'),states=ss,**z)
        med={m:float(np.median([r['seconds'] for r in times if r['method']==m])) for m in models}
        gates={}
        for m in models:
            if m=='fp32':continue
            diffs=[]
            for g,*_ in groups:
                b=next(r for r in rows if r['group']==g and r['method']=='fp32');v=next(r for r in rows if r['group']==g and r['method']==m)
                for key in ('held','fixed'):
                    x,y=b[key],v[key];diffs.append(dict(group=g,threshold=key,ap_delta=y['ap']-x['ap'],recall_delta=y['recall']-x['recall'],fpr_delta=y['fpr']-x['fpr']))
            gates[m]=dict(speedup=med['fp32']/med[m],diffs=diffs,pass_=bool(med['fp32']/med[m]>=1.25 and all(d['ap_delta']>=-.01 and abs(d['recall_delta'])<=.02 and abs(d['fpr_delta'])<=.02 for d in diffs)))
        result=dict(schema='b32.actual_reader_quant.v1',weights=weights,torch=torch.__version__,threads=1,source_threshold=th,rows=rows,times=times,median_seconds=med,gates=gates,failures=errors,quantization=qreceipt,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Public development controls; selective quantized Conv2D only, float residual/norm/stem; forward timing not full CLI.')
        (a.out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('B32_DONE',json.dumps(dict(median_seconds=med,gates=gates,failures=errors,peak_rss_kib=result['peak_rss_kib'])),flush=True)
if __name__=='__main__':main()
