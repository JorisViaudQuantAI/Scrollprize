"""B36: actual contextual student fits and selective INT8 CPU kernels."""
import argparse,copy,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from panels import load
from fast_mad import reference
from transfer35 import pos_threshold,counts
class Block(nn.Sequential):
    def __init__(self,a,b,stride=1):super().__init__(nn.Conv2d(a,b,3,stride,1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU())
class Student(nn.Module):
    def __init__(self):
        super().__init__();self.a=Block(17,16);self.b=Block(16,24,2);self.c=Block(24,32,2);self.d=Block(32+24,24);self.e=Block(24+16,16);self.o=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(a);c=self.c(b);d=self.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='nearest'),b),1));e=self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='nearest'),a),1));return self.o(e)
class QConv(nn.Module):
    def __init__(self,m,stat):
        super().__init__();self.sc,self.z=self.affine(stat[:2]);so,zo=self.affine(stat[2:]);self.q=nn.quantized.Conv2d(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None)
        w=m.weight.detach();sc=w.abs().flatten(1).amax(1).clamp_min(1e-8)/127;qw=torch.quantize_per_channel(w,sc.double(),torch.zeros(len(sc),dtype=torch.int64),0,torch.qint8);self.q.set_weight_bias(qw,m.bias.detach() if m.bias is not None else None);self.q.scale=so;self.q.zero_point=zo
    @staticmethod
    def affine(stat):
        lo=min(0.,stat[0]);hi=max(0.,stat[1]);sc=max((hi-lo)/255,1e-8);return sc,max(0,min(255,round(-lo/sc)))
    def forward(self,x):return self.q(torch.quantize_per_tensor(x,self.sc,self.z,torch.quint8)).dequantize()
def norm(raw,reverse):
    return torch.from_numpy(np.stack([reference(np.moveaxis(np.moveaxis(v[2:19][::-1] if reverse else v[2:19],0,2).astype(np.float32,order='C'),2,0)) for v in raw]))
def train(x,s,teacher,tr,kind,seed):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed);m=Student();opt=torch.optim.AdamW(m.parameters(),lr=.002);s=torch.from_numpy(s);teacher=torch.from_numpy(teacher);losses=[];start=time.perf_counter();m.train()
    for step in range(192):
        a=[];y=[];q=[]
        for _ in range(2):
            for trial in range(1000):
                i=int(rng.choice(tr));j,k=map(int,rng.integers(129,size=2));v=s[i,j:j+64,k:k+64]
                if (v>0).any():break
            else:raise RuntimeError('no known training sample')
            a.append(x[i,:,64+j:192+j,64+k:192+k]);y.append(v);q.append(teacher[i,j:j+64,k:k+64])
        lab=torch.stack(y);teach=torch.stack(q);known=lab>0;logits=m(torch.stack(a))[:,0,32:96,32:96];hard=F.binary_cross_entropy_with_logits(logits[known],(lab[known]==2).float());soft=F.binary_cross_entropy_with_logits(logits,teach);loss=hard if kind=='label' else .5*(hard+soft);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
    assert len(losses)==192 and np.isfinite(losses).all();return m.eval(),dict(steps=len(losses),seconds=time.perf_counter()-start,final_loss=float(np.mean(losses[-16:])))
def quantize(m,x,tr):
    mods={k:v for k,v in m.named_modules() if type(v) is nn.Conv2d};stats={k:[float('inf'),-float('inf'),float('inf'),-float('inf')] for k in mods}
    def hook(k):
        def h(module,ins,out):
            a,b=torch.aminmax(ins[0]);c,d=torch.aminmax(out);v=stats[k];v[:]=[min(v[0],float(a)),max(v[1],float(b)),min(v[2],float(c)),max(v[3],float(d))]
        return h
    hs=[v.register_forward_hook(hook(k)) for k,v in mods.items()]
    try:
        with torch.inference_mode():
            for i in tr:m(x[i:i+1])
    finally:
        for h in hs:h.remove()
    q=copy.deepcopy(m)
    for k,v in mods.items():
        p,key=k.rsplit('.',1) if '.' in k else ('',k);setattr(q.get_submodule(p),key,QConv(v,stats[k]))
    return q.eval(),len(mods)
def predict(m,x):
    with torch.inference_mode():return np.stack([m(a[None])[0,0,96:288,96:288].sigmoid().numpy() for a in x])
def latency(m,x):
    with torch.inference_mode():
        for _ in range(2):m(x[:1])
        ts=[]
        for _ in range(5):t=time.perf_counter();m(x[:1]);ts.append(time.perf_counter()-t)
    return float(np.median(ts))
def score(p,s,t,teacher):
    r=counts(p>=t,s);k=s>0;r.update(ap=float(average_precision_score(s[k]==2,p[k])),teacher_mae=float(np.abs(p-teacher).mean()),threshold=t);return r
def run(d,out):
    tic=time.perf_counter();torch.set_num_interop_threads(1);torch.set_num_threads(1);torch.backends.quantized.engine='x86';warnings.filterwarnings('ignore',category=DeprecationWarning);a,b,p,q=load(d);x=norm(a['raw'],True);x2=[norm(b['raw'],v) for v in (False,True)];s=a['states'];s2=b['states'];teacher=p['probabilities'][0];teacher2=q['probabilities'][:,0];w=a['names'];assert list(w)==['w058']*6+['w060']*6
    folds=[(np.array([0,1,2,4]),np.array([3,5]),['w060']),(np.array([6,7,8,10]),np.array([9,11]),['w058']),(np.array([0,1,2,4,6,7,8,10]),np.array([3,5,9,11]),['0841_ascending','0841_descending'])];rows=[];fits=[];gates=[];failures=[];out.mkdir(exist_ok=True)
    for fold,(tr,cal,held) in enumerate(folds):
        assert not np.intersect1d(tr,cal).size;tt=pos_threshold(teacher[cal],s[cal],4,5)
        groups=[]
        for target in held:
            if target.startswith('0841'):
                j=0 if target.endswith('ascending') else 1;groups.append((target,x2[j],s2,teacher2[j]))
            else:
                te=np.flatnonzero(w==target);assert not np.intersect1d(tr,te).size and not np.intersect1d(cal,te).size;groups.append((target,x[te],s[te],teacher[te]))
        for kind in ('label','distill'):
            m,rec=train(x,s,teacher,tr,kind,3600+fold);torch.set_num_threads(1);rec.update(fold=fold,kind=kind,train=tr.tolist(),calibration=cal.tolist(),parameters=sum(p.numel() for p in m.parameters()));fits.append(rec);torch.save(m.state_dict(),out/f'fold{fold}_{kind}.pt');models={'fp32':m}
            try:models['int8'],n=quantize(m,x,tr);rec['quantized_conv2d']=n
            except Exception as e:failures.append(dict(fold=fold,kind=kind,error=repr(e)))
            pc=predict(m,x[cal]);th=pos_threshold(pc,s[cal],4,5)
            for target,xx,ss,tch in groups:
                ref=score(tch,ss,tt,tch)
                for arm,net in models.items():
                    v=predict(net,xx);calp=predict(net,x[cal]);t=pos_threshold(calp,s[cal],4,5);r=score(v,ss,t,tch);r0=score(v,ss,th,tch);row=dict(fold=fold,kind=kind,target=target,arm=arm,teacher=ref,student=r,fixed_float_threshold=r0,latency_seconds=latency(net,xx));row['pass']=bool(r['ap']>=ref['ap']-.02 and r['recall']>=ref['recall']-.02 and r['fpr']<=ref['fpr']+.02);rows.append(row);print('B36_ROW',json.dumps(row),flush=True)
            print('B36_FIT',json.dumps(rec),flush=True)
    return dict(schema='b36.context_student.v1',fits=fits,rows=rows,failures=failures,wall_seconds=time.perf_counter()-tic,scope='Actual contextual student from scratch; tiny exposed development panel, same parent calibration and one different parent test, no Reader equivalence or deployment.')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('out',type=Path);v=a.parse_args();r=run(v.data,v.out);(v.out/'results.json').write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');print('B36_DONE',json.dumps(dict(fits=len(r['fits']),rows=len(r['rows']),passes=sum(x['pass'] for x in r['rows']),seconds=r['wall_seconds'],failures=r['failures'])),flush=True)
