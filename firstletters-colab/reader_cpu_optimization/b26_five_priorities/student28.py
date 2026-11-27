"""B28: local compact 2.5D student, real Reader v3 teacher, pruning + static int8."""
import copy,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.ao import quantization as Q
from sklearn.metrics import average_precision_score
from cascade import PRED,sha,threshold,metric
from fast_mad import reference
PINS={'panel1447/panel.npz':'0f529972da360305aaad80d3400f209230427aa4ad9ad45b8df0f5807e4c9fb4','panel841/panel.npz':'926a284be12043aff107107c97b04a0af472e4e37c27a5277131891bad7d6357','pred841/predictions.npz':'2fb79c53bc83c4d008a9bcffaad7fd788691140e24c7012440b61553ffb8c691'}

class Student(nn.Module):
    def __init__(self,width=16):
        super().__init__();self.quant=Q.QuantStub();self.a=nn.Conv2d(17,width,3,padding=1);self.ar=nn.ReLU();self.b=nn.Conv2d(width,width,3,padding=1);self.br=nn.ReLU();self.c=nn.Conv2d(width,1,1);self.dequant=Q.DeQuantStub()
    def forward(self,x):return self.dequant(self.c(self.br(self.b(self.ar(self.a(self.quant(x)))))))

def normalize(raw,reverse):
    # Match upstream HWZ layout before its float32 normalization.
    result=[]
    for v in raw:
        a=v[2:19];a=a[::-1] if reverse else a
        hwz=np.moveaxis(a,0,2).astype(np.float32,order='C');result.append(reference(np.moveaxis(hwz,2,0)))
    return torch.from_numpy(np.stack(result))

def read_panel(d,name):
    f=d/name/'panel.npz';assert sha(f)==PINS[name+'/panel.npz'];z=np.load(f,allow_pickle=False);p=json.loads((d/name/'prepare.json').read_text());raw=z['raw'];s=z['states'];orig=z['origins_yx'];names=z['names']
    h=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
    if name=='panel841':
        assert [h(x) for x in raw]==[r['raw_sha256'] for r in p['prepared']];assert [h(x) for x in s]==[r['states_sha256'] for r in p['prepared']]
    else:assert [h(x) for x in raw]==p['raw_sha256'] and [h(x) for x in s]==p['states_sha256']
    return raw,s,orig,names

def train(x,s,t,tr,kind,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=Student();opt=torch.optim.Adam(m.parameters(),lr=.003);losses=[];known_steps=0
    x=x[:,:,94:290,94:290];s=torch.from_numpy(s);t=torch.from_numpy(t);start=time.perf_counter();m.train()
    for step in range(80):
        a=[];y=[];q=[]
        for _ in range(4):
            i=int(rng.choice(tr));j,k=rng.integers(0,145,size=2);a.append(x[i,:,j:j+52,k:k+52]);y.append(s[i,j:j+48,k:k+48]);q.append(t[i,j:j+48,k:k+48])
        inp=torch.stack(a);lab=torch.stack(y);teacher=torch.stack(q);known=lab>0;logit=m(inp)[:,0,2:50,2:50];kl=F.binary_cross_entropy_with_logits(logit,teacher)
        if known.any():
            sup=F.binary_cross_entropy_with_logits(logit[known],(lab[known]==2).float());known_steps+=1
            loss=sup if kind=='label' else .5*(sup+kl)
        elif kind=='distill':loss=kl
        else:continue
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
    assert losses and all(np.isfinite(losses));m.eval();return m,dict(seconds=time.perf_counter()-start,steps=len(losses),known_steps=known_steps,final_loss=float(np.mean(losses[-10:])))

def prune(m):
    n=Student(12).eval();i=m.a.weight.abs().sum((1,2,3)).topk(12).indices.sort().values;j=m.b.weight.abs().sum((1,2,3)).topk(12).indices.sort().values
    with torch.no_grad():
        n.a.weight.copy_(m.a.weight[i]);n.a.bias.copy_(m.a.bias[i]);n.b.weight.copy_(m.b.weight[j][:,i]);n.b.bias.copy_(m.b.bias[j]);n.c.weight.copy_(m.c.weight[:,j]);n.c.bias.copy_(m.c.bias)
    return n

def quantize(m,x,tr):
    q=copy.deepcopy(m).eval();Q.fuse_modules(q,[['a','ar'],['b','br']],inplace=True);q.qconfig=Q.get_default_qconfig('x86');Q.prepare(q,inplace=True)
    with torch.inference_mode():
        for i in tr:q(x[i:i+1,:,94:290,94:290])
    Q.convert(q,inplace=True);assert q.a.weight().dtype==torch.qint8 and q.b.weight().dtype==torch.qint8 and q.c.weight().dtype==torch.qint8
    return q

def predict(m,x):
    with torch.inference_mode():return np.stack([m(v[None])[:,0,96:288,96:288].sigmoid().numpy()[0] for v in x])

def latency(m,x):
    with torch.inference_mode():
        for _ in range(2):m(x[:1])
        t=[]
        for _ in range(5):a=time.perf_counter();m(x[:1]);t.append(time.perf_counter()-a)
    return float(np.median(t))

def summarize(q,s,teacher,t,blank,ft=None):
    out=metric(q,s,t,blank);out.update(teacher_mae=float(np.abs(q-teacher).mean()),teacher_max_error=float(np.abs(q-teacher).max()),fixed_0_5=metric(q,s,.5,blank))
    if ft is not None:out['float_threshold_metrics']=metric(q,s,ft,blank)
    aps=[]
    for i in range(len(s)):
        mask=s[i]>0;z=s[i][mask]==2;aps.append(float(average_precision_score(z,q[i][mask])) if z.any() and not z.all() else None)
    out['per_crop_AP']=aps;return out

def main():
    t0=time.perf_counter();d=Path(sys.argv[1]);out=Path(sys.argv[2]);source=Path(sys.argv[3]);assert sha(source)==PRED;assert sha(d/'pred841/predictions.npz')==PINS['pred841/predictions.npz'];torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.quantized.engine='x86';warnings.filterwarnings('ignore',category=DeprecationWarning);warnings.filterwarnings('ignore',category=UserWarning)
    raw,s,orig,names=read_panel(d,'panel1447');v=np.load(source,allow_pickle=False);assert np.array_equal(v['states'],s) and np.array_equal(v['origins_yx'],orig) and np.array_equal(v['windings'],names);teach=v['probabilities'][0];blank=v['blank'];x=normalize(raw,True)
    raw2,s2,orig2,names2=read_panel(d,'panel841');v2=np.load(d/'pred841/predictions.npz',allow_pickle=False);assert np.array_equal(v2['states'],s2) and np.array_equal(v2['origins_yx'],orig2);target=[normalize(raw2,r) for r in (False,True)];t2=v2['probabilities'][:,0];b2=v2['blank'];assert v2['directions'].tolist()==['ascending','descending']
    rows=[];fits=[];fails=[];baseline=[];models={}
    for fold,(tr,ca,held) in enumerate([(np.array([0,1,2,4]),np.array([3,5]),[("w060",x[6:],s[6:],teach[6:],blank[6:])]),(np.array([6,7,8,10]),np.array([9,11]),[("w058",x[:6],s[:6],teach[:6],blank[:6])]),(np.array([0,1,2,4,6,7,8,10]),np.array([3,5,9,11]),[("0841_asc",target[0],s2,t2[0],b2),("0841_desc",target[1],s2,t2[1],b2)])]):
        assert not np.intersect1d(tr,ca).size;tt=threshold(teach[ca],s[ca]);
        for name,xx,ss,teacher,bb in held:baseline.append(dict(fold=fold,target=name,threshold=tt,**metric(teacher,ss,tt,bb)))
        for kind in ('label','distill'):
            m,f=train(x,s,teach,tr,kind,2800+fold);f.update(fold=fold,kind=kind,train=tr.tolist(),cal=ca.tolist());fits.append(f);models[f'{fold}_{kind}']={k:v.detach().cpu().numpy().tolist() for k,v in m.state_dict().items()};modes={'float':m,'pruned':prune(m)}
            for key,base in list(modes.items()):
                try:modes[key+'_int8']=quantize(base,x,tr)
                except Exception as e:fails.append(dict(fold=fold,kind=kind,mode=key+'_int8',error=str(e)))
            torch.set_num_threads(1);float_t={};float_q={}
            for mode,mm in modes.items():
                width=12 if mode.startswith('pruned') else 16;params=(17*width*9+width)+(width*width*9+width)+(width+1);macs=(17*width*9+width*width*9+width)*384*384
                cv=predict(mm,x[ca]);th=threshold(cv,s[ca]);cal=metric(cv,s[ca],th,blank[ca]);
                if 'int8' not in mode:float_t[mode]=th
                for name,xx,ss,teacher,bb in held:
                    q=predict(mm,xx);base_mode=mode.replace('_int8','');drift=None
                    if 'int8' in mode:
                        ref=float_q[(name,base_mode)];e=np.abs(q-ref);drift=dict(mae=float(e.mean()),max=float(e.max()),changed_fixed_0_5=int(((q>=.5)!=(ref>=.5)).sum()),changed_float_threshold=int(((q>=float_t[base_mode])!=(ref>=float_t[base_mode])).sum()),n=int(q.size))
                    else:float_q[name,mode]=q
                    rr=dict(fold=fold,target=name,kind=kind,mode=mode,params=params,macs=macs,threshold=th,calibration_recall=cal['recall'],calibration_fpr=cal['fpr'],latency_seconds=latency(mm,xx),int8_drift=drift,**summarize(q,ss,teacher,th,bb,float_t[base_mode] if 'int8' in mode else None));rows.append(rr);print('B28',kind,mode,name,round(rr['recall'],4),round(rr['fpr'],4),round(rr['ap'],4),round(rr['latency_seconds'],5),flush=True)
            torch.set_num_threads(2)
    gate={}
    for k in ('label','distill'):
        for mode in ('float','pruned','float_int8','pruned_int8'):
            rr=[r for r in rows if r['kind']==k and r['mode']==mode];g=[]
            for r in rr:
                base=next(v for v in baseline if v['target']==r['target']);g.append(dict(target=r['target'],pass_=bool(r['recall']>=.6 and r['fpr']<=base['fpr'] and r['ap']>=base['ap']-.05)))
            gate[k+'_'+mode]=dict(pass_=len(g)==4 and all(r['pass_'] for r in g),rows=g)
    res=dict(schema='b28.reader_student.v1',pins=PINS,teacher_sha256=PRED,torch=torch.__version__,numpy=np.__version__,fits=fits,baseline=baseline,rows=rows,failures=fails,gates=gate,wall_seconds=time.perf_counter()-t0,scope='Local student trained on actual Reader scores, limited 1447 development panel; 0841 distinct-parent exposed test. Not actual v4 or full-model quantization.')
    out.write_text(json.dumps(res,separators=(',',':'))+'\n');(out.parent/'student_weights.json').write_text(json.dumps(models,separators=(',',':'))+'\n');print('B28_DONE',json.dumps({'wall_seconds':res['wall_seconds'],'fits':len(fits),'rows':len(rows),'failures':fails,'gates':gate}),flush=True)
if __name__=='__main__':main()
