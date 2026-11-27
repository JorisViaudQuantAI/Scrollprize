"""B22: exact native render and fixed cross-parent Reader calibration."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'firstletters-colab/reader_cpu_optimization/b18_paired'))
from reader_1447 import load_model,predict,metrics,threshold,sha
sys.path.insert(0,str(ROOT/'firstletters-colab/studio_oblique_stress'))
from assets import acquire_geometry
from native import load_surface,SupportedNative
sys.path.insert(0,str(ROOT/'firstletters-colab/studio_control'))
from run_public_control import FILES,BASE,verified_file

PANEL=((480,1248,0,4893),(480,1632,4049,5709),(864,1632,11005,18119))
DIRS=('ascending','descending')
ARMS=('center','minus2','plus2','bf16_center','bf16_minus2','bf16_plus2')
PINS={'assets.py':'cac664fdafe3f982e84354281601dd42137dc294','native.py':'16ce7ac2f5d447e5378f360251a1689a4926ce18'}
CAL=Path(__file__).with_name('calibration.json')


def check_blob(path,expected):
    b=path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=expected:
        raise ValueError('unexpected upstream helper '+str(path))


def prepare(args):
    cache=args.cache;cache.mkdir(parents=True,exist_ok=True)
    for n,(h,size) in FILES.items():verified_file(BASE+n,cache/n,h,size)
    l=np.load(cache/'labels.npy',allow_pickle=False)
    ct=np.load(cache/'ct-tiles.npy',mmap_mode='r',allow_pickle=False)
    tiles=np.load(cache/'tiles.npy',allow_pickle=False)
    assert l.shape==(2773,2609) and ct.shape==(28,512,512,16)
    for n,h in PINS.items():check_blob(ROOT/'firstletters-colab/studio_oblique_stress'/n,h)
    check_blob(ROOT/'gemini_lane/stream_snap_render.py','042fee4984205615b39d669c6f01500ecdff74a3')
    panel=[]
    for y in range(96,l.shape[0]-383,384):
        for x in range(96,l.shape[1]-383,384):
            s=l[y+96:y+288,x+96:x+288]
            p,n=int((s==2).sum()),int((s==1).sum())
            if (p>=512 and n>=512) or (p==0 and n>=4096):panel.append((y,x,p,n))
    assert tuple(panel)==PANEL
    mesh,recipe,prov=acquire_geometry(cache/'geometry')
    surf=load_surface(args.author,mesh,recipe)
    volume=SupportedNative(recipe['url'],cache/'native',byte_budget=512*1024**2)
    assert list(volume.shape)==[19400,7703,7703] and list(volume.chunks)==[128,128,128]
    out=[];fail=[]
    for y,x,p,n in PANEL:
        try:
            raw=np.empty((21,384,384),np.uint8)
            for j in range(0,384,64):
                xyz,normal,valid=surf.points(np.arange(y+j,y+j+64),np.arange(x,x+384),9.36)
                if not valid.all():raise ValueError('invalid full-context geometry at strip '+str(j))
                offsets=np.asarray(np.arange(-10,11,dtype=np.float64)*float(surf.ct_um),dtype=np.float32)/surf.ct_um
                coords=xyz[None]+normal[None]*offsets[:,None,None,None]
                raw[:,j:j+64]=np.rint(volume.sample_exact(coords)).clip(0,255).astype(np.uint8)
            # Source-author prepared samples must agree on every covered pixel.
            ncheck=0;bad=0;covered=np.zeros((384,384),np.uint8)
            for tile,(ty,tx) in zip(ct,tiles):
                ya,yb=max(y,ty),min(y+384,ty+512);xa,xb=max(x,tx),min(x+384,tx+512)
                if ya>=yb or xa>=xb:continue
                observed=raw[2:18,ya-y:yb-y,xa-x:xb-x].transpose(1,2,0)
                ref=tile[ya-ty:yb-ty,xa-tx:xb-tx];bad+=int((observed!=ref).sum());ncheck+=int(ref.size)
                covered[ya-y:yb-y,xa-x:xb-x]+=1
            s=l[y+96:y+288,x+96:x+288].copy()
            if ncheck==0 or bad or (covered>1).any() or not (covered[96:288,96:288][s>0]==1).all():
                raise ValueError(f'identity reconstruction failed: {bad}/{ncheck} mismatched')
            rec=dict(y=y,x=x,ink=p,bg=n,identity_values=ncheck,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),states_sha256=hashlib.sha256(s.tobytes()).hexdigest())
            out.append((rec,raw,s));print('B22_PREPARED',json.dumps(rec),flush=True)
        except Exception as e:
            fail.append(dict(y=y,x=x,error=str(e)));print('B22_PREP_FAILURE',json.dumps(fail[-1]),flush=True)
    receipt=dict(planned=PANEL,prepared=[v[0] for v in out],failures=fail,geometry=prov,native=volume.receipt(),calibration_sha256=sha(CAL))
    (args.out/'prepare.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('B22_PREP_DONE',json.dumps({'prepared':len(out),'failures':fail,'native_bytes':volume.new_http_bytes,'receipt_sha256':sha(args.out/'prepare.json')}),flush=True)
    if len(out)!=len(PANEL):raise RuntimeError('frozen panel incomplete; no model inference')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('author',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high')
    cal=json.loads(CAL.read_text());th=cal['thresholds'];assert cal['sha256']=='baf48b9301c9697c0dd8f4efbb7ffde9c2f891305f08d39194323eb860bd2d81'
    panel=prepare(a);model,cfg,weights=load_model(a.checkpoint);pred=[];states=[];times=[]
    for rec,raw,s in panel:
        pp=[]
        for sign in DIRS:
            q=[]
            for arm,lo,amp in zip(ARMS,(2,0,4,2,0,4),(False,False,False,True,True,True)):
                band=raw[lo:lo+17];band=band if sign=='ascending' else band[::-1]
                t=time.perf_counter();v=predict(model,cfg,band,amp);elapsed=time.perf_counter()-t
                q.append(v);times.append(dict(y=rec['y'],x=rec['x'],direction=sign,arm=arm,seconds=elapsed))
            pp.append(np.stack(q))
        pred.append(np.stack(pp));states.append(s)
    p=np.stack(pred,axis=2);s=np.stack(states);blank=np.array([r[0]['ink']==0 for r in panel]);rows=[];errors=[]
    np.savez_compressed(a.out/'predictions.npz',probabilities=p,states=s,blank=blank,directions=np.array(DIRS),arms=np.array(ARMS),origins_yx=np.array([[r[0]['y'],r[0]['x']] for r in panel]))
    for i,sign in enumerate(DIRS):
        q=p[i];center=q[0];median=np.median(q[:3],axis=0);bm=np.median(q[3:],axis=0);opp=p[1-i,0]
        methods=dict(center=center,median3=median,contrast=center-opp,product=center*(1-opp),bf16_center=q[3],bf16_median3=bm)
        for name,z in methods.items():
            ref={'bf16_center':'center','bf16_median3':'median3'}.get(name,name)
            t=th[ref];oracle=threshold(z,s)
            row=dict(direction=sign,method=name,threshold=t,held=metrics(z,s,t,blank),oracle=metrics(z,s,oracle,blank),fixed_0_5=metrics(z,s,.5,blank) if name in ('center','median3','bf16_center','bf16_median3') else None)
            rows.append(row);print('B22_ROW',json.dumps(row),flush=True)
        for name,x,y,t in [('center',center,q[3],th['center']),('median3',median,bm,th['median3'])]:
            e=np.abs(x-y);errors.append(dict(direction=sign,method=name,max_error=float(e.max()),mean_error=float(e.mean()),changed_fixed_0_5=int(((x>=.5)!=(y>=.5)).sum()),changed_transferred=int(((x>=t)!=(y>=t)).sum()),n=int(e.size)))
    gates={}
    for sign in DIRS:
        a0=next(r['held'] for r in rows if r['direction']==sign and r['method']=='center');m=next(r['held'] for r in rows if r['direction']==sign and r['method']=='median3')
        dr=100*(m['recall']-a0['recall']);df=100*(m['fpr']-a0['fpr'])
        gates[sign]=dict(delta_recall_pp=dr,delta_fpr_pp=df,pass_=bool(dr>=-2 and df<=-2 and m['recall']>=.6))
    result=dict(schema='b22.reader_cross_parent.v1',rows=rows,forwards=len(times),times=times,gates=gates,errors=errors,weights=weights,prediction_sha256=sha(a.out/'predictions.npz'),scope='Distinct parent transfer, exposed donor labels and unverified orientation, no independent blind manuscript validation.')
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('B22_DONE',json.dumps(dict(forwards=len(times),gates=gates,errors=errors,prediction_sha256=result['prediction_sha256'],median_seconds={arm:float(np.median([x['seconds'] for x in times if x['arm']==arm])) for arm in ARMS})),flush=True)

if __name__=='__main__':main()
