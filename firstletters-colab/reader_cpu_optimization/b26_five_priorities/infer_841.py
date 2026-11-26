"""B26: isolated Reader inference on B22 staged native data."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'b18_paired'))
from reader_1447 import load_model,predict,metrics,threshold,sha
CAL=HERE.parent/'b22_cross_parent'/'calibration.json'
DIRS=('ascending','descending')
ARMS=('center','minus2','plus2','bf16_center','bf16_minus2','bf16_plus2')
def load_panel(d):
    with np.load(d/'panel.npz',allow_pickle=False) as z:
        raw=z['raw'];s=z['states'];orig=z['origins_yx']
    receipt=json.loads((d/'prepare.json').read_text())
    assert raw.shape==(3,21,384,384) and s.shape==(3,192,192)
    panel=[]
    for j,r in enumerate(receipt['prepared']):
        assert orig[j].tolist()==[r['y'],r['x']]
        assert hashlib.sha256(raw[j].tobytes()).hexdigest()==r['raw_sha256']
        assert hashlib.sha256(s[j].tobytes()).hexdigest()==r['states_sha256']
        panel.append((r,raw[j],s[j]))
    return panel
def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('author',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high')
    cal=json.loads(CAL.read_text());th=cal['thresholds'];assert cal['sha256']=='baf48b9301c9697c0dd8f4efbb7ffde9c2f891305f08d39194323eb860bd2d81'
    panel=load_panel(a.cache);print('B26_MODEL_LOADING',flush=True);model,cfg,weights=load_model(a.checkpoint);print('B26_MODEL_LOADED',weights,flush=True);pred=[];states=[];times=[]
    for rec,raw,s in panel:
        pp=[]
        for sign in DIRS:
            q=[]
            for arm,lo,amp in zip(ARMS,(2,0,4,2,0,4),(False,False,False,True,True,True)):
                band=raw[lo:lo+17];band=band if sign=='ascending' else band[::-1]
                t=time.perf_counter();v=predict(model,cfg,band,amp);elapsed=time.perf_counter()-t
                q.append(v);times.append(dict(y=rec['y'],x=rec['x'],direction=sign,arm=arm,seconds=elapsed));print('B26_FORWARD',json.dumps(times[-1]),flush=True)
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
            rows.append(row);print('B26_ROW',json.dumps(row),flush=True)
        for name,x,y,t in [('center',center,q[3],th['center']),('median3',median,bm,th['median3'])]:
            e=np.abs(x-y);errors.append(dict(direction=sign,method=name,max_error=float(e.max()),mean_error=float(e.mean()),changed_fixed_0_5=int(((x>=.5)!=(y>=.5)).sum()),changed_transferred=int(((x>=t)!=(y>=t)).sum()),n=int(e.size)))
    gates={}
    for sign in DIRS:
        a0=next(r['held'] for r in rows if r['direction']==sign and r['method']=='center');m=next(r['held'] for r in rows if r['direction']==sign and r['method']=='median3')
        dr=100*(m['recall']-a0['recall']);df=100*(m['fpr']-a0['fpr'])
        gates[sign]=dict(delta_recall_pp=dr,delta_fpr_pp=df,pass_=bool(dr>=-2 and df<=-2 and m['recall']>=.6))
    result=dict(schema='b26.reader_cross_parent.v1',rows=rows,forwards=len(times),times=times,gates=gates,errors=errors,weights=weights,prediction_sha256=sha(a.out/'predictions.npz'),scope='Distinct parent transfer, exposed donor labels and unverified orientation, no independent blind manuscript validation.')
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('B26_DONE',json.dumps(dict(forwards=len(times),gates=gates,errors=errors,prediction_sha256=result['prediction_sha256'],median_seconds={arm:float(np.median([x['seconds'] for x in times if x['arm']==arm])) for arm in ARMS})),flush=True)

if __name__=='__main__':main()
