"""Public control A/B of one-load vs per-direction model construction."""
import argparse,json,time,sys,tempfile
from pathlib import Path
import numpy as np
import torch
import tifffile
from staged_rv3 import Burst,stage_arrays,load_stage,sha
PIN='926a284be12043aff107107c97b04a0af472e4e37c27a5277131891bad7d6357'
def main():
    p=argparse.ArgumentParser();p.add_argument('panel',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert sha(a.panel)==PIN
    with np.load(a.panel,allow_pickle=False) as z:raw=z['raw']
    assert raw.shape==(3,21,384,384)
    a.out.mkdir(parents=True,exist_ok=False);stages=[]
    # Fixed public contexts, use partial support to exercise weighting.
    for i in (0,1):
        mask=np.ones((384,384),bool);mask[:12,:]=False;mask[180:196,180:196]=False
        path=a.out/f'stage{i}';stage_arrays(path,raw[i,2:19],mask,{'panel_sha256':PIN,'index':i,'mask':'artificial support exclusion'});stages.append(path)
    runs={};receipts=[]
    for mode in ('reload','reuse'):
        obj=Burst(a.checkpoint,allow_cpu=True);rows=[];t=time.perf_counter()
        for i,s in enumerate(stages):
            x,m,_=load_stage(s);w=a.out/f'{mode}-{i}'
            v=obj.run(x,m,w,batch_sizes=(1,),reload_each=(mode=='reload'),keep_tiff=True)
            for direction,r in v.items():
                valid=r['valid'];q=r['prob'];u=tifffile.imread(w/f'{direction}-1.tif');assert np.array_equal(u[valid],(q[valid]*255).astype(np.uint8))
                assert not (valid&~m).any() and valid.any()
                rows.append(dict(context=i,direction=direction,valid=int(valid.sum()),seconds=r['seconds'],loads=obj.loads))
            runs[mode,i]=v
        receipts.append(dict(mode=mode,wall_seconds=time.perf_counter()-t,loads=obj.loads,rows=rows));print('P12_BURST_MODE',json.dumps(receipts[-1]),flush=True)
    checks=0
    for i in range(len(stages)):
        for sign in ('forward','reverse'):
            r,s=runs['reload',i][sign],runs['reuse',i][sign]
            assert np.array_equal(r['valid'],s['valid']);assert np.array_equal(r['prob'],s['prob']);checks+=2
    assert receipts[0]['loads']==4 and receipts[1]['loads']==1
    result=dict(schema='p12.burst_ab.v1',checks=checks,model_load_counts=[4,1],receipts=receipts,scope='Public previously exposed 0841 inputs; CPU parity; one ordered timing pair, no L4 claim.')
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('P12_BURST_DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
