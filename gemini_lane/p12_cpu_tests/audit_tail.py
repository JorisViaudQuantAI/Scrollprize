"""Audit the proposed p995<.72 veto on archived public Reader crops only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
PINS={'b21/predictions.npz':'baf48b9301c9697c0dd8f4efbb7ffde9c2f891305f08d39194323eb860bd2d81','pred841/predictions.npz':'2fb79c53bc83c4d008a9bcffaad7fd788691140e24c7012440b61553ffb8c691'}
def run(d):
    for k,v in PINS.items():
        with (d/k).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==v
    a=np.load(d/'b21/predictions.npz',allow_pickle=False);b=np.load(d/'pred841/predictions.npz',allow_pickle=False)
    groups=[('1447',a['probabilities'][0],a['states']),('0841_ascending',b['probabilities'][0,0],b['states']),('0841_descending',b['probabilities'][1,0],b['states'])]
    rows=[]
    for name,p,s in groups:
        v=[]
        for q,t in zip(p,s):
            ink=t==2
            if ink.any():v.append((float(np.quantile(q,.995)),int(ink.sum()),int((ink&(q>=.5)).sum())))
        low=[x for x in v if x[0]<.72]
        rows.append(dict(group=name,positive_observations=len(v),low_tail_positive=len(low),low_tail_detected_at_half=sum(x[2]>0 for x in low),known_ink_in_low_tail=sum(x[1] for x in low),tp_at_half_in_low_tail=sum(x[2] for x in low)))
    return dict(schema='p12.tail_veto_counterexamples.v1',source_pins=PINS,rows=rows,scope='Public selected central192 crops on two exposed parents, 0841 orientations are paired hypotheses not independent crops. Not whole-mesh recall or unread adjudication.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();r=run(a.data);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
