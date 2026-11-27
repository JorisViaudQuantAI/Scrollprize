"""Aggregate-only arithmetic replay of Gemini P12's published receipt."""
import argparse,hashlib,json,math
from pathlib import Path
PIN='d58593ae614117050d984a0cbe5257601d94c51a'
def run(p):
    b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==PIN
    j=json.loads(b);r=j['rows'];t=j['totals'];area=sum(x['valid_ct_cm2'] for x in r);wall=sum(x['total_seconds'] for x in r);calls=sum(x['rv3_forward']['seconds']+x['rv3_reverse']['seconds'] for x in r)
    assert len(r)==t['meshes_completed']==5 and round(area,4)==t['total_cm2'] and round(wall,2)==t['sum_mesh_wall_seconds'] and round(calls,2)==t['sum_gpu_inference_seconds'] and round(calls/wall,4)==t['gpu_time_fraction']
    return dict(schema='p12.aggregate_cost_replay.v1',source_blob=PIN,area_cm2=area,sum_wall_seconds=wall,sum_inference_call_seconds=calls,other_seconds=wall-calls,inference_call_fraction=calls/wall,seconds_cm2_per_direction=calls/(2*area),idealized_wall_ratio_if_all_noncall_time_removed=wall/calls,idealized_area_in_5min_both_directions=300*area/calls,seconds_20cm2_both=20*calls/area,seconds_50cm2_both=50*calls/area,scope='Inference call wall time includes model setup, I/O, and synchronization; not CUDA kernel utilization or VM invoice. Linear extrapolation neglects startup/transfer and is not measured speed.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('out',type=Path);a=p.parse_args();r=run(a.receipt);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
