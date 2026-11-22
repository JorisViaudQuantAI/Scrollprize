#!/usr/bin/env python3
"""Independent completed-run check. Never runs inference or requests source data.

Prepared before either field's result was inspected. Tile indices 0,27,63 were
fixed by the root agent. Execute only after both frozen fields are complete.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

FREEZE = '80c109fdac5162df1f6c92094dd73bb1c24949a1'
EXPECTED = {
 'run.py':'f5d1fd5ab2bd781f4f524c6b375ac0e7469bdb2825a68784c2fa29496dca7377',
 'protocol.json':'649d4b5bc638751f926db30bb4fb0459b0ea12fb2271ebfcc14843a87946dbf1',
 'selection.json':'c3bdeb1e37b506f596af1bafab9fbd0c1bfc0271b8eed2e2ef644db8ce47b294',
}
ARMS = ('A_native_fine','B_degraded_fine','C_coarse')
TILE_INDICES = (0,27,63)
SCROLLS = ('PHerc0841','PHerc0814')


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(test, reason):
 if not bool(test):raise AssertionError(reason)
def mean4(a):
 a=np.asarray(a)
 h,w=a.shape
 require(h%4==w%4==0,'fine score dimensions')
 total=np.zeros((h//4,w//4),np.float64)
 for dy in range(4):
  for dx in range(4):total+=a[dy::4,dx::4].astype(np.float64)
 return (total/16).astype(np.float32)
def pool4(mask):
 h,w=mask.shape
 total=np.zeros((h//4,w//4),bool)
 for dy in range(4):
  for dx in range(4):total|=mask[dy::4,dx::4]
 return total

def independent_construction(a):
 require(a.shape==(64,256,256) and a.dtype==np.uint8,'native patch shape/dtype')
 total=np.zeros((16,64,64),np.uint32)
 for dz in range(4):
  for dy in range(4):
   for dx in range(4):total+=a[dz::4,dy::4,dx::4]
 c=np.rint(total.astype(np.float64)/64).astype(np.uint8)
 z=np.arange(64)//4;y=np.arange(256)//4;x=np.arange(256)//4
 b=c[np.ix_(z,y,x)]
 fm=np.logical_or.reduce(a!=0,axis=0)
 cm=pool4(fm)
 return b,c,fm,cm

def metric(score,state,support):
 known=support&(state!=0)
 truth=state[known]==2;values=np.asarray(score)[known]
 require(np.isfinite(values).all(),'nonfinite scored value')
 pos=int(np.count_nonzero(truth));neg=int(np.count_nonzero(~truth))
 r={'known':pos+neg,'ink':pos,'background':neg,'unknown_or_unsupported':int(known.size-pos-neg),
    'ap':None,'auc':None,'recall_at_background_5pct':None,'background_exceedance':None,'threshold_strict_greater':None}
 if pos and neg:
  bg=values[~truth];budget=math.floor(.05*neg)
  threshold=float(np.sort(bg)[neg-budget-1])
  calls=values>threshold
  r.update(ap=float(average_precision_score(truth,values)),auc=float(roc_auc_score(truth,values)),
   recall_at_background_5pct=float(np.count_nonzero(calls&truth)/pos),
   background_exceedance=float(np.count_nonzero(calls&~truth)/neg),threshold_strict_greater=threshold,
   allowed_background_count=budget,background_positive_count=int(np.count_nonzero(calls&~truth)),
   ink_positive_count=int(np.count_nonzero(calls&truth)))
  require(r['background_positive_count']<=budget,'ROC point exceeded fixed background allowance')
 return r

class Comparisons:
 def __init__(self):self.count=0;self.maximum_error=0.
 def check(self,expected,actual):
  for key,wanted in expected.items():
   got=actual[key];self.count+=1
   if wanted is None:require(got is None,'unexpected defined metric '+key)
   elif isinstance(wanted,(bool,int)):
    require(wanted==got,'discrete mismatch '+key)
   elif isinstance(wanted,(float,np.floating)):
    delta=abs(float(wanted)-float(got));self.maximum_error=max(self.maximum_error,delta)
    require(delta<=1e-12,'numerical mismatch '+key)
   else:require(wanted==got,'value mismatch '+key)

def independent_gate(metrics):
 qs=('q00','q01','q10','q11')
 if any(metrics[q][a]['recall_at_background_5pct'] is None or metrics[q][a]['ap'] is None for q in qs for a in ARMS):
  return {'pass':False,'reason':'missing_class_or_support','macro_recall':None}
 macro={a:math.fsum(metrics[q][a]['recall_at_background_5pct'] for q in qs)/4 for a in ARMS}
 a,b,c=(macro[k] for k in ARMS)
 if 1-c<.05 or 1-b<=0:return {'pass':False,'reason':'baseline_ceiling_prevents_meaningful_gain','macro_recall':macro}
 rc=(a-c)/(1-c);rb=(a-b)/(1-b)
 nonworse=sum(metrics[q][ARMS[0]]['ap']>=metrics[q][ARMS[2]]['ap'] for q in qs)
 passed=rc>=.20 and a-c>=.05 and rb>=.10 and nonworse>=3
 return {'pass':bool(passed),'reason':'all_prespecified_gates_pass' if passed else 'prespecified_gain_gate_failed',
  'macro_recall':macro,'relative_miss_reduction_vs_C':rc,'absolute_macro_recall_gain_vs_C':a-c,
  'relative_miss_reduction_vs_B':rb,'quadrants_AP_nonworse_vs_C':nonworse,
  'not_statistical_significance_or_letter_reading':True}

def verify(args):
 repo=Path(args.repo);run=Path(args.run);dest=Path(args.review_dir)
 result=read(run/'result.json')
 require(result['status']=='complete','Both fields must be complete before this checker reads score arrays')
 require(tuple(f['scroll'] for f in result['fields'])==SCROLLS,'exact two complete fields required')
 require(result['freeze_reference']==FREEZE,'unexpected experiment freeze')
 lane=repo/'firstletters-colab/native_resolution';paths=[lane/k for k in EXPECTED]
 command=['git','ls-tree','-r','--full-tree',FREEZE,'--',*[str(p.relative_to(repo)) for p in paths]]
 tree=subprocess.check_output(command,cwd=repo,text=True,env={**os.environ,'GIT_NO_LAZY_FETCH':'1'})
 blobs={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in tree.splitlines()}
 for p in paths:
  b=p.read_bytes();require(hashlib.sha256(b).hexdigest()==EXPECTED[p.name],'file differs from reviewed freeze')
  blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  require(blob==blobs[str(p.relative_to(repo))],'Git freeze blob mismatch')
 for field,key in [('run.py','runner_sha256'),('protocol.json','protocol_sha256'),('selection.json','selection_sha256')]:
  require(result[key]==EXPECTED[field],'receipt source identity mismatch')
 protocol=read(lane/'protocol.json');selection=read(lane/'selection.json')
 manifest_path=repo/'firstletters-colab/transversal_lab/hecate_donor_manifest.json'
 require(sha(manifest_path)==protocol['pins']['donor_manifest_sha256'],'manifest pin')
 donors={d['scroll']:d for d in read(manifest_path)['donors']}
 choices={r['scroll']:r['choice'] for r in selection['results']}
 old={}
 for receipt,cache in args.prior_cache:
  require(sha(receipt) in protocol['pins']['prior_receipt_sha256'],'old source receipt pin')
  for row in read(receipt)['completed_assets']:
   name=hashlib.sha256(row['url'].encode()).hexdigest()
   if row['url'] in old:require(old[row['url']]['sha256']==row['sha256'],'inconsistent prior source')
   old[row['url']]={**row,'path':Path(cache)/name}
 source=read(run/'source_receipt.json');require(sha(run/'source_receipt.json')==result['source_receipt_sha256'],'source receipt hash')
 verified_bytes=0;newbytes=0
 for row in source['completed_assets']:
  if row['reused_old_cache']:
   previous=old[row['url']];require(all(previous[k]==row[k] for k in ['bytes','sha256']),'reused source not pinned')
   p=previous['path']
  else:
   p=run/'source_cache'/hashlib.sha256(row['url'].encode()).hexdigest();newbytes+=row['bytes']
  require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'cached object bytes/hash')
  verified_bytes+=row['bytes']
 require(newbytes==source['new_application_body_bytes']==result['new_application_body_bytes'],'new byte accounting')
 require(newbytes<=protocol['limits']['new_body_bytes'],'byte budget')
 require(source['requests_including_redirects']==result['requests_including_redirects']<=protocol['limits']['requests_including_redirects'],'request accounting')
 require(not source['failed_assets'],'completed experiment unexpectedly retains source failures')
 require(sha(args.coarse_checkpoint)==protocol['coarse_checkpoint']['sha256'],'coarse checkpoint pin')

 # Only this pure helper is executed from the frozen source; no model code runs.
 nodes=[n for n in ast.parse((lane/'run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='degrade']
 require(len(nodes)==1,'degrade helper definition')
 namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-degrade-helper>','exec'),namespace)
 check=Comparisons();field_rows=[];paired=[];forward_count=0;score_reductions=0
 pre=read(run/'pre_field_freeze.json');decision=read(run/'pilot_decision.json')
 require(pre['pilot_decision_sha256']==sha(run/'pilot_decision.json'),'pre-field decision hash')
 require(decision==result['pilot_decision'] and pre['no_field_scores_seen'] is True,'pilot decision receipt')
 inputs={r['scroll']:r for r in pre['field_inputs']}
 regions={'field':(slice(None),slice(None)),'q00':(slice(0,128),slice(0,128)),
  'q01':(slice(0,128),slice(128,256)),'q10':(slice(128,256),slice(0,128)),
  'q11':(slice(128,256),slice(128,256))}
 for row in result['fields']:
  scroll=row['scroll'];d=donors[scroll];choice=choices[scroll];folder=run/scroll
  require(read(folder/'result.json')==row,'field result differs from final receipt')
  require(sha(folder/'scores.npz')==row['scores_sha256'],'score packet hash')
  require(sha(folder/'fine_context_uint8.npy')==row['fine_context_sha256']==inputs[scroll]['fine_sha256'],'field input hash')
  require(sha(folder/'labels.npy')==row['labels_sha256']==inputs[scroll]['labels_sha256'],'field label hash')
  with np.load(folder/'scores.npz',allow_pickle=False) as packet:arrays={k:packet[k].copy() for k in packet.files}
  raw=np.load(folder/'fine_context_uint8.npy',mmap_mode='r',allow_pickle=False)
  require(raw.shape==(64,1152,1152) and raw.dtype==np.uint8,'whole fine input geometry')
  labpath=old[d['label_url']]['path'];require(sha(labpath)==d['label_sha256'],'categorical source pin')
  labels=np.asarray(Image.open(labpath));y0,x0=choice['core_origin_target_yx']
  yy=((2*(y0+np.arange(256,dtype=np.int64))+1)*20)//39
  xx=((2*(x0+np.arange(256,dtype=np.int64))+1)*20)//39
  state=labels[np.ix_(yy,xx)]
  np.testing.assert_array_equal(state,arrays['labels']);np.testing.assert_array_equal(state,np.load(folder/'labels.npy',allow_pickle=False))
  counts={'known':int(np.count_nonzero(state)),'ink':int(np.count_nonzero(state==2)),'background':int(np.count_nonzero(state==1)),'unknown':int(np.count_nonzero(state==0))}
  require(all(choice['labels'][k]==v for k,v in counts.items()),'selected counts changed')
  availability=np.any(raw[:,64:1088,64:1088]!=0,axis=0)
  common_support=pool4(availability)
  np.testing.assert_array_equal(common_support,arrays['support'])
  np.testing.assert_array_equal(common_support,np.load(folder/'common_support.npy',allow_pickle=False))
  require(int(common_support.sum())==row['scored_support_pixels'] and row['all_conditions_same_scored_support'] is True,'shared support receipt')
  for index in TILE_INDICES:
   ty,tx=divmod(index,8);y,x=ty*32,tx*32
   a=np.ascontiguousarray(raw[:,4*y:4*y+256,4*x:4*x+256]);expected=independent_construction(a)
   got=namespace['degrade'](a)
   for e,g in zip(expected,got):np.testing.assert_array_equal(e,g)
   b,c,fm,cm=expected
   np.testing.assert_array_equal(common_support[y:y+32,x:x+32],cm[16:48,16:48])
   paired.append({'scroll':scroll,'tile_index':index,'A_sha256':arrsha(a),'B_sha256':arrsha(b),'C_sha256':arrsha(c),
    'fine_mask_sha256':arrsha(fm),'coarse_mask_sha256':arrsha(cm),'independent_integer_box_average_matches_frozen_helper':True})
  recomputed={}
  for arm in ARMS:
   common=arrays[arm];require(common.shape==(256,256) and np.isfinite(common).all(),'complete common map')
   np.testing.assert_array_equal(common,np.load(folder/(arm+'_common.npy'),allow_pickle=False))
   if arm!=ARMS[2]:
    fine=arrays[arm+'_fine_grid'];require(fine.shape==(1024,1024) and np.isfinite(fine).all(),'complete fine map')
    np.testing.assert_array_equal(fine,np.load(folder/(arm+'_fine.npy'),allow_pickle=False))
    np.testing.assert_array_equal(mean4(fine),common);score_reductions+=1
   require(len(row['forward_seconds'][arm])==64,'field forward count per arm')
   require(all(t>0 for t in row['forward_seconds'][arm]),'positive forward timings')
   forward_count+=len(row['forward_seconds'][arm])
  for name,region in regions.items():
   recomputed[name]={}
   for arm in ARMS:
    got=metric(arrays[arm][region],state[region],common_support[region]);check.check(got,row['metrics'][name][arm]);recomputed[name][arm]=got
  g=independent_gate(recomputed);require(g['pass']==row['gate']['pass'] and g['reason']==row['gate']['reason'],'field advancement decision')
  if g['macro_recall'] is not None:check.check(g['macro_recall'],row['gate']['macro_recall'])
  for k,v in g.items():
   if k!='macro_recall':check.check({k:v},row['gate'])
  field_rows.append({'scroll':scroll,'label_counts':counts,'support_pixels':int(common_support.sum()),'gate':g,
   'field_metrics':recomputed['field'],'quadrant_metrics':{q:recomputed[q] for q in regions if q!='field'}})
  del raw,arrays

 # Independently verify the pilot's selected orders and numerical/backend gate.
 with np.load(run/'pilot_inputs.npz',allow_pickle=False) as p:pilot={k:p[k].copy() for k in p.files}
 with np.load(run/'pilot_scores.npz',allow_pickle=False) as p:pilots={k:p[k].copy() for k in p.files}
 pb,pc,pfm,pcm=independent_construction(pilot['fine'])
 for e,k in [(pc,'coarse'),(pfm,'fine_valid'),(pcm,'coarse_valid')]:np.testing.assert_array_equal(e,pilot[k])
 pm={};times={}
 for model in ('fine','coarse'):
  pm[model]={};times[model]=[]
  for order in ('source','reverse'):
   s=pilots[f'{model}_{order}_fp32'];core=mean4(s[64:192,64:192]) if model=='fine' else s[16:48,16:48]
   got=metric(core,pilot['labels'],pcm[16:48,16:48]);check.check(got,decision['pilot_fp32'][model][order]['metrics']);pm[model][order]=got
   times[model].append(decision['pilot_fp32'][model][order]['seconds'])
  chosen='source' if pm[model]['source']['auc']>pm[model]['reverse']['auc'] else 'reverse'
  require(decision['orders'][model]==chosen,'pilot order selection')
 fmean=math.fsum(times['fine'])/2;cmean=math.fsum(times['coarse'])/2;fp32_projection=256*fmean+128*cmean
 require(abs(fp32_projection-decision['projected_field_fp32_seconds'])<1e-9,'fp32 runtime projection')
 parity=decision['bf16_parity'];expect_precision='fp32';selected_fine_mean=fmean
 if fp32_projection>3600:
  require(len(parity)==2,'missing predeclared parity trial')
  for order,rec in zip(('source','reverse'),parity):
   s=pilots['fine_'+order+'_bf16'];base=pilots['fine_'+order+'_fp32']
   delta=float(np.max(np.abs(s.astype(np.float64)-base.astype(np.float64))))
   apdiff=abs(metric(mean4(s[64:192,64:192]),pilot['labels'],pcm[16:48,16:48])['ap']-pm['fine'][order]['ap'])
   check.check({'order':order,'max_abs_probability_difference':delta,'absolute_AP_difference':apdiff,'pass':delta<=.005 and apdiff<=.01},rec)
  bfmean=math.fsum(r['seconds'] for r in parity)/2
  if all(r['pass'] for r in parity) and bfmean<fmean:expect_precision='bf16';selected_fine_mean=bfmean
 else:require(not parity,'unexpected parity trials below trigger')
 require(decision['fine_precision']==expect_precision and decision['coarse_precision']=='fp32','selected precision')
 require(abs(256*selected_fine_mean+128*cmean-decision['projected_selected_field_seconds'])<1e-9,'selected runtime projection')
 require(forward_count==384,'exact full-field count')
 require(result['forwards_completed']==forward_count+4+len(parity),'total forward count')
 require(pre['forwards_before_fields']==4+len(parity),'pre-field forward count')
 require(result['numeric_gate_pass']==all(f['gate']['pass'] for f in field_rows),'two-field advancement decision')
 require(result['model_buffers_unchanged'] is True and result['training_steps']==0 and result['letter_reading_claim'] is False,'frozen execution claims')
 summary={'schema':'scrollprize.native-resolution-independent-review.v1','status':'pass','freeze_reference':FREEZE,
  'review_script_sha256':sha(__file__),'result_sha256':sha(run/'result.json'),'reviewed_source_sha256':EXPECTED,
  'source_assets_verified':len(source['completed_assets']),'source_asset_bytes_verified':verified_bytes,
  'new_application_body_bytes':newbytes,'paired_construction_tile_indices':list(TILE_INDICES),'paired_input_reconstructions':paired,
  'fine_to_common_maps_exactly_equal':score_reductions,'same_support_for_all_arms':True,
  'scalar_metric_comparisons':check.count,'maximum_absolute_scalar_metric_error':check.maximum_error,
  'field_forwards':forward_count,'total_forwards':result['forwards_completed'],'pilot_orders':decision['orders'],
  'fine_precision':decision['fine_precision'],'numeric_gate_pass':result['numeric_gate_pass'],'fields':field_rows,
  'new_model_forwards_during_review':0,'network_requests_during_review':0,
  'method':'Independent integer block sums and indexing, NumPy score/support reconstruction, sklearn AP/AUC and separate order-statistic ROC points; comparison against the isolated frozen degradation helper; no inference code called.',
  'limitations':['Per-forward A/B/C tensors were not retained: deterministic paired construction is independently reconstructed from retained fine inputs and checked against the frozen helper, rather than verified through saved per-forward tensor hashes.','Common maps and metrics are independently checked from saved scores, not independently regenerated model outputs.','Model-assisted annotations, prior donor exposure and label-selected development fields remain; review does not establish photographic ground truth, independent generalisation, letters or a prize.']}
 dest.mkdir(exist_ok=True,parents=True);(dest/'independent_review.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':'pass','metric_comparisons':check.count,'maximum_metric_error':check.maximum_error,'numeric_gate_pass':summary['numeric_gate_pass'],'field_forwards':forward_count,'review':str(dest/'independent_review.json')},indent=2))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--execute-after-both-fields-complete',action='store_true',required=True)
 p.add_argument('--repo',required=True);p.add_argument('--run',required=True);p.add_argument('--review-dir',required=True)
 p.add_argument('--coarse-checkpoint',required=True);p.add_argument('--prior-cache',action='append',nargs=2,required=True)
 args=p.parse_args();verify(args)
if __name__=='__main__':main()
