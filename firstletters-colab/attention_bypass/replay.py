#!/usr/bin/env python3
"""After-outcome portable replay of the cached-logit retrospective analysis.

This is not the originally executed script. Original and replay identities are
recorded separately. No networking, model imports, or model forwards occur.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
from PIL import Image

HERE=Path(__file__).resolve().parent
P=HERE/'protocol.json'
EXPECTED_PROTOCOL='eff81c511c28f8ac4fb9d48dea28a4ef7b2de397edf01faa7db7f8390f05070b'
ARMS=('released','whole_depth_max')
ORDERS=('source','reverse')
EPS=np.finfo(np.float32).eps

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def verify(path,expected):
    if sha(path)!=expected:raise ValueError('Pinned identity changed: '+str(path))

def threshold_background(scores):
    scores=np.asarray(scores)
    if len(scores)==0:return None
    k=math.floor(.05*len(scores))
    return float(np.sort(scores)[::-1][k])

def metrics(labels,scores,threshold=None):
    n=int(len(labels));ni=int(np.sum(labels));nb=n-ni
    mixed=ni>0 and nb>0
    t=threshold if threshold is not None else threshold_background(scores[labels==0])
    selected=np.zeros(n,bool) if t is None else scores>t
    return {'n':n,'ink':ni,'background':nb,
            'ap':float(average_precision_score(labels,scores)) if mixed else None,
            'auc':float(roc_auc_score(labels,scores)) if mixed else None,
            'threshold_strict_greater':t,
            'recall':float(selected[labels==1].mean()) if ni and t is not None else None,
            'background_rate':float(selected[labels==0].mean()) if nb and t is not None else None,
            'selected_ink':int(selected[labels==1].sum()) if t is not None else None,
            'selected_background':int(selected[labels==0].sum()) if t is not None else None}

def load_labels(donor,roots):
    key=hashlib.sha256(donor['label_url'].encode()).hexdigest()
    candidates=[Path(root)/key for root in roots]
    present=[p for p in candidates if p.exists()]
    if not present:raise FileNotFoundError('Required old label cache missing')
    path=present[0];verify(path,donor['label_sha256'])
    if donor['label_format']=='png':
        with Image.open(path) as im:
            if im.mode not in ('L','P'):raise ValueError('Unexpected label mode')
            labels=np.asarray(im).copy()
    elif donor['label_format']=='npy':labels=np.load(path,allow_pickle=False)
    else:raise ValueError('Unexpected label format')
    if list(labels.shape)!=donor['label_shape_yx'] or not np.isin(labels,[0,1,2]).all():raise ValueError('Invalid labels')
    return labels

def indices(donor,context):
    if abs(donor['label_pitch_um']-9.36)>1e-8:raise ValueError('Label pitch differs')
    y,x=context['origin_target_yx']
    yy=((2*(y+np.arange(16,48,dtype=np.int64))+1)*20)//39
    xx=((2*(x+np.arange(16,48,dtype=np.int64))+1)*20)//39
    return yy,xx

def coherent_depth(volume,order,labels,support,scores,measured):
    t0=measured['released']['threshold_strict_greater'];t1=measured['whole_depth_max']['threshold_strict_greater']
    if t0 is None or t1 is None:return {'status':'no_known_background_operating_point'}
    selected=(labels==2)&support&(scores['whole_depth_max']>t1)&~(scores['released']>t0)
    depth=np.argmax(volume,axis=0)
    if order=='reverse':depth=15-depth
    edge_count=0;good_count=0
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        edges=selected[a]&selected[b]
        edge_count+=int(edges.sum());good_count+=int((edges&(np.abs(depth[a]-depth[b])<=1)).sum())
    return {'status':'supported' if edge_count>=20 else 'insufficient_recovered_ink_edges',
            'newly_recalled_ink':int(selected.sum()),'source_order_argmax_depth_histogram':np.bincount(depth[selected],minlength=16).tolist(),
            'adjacent_recovered_ink_edges':edge_count,'edges_depth_difference_at_most_one':good_count,
            'coherence_fraction':float(good_count/edge_count) if edge_count>=20 else None}

def summarize(records):
    rows=[]
    for donor in ('PHerc0009B','PHerc0841','PHerc0814'):
        for order in ORDERS:
            allrows=[r for r in records if r['scroll']==donor and r['order']==order]
            mixed=[r for r in allrows if r['class_status']=='mixed']
            counts={k:sum(r['class_status']==k for r in allrows) for k in ('mixed','background_only','ink_only','no_known_supported_labels')}
            item={'scroll':donor,'order':order,'all_contexts':len(allrows),'context_classes':counts,'mixed_contexts':len(mixed)}
            for arm in ARMS:
                item[arm]={'macro_recall':float(np.mean([r['metrics'][arm]['recall'] for r in mixed])) if mixed else None,
                           'macro_ap':float(np.mean([r['metrics'][arm]['ap'] for r in mixed])) if mixed else None}
            if mixed:
                a,b=item['whole_depth_max']['macro_recall'],item['released']['macro_recall']
                item['absolute_macro_recall_gain']=a-b
                item['relative_macro_miss_reduction']=(a-b)/(1-b) if b<1 else None
                item['AP_nonworse_contexts']=sum(r['metrics']['whole_depth_max']['ap']>=r['metrics']['released']['ap'] for r in mixed)
                item['AP_nonworse_fraction']=item['AP_nonworse_contexts']/len(mixed)
                edges=sum(r['depth_coherence']['adjacent_recovered_ink_edges'] for r in mixed)
                good=sum(r['depth_coherence']['edges_depth_difference_at_most_one'] for r in mixed)
                item['aggregate_recovered_ink_depth_edges']=edges
                item['aggregate_depth_coherence_fraction']=good/edges if edges>=20 else None
            else:item.update(absolute_macro_recall_gain=None,relative_macro_miss_reduction=None,AP_nonworse_contexts=None,AP_nonworse_fraction=None)
            rows.append(item)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run-root',type=Path,required=True,help='Original run directory containing features/ and source-cache/')
    parser.add_argument('--old-role-root',type=Path,required=True,help='Original role-amended run directory containing features/ and source-cache/')
    parser.add_argument('--manifest',type=Path,help='Optional relocated public donor manifest; the pinned hash must match')
    parser.add_argument('--output',type=Path,required=True,help='Fresh JSON path outside any Git checkout')
    args=parser.parse_args()
    start=time.monotonic();verify(P,EXPECTED_PROTOCOL);portable=json.loads(P.read_text())
    protocol=portable['fixed_analysis'];assets=portable['assets']
    output=args.output.resolve()
    if output.exists():raise ValueError('Replay output already exists')
    if any((p/'.git').exists() for p in output.parents):raise ValueError('Replay output must remain outside Git')
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=args.manifest or (HERE/assets['donor_manifest_relative_path']).resolve()
    verify(manifest_path,assets['donor_manifest_sha256'])
    manifest=json.loads(manifest_path.read_text())
    roots={'old_run_root':args.old_run_root,'old_role_root':args.old_role_root}
    label_roots=[root/'source-cache' for root in roots.values()]
    # Generated checks on strict thresholds including ties and fewer than20 backgrounds.
    for n in (1,5,19,20,21,100):
        for bg in (np.arange(n,dtype=np.float32),np.zeros(n,dtype=np.float32)):
            assert int((bg>threshold_background(bg)).sum())<=math.floor(.05*n)
    rows=[];arrays={};hashes={};seen_by_donor={};overlap_counts={}
    for donor in manifest['donors']:
        labels=load_labels(donor,label_roots)
        seen_by_donor[donor['scroll']]=set();overlap_counts[donor['scroll']]=0
        base=roots[assets['feature_cache_root_by_original_split'][donor['split']]]
        for number,context in enumerate(donor['contexts']):
            yy,xx=indices(donor,context);lab=labels[np.ix_(yy,xx)]
            ids=(yy[:,None]*donor['label_shape_yx'][1]+xx[None,:]).ravel()
            overlap_counts[donor['scroll']]+=sum(int(v) in seen_by_donor[donor['scroll']] for v in ids)
            seen_by_donor[donor['scroll']].update(map(int,ids))
            for order in ORDERS:
                relative=f"features/{donor['split']}-{number:02d}-{order}.npz"
                feature_path=base/relative
                public_key=donor['scroll']+'/'+feature_path.name
                expected=assets['feature_sha256'][public_key]
                verify(feature_path,expected);hashes[public_key]=expected
                with np.load(feature_path,allow_pickle=False) as cached:
                    volume=cached['volume'];baseline=cached['baseline'];support=cached['support']
                if volume.shape!=(16,32,32) or baseline.shape!=(32,32) or support.shape!=(32,32) or support.dtype!=np.bool_:raise ValueError('Cached schema changed')
                if not np.isfinite(volume).all() or not np.isfinite(baseline).all():raise ValueError('Nonfinite old cached scores')
                known=support&(lab!=0);y=(lab[known]==2).astype(np.uint8)
                score={'released':np.clip(expit(baseline.astype(np.float32)),EPS,1-EPS),
                       'whole_depth_max':np.clip(expit(np.max(volume,axis=0).astype(np.float32)),EPS,1-EPS)}
                measured={arm:metrics(y,score[arm][known]) for arm in ARMS}
                ni=int(y.sum());nb=int(len(y)-ni)
                status='mixed' if ni and nb else 'background_only' if nb else 'ink_only' if ni else 'no_known_supported_labels'
                row={'scroll':donor['scroll'],'original_split':donor['split'],'context':context['id'],'order':order,'class_status':status,
                     'known':len(y),'ink':ni,'background':nb,'supported_pixels':int(support.sum()),'unknown_supported_pixels':int((support&(lab==0)).sum()),
                     'metrics':measured,'maximum_probability_below_released_by_more_than_2e_6':int((known&(score['whole_depth_max']<score['released']-2e-6)).sum()),
                     'depth_coherence':coherent_depth(volume,order,lab,support,score,measured)}
                rows.append(row);arrays[(donor['scroll'],context['id'],order)]={'labels':y,**{arm:score[arm][known] for arm in ARMS}}
    if len(rows)!=96 or len(hashes)!=96:raise ValueError('Original context set incomplete')
    if any(overlap_counts.values()):raise ValueError('Unexpected duplicated score-core labels; preserve failure instead of silently deduplicating')
    transfer_thresholds={}
    for order in ORDERS:
        transfer_thresholds[order]={}
        calibration=[(r,arrays[(r['scroll'],r['context'],order)]) for r in rows if r['scroll']=='PHerc0009B' and r['order']==order]
        for arm in ARMS:
            background=np.concatenate([a[arm][a['labels']==0] for _,a in calibration])
            t=threshold_background(background)
            if t is None:raise ValueError('Original background calibration absent')
            transfer_thresholds[order][arm]={'threshold_strict_greater':t,'background':len(background),'rate':float((background>t).mean())}
            for row in rows:
                if row['order']!=order:continue
                data=arrays[(row['scroll'],row['context'],order)]
                row.setdefault('transferred_PHerc0009B_5pct_threshold',{})[arm]=metrics(data['labels'],data[arm],t)
    summary=summarize(rows);screens=[]
    for donor in protocol['retrospective_mechanism_screen']['donors']:
        item=next(r for r in summary if r['scroll']==donor and r['order']=='reverse')
        reasons=[]
        if item['mixed_contexts']<2:reasons.append('fewer_than_two_mixed_contexts')
        else:
            if item['released']['macro_recall']>.95:reasons.append('released_recall_ceiling')
            if item['absolute_macro_recall_gain']<.05:reasons.append('absolute_macro_recall_gain_below_0.05')
            if item['relative_macro_miss_reduction'] is None or item['relative_macro_miss_reduction']<.20:reasons.append('relative_macro_miss_reduction_below_0.20')
            if item['AP_nonworse_fraction']<.75:reasons.append('AP_nonworse_fraction_below_0.75')
        screens.append({'scroll':donor,'order':'reverse','pass':not reasons,'reasons':reasons})
    analysis={'contexts':rows,'donor_summary':summary,'whole_search_background_calibration':transfer_thresholds,
              'retrospective_mechanism_screen':screens,'both_donors_screen_pass':all(s['pass'] for s in screens),
              'overlapping_score_label_ids':overlap_counts}
    encoded=json.dumps(analysis,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    evidence=json.loads((HERE/'evidence.json').read_text())
    match=(analysis==evidence['results'] and hashes==evidence['source_features_sha256']
           and digest==evidence['analysis_results_sha256'])
    result={'schema':'scrollprize.cached-attention-bypass.replay.v1',
            'original_execution':portable['original_execution'],
            'after_outcome_replay':{'code_sha256':sha(__file__),'protocol_sha256':sha(P),
                                    'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'analysis_body_wall_seconds':time.monotonic()-start},
            'source_features_sha256':hashes,'analysis_results_sha256':digest,'results':analysis,
            'exact_replay_match':match,'new_model_forwards':0,'network_requests':0,
            'native_resolution_outputs_read':False,'native_resolution_gates_unchanged':True,
            'limitations':protocol['limitations'],'not_a_reading_or_prize_claim':True}
    save(output,result)
    print(json.dumps({'status':'complete' if match else 'replay_mismatch','contexts':len(rows),
                      'exact_replay_match':match,'new_forwards':0,'analysis_results_sha256':digest,
                      'replay_result_sha256':sha(output)},indent=2,allow_nan=False))
    if not match:raise ValueError('Replay differs from the recorded original calculation')

if __name__=='__main__':main()
