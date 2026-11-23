// MIT; deterministic dependency-free geometry stress test and paired-ray audit.
function evaluate(input) {
  const C={step:0.5,lo:-40,hi:40,T:41,maxJump:2,penalty:0.15,minScore:0.35,minMargin:1,excludeRadius:2,seed:20261123};
  const assert=(b,m)=>{if(!b)throw Error(m)};
  const D=Math.round((C.hi-C.lo)/C.step)+1;
  const dep=i=>C.lo+i*C.step;
  const ix=x=>Math.round((x-C.lo)/C.step);
  const argmax=a=>{let j=0;for(let i=1;i<a.length;i++)if(a[i]>a[j])j=i;return j};
  function track(scores,anchor) {
    const T=scores.length, N=scores[0].length;
    assert(anchor>=0&&anchor<N&&scores.every(s=>s.length===N&&s.every(Number.isFinite)),"invalid input");
    const F=Array.from({length:T},()=>Array(N).fill(-Infinity));
    const B=Array.from({length:T},()=>Array(N).fill(-Infinity));
    const P=Array.from({length:T},()=>Array(N).fill(-1));
    F[0][anchor]=scores[0][anchor];
    for(let t=1;t<T;t++)for(let k=0;k<N;k++) {
      for(let p=Math.max(0,k-C.maxJump);p<=Math.min(N-1,k+C.maxJump);p++){
        const v=F[t-1][p]-C.penalty*Math.abs(k-p)+scores[t][k];
        if(v>F[t][k]){F[t][k]=v;P[t][k]=p}
      }
    }
    B[T-1].fill(0);
    for(let t=T-2;t>=0;t--)for(let k=0;k<N;k++)for(let p=Math.max(0,k-C.maxJump);p<=Math.min(N-1,k+C.maxJump);p++){
      B[t][k]=Math.max(B[t][k],B[t+1][p]+scores[t+1][p]-C.penalty*Math.abs(k-p));
    }
    const path=Array(T);path[T-1]=argmax(F[T-1]);
    for(let t=T-1;t>0;t--)path[t-1]=P[t][path[t]];
    const margins=path.map((k,t)=>{
      let alt=-Infinity;for(let j=0;j<N;j++)if(Math.abs(j-k)>C.excludeRadius)alt=Math.max(alt,F[t][j]+B[t][j]);
      return F[t][k]+B[t][k]-alt;
    });
    return {path,accepted:path.map((k,t)=>scores[t][k]>=C.minScore&&margins[t]>=C.minMargin)};
  }
  const check=track([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]],0);
  assert(JSON.stringify(check.path)==="[0,1,2]","known ridge");
  const jump=track([[1,0,0,0,0,0],[0,0,0,0,0,1]],0);
  assert(jump.path[1]<=C.maxJump,"jump constraint");
  const blank=track([[1,0,0,0],[0,0,0,0]],0);
  assert(!blank.accepted[1],"unsupported column must abstain");
  const raw=input.records;
  assert(Array.isArray(raw)&&raw.length===24,"unexpected record count");
  let seed=C.seed>>>0;
  const random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296};
  const gaussian=(x,c)=>Math.exp(-0.5*((x-c)/0.65)**2);
  function metrics(){return {n:0,accepted:0,within_one:0,unsafe_over_two:0,abs_error_sum:0}}
  function add(m,accepted,pred,truth){
    m.n++;if(!accepted)return;m.accepted++;const err=Math.abs(pred-truth);
    m.within_one+=Number(err<=1+1e-9);m.unsafe_over_two+=Number(err>2+1e-9);m.abs_error_sum+=err;
  }
  function finish(m){return {...m,coverage:m.accepted/m.n,conditional_within_one:m.accepted?m.within_one/m.accepted:null,conditional_mae:m.accepted?m.abs_error_sum/m.accepted:null}}
  const scenarios=["competing_crest","target_outage","approaching_crest"];
  const A=Object.fromEntries(scenarios.map(s=>[s,{greedy:metrics(),anchored:metrics(),cases:0}]));
  let eligible=0;
  for(const r of raw){
    const peaks=r.rays.raw.peak_offsets_vox;
    const landmark=r.fixed_landmark_raw_offset_vox;
    if(!Number.isFinite(landmark)||r.legacy_raw_nearest.indices.length!==1||!peaks.includes(landmark)||peaks.length<2)continue;
    eligible++;
    const others=peaks.filter(p=>p!==landmark);
    const near=others.reduce((a,b)=>Math.abs(a-landmark)<=Math.abs(b-landmark)?a:b);
    for(const scenario of scenarios){
      const a=A[scenario],truth=[],scores=[];
      a.cases++;
      for(let t=0;t<C.T;t++){
        const shift=0.5*Math.round(6*Math.sin(2*Math.PI*t/(C.T-1)));
        const pulse=Math.sin(Math.PI*t/(C.T-1))**2;
        truth.push(landmark+shift);
        let amp=1-0.55*pulse;
        if(scenario==="target_outage"&&t>=17&&t<=23)amp=0;
        const row=[];
        for(let k=0;k<D;k++){
          const d=dep(k);
          let v=amp*gaussian(d,truth[t]);
          for(const p of others){
            const closer=scenario==="approaching_crest"&&p===near?0.65*pulse*(landmark-p):0;
            v=Math.max(v,(0.3+0.65*pulse)*gaussian(d,p+shift+closer));
          }
          row.push(Math.min(1,v+0.04*random()));
        }
        scores.push(row);
      }
      const result=track(scores,ix(truth[0]));
      for(let t=1;t<C.T;t++){
        const j=argmax(scores[t]);
        add(a.greedy,scores[t][j]>=C.minScore,dep(j),truth[t]);
        add(a.anchored,result.accepted[t],dep(result.path[t]),truth[t]);
      }
    }
  }
  const simulation=Object.fromEntries(Object.entries(A).map(([name,a])=>[name,{cases:a.cases,greedy:finish(a.greedy),anchored:finish(a.anchored),gate:a.anchored.accepted>=0.5*a.anchored.n&&a.anchored.unsafe_over_two===0&&a.anchored.within_one>=0.9*a.anchored.accepted}]));
  const G={all:{n:0,accepted:0,mismatched:0,landmark_distance_over_two:0},global_cap:{n:0,accepted:0,mismatched:0,landmark_distance_over_two:0},local_guard:{n:0,accepted:0,mismatched:0,landmark_distance_over_two:0}};
  let ambiguous=0;
  for(const r of raw) {
    const p=r.rays.raw.peak_offsets_vox, l=r.fixed_landmark_raw_offset_vox, d=r.along_raw_normal_vox, s=r.rays.snapped_uv;
    const c=s.nearest_peak_to_center, f=s.fixed_raw_landmark?.nearest_crest_to_landmark_3d;
    if(r.legacy_raw_nearest.indices.length!==1||!Number.isFinite(l)||!Number.isFinite(d)||!c||!f||c.indices.length!==1||f.indices.length!==1||!p.includes(l)||p.length<2){ambiguous++;continue}
    const mismatch=c.indices[0]!==f.indices[0];
    const targetDist=Math.abs(l-d);
    const otherDist=Math.min(...p.filter(x=>x!==l).map(x=>Math.abs(x-d)));
    const accepts={all:true,global_cap:Math.abs(d)<=4,local_guard:targetDist<=2&&otherDist-targetDist>=2};
    for(const [name,accepted] of Object.entries(accepts)){
      const g=G[name];g.n++;if(!accepted)continue;
      g.accepted++;g.mismatched+=Number(mismatch);g.landmark_distance_over_two+=Number(f.minimum>2);
    }
  }
  return {schema:1,constants:C,synthetic_tests:3,source_public_sites:raw.length,synthetic_eligible_sites:eligible,simulation,simulation_gate:Object.values(simulation).every(x=>x.gate),real_paired_geometry:{excluded_ambiguous_or_missing:ambiguous,methods:G,gate:G.local_guard.accepted>=0.5*G.local_guard.n&&G.local_guard.mismatched===0},limitations:["The first experiment simulates spatial strips and m7-like scores using only real published raw crest locations. It is not real CT tracking or an ink test.","The second experiment is retrospective on previously published disjoint public rays; its endpoint is tilted-ray crest correspondence, not anatomical truth.","A globally optimal path can switch sheets; min-marginal support is not a proof of identity. No private or unread candidate data, CNN forwards or new scan acquisition."]};
}
export { evaluate };
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if(process.argv.length!==3) throw Error('usage: node run.mjs geometry.json');
  const bytes=readFileSync(process.argv[2]);
  const sha=createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex');
  if(sha!=='3e6d227294ad5d8020554a959e1a1494a427c95e')throw Error('source blob hash mismatch: '+sha);
  const start=performance.now(); const result=evaluate(JSON.parse(bytes));
  console.log(JSON.stringify({...result,source_git_blob_sha1:sha,elapsed_ms:performance.now()-start,node_version:process.version},null,2));
}
