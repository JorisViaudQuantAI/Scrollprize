// Node 18+, run from any directory. Source is pinned Git blob.
import fs from 'node:fs';
import crypto from 'node:crypto';
function evaluate(input) {
  const cells=input.cells, cal=cells.filter(c=>c.split==="calibration"), test=cells.filter(c=>c.split==="validation");
  if(cal.length!==12||test.length!==12||cells.some(c=>!c.anatomically_eligible||c.native_profile.length!==28||c.native_profile.some(v=>!Number.isFinite(v)))) throw Error("unexpected source");
  const range=Array.from({length:17},(_,i)=>i-8), sets=[];
  for(let a=-8;a<=6;a++)for(let b=a+1;b<=7;b++)for(let c=b+1;c<=8;c++){const xs=[a,b,c];sets.push({xs,penalty:a*a+b*b+c*c,covered:range.map((v,i)=>xs.some(x=>Math.abs(v-x)<=1)?i:-1).filter(i=>i>=0)})}
  function pearson(a,b){let ma=a.reduce((s,v)=>s+v,0)/a.length,mb=b.reduce((s,v)=>s+v,0)/b.length,aa=0,bb=0,ab=0;for(let i=0;i<a.length;i++){let x=a[i]-ma,y=b[i]-mb;aa+=x*x;bb+=y*y;ab+=x*y}return aa*bb>0?ab/Math.sqrt(aa*bb):0;}
  const library=range.map(c=>cal.map(t=>t.native_profile.slice(8-c,20-c)));
  function propose(y){const scores=library.map(templates=>templates.map(t=>pearson(y,t)).sort((a,b)=>b-a).slice(0,3).reduce((a,b)=>a+b,0)/3);const mx=Math.max(...scores),weights=scores.map(s=>Math.exp((s-mx)/.10));let best=null,bestValue=-Infinity;for(const s of sets){const value=s.covered.reduce((sum,i)=>sum+weights[i],0);if(value>bestValue+1e-12||Math.abs(value-bestValue)<=1e-12&&s.penalty<best.penalty){best=s;bestValue=value}}const one=range.slice().sort((a,b)=>scores[b+8]-scores[a+8]||Math.abs(a)-Math.abs(b)||a-b)[0];return {adaptive:best.xs,uniform:[-6,0,6],single:[one],scores}}
  const cases=[];
  for(const c of test)for(const d of range){const truth=-d, q=propose(c.native_profile.slice(8+d,20+d));const distance={};for(const key of ["uniform","adaptive","single"])distance[key]=Math.min(...q[key].map(v=>Math.abs(v-truth)));cases.push({cell:c.id,injected_offset:d,truth,...q,distance})}
  function stats(rows){const ans={n:rows.length};for(const key of ["uniform","adaptive","single"]){const ds=rows.map(r=>r.distance[key]);ans[key]={within1:ds.filter(d=>d<=1).length/rows.length,within2:ds.filter(d=>d<=2).length/rows.length,mean:ds.reduce((a,b)=>a+b,0)/rows.length,max:Math.max(...ds)}}ans.paired={wins:rows.filter(r=>r.distance.adaptive<r.distance.uniform).length,losses:rows.filter(r=>r.distance.adaptive>r.distance.uniform).length,ties:rows.filter(r=>r.distance.adaptive===r.distance.uniform).length};return ans}
  const overall=stats(cases),perCell=Object.fromEntries(test.map(c=>[c.id,stats(cases.filter(r=>r.cell===c.id))])),perOffset=Object.fromEntries(range.map(d=>[d,stats(cases.filter(r=>r.injected_offset===d))]));
  const checks={gain10pp:overall.adaptive.within1-overall.uniform.within1>=.1-1e-12,within2_90:overall.adaptive.within2>=.9,noCellDrop10pp:Object.values(perCell).every(s=>s.adaptive.within1>=s.uniform.within1-.1-1e-12)};
  if(sets.length!==680||cases.length!==204||Math.abs(pearson([1,2,3],[3,5,7])-1)>1e-12||cases.some(c=>new Set(c.adaptive).size!==3||c.adaptive.some(v=>!range.includes(v))))throw Error("self-test failed");
  return {experiment:"depth_hypotheses_0841_retrospective_v1",source_blob_sha1:"b3137ea8c52d5ac0cb6d66137aa1a4f9e1539a64",self_tests:"pass",overall,perCell,perOffset,checks,passed:Object.values(checks).every(Boolean),cases,limitations:"real archived mean CT profiles, artificial offsets, same previously examined scroll and no ink model; reference is stored band convention only"};
}
const bytes=fs.readFileSync(process.argv[2]);
const hash=crypto.createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex');
if(hash!=='b3137ea8c52d5ac0cb6d66137aa1a4f9e1539a64')throw Error('wrong Git blob');
console.log(JSON.stringify(evaluate(JSON.parse(bytes)),null,2));
