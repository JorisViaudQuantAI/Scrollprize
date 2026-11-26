import fs from 'node:fs';
import crypto from 'node:crypto';
function assert(ok,msg){if(!ok)throw Error(msg)}
function bounds(n,b,a){assert(Number.isInteger(n)&&n>=0&&[b,...a].every(x=>Number.isInteger(x)&&x>=0&&x<=n),"bad marginal");return [Math.max(0,...a.map(x=>x-b)),Math.min(n-b,a.reduce((s,x)=>s+x,0))]}
function test(){
  let checks=0;
  for(let n=0;n<=5;n++){
    const masks=1<<n;const count=x=>{let c=0;while(x){c+=x&1;x>>=1}return c};
    for(let k=1;k<=3;k++){
      const stats=new Map();
      function visit(arr){if(arr.length<k+1){for(let m=0;m<masks;m++)visit([...arr,m]);return;}
        const key=arr.map(count).join(","),u=arr.slice(1).reduce((x,y)=>x|y,0),gain=count(u&~arr[0]);let r=stats.get(key)||[Infinity,-Infinity];r=[Math.min(r[0],gain),Math.max(r[1],gain)];stats.set(key,r);
      }
      visit([]);
      for(const [key,v] of stats){const [b,...a]=key.split(",").map(Number);const z=bounds(n,b,a);assert(z[0]===v[0]&&z[1]===v[1],"oracle mismatch");checks++}
    }
  }
  let rejected=0;for(const a of [[2,3,[1]],[2,0,[3]],[2,0,[-1]],[2,0,[.5]]]){try{bounds(...a)}catch{rejected++}}assert(rejected===4,"bad marginals accepted");return {exhaustive_cardinality_cases:checks,invalid_cases:rejected}
}
function run(j){
 const arms=["identity","normal_only","anchor_only","rigid"],expected=["x_minus20","x_minus5","x_plus5","x_plus20","y_minus20","y_minus5","y_plus5","y_plus20"];
 const specs=[["identity_plus_normal","identity",["normal_only"]],["identity_plus_anchor","identity",["anchor_only"]],["identity_plus_rigid","identity",["rigid"]],["identity_plus_all","identity",["normal_only","anchor_only","rigid"]],["rigid_plus_anchor","rigid",["anchor_only"]],["rigid_plus_normal","rigid",["normal_only"]],["rigid_plus_hybrids","rigid",["anchor_only","normal_only"]]];
 assert(j.per_core.length===8&&j.blocks.length===8,"block count");
 const rows=[];let marginalChecks=0;
 for(let i=0;i<8;i++){const block=j.per_core[i],g=j.blocks[i];assert(block.block===expected[i]&&g.name===expected[i],"block name");
 const ids=[250,288,326,357,358,359,360,361,362,395,399,430,432,469,471,472,509];
 assert(block.cores.length===17&&block.cores.every((c,k)=>c.author_index===ids[k]),"core identity");
 for(const c of block.cores)for(const a of arms){const m=c.metrics[a],b=c.metrics.identity;
 for(const q of ["ink_pixels","background_pixels","unknown_pixels"]){assert(m[q]===b[q]&&Number.isInteger(m[q])&&m[q]>=0,"population mismatch");marginalChecks++}
 assert(m.ink_pixels+m.background_pixels+m.unknown_pixels===1024,"core size");
 bounds(m.ink_pixels,m.true_positive_at_0_5,[]);bounds(m.background_pixels,m.false_positive_at_0_5,[])}
 for(const group of ["all","mixed","pure_background","pure_ink"]){const cat={mixed:"mixed",pure_background:"background",pure_ink:"ink"}[group];const cs=group==="all"?block.cores:block.cores.filter(c=>c.category===cat);
 assert(cs.length===({all:17,mixed:11,pure_background:3,pure_ink:3}[group]),"group count");
 for(const a of arms)for(const q of ["ink_pixels","background_pixels","unknown_pixels","true_positive_at_0_5","false_positive_at_0_5"]){assert(cs.reduce((s,c)=>s+c.metrics[a][q],0)===g.metrics[a][group][q],"published aggregate mismatch");marginalChecks++}
 for(const [policy,base,other] of specs){const result={block:block.block,group,policy,baseline:base,other};
 for(const [name,total,hit] of [["ink","ink_pixels","true_positive_at_0_5"],["background","background_pixels","false_positive_at_0_5"]]){
 let n=0,b=0,lo=0,hi=0,pooledOther=other.map(()=>0);
 for(const c of cs){const m=c.metrics[base],a=other.map(x=>c.metrics[x][hit]),v=bounds(m[total],m[hit],a);n+=m[total];b+=m[hit];lo+=v[0];hi+=v[1];a.forEach((x,k)=>pooledOther[k]+=x)}
 const p=bounds(n,b,pooledOther);assert(lo>=p[0]&&hi<=p[1],"stratification weakened bounds");result[name]={n,baseline:b,added:[lo,hi],pooled_added:p,union:[b+lo,b+hi],gain_fraction:n?[lo/n,hi/n]:null};
 }rows.push(result)}
 }
 }
 return {marginal_checks:marginalChecks,rows}
}
const source=process.argv[2];if(!source)throw Error('usage: node audit.mjs <pinned results.json>');
const bytes=fs.readFileSync(source);const blob=crypto.createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex');
assert(blob==='403fd08e9619783438b67a73634850456b23ac94','unexpected source git blob');
const t=test(), a=run(JSON.parse(bytes));console.log(JSON.stringify({source:{commit:'8f2e576ff961a4f247fb545b5317b223988e57be',blob,path:'firstletters-colab/studio_pose_factors/evidence/results.json',archive_available:false},test:t,marginal_checks:a.marginal_checks,rows:a.rows,scope:'Retrospective sharp cardinality bounds, same block/core/support, no pixelwise overlap, no new inference or independent ink labels.'},null,2));
