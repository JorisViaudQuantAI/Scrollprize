// B6: exact minimax zero-event audit design for the fixed public B5 cohort.
import fs from "node:fs";
import crypto from "node:crypto";
function design(windows,key){
 const seen=new Set(Object.values(key)), touched=new Set(windows.filter(w=>seen.has(w.window)).map(w=>w.winding));
 const strata=["touched_scored","touched_abstain","untouched_scored","untouched_abstain"];
 const counts=strata.map(s=>windows.filter(w=>!seen.has(w.window)&&((touched.has(w.winding)?"touched":"untouched")+"_"+(w.line===null?"abstain":"scored"))===s).length);
 const N=counts.reduce((s,n)=>s+n,0);
 function probability(ns,ks,m){
  if(ns.length!==ks.length||!Number.isInteger(m)||m<0||m>ns.reduce((s,n)=>s+n,0)||ns.some((n,i)=>!Number.isInteger(n)||!Number.isInteger(ks[i])||ks[i]<0||ks[i]>n))throw Error("invalid counts");
  // Greedy maximization of product over flagged-count allocations. Each stratum's
  // marginal ratio (N_h-n_h-m_h)/(N_h-m_h) is nonincreasing, so selecting
  // the largest remaining marginal at each step is globally optimal.
  const miss=ns.map(()=>0);let num=1n,den=1n;
  for(let j=0;j<m;j++){let best=-1,bn=0,bd=1;
   for(let i=0;i<ns.length;i++){let d=ns[i]-miss[i],n=Math.max(0,d-ks[i]);if(d>0&&(best<0||n*bd>bn*d)){best=i;bn=n;bd=d}}
   miss[best]++;num*=BigInt(bn);den*=BigInt(bd);
  }
  return {num,den,miss,p:Number(num)/Number(den)};
 }
 const less=(a,b)=>a.num*b.den<b.num*a.den;
 const rejected=q=>20n*q.num<q.den;
 function optimum(budget,m){
  let best=null;
  function visit(i,left,ks){
   if(i===counts.length){if(left)return;const q=probability(counts,ks,m);if(!best||less(q,best.q)||(!less(best.q,q)&&!less(q,best.q)&&ks.reduce((s,k,j)=>s+(k/counts[j]-budget/N)**2,0)<best.imbalance)){best={ks:[...ks],q,imbalance:ks.reduce((s,k,j)=>s+(k/counts[j]-budget/N)**2,0)}}return;}
   // Every nonempty stratum gets at least one draw; no unsampled risk bucket.
   const lo=Math.max(1,left-counts.slice(i+1).reduce((s,n)=>s+n,0)),hi=Math.min(counts[i],left-(counts.length-i-1));
   for(let k=lo;k<=hi;k++)visit(i+1,left-k,[...ks,k]);
  }
  visit(0,budget,[]);return best;
 }
 function record(budget,m){const x=optimum(budget,m),s=probability([N],[budget],m);return {budget,target_missed:m,allocation:Object.fromEntries(strata.map((v,i)=>[v,x.ks[i]])),worst_flag_allocation:Object.fromEntries(strata.map((v,i)=>[v,x.q.miss[i]])),worst_p_zero:x.q.p,worst_p_zero_exact:`${x.q.num}/${x.q.den}`,reject_if_zero_95pct:rejected(x.q),srs_p_zero:s.p,srs_reject_if_zero_95pct:rejected(s)}}
 const plans=[];for(const budget of [30,45,60])for(const m of [3,6,12])plans.push(record(budget,m));
 const minSRS={};for(const m of [1,3,6,12]){let n=0;while(!rejected(probability([N],[n],m)))n++;minSRS[m]=n;}
 // Independent exhaustive oracle over small allocation spaces.
 let checks=0;
 function brute(ns,ks,m){let best=null;function rec(i,left,x){if(i===ns.length){if(left)return;let num=1n,den=1n;for(let h=0;h<ns.length;h++)for(let j=0;j<x[h];j++){num*=BigInt(Math.max(0,ns[h]-ks[h]-j));den*=BigInt(ns[h]-j)}const q={num,den};if(!best||less(best,q))best=q;return}for(let j=0;j<=Math.min(left,ns[i]);j++)rec(i+1,left-j,[...x,j])}rec(0,m,[]);return best;}
 for(let a=1;a<=4;a++)for(let b=1;b<=4;b++)for(let ka=0;ka<=a;ka++)for(let kb=0;kb<=b;kb++)for(let m=0;m<=a+b;m++){let q=probability([a,b],[ka,kb],m),v=brute([a,b],[ka,kb],m);if(less(q,v)||less(v,q))throw Error("greedy differs from exhaustive oracle");checks++}
 if(counts.join(",")!=="23,23,38,25"||N!==109)throw Error("changed public cohort");
 return {schema:"b6.stratified_zero_event_design.v1",remaining:N,stratum_counts:Object.fromEntries(strata.map((s,i)=>[s,counts[i]])),oracle_cases_passed:checks,plans,minimum_additional_srs_reviews_for_zero_event_exclusion:minSRS,assumptions:["Conditional on the fixed existing 40-window review; new labels must be blinded, consistently defined, and fixed before sampling.","Independent uniform samples without replacement within each frozen stratum; no replacement of difficult windows; no peeking/optional stopping.","At each target m, the worst-case p0 maximizes over all allocations of at least m flagged windows; monotonicity makes m sufficient. Reject only for exact p<0.05, if zero flags.","Counts are window-level visual flags, not ink recall, physical sites or independence across windings. A chart-touch constraint requires a separate design; do not apply these p-values to one-per-winding sampling.","No real additional review or GPU cost measured; design only."]};
}
function pinned(path,sha){const b=fs.readFileSync(path);const h=crypto.createHash("sha1").update("blob "+b.length+"\0").update(b).digest("hex");if(h!==sha)throw Error("source pin mismatch");return JSON.parse(b);}
console.log(JSON.stringify(design(pinned(process.argv[2],"d6a337f299fa5b893dbc84f6de23542adbbd83a6"),pinned(process.argv[3],"7aa64aa902924bff95bdec3f5cb9360154e8e0f5")),null,2));
