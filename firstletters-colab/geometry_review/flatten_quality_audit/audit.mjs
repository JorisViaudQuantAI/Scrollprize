// B8 public 0211 paired chart audit; no candidate data or new inference.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
function audit(raw,flat,windows,key){
 const assert=(x,m)=>{if(!x)throw Error(m)},round=x=>Math.round(x*1e6)/1e6,mean=xs=>xs.reduce((s,x)=>s+x,0)/xs.length;
 function normalize(obj){
  const m=new Map;
  for(const [k,v] of Object.entries(obj)){
   const id=k.split("/").at(-1),a=v.all;
   assert(/^w\d{3}$/.test(id)&&!m.has(id)&&a&&Number.isInteger(a.cells)&&a.cells>0,"bad chart identity/cell count");
   for(const f of ["shear8","stretch_turn","stretch_z","rough"]){assert(Array.isArray(a[f])&&a[f].length===3&&a[f].every(Number.isFinite),"bad quality vector")}
   for(const f of ["shear8","stretch_turn","stretch_z"]){assert(a[f][0]<=a[f][1]&&a[f][1]<=a[f][2],"unordered percentiles")}
   assert(a.shear8_abs_le8>=0&&a.shear8_abs_le8<=1&&a.rough[2]>=0&&a.rough[2]<=1&&a.rough[1]>=a.rough[0]&&a.stretch_turn[0]>0&&a.stretch_z[0]>0,"bad quality measure");
   m.set(id,a);
  }
  return m;
 }
 const a=normalize(raw),b=normalize(flat),ids=[...a.keys()].sort();
 assert(ids.length===60&&b.size===60&&ids.every(x=>b.has(x)),"expected matched 60 charts");
 const wkey=w=>w.winding.replace(/^fw/,"w");
 assert(windows.length===149&&new Set(windows.map(w=>w.window)).size===149&&windows.every(w=>b.has(wkey(w))&&Number.isFinite(w.valid_cm2)&&w.valid_cm2>0),"invalid windows");
 assert(new Set(windows.map(wkey)).size===60,"changed chart/window scope");
 const seen=new Set(Object.values(key));
 assert(seen.size===40&&[...seen].every(id=>windows.some(w=>w.window===id)),"invalid review key");
 const flagKeys=["A19","A16","A02"];
 assert(flagKeys.every(k=>typeof key[k]==="string"),"flag keys absent");
 const flags=new Set(flagKeys.map(k=>key[k]));
 const metricsFn={shear_share:x=>x.shear8_abs_le8,shear_central_span:x=>x.shear8[2]-x.shear8[0],stretch_turn_span:x=>x.stretch_turn[2]-x.stretch_turn[0],stretch_z_span:x=>x.stretch_z[2]-x.stretch_z[0],rough_gt20_share:x=>x.rough[2],rough_p95:x=>x.rough[1],cells:x=>x.cells};
 const metrics={};
 for(const [k,f] of Object.entries(metricsFn)){
  const before=ids.map(id=>f(a.get(id))),after=ids.map(id=>f(b.get(id))),d=after.map((v,i)=>v-before[i]);
  metrics[k]={raw_chart_macro:round(mean(before)),flat_chart_macro:round(mean(after)),mean_delta:round(mean(d)),increased:d.filter(x=>x>1e-10).length,decreased:d.filter(x=>x< -1e-10).length,tied:d.filter(x=>Math.abs(x)<=1e-10).length,flat_min:round(Math.min(...after)),flat_max:round(Math.max(...after))};
 }
 const area=xs=>round(xs.reduce((s,w)=>s+w.valid_cm2,0));
 function gateReport(pass){
  const keep=windows.filter(w=>pass(b.get(wkey(w)))),drop=windows.filter(w=>!pass(b.get(wkey(w))));
  return {pass_charts:ids.filter(id=>pass(b.get(id))).length,excluded_charts:ids.filter(id=>!pass(b.get(id))).length,excluded_windows:drop.length,excluded_summed_window_area_cm2:area(drop),excluded_reviewed_windows:drop.filter(w=>seen.has(w.window)).length,excluded_published_visual_flags:drop.filter(w=>flags.has(w.window)).length,excluded_unreviewed_windows:drop.filter(w=>!seen.has(w.window)).length,excluded_abstention_windows:drop.filter(w=>w.line===null).length,kept_windows:keep.length};
 }
 const thresholds=[.90,.95,.99].map(t=>({threshold:t,...gateReport(x=>x.shear8_abs_le8>=t)}));
 const stretch=x=>[x.stretch_turn,x.stretch_z].every(v=>v[0]>=.95&&v[2]<=1.05);
 const cellTotals={raw:ids.reduce((s,id)=>s+a.get(id).cells,0),flat:ids.reduce((s,id)=>s+b.get(id).cells,0)};
 const allAbstain=ids.filter(id=>windows.filter(w=>wkey(w)===id).every(w=>w.line===null));
 const scored=ids.filter(id=>windows.filter(w=>wkey(w)===id).some(w=>w.line!==null));
 const qualityFor=g=>({charts:g.length,flat_shear_share_macro:g.length?round(mean(g.map(id=>b.get(id).shear8_abs_le8))):null,flat_rough_gt20_share_macro:g.length?round(mean(g.map(id=>b.get(id).rough[2]))):null});
 let tests=0;
 const example=b.get(ids[0]);
 const badInputs=[{w010:{all:{...example,shear8:[1,0,2]}}},{w010:{all:{...example,cells:0}}},{w010:{all:{...example,shear8_abs_le8:1.1}}},{w010:{all:example},"fitted/w010":{all:example}}];
 for(const v of badInputs){let failed=false;try{normalize(v)}catch{failed=true}assert(failed,"malformed chart accepted");tests++}
 assert(thresholds.every(t=>t.excluded_windows+t.kept_windows===149),"partition mismatch");tests++;
 return {schema:"b8.flatten_quality_audit.v1",paired_charts:ids.length,tests_passed:tests,metrics,cell_totals:cellTotals,cell_count_ratio:round(cellTotals.flat/cellTotals.raw),hypothetical_flat_shear_gates:thresholds,hypothetical_rounded_central_stretch_gate:{range:[.95,1.05],...gateReport(stretch)},all_abstain_quality:qualityFor(allAbstain),some_score_quality:qualityFor(scored),limitations:["Matched chart IDs, not matched vertices or cell masks; changed cell counts mean macro per-chart summaries are not a pooled physical area or pixel-level effect.","gridq cells counts one-step valid cells, not the exact shear or rough denominators; do not cell-weight those fractions as exact pooled shares.","Stretch percentiles rounded to 0.01; central range gate descriptive, not an exact fraction or joint confidence statement.","Published visual flags include weak/unverified and known-location observations, not independent ink truth. No validated exclusion threshold or new review."]};
}
const names=["tables/wl_259_gate_final__gridq_fitted_w010_069.json","tables/flat211f__gridq_flat.json","maps_export211/windows.json","look/P1/KEY_sealed.json"],pins=["7a04049ad8aab9d56013db0cc2dd4409b835094b","1e715cc751cbe855ef4d8f4bef8d18ec9a1f57cb","d6a337f299fa5b893dbc84f6de23542adbbd83a6","7aa64aa902924bff95bdec3f5cb9360154e8e0f5"];
const inputs=names.map((n,i)=>{let b=fs.readFileSync(path.join(process.argv[2],n));let h=crypto.createHash("sha1").update("blob "+b.length+"\0").update(b).digest("hex");if(h!==pins[i])throw Error("pin mismatch "+n);return JSON.parse(b)});
console.log(JSON.stringify(audit(...inputs),null,2));
