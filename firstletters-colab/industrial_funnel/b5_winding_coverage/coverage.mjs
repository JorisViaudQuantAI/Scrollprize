// B5 public already-awarded control. No new inference or ink labels.
import fs from "node:fs";
import crypto from "node:crypto";
function coverage(windows,key){
 const assert=(c,m)=>{if(!c)throw Error(m)};
 const round=x=>Math.round(x*1e6)/1e6;
 const ids=new Set(windows.map(x=>x.window)), ranks=new Set(windows.map(x=>x.rank));
 assert(windows.length===ids.size&&windows.length===ranks.size&&windows.every(w=>Number.isInteger(w.rank)&&w.rank>=1&&w.rank<=windows.length),"bad identities");
 assert(windows.every(w=>typeof w.winding==="string"&&Number.isInteger(w.x0)&&Number.isInteger(w.x1)&&w.x1>w.x0&&Number.isFinite(w.valid_cm2)&&w.valid_cm2>0&&(w.line===null?w.lr===-99:Number.isFinite(w.line)&&w.lr===w.line)),"invalid record");
 const reviewed=new Set(Object.values(key));assert(reviewed.size===Object.keys(key).length&&[...reviewed].every(x=>ids.has(x)),"bad sealed key");
 const by=new Map();for(const w of windows){if(!by.has(w.winding))by.set(w.winding,[]);by.get(w.winding).push(w)}
 const union=ws=>{let end=-Infinity,total=0;for(const w of [...ws].sort((x,y)=>x.x0-y.x0||x.x1-y.x1)){total+=Math.max(0,w.x1-Math.max(end,w.x0));end=Math.max(end,w.x1)}return total};
 const sumArea=ws=>round(ws.reduce((s,w)=>s+w.valid_cm2,0));
 const groups=[...by.values()];const partition=sel=>{
 let rows=windows.filter(sel),counts=groups.map(g=>g.filter(sel).length), totalUnion=groups.reduce((s,g)=>s+union(g),0), selectedUnion=groups.reduce((s,g)=>s+union(g.filter(sel)),0);
 return {windows:rows.length,window_area_sum_cm2:sumArea(rows),windings_touched:counts.filter(n=>n>0).length,windings_complete:counts.filter((n,i)=>n===groups[i].length).length,windings_untouched:counts.filter(n=>n===0).length,chart_x_union_px:selectedUnion,chart_x_union_fraction:round(selectedUnion/totalUnion),no_line_score:rows.filter(w=>w.line===null).length};
 };
 const top=w=>w.rank<=25, seen=w=>reviewed.has(w.window);
 const rest=windows.filter(w=>!seen(w)), untouched=groups.filter(g=>!g.some(seen));
 const noScoreOnly=groups.filter(g=>g.every(w=>w.line===null));
 const reviewedWindings=new Set(windows.filter(seen).map(w=>w.winding));
 const strata={};for(const w of rest){let k=(reviewedWindings.has(w.winding)?"touched":"untouched")+"_"+(w.line===null?"abstain":"scored");if(!strata[k])strata[k]={windows:0,area_cm2:0};strata[k].windows++;strata[k].area_cm2+=w.valid_cm2}for(const s of Object.values(strata))s.area_cm2=round(s.area_cm2);
 const total=partition(()=>true);
 return {schema:"b5.winding_coverage.v1",all:total,top25:partition(top),sealed_review:partition(seen),unreviewed:partition(w=>!seen(w)),no_line_score:partition(w=>w.line===null),all_abstain_windings:noScoreOnly.length,untouched_all_abstain_windings:untouched.filter(g=>g.every(w=>w.line===null)).length,remaining_strata:strata,minimum_additional_reviews_for_chart_touch:untouched.length,untouched_window_count:untouched.reduce((s,g)=>s+g.length,0),untouched_area_cm2:sumArea(untouched.flat()),sum_window_width_px:windows.reduce((s,w)=>s+w.x1-w.x0,0),limitations:["Window area sum is not unique physical surface area; chart x-interval union is one-dimensional, not valid-pixel coverage or anatomy.","A touched winding can still contain missed ink; no new review labels, no ink recall or deployment claim."]};
}
function test(){
 const w=(rank,winding,x0,x1,line=1)=>({window:`w${rank}`,rank,winding,x0,x1,valid_cm2:1,line,lr:line===null?-99:line});
 const ws=[w(1,"a",0,10),w(2,"a",5,15),w(3,"b",2,4,null),w(4,"c",0,1,null)];
 const v=coverage(ws,{A:"w1"});
 if(v.all.chart_x_union_px!==18||v.sealed_review.chart_x_union_px!==10||v.minimum_additional_reviews_for_chart_touch!==2||v.untouched_all_abstain_windings!==2||v.sum_window_width_px!==23)throw Error("union/coverage regression");
 let n=1;for(const [ww,k] of [[ws,{A:"missing"}],[[...ws,ws[0]],{A:"w1"}],[[{...ws[0],x1:-1},...ws.slice(1)],{A:"w1"}],[[{...ws[0],lr:-99},...ws.slice(1)],{A:"w1"}],[ws,{A:"w1",B:"w1"}]]){let failed=false;try{coverage(ww,k)}catch{failed=true}if(!failed)throw Error("bad input accepted");n++}return n;
}
function pinned(path,sha){const b=fs.readFileSync(path);const h=crypto.createHash("sha1").update("blob "+b.length+"\0").update(b).digest("hex");if(h!==sha)throw Error("source pin mismatch");return JSON.parse(b);}
const tests=test();
if(process.argv.includes("--self-test")){console.log(JSON.stringify({tests_passed:tests}));}else{const result=coverage(pinned(process.argv[2],"d6a337f299fa5b893dbc84f6de23542adbbd83a6"),pinned(process.argv[3],"7aa64aa902924bff95bdec3f5cb9360154e8e0f5"));console.log(JSON.stringify({tests_passed:tests,source_commit:"1770a7b9f223a016167fda440f1867848373119d",result},null,2));}
