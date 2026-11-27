import fs from 'node:fs';
function assert(ok,msg){if(!ok)throw Error(msg)}
function validate(reg){
 assert(reg.as_of==="2026-11-26"&&reg.first_letters.remaining_awards===5,"unexpected snapshot");
 const fl=reg.first_letters.eligible;assert(fl.length===17&&new Set(fl.map(x=>x.id)).size===17&&new Set(fl.map(x=>x.volume)).size===17,"duplicate or missing entry");
 for(const x of fl)assert(/^PHerc\d{4}[AB]?$/.test(x.id)&&/^\d{14}$/.test(x.volume),"invalid identifier");
 assert(reg.grand_prize.eligible_ids.length===13,"grand list");
 const controls=reg.controls;assert(new Set(controls.map(x=>x.id)).size===controls.length,"duplicate control");
 for(const x of controls)assert(!fl.some(y=>y.id===x.id),"control leaked into FL eligibility");
 assert(controls.find(x=>x.id==="PHerc0846A").parent_group===controls.find(x=>x.id==="PHerc0846B").parent_group,"split parent");
 return reg;
}
function route(reg,id,volume,objective,verifiedOn){
 if(verifiedOn!==reg.as_of)return {decision:"RECHECK_OFFICIAL_RULES",dispatch_allowed:false};
 if(!["first_letters","grand_prize","validation"].includes(objective))return {decision:"UNKNOWN_OBJECTIVE",dispatch_allowed:false};
 const control=reg.controls.find(x=>x.id===id);
 const entry=reg.first_letters.eligible.find(x=>x.id===id);
 if(objective==="first_letters")return {decision:entry&&entry.volume===volume?"ELIGIBLE_FOR_REVIEW":entry?"VOLUME_MISMATCH":control?"CONTROL_NOT_FIRST_LETTERS":"NOT_ON_ALLOWLIST",dispatch_allowed:false};
 if(objective==="grand_prize")return {decision:reg.grand_prize.eligible_ids.includes(id)?"GRAND_PRIZE_ID_REVIEW_VERIFY_VOLUME":"NOT_ON_ALLOWLIST",dispatch_allowed:false};
 return {decision:control?"PUBLIC_CONTROL_REVIEW_PROVENANCE":"UNREGISTERED_CONTROL",parent_group:control?.parent_group||null,dispatch_allowed:false};
}
function selftest(reg){
 validate(reg);const d=reg.as_of;let n=0;
 for(const x of reg.first_letters.eligible){assert(route(reg,x.id,x.volume,"first_letters",d).decision==="ELIGIBLE_FOR_REVIEW","valid entry");assert(route(reg,x.id,"bad","first_letters",d).decision==="VOLUME_MISMATCH","wrong volume");n+=2}
 for(const x of reg.controls){assert(route(reg,x.id,"","first_letters",d).decision==="CONTROL_NOT_FIRST_LETTERS","control routing");assert(route(reg,x.id,"","validation",d).decision==="PUBLIC_CONTROL_REVIEW_PROVENANCE","control validation");n+=2}
 for(const id of ["PHerc0211","PHerc1447"]){assert(route(reg,id,"","grand_prize",d).decision==="GRAND_PRIZE_ID_REVIEW_VERIFY_VOLUME","grand not preserved");n++}
 assert(route(reg,"PHerc0175A",reg.first_letters.eligible[1].volume,"first_letters","2026-11-27").decision==="RECHECK_OFFICIAL_RULES","stale snapshot");n++;
 assert(route(reg,"PHerc9999","","first_letters",d).decision==="NOT_ON_ALLOWLIST","unknown entry");n++;
 assert(route(reg,"PHerc0175A",reg.first_letters.eligible[1].volume,"bad",d).decision==="UNKNOWN_OBJECTIVE","unknown objective");n++;
 assert(route(reg,"PHerc0175A",reg.first_letters.eligible[1].volume,"first_letters",d).dispatch_allowed===false,"automatic dispatch");n++;
 return n;
}
const reg=validate(JSON.parse(fs.readFileSync(new URL('./prize_registry_20261126.json',import.meta.url),'utf8')));
const args=process.argv.slice(2);
if(args[0]==='--selftest'){console.log(JSON.stringify({passed:selftest(reg)}));}else{if(args.length!==4)throw Error('usage: node route.mjs ID VOLUME OBJECTIVE VERIFIED_ON');console.log(JSON.stringify(route(reg,...args)));}
