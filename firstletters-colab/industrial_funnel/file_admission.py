"""A03 offline, single-operator admission verifier.

A pinned manifest bounds authorized attempts, not actual hardware. No dispatch.
File receipts remain evidence even when not reusable. Never export unit details.
"""
from __future__ import annotations
import json
from control_plane import (validate_spec,digest,require_sha,vector,add,zeros,within,
                           PREDECESSOR,INFERENCE,SCROLL)
from order_audit import is_experiment,assignment
from file_receipts import validate as validate_receipt

def validate_file_envelope(spec,attempt,envelope,store):
    """Recheck shared identity/receipt rules independently of a producer reader.

    A storage backend may implement an older or more permissive receipt schema.
    Its successful read is not evidence of compatibility with the shared policy.
    The store must implement the existing verify(SHA256) byte-integrity contract.
    """
    s=validate_spec(spec)
    if not isinstance(envelope,dict) or set(envelope)!={"schema","spec","receipt","sha256"}:
        raise ValueError("Envelope schema mismatch")
    if envelope["schema"]!="funnel.file_receipt.v1" or envelope["spec"]!=s:
        raise ValueError("Envelope spec or schema mismatch")
    if not isinstance(envelope["receipt"],dict):
        raise ValueError("Expected receipt object")
    if digest({k:v for k,v in envelope.items() if k!="sha256"})!=envelope["sha256"]:
        raise ValueError("Envelope hash mismatch")
    validate_receipt(s,attempt,envelope["receipt"],store)
    return envelope

def freeze_batch(epoch,eligible,cap,entries):
    if not isinstance(epoch,str) or not epoch or not isinstance(eligible,list) or not eligible:
        raise ValueError("Epoch and dated eligibility required")
    if len(set(eligible))!=len(eligible) or any(not isinstance(s,str) or not SCROLL.fullmatch(s) for s in eligible):
        raise ValueError("Invalid eligibility set")
    cap=vector(cap);rows={};reserved=zeros()
    if not isinstance(entries,list) or not entries:raise ValueError("Empty batch")
    for e in entries:
        if set(e)!={"spec","grants"}:raise ValueError("Invalid entry")
        s=validate_spec(e["spec"]);j=digest(s)
        if j in rows:raise ValueError("Duplicate job")
        if s["stage"]!="CONTROL" and s["scroll"] not in eligible:raise ValueError("Ineligible scroll")
        if not isinstance(e["grants"],list) or not e["grants"]:raise ValueError("Explicit authorized attempts required")
        gg=[vector(g) for g in e["grants"]]
        for g in gg:
            if not any(g.values()):raise ValueError("Zero grant")
            reserved=add(reserved,g)
        rows[j]={"spec":s,"grants":gg}
    if not within(reserved,cap):raise ValueError("Batch worst-case attempts exceed cap")
    for j,e in rows.items():
        s=e["spec"];parents=[]
        for p in s["deps"]:
            if p not in rows:raise ValueError("Unregistered parent")
            q=rows[p]["spec"];parents.append(q)
            if is_experiment(q):raise ValueError("Experimental result cannot parent standard stage")
            if q["stage"]!="CONTROL" and any(q[k]!=s[k] for k in ("scroll","unit","frame")):
                raise ValueError("Unregistered frame/unit transform")
        expected=PREDECESSOR[s["stage"]]
        req=([expected] if expected else [])+(["CONTROL"] if s["stage"] in INFERENCE else [])
        if sorted(q["stage"] for q in parents)!=sorted(req):raise ValueError("Stage lineage mismatch")
        if s["stage"] in INFERENCE and not any(q["stage"]=="CONTROL" and q["calibration"]==s["calibration"] for q in parents):
            raise ValueError("Calibration lineage mismatch")
    # Fixed stage predecessor makes cycles impossible; retain explicit cycle check.
    visiting=set();done=set()
    def visit(j):
        if j in visiting:raise ValueError("Cycle")
        if j in done:return
        visiting.add(j)
        for p in rows[j]["spec"]["deps"]:visit(p)
        visiting.remove(j);done.add(j)
    for j in rows:visit(j)
    obj={"schema":"funnel.file_batch.v1","epoch":epoch,"eligible":sorted(eligible),
         "cap":cap,"authorized_worst_case":reserved,"entries":rows}
    obj["manifest_hash"]=digest(obj)
    return obj

def validate_batch(manifest,pin):
    require_sha(pin)
    if not isinstance(manifest,dict) or manifest.get("manifest_hash")!=pin:raise ValueError("Untrusted batch pin")
    try:expected=freeze_batch(manifest["epoch"],manifest["eligible"],manifest["cap"],list(manifest["entries"].values()))
    except (KeyError,TypeError) as e:raise ValueError("Invalid manifest") from e
    if expected!=manifest:raise ValueError("Manifest changed")
    return manifest

def inspect_batch(files,manifest,pin,now,revoked=()):
    """Return private per-job state. Only a separately authorized operator may run.

    Conservative expiry check on ancestor controls: historical work is kept but
    not approved for new work after a control expires. The caller supplies trusted
    current time/revocation snapshot; there is no authenticated policy service.
    """
    if type(now)!=int:raise ValueError("Explicit inspection time required")
    revoked=set(revoked)
    for h in revoked:require_sha(h)
    m=validate_batch(manifest,pin);entries=m["entries"];states={};spent=zeros();over=False
    # Inspect every receipt before determining readiness, even failed/blocked jobs.
    for j,e in sorted(entries.items()):
        s=e["spec"];r={"state":"PENDING","receipt":None,"attempts":0,"next_attempt":1,"reason":None}
        states[j]=r
        try:
            paths=list(files.receipts.glob(j+".*.json"));numbers=[]
            for p in paths:
                name=p.name.split(".")
                if len(name)!=3 or not name[1].isdigit() or str(int(name[1]))!=name[1]:raise ValueError("Invalid attempt filename")
                numbers.append(int(name[1]))
            numbers.sort()
            if numbers!=list(range(1,len(numbers)+1)) or len(numbers)>len(e["grants"]):raise ValueError("Unauthorized/noncontiguous attempt")
            r["attempts"]=len(numbers);r["next_attempt"]=len(numbers)+1
            rows=[]
            for n in numbers:
                q=validate_file_envelope(s,n,files.read(s,n),files.store);rr=q["receipt"]
                spent=add(spent,rr["actual"])
                if rr["grant"]!=e["grants"][n-1]:raise ValueError("Grant differs from authorized manifest")
                if not within(rr["actual"],e["grants"][n-1]):over=True
                rows.append(q)
            complete=[q for q in rows if q["receipt"]["outcome"]=="COMPLETE"]
            if len(complete)>1 or (complete and complete[0] is not rows[-1]):
                raise ValueError("Conflicting completed attempts")
            if complete:r["receipt"]=complete[0]
            elif len(numbers)==len(e["grants"]):r.update(state="BLOCKED",reason="ATTEMPTS_EXHAUSTED")
            if is_experiment(s):assignment(s,files.store)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            r.update(state="BLOCKED",reason="RECEIPT_OR_SPEC_INVALID",detail=type(exc).__name__)
            # Unknown/corrupt costs cannot be declared zero or budget safe.
            over=True
    finished=set()
    def resolve(j):
        if j in finished:return states[j]
        r=states[j];s=entries[j]["spec"]
        parents=[resolve(p) for p in s["deps"]]
        if r["state"]=="BLOCKED":finished.add(j);return r
        if any(q["state"]!="COMPLETE" for q in parents):
            r.update(state="BLOCKED" if any(q["state"] in ("BLOCKED","EXPERIMENT_COMPLETE") for q in parents) else "WAITING_PARENT",
                     reason="PARENT_NOT_ADMITTED")
        else:
            try:
                for q in parents:
                    if not set(q["receipt"]["receipt"]["outputs"].values())<=set(s["inputs"].values()):
                        raise ValueError("Parent outputs not bound")
                for h in s["inputs"].values():files.store.verify(h)
                if s["stage"] in INFERENCE|{"CONTROL"} and s["calibration"] in revoked:
                    r.update(state="BLOCKED",reason="CONTROL_REVOKED")
                elif s["stage"]=="CONTROL" and r["receipt"] is not None:
                    rr=r["receipt"]["receipt"]
                    if not rr["control_pass"] or rr["valid_until"]<=now:r.update(state="BLOCKED",reason="CONTROL_FAILED_OR_STALE")
                    else:r["state"]="COMPLETE"
                elif r["receipt"] is not None:r["state"]="EXPERIMENT_COMPLETE" if is_experiment(s) else "COMPLETE"
                else:r["state"]="READY_FOR_OPERATOR"
            except (ValueError,KeyError,TypeError,OSError):
                r.update(state="BLOCKED",reason="INPUT_OR_LINEAGE_INVALID")
        finished.add(j);return r
    for j in entries:resolve(j)
    halted=over or not within(spent,m["cap"])
    if halted:
        for r in states.values():
            if r["state"] in ("READY_FOR_OPERATOR","WAITING_PARENT"):r.update(state="BLOCKED",reason="BATCH_INTEGRITY_OR_RESOURCE_STOP")
    return {"manifest_hash":pin,"states":states,"spent":spent,
            "authorized_worst_case":m["authorized_worst_case"],"halted":halted,
            "costs_complete":not over,"dispatch_authorized":False,
            "scope":"Batch-local checks; no cross-batch spending, liveness or execution authorization."}

def evidence_bound_audit(files,manifest,pin,plan_hash,adjudication_hashes,now,revoked=()):
    """Reduce private paired evidence; labels alone never establish an audit."""
    from order_audit import validate_plan,audit_summary,PAIRED
    plan=validate_plan(json.loads(files.store.verify(plan_hash).read_text()))
    inspected=inspect_batch(files,manifest,pin,now,revoked)
    entries=manifest["entries"];states=inspected["states"]
    index={(r["scroll"],r["unit"],r["frame"]):i for i,r in enumerate(plan["population"])}
    pairs={i for s in plan["strata"].values() for i in s["paired_indices"]}
    jobs={}
    for j,e in entries.items():
        s=e["spec"]
        if not is_experiment(s) or s["inputs"].get("order_audit_plan")!=plan_hash:continue
        key=(s["scroll"],s["unit"],s["frame"])
        if key not in index or index[key] in jobs:raise ValueError("Cohort/job bijection violated")
        i=index[key];assignment(s,files.store);jobs[i]=j
    if any(type(i)!=str or not i.isdigit() or str(int(i))!=i or int(i) not in pairs for i in adjudication_hashes):
        raise ValueError("Review outside planned paired population")
    labels={};invalid=0
    for i in sorted(pairs):
        if str(i) not in adjudication_hashes:continue
        j=jobs.get(i);r=states.get(j,{})
        try:
            if r.get("state")!="EXPERIMENT_COMPLETE":raise ValueError("Pair not admitted")
            s=entries[j]["spec"];rec=r["receipt"]
            if assignment(s,files.store)!=PAIRED:raise ValueError("Not assigned pair")
            q=json.loads(files.store.verify(adjudication_hashes[str(i)]).read_text())
            keys={"schema","plan_hash","index","job","receipt_hash","normal_order","producer","reviewer","outcome"}
            if set(q)!=keys or q["schema"]!="funnel.order_adjudication.v1" or q["plan_hash"]!=plan["plan_hash"] or type(q["index"])!=int or q["index"]!=i or q["job"]!=j or q["receipt_hash"]!=rec["sha256"] or q["normal_order"]!=plan["population"][i]["normal_order"]:
                raise ValueError("Adjudication not bound to exact pair")
            if any(not isinstance(q[k],str) or not q[k] for k in ("producer","reviewer")) or q["producer"]==q["reviewer"]:
                raise ValueError("Distinct declared reviewer required")
            allowed=("NO_SINGLE_ORDER_EXPOSURE","UNRESOLVED") if q["normal_order"]=="unknown" else ("MISSED_DETECTABLE_TEXT","NO_ADDITIONAL_DETECTABLE_TEXT","UNRESOLVED")
            if q["outcome"] not in allowed:raise ValueError("Invalid review outcome")
            labels[str(i)]=q["outcome"]
        except (ValueError,KeyError,TypeError,OSError):invalid+=1
    q=audit_summary(plan,labels)
    full=len(jobs)==len(index) and all(states[j]["state"]=="EXPERIMENT_COMPLETE" for j in jobs.values()) and not inspected["halted"]
    q["cohort_complete"]=full;q["planned_cohort_units"]=len(index)
    q["completed_cohort_units"]=sum(states[j]["state"]=="EXPERIMENT_COMPLETE" for j in jobs.values())
    q["invalid_adjudications"]=invalid
    q["evidence_complete"]=q["complete"] and full and invalid==0
    if not q["evidence_complete"]:
        q["complete"]=False;q["paired_investigation_required"]=True
        for k in ("HT_detectable_total","design_variance","simultaneous_zero_event_upper_count"):q[k]=None
        for r in q["strata"].values():
            r.pop("zero_event_upper_count",None);r["HT_detectable_total"]=None;r["design_variance"]=None
    q["evidence_verified"]=q["evidence_complete"]
    q["promotion_allowed"]=False
    q["evidence_scope"]="Verified byte/lineage/grant binding in one pinned batch; reviewer labels and scientific semantics not independently authenticated."
    return q
