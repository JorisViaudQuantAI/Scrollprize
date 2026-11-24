"""A02 experimental order allocation, no production promotion.

A plan binds the complete supplied registered-chart population, the normal rule,
three recipe/calibration pairs and a pre-outcome audit draw. Unknown normals always
get both orders. The manifest is private. No claims about anatomical correctness.
"""
from __future__ import annotations
import hashlib
import json
import math
import random

MODE = "single_order_plus_audit"
PAIRED = ["asc", "desc"]
ROLES = {"asc_map", "desc_map", "asc_support", "desc_support"}

def freeze_plan(population, recipes, seed):
    from control_plane import SCROLL, canonical, digest, require_sha
    if type(seed) is not int or not isinstance(population,list) or not population:
        raise ValueError("Nonempty population and integer seed required")
    if not isinstance(recipes,dict) or set(recipes)!={"paired","asc","desc"}:
        raise ValueError("Three explicit recipes required")
    for r in recipes.values():
        if not isinstance(r,dict) or set(r)!={"recipe","calibration"}:raise ValueError("Invalid recipe record")
        for h in r.values():require_sha(h)
    for k in ("recipe","calibration"):
        if len({r[k] for r in recipes.values()})!=3:raise ValueError("Distinct recipe/calibration keys required")
    rows=[];seen=set()
    for r in population:
        if not isinstance(r,dict) or set(r)!={"scroll","unit","frame","geometry","normal_order","stratum"}:
            raise ValueError("Invalid population record")
        if not isinstance(r["scroll"],str) or not SCROLL.fullmatch(r["scroll"]):
            raise ValueError("Invalid scroll")
        if not isinstance(r["unit"],str) or not 0<len(r["unit"])<=128 or not isinstance(r["stratum"],str) or not r["stratum"]:
            raise ValueError("Invalid private chart/stratum")
        for k in ("frame","geometry"):require_sha(r[k])
        if r["normal_order"] not in ("asc","desc","unknown"):raise ValueError("Invalid normal-order assertion")
        # A chart may not be listed twice under different geometry hypotheses in
        # one plan; distinct immutable chart IDs are needed for distinct versions.
        key=(r["scroll"],r["unit"],r["frame"])
        if key in seen:raise ValueError("Duplicate chart in audit population")
        seen.add(key);rows.append(dict(r))
    rows.sort(key=lambda r:(r["scroll"],r["unit"],r["frame"]))
    ph=digest(rows);groups={}
    for i,r in enumerate(rows):
        k=canonical([r["scroll"],r["stratum"],r["normal_order"]])
        groups.setdefault(k,[]).append(i)
    strata={}
    for k,ids in sorted(groups.items()):
        N=len(ids);unknown=json.loads(k)[2]=="unknown"
        n=N if unknown else (15*N+99)//100
        rng=random.Random(int(hashlib.sha256(canonical([seed,ph,k]).encode()).hexdigest(),16))
        chosen=sorted(rng.sample(ids,n))
        strata[k]={"N":N,"n":n,"probability":n/N,"paired_indices":chosen}
    obj={"schema":"funnel.order_audit.v1","mode":MODE,"population":rows,"population_hash":ph,
         "recipes":recipes,"seed":seed,"audit_rate_percent":15,"strata":strata}
    obj["plan_hash"]=digest(obj)
    return obj

def validate_plan(plan):
    if not isinstance(plan,dict):raise ValueError("Invalid order audit plan")
    try:expected=freeze_plan(plan["population"],plan["recipes"],plan["seed"])
    except (KeyError,TypeError) as e:raise ValueError("Invalid order audit plan") from e
    if expected!=plan:raise ValueError("Frozen order audit draw mismatch")
    return plan

def is_experiment(spec):
    return spec.get("order_policy")==MODE

def assignment(spec,store):
    """Return exact expected orders from the immutable plan, never from a worker."""
    if not is_experiment(spec):return list(PAIRED)
    if spec["stage"]!="COARSE":raise ValueError("Experimental order policy only for COARSE")
    from control_plane import canonical
    try:p=store.verify(spec["inputs"]["order_audit_plan"]);plan=validate_plan(json.loads(p.read_text()))
    except (KeyError,TypeError) as e:raise ValueError("Missing frozen order audit plan") from e
    matches=[(i,r) for i,r in enumerate(plan["population"]) if all(r[k]==spec[k] for k in ("scroll","unit","frame"))]
    if len(matches)!=1:raise ValueError("Chart absent from frozen audit population")
    i,r=matches[0]
    if r["geometry"]!=spec["inputs"].get("geometry"):raise ValueError("Frozen geometry mismatch")
    key=canonical([r["scroll"],r["stratum"],r["normal_order"]])
    paired=i in plan["strata"][key]["paired_indices"]
    arm="paired" if paired else r["normal_order"]
    if arm not in plan["recipes"]:raise ValueError("Unknown normal requires paired")
    if any(spec[k]!=plan["recipes"][arm][k] for k in ("recipe","calibration")):
        raise ValueError("Frozen order recipe/calibration mismatch")
    return list(PAIRED) if paired else [arm]

def validate_outputs(spec,receipt,store):
    expected=assignment(spec,store)
    outputs=receipt["outputs"]
    needed={o+"_"+suffix for o in expected for suffix in ("map","support")}
    if receipt.get("orders")!=expected or not needed|{"metadata"}<=set(outputs):
        raise ValueError("Incomplete or mismatched assigned-order evidence")
    if is_experiment(spec) and set(outputs)&ROLES!=needed:
        raise ValueError("Unassigned order cannot masquerade as measured")

def audit_summary(plan, adjudications, alpha=.05):
    """Unchecked mathematical reducer: miss / no miss / unresolved.

    Operational callers must use file_admission.evidence_bound_audit.
    
    No-hit estimates concern the frozen eligible cohort and reference recipe.
    We reuse survey.summarize, not IID inference on pixels or known letters.
    """
    from control_plane import digest
    from survey import summarize
    p=validate_plan(plan)
    strata={};unknown=set();known=set()
    for k,v in p["strata"].items():
        sample=[str(i) for i in v["paired_indices"]]
        if json.loads(k)[2]=="unknown":
            unknown.update(sample);continue
        known.update(sample)
        strata[k]={"N":v["N"],"n":v["n"],"inclusion_probability":v["probability"],"sample":sample}
    if set(adjudications)-known-unknown:raise ValueError("Adjudication outside paired cohort")
    allowed={"MISSED_DETECTABLE_TEXT":"DETECTABLE_POSITIVE","NO_ADDITIONAL_DETECTABLE_TEXT":"NO_QUALIFYING_READING","UNRESOLVED":"UNRESOLVED"}
    if any(adjudications[i] not in allowed for i in known&set(adjudications)):
        raise ValueError("Invalid known-order adjudication")
    if any(adjudications[i] not in ("NO_SINGLE_ORDER_EXPOSURE","UNRESOLVED") for i in unknown&set(adjudications)):
        raise ValueError("Unknown normal has no single-order counterfactual")
    unknown_reviewed=sum(adjudications.get(i)=="NO_SINGLE_ORDER_EXPOSURE" for i in unknown)
    unknown_complete=unknown_reviewed==len(unknown)
    d={"schema":"funnel.survey.v1","population_hash":p["population_hash"],"seed":p["seed"],
       "strata":strata,"budget_feasible":True}
    d["draw_hash"]=digest(d)
    if strata:
        q=summarize(d,{i:allowed[adjudications[i]] for i in known&set(adjudications)},alpha)
    else:
        q={"complete":True,"population_units":0,"sample_units":0,"reviewed_units":0,"strata":{},
           "HT_detectable_total":None,"design_variance":None,"simultaneous_zero_event_upper_count":None,"alpha":alpha}
    q["complete"]=q["complete"] and unknown_complete
    if not q["complete"]:
        q["simultaneous_zero_event_upper_count"]=None
        q["HT_detectable_total"]=None;q["design_variance"]=None
        for r in q["strata"].values():r.pop("zero_event_upper_count",None)
    q["unknown_order_paired_units"]=len(unknown);q["unknown_order_pair_adjudicated"]=unknown_reviewed
    q["scope"]="Only extra detectable-text misses under assigned normal rule in frozen known-order cohort. Unknown-order units paired by design. Adjudications are externally supplied, not independently authenticated by this helper. Not latent ink or population recall."
    q["evidence_verified"]=False
    q["experiment_only"]=True
    q["promotion_allowed"]=False
    q["paired_investigation_required"]=any(v=="MISSED_DETECTABLE_TEXT" for v in adjudications.values()) or not q["complete"]
    return q
