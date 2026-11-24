"""Frozen stratified exploration, separate from ranking.

Draw first, never replace an expensive or inconvenient selected unit.
Inference targets detectable positives under an agreed independent review recipe,
not latent ink or the probability of winning. Population must be physically
deduplicated and frozen by a verified producer; this module cannot certify it.
"""
from __future__ import annotations
import hashlib
import math
import random
from control_plane import add, canonical, digest, vector, within, zeros

OUTCOMES={"DETECTABLE_POSITIVE","NO_QUALIFYING_READING","UNRESOLVED"}

def freeze_draw(population,allocations,seed,budget):
    if type(seed)!=int:raise ValueError("Frozen integer seed required")
    if not isinstance(population,list) or not population:raise ValueError("Nonempty frozen population required")
    rows=[];ids=set();groups={}
    for row in population:
        if not isinstance(row,dict) or set(row)!={"unit","stratum","grant"}:raise ValueError("Population schema mismatch")
        unit,stratum=row["unit"],row["stratum"]
        if not isinstance(unit,str) or not unit or unit in ids or not isinstance(stratum,str) or not stratum:
            raise ValueError("Duplicate or invalid physical survey unit")
        ids.add(unit);q={"unit":unit,"stratum":stratum,"grant":vector(row["grant"])}
        rows.append(q);groups.setdefault(stratum,[]).append(q)
    rows.sort(key=lambda q:q["unit"]);p_hash=digest(rows)
    if set(allocations)!=set(groups):raise ValueError("Every stratum must have a draw")
    total=zeros();strata={}
    for key,rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda q:q["unit"]);N=len(rr);n=allocations[key]
        if type(n)!=int or not 1<=n<=N:raise ValueError("Invalid stratum sample size")
        rng=random.Random(int(hashlib.sha256(canonical([seed,p_hash,key]).encode()).hexdigest(),16))
        picked=sorted(rng.sample(rr,n),key=lambda q:q["unit"])
        for q in picked:total=add(total,q["grant"])
        strata[key]={"N":N,"n":n,"inclusion_probability":n/N,"sample":[q["unit"] for q in picked]}
    cap=vector(budget)
    result={"schema":"funnel.survey.v1","population_hash":p_hash,"seed":seed,"strata":strata,
            "reserved":total,"budget":cap,"budget_feasible":within(total,cap)}
    result["draw_hash"]=digest(result)
    return result

def zero_upper_count(N,n,alpha):
    """Largest number of positives compatible with zero observed at level alpha.
    
    Exact SRS without replacement: P(X=0 | M) = choose(N-M,n)/choose(N,n).
    Invert the monotone probability in log space; no IID assumption for pixels.
    """
    if type(N)!=int or type(n)!=int or not 1<=n<=N or not 0<alpha<1:raise ValueError("Invalid bound parameters")
    if n==N:return 0
    target=math.log(alpha)
    def feasible(M):
        if M==0:return True
        if M>N-n:return False
        return math.fsum(math.log1p(-M/(N-i)) for i in range(n))>=target-1e-12
    lo,hi=0,N-n+1
    while hi-lo>1:
        mid=(lo+hi)//2
        if feasible(mid):lo=mid
        else:hi=mid
    return lo

def summarize(draw,results,alpha=.05):
    if not isinstance(draw,dict) or draw.get("draw_hash")!=digest({k:v for k,v in draw.items() if k!="draw_hash"}):
        raise ValueError("Frozen draw fingerprint mismatch")
    if not 0<alpha<1:raise ValueError("Invalid alpha")
    expected={u for s in draw["strata"].values() for u in s["sample"]}
    if set(results)-expected or any(v not in OUTCOMES for v in results.values()):
        raise ValueError("Unknown survey unit or outcome")
    rows={};all_done=bool(draw["budget_feasible"]);all_zero=True;ht=0.;var=0.;variance_available=True
    for key,s in draw["strata"].items():
        vals=[results.get(u,"UNRESOLVED") for u in s["sample"]]
        N,n=s["N"],s["n"];k=vals.count("DETECTABLE_POSITIVE");unresolved=vals.count("UNRESOLVED")
        complete=unresolved==0 and draw["budget_feasible"]
        all_done &= complete;all_zero &= k==0
        r={"N":N,"n":n,"reviewed":n-unresolved,"detected":k,"unresolved":unresolved,
           "inclusion_probability":s["inclusion_probability"],"HT_detectable_total":None,"design_variance":None}
        if complete:
            p=k/n;r["HT_detectable_total"]=N*p;ht+=N*p
            if n==N:vv=0.
            elif n>1:vv=N*N*(1-n/N)*p*(1-p)/(n-1)
            else:vv=None
            r["design_variance"]=vv
            if vv is None:variance_available=False
            else:var+=vv
        rows[key]=r
    # Simultaneous one-sided bounds only for a complete zero-event survey.
    # Positive events require a separately specified interval procedure.
    upper=None
    if all_done and all_zero:
        adj=alpha/len(rows)
        for r in rows.values():r["zero_event_upper_count"]=zero_upper_count(r["N"],r["n"],adj)
        upper=sum(r["zero_event_upper_count"] for r in rows.values())
    return {"complete":all_done,"population_units":sum(r["N"] for r in rows.values()),
            "sample_units":sum(r["n"] for r in rows.values()),
            "reviewed_units":sum(r["reviewed"] for r in rows.values()),"strata":rows,
            "HT_detectable_total":ht if all_done else None,
            "design_variance":var if all_done and variance_available else None,
            "simultaneous_zero_event_upper_count":upper,"alpha":alpha,
            "scope":"Detectable positives under the frozen review recipe; not latent ink, papyrus area, or prize probability."}
