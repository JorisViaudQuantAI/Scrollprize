"""074 read-only cost and audit-power analysis from pinned public T17 timings."""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
import unittest
from pathlib import Path
from survey import zero_upper_count
from control_plane import digest
from order_audit import freeze_plan

HERE=Path(__file__).resolve().parent
SOURCE_BLOB="610e115ac1c6775aebc92e435b0c53264ca97aad"
def no_hit(N,M,n):
    return math.comb(N-M,n)/math.comb(N,n) if N-M>=n else 0.
def min_audit(N,M,alpha=.05):
    return next(n for n in range(1,N+1) if no_hit(N,M,n)<alpha)
def timing_case(scroll,mode,data,render):
    orders=data["studio_s96"]["orders"]
    secs={o:float(orders["studio_"+o]["seconds"]) for o in ("asc","desc")}
    share={o:float(orders["studio_"+o]["frac_gt_07"]) for o in ("asc","desc")}
    if not all(math.isfinite(v) and v>0 for v in secs.values()):raise ValueError("Invalid timing")
    pair=sum(secs.values());arms=[]
    for o in ("asc","desc"):
        other="desc" if o=="asc" else "asc"
        for u in (0.,.25,.5):
            # A paired audit of 15% of the known-normal fraction plus certainty
            # pairs for all unknown normals. Homogeneous per-chart costs assumed.
            paired_fraction=u+(1-u)*.15
            expected=secs[o]+paired_fraction*secs[other]
            arms.append({"rule_order":o,"unknown_normal_fraction":u,
                "paired_fraction":paired_fraction,"expected_inference_seconds":expected,
                "inference_reduction_fraction":1-expected/pair,
                "render_plus_inference_reduction_fraction":None if render is None else 1-(render+expected)/(render+pair)})
    return {"scroll":scroll,"recipe":mode,"measured_s96_seconds":secs,
            "observed_share_gt07":share,"larger_share_order":max(share,key=share.get),
            "measured_paired_inference_seconds":pair,"measured_render_seconds":render,
            "counterfactuals":arms}
def run(source):
    raw=source.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if blob!=SOURCE_BLOB:raise ValueError("T17 source blob mismatch")
    j=json.loads(raw)
    cases=[timing_case("PHerc0211",mode,v,float(v["render_seconds"]))
           for mode,v in j["arm1_pherc0211_matrix"]["snap_modes"].items()]
    cases.append(timing_case("PHerc0306B","T17",j["arm2_pherc0306b_layer_card"],None))
    N=120;n=(15*N+99)//100
    upper=zero_upper_count(N,n,.05)
    brute=max(M for M in range(N+1) if no_hit(N,M,n)>=.05)
    if upper!=brute:raise AssertionError("Finite population bound mismatch")
    # Frozen small-stratum demonstration: 20 known asc, 20 desc, 2 unknown;
    # actual plan arithmetic, not Claude's measured population.
    recipes={k:{"recipe":digest(["r",k]),"calibration":digest(["c",k])} for k in ("paired","asc","desc")}
    population=[{"scroll":"PHerc0211","unit":f"synthetic-{i}-{k}","frame":digest("frame"),
                 "geometry":digest([i,k]),"normal_order":k,"stratum":"fixture"} 
                for k,c in (("asc",20),("desc",20),("unknown",2)) for i in range(c)]
    plan=freeze_plan(population,recipes,7401)
    return {"source_git_blob":blob,"source_sha256":hashlib.sha256(raw).hexdigest(),
            "cases":cases,"statistical_illustration":{"N":N,"n":n,"one_sided_alpha":.05,
             "probability_audit_misses_all":{str(M):no_hit(N,M,n) for M in (1,2,5,10)},
             "zero_event_upper_count":upper,"zero_event_upper_fraction":upper/N,
             "minimum_n_to_exclude_at_least_one_miss":min_audit(N,1),
             "minimum_n_to_exclude_at_least_six_misses":min_audit(N,6),
             "scope":"Hypothetical fixed cohort, uniform audit without replacement; no observed miss counts."},
             "synthetic_plan":{"population":len(population),
                               "paired_units":sum(s["n"] for s in plan["strata"].values()),
                               "singleton_stratum_pair_rate":1.},
             "scope":"Counterfactual timing arithmetic on three recorded public positive configurations. No new GPU run, rule prediction, measured detection recall, or full pipeline savings. Timing excludes unspecified acquisition/control/review overhead."}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_*.py")
    t=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"schema":"funnel.amendment074.v1","tests_run":t.testsRun,
            "failures":[{"test":str(x),"traceback":y} for x,y in t.failures],
            "errors":[{"test":str(x),"traceback":y} for x,y in t.errors],
            "skipped":len(t.skipped),"tests_pass":t.wasSuccessful() and not t.skipped,
            "source_sha256":{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()
                for n in ("AMENDMENT-002.md","order_audit.py","control_plane.py","file_receipts.py","survey.py","test_order_audit.py","test_control_plane.py","test_file_receipts.py","verify074.py")},
            "python":platform.python_version()}
    if report["tests_pass"]:
        try:report["analysis"]=run(a.source)
        except Exception as e:report["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    report["pass"]=report["tests_pass"] and "analysis" in report
    report["wall_seconds"]=time.monotonic()-start
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)
    sys.exit(0 if report["pass"] else 1)
