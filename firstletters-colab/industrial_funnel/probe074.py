"""Reproduce pre-A03 gaps using isolated, pinned A02 modules and synthetic bytes."""
import json
from pathlib import Path
import tempfile
import sys
from control_plane import digest,canonical
from file_receipts import FileReceipts,job_key
from order_audit import freeze_plan,audit_summary
def res(n):return dict(cpu_ms=n,gpu_ms=0,human_ms=0,io_bytes=0)
def run():
    with tempfile.TemporaryDirectory() as tmp:
        f=FileReceipts(Path(tmp)/"private")
        s=dict(scroll="PHerc0125",unit="fixture",frame=digest("frame"),stage="COARSE",recipe=digest("recipe"),calibration=digest("cal"),deps=[digest("absent-parent")],inputs={"geometry":f.store.put(b"synthetic")})
        receipt=dict(job=job_key(s),attempt=1,outcome="COMPLETE",worker_stopped=True,grant=res(1000),actual=res(100),outputs={k:f.store.put(k.encode()) for k in ("metadata","asc_map","desc_map","asc_support","desc_support")},failure=None,orders=["asc","desc"])
        f.publish(s,1,receipt)
        a=f.resume(s,now=100)
    p=freeze_plan([dict(scroll="PHerc0125",unit=f"fixture-{i}",frame=digest("frame"),geometry=digest(i),normal_order="asc",stratum="fixture") for i in range(120)],{k:dict(recipe=digest(["recipe",k]),calibration=digest(["cal",k])) for k in ("asc","desc","paired")},7401)
    pair={i for q in p["strata"].values() for i in q["paired_indices"]}
    b=audit_summary(p,{str(i):"NO_ADDITIONAL_DETECTABLE_TEXT" for i in pair})
    return {"synthetic_only":True,"local_resume_completed_with_absent_parent":a["completed"],"old_chain_eligible_with_absent_parent":a["eligible_for_standard_chain"],"self_declared_grant_cpu_ms":1000,"fixture_external_expected_grant_cpu_ms":10,"paired_maps_executed":0,"audit_labels":len(pair),"old_audit_complete_without_evidence":b["complete"],"old_zero_event_upper_count_without_evidence":b["simultaneous_zero_event_upper_count"],"reproduced":bool(a["eligible_for_standard_chain"] and b["complete"] and b["simultaneous_zero_event_upper_count"]==17)}
if __name__=="__main__":
    q=run();Path(sys.argv[1]).write_text(json.dumps(q,indent=2)+"\n")
    print("OLD_PROBE_JSON="+json.dumps(q,separators=(",",":")))
    sys.exit(0 if q["reproduced"] else 1)
