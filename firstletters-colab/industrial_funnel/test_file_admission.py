"""A03 synthetic fault-injection; no scans, model weights, or private data."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from control_plane import digest,canonical
from file_receipts import FileReceipts,job_key
from file_admission import freeze_batch,inspect_batch,evidence_bound_audit
from order_audit import freeze_plan,assignment,MODE
import test_control_plane as fixtures

res=fixtures.res
spec=fixtures.spec
CAL=fixtures.CAL
FRAME=fixtures.FRAME

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f=FileReceipts(Path(self.tmp.name)/"private")
        self.entries=[]
    def put(self,label):
        return self.f.store.put(canonical(label).encode())
    def add(self,s,attempts=1,grant=None):
        self.entries.append({"spec":s,"grants":[grant or res(cpu=10) for _ in range(attempts)]})
        return s
    def publish(self,s,outputs=None,n=1,**kw):
        r=dict(job=job_key(s),attempt=n,outcome="COMPLETE",worker_stopped=True,
               grant=res(cpu=10),actual=res(cpu=1),outputs=outputs or {"evidence":self.put(["out",job_key(s),n])},failure=None)
        if s["stage"]=="CONTROL":
            r.update(calibration=s["calibration"],control_pass=True,valid_until=200)
        if s["stage"] in ("COARSE","DEEP"):
            orders=assignment(s,self.f.store)
            r["orders"]=orders
            r["outputs"]={k:self.put([job_key(s),k]) for k in ["metadata"]+[o+"_"+v for o in orders for v in ("map","support")]}
        r.update(kw)
        if r["outcome"]!="COMPLETE":
            for k in ("orders","calibration","control_pass","valid_until","review_status"):r.pop(k,None)
        self.f.publish(s,n,r)
        return self.f.read(s,n)
    def manifest(self,cap=None):
        return freeze_batch("fixture-20261124",["PHerc0125"],cap or res(cpu=10000),self.entries)
    def inspect(self,m=None,now=100,revoked=()):
        m=m or self.manifest()
        return inspect_batch(self.f,m,m["manifest_hash"],now,revoked)
    def state(self,s,m=None,now=100,revoked=()):
        return self.inspect(m,now,revoked)["states"][job_key(s)]
    def chain(self,geometry=True,control=True,passed=True,coarse=False,coarse_attempts=1,**kw):
        r=self.add(spec(unit="fixture"))
        rr=self.publish(r)
        gh=self.put("geometry")
        g=self.add(spec("GEOMETRY",unit=r["unit"],deps=[job_key(r)],inputs=rr["receipt"]["outputs"]))
        if geometry:self.publish(g,{"geometry":gh})
        c=self.add(spec("CONTROL",unit="fixture-control",scroll="PHerc0211",calibration=CAL))
        ch=self.put("control")
        if control:self.publish(c,{"control":ch},control_pass=passed)
        s=self.add(spec("COARSE",unit=r["unit"],deps=[job_key(g),job_key(c)],calibration=CAL,inputs={"geometry":gh,"control":ch}),coarse_attempts)
        if coarse:self.publish(s,**kw)
        return r,g,c,s
    def test_ready_and_complete(self):
        _,_,_,s=self.chain()
        q=self.inspect()
        self.assertEqual(q["states"][job_key(s)]["state"],"READY_FOR_OPERATOR")
        self.assertFalse(q["dispatch_authorized"])
        self.publish(s)
        self.assertEqual(self.state(s)["state"],"COMPLETE")
        self.assertEqual(self.inspect()["spent"]["cpu_ms"],4)
    def test_resume_is_only_local(self):
        _,_,_,s=self.chain(control=False,coarse=True)
        local=self.f.resume(s,now=100)
        self.assertTrue(local["completed"])
        self.assertFalse(local["eligible_for_standard_chain"])
        self.assertTrue(local["admission_check_required"])
        self.assertEqual(self.state(s)["state"],"WAITING_PARENT")
    def test_missing_geometry(self):
        _,_,_,s=self.chain(geometry=False,coarse=True)
        self.assertEqual(self.state(s)["state"],"WAITING_PARENT")
    def test_failed_and_stale_control(self):
        _,_,_,s=self.chain(passed=False,coarse=True)
        self.assertEqual(self.state(s)["state"],"BLOCKED")
    def test_expired_control(self):
        _,_,_,s=self.chain(coarse=True)
        self.assertEqual(self.state(s,now=200)["state"],"BLOCKED")
    def test_revoked_control(self):
        _,_,_,s=self.chain(coarse=True)
        self.assertEqual(self.state(s,revoked=[CAL])["state"],"BLOCKED")
    def test_unregistered_parent(self):
        self.add(spec("COARSE",deps=[digest("missing")],calibration=CAL))
        with self.assertRaises(ValueError):self.manifest()
    def test_wrong_parent_stage(self):
        r=self.add(spec())
        self.add(spec("COARSE",deps=[job_key(r)],calibration=CAL))
        with self.assertRaises(ValueError):self.manifest()
    def test_wrong_frame(self):
        r=self.add(spec())
        self.add(spec("GEOMETRY",deps=[job_key(r)],frame=digest("other")))
        with self.assertRaises(ValueError):self.manifest()
    def test_wrong_calibration(self):
        _,_,_,s=self.chain()
        self.entries[-1]["spec"]={**s,"calibration":digest("other")}
        with self.assertRaises(ValueError):self.manifest()
    def test_missing_parent_output_binding(self):
        _,_,_,s=self.chain()
        s={**s,"inputs":{"geometry":self.put("unrelated geometry"),"control":s["inputs"]["control"]}}
        self.entries[-1]["spec"]=s
        self.publish(s)
        self.assertEqual(self.state(s)["state"],"BLOCKED")
    def test_cap_reserves_all_retries(self):
        self.chain(coarse_attempts=2)
        with self.assertRaises(ValueError):self.manifest(res(cpu=49))
        self.assertEqual(self.manifest(res(cpu=50))["authorized_worst_case"]["cpu_ms"],50)
    def test_manifest_pin(self):
        self.chain()
        m=self.manifest()
        tampered=deepcopy(m)
        tampered["cap"]["cpu_ms"]+=1
        with self.assertRaises(ValueError):inspect_batch(self.f,tampered,m["manifest_hash"],100)
        with self.assertRaises(ValueError):inspect_batch(self.f,m,digest("other"),100)
    def test_grant_escalation(self):
        _,_,_,s=self.chain(coarse=True,grant=res(cpu=1000),actual=res(cpu=100))
        q=self.inspect()
        self.assertTrue(q["halted"])
        self.assertEqual(q["states"][job_key(s)]["state"],"BLOCKED")
        self.assertFalse(q["costs_complete"])
    def test_unauthorized_attempt(self):
        _,_,_,s=self.chain()
        self.publish(s,n=2)
        self.assertTrue(self.inspect()["halted"])
    def test_partial_then_complete(self):
        _,_,_,s=self.chain(coarse_attempts=2)
        self.publish(s,outcome="PARTIAL",failure="SESSION_LOST")
        self.assertEqual(self.state(s)["state"],"READY_FOR_OPERATOR")
        self.publish(s,n=2)
        self.assertEqual(self.state(s)["state"],"COMPLETE")
        self.assertEqual(self.inspect()["spent"]["cpu_ms"],5)
    def test_failed_exhausted(self):
        _,_,_,s=self.chain()
        self.publish(s,outcome="FAILED",failure="GPU_UNAVAILABLE")
        self.assertEqual(self.state(s)["reason"],"ATTEMPTS_EXHAUSTED")
    def test_conflicting_completed_attempts(self):
        _,_,_,s=self.chain(coarse_attempts=2,coarse=True)
        self.publish(s,n=2)
        self.assertTrue(self.inspect()["halted"])
    def test_overrun_halts_batch(self):
        _,_,_,s=self.chain(coarse_attempts=2)
        self.publish(s,outcome="FAILED",failure="RESOURCE_OVERRUN",actual=res(cpu=11))
        self.assertTrue(self.inspect()["halted"])
        self.assertEqual(self.state(s)["reason"],"BATCH_INTEGRITY_OR_RESOURCE_STOP")
    def test_corrupt_parent_bytes(self):
        _,g,_,s=self.chain(coarse=True)
        h=self.f.read(g,1)["receipt"]["outputs"]["geometry"]
        (self.f.store.root/h).write_bytes(b"corrupted")
        q=self.inspect()
        self.assertTrue(q["halted"])
        self.assertEqual(q["states"][job_key(s)]["state"],"BLOCKED")
    def cohort(self,missing=None):
        pop=[]
        for i in range(4):
            pop.append(dict(scroll="PHerc0125",unit=f"fixture-{i}",frame=FRAME,
                geometry=self.put(["geometry",i]),normal_order="asc" if i<3 else "unknown",stratum="fixture"))
        recipes={k:{"recipe":digest(["recipe",k]),"calibration":digest(["cal",k])} for k in ("paired","asc","desc")}
        p=freeze_plan(pop,recipes,7401)
        ph=self.f.store.put(canonical(p).encode())
        pair={i for v in p["strata"].values() for i in v["paired_indices"]}
        cs={};jobs={};labels={}
        for i,row in enumerate(p["population"]):
            arm="paired" if i in pair else row["normal_order"]
            if arm not in cs:
                c=self.add(spec("CONTROL",unit="fixture-control-"+arm,scroll="PHerc0211",calibration=recipes[arm]["calibration"]))
                cc=self.publish(c);cs[arm]=(c,cc["receipt"]["outputs"])
            c,cout=cs[arm]
            r=self.add(spec(unit=row["unit"]));rr=self.publish(r)
            g=self.add(spec("GEOMETRY",unit=row["unit"],deps=[job_key(r)],inputs=rr["receipt"]["outputs"]))
            self.publish(g,{"geometry":row["geometry"]})
            s=self.add({**spec("COARSE",unit=row["unit"],deps=[job_key(g),job_key(c)],inputs={"geometry":row["geometry"],"order_audit_plan":ph,**cout},**recipes[arm]),"order_policy":MODE})
            jobs[i]=s
            if i==missing:continue
            q=self.publish(s)
            if i in pair:
                labels[str(i)]=self.put(dict(schema="funnel.order_adjudication.v1",plan_hash=p["plan_hash"],index=i,job=job_key(s),receipt_hash=q["sha256"],normal_order=row["normal_order"],producer="fixture-producer",reviewer="fixture-reviewer",outcome="NO_SINGLE_ORDER_EXPOSURE" if row["normal_order"]=="unknown" else "NO_ADDITIONAL_DETECTABLE_TEXT"))
        return p,ph,jobs,labels
    def bound(self,ph,labels,revoked=()):
        m=self.manifest()
        return evidence_bound_audit(self.f,m,m["manifest_hash"],ph,labels,100,revoked)
    def test_full_audit_bound(self):
        _,ph,_,labels=self.cohort()
        q=self.bound(ph,labels)
        self.assertTrue(q["evidence_complete"])
        self.assertTrue(q["evidence_verified"])
        self.assertIsNotNone(q["simultaneous_zero_event_upper_count"])
        self.assertFalse(q["promotion_allowed"])
        self.assertEqual(q["completed_cohort_units"],4)
    def test_missing_pair_review(self):
        _,ph,_,labels=self.cohort()
        labels.pop(next(iter(labels)))
        q=self.bound(ph,labels)
        self.assertFalse(q["evidence_complete"])
        self.assertIsNone(q["simultaneous_zero_event_upper_count"])
    def test_missing_pair_receipt(self):
        _,ph,jobs,labels=self.cohort()
        i=int(next(iter(labels)))
        self.f._path(jobs[i],1).unlink()
        q=self.bound(ph,labels)
        self.assertFalse(q["evidence_complete"])
        self.assertEqual(q["invalid_adjudications"],1)
    def test_missing_single_receipt(self):
        p,ph,jobs,labels=self.cohort()
        i=next(i for i in jobs if str(i) not in labels)
        self.f._path(jobs[i],1).unlink()
        q=self.bound(ph,labels)
        self.assertFalse(q["cohort_complete"])
        self.assertIsNone(q["HT_detectable_total"])
    def test_bad_review_binding(self):
        _,ph,_,labels=self.cohort()
        i=next(iter(labels));r=__import__("json").loads(self.f.store.verify(labels[i]).read_text())
        r["receipt_hash"]=digest("other");labels[i]=self.put(r)
        q=self.bound(ph,labels)
        self.assertEqual(q["invalid_adjudications"],1)
        self.assertIsNone(q["simultaneous_zero_event_upper_count"])
    def test_same_reviewer(self):
        _,ph,_,labels=self.cohort()
        i=next(iter(labels));r=__import__("json").loads(self.f.store.verify(labels[i]).read_text())
        r["reviewer"]=r["producer"];labels[i]=self.put(r)
        self.assertFalse(self.bound(ph,labels)["evidence_complete"])
    def test_extra_review(self):
        _,ph,_,labels=self.cohort()
        labels["999"]=next(iter(labels.values()))
        with self.assertRaises(ValueError):self.bound(ph,labels)
    def test_audited_miss(self):
        p,ph,_,labels=self.cohort()
        i=next(i for i in labels if p["population"][int(i)]["normal_order"]!="unknown")
        r=__import__("json").loads(self.f.store.verify(labels[i]).read_text())
        r["outcome"]="MISSED_DETECTABLE_TEXT";labels[i]=self.put(r)
        q=self.bound(ph,labels)
        self.assertTrue(q["evidence_complete"])
        self.assertEqual(q["HT_detectable_total"],3)
        self.assertTrue(q["paired_investigation_required"])
        self.assertIsNone(q["simultaneous_zero_event_upper_count"])
    def test_experimental_parent_forbidden(self):
        _,_,jobs,_=self.cohort()
        s=jobs[0]
        self.add(spec("RANK",unit=s["unit"],deps=[job_key(s)],inputs=self.f.read(s,1)["receipt"]["outputs"]))
        with self.assertRaises(ValueError):self.manifest()
    def test_revoked_pair_controls_no_bound(self):
        p,ph,_,labels=self.cohort()
        q=self.bound(ph,labels,[p["recipes"]["paired"]["calibration"]])
        self.assertFalse(q["evidence_complete"])
        self.assertIsNone(q["simultaneous_zero_event_upper_count"])

if __name__=="__main__":unittest.main()
