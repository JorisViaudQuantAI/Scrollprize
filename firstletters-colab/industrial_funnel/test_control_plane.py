"""Synthetic fault-injection tests. No scans, network, model weights or private data."""
import concurrent.futures
import itertools
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from control_plane import Coordinator, Store, digest, zeros, vector
from survey import freeze_draw, summarize, zero_upper_count

def res(cpu=0,gpu=0,human=0,io=0):
    return {"cpu_ms":cpu,"gpu_ms":gpu,"human_ms":human,"io_bytes":io}

def caps(n=100):
    return {x:res(n,n,n,n) for x in ("exploit","repair","explore")}

FRAME=digest("synthetic-frame")
CAL=digest("synthetic-exact-recipe-and-stratum")
SCROLLS=["PHerc0125","PHerc0175A"]

def spec(stage="REGISTER",unit="synthetic-unit",scroll=SCROLLS[0],deps=(),inputs=None,calibration=None,recipe=None,frame=FRAME):
    return dict(scroll=scroll,unit=unit,frame=frame,stage=stage,recipe=recipe or digest(["recipe",stage]),
                inputs=inputs or {},deps=list(deps),calibration=calibration)

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/"private"
        self.c=Coordinator(self.root,"epoch-fixture",caps(),SCROLLS,lease_seconds=10,max_attempts=2)
    def create(self,s=None,lane="exploit",priority=1,grant=None):
        return self.c.enqueue(s or spec(),lane,priority,grant or res(cpu=10),now=100)
    def outputs(self,label,roles=("evidence",)):
        return {r:self.c.store.put((label+":"+r).encode()) for r in roles}
    def receipt(self,l,outputs=None,**kw):
        return {"job":l["job"],"outcome":"COMPLETE","worker_stopped":True,
                "actual":res(cpu=1),"outputs":outputs if outputs is not None else self.outputs(l["job"]),**kw}
    def complete(self,j,outputs=None,now=101,**kw):
        l=self.c.lease("exploit","synthetic-worker",now=now)
        self.assertIsNotNone(l);self.assertEqual(l["job"],j)
        r=self.receipt(l,outputs,**kw)
        self.assertEqual(self.c.finish(l["token"],r,now=now+1),"SUCCEEDED")
        return r["outputs"]
    def geometry(self):
        j=self.create();o=self.complete(j)
        g=self.create(spec("GEOMETRY",deps=[j],inputs=o))
        return g,self.complete(g,now=103)
    def control(self,until=200,passed=True,unit="synthetic-public-control",cal=CAL):
        j=self.create(spec("CONTROL",unit=unit,scroll="PHerc0211",calibration=cal))
        l=self.c.lease("exploit","synthetic-worker",now=105)
        self.assertEqual(l["job"],j)
        o=self.outputs(unit)
        status=self.c.finish(l["token"],self.receipt(l,o,control_pass=passed,calibration=cal,valid_until=until),now=106)
        self.assertEqual(status,"SUCCEEDED" if passed else "FAILED")
        return j,o
    def coarse(self,until=200):
        g,o=self.geometry();c,co=self.control(until=until)
        j=self.create(spec("COARSE",deps=[g,c],inputs={**o,"control":co["evidence"]},calibration=CAL))
        return j
    def pair(self,label):
        return self.outputs(label,("asc_map","desc_map","asc_support","desc_support","metadata"))
    def test_complete_dependency_pipeline(self):
        j=self.coarse()
        o=self.complete(j,self.pair("coarse"),now=107,orders=["asc","desc"])
        r=self.create(spec("RANK",deps=[j],inputs=o))
        ro=self.complete(r,now=109)
        cal2=digest("fine-calibration")
        c,co=self.control(unit="fine-public-control",cal=cal2)
        d=self.create(spec("DEEP",deps=[r,c],inputs={**ro,"control":co["evidence"]},calibration=cal2))
        out=self.complete(d,self.pair("deep"),now=111,orders=["asc","desc"])
        v=self.create(spec("REVIEW",deps=[d],inputs=out))
        self.complete(v,now=113,review_status="REVIEWED_NO_QUALIFYING_READING")
        self.assertEqual(sum(q["n"] for q in self.c.summary()["counts"] if q["state"]=="SUCCEEDED"),8)
    def test_same_job_idempotent_conflicting_policy(self):
        j=self.create();self.assertEqual(j,self.create())
        with self.assertRaises(ValueError):self.create(priority=2)
        self.assertEqual(len(self.c.summary()["counts"]),1)
    def test_epoch_policy_immutable_and_restart(self):
        j=self.create();o=self.complete(j)
        c=Coordinator(self.root,"epoch-fixture",caps(),SCROLLS,10,2)
        self.assertEqual(c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],1)
        self.assertIsNone(c.lease("exploit","restart-worker",103))
        with self.assertRaises(ValueError):Coordinator(self.root,"epoch-fixture",caps(200),SCROLLS,10,2)
    def test_concurrent_claim_one_winner(self):
        self.create()
        c2=Coordinator(self.root,"epoch-fixture",caps(),SCROLLS,10,2)
        barrier=threading.Barrier(2)
        def claim(co):
            barrier.wait(timeout=5)
            return co.lease("exploit","concurrent-fixture",101)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            result=list(ex.map(claim,[self.c,c2]))
        self.assertEqual(sum(x is not None for x in result),1)
        self.assertEqual(self.c.summary()["resources"]["reserved"]["exploit"]["cpu_ms"],10)
    def test_concurrent_vector_budget(self):
        self.create(spec(unit="a"),grant=res(gpu=60,io=50))
        self.create(spec(unit="b"),grant=res(gpu=60,io=50))
        c2=Coordinator(self.root,"epoch-fixture",caps(),SCROLLS,10,2)
        barrier=threading.Barrier(2)
        def claim(co):
            barrier.wait(timeout=5);return co.lease("exploit","synthetic",101)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:ans=list(ex.map(claim,[self.c,c2]))
        self.assertEqual(sum(x is not None for x in ans),1)
        self.assertEqual(self.c.summary()["operational_reasons"]["PENDING_RESOURCE_CAP"],1)
    def test_separate_lane_caps_and_infeasible_grant(self):
        self.create(spec(unit="impossible"),grant=res(gpu=101))
        self.create(spec(unit="uses-cap"),grant=res(gpu=100))
        self.create(spec(unit="waiting"),grant=res(gpu=1),priority=0)
        self.create(spec(unit="explore"),lane="explore",grant=res(gpu=100))
        self.assertIsNotNone(self.c.lease("exploit","w",101))
        self.assertIsNone(self.c.lease("exploit","w",101))
        self.assertIsNotNone(self.c.lease("explore","w",101))
        s=self.c.summary();self.assertEqual(s["operational_reasons"]["GRANT_EXCEEDS_LANE_CAP"],1)
        self.assertEqual(s["operational_reasons"]["PENDING_RESOURCE_CAP"],1)
    def test_scroll_fairness_not_raw_priority(self):
        self.create(spec(unit="a"),priority=99)
        self.create(spec(unit="b"),priority=98)
        self.create(spec(unit="c",scroll=SCROLLS[1]),priority=1)
        l=self.c.lease("exploit","w",101)
        self.assertEqual(l["spec"]["scroll"],SCROLLS[0])
        self.c.finish(l["token"],self.receipt(l),102)
        l=self.c.lease("exploit","w",103)
        self.assertEqual(l["spec"]["scroll"],SCROLLS[1])
    def test_expiry_fences_and_charges_grant(self):
        j=self.create();l=self.c.lease("exploit","w",101)
        self.assertEqual(self.c.expire(111),1)
        self.assertEqual(self.c.expire(112),0)
        with self.assertRaises(ValueError):self.c.finish(l["token"],self.receipt(l),112)
        self.assertEqual(self.c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],10)
        self.assertIsNone(self.c.lease("exploit","w2",112))
        with self.assertRaises(ValueError):self.c.reconcile_lost(j,l["token"],False,112)
        self.assertEqual(self.c.reconcile_lost(j,l["token"],True,112),"PENDING")
        l2=self.c.lease("exploit","w2",113)
        self.assertNotEqual(l["token"],l2["token"])
        self.assertEqual(self.c.finish(l2["token"],self.receipt(l2),114),"SUCCEEDED")
        self.assertEqual(self.c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],11)
    def test_stale_finish_itself_charges_expiry(self):
        self.create();l=self.c.lease("exploit","w",101)
        with self.assertRaises(ValueError):self.c.finish(l["token"],self.receipt(l),111)
        self.assertEqual(self.c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],10)
    def test_duplicate_receipt_exact_and_conflict(self):
        self.create();l=self.c.lease("exploit","w",101);r=self.receipt(l)
        self.assertEqual(self.c.finish(l["token"],r,102),"SUCCEEDED")
        self.assertEqual(self.c.finish(l["token"],r,120),"SUCCEEDED")
        with self.assertRaises(ValueError):self.c.finish(l["token"],{**r,"actual":res(cpu=2)},120)
        self.assertEqual(self.c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],1)
    def test_overrun_halts_all_lanes(self):
        self.create();self.create(spec(unit="second"),lane="repair")
        l=self.c.lease("exploit","w",101)
        self.assertEqual(self.c.finish(l["token"],self.receipt(l,actual=res(cpu=11)),102),"OVERRUN")
        self.assertTrue(self.c.summary()["halted"]);self.assertIsNone(self.c.lease("repair","w2",103))
        self.assertEqual(self.c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],11)
    def test_retry_limit_partial_receipt_never_unblocks(self):
        j=self.create();partial=self.outputs("partial")
        self.create(spec("GEOMETRY",deps=[j],inputs=partial))
        for at in (101,103):
            l=self.c.lease("exploit","w",at);self.assertEqual(l["job"],j)
            self.assertEqual(self.c.finish(l["token"],self.receipt(l,partial,outcome="FAILED",failure="TIMEOUT",actual=res(cpu=2)),at+1),"FAILED")
        self.assertIsNone(self.c.lease("exploit","w",105))
        self.assertEqual(self.c.summary()["operational_reasons"]["UPSTREAM_BLOCKED"],1)
        self.assertEqual(self.c.summary()["resources"]["spent"]["exploit"]["cpu_ms"],4)
    def test_missing_unknown_wrong_parent_and_cross_frame(self):
        with self.assertRaises(ValueError):self.create(spec("RANK",deps=[digest("unknown")]))
        j=self.create()
        with self.assertRaises(ValueError):self.create(spec("COARSE",deps=[j],calibration=CAL))
        with self.assertRaises(ValueError):self.create(spec("GEOMETRY",deps=[j],frame=digest("other-frame")))
        with self.assertRaises(ValueError):self.create(spec("GEOMETRY",deps=[j],unit="other-unit"))
    def test_changed_recipe_is_new_job_not_duplicate(self):
        a=self.create();b=self.create(spec(recipe=digest("new-recipe")))
        self.assertNotEqual(a,b)
    def test_unregistered_scroll_not_admitted(self):
        with self.assertRaises(ValueError):self.create(spec(scroll="PHerc9999"))
    def test_control_key_mismatch(self):
        g,o=self.geometry();c,co=self.control()
        with self.assertRaises(ValueError):self.create(spec("COARSE",deps=[g,c],inputs={**o,"control":co["evidence"]},calibration=digest("different")))
    def test_failed_control_blocks(self):
        g,o=self.geometry();c,co=self.control(passed=False)
        self.create(spec("COARSE",deps=[g,c],inputs={**o,"control":co["evidence"]},calibration=CAL))
        self.assertIsNone(self.c.lease("exploit","w",107))
        self.assertEqual(self.c.summary()["operational_reasons"]["UPSTREAM_BLOCKED"],1)
    def test_stale_control_before_dispatch(self):
        j=self.coarse(until=108)
        self.assertIsNone(self.c.lease("exploit","w",108))
        self.assertEqual(self.c.summary()["operational_reasons"]["CONTROL_STALE_OR_INVALID"],1)
    def test_control_expired_during_compute_quarantines(self):
        j=self.coarse(until=108);l=self.c.lease("exploit","w",107)
        self.assertEqual(self.c.finish(l["token"],self.receipt(l,self.pair("pair"),orders=["asc","desc"]),108),"FAILED")
        self.assertEqual(self.c.summary()["operational_reasons"]["CONTROL_STALE_OR_INVALID"],1)
    def test_missing_order_or_mask_rejected(self):
        j=self.coarse();l=self.c.lease("exploit","w",107);o=self.pair("coarse")
        bad={k:v for k,v in o.items() if k!="desc_support"}
        with self.assertRaises(ValueError):self.c.finish(l["token"],self.receipt(l,bad,orders=["asc","desc"]),108)
        with self.assertRaises(ValueError):self.c.finish(l["token"],self.receipt(l,o,orders=["asc"]),108)
        self.assertEqual(self.c.finish(l["token"],self.receipt(l,o,orders=["asc","desc"]),108),"SUCCEEDED")
    def test_corrupt_input_before_dispatch(self):
        h=self.c.store.put(b"synthetic-input")
        self.create(spec(inputs={"input":h}));(self.c.store.root/h).write_bytes(b"changed")
        self.assertIsNone(self.c.lease("exploit","w",101))
        self.assertEqual(self.c.summary()["operational_reasons"]["INPUT_MISSING_OR_CORRUPT"],1)
    def test_corrupt_input_during_compute_quarantines(self):
        h=self.c.store.put(b"synthetic-input");self.create(spec(inputs={"input":h}))
        l=self.c.lease("exploit","w",101);(self.c.store.root/h).write_bytes(b"changed")
        self.assertEqual(self.c.finish(l["token"],self.receipt(l),102),"FAILED")
    def test_corrupt_output_rejected(self):
        self.create();l=self.c.lease("exploit","w",101);o=self.outputs("bad")
        (self.c.store.root/o["evidence"]).write_bytes(b"corrupted")
        with self.assertRaises(ValueError):self.c.finish(l["token"],self.receipt(l,o),102)
        self.assertEqual(self.c.summary()["resources"]["reserved"]["exploit"]["cpu_ms"],10)
    def test_lineage_outputs_must_be_inputs(self):
        j=self.create();self.complete(j)
        other=self.outputs("other")
        self.create(spec("GEOMETRY",deps=[j],inputs=other))
        self.assertIsNone(self.c.lease("exploit","w",103))
        self.assertEqual(self.c.summary()["operational_reasons"]["LINEAGE_HASH_MISMATCH"],1)
    def test_stopped_worker_required(self):
        self.create();l=self.c.lease("exploit","w",101)
        with self.assertRaises(ValueError):self.c.finish(l["token"],self.receipt(l,worker_stopped=False),102)
    def test_private_summary_no_identifiers(self):
        j=self.create(spec(unit="SECRET_FIXTURE_ID"));l=self.c.lease("exploit","SECRET_WORKER",101)
        r=json.dumps(self.c.summary())
        for x in ("SECRET_FIXTURE_ID","SECRET_WORKER",j,l["token"],FRAME):self.assertNotIn(x,r)
        self.assertNotIn("blank",r.lower())
    def test_integer_resource_validation(self):
        for x in (res(cpu=-1),res(cpu=True),res(cpu=1.5),{"gpu_ms":1}):
            with self.assertRaises(ValueError):vector(x)
    def test_store_path_traversal_and_symlink(self):
        with self.assertRaises(ValueError):self.c.store.verify("../private")
        h=digest("fake");(self.c.store.root/h).symlink_to(self.root/"missing")
        with self.assertRaises(ValueError):self.c.store.verify(h)

class SurveyTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{"unit":f"synthetic-{s}-{i}","stratum":s,"grant":res(gpu=2)} for s in ("low","unknown") for i in range(10)]
        self.alloc={"low":3,"unknown":4}
    def draw(self,budget=None):
        return freeze_draw(self.rows,self.alloc,72,budget or res(gpu=20))
    def test_frozen_deterministic_order_independent(self):
        a=self.draw();b=freeze_draw(self.rows[::-1],self.alloc,72,res(gpu=20))
        self.assertEqual(a,b);self.assertEqual(a["strata"]["low"]["inclusion_probability"],.3)
        self.assertEqual(a["reserved"]["gpu_ms"],14)
        self.assertTrue(a["budget_feasible"])
    def test_budget_infeasible_no_substitution(self):
        a=self.draw();b=self.draw(res(gpu=12))
        self.assertEqual(a["strata"],b["strata"]);self.assertFalse(b["budget_feasible"])
        rr={u:"NO_QUALIFYING_READING" for s in b["strata"].values() for u in s["sample"]}
        self.assertFalse(summarize(b,rr)["complete"])
    def test_no_missing_strata_or_duplicate_units(self):
        with self.assertRaises(ValueError):freeze_draw(self.rows,{"low":3},72,res(gpu=20))
        with self.assertRaises(ValueError):freeze_draw(self.rows+[self.rows[0]],self.alloc,72,res(gpu=20))
    def test_incomplete_or_unknown_no_bound(self):
        a=self.draw();rr={u:"NO_QUALIFYING_READING" for s in a["strata"].values() for u in s["sample"]}
        rr.pop(next(iter(rr)));q=summarize(a,rr)
        self.assertFalse(q["complete"]);self.assertIsNone(q["simultaneous_zero_event_upper_count"])
        self.assertIsNone(q["HT_detectable_total"])
        with self.assertRaises(ValueError):summarize(a,{"not-sampled":"NO_QUALIFYING_READING"})
    def test_exact_zero_bound_vs_combinatorial_oracle(self):
        for N in range(1,26):
            for n in range(1,N+1):
                for alpha in (.05,.025,.1):
                    expected=max(M for M in range(N+1) if (math.comb(N-M,n)/math.comb(N,n) if N-M>=n else 0)>=alpha-1e-14)
                    self.assertEqual(zero_upper_count(N,n,alpha),expected)
    def test_complete_zero_simultaneous_bound(self):
        a=self.draw();rr={u:"NO_QUALIFYING_READING" for s in a["strata"].values() for u in s["sample"]}
        q=summarize(a,rr)
        self.assertTrue(q["complete"]);self.assertEqual(q["HT_detectable_total"],0)
        self.assertEqual(q["simultaneous_zero_event_upper_count"],zero_upper_count(10,3,.025)+zero_upper_count(10,4,.025))
        self.assertGreater(q["simultaneous_zero_event_upper_count"],0)
    def test_census_zero_bound_and_positive_HT(self):
        a=freeze_draw(self.rows,{"low":10,"unknown":10},72,res(gpu=40))
        rr={u:"NO_QUALIFYING_READING" for s in a["strata"].values() for u in s["sample"]}
        self.assertEqual(summarize(a,rr)["simultaneous_zero_event_upper_count"],0)
        rr[next(iter(rr))]="DETECTABLE_POSITIVE"
        q=summarize(a,rr);self.assertEqual(q["HT_detectable_total"],1)
        self.assertEqual(q["design_variance"],0);self.assertIsNone(q["simultaneous_zero_event_upper_count"])
    def test_sample_positive_HT(self):
        a=self.draw();rr={u:"NO_QUALIFYING_READING" for s in a["strata"].values() for u in s["sample"]}
        rr[a["strata"]["low"]["sample"][0]]="DETECTABLE_POSITIVE"
        q=summarize(a,rr);self.assertAlmostEqual(q["HT_detectable_total"],10/3)
        self.assertGreater(q["design_variance"],0);self.assertIsNone(q["simultaneous_zero_event_upper_count"])
    def test_tampered_draw_rejected(self):
        a=self.draw();a["strata"]["low"]["n"]=2
        with self.assertRaises(ValueError):summarize(a,{})
if __name__=="__main__":
    unittest.main()
