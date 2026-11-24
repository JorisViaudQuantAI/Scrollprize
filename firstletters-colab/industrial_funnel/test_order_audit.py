"""G2: regression tests for experimental order plans, both receipt backends."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from control_plane import Coordinator, digest, canonical, validate_spec
from file_receipts import FileReceipts,job_key
from order_audit import MODE, PAIRED, freeze_plan, validate_plan, assignment, validate_outputs, audit_summary
import test_control_plane as fixtures

res=fixtures.res;caps=fixtures.caps;spec=fixtures.spec;FRAME=fixtures.FRAME
def recipes():
    return {k:{"recipe":digest(["recipe074",k]),"calibration":digest(["cal074",k])} for k in ("paired","asc","desc")}
def population():
    rows=[]
    for order,n in (("asc",20),("desc",20),("unknown",2)):
        for i in range(n):
            rows.append({"scroll":"PHerc0125","unit":f"fixture-{order}-{i:02d}","frame":FRAME,
                         "geometry":digest([order,i]),"normal_order":order,"stratum":"synthetic"})
    return rows
def pair_indices(p):
    return {i for s in p["strata"].values() for i in s["paired_indices"]}
def plan_specs(store,p):
    h=store.put(canonical(p).encode())
    ids=pair_indices(p)
    out=[]
    for i,r in enumerate(p["population"]):
        arm="paired" if i in ids else r["normal_order"]
        out.append({**spec("COARSE",unit=r["unit"],scroll=r["scroll"],frame=r["frame"],
                         inputs={"geometry":r["geometry"],"order_audit_plan":h},
                         **p["recipes"][arm]),"order_policy":MODE})
    return out

class OrderTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name)/"private";self.f=FileReceipts(self.root)
        self.p=freeze_plan(population(),recipes(),7401)
        self.ss=plan_specs(self.f.store,self.p)
        self.known_single=next(i for i,r in enumerate(self.p["population"]) if r["normal_order"]=="asc" and i not in pair_indices(self.p))
        self.known_pair=next(i for i,r in enumerate(self.p["population"]) if r["normal_order"]=="desc" and i in pair_indices(self.p))
        self.unknown=next(i for i,r in enumerate(self.p["population"]) if r["normal_order"]=="unknown")
    def put_geometry(self,r):
        # Test rows use the canonical digest of this exact synthetic payload.
        order=r["normal_order"];i=int(r["unit"].rsplit("-",1)[1])
        h=self.f.store.put(canonical([order,i]).encode())
        self.assertEqual(h,r["geometry"])
    def outputs(self,orders):
        return {k:self.f.store.put(("fixture:"+k).encode()) for k in ["metadata"]+[o+"_"+s for o in orders for s in ("map","support")]}
    def receipt(self,s,orders,**kw):
        return dict(job=job_key(s),attempt=1,outcome="COMPLETE",worker_stopped=True,
            grant=res(cpu=10),actual=res(cpu=2),outputs=self.outputs(orders),failure=None,orders=orders,**kw)
    def test_legacy_fingerprint_preserved(self):
        s=spec("COARSE",calibration=fixtures.CAL)
        self.assertEqual(job_key(s),digest(s));self.assertEqual(validate_spec(s),s)
        self.assertEqual(assignment(s,self.f.store),PAIRED)
    def test_deterministic_stratified_and_unknown(self):
        self.assertEqual(self.p,freeze_plan(population()[::-1],recipes(),7401))
        self.assertEqual(len(pair_indices(self.p)),8)
        self.assertEqual(assignment(self.ss[self.unknown],self.f.store),PAIRED)
        self.assertEqual(assignment(self.ss[self.known_pair],self.f.store),PAIRED)
        self.assertEqual(assignment(self.ss[self.known_single],self.f.store),["asc"])
        for k,v in self.p["strata"].items():
            self.assertEqual(v["probability"],1. if json.loads(k)[2]=="unknown" else .15)
    def test_small_strata_round_up(self):
        r=population()[:1]
        p=freeze_plan(r,recipes(),7401)
        self.assertEqual(len(pair_indices(p)),1)
        self.assertEqual(next(iter(p["strata"].values()))["probability"],1.)
    def test_duplicate_and_invalid_population(self):
        r=population()
        for rr in (r+[r[0]], [{**r[0],"normal_order":"guessed"}]):
            with self.assertRaises(ValueError):freeze_plan(rr,recipes(),7401)
        bad=recipes();bad["asc"]=bad["paired"]
        with self.assertRaises(ValueError):freeze_plan(r,bad,7401)
    def test_tampered_draw_rejected_even_rehashed(self):
        p=copy.deepcopy(self.p);v=next(iter(p["strata"].values()));v["paired_indices"]=v["paired_indices"][1:]
        p["plan_hash"]=digest({k:v for k,v in p.items() if k!="plan_hash"})
        with self.assertRaises(ValueError):validate_plan(p)
        h=self.f.store.put(canonical(p).encode())
        s=copy.deepcopy(self.ss[0]);s["inputs"]["order_audit_plan"]=h
        with self.assertRaises(ValueError):assignment(s,self.f.store)
    def test_missing_corrupt_geometry_recipe_calibration(self):
        s=self.ss[self.known_single]
        for k,v in (("recipe",digest("wrong")),("calibration",digest("wrong")),("unit","unregistered")):
            with self.assertRaises(ValueError):assignment({**s,k:v},self.f.store)
        for k in ("geometry","order_audit_plan"):
            with self.assertRaises(ValueError):assignment({**s,"inputs":{**s["inputs"],k:digest("wrong")}},self.f.store)
        h=s["inputs"]["order_audit_plan"]
        (self.f.store.root/h).write_bytes(b"corrupt")
        with self.assertRaises(ValueError):assignment(s,self.f.store)
    def test_policy_cannot_extend_deep_or_hide(self):
        s=self.ss[self.known_single]
        for stage in ("DEEP","CONTROL","RANK"):
            with self.assertRaises(ValueError):validate_spec({**s,"stage":stage})
        with self.assertRaises(ValueError):validate_spec({k:v for k,v in s.items() if k!="order_policy"})
        with self.assertRaises(ValueError):validate_spec({**s,"order_policy":"free_single"})
    def test_file_single_cannot_masquerade_as_pair(self):
        i=self.known_single;s=self.ss[i];self.put_geometry(self.p["population"][i])
        with self.assertRaises(ValueError):self.f.publish(s,1,self.receipt(s,PAIRED))
        r=self.receipt(s,["asc"])
        with self.assertRaises(ValueError):self.f.publish(s,1,{**r,"outputs":{**r["outputs"],"desc_map":r["outputs"]["asc_map"]}})
        with self.assertRaises(ValueError):self.f.publish(s,1,{**r,"outputs":{k:v for k,v in r["outputs"].items() if k!="asc_support"}})
        self.f.publish(s,1,r)
        q=FileReceipts(self.root).resume(s)
        self.assertTrue(q["completed"]);self.assertTrue(q["experiment_only"]);self.assertFalse(q["eligible_for_standard_chain"])
    def test_file_pair_cannot_masquerade_as_single(self):
        for i in (self.known_pair,self.unknown):
            s=self.ss[i];self.put_geometry(self.p["population"][i])
            with self.assertRaises(ValueError):self.f.publish(s,1,self.receipt(s,["asc"]))
            self.f.publish(s,1,self.receipt(s,PAIRED))
            self.assertFalse(self.f.resume(s)["eligible_for_standard_chain"])
    def test_file_failed_experiment_no_completion(self):
        s=self.ss[self.known_single]
        r=self.receipt(s,["asc"])
        r.pop("orders");r.update(outcome="PARTIAL",failure="SESSION_LOST")
        self.f.publish(s,1,r)
        q=self.f.resume(s);self.assertFalse(q["completed"]);self.assertFalse(q["eligible_for_standard_chain"])
    def labels(self):
        return {str(i):("NO_SINGLE_ORDER_EXPOSURE" if self.p["population"][i]["normal_order"]=="unknown"
            else "NO_ADDITIONAL_DETECTABLE_TEXT") for i in pair_indices(self.p)}
    def test_audit_complete_zero_no_promotion(self):
        q=audit_summary(self.p,self.labels())
        self.assertTrue(q["complete"]);self.assertEqual(q["sample_units"],6)
        self.assertEqual(q["unknown_order_paired_units"],2);self.assertEqual(q["unknown_order_pair_adjudicated"],2)
        self.assertGreater(q["simultaneous_zero_event_upper_count"],0)
        self.assertFalse(q["promotion_allowed"])
    def test_audit_miss_requires_investigation(self):
        r=self.labels();r[str(self.known_pair)]="MISSED_DETECTABLE_TEXT"
        q=audit_summary(self.p,r)
        self.assertTrue(q["paired_investigation_required"]);self.assertIsNone(q["simultaneous_zero_event_upper_count"])
        self.assertAlmostEqual(q["HT_detectable_total"],20/3)
    def test_audit_missing_any_pair_suppresses_bound(self):
        for i in (self.known_pair,self.unknown):
            r=self.labels();r.pop(str(i))
            q=audit_summary(self.p,r);self.assertFalse(q["complete"]);self.assertIsNone(q["simultaneous_zero_event_upper_count"])
            self.assertIsNone(q["HT_detectable_total"])
    def test_unknown_not_counted_as_no_miss(self):
        r=self.labels();r[str(self.unknown)]="NO_ADDITIONAL_DETECTABLE_TEXT"
        with self.assertRaises(ValueError):audit_summary(self.p,r)
        with self.assertRaises(ValueError):audit_summary(self.p,{**self.labels(),str(self.known_single):"NO_ADDITIONAL_DETECTABLE_TEXT"})
    def test_all_unknown_no_counterfactual(self):
        p=freeze_plan([r for r in population() if r["normal_order"]=="unknown"],recipes(),7401)
        q=audit_summary(p,{str(i):"NO_SINGLE_ORDER_EXPOSURE" for i in pair_indices(p)})
        self.assertTrue(q["complete"]);self.assertEqual(q["population_units"],0)
        self.assertIsNone(q["simultaneous_zero_event_upper_count"]);self.assertFalse(q["promotion_allowed"])

class CoordinatorOrderTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.c=Coordinator(Path(t.name)/"private","g2",caps(),["PHerc0125"],lease_seconds=10,max_attempts=2)
    def finish(self,j,out,**extra):
        l=self.c.lease("exploit","worker",101);self.assertEqual(l["job"],j)
        r={"job":j,"outcome":"COMPLETE","worker_stopped":True,"actual":res(cpu=1),"outputs":out,**extra}
        self.assertEqual(self.c.finish(l["token"],r,102),"SUCCEEDED")
        return out
    def build(self,want_pair=False):
        p=freeze_plan(population(),recipes(),7401);indices=pair_indices(p)
        i=next(i for i,r in enumerate(p["population"]) if r["normal_order"]=="asc" and (i in indices)==want_pair)
        s=plan_specs(self.c.store,p)[i];unit=s["unit"]
        ghash=self.c.store.put(canonical(["asc",int(unit.rsplit("-",1)[1])]).encode())
        j=self.c.enqueue(spec(unit=unit),"exploit",1,res(cpu=10),100)
        o=self.finish(j,{"registered":self.c.store.put(b"registered")})
        g=self.c.enqueue(spec("GEOMETRY",unit=unit,deps=[j],inputs=o),"exploit",1,res(cpu=10),100)
        go=self.finish(g,{"geometry":ghash})
        ctrl=self.c.enqueue(spec("CONTROL",unit="public-fixture",scroll="PHerc0211",calibration=s["calibration"]),"exploit",1,res(cpu=10),100)
        co=self.finish(ctrl,{"control":self.c.store.put(b"control")},control_pass=True,calibration=s["calibration"],valid_until=200)
        s["deps"]=[g,ctrl];s["inputs"].update(go);s["inputs"].update(co)
        return s
    def output(self,orders):
        return {k:self.c.store.put(("fixture:"+k).encode()) for k in ["metadata"]+[o+"_"+s for o in orders for s in ("map","support")]}
    def test_actual_single_and_standard_rank_block(self):
        s=self.build();j=self.c.enqueue(s,"exploit",1,res(cpu=10),103)
        o=self.finish(j,self.output(["asc"]),orders=["asc"])
        with self.assertRaises(ValueError):self.c.enqueue(spec("RANK",unit=s["unit"],deps=[j],inputs=o),"exploit",1,res(cpu=10),104)
    def test_actual_pair_requires_both_and_standard_rank_block(self):
        s=self.build(True);j=self.c.enqueue(s,"exploit",1,res(cpu=10),103)
        l=self.c.lease("exploit","worker",104);self.assertEqual(j,l["job"])
        r={"job":j,"outcome":"COMPLETE","worker_stopped":True,"actual":res(cpu=1),"outputs":self.output(["asc"]),"orders":["asc"]}
        with self.assertRaises(ValueError):self.c.finish(l["token"],r,105)
        o=self.output(PAIRED);self.assertEqual(self.c.finish(l["token"],{**r,"outputs":o,"orders":PAIRED},105),"SUCCEEDED")
        with self.assertRaises(ValueError):self.c.enqueue(spec("RANK",unit=s["unit"],deps=[j],inputs=o),"exploit",1,res(cpu=10),106)
    def test_plan_changed_after_enqueue_blocks(self):
        s=self.build();j=self.c.enqueue(s,"exploit",1,res(cpu=10),103)
        (self.c.store.root/s["inputs"]["order_audit_plan"]).write_bytes(b"bad")
        self.assertIsNone(self.c.lease("exploit","worker",104))
        self.assertEqual(self.c.summary()["operational_reasons"]["INPUT_MISSING_OR_CORRUPT"],1)
if __name__=="__main__":unittest.main()
