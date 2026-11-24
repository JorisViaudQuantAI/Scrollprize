"""Synthetic file receipt tests for amendment A01; no network or scan."""
import json
from pathlib import Path
import tempfile
import unittest
from control_plane import digest
from file_receipts import FileReceipts, job_key
from test_control_plane import spec, res, CAL

class FileTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name)/"private";self.a=FileReceipts(self.root)
        self.s=spec()
        self.out={"evidence":self.a.store.put(b"synthetic output")}
    def make(self,attempt=1,**kw):
        return {"job":job_key(self.s),"attempt":attempt,"outcome":"COMPLETE","worker_stopped":True,
                "grant":res(cpu=10),"actual":res(cpu=2),"outputs":self.out,"failure":None,**kw}
    def test_restart_and_replay(self):
        r=self.make();h=self.a.publish(self.s,1,r)
        b=FileReceipts(self.root)
        self.assertEqual(b.publish(self.s,1,r),h)
        q=b.resume(self.s);self.assertTrue(q["completed"]);self.assertEqual(q["next_attempt"],2)
        self.assertEqual(q["receipt"]["sha256"],h)
    def test_partial_orphan_not_complete(self):
        (self.a.receipts/".partial-fixture").write_text('{"broken":')
        self.assertFalse(self.a.resume(self.s)["completed"])
        r=self.make(outcome="PARTIAL",failure="SESSION_LOST")
        self.a.publish(self.s,1,r)
        q=self.a.resume(self.s);self.assertFalse(q["completed"]);self.assertEqual(q["next_attempt"],2)
        self.a.publish(self.s,2,self.make(2));self.assertTrue(self.a.resume(self.s)["completed"])
    def test_conflicting_attempt_immutable(self):
        self.a.publish(self.s,1,self.make())
        with self.assertRaises(ValueError):self.a.publish(self.s,1,self.make(actual=res(cpu=3)))
        self.assertEqual(self.a.read(self.s,1)["receipt"]["actual"]["cpu_ms"],2)
    def test_multiple_successes_reconcile_not_choose(self):
        self.a.publish(self.s,1,self.make());self.a.publish(self.s,2,self.make(2))
        with self.assertRaises(ValueError):self.a.resume(self.s)
    def test_wrong_identity_or_attempt(self):
        with self.assertRaises(ValueError):self.a.publish(self.s,1,self.make(job=digest("wrong")))
        with self.assertRaises(ValueError):self.a.publish(self.s,1,self.make(2))
    def test_corrupt_artifact_after_restart(self):
        self.a.publish(self.s,1,self.make())
        (self.a.store.root/self.out["evidence"]).write_bytes(b"corrupt")
        with self.assertRaises(ValueError):FileReceipts(self.root).resume(self.s)
    def test_corrupt_envelope(self):
        self.a.publish(self.s,1,self.make())
        p=self.a._path(self.s,1);obj=json.loads(p.read_text());obj["receipt"]["actual"]["cpu_ms"]=1;p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):self.a.resume(self.s)
    def test_overrun_not_complete_and_failed_charge_retained(self):
        with self.assertRaises(ValueError):self.a.publish(self.s,1,self.make(actual=res(cpu=11)))
        self.a.publish(self.s,1,self.make(actual=res(cpu=11),outcome="FAILED",failure="RESOURCE_OVERRUN"))
        self.assertFalse(self.a.resume(self.s)["completed"])
        self.assertEqual(self.a.read(self.s,1)["receipt"]["actual"]["cpu_ms"],11)
    def test_pair_roles_missing(self):
        self.s=spec("COARSE",calibration=CAL)
        with self.assertRaises(ValueError):self.a.publish(self.s,1,self.make(orders=["asc","desc"]))
    def test_control_expired_not_ready(self):
        self.s=spec("CONTROL",calibration=CAL)
        self.a.publish(self.s,1,self.make(calibration=CAL,control_pass=True,valid_until=100))
        self.assertTrue(self.a.resume(self.s,99)["completed"])
        self.assertFalse(self.a.resume(self.s,100)["completed"])
    def test_worker_stopped_required(self):
        with self.assertRaises(ValueError):self.a.publish(self.s,1,self.make(worker_stopped=False))
if __name__=="__main__":unittest.main()
