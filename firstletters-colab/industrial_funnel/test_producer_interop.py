"""Offline admission regressions for permissive producer readers.

Only synthetic bytes are used. The real pinned Gemini producer is exercised by
producer_interop.py replay, independently of these shared-policy regressions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from control_plane import canonical, digest
from file_admission import freeze_batch, inspect_batch
from file_receipts import FileReceipts, job_key
from order_audit import MODE, freeze_plan
from producer_interop import ReadOnlyCAS, inspect_envelope, _load_pinned_producer


def resources(n):
    return {"cpu_ms": n, "gpu_ms": 0, "human_ms": 0, "io_bytes": 0}


def spec(stage="REGISTER", *, deps=(), inputs=None, calibration=None):
    return {
        "scroll": "PHerc0125", "unit": "synthetic-interop", "frame": digest("synthetic-frame"),
        "stage": stage, "recipe": digest(["synthetic-recipe", stage]),
        "inputs": inputs or {}, "deps": sorted(deps), "calibration": calibration,
    }


def receipt(s, outputs, **extra):
    return {
        "job": job_key(s), "attempt": 1, "outcome": "COMPLETE", "worker_stopped": True,
        "grant": resources(10), "actual": resources(1), "outputs": outputs, "failure": None, **extra,
    }


def envelope(s, r):
    body = {"schema": "funnel.file_receipt.v1", "spec": s, "receipt": r}
    return {**body, "sha256": digest(body)}


class PermissiveProducerReader:
    """Storage fixture modelling a producer with different semantic validation.

    CAS remains the existing shared byte-verifying implementation. The fixture
    does not purport to defend against a malicious store or arbitrary code.
    """

    def __init__(self, files):
        self.receipts = files.receipts
        self.store = files.store

    def read(self, s, attempt):
        return json.loads((self.receipts / f"{job_key(s)}.{attempt}.json").read_text())


class ProducerInteropTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.files = FileReceipts(self.root / "private")
        self.backend = PermissiveProducerReader(self.files)

    def put(self, s, r):
        # Intentionally preserves the producer's exact envelope, even when its
        # schema differs; only a temporary synthetic fixture is written here.
        obj = envelope(s, r)
        path = self.files.receipts / f"{job_key(s)}.{r['attempt']}.json"
        path.write_text(canonical(obj) + "\n")
        return obj

    def one_register(self):
        s = spec()
        r = receipt(s, {"evidence": self.files.store.put(b"synthetic-evidence")})
        return s, r, self.put(s, r)

    def batch(self, specs, attempts=1):
        return freeze_batch("synthetic-interop", ["PHerc0125"], resources(10 * len(specs) * attempts),
                            [{"spec": s, "grants": [resources(10)] * attempts} for s in specs])

    def inspect_batch(self, batch):
        return inspect_batch(self.backend, batch, batch["manifest_hash"], 100)

    def dag(self, single):
        cal = digest("synthetic-calibration")
        register = spec()
        reg = {"registration": self.files.store.put(b"synthetic-registration")}
        self.put(register, receipt(register, reg))
        control = spec("CONTROL", calibration=cal)
        ctl = {"control": self.files.store.put(b"synthetic-control")}
        self.put(control, receipt(control, ctl, calibration=cal, control_pass=True, valid_until=999))
        geometry = spec("GEOMETRY", deps=[job_key(register)], inputs=reg)
        geo = {"geometry": self.files.store.put(b"synthetic-geometry")}
        self.put(geometry, receipt(geometry, geo))
        coarse = spec("COARSE", deps=[job_key(geometry), job_key(control)], inputs={**geo, **ctl}, calibration=cal)
        orders = ["asc"] if single else ["asc", "desc"]
        keys = ["metadata"] + [o + "_" + r for o in orders for r in ("map", "support")]
        maps = {k: self.files.store.put(k.encode()) for k in keys}
        r = receipt(coarse, maps, orders=orders)
        if single:
            r.update(order_mode="single_order_audit_15pct", audit_policy={
                "audit_rate": .15, "normal_orientation_ambiguous": False,
                "audit_selected": False, "selected_order": "asc", "paired_calibration_key": cal,
            })
        self.put(coarse, r)
        rank = spec("RANK", deps=[job_key(coarse)], inputs=maps)
        return self.batch([register, control, geometry, coarse, rank]), coarse, rank

    def test_inspection_is_read_only_and_never_admits_science(self):
        s, r, obj = self.one_register()
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        snapshot = copy.deepcopy(obj)
        result = inspect_envelope(obj, ReadOnlyCAS(self.files.store.root), expected_spec=s, expected_attempt=1)
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(snapshot, obj)
        self.assertTrue(result["compatible_with_shared_receipts"])
        self.assertTrue(result["external_identity_supplied"])
        self.assertFalse(result["production_complete"])
        self.assertFalse(result["dispatch_authorized"])
        self.assertEqual(result["scientific_admission"], "NOT_ESTABLISHED")
        self.assertIn("NATIVE_ARRAY_SEMANTICS_NOT_CHECKED", result["missing_evidence"])

    def test_self_consistency_is_not_external_identity(self):
        s, r, obj = self.one_register()
        self.assertFalse(inspect_envelope(obj, self.files.store)["external_identity_supplied"])
        other = {**s, "unit": "different-synthetic-unit"}
        self.assertFalse(inspect_envelope(obj, self.files.store, expected_spec=other, expected_attempt=1)["compatible_with_shared_receipts"])

    def test_paired_extension_is_rejected_without_rewriting(self):
        s = spec("COARSE", calibration=digest("synthetic-cal"))
        out = {k: self.files.store.put(k.encode()) for k in ("asc_map", "desc_map", "asc_support", "desc_support", "metadata")}
        r = receipt(s, out, orders=["asc", "desc"], order_mode="paired_both_orders")
        obj = self.put(s, r)
        before = copy.deepcopy(obj)
        self.assertFalse(inspect_envelope(obj, self.files.store)["compatible_with_shared_receipts"])
        self.assertEqual(before, obj)

    def test_single_order_audit_flags_are_not_retroactive_plan(self):
        batch, coarse, rank = self.dag(single=True)
        obj = self.backend.read(coarse, 1)
        result = inspect_envelope(obj, self.files.store)
        self.assertFalse(result["compatible_with_shared_receipts"])
        self.assertIn("SHARED_FROZEN_ORDER_PLAN_NOT_BOUND", result["missing_evidence"])
        self.assertNotIn("order_policy", obj["spec"])
        self.assertNotIn("order_audit_plan", obj["spec"]["inputs"])

    def test_a03_blocks_actual_producer_schema_through_permissive_reader(self):
        batch, coarse, rank = self.dag(single=True)
        view = self.inspect_batch(batch)
        self.assertEqual(view["states"][job_key(coarse)]["state"], "BLOCKED")
        self.assertEqual(view["states"][job_key(rank)]["state"], "BLOCKED")
        self.assertTrue(view["halted"])
        self.assertFalse(view["costs_complete"])
        self.assertFalse(view["dispatch_authorized"])

    def test_a03_preserves_canonical_paired_reader_compatibility(self):
        batch, coarse, rank = self.dag(single=False)
        view = self.inspect_batch(batch)
        self.assertEqual(view["states"][job_key(coarse)]["state"], "COMPLETE")
        self.assertEqual(view["states"][job_key(rank)]["state"], "READY_FOR_OPERATOR")
        self.assertFalse(view["halted"])
        self.assertTrue(view["costs_complete"])
        self.assertFalse(view["dispatch_authorized"])

    def test_a03_rechecks_envelope_identity_hash_and_shape(self):
        s, r, obj = self.one_register()
        batch = self.batch([s])
        path = self.files.receipts / f"{job_key(s)}.1.json"
        variants = [
            {**obj, "sha256": "0" * 64},
            {**obj, "schema": "other.schema"},
            {**obj, "extra": True},
            envelope({**s, "unit": "another-unit"}, r),
            envelope(s, []),
            envelope(s, {**r, "job": digest("another-job")}),
        ]
        for bad in variants:
            with self.subTest(bad=bad):
                path.write_text(canonical(bad))
                view = self.inspect_batch(batch)
                self.assertEqual(view["states"][job_key(s)]["state"], "BLOCKED")
                self.assertTrue(view["halted"])
                self.assertFalse(view["costs_complete"])

    def test_a03_rechecks_cas_bytes_after_permissive_read(self):
        s, r, obj = self.one_register()
        self.files.store.verify(r["outputs"]["evidence"]).write_bytes(b"synthetic-corruption")
        view = self.inspect_batch(self.batch([s]))
        self.assertEqual(view["states"][job_key(s)]["state"], "BLOCKED")
        self.assertTrue(view["halted"])
        self.assertFalse(view["costs_complete"])

    def test_completed_then_failed_is_not_reusable(self):
        s, r, obj = self.one_register()
        self.put(s, {**r, "attempt": 2, "outcome": "FAILED", "failure": "TIMEOUT"})
        self.assertTrue(self.files.resume(s)["completed"])
        view = self.inspect_batch(self.batch([s], attempts=2))
        self.assertEqual(view["states"][job_key(s)]["state"], "BLOCKED")
        self.assertTrue(view["halted"])

    def test_frozen_shared_plan_is_preserved_and_not_promoted(self):
        geometry = self.files.store.put(b"synthetic-plan-geometry")
        s = spec("COARSE", calibration=digest("temporary"))
        population = [{"scroll": s["scroll"], "unit": s["unit"], "frame": s["frame"],
                       "geometry": geometry, "normal_order": "asc", "stratum": "synthetic"}]
        recipes = {arm: {"recipe": digest(["recipe", arm]), "calibration": digest(["cal", arm])}
                   for arm in ("paired", "asc", "desc")}
        plan = freeze_plan(population, recipes, 7401)
        plan_sha = self.files.store.put(canonical(plan).encode())
        s.update(order_policy=MODE, inputs={"geometry": geometry, "order_audit_plan": plan_sha}, **recipes["paired"])
        out = {k: self.files.store.put(k.encode()) for k in ("asc_map", "desc_map", "asc_support", "desc_support", "metadata")}
        obj = self.put(s, receipt(s, out, orders=["asc", "desc"]))
        result = inspect_envelope(obj, self.files.store)
        self.assertTrue(result["compatible_with_shared_receipts"])
        self.assertFalse(result["production_complete"])
        self.assertEqual(obj["spec"]["order_policy"], MODE)
        self.assertEqual(obj["spec"]["inputs"]["order_audit_plan"], plan_sha)

    def test_source_pin_is_checked_before_import(self):
        path = self.root / "producer.py"
        path.write_text('raise AssertionError("must not execute unpinned source")\n')
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _load_pinned_producer(path, "0" * 64)

    def test_cli_reads_existing_cas_and_returns_status(self):
        s, r, obj = self.one_register()
        path = self.files.receipts / f"{job_key(s)}.1.json"
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name("producer_interop.py")),
                                    "inspect", "--receipt", str(path), "--cas", str(self.files.store.root)],
                                   capture_output=True, text=True, check=True)
        result = json.loads(completed.stdout)
        self.assertTrue(result["compatible_with_shared_receipts"])
        self.assertFalse(result["production_complete"])
        self.assertFalse(result["dispatch_authorized"])


if __name__ == "__main__":
    unittest.main()
