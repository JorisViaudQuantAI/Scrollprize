"""A03 regression review: both order policies and prior-attempt cost accounting.

Complements the producer interop rejection cases with valid full DAGs. Every
payload is synthetic; this checks admission contracts, not scientific evidence.
"""
import json
from pathlib import Path
import tempfile
import unittest

from control_plane import canonical, digest
from file_admission import freeze_batch, inspect_batch
from file_receipts import FileReceipts, job_key
from order_audit import MODE, freeze_plan


def resources(cpu):
    return {"cpu_ms": cpu, "gpu_ms": 0, "human_ms": 0, "io_bytes": 0}


class UncheckedReader:
    def __init__(self, shared):
        self.receipts, self.store = shared.receipts, shared.store

    def read(self, spec, attempt):
        return json.loads((self.receipts / f"{job_key(spec)}.{attempt}.json").read_text())


class AdmissionPolicyCompatibilityTests(unittest.TestCase):
    def test_valid_order_arms_after_partial_attempt_keep_complete_costs(self):
        for arm in ("legacy_paired", "known_single", "known_paired", "unknown_paired"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                shared = FileReceipts(Path(tmp) / "private")
                frame = digest("synthetic-frame-a03-review")
                recipes = {a: {"recipe": digest(["synthetic-recipe", a]),
                               "calibration": digest(["synthetic-calibration", a])}
                           for a in ("paired", "asc", "desc")}
                population = [{"scroll": "PHerc0125", "unit": f"synthetic-{order}-{i}",
                               "frame": frame, "normal_order": order, "stratum": "synthetic",
                               "geometry": shared.store.put(canonical([order, i]).encode())}
                              for order, count in (("asc", 10), ("unknown", 1)) for i in range(count)]
                plan = freeze_plan(population, recipes, 7401)
                paired = {i for s in plan["strata"].values() for i in s["paired_indices"]}
                selected = next(i for i, row in enumerate(plan["population"])
                                if (row["normal_order"] == "unknown" if arm == "unknown_paired" else
                                    row["normal_order"] == "asc" and (i not in paired if arm == "known_single" else i in paired)))
                row = plan["population"][selected]
                orders = ["asc"] if arm == "known_single" else ["asc", "desc"]
                recipe_arm = "asc" if arm == "known_single" else "paired"
                calibration = recipes[recipe_arm]["calibration"]

                def spec(stage, *, deps=(), inputs=None, cal=None):
                    return {"scroll": row["scroll"], "unit": row["unit"], "frame": frame,
                            "stage": stage, "recipe": digest(["synthetic-stage", stage]),
                            "inputs": inputs or {}, "deps": sorted(deps), "calibration": cal}

                def publish(s, out, *, attempt=1, actual=1, **extra):
                    receipt = {"job": job_key(s), "attempt": attempt, "outcome": "COMPLETE",
                               "worker_stopped": True, "grant": resources(10), "actual": resources(actual),
                               "outputs": out, "failure": None, **extra}
                    shared.publish(s, attempt, receipt)

                register = spec("REGISTER")
                reg = {"registration": shared.store.put(b"synthetic-registration")}
                publish(register, reg)
                control = spec("CONTROL", cal=calibration)
                ctl = {"control": shared.store.put(b"synthetic-control")}
                publish(control, ctl, calibration=calibration, control_pass=True, valid_until=999)
                geometry = spec("GEOMETRY", deps=[job_key(register)], inputs=reg)
                geo = {"geometry": row["geometry"]}
                publish(geometry, geo)
                coarse = spec("COARSE", deps=[job_key(geometry), job_key(control)], inputs={**geo, **ctl}, cal=calibration)
                if arm != "legacy_paired":
                    coarse.update(order_policy=MODE, **recipes[recipe_arm])
                    coarse["inputs"]["order_audit_plan"] = shared.store.put(canonical(plan).encode())
                keys = ["metadata"] + [order + "_" + role for order in orders for role in ("map", "support")]
                maps = {key: shared.store.put(("synthetic-" + key).encode()) for key in keys}
                publish(coarse, {}, actual=3, outcome="PARTIAL", failure="SESSION_LOST")
                publish(coarse, maps, attempt=2, actual=4, orders=orders)
                entries = [{"spec": s, "grants": [resources(10)]} for s in (register, control, geometry)]
                entries.append({"spec": coarse, "grants": [resources(10), resources(10)]})
                rank = spec("RANK", deps=[job_key(coarse)], inputs=maps)
                with_rank = entries + [{"spec": rank, "grants": [resources(10)]}]
                if arm == "legacy_paired":
                    entries = with_rank
                else:
                    with self.assertRaises(ValueError):
                        freeze_batch("synthetic", ["PHerc0125"], resources(60), with_rank)
                batch = freeze_batch("synthetic", ["PHerc0125"], resources(60), entries)
                shared_result = inspect_batch(shared, batch, batch["manifest_hash"], 100)
                permissive_result = inspect_batch(UncheckedReader(shared), batch, batch["manifest_hash"], 100)
                self.assertEqual(shared_result, permissive_result)
                self.assertFalse(shared_result["halted"])
                self.assertTrue(shared_result["costs_complete"])
                self.assertFalse(shared_result["dispatch_authorized"])
                self.assertEqual(shared_result["spent"], resources(10))
                state = shared_result["states"][job_key(coarse)]
                self.assertEqual(state["attempts"], 2)
                self.assertEqual(state["state"], "COMPLETE" if arm == "legacy_paired" else "EXPERIMENT_COMPLETE")
                if arm == "legacy_paired":
                    self.assertEqual(shared_result["states"][job_key(rank)]["state"], "READY_FOR_OPERATOR")


if __name__ == "__main__":
    unittest.main()
