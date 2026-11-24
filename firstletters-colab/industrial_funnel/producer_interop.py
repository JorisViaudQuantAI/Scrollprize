"""Read-only receipt compatibility inspection and opt-in synthetic producer replay.

No historical receipt is rewritten, no order plan is retroactively created, and
no run is dispatched. Receipt integrity, batch admission and native-array
semantics remain separate checks delegated to their existing shared validators.
The ``replay`` command writes synthetic fixtures only in a temporary directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from control_plane import canonical, digest, file_sha, require_sha, validate_spec
from file_admission import freeze_batch, inspect_batch, validate_file_envelope
from file_receipts import job_key
from order_audit import MODE, freeze_plan


PRODUCER_COMMIT = "da94312727c7e95f7f7fdc2d7481d502265434c5"
PRODUCER_SHA256 = "7a43443989eaaa08d0c9bf1f8f9b98fac95bf768fbaa48ccebd3772589c57bec"
LEAK_GUARD_SHA256 = "01b8142e63be80be1462913f9c397f08083685b0b63f03a41df3362dba48193c"


class ReadOnlyCAS:
    """Existing on-disk CAS reader; never creates directories or artifacts."""

    def __init__(self, root):
        path = Path(root)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("An existing nonsymlink CAS directory is required")
        self.root = path.resolve(strict=True)

    def verify(self, sha):
        path = self.root / require_sha(sha)
        if path.is_symlink() or not path.is_file() or file_sha(path) != sha:
            raise ValueError("Missing or corrupt CAS artifact")
        return path


def inspect_envelope(envelope, store, *, expected_spec=None, expected_attempt=None):
    """Diagnose compatibility without upgrading producer claims to admission.

    Omitting expected identity checks self-consistency only. A pinned batch must
    still bind the spec, dependencies, grants, controls and current inspection
    time. No inference is made from producer flags about scientific sufficiency.
    """
    result = {
        "schema": "funnel.producer_interop_inspection.v1",
        "compatible_with_shared_receipts": False,
        "external_identity_supplied": expected_spec is not None and expected_attempt is not None,
        "issues": [],
        "producer_declarations": {},
        "missing_evidence": [
            "PINNED_BATCH_ADMISSION_NOT_CHECKED",
            "NATIVE_ARRAY_SEMANTICS_NOT_CHECKED",
            "SCIENTIFIC_ADMISSION_NOT_ESTABLISHED",
        ],
        "scientific_admission": "NOT_ESTABLISHED",
        "production_complete": False,
        "dispatch_authorized": False,
        "scope": "Shared receipt compatibility only; producer metadata is declarative.",
    }
    try:
        if not isinstance(envelope, dict):
            raise ValueError("Expected an envelope object")
        spec = envelope["spec"] if expected_spec is None else expected_spec
        receipt = envelope["receipt"]
        attempt = receipt["attempt"] if expected_attempt is None else expected_attempt
        if isinstance(receipt, dict):
            declared = result["producer_declarations"]
            if "orders" in receipt:
                value = receipt["orders"]
                declared["orders"] = copy.deepcopy(value) if isinstance(value, list) and all(
                    isinstance(order, str) for order in value
                ) else {"invalid_type": type(value).__name__}
            if "order_mode" in receipt:
                value = receipt["order_mode"]
                declared["order_mode"] = value if isinstance(value, str) else {"invalid_type": type(value).__name__}
            if "audit_policy" in receipt:
                declared["receipt_audit_policy_present"] = True
                result["issues"].append({
                    "code": "RECEIPT_AUDIT_POLICY_IS_NOT_A_FROZEN_PLAN",
                    "detail": "A producer audit flag cannot replace the shared spec-bound plan.",
                })
            if receipt.get("order_mode") == "single_order_audit_15pct" and (
                not isinstance(spec, dict) or spec.get("order_policy") != MODE
                or "order_audit_plan" not in spec.get("inputs", {})
            ):
                result["missing_evidence"].append("SHARED_FROZEN_ORDER_PLAN_NOT_BOUND")
        validate_file_envelope(spec, attempt, envelope, store)
        result["compatible_with_shared_receipts"] = True
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
        result["issues"].append({"code": "SHARED_RECEIPT_REJECTED", "detail": str(exc)})
    result["status"] = (
        "SHARED_RECEIPT_COMPATIBLE" if result["compatible_with_shared_receipts"]
        else "INCOMPATIBLE_WITH_SHARED_RECEIPTS"
    )
    return result


def _res(cpu=0, gpu=0, io=0):
    return {"cpu_ms": cpu, "gpu_ms": gpu, "human_ms": 0, "io_bytes": io}


def _spec(stage="REGISTER", *, unit="synthetic-interop", deps=(), inputs=None, calibration=None):
    return {
        "scroll": "PHerc0125", "unit": unit, "frame": digest("synthetic-frame"),
        "stage": stage, "recipe": digest(["synthetic-recipe", stage]),
        "inputs": dict(inputs or {}), "deps": sorted(deps), "calibration": calibration,
    }


def _receipt(spec, outputs, *, attempt=1, **extra):
    return {
        "job": job_key(spec), "attempt": attempt, "outcome": "COMPLETE",
        "worker_stopped": True, "grant": _res(cpu=1000, gpu=1000, io=10000),
        "actual": _res(cpu=1), "outputs": outputs, "failure": None, **extra,
    }


def _accepts(call):
    try:
        call()
        return {"accepted": True}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return {"accepted": False, "error": type(exc).__name__, "detail": str(exc)}


def _load_pinned_producer(path, expected_sha):
    """Import only the explicitly supplied, hash-pinned local producer source."""
    path = Path(path).resolve(strict=True)
    require_sha(expected_sha)
    if file_sha(path) != expected_sha:
        raise ValueError("Producer source SHA256 differs from the supplied pin")
    guard = path.with_name("leak_guard.py")
    if file_sha(guard) != LEAK_GUARD_SHA256:
        raise ValueError("The producer leak_guard dependency differs from the pinned version")
    old_guard = sys.modules.get("leak_guard")
    try:
        guard_spec = importlib.util.spec_from_file_location("leak_guard", guard)
        guard_module = importlib.util.module_from_spec(guard_spec)
        guard_spec.loader.exec_module(guard_module)
        sys.modules["leak_guard"] = guard_module
        producer_spec = importlib.util.spec_from_file_location("pinned_interop_producer", path)
        producer = importlib.util.module_from_spec(producer_spec)
        producer_spec.loader.exec_module(producer)
    finally:
        if old_guard is None:
            sys.modules.pop("leak_guard", None)
        else:
            sys.modules["leak_guard"] = old_guard
    return producer


def _replay_dag(producer, root):
    """Existing producer reader must not widen A03's canonical admission policy."""
    files = producer.CrossPlatformFileReceipts(root)
    store = files.store
    cal = digest("synthetic-dag-calibration")
    register = _spec()
    reg_out = {"registration": store.put(b"synthetic-registration")}
    control = _spec("CONTROL", calibration=cal)
    ctl_out = {"control": store.put(b"synthetic-control")}
    geometry = _spec("GEOMETRY", deps=[job_key(register)], inputs=reg_out)
    geo_out = {"geometry": store.put(b"synthetic-geometry")}
    coarse = _spec("COARSE", deps=[job_key(geometry), job_key(control)],
                   inputs={**geo_out, **ctl_out}, calibration=cal)
    coarse_out = {k: store.put(k.encode()) for k in ("asc_map", "asc_support", "metadata")}
    rank = _spec("RANK", deps=[job_key(coarse)], inputs=coarse_out)
    rows = [
        (register, _receipt(register, reg_out)),
        (control, _receipt(control, ctl_out, calibration=cal, control_pass=True, valid_until=999)),
        (geometry, _receipt(geometry, geo_out)),
        (coarse, _receipt(coarse, coarse_out, orders=["asc"], order_mode="single_order_audit_15pct",
                         audit_policy={"audit_rate": .15, "normal_orientation_ambiguous": False,
                                       "audit_selected": False, "selected_order": "asc",
                                       "paired_calibration_key": cal})),
    ]
    for spec, receipt in rows:
        files.publish(spec, 1, receipt)
    specs = [register, control, geometry, coarse, rank]
    batch = freeze_batch("synthetic-interop", ["PHerc0125"], _res(cpu=5000, gpu=5000, io=50000),
                         [{"spec": s, "grants": [_res(cpu=1000, gpu=1000, io=10000)]} for s in specs])
    view = inspect_batch(files, batch, batch["manifest_hash"], 100)
    return {
        "producer_reader_accepts_single_order_as_legacy_spec": True,
        "shared_a03_halted": view["halted"], "costs_complete": view["costs_complete"],
        "dispatch_authorized": view["dispatch_authorized"],
        "coarse_state": view["states"][job_key(coarse)]["state"],
        "rank_state": view["states"][job_key(rank)]["state"],
    }


def replay_producer(producer_path, producer_sha256=PRODUCER_SHA256):
    """Reproduce contradictions on synthetic data using the real pinned code.

    The native fixture uses the existing analytic test fixture and validator;
    no detector is run and no real papyrus evidence is changed or diagnosed.
    """
    producer = _load_pinned_producer(producer_path, producer_sha256)
    from file_receipts import validate as validate_receipt
    from native_evidence import validate_native
    from test_native_evidence import NativeEvidenceTests
    import numpy as np

    cases = {}
    with tempfile.TemporaryDirectory(prefix="producer-interop-replay-") as tmp:
        root = Path(tmp)
        files = producer.CrossPlatformFileReceipts(root / "receipts-test")
        store = files.store
        cal = digest("synthetic-calibration")
        spec = _spec("COARSE", calibration=cal)
        outputs = {k: store.put(k.encode()) for k in
                   ("asc_map", "desc_map", "asc_support", "desc_support", "metadata")}
        legacy = _receipt(spec, outputs, orders=["asc", "desc"])
        extended = {**legacy, "order_mode": "paired_both_orders"}
        cases["paired_receipt_schema"] = {
            "legacy_shared": _accepts(lambda: validate_receipt(spec, 1, legacy, store)),
            "extended_producer": _accepts(lambda: producer.validate_r1_receipt(spec, 1, extended, store)),
            "extended_shared": _accepts(lambda: validate_receipt(spec, 1, extended, store)),
        }
        recipes = {arm: {"recipe": digest(["recipe", arm]), "calibration": digest(["cal", arm])}
                   for arm in ("paired", "asc", "desc")}
        geometry = store.put(b"synthetic-frozen-geometry")
        population = [{"scroll": "PHerc0125", "unit": "synthetic-planned", "frame": spec["frame"],
                       "geometry": geometry, "normal_order": "asc", "stratum": "synthetic"}]
        plan = freeze_plan(population, recipes, 7401)
        plan_sha = store.put(canonical(plan).encode())
        planned = {**_spec("COARSE", unit="synthetic-planned", calibration=recipes["paired"]["calibration"],
                          inputs={"geometry": geometry, "order_audit_plan": plan_sha}),
                   "recipe": recipes["paired"]["recipe"], "order_policy": MODE}
        cases["frozen_spec_schema"] = {
            "shared": _accepts(lambda: validate_spec(planned)),
            "producer": _accepts(lambda: producer.validate_job_spec(planned)),
        }
        audit = {"audit_rate": .15, "normal_orientation_ambiguous": False, "audit_selected": False,
                 "selected_order": "asc", "paired_calibration_key": digest("synthetic-paired-cal")}
        accepted = 0
        for i in range(20):
            single = _spec("COARSE", unit=f"synthetic-{i:02d}", calibration=cal)
            receipt = _receipt(single, {k: outputs[k] for k in ("asc_map", "asc_support", "metadata")},
                               orders=["asc"], order_mode="single_order_audit_15pct", audit_policy=audit)
            producer.validate_r1_receipt(single, 1, receipt, store)
            accepted += 1
        cases["unfrozen_audit_declaration"] = {
            "accepted_units": accepted, "paired_units": 0, "declared_rate": .15, "frozen_plan_present": False,
        }
        register = _spec()
        first = _receipt(register, {"evidence": store.put(b"synthetic-register-evidence")})
        files.publish(register, 1, first)
        files.publish(register, 2, {**first, "attempt": 2, "outcome": "FAILED", "failure": "TIMEOUT"})
        batch = freeze_batch("synthetic-resume", ["PHerc0125"], _res(cpu=2000, gpu=2000, io=20000),
                             [{"spec": register, "grants": [first["grant"], first["grant"]]}])
        inspected = inspect_batch(files, batch, batch["manifest_hash"], 100)
        cases["resume_is_not_admission"] = {
            "producer_resume_completed": files.resume(register, now=100)["completed"],
            "a03_state": inspected["states"][job_key(register)]["state"],
            "a03_halted": inspected["halted"],
        }
        fixture = NativeEvidenceTests()
        fixture.setUp()
        try:
            manifest = fixture.make()
            probability = fixture.read_role(manifest, "probability")
            probability[100, 100] = np.nan
            fixture.write_role(manifest, "probability", probability)
            native = validate_native(manifest, fixture.root)
            bridge_store = producer.CrossPlatformFileReceipts(root / "bridge")
            pixels = 288 * 288
            voxel = 9.362
            summary = {
                "scroll": "PHerc0125", "voxel_um": voxel, "num_slices": 31,
                "valid_ct_pixels": pixels, "valid_ct_cm2": pixels * (voxel / 10000) ** 2,
                "total_seconds": 1, "detectors": {"studio_asc": {"seconds": .1}, "studio_desc": {"seconds": .1}},
            }
            probability_bytes = (fixture.root / manifest["arrays"]["probability"]["path"]).read_bytes()
            support_bytes = (fixture.root / manifest["arrays"]["source_valid"]["path"]).read_bytes()
            kwargs = {
                "repo": bridge_store, "summary": summary, "stage": "COARSE",
                "unit_private_id": "synthetic-nonfinite-array", "frame_hash": digest("synthetic-frame"),
                "recipe_hash": digest("synthetic-recipe"), "calibration_key": cal, "deps": [], "inputs": {},
                "artifact_payloads": {"asc_map": probability_bytes, "desc_map": probability_bytes,
                                      "asc_support": support_bytes, "desc_support": support_bytes},
                "geometry_audit": {"parameterization_method": "flatboi", "shear_abs_p95_deg": 1.0,
                                   "pct_area_gt_20deg": 0.0, "area_inflation_true_over_nominal": 1.0},
                "telemetry": {"gpu_name": "synthetic-no-hardware", "peak_allocated_mib": 1,
                              "peak_reserved_mib": 1, "vram_cap_fraction": .5, "max_temp_c": 30,
                              "post_run_temp_c": 30, "worker_stopped": True},
                "grant": _res(cpu=2000, gpu=2000, io=10000000),
            }
            emitted = producer.emit_r1_receipt_from_rtx4060_summary(**kwargs)
            envelope = bridge_store.read(emitted["spec"], 1)
            metadata = bridge_store.store.read_json(emitted["metadata_sha256"])
            cases["native_arrays_vs_metadata"] = {
                "fixture": "Synthetic 288x288 float32 NPY with one supported NaN; correct SHA256",
                "native_valid": native["valid"], "native_errors": [e["code"] for e in native["errors"]],
                "producer_outcome": emitted["outcome"],
                "producer_negative_claim_flag": emitted["admissible_for_negative_claim"],
                "flag_is_scientific_authorization": False,
                "source_support_pixels_declared": metadata["sampling_contract"]["source_support_pixels"],
                "exclusion_flags_inserted_by_bridge": {
                    k: metadata["sampling_contract"][k]
                    for k in ("padded_pixels_excluded_from_area", "missing_chunk_fill_excluded")
                },
                "shared_inspection": inspect_envelope(envelope, bridge_store.store),
            }
            # Use finite arrays for the independent mixed-model-family case.
            probability[100, 100] = .25
            fixture.write_role(manifest, "probability", probability)
            finite_bytes = (fixture.root / manifest["arrays"]["probability"]["path"]).read_bytes()
            kwargs["artifact_payloads"] = {**kwargs["artifact_payloads"], "asc_map": finite_bytes, "desc_map": finite_bytes}
            kwargs["unit_private_id"] = "synthetic-mixed-model-family"
            kwargs["summary"] = {**summary, "detectors": {"studio_desc": {"seconds": .1}, "ahn_s0_asc": {"seconds": .1}}}
            mixed = producer.emit_r1_receipt_from_rtx4060_summary(**kwargs)
            cases["mixed_model_family_pair"] = {
                "producer_outcome": mixed["outcome"], "orders": mixed["receipt"]["orders"],
                "detectors": ["studio_desc", "ahn_s0_asc"], "same_model_family_pair": False,
            }
        finally:
            fixture.doCleanups()
        cases["shared_admission_backend_independence"] = _replay_dag(producer, root / "dag")
    return {
        "schema": "funnel.producer_interop_replay.v1", "synthetic_only": True,
        "producer_commit": PRODUCER_COMMIT if producer_sha256 == PRODUCER_SHA256 else None,
        "producer_source_sha256": producer_sha256, "producer_dependency_sha256": LEAK_GUARD_SHA256,
        "shared_admission_source_sha256": file_sha(Path(__file__).with_name("file_admission.py")),
        "cases": cases, "scientific_admission": "NOT_ESTABLISHED", "production_complete": False,
        "dispatch_authorized": False, "new_model_forwards": 0,
        "scope": "Executable interoperability contradictions on synthetic data; no finding about real papyrus quality or text.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Read one existing receipt and CAS")
    inspect_parser.add_argument("--receipt", required=True, type=Path)
    inspect_parser.add_argument("--cas", required=True, type=Path)
    replay_parser = commands.add_parser("replay", help="Run contradictory synthetic fixtures on pinned producer code")
    replay_parser.add_argument("--producer-path", required=True, type=Path)
    replay_parser.add_argument("--producer-sha256", default=PRODUCER_SHA256)
    args = parser.parse_args(argv)
    if args.command == "replay":
        result = replay_producer(args.producer_path, args.producer_sha256)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    try:
        envelope = json.loads(args.receipt.read_text())
        result = inspect_envelope(envelope, ReadOnlyCAS(args.cas))
    except (ValueError, OSError) as exc:
        result = {"status": "UNREADABLE_INPUT", "error": str(exc), "production_complete": False,
                  "scientific_admission": "NOT_ESTABLISHED", "dispatch_authorized": False}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("compatible_with_shared_receipts") else 2


if __name__ == "__main__":
    raise SystemExit(main())
