# Copyright (c) 2026 Scrollprize Contributors (MIT License)
"""CPU-only unit tests for ``gemini_lane/industrial_funnel_adapter.py``.

Verifies cross-platform Windows/POSIX R1 FileReceipts, dual-order & A01 B8/S2
single-order-plus-15%-audit modes, Exp T15/T16/T17/Q33 semantic contracts,
S0 19-scroll inventory joining, 64-vx vs <=3-vx normal-correspondence dedup,
and leak_guard privacy hygiene. Zero torch/CUDA imports, zero network calls.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

GEMINI_DIR = Path(__file__).resolve().parent
if str(GEMINI_DIR) not in sys.path:
    sys.path.insert(0, str(GEMINI_DIR))

import industrial_funnel_adapter as IFA  # noqa: E402
import leak_guard as LG  # noqa: E402


def _res(cpu: int = 0, gpu: int = 0, human: int = 0, io: int = 0) -> dict[str, int]:
    return {"cpu_ms": cpu, "gpu_ms": gpu, "human_ms": human, "io_bytes": io}


def _spec(
    stage: str = "REGISTER",
    scroll: str = "PHerc0125",
    unit: str = "synthetic-unit-01",
    deps: list[str] | None = None,
    inputs: dict[str, str] | None = None,
    calibration: str | None = None,
) -> dict:
    return {
        "scroll": scroll,
        "unit": unit,
        "frame": IFA.digest_obj("synthetic-frame-v1"),
        "stage": stage,
        "recipe": IFA.digest_obj(["synthetic-recipe", stage]),
        "inputs": inputs or {},
        "deps": list(deps or []),
        "calibration": calibration,
    }


class CrossPlatformFileReceiptsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ifunnel-adapter-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "private"
        self.repo = IFA.CrossPlatformFileReceipts(self.root)
        self.cal = IFA.digest_obj("synthetic-calibration-stratum-v1")
        self.spec_reg = _spec("REGISTER")
        self.out_reg = {"evidence": self.repo.store.put(b"synthetic-register-payload")}

    def _make_receipt(self, spec: dict, attempt: int = 1, outputs: dict | None = None, **kw) -> dict:
        return {
            "job": IFA.job_key(spec),
            "attempt": attempt,
            "outcome": "COMPLETE",
            "worker_stopped": True,
            "grant": _res(cpu=1000, gpu=1000, io=10_000),
            "actual": _res(cpu=200, gpu=100, io=1024),
            "outputs": outputs if outputs is not None else self.out_reg,
            "failure": None,
            **kw,
        }

    def test_private_root_rejects_git_worktree(self) -> None:
        inside_repo = GEMINI_DIR / "_forbidden_private_dir"
        with self.assertRaises(ValueError):
            IFA.private_root_cross_platform(inside_repo)

    def test_restart_idempotent_replay_and_conflict_rejection(self) -> None:
        r1 = self._make_receipt(self.spec_reg)
        sha1 = self.repo.publish(self.spec_reg, 1, r1)
        reopened = IFA.CrossPlatformFileReceipts(self.root)
        self.assertEqual(reopened.publish(self.spec_reg, 1, r1), sha1)
        res = reopened.resume(self.spec_reg)
        self.assertTrue(res["completed"])
        self.assertEqual(res["next_attempt"], 2)
        self.assertEqual(res["receipt"]["sha256"], sha1)

        # Conflicting overwrite on attempt 1 must be rejected
        r1_conflict = self._make_receipt(self.spec_reg, actual=_res(cpu=201, gpu=100, io=1024))
        with self.assertRaises(ValueError):
            reopened.publish(self.spec_reg, 1, r1_conflict)

    def test_partial_orphan_failed_overrun_and_reconcile(self) -> None:
        (self.repo.receipts / ".partial-crash-orphan").write_text('{"truncated":', encoding="utf-8")
        self.assertFalse(self.repo.resume(self.spec_reg)["completed"])

        # Resource overrun cannot be marked COMPLETE
        with self.assertRaises(ValueError):
            self.repo.publish(self.spec_reg, 1, self._make_receipt(self.spec_reg, actual=_res(cpu=5000)))

        # Record failed attempt 1, then succeed on attempt 2
        r_fail = self._make_receipt(
            self.spec_reg, attempt=1, outcome="FAILED", failure="SESSION_LOST", actual=_res(cpu=300)
        )
        self.repo.publish(self.spec_reg, 1, r_fail)
        st = self.repo.resume(self.spec_reg)
        self.assertFalse(st["completed"])
        self.assertEqual(st["next_attempt"], 2)

        self.repo.publish(self.spec_reg, 2, self._make_receipt(self.spec_reg, attempt=2))
        self.assertTrue(self.repo.resume(self.spec_reg)["completed"])

        # Competing second COMPLETE attempt triggers explicit operator reconciliation error
        self.repo.publish(self.spec_reg, 3, self._make_receipt(self.spec_reg, attempt=3))
        with self.assertRaises(ValueError):
            self.repo.resume(self.spec_reg)

    def test_cas_corruption_and_path_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.repo.store.verify("../escape")
        self.repo.publish(self.spec_reg, 1, self._make_receipt(self.spec_reg))
        (self.repo.store.root / self.out_reg["evidence"]).write_bytes(b"corrupted-bytes")
        with self.assertRaises(ValueError):
            IFA.CrossPlatformFileReceipts(self.root).resume(self.spec_reg)

    def test_control_expiry_and_worker_stopped(self) -> None:
        spec_ctl = _spec("CONTROL", scroll="PHerc0211", calibration=self.cal)
        with self.assertRaises(ValueError):
            self.repo.publish(
                spec_ctl,
                1,
                self._make_receipt(
                    spec_ctl,
                    worker_stopped=False,
                    calibration=self.cal,
                    control_pass=True,
                    valid_until=500,
                ),
            )
        self.repo.publish(
            spec_ctl,
            1,
            self._make_receipt(spec_ctl, calibration=self.cal, control_pass=True, valid_until=500),
        )
        self.assertTrue(self.repo.resume(spec_ctl, now=499)["completed"])
        self.assertFalse(self.repo.resume(spec_ctl, now=500)["completed"])

    def test_dual_order_and_a01_single_order_15pct_audit_modes(self) -> None:
        spec_coarse = _spec("COARSE", calibration=self.cal)
        spec_deep = _spec("DEEP", calibration=self.cal)

        asc_out = {
            "asc_map": self.repo.store.put(b"asc-map"),
            "asc_support": self.repo.store.put(b"asc-sup"),
            "metadata": self.repo.store.put(b'{"ok":true}'),
        }
        both_out = {
            **asc_out,
            "desc_map": self.repo.store.put(b"desc-map"),
            "desc_support": self.repo.store.put(b"desc-sup"),
        }
        paired_cal = IFA.digest_obj("paired-calibration-key-0211")

        # 1. Default paired_both_orders rejects single-order payload
        with self.assertRaises(ValueError):
            self.repo.publish(spec_coarse, 1, self._make_receipt(spec_coarse, outputs=asc_out, orders=["asc"]))

        # 2. A01 B8/S2 single_order_audit_15pct accepts unambiguous non-audit COARSE unit
        audit_ok = {
            "audit_rate": 0.15,
            "audit_selected": False,
            "normal_orientation_ambiguous": False,
            "selected_order": "asc",
            "paired_calibration_key": paired_cal,
        }
        self.repo.publish(
            spec_coarse,
            1,
            self._make_receipt(
                spec_coarse,
                outputs=asc_out,
                orders=["asc"],
                order_mode="single_order_audit_15pct",
                audit_policy=audit_ok,
            ),
        )
        self.assertTrue(self.repo.resume(spec_coarse)["completed"])

        # 3. Audit-selected (15% draw) OR ambiguous normal MUST supply BOTH orders
        audit_sel = {**audit_ok, "audit_selected": True}
        with self.assertRaises(ValueError):
            self.repo.publish(
                spec_coarse,
                2,
                self._make_receipt(
                    spec_coarse,
                    attempt=2,
                    outputs=asc_out,
                    orders=["asc"],
                    order_mode="single_order_audit_15pct",
                    audit_policy=audit_sel,
                ),
            )
        # Audit-selected with BOTH orders succeeds
        spec_coarse_aud = _spec("COARSE", unit="synthetic-audit-unit", calibration=self.cal)
        self.repo.publish(
            spec_coarse_aud,
            1,
            self._make_receipt(
                spec_coarse_aud,
                outputs=both_out,
                orders=["asc", "desc"],
                order_mode="single_order_audit_15pct",
                audit_policy=audit_sel,
            ),
        )

        # 4. Audit rate < 0.15 rejected
        with self.assertRaises(ValueError):
            self.repo.publish(
                spec_coarse,
                3,
                self._make_receipt(
                    spec_coarse,
                    attempt=3,
                    outputs=asc_out,
                    orders=["asc"],
                    order_mode="single_order_audit_15pct",
                    audit_policy={**audit_ok, "audit_rate": 0.10},
                ),
            )

        # 5. single_order_audit_15pct is FORBIDDEN on DEEP (S4 always requires both orders)
        with self.assertRaises(ValueError):
            self.repo.publish(
                spec_deep,
                1,
                self._make_receipt(
                    spec_deep,
                    outputs=asc_out,
                    orders=["asc"],
                    order_mode="single_order_audit_15pct",
                    audit_policy=audit_ok,
                ),
            )
        self.repo.publish(spec_deep, 1, self._make_receipt(spec_deep, outputs=both_out, orders=["asc", "desc"]))
        self.assertTrue(self.repo.resume(spec_deep)["completed"])

    def test_resume_rejects_failed_after_complete_and_a02_spec_supported(self) -> None:
        # Conflicting FAILED attempt after a COMPLETE attempt must fail resume
        spec_conf = _spec("COARSE", scroll="PHerc0826", calibration=self.cal)
        both_out = {
            "asc_map": self.repo.store.put(b"asc-map"),
            "asc_support": self.repo.store.put(b"asc-sup"),
            "desc_map": self.repo.store.put(b"desc-map"),
            "desc_support": self.repo.store.put(b"desc-sup"),
            "metadata": self.repo.store.put(b'{"ok":true}'),
        }
        self.repo.publish(spec_conf, 1, self._make_receipt(spec_conf, attempt=1, outputs=both_out, orders=["asc", "desc"]))
        self.repo.publish(
            spec_conf,
            2,
            self._make_receipt(spec_conf, attempt=2, outcome="FAILED", failure="SESSION_LOST"),
        )
        with self.assertRaisesRegex(ValueError, "Non-terminal COMPLETE"):
            self.repo.resume(spec_conf)

        # A02 order_policy='single_order_plus_audit' spec schema validation
        plan_sha = self.repo.store.put(b'{"schema":"funnel.order_audit.v1"}')
        geom_sha = self.repo.store.put(b"geom")
        a02_spec = IFA.validate_job_spec(
            {
                "scroll": "PHerc0826",
                "unit": "u_a02",
                "frame": IFA.digest_obj("f0"),
                "stage": "COARSE",
                "recipe": IFA.digest_obj("r0"),
                "inputs": {"geometry": geom_sha, "order_audit_plan": plan_sha},
                "deps": [],
                "calibration": self.cal,
                "order_policy": "single_order_plus_audit",
            }
        )
        self.assertEqual(a02_spec["order_policy"], "single_order_plus_audit")
        with self.assertRaises(ValueError):
            IFA.validate_job_spec(
                {
                    "scroll": "PHerc0826",
                    "unit": "u_bad",
                    "frame": IFA.digest_obj("f0"),
                    "stage": "COARSE",
                    "recipe": IFA.digest_obj("r0"),
                    "inputs": {"geometry": geom_sha, "order_audit_plan": plan_sha},
                    "deps": [],
                    "calibration": self.cal,
                }
            )


class DomainContractsAndBridgeTests(unittest.TestCase):
    def test_2d_shear_quarantine_and_t2_vs_flatboi_distinction(self) -> None:
        # Winner-grade / PHerc0125 pscamillo mesh passes without quarantine
        ok = IFA.validate_geometry_contract(
            {
                "parameterization_method": "raw_fit_grid",
                "shear_abs_p95_deg": 5.339,
                "pct_area_gt_20deg": 0.198,
                "area_inflation_true_over_nominal": 1.0408,
                "snap_passes": 2,
            }
        )
        self.assertFalse(ok["shear_quarantined"])
        self.assertTrue(ok["admissible_for_negative_claim"])
        self.assertFalse(ok["snap_is_identity_certificate"])

        # HF PHerc0826 / PHerc0191 high-shear mesh is quarantined
        quar = IFA.validate_geometry_contract(
            {
                "parameterization_method": "raw_fit_grid",
                "shear_abs_p95_deg": 58.963,
                "pct_area_gt_20deg": 70.802,
                "area_inflation_true_over_nominal": 1.9571,
                "snap_passes": 2,
            }
        )
        self.assertTrue(quar["shear_quarantined"])
        self.assertFalse(quar["admissible_for_negative_claim"])
        self.assertEqual(quar["recommended_remediation"], "run_2d_flattening_flatboi_or_metric_flatten")

        # 1D Exp T2 resampling cannot claim 2D flattening equivalence
        with self.assertRaises(ValueError):
            IFA.validate_geometry_contract(
                {
                    "parameterization_method": "1d_arc_length_resample_t2",
                    "claims_2d_flattening": True,
                    "shear_abs_p95_deg": 13.58,
                    "pct_area_gt_20deg": 0.0,
                    "area_inflation_true_over_nominal": 0.9987,
                }
            )

        t2_honest = IFA.validate_geometry_contract(
            {
                "parameterization_method": "1d_arc_length_resample_t2",
                "shear_abs_p95_deg": 13.58,
                "pct_area_gt_20deg": 0.0,
                "area_inflation_true_over_nominal": 0.9987,
            }
        )
        self.assertTrue(t2_honest["is_1d_t2_resampled"])
        self.assertFalse(t2_honest["is_2d_flattened"])

    def test_pixel_center_exact_slice_lists_and_non_padded_support(self) -> None:
        self.assertEqual(IFA.expected_slice_indices("studio", "desc", num_slices=31), IFA.STUDIO_DESC_SLICES_31)
        self.assertEqual(IFA.expected_slice_indices("studio", "asc", num_slices=31), IFA.STUDIO_ASC_SLICES_31)
        self.assertEqual(IFA.expected_slice_indices("ahn_s0", "asc", num_slices=31), IFA.AHN_S0_ASC_SLICES_31)
        self.assertEqual(IFA.expected_slice_indices("ahn_s0", "desc", num_slices=31), IFA.AHN_S0_DESC_SLICES_31)

        px = 732641
        vox = 9.362
        cm2 = round(px * ((vox / 10000.0) ** 2), 4)
        base_samp = {
            "pixel_center_frame": "vertex_aligned_p_times_scale",
            "voxel_um": vox,
            "num_slices": 31,
            "mesh_index": 15,
            "slice_step_vx": 1.0,
            "source_support_pixels": px,
            "nominal_pixels": 800000,
            "valid_ct_cm2": cm2,
            "padded_pixels_excluded_from_area": True,
            "missing_chunk_fill_excluded": True,
        }
        dets_ok = {
            "studio_desc": {"family": "studio", "order": "desc", "slice_indices": IFA.STUDIO_DESC_SLICES_31},
            "studio_asc": {"family": "studio", "order": "asc", "slice_indices": IFA.STUDIO_ASC_SLICES_31},
            "ahn_s0_asc": {"family": "ahn_s0", "order": "asc", "slice_indices": IFA.AHN_S0_ASC_SLICES_31},
            "ahn_s0_desc": {"family": "ahn_s0", "order": "desc", "slice_indices": IFA.AHN_S0_DESC_SLICES_31},
        }
        res = IFA.validate_sampling_and_detector_contract(base_samp, dets_ok)
        self.assertFalse(res["half_pixel_offset_applied"])

        # Reversed slice list for studio_desc must fail
        dets_bad = {
            "studio_desc": {"family": "studio", "order": "desc", "slice_indices": IFA.STUDIO_ASC_SLICES_31},
        }
        with self.assertRaises(ValueError):
            IFA.validate_sampling_and_detector_contract(base_samp, dets_bad)

        # Inflated/padded area mismatch must fail
        with self.assertRaises(ValueError):
            IFA.validate_sampling_and_detector_contract({**base_samp, "valid_ct_cm2": cm2 + 0.5}, dets_ok)

        # Missing padded exclusion flag must fail
        with self.assertRaises(ValueError):
            IFA.validate_sampling_and_detector_contract(
                {**base_samp, "padded_pixels_excluded_from_area": False}, dets_ok
            )

    def test_q34_non_gating_q33_recall_and_telemetry(self) -> None:
        tel = {
            "gpu_name": "RTX 4060 Laptop 8 GiB",
            "peak_allocated_mib": 1608,
            "peak_reserved_mib": 1740,
            "vram_cap_fraction": 0.35,
            "max_temp_c": 74,
            "post_run_temp_c": 63,
            "batch_pause_ms": 350,
            "hardware_or_oom_incidents": 0,
            "worker_stopped": True,
        }
        diag = {
            "q34_used_as_hard_gate": False,
            "q34_diagnostic_score": 0.0827,
            "q33_mixed_region_recall": {
                "reported": True,
                "stratum": "pherc0211_public_control",
                "recall_at_budget": 0.92,
                "worst_stratum_recall": 0.87,
            },
        }
        out = IFA.validate_diagnostics_and_telemetry(diag, tel)
        self.assertFalse(out["q34_used_as_hard_gate"])
        self.assertTrue(out["q34_bleed_through_floor_acknowledged"])

        # Attempting to use Q34 as hard gate must fail
        with self.assertRaises(ValueError):
            IFA.validate_diagnostics_and_telemetry({**diag, "q34_used_as_hard_gate": True}, tel)

        # Thermal limit breach must fail
        with self.assertRaises(ValueError):
            IFA.validate_diagnostics_and_telemetry(diag, {**tel, "post_run_temp_c": 85})

    def test_rtx4060_summary_bridge_and_public_sanitization(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ifunnel-bridge-") as tmp:
            repo = IFA.CrossPlatformFileReceipts(Path(tmp) / "private")
            src_in = repo.store.put(b"synthetic-geometry-input")
            cal_key = IFA.digest_obj("rtx4060-studio-s96-fp16-v1")
            px = 732641
            vox = 9.362
            cm2 = round(px * ((vox / 10000.0) ** 2), 4)
            summary = {
                "scroll": "PHerc0211",
                "mesh_tag": "public_control_0211",
                "voxel_um": vox,
                "num_slices": 31,
                "snap_passes": 2,
                "uv_crop": [12, 76, 184, 214],  # private key in raw summary; must not leak to public aggregate
                "valid_ct_pixels": px,
                "valid_ct_cm2": cm2,
                "ct_mib_fetched": 340.0,
                "m7_mib_fetched": 159.5,
                "total_seconds": 100.0,
                "detectors": {
                    "studio_desc": {"valid_pixels": px, "valid_cm2": cm2, "p995": 0.926, "frac_gt_07": 0.286, "seconds": 25.0},
                    "studio_asc": {"valid_pixels": px, "valid_cm2": cm2, "p995": 0.878, "frac_gt_07": 0.042, "seconds": 25.0},
                },
                "top_hotspots": [{"rank": 1, "y": 10, "x": 20}],
            }
            emitted = IFA.emit_r1_receipt_from_rtx4060_summary(
                repo,
                summary=summary,
                stage="COARSE",
                unit_private_id="SECRET_UNIT_PRIVATE_99",
                frame_hash=IFA.digest_obj("frame-0211"),
                recipe_hash=IFA.digest_obj("studio-s96-snap2"),
                calibration_key=cal_key,
                deps=[IFA.digest_obj("dep-geom"), IFA.digest_obj("dep-ctl")],
                inputs={"geom": src_in},
                artifact_payloads={
                    "asc_map": b"asc-prob-bytes",
                    "desc_map": b"desc-prob-bytes",
                    "asc_support": b"asc-sup-bytes",
                    "desc_support": b"desc-sup-bytes",
                },
                geometry_audit={
                    "parameterization_method": "flatboi",
                    "shear_abs_p95_deg": 1.756,
                    "pct_area_gt_20deg": 0.044,
                    "area_inflation_true_over_nominal": 0.9999,
                },
                telemetry={
                    "gpu_name": "RTX 4060 Laptop 8 GiB",
                    "peak_allocated_mib": 1905,
                    "peak_reserved_mib": 2108,
                    "vram_cap_fraction": 0.50,
                    "max_temp_c": 74,
                    "post_run_temp_c": 57,
                    "batch_pause_ms": 300,
                    "hardware_or_oom_incidents": 0,
                    "worker_stopped": True,
                },
                grant=_res(cpu=200_000, gpu=200_000, io=700_000_000),
            )
            self.assertEqual(emitted["outcome"], "COMPLETE")
            self.assertFalse(emitted["shear_quarantined"])
            self.assertNotIn("order_mode", emitted["receipt"])  # Strict file_receipts.validate parity
            env = repo.read(emitted["spec"], 1)
            meta_doc = repo.store.read_json(emitted["metadata_sha256"])
            pub = IFA.sanitize_receipt_for_public_aggregate(env, meta_doc)
            serialized = json.dumps(pub)
            self.assertNotIn("SECRET_UNIT_PRIVATE_99", serialized)
            self.assertNotIn("uv_crop", serialized)
            self.assertNotIn("top_hotspots", serialized)
            IFA.assert_public_export_clean(pub)

            # Accept mit_arap_2d as valid 2D flattening method
            arap_geom = IFA.validate_geometry_contract(
                {
                    "parameterization_method": "mit_arap_2d",
                    "shear_abs_p95_deg": 7.371,
                    "pct_area_gt_20deg": 0.151,
                    "area_inflation_true_over_nominal": 0.9893,
                }
            )
            self.assertTrue(arap_geom["is_2d_flattened"])
            self.assertFalse(arap_geom["shear_quarantined"])

            # Reject mixed-family pairing (studio_desc + ahn_s0_asc)
            bad_summary = dict(summary)
            bad_summary["detectors"] = {
                "studio_desc": summary["detectors"]["studio_desc"],
                "ahn_s0_asc": summary["detectors"]["studio_asc"],
            }
            with self.assertRaisesRegex(ValueError, "Mixed-family"):
                IFA.emit_r1_receipt_from_rtx4060_summary(
                    repo,
                    summary=bad_summary,
                    stage="COARSE",
                    unit_private_id="MIXED_BAD_UNIT",
                    frame_hash=IFA.digest_obj("frame-0211"),
                    recipe_hash=IFA.digest_obj("studio-s96-snap2"),
                    calibration_key=cal_key,
                    deps=[],
                    inputs={"geom": src_in},
                    artifact_payloads={
                        "asc_map": b"a",
                        "desc_map": b"d",
                        "asc_support": b"as",
                        "desc_support": b"ds",
                    },
                    geometry_audit={
                        "parameterization_method": "mit_arap_2d",
                        "shear_abs_p95_deg": 7.371,
                        "pct_area_gt_20deg": 0.151,
                        "area_inflation_true_over_nominal": 0.9893,
                    },
                    telemetry={
                        "gpu_name": "RTX 4060 Laptop 8 GiB",
                        "peak_allocated_mib": 1905,
                        "peak_reserved_mib": 2108,
                        "vram_cap_fraction": 0.50,
                        "max_temp_c": 74,
                        "post_run_temp_c": 57,
                        "batch_pause_ms": 300,
                        "hardware_or_oom_incidents": 0,
                        "worker_stopped": True,
                    },
                    grant=_res(cpu=200_000, gpu=200_000, io=700_000_000),
                )

            # Reject .npy artifact with NaN score
            import io as _io
            nan_buf = _io.BytesIO()
            np.save(nan_buf, np.array([[0.1, np.nan]], dtype=np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "NONFINITE_SCORE"):
                IFA.emit_r1_receipt_from_rtx4060_summary(
                    repo,
                    summary=summary,
                    stage="COARSE",
                    unit_private_id="NAN_BAD_UNIT",
                    frame_hash=IFA.digest_obj("frame-0211"),
                    recipe_hash=IFA.digest_obj("studio-s96-snap2"),
                    calibration_key=cal_key,
                    deps=[],
                    inputs={"geom": src_in},
                    artifact_payloads={
                        "asc_map": nan_buf.getvalue(),
                        "desc_map": b"d",
                        "asc_support": b"as",
                        "desc_support": b"ds",
                    },
                    geometry_audit={
                        "parameterization_method": "mit_arap_2d",
                        "shear_abs_p95_deg": 7.371,
                        "pct_area_gt_20deg": 0.151,
                        "area_inflation_true_over_nominal": 0.9893,
                    },
                    telemetry={
                        "gpu_name": "RTX 4060 Laptop 8 GiB",
                        "peak_allocated_mib": 1905,
                        "peak_reserved_mib": 2108,
                        "vram_cap_fraction": 0.50,
                        "max_temp_c": 74,
                        "post_run_temp_c": 57,
                        "batch_pause_ms": 300,
                        "hardware_or_oom_incidents": 0,
                        "worker_stopped": True,
                    },
                    grant=_res(cpu=200_000, gpu=200_000, io=700_000_000),
                )

    def test_privacy_guard_catches_banned_keys_paths_and_tokens(self) -> None:
        self.assertTrue(IFA.find_privacy_violations({"uv_crop": [1, 2, 3, 4]}))
        self.assertTrue(IFA.find_privacy_violations({"top_hotspots": [{"x": 1}]}))
        bad_win = "C:" + "\\" + "Temp\\secret.json"
        self.assertTrue(IFA.find_privacy_violations({"note": bad_win}))
        self.assertTrue(IFA.find_privacy_violations({"note": "ghp_" + "A" * 24}))


class NormalCorrespondenceDedupAndS0Tests(unittest.TestCase):
    def test_64vx_cell_fails_while_3vx_normal_separates_parallel_sheets(self) -> None:
        # Two flat parallel sheets 8 voxels apart inside the SAME 64x64x64 cell
        gy, gx = np.meshgrid(np.linspace(8.0, 56.0, 8), np.linspace(8.0, 56.0, 8), indexing="ij")
        z_a = np.full_like(gx, 20.0)
        z_b_distinct = np.full_like(gx, 28.0)  # 8 vx away (next papyrus winding)
        z_b_snapped = np.full_like(gx, 21.5)   # 1.5 vx away (collapsed by m7 snap)
        valid = np.ones_like(gx, dtype=bool)

        sep = IFA.evaluate_normal_correspondence_dedup(
            (gx, gy, z_a, valid),
            (gx, gy, z_b_distinct, valid),
        )
        self.assertEqual(sep["cell_64vx_same_vert_frac"], 1.0)
        self.assertEqual(sep["normal_correspondence_frac"], 0.0)
        self.assertAlmostEqual(sep["surface_dist_median_vox"], 8.0, places=2)

        col = IFA.evaluate_normal_correspondence_dedup(
            (gx, gy, z_a, valid),
            (gx, gy, z_b_snapped, valid),
        )
        self.assertEqual(col["cell_64vx_same_vert_frac"], 1.0)
        self.assertEqual(col["normal_correspondence_frac"], 1.0)
        self.assertAlmostEqual(col["surface_dist_median_vox"], 1.5, places=2)

    def test_s0_joiner_on_synthetic_fixture_and_privacy_clean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ifunnel-s0-") as tmp:
            root = Path(tmp)
            s0_csv = root / "S0.csv"
            s0_csv.write_text(
                "scroll,status,z_fraction,z_slice,slabs,clean_slabs,edge_rise_um_q75,merged_frac,coherence,air_frac\n"
                'PHerc. 0125,"eligible, unread",0.2,4000,4,2,18.0,0.05,0.60,0.35\n'
                'PHerc. 0268,"eligible, unread",0.2,3000,4,0,32.0,0.50,0.30,0.10\n'
                'PHerc. 0826,"eligible, unread",0.56,9500,4,3,14.0,0.04,0.72,0.42\n'
                'PHerc. 0211,read (control),0.32,6200,4,2,18.0,0.08,0.62,0.31\n',
                encoding="utf-8",
            )
            cov = root / "cov.json"
            cov.write_text(
                json.dumps(
                    {
                        "scrolls": [
                            {"scroll": "PHerc0125", "voxel_um": 9.362, "pscamillo_meshes": 340, "s3_has_m7_and_normal_grids": True},
                            {"scroll": "PHerc0268", "voxel_um": 8.64, "pscamillo_meshes": 0, "s3_has_m7_and_normal_grids": True},
                            {"scroll": "PHerc0826", "voxel_um": 9.362, "hf_spiral_fit_runs": 3, "s3_has_m7_and_normal_grids": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            led = root / "led.json"
            led.write_text(
                json.dumps(
                    {
                        "meshes": [
                            {"scroll": "PHerc0125", "valid_ct_cm2": 4.0, "coarse_pass": {"total_seconds": 1200.0}, "two_family_followup": None}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            gq = root / "gq.json"
            gq.write_text(
                json.dumps(
                    {
                        "sets": [
                            {
                                "set_id": "pherc0826_hf_spiral_fits",
                                "pooled": {
                                    "meshes": 33,
                                    "shear_1cell": {"abs_median_deg": 28.0, "abs_p95_deg": 58.0, "pct_area_gt_20deg": 70.0},
                                    "area_inflation_true_over_nominal": 1.95,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            q22 = root / "q22.json"
            q22.write_text(
                json.dumps(
                    {
                        "scrolls": [
                            {
                                "scroll_id": "PHerc0125",
                                "affine_to_ref_control": {"gain": 1.38, "bias": -53.7},
                                "studio_envelope_check": {"flagged": True, "flag_reasons": ["abs_bias_gt_15"]},
                                "nonzero_ct": {"p50": 109.0, "frac_ge_200": 0.0033},
                                "post_nyul_udupa_validation": {"flagged_after_nyul": False},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            joined = IFA.build_s0_19scroll_inventory(
                s0_csv_path=s0_csv,
                coverage_matrix_path=cov,
                screen_ledger_path=led,
                grid_quality_path=gq,
                q22_audit_path=q22,
                verify_pinned_s0_blob=False,
            )
            by_sc = {r["scroll"]: r for r in joined["eligible_scrolls"]}
            self.assertEqual(by_sc["PHerc0268"]["recommended_next_stage"], "S0_GEOMETRY_FIRST_REPAIR")
            self.assertEqual(by_sc["PHerc0826"]["recommended_next_stage"], "S2_2D_FLATTEN_BEFORE_SCREEN")
            self.assertEqual(by_sc["PHerc0125"]["recommended_next_stage"], "S2_COARSE_SCREEN_EXISTING_CHARTS")
            self.assertNotIn("z_slice", json.dumps(joined))
            IFA.assert_public_export_clean(joined)

    def test_committed_adapter_and_validation_json_have_zero_leaks_and_zero_torch(self) -> None:
        adapter_path = GEMINI_DIR / "industrial_funnel_adapter.py"
        adapter_txt = adapter_path.read_text(encoding="utf-8")
        gpu_modules = ("torch", "cupy", "jax", "tensorflow", "numba.cuda", "pycuda")
        tree = ast.parse(adapter_txt)
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            for mod in mods:
                self.assertFalse(
                    any(mod == g or mod.startswith(g + ".") for g in gpu_modules),
                    f"industrial_funnel_adapter.py imports forbidden GPU module {mod!r}",
                )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import industrial_funnel_adapter; "
                f"print([m for m in {gpu_modules!r} if m in sys.modules])",
                str(GEMINI_DIR),
            ],
            capture_output=True,
            text=True,
            env={"CUDA_VISIBLE_DEVICES": "-1", **os.environ},
            timeout=60,
            check=True,
        )
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "[]")
        self.assertEqual(LG.find_leaks(adapter_txt), [])
        val_path = GEMINI_DIR / "evidence" / "industrial_funnel_validation.json"
        if val_path.is_file():
            val_txt = val_path.read_text(encoding="utf-8")
            self.assertEqual(LG.find_leaks(val_txt), [])
            data = json.loads(val_txt)
            IFA.assert_public_export_clean(data)
            self.assertEqual(data["s0_19scroll_inventory"]["summary"]["eligible_scrolls"], 19)
            self.assertEqual(data["windows_compatibility_audit"]["patched_replay"]["passed_count"], 48)
            self.assertEqual(data["windows_compatibility_audit"]["patched_replay"]["tests_run"], 48)
        self.assertEqual(LG.find_leaks("cross-agent interop check", user="agent"), [])

    def test_g4_boundary_vertices_achieve_100pct_self_correspondence(self) -> None:
        # Irregular valid mask leaving right/bottom boundary vertices without outgoing valid quads
        vy, vx = np.meshgrid(np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32), indexing="ij")
        x = vx * 15.0
        y = vy * 15.0
        z = np.full_like(x, 100.0)
        x[3:6, 6:12] = -1.0
        y[3:6, 6:12] = -1.0
        z[3:6, 6:12] = -1.0
        self_rep = IFA.evaluate_normal_correspondence_dedup(
            (x, y, z, x >= 0),
            (x, y, z, x >= 0),
        )
        self.assertEqual(self_rep["normal_correspondence_frac"], 1.0)
        self.assertEqual(self_rep["euclid_le_threshold_frac"], 1.0)


if __name__ == "__main__":
    unittest.main()
