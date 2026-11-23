# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline unit tests for Exp T4 (Q22): Multi-Scroll CT Radiometric Shift Audit."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

from gemini_lane.q22_radiometric_shift_audit import (
    AFFINE_LANDMARK_PERCENTILES,
    DEFAULT_NYUL_PERCENTILES,
    DEFAULT_SCROLL_SPECS,
    MIB,
    BudgetExceededError,
    BudgetedRemoteZarrV2,
    MultiScrollByteBudget,
    ScrollSpec,
    assert_no_local_paths,
    assert_receipt_privacy,
    compute_ct_stats_from_uint8_hist,
    compute_ct_stats_from_values,
    evaluate_studio_envelope_flags,
    fit_affine_to_reference,
    nyul_udupa_standardize_stack,
    run_radiometric_shift_audit,
)


def _write_local_zarr_v2(
    root: Path,
    arr: np.ndarray,
    chunks: tuple[int, int, int] = (16, 16, 16),
    *,
    dimension_separator: str = ".",
    compress: bool = True,
) -> Path:
    """Create a minimal local Zarr v2 level-0 volume for offline tests."""
    lvl = root / "0"
    lvl.mkdir(parents=True, exist_ok=True)
    compressor = {"id": "zlib", "level": 1} if compress else None
    meta = {
        "chunks": list(chunks),
        "compressor": compressor,
        "dimension_separator": dimension_separator,
        "dtype": arr.dtype.str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(arr.shape),
        "zarr_format": 2,
    }
    (lvl / ".zarray").write_text(json.dumps(meta), encoding="utf-8")

    sz, sy, sx = arr.shape
    cz, cy, cx = chunks
    nz = (sz + cz - 1) // cz
    ny = (sy + cy - 1) // cy
    nx = (sx + cx - 1) // cx
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                sub = arr[
                    iz * cz : min((iz + 1) * cz, sz),
                    iy * cy : min((iy + 1) * cy, sy),
                    ix * cx : min((ix + 1) * cx, sx),
                ]
                if not np.any(sub):
                    continue  # leave sparse chunk as missing (404)
                padded = np.zeros(chunks, dtype=arr.dtype)
                padded[: sub.shape[0], : sub.shape[1], : sub.shape[2]] = sub
                raw = padded.tobytes(order="C")
                payload = zlib.compress(raw, level=1) if compress else raw
                key = dimension_separator.join((str(iz), str(iy), str(ix)))
                out_path = lvl / key
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(payload)
    return root


class TestNyulUdupaStandardizer(unittest.TestCase):
    def test_piecewise_linear_landmark_alignment(self) -> None:
        rng = np.random.default_rng(42)
        # Skewed shifted distribution with bright tail
        base = rng.normal(loc=145.0, scale=28.0, size=(8, 64, 64))
        base = np.clip(np.round(base), 10, 245).astype(np.float32)
        # Inject zeros and invalid masked pixels
        mask = np.ones((64, 64), dtype=bool)
        mask[:8, :8] = False
        base[:, -5:, -5:] = 0.0

        ref_landmarks = (22.0, 58.0, 82.0, 108.0, 134.0, 162.0, 195.0)
        std = nyul_udupa_standardize_stack(
            base,
            valid_mask=mask,
            ref_landmarks=ref_landmarks,
            percentiles=DEFAULT_NYUL_PERCENTILES,
            out_range=(1.0, 255.0),
        )
        self.assertEqual(std.shape, base.shape)
        self.assertEqual(std.dtype, np.float32)

        # Invalid & zero pixels must remain exact 0.0
        self.assertTrue(np.all(std[:, :8, :8] == 0.0))
        self.assertTrue(np.all(std[:, -5:, -5:] == 0.0))

        fg = np.broadcast_to(mask[None, :, :], base.shape) & (base > 0.0)
        out_vals = std[fg]
        self.assertGreaterEqual(float(out_vals.min()), 1.0)
        self.assertLessEqual(float(out_vals.max()), 255.0)

        out_pcts = np.percentile(out_vals, DEFAULT_NYUL_PERCENTILES)
        np.testing.assert_allclose(out_pcts, ref_landmarks, atol=0.35)

    def test_dict_ref_landmarks_and_nan_preservation(self) -> None:
        arr = np.linspace(20.0, 220.0, 4 * 32 * 32, dtype=np.float32).reshape(4, 32, 32)
        arr[0, 0, 0] = np.nan
        arr[0, 0, 1] = -5.0
        arr[0, 0, 2] = 0.0
        ref_dict = {
            "p01": 15.0,
            "p10": 50.0,
            "p25": 75.0,
            "p50": 100.0,
            "p75": 125.0,
            "p90": 150.0,
            "p99": 185.0,
        }
        out = nyul_udupa_standardize_stack(arr, None, ref_dict)
        self.assertEqual(float(out[0, 0, 0]), 0.0)
        self.assertEqual(float(out[0, 0, 1]), 0.0)
        self.assertEqual(float(out[0, 0, 2]), 0.0)
        self.assertTrue(np.isfinite(out).all())

        fg = np.isfinite(arr) & (arr > 0.0)
        np.testing.assert_allclose(
            np.percentile(out[fg], DEFAULT_NYUL_PERCENTILES),
            [15.0, 50.0, 75.0, 100.0, 125.0, 150.0, 185.0],
            atol=0.1,
        )

    def test_degenerate_constant_and_empty_arrays(self) -> None:
        ref_lm = (20.0, 55.0, 80.0, 110.0, 140.0, 165.0, 190.0)

        # Constant positive foreground with zero border
        const = np.full((2, 16, 16), 95.0, dtype=np.float32)
        const[:, 0, :] = 0.0
        out_const = nyul_udupa_standardize_stack(const, None, ref_lm)
        self.assertTrue(np.isfinite(out_const).all())
        self.assertTrue(np.all(out_const[:, 0, :] == 0.0))
        self.assertTrue(np.allclose(out_const[:, 1:, :], 110.0))

        # All-zero / empty valid mask
        zeros = np.zeros((2, 8, 8), dtype=np.float32)
        out_zero = nyul_udupa_standardize_stack(zeros, None, ref_lm)
        self.assertTrue(np.all(out_zero == 0.0))

        # Partially collapsed landmarks (saturated high quantiles at 255)
        sat = np.concatenate([np.arange(50, 200, dtype=np.float32), np.full(200, 255.0, dtype=np.float32)])
        out_sat = nyul_udupa_standardize_stack(sat, None, ref_lm)
        self.assertTrue(np.isfinite(out_sat).all())
        self.assertGreaterEqual(float(out_sat.min()), 1.0)
        self.assertLessEqual(float(out_sat.max()), 255.0)

    def test_clipping_to_out_range(self) -> None:
        arr = np.array([1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 5000.0], dtype=np.float32)
        ref_lm = (5.0, 20.0, 40.0, 80.0, 120.0, 160.0, 240.0)
        out = nyul_udupa_standardize_stack(arr, None, ref_lm, out_range=(1.0, 255.0))
        self.assertGreaterEqual(float(out.min()), 1.0)
        self.assertLessEqual(float(out.max()), 255.0)


class TestStatsAndDriftFlagging(unittest.TestCase):
    def test_uint8_histogram_matches_dense_stats(self) -> None:
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 256, size=(32, 32, 32), dtype=np.uint8)
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.int64)

        s_hist = compute_ct_stats_from_uint8_hist(hist)
        s_dense = compute_ct_stats_from_values(arr.astype(np.float32))

        for k in (
            "p01",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "mean",
            "std",
            "mad",
            "robust_sigma",
            "skewness",
            "frac_ge_200",
            "frac_ge_220",
        ):
            self.assertAlmostEqual(
                float(s_hist["nonzero_ct"][k]),
                float(s_dense["nonzero_ct"][k]),
                places=3,
                msg=f"Mismatch on nonzero_ct[{k}]",
            )
            self.assertAlmostEqual(
                float(s_hist["papyrus_wall_core_gt_p50"][k]),
                float(s_dense["papyrus_wall_core_gt_p50"][k]),
                places=3,
                msg=f"Mismatch on papyrus_wall_core_gt_p50[{k}]",
            )

    def test_affine_fit_and_studio_envelope_flags(self) -> None:
        ref = [60.0, 85.0, 110.0, 135.0, 160.0]

        # Exact affine transform: ref = 1.0 * src + 0.0 -> unflagged
        fit_id = fit_affine_to_reference(ref, ref)
        self.assertAlmostEqual(fit_id["gain"], 1.0, places=5)
        self.assertAlmostEqual(fit_id["bias"], 0.0, places=4)
        flg_ok = evaluate_studio_envelope_flags(
            gain=fit_id["gain"], bias=fit_id["bias"], frac_ge_200=0.02
        )
        self.assertFalse(flg_ok["flagged"])
        self.assertEqual(flg_ok["flag_reasons"], [])

        # Gain outside [0.8, 1.2]: src stretched -> gain_to_ref = 0.65
        src_wide = [(y - 10.0) / 0.65 for y in ref]
        fit_g = fit_affine_to_reference(src_wide, ref)
        self.assertAlmostEqual(fit_g["gain"], 0.65, places=4)
        flg_g = evaluate_studio_envelope_flags(
            gain=fit_g["gain"], bias=fit_g["bias"], frac_ge_200=0.01
        )
        self.assertTrue(flg_g["flagged"])
        self.assertTrue(flg_g["flag_gain"])
        self.assertFalse(flg_g["flag_bias"])
        self.assertFalse(flg_g["flag_clip_200"])

        # Bias > 15
        src_shifted = [y - 22.0 for y in ref]
        fit_b = fit_affine_to_reference(src_shifted, ref)
        self.assertAlmostEqual(fit_b["bias"], 22.0, places=3)
        flg_b = evaluate_studio_envelope_flags(
            gain=fit_b["gain"], bias=fit_b["bias"], frac_ge_200=0.01
        )
        self.assertTrue(flg_b["flagged"])
        self.assertTrue(flg_b["flag_bias"])

        # Clipping >= 200 exceeds 5%
        flg_c = evaluate_studio_envelope_flags(gain=1.0, bias=0.0, frac_ge_200=0.08)
        self.assertTrue(flg_c["flagged"])
        self.assertTrue(flg_c["flag_clip_200"])


class TestBudgetEnforcementAndReceiptPrivacy(unittest.TestCase):
    def test_budget_enforces_per_scroll_and_global_caps(self) -> None:
        budget = MultiScrollByteBudget(per_scroll_limit_bytes=1000, global_limit_bytes=1500)
        budget.charge("S1", 600)
        with self.assertRaises(BudgetExceededError):
            budget.check_planned("S1", 401)
        with self.assertRaises(BudgetExceededError):
            budget.charge("S1", 401)
        budget.charge("S1", 400)
        self.assertEqual(budget.scroll_bytes("S1"), 1000)

        # S2 has 1000 per-scroll cap, but global only has 500 remaining
        self.assertEqual(budget.remaining("S2"), 500)
        with self.assertRaises(BudgetExceededError):
            budget.charge("S2", 501)
        budget.charge("S2", 500)
        self.assertEqual(budget.total_bytes, 1500)
        self.assertTrue(budget.summary()["within_limits"])

    def test_budgeted_zarr_rejects_oversized_local_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "tiny.zarr"
            arr = np.full((16, 16, 16), 120, dtype=np.uint8)
            _write_local_zarr_v2(root, arr, chunks=(16, 16, 16), compress=False)
            # Chunk is 4096 bytes raw; allow .zarray (~220 B) but deny chunk
            budget = MultiScrollByteBudget(per_scroll_limit_bytes=1000, global_limit_bytes=2000)
            with BudgetedRemoteZarrV2(root, scroll_id="TEST", budget=budget) as z:
                with self.assertRaises(BudgetExceededError):
                    z.get_chunk((0, 0, 0))

    def test_end_to_end_offline_audit_and_privacy_checks(self) -> None:
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            specs = [
                ScrollSpec("PHerc0841", "control", True, 9.366, 113, "PHerc0841/volumes/ctrl1.zarr"),
                ScrollSpec("PHerc0343", "control", True, 8.640, 116, "PHerc0343/volumes/ctrl2.zarr"),
                ScrollSpec("PHerc0306B", "control", True, 8.640, 116, "PHerc0306B/volumes/ctrl3.zarr"),
                ScrollSpec("PHerc0125", "eligible_unread", False, 9.362, 113, "PHerc0125/volumes/u1.zarr"),
                ScrollSpec("PHerc1218", "eligible_unread", False, 8.640, 116, "PHerc1218/volumes/u2.zarr"),
            ]
            overrides: dict[str, Path] = {}
            for sp in specs:
                vol = np.zeros((48, 48, 48), dtype=np.uint8)
                if sp.scroll_id == "PHerc1218":
                    # Strong radiometric shift + >5% clipping >= 200
                    vals = np.clip(np.round(rng.normal(175.0, 32.0, size=(48, 48, 48))), 1, 255).astype(np.uint8)
                    vol[:] = vals
                    # Leave exact center chunk sparse (<10% non-zero) to exercise +/-1 fallback
                    vol[16:32, 16:32, 16:32] = 0
                else:
                    vals = np.clip(np.round(rng.normal(110.0, 24.0, size=(48, 48, 48))), 1, 255).astype(np.uint8)
                    vol[:] = vals
                overrides[sp.scroll_id] = _write_local_zarr_v2(
                    base / f"{sp.scroll_id}.zarr", vol, chunks=(16, 16, 16)
                )

            receipt = run_radiometric_shift_audit(
                specs,
                root_overrides=overrides,
                per_scroll_limit_bytes=6 * MIB,
                global_limit_bytes=100 * MIB,
                target_chunks=3,
                min_chunks=2,
            )

            # Schema & privacy checks
            assert_receipt_privacy(receipt)
            self.assertEqual(receipt["schema_version"], "q22_radiometric_shift_audit_v1")
            self.assertTrue(receipt["budget"]["within_limits"])
            self.assertIn("PHerc1218", receipt["summary"]["flagged_eligible_unread"])
            self.assertNotIn("PHerc0125", receipt["summary"]["flagged_eligible_unread"])
            self.assertTrue(receipt["summary"]["all_scrolls_unflagged_after_nyul"])

            # Verify unread entries have zero chunk indices / coordinates and zero local paths
            text = json.dumps(receipt)
            win_prefix = "C:" + "\\" + "Users"
            posix_win_prefix = "C:" + "/" + "Users"
            self.assertNotIn(win_prefix, text)
            self.assertNotIn(posix_win_prefix, text)
            for rec in receipt["scrolls"]:
                if rec["role"] == "eligible_unread":
                    self.assertNotIn("control_chunk_indices", rec["sampling"])
                    self.assertNotIn("chunk_indices", rec["sampling"])
                    self.assertGreaterEqual(rec["sampling"]["chunks_accepted"], 2)
                else:
                    self.assertIn("control_chunk_indices", rec["sampling"])

    def test_privacy_guards_reject_leaks(self) -> None:
        fake_win_path = "C:" + "\\Users\\alice\\secret.json"
        fake_tok = "ghp_" + "1234567890abcdef1234567890"
        with self.assertRaises(ValueError):
            assert_no_local_paths({"bad": fake_win_path})
        with self.assertRaises(ValueError):
            assert_no_local_paths({"token": fake_tok})
        with self.assertRaises(ValueError):
            assert_receipt_privacy(
                {
                    "budget": {},
                    "scrolls": [
                        {
                            "scroll_id": "PHerc0125",
                            "role": "eligible_unread",
                            "sampling": {"chunk_indices": [[10, 12, 14]]},
                        }
                    ],
                }
            )
        with self.assertRaises(ValueError):
            assert_receipt_privacy(
                {
                    "budget": {},
                    "scrolls": [
                        {
                            "scroll_id": "PHerc0125",
                            "role": "eligible_unread",
                            "note": "fetched 0/12.14.16",
                        }
                    ],
                }
            )

    def test_default_manifest_has_expected_13_scrolls(self) -> None:
        self.assertEqual(len(DEFAULT_SCROLL_SPECS), 13)
        controls = [s.scroll_id for s in DEFAULT_SCROLL_SPECS if s.role == "control"]
        unread = [s.scroll_id for s in DEFAULT_SCROLL_SPECS if s.role == "eligible_unread"]
        self.assertEqual(controls, ["PHerc0841", "PHerc0343", "PHerc0306B", "PHerc0211"])
        self.assertEqual(
            unread,
            [
                "PHerc0125",
                "PHerc1203",
                "PHerc0191",
                "PHerc0826",
                "PHerc0846A",
                "PHerc0175A",
                "PHerc0358",
                "PHerc1218",
                "PHerc1545",
            ],
        )
        # Ensure PHerc0175A uses exact PHerc0175A prefix (not PHerc0521).
        spec_0175a = next(s for s in DEFAULT_SCROLL_SPECS if s.scroll_id == "PHerc0175A")
        self.assertTrue(spec_0175a.relative_path.startswith("PHerc0175A/volumes/"))
        self.assertEqual(AFFINE_LANDMARK_PERCENTILES, (10, 25, 50, 75, 90))

    def test_committed_evidence_receipt_schema_and_privacy(self) -> None:
        receipt_path = Path(__file__).resolve().parent / "evidence" / "q22_radiometric_shift_audit.json"
        if not receipt_path.is_file():
            self.skipTest("live evidence receipt not generated yet")
        raw = receipt_path.read_text(encoding="utf-8")
        receipt = json.loads(raw)
        assert_receipt_privacy(receipt)
        self.assertEqual(receipt["schema_version"], "q22_radiometric_shift_audit_v1")
        self.assertEqual(receipt["summary"]["total_scrolls_audited"], 13)
        self.assertTrue(receipt["budget"]["within_limits"])
        self.assertLessEqual(float(receipt["budget"]["global_mib"]), 100.0)
        for sid, info in receipt["budget"]["per_scroll"].items():
            self.assertLessEqual(float(info["mib"]), 6.0, msg=f"{sid} exceeded 6 MiB")
        for rec in receipt["scrolls"]:
            self.assertIn("papyrus_wall_core_gt_p50", rec)
            self.assertIn("papyrus_wall_upper_tercile_gt_p66_7", rec)
            if rec["role"] == "eligible_unread":
                self.assertNotIn("control_chunk_indices", rec["sampling"])
                self.assertNotIn("chunk_indices", rec["sampling"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
