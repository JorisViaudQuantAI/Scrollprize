# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only tests for ``seismic_rgt_unwrapper`` (Exp T9-Unwrap, Q46)."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import seismic_rgt_unwrapper as RGT  # noqa: E402

MODULE_PATH = HERE / "seismic_rgt_unwrapper.py"
EVIDENCE_PATH = HERE / "evidence" / "seismic_rgt_crushed_preflight.json"
GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba.cuda", "pycuda")


def make_crushed_stack(
    shape=(48, 64, 64), n_wraps=5, spacing=6.0, sigma=1.0, noise=4.0, seed=0,
    *, fold=3.0, pinch=1.5, gap=True,
):
    """Folded multi-wrap stack: spacing 5-7 vx (mean ``spacing``), one pinch, one delamination gap.

    Returns ``(ct_uint8, true_heights (n_wraps, A, B), gap_mask (A, B), gap_sheet_index)``.
    """
    rng = np.random.default_rng(seed)
    d, a, b = shape
    aa, bb = np.meshgrid(np.arange(a, dtype=float), np.arange(b, dtype=float), indexing="ij")
    fold_f = fold * np.sin(2 * np.pi * aa / a) * np.cos(np.pi * bb / b)
    local_sp = spacing + 1.0 * np.sin(2 * np.pi * bb / b)
    pinch_f = pinch * np.exp(-((aa - 0.65 * a) ** 2 + (bb - 0.35 * b) ** 2) / (2 * 6.0 ** 2))
    c = (n_wraps - 1) / 2
    heights = []
    for m in range(n_wraps):
        h = d / 2 + fold_f + (m - c) * local_sp
        if m == 2:
            h = h + pinch_f
        heights.append(h)
    gap_mask = ((aa - 0.3 * a) ** 2 + (bb - 0.7 * b) ** 2 < 36.0) if gap else np.zeros((a, b), bool)
    z = np.arange(d, dtype=float)[:, None, None]
    vol = np.zeros(shape)
    for m, h in enumerate(heights):
        texture = 1.0 + 0.35 * ndi.gaussian_filter(rng.normal(0, 1, (a, b)), 1.2) / 0.25
        amp = np.clip(texture, 0.4, 1.6) * (np.where(gap_mask, 0.0, 1.0) if m == 1 else 1.0)
        vol += amp * np.exp(-(z - h[None]) ** 2 / (2 * sigma ** 2))
    ct = np.clip(40 + 150 * vol + rng.normal(0, noise, shape), 1, 255).astype(np.uint8)
    return ct, np.stack(heights), gap_mask, 1


def make_pinched_pair(shape=(32, 64, 64), far_sep=7.0, pinch_sep=3.5, sigma=1.0, noise=3.0, seed=1):
    """Two wraps ``far_sep`` apart that pinch to ``pinch_sep`` around the box centre.

    Returns ``(ct_uint8, m7_like_uint8, true_heights (2, A, B), pinch_core_mask)``; the m7-like
    binary thresholds the clean sheet sum, so the two wraps fuse into one run at the pinch.
    """
    rng = np.random.default_rng(seed)
    d, a, b = shape
    aa, bb = np.meshgrid(np.arange(a, dtype=float), np.arange(b, dtype=float), indexing="ij")
    bump = np.exp(-(((aa - a / 2) ** 2 + (bb - b / 2) ** 2) / 9.0 ** 2) ** 2)
    half = 0.5 * (far_sep - (far_sep - pinch_sep) * bump)
    h0, h1 = d / 2 - half, d / 2 + half
    z = np.arange(d, dtype=float)[:, None, None]
    vol = np.zeros(shape)
    clean = np.zeros(shape)
    for h in (h0, h1):
        texture = 1.0 + 0.35 * ndi.gaussian_filter(rng.normal(0, 1, (a, b)), 1.2) / 0.25
        g = np.exp(-(z - h[None]) ** 2 / (2 * sigma ** 2))
        vol += np.clip(texture, 0.4, 1.6) * g
        clean += g
    ct = np.clip(40 + 150 * vol + rng.normal(0, noise, shape), 1, 255).astype(np.uint8)
    m7 = np.where(clean > 0.3, 255, 0).astype(np.uint8)
    core = bump > 0.8
    return ct, m7, np.stack([h0, h1]), core


def match_sheets(sheets, true_h):
    """(extracted index, true index, rmse) per extracted sheet on its valid columns."""
    out = []
    for i, s in enumerate(sheets):
        v = s["valid"]
        d = [np.median(np.abs(s["height"][v] - t[v])) for t in true_h]
        m = int(np.argmin(d))
        err = s["height"][v] - true_h[m][v]
        out.append((i, m, float(np.sqrt(np.mean(err ** 2)))))
    return out


class SyntheticCrushedStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct, cls.true_h, cls.gap_mask, cls.gap_sheet = make_crushed_stack()
        cls.metrics, cls.det = RGT.compare_rgt_vs_single_sheet_baseline(
            cls.ct, None, voxel_um=9.362, min_spacing_vx=4.0, max_spacing_vx=18.0, return_details=True
        )

    def test_structure_tensor_recovers_normal_axis_and_planarity(self):
        normals, plan, axis = RGT.estimate_structure_tensor_dip_3d(self.ct)
        self.assertEqual(axis, 0)
        self.assertEqual(normals.shape, self.ct.shape + (3,))
        self.assertTrue(np.all(normals[..., 0] >= 0))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-3)
        self.assertGreater(float(np.median(plan)), 0.75)
        self.assertTrue(np.all((plan >= 0) & (plan <= 1)))

    def test_wavenumber_recovers_wrap_spacing(self):
        k, info = RGT.estimate_local_wrap_wavenumber_3d(self.ct, 0, min_spacing_vx=4.0, max_spacing_vx=18.0)
        self.assertEqual(k.shape, self.ct.shape)
        self.assertLess(abs(info["mean_wrap_spacing_vx"] - 6.0), 1.0)
        self.assertLess(abs(info["median_wrap_spacing_vx"] - 6.0), 1.0)
        lam = 2 * np.pi / k
        self.assertTrue(np.all((lam >= 4.0 - 1e-4) & (lam <= 18.0 + 1e-4)))

    def test_rgt_phase_is_strictly_monotonic_with_phasor(self):
        rgt = self.det["rgt"]
        self.assertTrue(np.all(np.diff(rgt["phi"], axis=0) > 0))
        np.testing.assert_allclose(rgt["phasor_cos"] ** 2 + rgt["phasor_sin"] ** 2, 1.0, atol=1e-4)
        self.assertGreater(rgt["info"]["phase_normal_coherence"], 0.9)

    def test_multi_sheet_recovery_non_crossing_and_subvoxel(self):
        m = self.metrics
        self.assertGreaterEqual(m["sheet_count_rgt"], 4)
        self.assertEqual(m["crossing_fraction_rgt"], 0.0)
        matches = match_sheets(self.det["wheeler"]["sheets"], self.true_h)
        self.assertEqual(len({t for _, t, _ in matches}), len(matches), "two extracted sheets on one wrap")
        for _, _, rmse in matches:
            self.assertLess(rmse, 0.65)
        sheets = self.det["wheeler"]["sheets"]
        for lo, hi in zip(sheets[:-1], sheets[1:]):
            jv = lo["valid"] & hi["valid"]
            self.assertTrue(np.all(hi["height"][jv] > lo["height"][jv]))
        gap = [s for s, (_, t, _) in zip(sheets, matches) if t == self.gap_sheet]
        if gap:
            self.assertLess(float(gap[0]["valid"][self.gap_mask].mean()), 0.2, "sheet bridged the delamination gap")

    def test_wheeler_domain_flattens_horizontal_slices(self):
        m = self.metrics
        self.assertLess(m["wheeler_slice_variance_ratio"], 0.5 * m["cartesian_slice_variance_ratio"])
        self.assertGreater(m["wheeler_peak_contrast"], 0.2)

    def test_beats_single_sheet_baseline_on_usable_area(self):
        m = self.metrics
        self.assertEqual(m["sheet_count_single_baseline"], 1)
        self.assertGreater(m["usable_cm2_single_sheet_baseline"], 0.0)
        self.assertGreaterEqual(m["area_gain_ratio_vs_baseline"], 2.5)
        self.assertGreater(m["usable_cm2_rgt_total"], m["usable_cm2_best_sheet_rgt"])
        self.assertIsNotNone(m["median_f_parallel_rgt"])
        self.assertTrue(set(m) <= RGT.PUBLIC_RGT_METRIC_KEYS)


class PinchedWrapTests(unittest.TestCase):
    def test_pinch_stays_non_crossing_and_continuous(self):
        ct, m7, true_h, core = make_pinched_pair()
        lab, n = ndi.label(m7 >= 128, structure=np.ones((3, 3, 3), bool))
        self.assertEqual(n, 1, "26-connected binary should merge the pinched wraps into one component")

        m, det = RGT.compare_rgt_vs_single_sheet_baseline(ct, m7, voxel_um=9.362, return_details=True)
        self.assertEqual(m["sheet_count_rgt"], 2)
        self.assertEqual(m["crossing_fraction_rgt"], 0.0)
        sheets = det["wheeler"]["sheets"]
        for (_, t, rmse), s in zip(match_sheets(sheets, true_h), sheets):
            self.assertLess(rmse, 0.65)
            self.assertGreater(float(s["valid"][core].mean()), 0.9, "RGT sheet dropped the pinch")
            self.assertGreater(float(RGT._largest_component_2d(s["valid"]).mean()), 0.9)
            err = np.abs(s["height"] - true_h[t])[core & s["valid"]]
            self.assertLess(float(err.max()), 1.0)
        jv = sheets[0]["valid"] & sheets[1]["valid"]
        self.assertTrue(np.all(sheets[1]["height"][jv] > sheets[0]["height"][jv]))
        base_core = float(det["baseline_valid"][core].mean())
        self.assertLess(base_core, 0.5, "single-sheet column tracking should drop the pinch")


class ByteBudgetTests(unittest.TestCase):
    def test_budget_raises_before_exceeding_cap(self):
        b = RGT.ByteBudget(RGT.GLOBAL_HTTP_CAP_BYTES)
        self.assertEqual(b.cap, 83_886_080)
        b.charge(80 * 1024 * 1024 - 10)
        with self.assertRaises(RGT.BudgetExceeded):
            b.precheck(11)
        self.assertEqual(b.used, 80 * 1024 * 1024 - 10)
        with self.assertRaises(RGT.BudgetExceeded):
            b.charge(11)
        with self.assertRaises(ValueError):
            RGT.ByteBudget(0)

    def test_uncompressed_box_read_is_prechecked(self):
        class FakeReader:
            chunks = (128, 128, 128)
            _decode = None
            _expected_chunk_bytes = 128 ** 3

            def read_box(self, lo, hi):  # pragma: no cover - must not be reached
                raise AssertionError("read attempted past the budget")

        b = RGT.ByteBudget(128 ** 3 + 1)
        with self.assertRaises(RGT.BudgetExceeded):
            RGT._read_box_budgeted(FakeReader(), (100, 0, 0), (164, 96, 96), b)
        self.assertEqual(b.used, 0)


class CpuOnlyComplianceTests(unittest.TestCase):
    def test_ast_has_no_gpu_imports(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        for n in names:
            self.assertFalse(any(n == g or n.startswith(g + ".") for g in GPU_MODULES), n)

    def test_subprocess_import_loads_no_gpu_library(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import seismic_rgt_unwrapper; "
            f"print([m for m in {GPU_MODULES!r} if m in sys.modules])"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        out = subprocess.run([sys.executable, "-c", code, str(HERE)], capture_output=True, text=True,
                             env=env, timeout=180, check=True)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")


class PrivacyAndReceiptTests(unittest.TestCase):
    def _unread(self, **extra):
        return {"boxes": [{"scroll": "PHerc1545", "metrics": {"sheet_count_rgt": 3}, **extra}]}

    def test_validator_rejects_leaks(self):
        RGT.validate_public_receipt_privacy(self._unread())
        bad = [
            {"note": "C:\\Users\\someone\\data\\x.zarr"},
            {"note": "/Users/someone/scroll"},
            {"note": "/home/someone/scroll"},
            {"note": "file:///tmp/x"},
            {"note": "token ghp_" + "a" * 20},
            {"boxes": [{"scroll": "PHerc0306B", "bbox": [0, 0, 0, 1, 1, 1]}]},
            self._unread(origin_zyx=[1, 2, 3]),
            self._unread(chunk_indices=[[1, 2, 3]]),
            self._unread(nested={"z0": 5}),
            self._unread(note="fetched 12.30.4"),
            self._unread(url="https://host/vol.zarr/0/12/30/4"),
            {"boxes": [{"scroll": "PHerc0846A", "metrics": {"seed_zyx_hint": 1}}]},
        ]
        for receipt in bad:
            with self.assertRaises(ValueError, msg=json.dumps(receipt)):
                RGT.validate_public_receipt_privacy(receipt)

    def test_sanitize_drops_non_public_keys(self):
        clean = RGT.sanitize_public_metrics({"sheet_count_rgt": np.int64(3), "origin_zyx": [1, 2, 3],
                                             "usable_cm2_rgt_total": np.float64(0.01), "x": "C:\\a"})
        self.assertEqual(clean, {"sheet_count_rgt": 3, "usable_cm2_rgt_total": 0.01})

    def test_committed_live_receipt(self):
        self.assertTrue(EVIDENCE_PATH.is_file(), "live preflight receipt missing")
        r = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        RGT.validate_public_receipt_privacy(r)
        self.assertEqual(r["schema"], "seismic_rgt_crushed_preflight/v1")
        self.assertTrue(r["cpu_only"])
        self.assertFalse(r["torch_imported"])
        self.assertIn(r["cuda_visible_devices"], ("", "-1"))
        http = r["http"]
        self.assertEqual(http["budget_bytes"], 83_886_080)
        self.assertLessEqual(http["http_bytes"], 83_886_080)
        self.assertFalse(http["persistent_ct_on_disk"])
        roles = {b["role"]: b for b in r["boxes"]}
        self.assertEqual(set(roles), {"public_control", "crushed_eligible_unread"})
        self.assertEqual(roles["public_control"]["scroll"], "PHerc0306B")
        self.assertIn(roles["crushed_eligible_unread"]["scroll"], RGT.UNREAD_SCROLLS)
        for b in r["boxes"]:
            m = b["metrics"]
            self.assertTrue(set(m) <= RGT.PUBLIC_RGT_METRIC_KEYS)
            self.assertEqual(m["box_shape_zyx"], list(RGT.BOX_SHAPE_ZYX))
            self.assertEqual(m["crossing_fraction_rgt"], 0.0)
            self.assertGreaterEqual(m["sheet_count_rgt"], 1)
            self.assertLessEqual(m["http_bytes"], http["http_bytes"])
        self.assertEqual(sum(b["metrics"]["http_bytes"] for b in r["boxes"]), http["http_bytes"])


if __name__ == "__main__":
    unittest.main()
