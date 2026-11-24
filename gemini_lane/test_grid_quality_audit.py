# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for ``grid_quality_audit`` (Exp T15-GridQuality-Audit)."""
from __future__ import annotations

import ast
import json
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid_quality_audit as GQA  # noqa: E402
import leak_guard  # noqa: E402

MODULE_PATH = HERE / "grid_quality_audit.py"
EVIDENCE_PATH = HERE / "evidence" / "grid_quality_audit.json"
GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba", "pycuda")


def plane_grid(rows=40, cols=40, a=(20.0, 0.0, 0.0), b=(0.0, 20.0, 0.0), z0=100.0):
    """Planar grid P[r, c] = origin + c*a + r*b; origin keeps every channel positive."""
    r, c = np.meshgrid(np.arange(rows, dtype=np.float64), np.arange(cols, dtype=np.float64), indexing="ij")
    origin = np.array([500.0, 500.0, z0])
    P = origin + c[..., None] * np.asarray(a) + r[..., None] * np.asarray(b)
    return P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), P[..., 2].astype(np.float32)


def reference_diag(d: str) -> dict:
    """Verbatim formulas of firstletters-colab/methods_ledger/code/grid_quality.py::diag (claude-023)."""
    x, y, z = [tifffile.imread(f"{d}/{c}.tif").astype(np.float64) for c in "xyz"]
    P = np.stack([x, y, z], -1)
    ok = (z > 0) & np.isfinite(z)
    a = P[:-1, 1:] - P[:-1, :-1]
    b = P[1:, :-1] - P[:-1, :-1]
    n = np.cross(a, b)
    m = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1]
    area = np.linalg.norm(n, axis=-1)
    nn = n / np.maximum(area, 1e-9)[..., None]
    nn[~m] = 0
    k = 9
    sm = np.stack([uniform_filter(nn[..., i], k, mode="nearest") for i in range(3)], -1)
    w = uniform_filter(m.astype(float), k, mode="nearest")
    sm = sm / np.maximum(np.linalg.norm(sm, axis=-1), 1e-9)[..., None]
    c = (nn * sm).sum(-1)[m & (w > 0.6)]
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    ca = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
    sh = np.abs(90 - np.degrees(np.arccos(np.clip(ca[m], -1, 1))))
    return dict(n=int(m.sum()), flip=float((c < 0).mean()), a45=float((ang > 45).mean()), a20=float((ang > 20).mean()),
                angp50=float(np.median(ang)), angp95=float(np.percentile(ang, 95)),
                shear50=float(np.median(sh)), shear95=float(np.percentile(sh, 95)))


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = dict(headers or {})
        self.closed = False

    def iter_content(self, chunk_size=1 << 16):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Serves ``routes[url]`` (a response or list of responses, consumed in order); records every URL."""

    def __init__(self, routes, default=None):
        self.routes = {k: (list(v) if isinstance(v, list) else [v]) for k, v in routes.items()}
        self.default = default
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        queue = self.routes.get(url)
        if queue:
            return queue.pop(0) if len(queue) > 1 else queue[0]
        if self.default is not None:
            return self.default()
        return FakeResponse(404)


def tif_bytes(arr) -> bytes:
    import io
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.asarray(arr, dtype=np.float32))
    return buf.getvalue()


class GridQualityMathTests(unittest.TestCase):
    def test_ideal_orthogonal_grid_has_zero_shear_unit_stretch_zero_roughness(self):
        s = GQA.grid_quality(*plane_grid(), meta={"scale": [0.05, 0.05]})
        self.assertEqual(s["shear_1cell"]["abs_p95_deg"], 0.0)
        self.assertEqual(s["shear_8cell"]["abs_p95_deg"], 0.0)
        self.assertAlmostEqual(s["stretch_along_u_columns"]["p50"], 1.0, places=6)
        self.assertAlmostEqual(s["stretch_along_v_rows"]["p50"], 1.0, places=6)
        self.assertAlmostEqual(s["area_inflation_true_over_nominal"], 1.0, places=6)
        self.assertEqual(s["roughness"]["p95_deg"], 0.0)
        self.assertEqual(s["roughness"]["pct_flip_dot_lt_0"], 0.0)

    def test_known_shear_matches_theoretical_angle_and_sign(self):
        theta = math.radians(30.0)       # rows leaned 30 deg: angle(a, b) = 60 deg -> shear = -30 deg
        b = (20.0 * math.sin(theta), 20.0 * math.cos(theta), 0.0)
        s = GQA.grid_quality(*plane_grid(b=b), nominal=(20.0, 20.0))
        for key in ("shear_1cell", "shear_8cell"):
            self.assertAlmostEqual(s[key]["signed_p05_deg"], -30.0, places=3)
            self.assertAlmostEqual(s[key]["signed_p95_deg"], -30.0, places=3)
            self.assertAlmostEqual(s[key]["abs_median_deg"], 30.0, places=3)
            self.assertEqual(s[key]["pct_cells_gt_20deg"], 100.0)
        # leaning the other way flips the sign, not the magnitude
        b_neg = (-b[0], b[1], 0.0)
        s2 = GQA.grid_quality(*plane_grid(b=b_neg), nominal=(20.0, 20.0))
        self.assertAlmostEqual(s2["shear_1cell"]["signed_p05_deg"], 30.0, places=3)
        # sheared parallelogram: area = |a||b| cos(theta) is inflated by cos(30 deg) (< 1 here)
        self.assertAlmostEqual(s["area_inflation_true_over_nominal"], math.cos(theta), places=4)

    def test_known_stretch_factor_and_area_inflation(self):
        s = GQA.grid_quality(*plane_grid(a=(40.0, 0.0, 0.0), b=(0.0, 10.0, 0.0)), nominal=(20.0, 20.0))
        self.assertAlmostEqual(s["stretch_along_u_columns"]["p50"], 2.0, places=6)
        self.assertAlmostEqual(s["stretch_along_v_rows"]["p50"], 0.5, places=6)
        self.assertEqual(s["stretch_along_u_columns"]["pct_gt_1p5x"], 100.0)
        self.assertEqual(s["stretch_along_u_columns"]["pct_off_by_gt_1p5x_either_way"], 100.0)
        self.assertEqual(s["stretch_along_v_rows"]["pct_gt_1p5x"], 0.0)
        self.assertEqual(s["stretch_along_v_rows"]["pct_off_by_gt_1p5x_either_way"], 100.0)
        self.assertAlmostEqual(s["area_inflation_true_over_nominal"], 1.0, places=6)   # 40 * 10 / (20 * 20)
        s3 = GQA.grid_quality(*plane_grid(a=(40.0, 0.0, 0.0)), nominal=(20.0, 20.0))
        self.assertAlmostEqual(s3["area_inflation_true_over_nominal"], 2.0, places=6)

    def test_nominal_step_from_meta_scale(self):
        self.assertEqual(GQA.nominal_step_vx({"scale": [0.05, 0.05]}), (20.0, 20.0))
        self.assertEqual(GQA.nominal_step_vx({"scale": [0.1, 0.05]}), (10.0, 20.0))
        self.assertEqual(GQA.nominal_step_vx({"scale": 0.05}), (20.0, 20.0))
        self.assertEqual(GQA.nominal_step_vx({}), (20.0, 20.0))
        self.assertEqual(GQA.nominal_step_vx({"scale": [0.0, 0.05]}), (20.0, 20.0))
        self.assertEqual(GQA.nominal_step_vx({"scale": [1.0, 1.0]}), (1.0, 1.0))

    def test_flipped_normals_are_detected_by_roughness(self):
        x, y, z = plane_grid(rows=30, cols=40)
        flat = GQA.grid_quality(x, y, z)
        self.assertEqual(flat["roughness"]["pct_flip_dot_lt_0"], 0.0)
        x2 = x.copy()
        x2[:, 21], x2[:, 22] = x[:, 22].copy(), x[:, 21].copy()   # swapped columns: one strip of back-facing cells
        folded = GQA.grid_quality(x2, y, z)
        self.assertGreater(folded["roughness"]["pct_flip_dot_lt_0"], 0.0)
        self.assertGreater(folded["roughness"]["pct_gt_45deg"], 0.0)
        self.assertGreater(folded["roughness"]["pct_flip_dot_lt_0"], 100.0 * 20 / (29 * 39) * 0.5)   # ~1 strip of 29 cells

    def test_matches_claude_reference_formulas_on_noisy_mesh_with_holes(self):
        rng = np.random.default_rng(7)
        x, y, z = plane_grid(rows=60, cols=70)
        x = x + rng.normal(0, 4, x.shape).astype(np.float32)
        y = y + rng.normal(0, 4, y.shape).astype(np.float32)
        z = z + rng.normal(0, 6, z.shape).astype(np.float32)
        hole = np.zeros(x.shape, bool)
        hole[10:20, 15:30] = True
        hole[40:60, 0:8] = True
        for a in (x, y, z):
            a[hole] = -1.0
        with tempfile.TemporaryDirectory() as td:
            for name, arr in zip("xyz", (x, y, z)):
                tifffile.imwrite(Path(td) / f"{name}.tif", arr)
            ref = reference_diag(td)
        cells = GQA.grid_quality_cells(x, y, z, nominal=(20.0, 20.0))
        self.assertEqual(int(cells["stretch_u"].size), ref["n"])
        rough, dot, sh = cells["rough_deg"], cells["rough_dot"], np.abs(cells["shear1"])
        self.assertAlmostEqual(float((dot < 0).mean()), ref["flip"], places=9)
        self.assertAlmostEqual(float((rough > 45).mean()), ref["a45"], places=9)
        self.assertAlmostEqual(float((rough > 20).mean()), ref["a20"], places=9)
        self.assertAlmostEqual(float(np.median(rough)), ref["angp50"], places=9)
        self.assertAlmostEqual(float(np.percentile(rough, 95)), ref["angp95"], places=9)
        self.assertAlmostEqual(float(np.median(sh)), ref["shear50"], places=9)
        self.assertAlmostEqual(float(np.percentile(sh, 95)), ref["shear95"], places=9)

    def test_sentinel_and_degenerate_inputs_do_not_crash(self):
        x = np.full((5, 5), -1.0, np.float32)
        self.assertEqual(GQA.grid_quality(x, x, x)["cells"], 0)
        one = np.ones((1, 7), np.float32)
        self.assertEqual(GQA.grid_quality(one, one, one)["cells"], 0)
        small = plane_grid(rows=5, cols=5)     # smaller than the 8-cell span and the 9x9 window
        s = GQA.grid_quality(*small)
        self.assertIsNone(s["shear_8cell"])
        self.assertEqual(s["shear_1cell"]["abs_p95_deg"], 0.0)

    def test_cartesian_rgt_height_field_has_zero_shear_and_unit_stretch(self):
        import seismic_rgt_unwrapper as RGT
        h = 20.0 + 3.0 * np.sin(np.linspace(0, 4, 50))[None, :] + np.zeros((40, 1))
        x, y, z, v = RGT.height_to_tifxyz(h, np.ones_like(h, bool), 0)
        x, y, z = (np.where(v, a + 1000.0, -1.0) for a in (x, y, z))
        s = GQA.grid_quality(x, y, z, nominal=(1.0, 1.0))
        self.assertLess(s["shear_1cell"]["abs_p95_deg"], 6.0)      # lateral axes stay orthogonal in projection
        self.assertEqual(s["stretch_along_v_rows"]["p50"], 1.0)    # rows run along an unsloped lateral axis
        self.assertGreaterEqual(s["stretch_along_u_columns"]["p05"], 1.0)
        self.assertLess(s["stretch_along_u_columns"]["p95"], 1.4)


class PoolAndSetTests(unittest.TestCase):
    def test_pool_equals_concatenation(self):
        c1 = GQA.grid_quality_cells(*plane_grid(rows=20, cols=20), nominal=(20.0, 20.0))
        c2 = GQA.grid_quality_cells(*plane_grid(rows=20, cols=20, a=(40.0, 0.0, 0.0)), nominal=(20.0, 20.0))
        pool = GQA.Pool()
        pool.add(c1)
        pool.add(c2)
        s = pool.summary()
        self.assertEqual(s["meshes"], 2)
        self.assertEqual(s["cells"], c1["stretch_u"].size + c2["stretch_u"].size)
        self.assertAlmostEqual(s["stretch_along_u_columns"]["pct_gt_1p5x"], 50.0, places=3)
        self.assertAlmostEqual(s["area_inflation_true_over_nominal"], 1.5, places=6)

    def test_pool_rejects_mixed_nominal_cell_areas(self):
        pool = GQA.Pool()
        pool.add(GQA.grid_quality_cells(*plane_grid(), nominal=(20.0, 20.0)))
        with self.assertRaises(ValueError):
            pool.add(GQA.grid_quality_cells(*plane_grid(), nominal=(1.0, 1.0)))

    def test_winding_ids_put_g1_first_and_dedupe(self):
        ids = GQA.winding_ids(3)
        self.assertEqual(ids[:4], list(GQA.G1_WINDINGS))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("w000", ids)
        self.assertEqual(ids.count("w030"), 1)

    def _mesh_routes(self, base, rows=20, cols=20, a=(20.0, 0.0, 0.0), size_pad=0):
        x, y, z = plane_grid(rows=rows, cols=cols, a=a)
        meta = json.dumps({"scale": [0.05, 0.05]}).encode()
        return {
            base + "x.tif": FakeResponse(200, tif_bytes(x)),
            base + "y.tif": FakeResponse(200, tif_bytes(y)),
            base + "z.tif": FakeResponse(200, tif_bytes(z)),
            base + "meta.json": FakeResponse(200, meta),
        }

    def test_run_mesh_set_with_fake_session_and_failure_status(self):
        routes = self._mesh_routes("https://h/a/")
        routes["https://h/b/x.tif"] = FakeResponse(429, b"slow down", {"Retry-After": "1"})
        http = GQA.BudgetedHttp(10 * 1024 * 1024, session=FakeSession(routes), retries=1, sleep=lambda s: None)
        out = GQA.run_mesh_set(http, [("a", "https://h/a/"), ("b", "https://h/b/")], set_id="t", source="fake",
                               row_ids=["a"], g1_ids=["a"])
        self.assertEqual(out["meshes_ok"], 1)
        self.assertEqual(out["meshes_failed"], 1)
        self.assertEqual(out["failures"][0], {"mesh_id": "b", "file": "x.tif", "http_status": 429})
        self.assertEqual(out["per_mesh"][0]["read_by_gemini_g1"], True)
        self.assertEqual(http.status_counts["429"], 2)       # one try + one retry, both recorded
        self.assertEqual(out["pooled"]["stretch_along_u_columns"]["p50"], 1.0)


class HttpBudgetTests(unittest.TestCase):
    def test_budget_is_a_hard_cap_mid_stream(self):
        body = b"x" * 40_000
        session = FakeSession({}, default=lambda: FakeResponse(200, body))
        http = GQA.BudgetedHttp(100_000, session=session, sleep=lambda s: None)
        self.assertEqual(http.get("u1")[0], 200)
        self.assertEqual(http.get("u2")[0], 200)
        with self.assertRaises(GQA.ByteBudgetExceeded):
            http.get("u3")
        # the stream is cut at the first 64 KiB chunk that crosses the cap, so overshoot is bounded by one chunk
        self.assertLessEqual(http.bytes_used, 100_000 + (1 << 16))

    def test_declared_content_length_over_cap_is_refused_before_reading(self):
        resp = FakeResponse(200, b"y" * 10, {"Content-Length": str(10_000_000)})
        http = GQA.BudgetedHttp(1_000_000, session=FakeSession({"u": resp}))
        with self.assertRaises(GQA.ByteBudgetExceeded):
            http.get("u")
        self.assertEqual(http.bytes_used, 0)
        self.assertTrue(resp.closed)

    def test_default_cap_is_60_mib_and_cli_rejects_more(self):
        self.assertEqual(GQA.BUDGET_BYTES, 60 * 1024 * 1024)
        with self.assertRaises(SystemExit), unittest_stderr_muted():
            GQA.main(["--budget-mib", "61", "--skip-live", "--skip-rgt", "--out", os.devnull])

    def test_non_200_statuses_are_counted_and_not_retried_for_404(self):
        http = GQA.BudgetedHttp(1_000_000, session=FakeSession({"u": FakeResponse(404, b"nope")}), sleep=lambda s: None)
        self.assertEqual(http.get("u"), (404, None))
        self.assertEqual(http.status_counts, {"404": 1})
        self.assertEqual(http.requests, 1)

    def test_run_mesh_set_marks_items_skipped_when_budget_runs_out(self):
        routes = {}
        for tag in "abc":
            routes.update(PoolAndSetTests._mesh_routes(self, f"https://h/{tag}/"))
        one_mesh = sum(len(r._body) for k, r in routes.items() if k.startswith("https://h/a/"))
        http = GQA.BudgetedHttp(int(one_mesh * 1.5), session=FakeSession(routes), sleep=lambda s: None)
        out = GQA.run_mesh_set(http, [(t, f"https://h/{t}/") for t in "abc"], set_id="t", source="fake")
        self.assertEqual(out["meshes_ok"], 1)
        self.assertEqual(out["skipped_budget_mesh_ids"], ["b", "c"])
        self.assertEqual(out["meshes_skipped_budget"], 2)
        self.assertLessEqual(http.bytes_used, http.budget_bytes + (1 << 16))


class unittest_stderr_muted:
    def __enter__(self):
        self._old = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, *exc):
        sys.stderr.close()
        sys.stderr = self._old
        return False


class CpuOnlyAndPrivacyTests(unittest.TestCase):
    def test_ast_has_no_gpu_imports(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertFalse(roots & set(GPU_MODULES), roots & set(GPU_MODULES))

    def test_subprocess_import_loads_no_gpu_module(self):
        code = (
            "import sys, os; os.environ['CUDA_VISIBLE_DEVICES']='-1'; sys.path.insert(0, sys.argv[1]); "
            "import grid_quality_audit as g; "
            "bad=[m for m in ('torch','cupy','jax','tensorflow','pycuda') if m in sys.modules]; "
            "print('BAD' if bad else 'OK', bad)"
        )
        res = subprocess.run([sys.executable, "-c", code, str(HERE)], capture_output=True, text=True, timeout=120,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(res.stdout.startswith("OK"), res.stdout)

    def test_assert_public_receipt_rejects_paths_users_and_unread_fields(self):
        GQA.assert_public_receipt({"a": 1, "http": {"x": 2}})
        for bad in ("C:" + chr(92) + "Users" + chr(92) + "someone", "/" + "home/someone/x", "uv_" + "crop", "bbox",
                    "top_" + "hotspots", ".t3/" + "worktrees"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GQA.assert_public_receipt({"note": f"see {bad}"})
        with mock.patch.object(leak_guard, "account_name", return_value="zorbulax"):
            with self.assertRaises(ValueError):
                GQA.assert_public_receipt({"note": "written by Zorbulax"})

    def test_receipt_guard_is_safe_for_a_root_account(self):
        # An account called "root" must not make the receipt guard reject the English word "root".
        self.assertEqual(leak_guard.find_user_leaks('{"note": "repo root and hecate_root"}', "root"), [])
        self.assertTrue(leak_guard.find_local_paths("/" + "root" + "/secret"))

    def test_code_verified_answers_match_the_sources_on_disk(self):
        ans = GQA.code_verified_answers()
        conv = ans["q_which_tifxyz_grid_coordinate_convention_does_stream_snap_render_use"]
        self.assertFalse(conv["render_surface_stack_contains_plus_0p5"])
        self.assertTrue(conv["numerical_probe_stub_reader"]["matches_grid_index_equals_pixel_times_scale"])
        self.assertFalse(conv["numerical_probe_stub_reader"]["matches_grid_index_equals_pixel_plus_half_times_scale"])
        flat = ans["q_per_surface_flattening_vs_raw_fit_grid_vs_spiral_winding_resample"]
        self.assertEqual(flat["modules_importing_spiral_winding_resample"], ["progress_prize_nov2026_verifier"])
        self.assertEqual(flat["modules_mentioning_flatten_spiral_checkpoint"], [])
        self.assertTrue(flat["screen_scroll_rtx4060_snaps_then_renders"])
        g1 = ans["q_which_pherc0826_pherc0191_fits_did_g1_read_and_on_which_grid"]
        self.assertEqual(g1["pherc0826_cpu_triage_mesh_ids"], list(GQA.G1_WINDINGS))
        self.assertEqual(g1["pherc0191_cpu_triage_mesh_ids"], list(GQA.G1_WINDINGS))
        self.assertEqual({m["mesh_id"] for m in g1["pherc0191_ink_detector_reads"]}, {"w030", "w050"})
        self.assertEqual(g1["pherc0826_ink_detector_reads"], [])
        self.assertTrue(g1["pherc0826_triage_no_ink_detector_run"])


class CommittedReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        cls.sets = {s["set_id"]: s for s in cls.receipt["sets"]}

    def test_committed_live_receipt(self):
        r = self.receipt
        self.assertEqual(r["schema"], GQA.SCHEMA)
        self.assertTrue(r["cpu_only"])
        self.assertEqual(r["cuda_visible_devices"], "-1")
        self.assertFalse(r["torch_imported"])
        self.assertFalse(r["http"]["persistent_downloads_on_disk"])
        self.assertLessEqual(r["http"]["http_bytes"], 60 * 1024 * 1024)
        self.assertLessEqual(r["http"]["budget_bytes"], 60 * 1024 * 1024)
        self.assertGreater(r["http"]["requests"], 100)
        self.assertEqual(sum(r["http"]["status_counts"].values()), r["http"]["requests"])
        for name in ("pherc0211_winner_mesh", "pherc0125_eligible_meshes_screened_negative", "pherc0191_hf_spiral_fits",
                     "pherc0826_hf_spiral_fits_slice-6528-7328", "pherc0826_hf_spiral_fits_slice-8928-9728",
                     "pherc0826_hf_spiral_fits_slice-9328-10128", "exp_t11_rgt_stitched_synthetic"):
            self.assertIn(name, self.sets)
        # failures are never silent: every non-200 status must show up in a set's failure list or the status counts
        failed = sum(s.get("meshes_failed", 0) for s in self.receipt["sets"])
        non200 = sum(v for k, v in r["http"]["status_counts"].items() if k != "200")
        if non200 == 0:
            self.assertEqual(failed, 0)
        else:
            self.assertTrue(failed > 0 or any(s.get("skipped_budget_mesh_ids") for s in self.receipt["sets"]))

    def test_committed_numbers_have_the_expected_ordering(self):
        win = self.sets["pherc0211_winner_mesh"]["pooled"]
        self.assertEqual(win["meshes"], 1)
        self.assertLess(win["shear_8cell"]["abs_p95_deg"], 3.0)
        self.assertAlmostEqual(win["stretch_along_u_columns"]["p50"], 1.0, delta=0.02)
        self.assertLess(win["area_inflation_true_over_nominal"] - 1.0, 0.02)
        self.assertLess(win["roughness"]["pct_flip_dot_lt_0"], 0.01)
        for set_id in ("pherc0191_hf_spiral_fits", "pherc0826_hf_spiral_fits_slice-6528-7328",
                       "pherc0826_hf_spiral_fits_slice-8928-9728", "pherc0826_hf_spiral_fits_slice-9328-10128"):
            sp = self.sets[set_id]["pooled"]
            self.assertGreater(sp["shear_8cell"]["abs_p95_deg"], 25.0, set_id)
            self.assertGreater(sp["shear_8cell"]["pct_area_gt_10deg"], 25.0, set_id)
            self.assertGreater(sp["stretch_along_u_columns"]["pct_off_by_gt_1p5x_either_way"], 10.0, set_id)
            self.assertGreater(sp["area_inflation_true_over_nominal"], 1.2, set_id)
            self.assertGreater(sp["roughness"]["p95_deg"], 2 * win["roughness"]["p95_deg"], set_id)
        rgt = self.sets["exp_t11_rgt_stitched_synthetic"]
        self.assertLess(rgt["pooled_after_snap"]["shear_8cell"]["abs_p95_deg"], 1.0)
        self.assertEqual(rgt["pooled_after_snap"]["stretch_along_u_columns"]["pct_off_by_gt_1p5x_either_way"], 0.0)
        self.assertEqual(rgt["crossing_fraction_rgt"], 0.0)

    def test_committed_receipt_is_public_safe_and_aggregate_only(self):
        text = EVIDENCE_PATH.read_text(encoding="utf-8")
        self.assertEqual(leak_guard.find_local_paths(text), [])
        for user in (leak_guard.account_name(), "root"):
            self.assertEqual(leak_guard.find_user_leaks(text, user), [], user)
        for tok in GQA.GENERIC_PRIVACY_TOKENS:
            self.assertNotIn(tok.lower(), text.lower())
        self.assertNotRegex(text, r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}")
        GQA.assert_public_receipt(self.receipt)
        # no raw vertex arrays: every list in per_mesh rows is a short tuple of summary numbers
        for s in self.receipt["sets"]:
            for row in s.get("per_mesh", []):
                for v in row.values():
                    if isinstance(v, list):
                        self.assertLessEqual(len(v), 3)


if __name__ == "__main__":
    unittest.main()
