# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gemini_lane/spiral_winding_resample.py."""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import spiral_winding_resample as swr  # noqa: E402

VOXEL_UM = 7.91
CM2_PER_VX2 = (VOXEL_UM / 1e4) ** 2
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"[A-Za-z]:/"),
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"\\Users\\[^\\]+\\", re.I),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/tmp/"),
    re.compile(r"file://", re.I),
)


# ---------------------------------------------------------------------------
# Synthetic crushed-ellipse spiral winding
# ---------------------------------------------------------------------------

def crushed_spiral_surface(
    t: np.ndarray,
    zi: np.ndarray,
    *,
    n_rows: int,
    a0: float = 500.0,
    b0: float = 100.0,
    phi_total: float = 0.3,
    crush: float = 0.1,
    growth: float = 0.08,
    dz: float = 20.0,
    center: tuple[float, float, float] = (3000.0, 3000.0, 500.0),
) -> np.ndarray:
    """One spiral wrap of a crushed ellipse whose crush axis and aspect vary with z.

    ``t`` is the spiral-fit angle (columns are equal ``t`` steps), ``zi`` the
    fractional row index.  Column spacing swings ~10-60 vx at ``dt = 0.1``.
    """
    s = zi / (n_rows - 1)
    phi = phi_total * s
    a = a0 * (1.0 + crush * np.sin(np.pi * s))
    b = b0 * (1.0 - 0.5 * crush * np.sin(np.pi * s))
    g = 1.0 + growth * t / (2.0 * np.pi)
    tp = t - phi
    ex = a * np.cos(tp) * g
    ey = b * np.sin(tp) * g
    x = center[0] + ex * np.cos(phi) - ey * np.sin(phi)
    y = center[1] + ex * np.sin(phi) + ey * np.cos(phi)
    z = center[2] + zi * dz + 0.0 * t
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def make_winding(n_rows: int = 60, dt: float = 0.1, **kw):
    t = np.arange(0.0, 2.0 * np.pi, dt)
    T, Z = np.meshgrid(t, np.arange(n_rows, dtype=np.float64))
    P = crushed_spiral_surface(T, Z, n_rows=n_rows, **kw)
    return P, np.ones(T.shape, dtype=bool), t


def dense_reference_area_vx2(t: np.ndarray, n_rows: int, factor: int = 10, **kw) -> float:
    td = np.linspace(t[0], t[-1], factor * (t.size - 1) + 1)
    zd = np.linspace(0.0, n_rows - 1.0, factor * (n_rows - 1) + 1)
    T, Z = np.meshgrid(td, zd)
    P = crushed_spiral_surface(T, Z, n_rows=n_rows, **kw)
    return swr.triangulated_area_vx2(P, np.ones(T.shape, dtype=bool))


META = {"format": "tifxyz", "type": "seg", "uuid": "synthetic_winding", "scale": [0.05, 0.05]}


def write_mesh(root: Path, P: np.ndarray, valid: np.ndarray, meta: dict | None = None) -> None:
    swr.write_tifxyz(root, P, valid, dict(meta or META))


# ---------------------------------------------------------------------------
# Exact QA metrics
# ---------------------------------------------------------------------------

class TestExactQA(unittest.TestCase):
    def _plane(self, n=11, step=20.0, shear_deg=0.0):
        i, j = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
        off = math.tan(math.radians(shear_deg)) * step
        P = np.stack([100.0 + j * step + i * off, 100.0 + 0.0 * i, 100.0 + i * step], axis=-1)
        return P, np.ones((n, n), dtype=bool)

    def test_flat_plane_area_steps_and_shear(self):
        P, V = self._plane()
        qa = swr.grid_qa(P, V, target_step_vx=20.0, voxel_um=VOXEL_UM, cell_vx=(20.0, 20.0))
        self.assertAlmostEqual(qa["area"]["triangulated_vx2"], 100 * 400.0, places=6)
        self.assertAlmostEqual(qa["area"]["triangulated_cm2"], 40000.0 * CM2_PER_VX2, places=12)
        self.assertAlmostEqual(qa["area"]["nominal_cm2"], 121 * 400.0 * CM2_PER_VX2, places=12)
        for k in ("du", "dv"):
            self.assertAlmostEqual(qa[k]["p05_vx"], 20.0)
            self.assertAlmostEqual(qa[k]["p95_vx"], 20.0)
            self.assertAlmostEqual(qa[k]["stretch_p50"], 1.0)
            self.assertAlmostEqual(qa[k]["p95_over_p05"], 1.0)
        self.assertAlmostEqual(qa["shear_deg"]["max"], 0.0, places=6)

    def test_sheared_plane_angle(self):
        P, V = self._plane(shear_deg=30.0)
        qa = swr.grid_qa(P, V, target_step_vx=20.0, voxel_um=VOXEL_UM)
        self.assertAlmostEqual(qa["shear_deg"]["p50"], 30.0, places=6)
        self.assertAlmostEqual(qa["shear_deg"]["max"], 30.0, places=6)
        # Shear does not change area of a parallelogram grid with the same base/height.
        self.assertAlmostEqual(qa["area"]["triangulated_vx2"], 100 * 400.0, places=4)

    def test_partial_quads_use_valid_triangles_only(self):
        P, V = self._plane(n=2)
        self.assertAlmostEqual(swr.triangulated_area_vx2(P, V), 400.0)
        for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            V2 = V.copy()
            V2[corner] = False
            self.assertAlmostEqual(swr.triangulated_area_vx2(P, V2), 200.0, msg=str(corner))
        V2 = V.copy()
        V2[0, 0] = V2[1, 1] = False
        self.assertEqual(swr.triangulated_area_vx2(P, V2), 0.0)

    def test_scale_and_voxel_resolution(self):
        self.assertEqual(swr.normalize_scale(0.05), (0.05, 0.05))
        self.assertEqual(swr.normalize_scale([0.1, 0.05]), (0.1, 0.05))
        with self.assertRaises(ValueError):
            swr.normalize_scale([0.1])
        with self.assertRaises(ValueError):
            swr.normalize_scale(-1)
        self.assertEqual(swr.meta_scale({}), ((0.05, 0.05), "default"))
        self.assertEqual(swr.resolve_voxel_um({"voxel_size_um": 3.24}), (3.24, "meta:voxel_size_um"))
        self.assertEqual(swr.resolve_voxel_um({"voxel_size_um": 3.24}, 7.91), (7.91, "cli"))
        self.assertEqual(swr.resolve_voxel_um({})[0], swr.DEFAULT_VOXEL_UM)


class TestSentinelIO(unittest.TestCase):
    def test_read_respects_sentinels_nan_and_zero(self):
        P, V, _ = make_winding(n_rows=6)
        P = P.copy()
        P[1, 2] = -1.0
        P[2, 3, 0] = np.nan
        P[3, 4] = 0.0
        P[4, 5, 2] = np.inf
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mesh"
            root.mkdir()
            import tifffile

            for ci, name in enumerate("xyz"):
                tifffile.imwrite(str(root / f"{name}.tif"), P[..., ci].astype(np.float32))
            (root / "meta.json").write_text(json.dumps({"scale": [0.05, 0.05]}), encoding="utf-8")
            g = swr.read_tifxyz(root)
        bad = [(1, 2), (2, 3), (3, 4), (4, 5)]
        for rc in bad:
            self.assertFalse(g.valid[rc], rc)
            np.testing.assert_array_equal(g.points[rc], [-1.0, -1.0, -1.0])
        self.assertEqual(int((~g.valid).sum()), len(bad))
        self.assertEqual(g.meta["scale"], [0.05, 0.05])

    def test_mask_tif_is_respected(self):
        P, V, _ = make_winding(n_rows=5)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mesh"
            write_mesh(root, P, V)
            m = np.full(V.shape, 255, dtype=np.uint8)
            m[2, 10] = 0
            import tifffile

            tifffile.imwrite(str(root / "mask.tif"), m)
            g = swr.read_tifxyz(root)
        self.assertFalse(g.valid[2, 10])
        self.assertEqual(int((~g.valid).sum()), 1)

    def test_sampler_never_interpolates_across_gaps(self):
        P = np.array([[[10.0, 10, 10], [-1, -1, -1], [30, 10, 10], [40, 10, 10]]])
        V = np.array([[True, False, True, True]])
        q = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]])
        p, ok = swr._sample_axis1(P, V, q, np.ones(q.shape, dtype=bool))
        np.testing.assert_array_equal(ok[0], [True, False, False, False, True, True, True])
        np.testing.assert_allclose(p[0, 5], [35.0, 10, 10])
        np.testing.assert_array_equal(p[0, 1], [-1.0, -1, -1])


# ---------------------------------------------------------------------------
# Resampling on a crushed non-isometric spiral winding
# ---------------------------------------------------------------------------

class TestCrushedSpiralResample(unittest.TestCase):
    N_ROWS = 60

    @classmethod
    def setUpClass(cls):
        cls.P, cls.V, cls.t = make_winding(n_rows=cls.N_ROWS)
        cls.ref_vx2 = dense_reference_area_vx2(cls.t, cls.N_ROWS)
        cls.common = swr.resample_spiral_winding(cls.P, cls.V, META, mode="common_column", voxel_um=VOXEL_UM)
        cls.reg = swr.resample_spiral_winding(
            cls.P, cls.V, META, mode="regularized_arclength", voxel_um=VOXEL_UM, max_shear_deg=15.0
        )

    def test_input_is_non_isometric(self):
        b = self.common.report["before"]
        self.assertLess(b["du"]["p05_vx"], 12.0)
        self.assertGreater(b["du"]["p95_vx"], 50.0)
        self.assertGreater(b["du"]["min_vx"], 9.0)
        self.assertLess(b["du"]["max_vx"], 62.0)
        self.assertGreater(b["du"]["p95_over_p05"], 4.0)
        # Vertex-count area badly under-reports the true area (cf. 1.99 vs 2.80 cm^2).
        self.assertLess(b["area"]["nominal_over_true"], 0.65)
        self.assertLess(abs(b["area"]["triangulated_vx2"] / self.ref_vx2 - 1.0), 0.005)

    def _check_area(self, res):
        r = res.report
        tb = r["before"]["area"]["triangulated_cm2"]
        ta = r["after"]["area"]["triangulated_cm2"]
        self.assertLess(abs(ta / tb - 1.0), 0.02)
        self.assertLess(abs(r["after"]["area"]["triangulated_vx2"] / self.ref_vx2 - 1.0), 0.02)
        self.assertAlmostEqual(r["area_change_frac"], ta / tb - 1.0, places=12)
        # Nominal vertex-count area becomes trustworthy after isometric resampling.
        self.assertLess(abs(r["after"]["area"]["nominal_over_true"] - 1.0), 0.10)
        self.assertAlmostEqual(
            swr.triangulated_area_vx2(res.points, res.valid) * CM2_PER_VX2, ta, places=12
        )

    def test_common_column_area_spacing_and_zero_added_shear(self):
        r = self.common.report
        self._check_area(self.common)
        self.assertLess(abs(r["after"]["du"]["p50_vx"] - 20.0), 0.5)
        self.assertLess(abs(r["after"]["dv"]["p50_vx"] - 20.0), 0.5)
        self.assertLess(r["du_p95_over_p05_after"], 0.5 * r["du_p95_over_p05_before"])
        self.assertLess(r["du_p95_over_p05_after"], 1.6)
        # A common column map keeps the source column geometry: no added shear.
        self.assertLess(r["after"]["shear_deg"]["max"], r["before"]["shear_deg"]["max"] + 1.0)
        self.assertEqual(self.common.meta["scale"], [0.05, 0.05])
        self.assertNotIn("regularization", r)

    def test_regularized_beats_common_with_bounded_shear(self):
        rr, rc = self.reg.report, self.common.report
        self._check_area(self.reg)
        info = rr["regularization"]
        self.assertIsNone(info["fallback_to_common_column"])
        self.assertLessEqual(rr["after"]["shear_deg"]["max"], 15.0 + 1e-9)
        self.assertLess(rr["after"]["du"]["log_err_mean"], 0.75 * rc["after"]["du"]["log_err_mean"])
        self.assertLess(rr["after"]["du"]["rel_err_p95"], rc["after"]["du"]["rel_err_p95"])
        self.assertLess(rr["du_p95_over_p05_after"], rc["du_p95_over_p05_after"])
        self.assertLess(abs(rr["after"]["du"]["p50_vx"] - 20.0), 0.5)
        self.assertGreater(info["lambda"], 0.0)  # the unregularized per-row map violates the cap
        self.assertGreater(info["evaluations"][0]["shear_stat_deg"], 15.0)

    def test_tighter_shear_cap_is_respected(self):
        for cap in (10.0, 12.0):
            res = swr.resample_spiral_winding(
                self.P, self.V, META, mode="regularized_arclength", voxel_um=VOXEL_UM, max_shear_deg=cap
            )
            info = res.report["regularization"]
            bound = max(cap, info["common_column_shear_stat_deg"])
            self.assertLessEqual(res.report["after"]["shear_deg"]["max"], bound + 1e-9, cap)
            self.assertLessEqual(
                res.report["after"]["du"]["log_err_mean"], info["common_column_du_log_err_mean"] + 1e-12
            )

    def test_cap_below_intrinsic_shear_falls_back_to_common(self):
        res = swr.resample_spiral_winding(
            self.P, self.V, META, mode="regularized_arclength", voxel_um=VOXEL_UM, max_shear_deg=1.0
        )
        info = res.report["regularization"]
        self.assertGreater(info["shear_cap_effective_deg"], 1.0)
        self.assertLessEqual(res.report["after"]["shear_deg"]["max"], info["shear_cap_effective_deg"] + 1e-9)

    def test_keep_rows_only_reparameterizes_columns(self):
        res = swr.resample_spiral_winding(
            self.P, self.V, META, mode="common_column", voxel_um=VOXEL_UM, resample_rows=False
        )
        self.assertEqual(res.points.shape[0], self.N_ROWS)
        self.assertEqual(res.meta["scale"], [0.05, 0.05])
        self.assertLess(abs(res.report["area_change_frac"]), 0.02)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            swr.resample_spiral_winding(self.P, self.V, META, mode="naive")
        with self.assertRaises(ValueError):
            swr.resample_spiral_winding(self.P, self.V, META, target_step_vx=0.0)
        with self.assertRaises(ValueError):
            swr.resample_spiral_winding(self.P, self.V, META, max_shear_deg=95.0)


class TestSentinelPreservation(unittest.TestCase):
    N_ROWS = 60
    HOLE_ROWS = (25, 32)  # inclusive
    HOLE_COLS = (20, 28)  # inclusive

    def _holed(self):
        P, V, t = make_winding(n_rows=self.N_ROWS)
        P, V = P.copy(), V.copy()
        r0, r1 = self.HOLE_ROWS
        c0, c1 = self.HOLE_COLS
        V[r0 : r1 + 1, c0 : c1 + 1] = False
        V[5, 40] = False
        V[50, 10] = False
        P[~V] = -1.0
        return P, V, t

    def test_no_output_vertex_inside_sentinel_gap(self):
        P, V, t = self._holed()
        td = np.linspace(t[0], t[-1], 20 * (t.size - 1) + 1)
        zd = np.linspace(0.0, self.N_ROWS - 1.0, 20 * (self.N_ROWS - 1) + 1)
        T, Z = np.meshgrid(td, zd)
        dense = crushed_spiral_surface(T, Z, n_rows=self.N_ROWS).reshape(-1, 3)
        tree = cKDTree(dense)
        r0, r1 = self.HOLE_ROWS
        c0, c1 = self.HOLE_COLS
        # Any valid output must be bracketed by valid sources: never strictly
        # between the last valid source line before the hole and the first after.
        t_lo, t_hi = t[c0 - 1] + 0.01, t[c1 + 1] - 0.01
        z_lo, z_hi = (r0 - 1) + 0.1, (r1 + 1) - 0.1
        for mode in swr.MODES:
            with self.subTest(mode=mode):
                res = swr.resample_spiral_winding(P, V, META, mode=mode, voxel_um=VOXEL_UM)
                pts = res.points[res.valid]
                self.assertTrue(np.all(pts > 0))
                np.testing.assert_array_equal(res.points[~res.valid], -1.0)
                dist, idx = tree.query(pts)
                self.assertLess(float(dist.max()), 3.0)  # outputs lie on the source surface
                tt, zz = T.reshape(-1)[idx], Z.reshape(-1)[idx]
                inside = (tt > t_lo) & (tt < t_hi) & (zz > z_lo) & (zz < z_hi)
                self.assertEqual(int(inside.sum()), 0)
                self.assertGreater(int((~res.valid).sum()), 0)
                ta = res.report["after"]["area"]["triangulated_vx2"]
                tb = res.report["before"]["area"]["triangulated_vx2"]
                self.assertLess(ta, tb * 1.01)
                self.assertGreater(ta, tb * 0.95)

    def test_written_sentinels_round_trip(self):
        P, V, _ = self._holed()
        res = swr.resample_spiral_winding(P, V, META, mode="regularized_arclength", voxel_um=VOXEL_UM)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            swr.write_tifxyz(out, res.points, res.valid, res.meta)
            g = swr.read_tifxyz(out)
        np.testing.assert_array_equal(g.valid, res.valid)
        np.testing.assert_allclose(g.points[g.valid], res.points[res.valid], rtol=0, atol=1e-3)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        P, V, _ = make_winding(n_rows=40)
        self.mesh = self.tmp / "winding_in"
        write_mesh(self.mesh, P, V, dict(META, voxel_size_um=VOXEL_UM))

    def tearDown(self):
        self._td.cleanup()

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = swr.main(argv)
        return rc, json.loads(buf.getvalue())

    def test_cli_round_trip(self):
        out = self.tmp / "winding_out"
        js = self.tmp / "qa" / "report.json"
        for mode in swr.MODES:
            with self.subTest(mode=mode):
                rc, brief = self._run([
                    "--mesh", str(self.mesh), "--out-dir", str(out), "--mode", mode,
                    "--target-step-vx", "20", "--max-shear-deg", "15", "--json-out", str(js),
                ])
                self.assertEqual(rc, 0)
                for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
                    self.assertTrue((out / f).is_file(), f)
                meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["scale"], [0.05, 0.05])
                self.assertEqual(meta["uuid"], "winding_out")
                self.assertEqual(meta["spiral_winding_resample"]["mode"], mode)
                self.assertEqual(len(meta["bbox"]), 2)
                text = js.read_text(encoding="utf-8")
                report = json.loads(text)
                for pat in FORBIDDEN_PATH_PATTERNS:
                    self.assertIsNone(pat.search(text), pat.pattern)
                    self.assertIsNone(pat.search(json.dumps(meta)), pat.pattern)
                self.assertEqual(report["mesh"], "winding_in")
                self.assertEqual(report["params"]["voxel_um_source"], "meta:voxel_size_um")
                g = swr.read_tifxyz(out)
                qa = swr.grid_qa(g.points, g.valid, voxel_um=VOXEL_UM, cell_vx=(20.0, 20.0))
                self.assertAlmostEqual(
                    qa["area"]["triangulated_cm2"], report["after"]["area"]["triangulated_cm2"], places=6
                )
                self.assertLess(abs(brief["area_change_frac"]), 0.02)
                self.assertLess(brief["du_p95_over_p05"][1], brief["du_p95_over_p05"][0])
                if mode == "regularized_arclength":
                    self.assertLessEqual(report["after"]["shear_deg"]["max"], 15.0 + 1e-9)

    def test_dry_run_writes_no_tifxyz(self):
        out = self.tmp / "never"
        js = self.tmp / "dry.json"
        rc, brief = self._run([
            "--mesh", str(self.mesh), "--out-dir", str(out), "--dry-run", "--json-out", str(js),
            "--voxel-um", "3.24", "--mode", "common_column",
        ])
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists())
        report = json.loads(js.read_text(encoding="utf-8"))
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["params"]["voxel_um"], 3.24)
        self.assertEqual(brief["mode"], "common_column")

    def test_missing_out_dir_without_dry_run(self):
        import contextlib

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(swr.main(["--mesh", str(self.mesh)]), 2)

    def test_subprocess_cpu_only(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        proc = subprocess.run(
            [sys.executable, str(HERE / "spiral_winding_resample.py"), "--mesh", str(self.mesh), "--dry-run"],
            capture_output=True, text=True, env=env, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        brief = json.loads(proc.stdout)
        self.assertEqual(brief["mode"], "regularized_arclength")
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); import spiral_winding_resample; "
             "print(sorted(m for m in ('torch', 'cupy', 'jax') if m in sys.modules))", str(HERE)],
            capture_output=True, text=True, env=env, timeout=120,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout.strip(), "[]")  # no GPU framework is ever imported


if __name__ == "__main__":
    unittest.main()
