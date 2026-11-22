# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline JSON/schema tests for ``colab_t4_firstletters_screener.ipynb`` (no network, no GPU)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import screen_scroll_rtx4060 as S  # noqa: E402

NOTEBOOK = HERE / "colab_t4_firstletters_screener.ipynb"
EVIDENCE = HERE / "evidence" / "pherc1203_onsheet_triage.json"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cells(nb: dict, cell_id: str) -> dict:
    return next(c for c in nb["cells"] if c["id"] == cell_id)


def _helper_namespace() -> dict:
    ns: dict = {"__name__": "colab_helpers"}
    exec(compile(_source(_cells(_load(), "helpers")), "helpers", "exec"), ns)
    return ns


class NotebookSchemaTests(unittest.TestCase):
    def test_nbformat_and_clean_cells(self):
        nb = _load()
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(nb["nbformat_minor"], 5)
        self.assertEqual(nb["metadata"]["accelerator"], "GPU")
        self.assertEqual(nb["metadata"]["colab"]["gpuType"], "T4")
        ids = [c["id"] for c in nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in nb["cells"]:
            self.assertIn(cell["cell_type"], {"markdown", "code"})
            self.assertIsInstance(cell["source"], list)
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [], cell["id"])
                self.assertIsNone(cell["execution_count"], cell["id"])

    def test_code_cells_are_valid_python(self):
        for cell in _load()["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(_source(cell), filename=cell["id"])

    def test_no_local_paths_or_private_markers(self):
        raw = NOTEBOOK.read_text(encoding="utf-8")
        for pattern in (r"(?<![A-Za-z])[A-Za-z]:\\{1,2}", r"C:/Users", r"/Users/", r"/home/", r"joris\b", "Documents"):
            self.assertIsNone(re.search(pattern, raw), pattern)

    def test_setup_clones_branch_and_installs_light_deps(self):
        setup = _source(_cells(_load(), "setup"))
        self.assertIn("JorisViaudQuantAI/Scrollprize", setup)
        self.assertIn('BRANCH = "gemini/coverage-gpu-lane"', setup)
        for dep in ("numcodecs", "safetensors", "tifffile", "opencv-python-headless", "requests"):
            self.assertIn(f'"{dep}"', setup)
        self.assertNotIn("torch", setup)

    def test_scenario_cells_present(self):
        nb = _load()
        gate = _source(_cells(nb, "gate"))
        self.assertIn("PHerc0306B", gate)
        self.assertIn("GATE_UV_CROP", gate)
        self.assertIn("snap_passes=0", gate)
        tier1 = _source(_cells(nb, "tier1"))
        self.assertIn("PHERC0125_TIER1", tier1)
        self.assertIn("snap_passes=2", tier1)
        self.assertIn("require_gate(GATE_REVIEWED)", tier1)
        p1203 = _source(_cells(nb, "pherc1203"))
        self.assertIn("PHERC1203_CLEAN", p1203)
        self.assertIn("snap_passes=2", p1203)
        self.assertIn("require_gate(GATE_REVIEWED)", p1203)
        burst = _source(_cells(nb, "burst"))
        self.assertIn("run_burst()", burst)
        self.assertIn("verify_upstream_gpu_burst.py", _source(_cells(nb, "helpers")))
        self.assertIn("GATE_REVIEWED = False", _source(_cells(nb, "gate-flag")))


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _helper_namespace()

    def test_gate_command(self):
        ns = self.ns
        cmd = ns["build_screen_cmd"]("PHerc0306B", [ns["GATE_MESH"]], snap_passes=0, uv_crop=ns["GATE_UV_CROP"])
        self.assertEqual(ns["GATE_UV_CROP"], "80:140,120:345")
        self.assertEqual(cmd[cmd.index("--uv-crop") + 1], "80:140,120:345")
        self.assertEqual(cmd[cmd.index("--detectors") + 1], "studio,ahn")
        self.assertEqual(cmd[cmd.index("--studio-stride") + 1], "48")
        self.assertEqual(cmd[cmd.index("--voxel-um") + 1], "8.64")
        self.assertEqual(S.parse_uv_crop(cmd[cmd.index("--uv-crop") + 1]), (80, 140, 120, 345))

    def test_command_parses_with_real_screener_cli(self):
        ns = self.ns
        meshes = [f"{ns['PHERC1203_BASE']}/{m}" for m in ns["PHERC1203_CLEAN"]]
        cmd = ns["build_screen_cmd"]("PHerc1203", meshes, snap_passes=2)
        args = S.build_arg_parser().parse_args(cmd[2:])
        self.assertEqual(args.scroll, "PHerc1203")
        self.assertEqual(args.snap_passes, 2)
        self.assertEqual(args.mesh, meshes)
        self.assertEqual(args.out_private_dir, Path("/content/scrollprize_private"))
        self.assertLessEqual(args.vram_fraction, 0.9)
        self.assertFalse(args.uv_crop)

    def test_pherc0125_tier1_excludes_codex_mesh(self):
        ns = self.ns
        self.assertEqual(
            ns["PHERC0125_TIER1"],
            ("z10544_w035", "z14144_w035", "z8736_w035", "z6944_w030", "z15936_w030"),
        )
        self.assertNotIn("z14144_w045", ns["PHERC0125_TIER1"])
        self.assertIn("z14144_w045", ns["PHERC0125_EXCLUDED"])
        self.assertRegex(ns["PSCAMILLO_COMMIT"], r"^[0-9a-f]{40}$")

    def test_pherc1203_clean_sheets(self):
        ns = self.ns
        self.assertEqual(len(ns["PHERC1203_CLEAN"]), 6)
        self.assertEqual(len(set(ns["PHERC1203_CLEAN"])), 6)
        self.assertEqual(ns["PHERC1203_CLEAN"][0], "auto_grown_20251005231446965")
        self.assertIn("auto_grown_20251005231446963", ns["PHERC1203_CLEAN"])
        self.assertNotIn("auto_grown_20250923163217042", ns["PHERC1203_CLEAN"])

    def test_scroll_urls_are_zarr_v2_roots(self):
        for name, cfg in self.ns["SCROLLS"].items():
            for key in ("ct", "m7"):
                self.assertTrue(cfg[key].startswith("https://vesuvius-challenge-open-data.s3"), name)
                self.assertTrue(cfg[key].endswith(".zarr"), name)
                self.assertNotIn("vcd3d", cfg[key], name)
            self.assertGreater(cfg["voxel_um"], 0)

    def test_zip_only_contains_allowlisted_files(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            sheet = out / "PHerc1203_x"
            sheet.mkdir(parents=True)
            for name in ("montage.png", "hotspots.png", "summary.json", "stack.npy", "map.png", "notes.txt"):
                (sheet / name).write_bytes(b"x")
            (out / "PHerc1203_batch_summary.json").write_text("{}", encoding="utf-8")
            zip_path, members = ns["package_private_outputs"](out, Path(tmp) / "z.zip")
            with zipfile.ZipFile(zip_path) as zf:
                names = sorted(zf.namelist())
            self.assertEqual(names, sorted(members))
            self.assertEqual(
                names,
                ["PHerc1203_x/hotspots.png", "PHerc1203_x/montage.png", "PHerc1203_x/summary.json"],
            )

    def test_require_gate_blocks_without_summary_or_review(self):
        ns = self.ns
        with tempfile.TemporaryDirectory() as tmp:
            ns["OUT_DIR"] = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                ns["require_gate"](True)
            gate_dir = Path(tmp) / "PHerc0306B_mesh_r80-140_c120-345"
            gate_dir.mkdir()
            stats = {"p99": 0.1, "p995": 0.2, "frac_gt_05": 0.0, "valid_cm2": 1.0}
            (gate_dir / "summary.json").write_text(
                json.dumps({"detectors": {k: stats for k in ns["GATE_DETECTOR_KEYS"]}}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                ns["require_gate"](False)
            ns["require_gate"](True)

    def test_outputs_dir_is_outside_clone(self):
        ns = self.ns
        with self.assertRaises(ValueError):
            Path(ns["OUT_DIR"]).relative_to(ns["REPO_DIR"])


class PublicEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(EVIDENCE.is_file(), "triage evidence not generated yet")
    def test_public_triage_json_is_sanitized(self):
        raw = EVIDENCE.read_text(encoding="utf-8")
        data = json.loads(raw)
        for pattern in (r"(?<![A-Za-z])[A-Za-z]:\\{1,2}", r"C:/Users", r"/Users/", r"joris", "Documents", r"\.png"):
            self.assertIsNone(re.search(pattern, raw), pattern)
        self.assertEqual(data["scroll"], "PHerc1203")
        self.assertEqual(len(data["sheets"]), 6)
        allowed = {
            "rank", "mesh_id", "uv_crop", "valid_vertex_fraction", "valid_vertex_count",
            "total_vertex_count", "on_sheet_raw", "on_sheet_after_snap", "median_disp_vx",
            "p90_disp_vx", "center_vs_outer_disp_diff", "onsheet_score", "valid_ct_cm2",
        }
        for sheet in data["sheets"]:
            self.assertEqual(set(sheet), allowed)
        scores = [s["onsheet_score"]["f_parallel"] for s in data["sheets"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertNotIn("detectors", data)
        self.assertNotIn("top_hotspots", data)


if __name__ == "__main__":
    unittest.main()
