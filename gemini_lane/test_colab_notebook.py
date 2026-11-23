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
from progress_prize_fp16 import prepare_hecate_parity_crops as P  # noqa: E402
from progress_prize_fp16 import verify_upstream_gpu_burst as B  # noqa: E402

NOTEBOOK = HERE / "colab_t4_firstletters_screener.ipynb"
EVIDENCE = HERE / "evidence" / "pherc1203_onsheet_triage.json"
EVIDENCE_T1B_SEP23 = HERE / "evidence" / "pherc0125_t1b_1203sep23_onsheet_triage.json"
EVIDENCE_0826_T2 = HERE / "evidence" / "pherc0826_1203t2_onsheet_triage.json"

UNREAD_SCREEN_CELL_IDS = (
    "tier1",
    "pherc1203",
    "pherc0191",
    "pherc0125-tier1b",
    "pherc1203-sep23",
    "pherc0826",
    "pherc1203-tier2",
)


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


def _calls_in_tree(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _is_require_gate_reviewed_stmt(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    call = stmt.value
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "require_gate"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "GATE_REVIEWED"
        and not call.keywords
    )


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

    def test_no_local_paths_secrets_or_private_markers(self):
        raw = NOTEBOOK.read_text(encoding="utf-8")
        forbidden = (
            r"(?<![A-Za-z])[A-Za-z]:\\{1,2}",
            r"C:/Users",
            r"/Users/",
            r"/home/",
            r"joris\b",
            "Documents",
            r"AppData",
            r"\.t3[\\/]worktrees",
            r"\b(?:gho|ghp|ghu|ghs|ghr)_[A-Za-z0-9]{10,}\b",
            r"\bhf_[A-Za-z0-9]{20,}\b",
            r"\bsk-[A-Za-z0-9_-]{20,}\b",
            r"\bAKIA[0-9A-Z]{16}\b",
        )
        for pattern in forbidden:
            self.assertIsNone(re.search(pattern, raw), pattern)

    def test_setup_clones_branch_and_installs_light_deps(self):
        setup = _source(_cells(_load(), "setup"))
        self.assertIn("JorisViaudQuantAI/Scrollprize", setup)
        self.assertIn('BRANCH = "gemini/coverage-gpu-lane"', setup)
        for dep in ("numcodecs", "safetensors", "tifffile", "opencv-python-headless", "requests", "zarr"):
            self.assertIn(f'"{dep}"', setup)
        self.assertNotIn("torch", setup)

    def test_gate_reviewed_lock_enforced_on_all_unread_screening_cells(self):
        nb = _load()
        self.assertIn("GATE_REVIEWED = False", _source(_cells(nb, "gate-flag")))

        discovered_unread: list[str] = []
        for cell in nb["cells"]:
            if cell["cell_type"] != "code" or cell["id"] in {"setup", "helpers", "gate", "gate-flag"}:
                continue
            tree = ast.parse(_source(cell), filename=cell["id"])
            if "build_screen_cmd" in _calls_in_tree(tree):
                discovered_unread.append(cell["id"])
                self.assertTrue(
                    tree.body and _is_require_gate_reviewed_stmt(tree.body[0]),
                    f"Unread screening cell {cell['id']!r} must start with require_gate(GATE_REVIEWED)",
                )

        self.assertEqual(tuple(discovered_unread), UNREAD_SCREEN_CELL_IDS)

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

        p0191 = _source(_cells(nb, "pherc0191"))
        self.assertIn("PHERC0191_WINDINGS", p0191)
        self.assertIn("PHERC0191_BASE", p0191)
        self.assertIn("snap_passes=2", p0191)
        self.assertIn("require_gate(GATE_REVIEWED)", p0191)

        burst = _source(_cells(nb, "burst"))
        self.assertIn("run_burst()", burst)
        burst_md = _source(_cells(nb, "burst-md"))
        self.assertIn("prepare_hecate_parity_crops.py", burst_md)
        self.assertIn("--stages studio,hecate", burst_md)
        helpers_src = _source(_cells(nb, "helpers"))
        self.assertIn("prepare_hecate_parity_crops.py", helpers_src)
        self.assertIn("verify_upstream_gpu_burst.py", helpers_src)

        t1b = _source(_cells(nb, "pherc0125-tier1b"))
        self.assertIn("PHERC0125_TIER1B", t1b)
        self.assertIn("snap_passes=2", t1b)
        self.assertIn("require_gate(GATE_REVIEWED)", t1b)

        sep23 = _source(_cells(nb, "pherc1203-sep23"))
        self.assertIn("PHERC1203_SEP23_CLEAN", sep23)
        self.assertIn("snap_passes=2", sep23)
        self.assertIn("require_gate(GATE_REVIEWED)", sep23)

        p0826 = _source(_cells(nb, "pherc0826"))
        self.assertIn("PHERC0826_WINDINGS", p0826)
        self.assertIn("PHERC0826_BASE", p0826)
        self.assertIn("snap_passes=2", p0826)
        self.assertIn("require_gate(GATE_REVIEWED)", p0826)

        t2 = _source(_cells(nb, "pherc1203-tier2"))
        self.assertIn("PHERC1203_TIER2_STRIPS", t2)
        self.assertIn("uv_crop=uv_crop", t2)
        self.assertIn("snap_passes=2", t2)
        self.assertIn("require_gate(GATE_REVIEWED)", t2)

        ids = [c["id"] for c in nb["cells"]]
        for cid in ("pherc0191", "burst", "pherc0125-tier1b", "pherc1203-sep23", "pherc0826", "pherc1203-tier2"):
            self.assertEqual(ids.index(cid), ids.index(f"{cid}-md") + 1, cid)
        self.assertLess(ids.index("pherc1203"), ids.index("pherc0191"))
        self.assertLess(ids.index("pherc0191"), ids.index("burst"))
        self.assertLess(ids.index("burst"), ids.index("pherc0125-tier1b"))
        self.assertLess(ids.index("pherc0125-tier1b"), ids.index("pherc1203-sep23"))
        self.assertLess(ids.index("pherc1203-sep23"), ids.index("pherc0826"))
        self.assertLess(ids.index("pherc0826"), ids.index("pherc1203-tier2"))
        self.assertLess(ids.index("pherc1203-tier2"), ids.index("export"))


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

    def test_pherc0191_command_parses_with_real_screener_cli(self):
        ns = self.ns
        meshes = [f"{ns['PHERC0191_BASE']}/{m}" for m in ns["PHERC0191_WINDINGS"]]
        cmd = ns["build_screen_cmd"]("PHerc0191", meshes, snap_passes=2)
        args = S.build_arg_parser().parse_args(cmd[2:])
        self.assertEqual(args.scroll, "PHerc0191")
        self.assertEqual(args.snap_passes, 2)
        self.assertEqual(args.mesh, meshes)
        self.assertEqual(args.voxel_um, 9.362)
        self.assertIn("/PHerc0191/volumes/20250821151635-9.362um-1.2m-113keV-masked.zarr", args.ct_zarr)
        self.assertIn("/PHerc0191/representations/predictions/surfaces/", args.m7_zarr)
        self.assertTrue(args.m7_zarr.endswith("surface-m7-L0-th0.2.zarr"))
        self.assertEqual(args.out_private_dir, Path("/content/scrollprize_private"))
        self.assertFalse(args.uv_crop)

    def test_pherc0191_windings_and_pinned_hf_mesh_source(self):
        ns = self.ns
        self.assertEqual(ns["PHERC0191_WINDINGS"], ("w020", "w030", "w040", "w050"))
        self.assertRegex(ns["HF_SPIRAL_REVISION"], r"^[0-9a-f]{40}$")
        base = ns["PHERC0191_BASE"]
        self.assertTrue(base.startswith("https://huggingface.co/datasets/rodriguescarson/eligible-scroll-spiral-fits/resolve/"))
        self.assertIn(ns["HF_SPIRAL_REVISION"], base)
        self.assertIn("2026-09-15_PHerc0191_slice-11600-12400_0-patch_w90/meshes/fitted", base)
        self.assertNotIn("/main/", base)
        self.assertEqual(ns["SCROLLS"]["PHerc0191"]["voxel_um"], 9.362)

    def test_pherc0125_tier1b_and_sep23_commands_parse_with_real_screener_cli(self):
        ns = self.ns
        expected_t1b = (
            "z9936_w035",
            "z9344_w015",
            "z11136_w035",
            "z15936_w035",
            "z8144_w030",
            "z14144_w050",
        )
        self.assertEqual(ns["PHERC0125_TIER1B"], expected_t1b)
        self.assertFalse(set(ns["PHERC0125_TIER1B"]) & set(ns["PHERC0125_TIER1"]))
        self.assertFalse(set(ns["PHERC0125_TIER1B"]) & set(ns["PHERC0125_EXCLUDED"]))

        t1b_meshes = [f"{ns['PSCAMILLO_BASE']}/{m}" for m in ns["PHERC0125_TIER1B"]]
        t1b_cmd = ns["build_screen_cmd"]("PHerc0125", t1b_meshes, snap_passes=2)
        t1b_args = S.build_arg_parser().parse_args(t1b_cmd[2:])
        self.assertEqual(t1b_args.scroll, "PHerc0125")
        self.assertEqual(t1b_args.snap_passes, 2)
        self.assertEqual(t1b_args.mesh, t1b_meshes)
        self.assertEqual(t1b_args.voxel_um, 9.362)
        self.assertEqual(t1b_args.out_private_dir, Path("/content/scrollprize_private"))
        self.assertFalse(t1b_args.uv_crop)

        self.assertEqual(ns["PHERC1203_SEP23_CLEAN"], ("auto_grown_20250923163217042",))
        self.assertFalse(set(ns["PHERC1203_SEP23_CLEAN"]) & set(ns["PHERC1203_CLEAN"]))
        sep23_meshes = [f"{ns['PHERC1203_BASE']}/{m}" for m in ns["PHERC1203_SEP23_CLEAN"]]
        sep23_cmd = ns["build_screen_cmd"]("PHerc1203", sep23_meshes, snap_passes=2)
        sep23_args = S.build_arg_parser().parse_args(sep23_cmd[2:])
        self.assertEqual(sep23_args.scroll, "PHerc1203")
        self.assertEqual(sep23_args.snap_passes, 2)
        self.assertEqual(sep23_args.mesh, sep23_meshes)
        self.assertEqual(sep23_args.voxel_um, 9.362)
        self.assertEqual(sep23_args.out_private_dir, Path("/content/scrollprize_private"))
        self.assertFalse(sep23_args.uv_crop)

    def test_pherc0826_and_pherc1203_tier2_commands_parse_with_real_screener_cli(self):
        ns = self.ns
        self.assertEqual(ns["PHERC0826_WINDINGS"], ("w030", "w050", "w040"))
        self.assertEqual(set(ns["PHERC0826_EXCLUDED"]), {"w020", "w027", "w051"})
        self.assertFalse(set(ns["PHERC0826_WINDINGS"]) & set(ns["PHERC0826_EXCLUDED"]))
        base = ns["PHERC0826_BASE"]
        self.assertTrue(base.startswith("https://huggingface.co/datasets/rodriguescarson/eligible-scroll-spiral-fits/resolve/"))
        self.assertIn(ns["HF_SPIRAL_REVISION"], base)
        self.assertIn("2026-09-15_PHerc0826_slice-8928-9728_0-patch/meshes/fitted", base)
        self.assertNotIn("/main/", base)

        meshes_0826 = [f"{base}/{m}" for m in ns["PHERC0826_WINDINGS"]]
        cmd_0826 = ns["build_screen_cmd"]("PHerc0826", meshes_0826, snap_passes=2)
        args_0826 = S.build_arg_parser().parse_args(cmd_0826[2:])
        self.assertEqual(args_0826.scroll, "PHerc0826")
        self.assertEqual(args_0826.snap_passes, 2)
        self.assertEqual(args_0826.mesh, meshes_0826)
        self.assertEqual(args_0826.voxel_um, 9.362)
        self.assertIn("/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr", args_0826.ct_zarr)
        self.assertIn("/PHerc0826/representations/predictions/surfaces/", args_0826.m7_zarr)
        self.assertTrue(args_0826.m7_zarr.endswith("surface-m7-L0-th0.2.zarr"))
        self.assertEqual(args_0826.out_private_dir, Path("/content/scrollprize_private"))
        self.assertFalse(args_0826.uv_crop)

        expected_strips = (
            ("auto_grown_20251005230830030", "0:165,0:68"),
            ("auto_grown_20251005221856743", "0:253,158:226"),
            ("auto_grown_20251005221856743", "0:253,0:62"),
        )
        self.assertEqual(ns["PHERC1203_TIER2_STRIPS"], expected_strips)
        for name, uv_crop in ns["PHERC1203_TIER2_STRIPS"]:
            cmd = ns["build_screen_cmd"]("PHerc1203", [f"{ns['PHERC1203_BASE']}/{name}"], snap_passes=2, uv_crop=uv_crop)
            args = S.build_arg_parser().parse_args(cmd[2:])
            self.assertEqual(args.scroll, "PHerc1203")
            self.assertEqual(args.snap_passes, 2)
            self.assertEqual(args.mesh, [f"{ns['PHERC1203_BASE']}/{name}"])
            self.assertEqual(args.uv_crop, uv_crop)
            self.assertIsNotNone(S.parse_uv_crop(args.uv_crop))
            self.assertEqual(args.out_private_dir, Path("/content/scrollprize_private"))

    def test_prepare_hecate_crops_command_parses_with_real_cli(self):
        ns = self.ns
        self.assertTrue((HERE.parent / ns["PREP_HECATE_SCRIPT"]).is_file())
        self.assertEqual(ns["HECATE_CROPS_DIR"], Path("/content/scrollprize_private/hecate_parity_crops"))
        self.assertEqual(ns["HECATE_CROP_2P4"], P.CROP_2P4_NAME)
        self.assertEqual(ns["HECATE_CROP_9P6"], P.CROP_9P6_NAME)

        # Default Colab path parses and validates outside git.
        default_cmd = ns["build_prepare_hecate_crops_cmd"]()
        self.assertEqual(Path(default_cmd[1]), ns["PREP_HECATE_SCRIPT"])
        default_args = P.build_parser().parse_args(default_cmd[2:])
        self.assertEqual(default_args.out_dir, Path("/content/scrollprize_private/hecate_parity_crops"))
        self.assertEqual(P.validate_args(default_args), [])

        with tempfile.TemporaryDirectory() as tmp:
            crops_dir = Path(tmp) / "hecate_parity_crops"
            cmd = ns["build_prepare_hecate_crops_cmd"](crops_dir)
            args = P.build_parser().parse_args(cmd[2:])
            self.assertEqual(args.out_dir, crops_dir)
            self.assertEqual(P.validate_args(args), [])

    def test_burst_command_parses_with_real_burst_cli_studio_and_hecate(self):
        ns = self.ns
        self.assertTrue((HERE.parent / ns["BURST_SCRIPT"]).is_file())
        with tempfile.TemporaryDirectory() as tmp:
            crops_dir = Path(tmp) / "hecate_parity_crops"
            crops_dir.mkdir()
            crop_2p4 = crops_dir / P.CROP_2P4_NAME
            crop_9p6 = crops_dir / P.CROP_9P6_NAME
            crop_2p4.write_bytes(b"dummy")
            crop_9p6.write_bytes(b"dummy")

            cmd = ns["build_burst_cmd"](Path(tmp) / "receipt.json", crops_dir=crops_dir)
            self.assertEqual(Path(cmd[1]), ns["BURST_SCRIPT"])
            args = B.build_parser().parse_args(cmd[2:])
            self.assertEqual(B.validate(args), [])
            planned = B.plan(args)

        self.assertEqual(args.stages, "studio,hecate")
        self.assertEqual(args.hecate_patch, "main")
        self.assertEqual(args.hecate_input_2p4, crop_2p4)
        self.assertEqual(args.hecate_input_9p6, crop_9p6)
        self.assertFalse(args.dry_run)
        self.assertLessEqual(args.vram_fraction, 0.35)
        self.assertEqual(args.batch, 1)
        self.assertEqual(args.studio_stride, 48)
        self.assertEqual(len([r for r in planned if r["script"] == "infer_release.py"]), 4)
        self.assertEqual(len([r for r in planned if r["script"] == "hecate.py"]), 8)

    def test_run_burst_orchestrates_prepare_then_burst_in_private_dir(self):
        ns = _helper_namespace()
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ns["REPO_DIR"] = HERE.parent
            ns["OUT_DIR"] = tmp_path / "private"
            crops_dir = ns["OUT_DIR"] / "hecate_parity_crops"

            def fake_run_streaming(cmd):
                commands.append(list(cmd))
                if Path(cmd[1]) == ns["PREP_HECATE_SCRIPT"]:
                    out = Path(cmd[cmd.index("--out-dir") + 1])
                    out.mkdir(parents=True, exist_ok=True)
                    (out / P.CROP_2P4_NAME).write_bytes(b"2p4")
                    (out / P.CROP_9P6_NAME).write_bytes(b"9p6")
                elif Path(cmd[1]) == ns["BURST_SCRIPT"]:
                    args = B.build_parser().parse_args(cmd[2:])
                    self.assertEqual(B.validate(args), [])
                    args.output.write_text('{"pass": true}\n', encoding="utf-8")

            ns["run_streaming"] = fake_run_streaming
            receipt = ns["run_burst"](crops_dir=crops_dir)
            self.assertIsNotNone(receipt)
            self.assertTrue(receipt.is_file())
            self.assertEqual(receipt.parent, ns["OUT_DIR"])
            self.assertEqual(len(commands), 2)
            self.assertEqual(Path(commands[0][1]), ns["PREP_HECATE_SCRIPT"])
            self.assertEqual(Path(commands[1][1]), ns["BURST_SCRIPT"])
            self.assertEqual(commands[1][commands[1].index("--stages") + 1], "studio,hecate")
            self.assertEqual(
                Path(commands[1][commands[1].index("--hecate-input-2p4") + 1]),
                crops_dir / P.CROP_2P4_NAME,
            )
            self.assertEqual(
                Path(commands[1][commands[1].index("--hecate-input-9p6") + 1]),
                crops_dir / P.CROP_9P6_NAME,
            )

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
            crops = out / "hecate_parity_crops"
            crops.mkdir()
            (crops / P.CROP_2P4_NAME).write_bytes(b"x")
            (crops / P.CROP_9P6_NAME).write_bytes(b"x")
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
        ns = _helper_namespace()
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

    def test_outputs_and_hecate_crops_dir_are_outside_clone(self):
        ns = self.ns
        with self.assertRaises(ValueError):
            Path(ns["OUT_DIR"]).relative_to(ns["REPO_DIR"])
        with self.assertRaises(ValueError):
            Path(ns["HECATE_CROPS_DIR"]).relative_to(ns["REPO_DIR"])
        Path(ns["HECATE_CROPS_DIR"]).relative_to(ns["OUT_DIR"])


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

    def test_new_notebook_targets_match_committed_cpu_triage_evidence(self):
        ns = _helper_namespace()
        t1b_data = json.loads(EVIDENCE_T1B_SEP23.read_text(encoding="utf-8"))
        t1b_meshes = tuple(t["mesh_id"] for t in t1b_data["targets"] if t["cohort"] == "PHerc0125_tier1b")
        sep23_meshes = tuple(t["mesh_id"] for t in t1b_data["targets"] if t["cohort"] == "PHerc1203_sep23_clean")
        self.assertEqual(ns["PHERC0125_TIER1B"], t1b_meshes)
        self.assertEqual(ns["PHERC1203_SEP23_CLEAN"], sep23_meshes)

        t2_data = json.loads(EVIDENCE_0826_T2.read_text(encoding="utf-8"))
        high_0826 = tuple(
            t["mesh_id"]
            for t in t2_data["targets"]
            if t["cohort"] == "PHerc0826_slice_8928_9728" and t["onsheet_score"]["f_parallel"] >= 0.8
        )
        t2_strips = tuple(
            (t["mesh_id"], f"{t['uv_crop'][0]}:{t['uv_crop'][1]},{t['uv_crop'][2]}:{t['uv_crop'][3]}")
            for t in t2_data["targets"]
            if t["cohort"] == "PHerc1203_tier2"
        )
        self.assertEqual(ns["PHERC0826_WINDINGS"], high_0826)
        self.assertEqual(ns["PHERC1203_TIER2_STRIPS"], t2_strips)
        self.assertEqual(ns["SCROLLS"]["PHerc0826"]["ct"], t2_data["public_sources"]["PHerc0826"]["ct_zarr_v2"])
        self.assertEqual(ns["SCROLLS"]["PHerc0826"]["m7"], t2_data["public_sources"]["PHerc0826"]["m7_zarr_v2"])


if __name__ == "__main__":
    unittest.main()
