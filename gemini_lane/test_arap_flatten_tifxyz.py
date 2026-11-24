# Copyright (c) 2026 Scrollprize Contributors (MIT License)
"""Offline CPU-only unit tests for ``gemini_lane/arap_flatten_tifxyz.py``."""
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
from PIL import Image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

GEMINI_DIR = Path(__file__).resolve().parent
if str(GEMINI_DIR) not in sys.path:
    sys.path.insert(0, str(GEMINI_DIR))

import arap_flatten_tifxyz as ARAP  # noqa: E402
import leak_guard as LG  # noqa: E402
import stream_snap_render as SSR  # noqa: E402

GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba.cuda", "pycuda")


def _make_sheared_curved_patch(
    *, h: int = 36, w: int = 44, shear_tan: float = 1.45, stretch_v: float = 2.2
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a curved 3D sheet sampled on a heavily sheared, stretched UV grid."""
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    u_phys = (jj * 20.0) + shear_tan * (ii * 20.0)
    v_phys = ii * (20.0 * stretch_v)
    radius = 700.0
    theta = (u_phys - u_phys.mean()) / radius
    x = radius * np.sin(theta) + 1200.0
    y = radius * (1.0 - np.cos(theta)) + 1200.0
    z = v_phys + 500.0
    pts = np.stack([x, y, z], axis=-1)
    valid = np.ones((h, w), dtype=bool)
    return pts, valid


class TestArapFlattenTifxyz(unittest.TestCase):
    def test_01_strongly_sheared_curved_grid_passes_2d_gate_after_arap(self) -> None:
        pts, valid = _make_sheared_curved_patch(shear_tan=3.2, stretch_v=2.2)
        res = ARAP.arap_flatten_grid(pts, valid, spacing_vx=20.0, iters=80)

        before = res["qa_before"]
        after = res["qa_after"]
        before_ok, before_reasons = ARAP.passes_shear_gate(before)
        self.assertFalse(before_ok)
        self.assertGreater(before["shear_8cell_abs_p95_deg"], 45.0)
        self.assertGreater(before["shear_8cell_pct_area_gt_20deg"], 90.0)
        self.assertTrue(any("shear" in r for r in before_reasons))

        self.assertTrue(res["passes_shear_gate"], msg=f"Failed gate: {res['gate_failure_reasons']}")
        self.assertEqual(res["gate_failure_reasons"], [])
        self.assertLess(after["shear_8cell_abs_p95_deg"], 5.0)
        self.assertEqual(after["shear_8cell_pct_area_gt_20deg"], 0.0)
        self.assertAlmostEqual(after["stretch_u_p05_p50_p95"][1], 1.0, delta=0.05)
        self.assertAlmostEqual(after["stretch_v_p05_p50_p95"][1], 1.0, delta=0.05)
        self.assertAlmostEqual(after["area_inflation_true_over_nominal"], 1.0, delta=0.05)
        self.assertEqual(res["solver"]["flipped_triangles"], 0)

    def test_02_sentinel_hole_not_bridged_and_git_repo_output_rejected(self) -> None:
        pts, valid = _make_sheared_curved_patch(h=24, w=28, shear_tan=0.8, stretch_v=1.5)
        # Punch a 6x6 sentinel hole in the interior; verify ARAP keeps an interior invalid void
        valid[9:15, 11:17] = False
        pts[~valid] = -1.0

        res = ARAP.arap_flatten_grid(pts, valid, spacing_vx=20.0, iters=60)
        self.assertTrue(res["passes_shear_gate"])
        out_val = res["valid"]
        self.assertGreater(int((~out_val).sum()), 12)
        # Interior bounding box of void should still have invalid sentinel nodes
        h_out, w_out = out_val.shape
        center_patch = out_val[h_out // 3 : (2 * h_out) // 3, w_out // 3 : (2 * w_out) // 3]
        self.assertFalse(bool(center_patch.all()))

        with self.assertRaisesRegex(ValueError, "private"):
            ARAP.flatten_tifxyz_to_dir(
                GEMINI_DIR / "_unused_src",
                GEMINI_DIR / "_forbidden_inside_git",
                voxel_um=9.362,
            )

    def test_03_directory_round_trip_writes_clean_tifxyz_with_mask_and_meta(self) -> None:
        pts, valid = _make_sheared_curved_patch(h=22, w=26, shear_tan=1.1, stretch_v=1.8)
        with tempfile.TemporaryDirectory(prefix="arap-roundtrip-", ignore_cleanup_errors=True) as td:
            root = Path(td)
            src = root / "src_mesh"
            out = root / "flat_mesh"
            SSR.write_tifxyz(
                src,
                pts[..., 0],
                pts[..., 1],
                pts[..., 2],
                meta={"format": "tifxyz", "scale": [0.05, 0.05]},
            )
            receipt = ARAP.flatten_tifxyz_to_dir(src, out, voxel_um=9.362, iters=60)
            self.assertTrue(receipt["passes_shear_gate"])
            for fn in ("x.tif", "y.tif", "z.tif", "mask.tif", "meta.json", "arap_qa.json"):
                self.assertTrue((out / fn).is_file(), f"Missing output {fn}")
            x, y, z, val_rt, meta = SSR.read_tifxyz(out)
            self.assertEqual(meta["parameterization_method"], "mit_arap_2d")
            self.assertEqual(meta["scale"], [0.05, 0.05])
            self.assertGreater(int(val_rt.sum()), 200)
            with Image.open(out / "mask.tif") as im:
                m = np.asarray(im)
            self.assertEqual(m.shape, val_rt.shape)
            self.assertEqual(int((m == 255).sum()), int(val_rt.sum()))

    def test_04_zero_torch_imports_and_zero_privacy_leaks(self) -> None:
        mod_path = GEMINI_DIR / "arap_flatten_tifxyz.py"
        mod_txt = mod_path.read_text(encoding="utf-8")
        tree = ast.parse(mod_txt)
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            for mod in mods:
                self.assertFalse(
                    any(mod == g or mod.startswith(g + ".") for g in GPU_MODULES),
                    f"arap_flatten_tifxyz.py imports forbidden GPU module {mod!r}",
                )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import arap_flatten_tifxyz; "
                f"print([m for m in {GPU_MODULES!r} if m in sys.modules])",
                str(GEMINI_DIR),
            ],
            capture_output=True,
            text=True,
            env={"CUDA_VISIBLE_DEVICES": "-1", **os.environ},
            timeout=60,
            check=True,
        )
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "[]")
        self.assertEqual(LG.find_leaks(mod_txt), [])
        self.assertEqual(LG.find_leaks((GEMINI_DIR / "test_arap_flatten_tifxyz.py").read_text(encoding="utf-8")), [])
        preflight = GEMINI_DIR / "evidence" / "pherc0826_arap_flatten_preflight.json"
        if preflight.is_file():
            self.assertEqual(LG.find_leaks(preflight.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
