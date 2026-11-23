"""Tiny generated fixtures only: raw export, overwrite and Git-location gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


CLI = Path(__file__).with_name("cli.py")


class RenderCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="surface-render-cli-")
        self.root = Path(self.temporary.name)
        self.volume_path = self.root / "generated_volume.npy"
        self.geometry_path = self.root / "generated_geometry.npz"
        self.origin = np.asarray([-2.0, 4.0, 11.0])
        z, y, x = np.indices((10, 10, 10), dtype=np.float64)
        volume = 2 * (x + self.origin[0]) - 3 * (y + self.origin[1]) + 0.25 * (z + self.origin[2]) - 12.5
        np.save(self.volume_path, volume.astype(np.float32))
        uv = np.asarray([[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]])
        xyz = np.column_stack((4 + uv[:, 0], 4 + uv[:, 1], np.full(4, 4.0))) + self.origin
        np.savez(self.geometry_path, uv=uv, xyz=xyz, triangles=np.asarray([[0, 1, 2], [0, 2, 3]]))

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, output, extra=(), *, module=False):
        command = ([sys.executable, "-m", "surface_sampling.cli"] if module
                   else [sys.executable, str(CLI)])
        command += [
            "--volume", str(self.volume_path), "--geometry", str(self.geometry_path),
            "--output", str(output), "--volume-origin", "-2", "4", "11",
            "--origin-uv", "0", "0", "--step-uv", "1", "1", "--shape-yx", "2", "2",
            "--depths=-0.5,0.5", "--samples-per-axis", "4",
        ]
        command += list(extra)
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            command, cwd=CLI.parent.parent, env=environment,
            capture_output=True, text=True, timeout=30, check=False,
        )

    def test_raw_float_export_mask_hashes_and_both_entrypoints(self):
        u, v = np.meshgrid(np.arange(2), np.arange(2))
        for module in (False, True):
            with self.subTest(module=module):
                output = self.root / ("module_output" if module else "standalone_output")
                result = self.call(output, module=module)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["valid_pixels"], 4)
                report_text = (output / "report.json").read_text()
                self.assertNotIn(str(self.root), report_text)
                self.assertNotIn(str(CLI.parent), report_text)
                report = json.loads(report_text)
                self.assertFalse(report["spec"]["automatic_intensity_scaling"])
                for role, path in (("volume", self.volume_path), ("geometry", self.geometry_path)):
                    self.assertEqual(report["source_files"][role]["filename"], path.name)
                    self.assertEqual(report["source_files"][role]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                for index, depth in enumerate((-0.5, 0.5)):
                    expected = 2 * (2 + u) - 3 * (8 + v) + 0.25 * (15 + depth) - 12.5
                    with Image.open(output / f"depth_{index:03d}.tif") as layer:
                        self.assertEqual(layer.mode, "F")
                        actual = np.asarray(layer)
                    self.assertEqual(actual.dtype, np.dtype("float32"))
                    np.testing.assert_array_equal(actual, expected.astype(np.float32))
                    self.assertTrue((actual < 0).all(), "negative raw CT values must survive export")
                with Image.open(output / "valid_mask.png") as mask:
                    self.assertEqual(mask.mode, "1")
                    self.assertTrue(np.asarray(mask).all())

    def test_existing_output_and_excessive_allocation_fail_before_work(self):
        output = self.root / "existing_output"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep this exact content\n")
        result = self.call(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(sentinel.read_text(), "keep this exact content\n")
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["keep.txt"])

        missing = self.root / "not_a_volume.npy"
        excessive = self.root / "excessive_output"
        result = self.call(excessive, extra=("--shape-yx", "1024", "1024", "--volume", str(missing)))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pixel count exceeds", result.stderr)
        self.assertNotIn("existing local", result.stderr)
        self.assertFalse(excessive.exists())

    def test_git_ancestor_and_symlink_routes_are_rejected(self):
        working_tree = self.root / "worktree"
        working_tree.mkdir()
        # Worktrees/submodules may use a .git FILE, not only a directory.
        (working_tree / ".git").write_text("gitdir: generated-placeholder\n")
        external_alias = self.root / "alias_to_worktree"
        external_alias.symlink_to(working_tree, target_is_directory=True)
        outside = self.root / "outside"
        outside.mkdir()
        internal_alias = working_tree / "alias_to_outside"
        internal_alias.symlink_to(outside, target_is_directory=True)
        for output in (working_tree / "output", external_alias / "output", internal_alias / "output"):
            with self.subTest(output=output.name):
                result = self.call(output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("outside every Git working tree", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
