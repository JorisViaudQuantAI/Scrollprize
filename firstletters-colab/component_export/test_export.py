"""Synthetic contract tests; no claim about an ink detector or manuscript.

The inference tests use the real tiler, depth interpolation, OpenCV and output
writers, with a deliberately simple NumPy predictor instead of model weights.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from component_export.export import (  # noqa: E402
    ComponentWriter, prediction_support_on_output_grid, read_provenance, sha256_file,
)

SPEC = importlib.util.spec_from_file_location("component_test_infer", PROJECT / "ink" / "infer_ink.py")
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)
HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class SyntheticPredictor:
    """Depth-order-sensitive test double; not trained and not an ink model."""

    nominal_um = 1.0
    native_range_um = (0.9, 1.1)
    display_rescale = (0.25, 0.75)

    def __init__(self, checkpoint="synthetic", depth=3, patch=8):
        self.depth, self.patch = depth, patch
        self.name = "synthetic-a" if depth % 2 else "synthetic-b"
        self.info = {
            "model": self.name, "checkpoint": str(checkpoint),
            "patch_zyx": [depth, patch, patch], "normalization": "synthetic divide_255",
        }
        self.calls = 0
        self.examples = 0

    def normalize(self, patch):
        return patch.astype(np.float32) / 255.0

    def predict(self, batch):
        self.calls += 1
        self.examples += len(batch)
        return np.float32(0.2) + np.float32(0.45) * batch[:, 0] + np.float32(0.25) * batch[:, -1]


def sampling(masses=(1.0, 1.0, 1.0)):
    return {"model_plane_source_weight_masses": list(masses)}


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scrollprize-components-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def writer(self, name="export", count=1, shape=(2, 3)):
        return ComponentWriter(self.base / name, run_metadata={"output_shape_yx": list(shape)},
                               expected_components=count)

    def component(self, writer, scores=None, weights=None, masses=(1.0, 1.0, 1.0)):
        scores = np.arange(6, dtype=np.float32).reshape(2, 3) / 10 if scores is None else scores
        weights = np.ones(scores.shape, np.float32) if weights is None else weights
        writer.write_component(model_index=0, model_info={"model": "synthetic"},
                               checkpoint_sha256="0" * 64, direction="forward", shift=0.0,
                               probability=scores, weight_sum=weights, sampling=sampling(masses))

    def test_atomic_component_and_receipt_last_hashes_unknown_geometry(self):
        writer = self.writer()
        self.component(writer)
        self.assertFalse((writer.root / "receipt.json").exists())
        folder = writer.root / "component-0000"
        record = json.loads((folder / "component.json").read_text())
        self.assertTrue(record["full_depth_context"])
        self.assertIsNone(record["declared_geometry"]["normal_sha256"])
        self.assertIsNone(record["declared_geometry"]["normal_convention"])
        self.assertEqual(record["acquisition_coverage_status"].split(":")[0], "unknown")
        for filename, digest in record["files_sha256"].items():
            self.assertEqual(sha256_file(folder / filename), digest)
        receipt = writer.finish()
        self.assertEqual(receipt["components"][0]["component_json_sha256"], sha256_file(folder / "component.json"))
        self.assertEqual(sorted(p.name for p in writer.root.iterdir()), ["component-0000", "receipt.json"])
        with self.assertRaises(RuntimeError):
            self.component(writer)

    def test_partial_depth_preserves_computed_score_but_not_complete_depth_support(self):
        writer = self.writer()
        self.component(writer, masses=(1.0, 1.0, 0.25))
        folder = writer.root / "component-0000"
        self.assertTrue(np.load(folder / "prediction_support.npy").all())
        self.assertFalse(np.load(folder / "complete_depth_support.npy").any())
        self.assertTrue(np.isfinite(np.load(folder / "probability.npy")).all())
        self.assertFalse(json.loads((folder / "component.json").read_text())["full_depth_context"])

    def test_missing_prediction_is_nan_and_does_not_mutate_legacy_array(self):
        writer = self.writer()
        scores = np.full((2, 3), 0.625, np.float32)
        original = scores.copy()
        weights = np.array([[0, 1e-6, 1], [2, 0.1, 1]], np.float32)
        self.component(writer, scores, weights)
        folder = writer.root / "component-0000"
        support = np.load(folder / "prediction_support.npy")
        np.testing.assert_array_equal(support, [[False, False, True], [True, True, True]])
        saved = np.load(folder / "probability.npy")
        self.assertTrue(np.isnan(saved[~support]).all())
        np.testing.assert_array_equal(saved[support], original[support])
        np.testing.assert_array_equal(scores, original)

    @unittest.skipUnless(HAS_CV2, "Existing inference OpenCV dependency is needed")
    def test_resize_cannot_promote_partially_supported_pixels(self):
        weights = np.array([[0, 1], [1, 1]], np.float32)
        output = prediction_support_on_output_grid(weights, (4, 4))
        expected = np.ones((4, 4), bool)
        expected[:3, :3] = False
        np.testing.assert_array_equal(output, expected)
        # Nearest-neighbor transport would incorrectly admit five of these pixels.
        self.assertFalse(output[2, 2])

    def test_failed_component_is_not_published_and_run_is_incomplete(self):
        writer = self.writer(count=2)
        self.component(writer)
        with mock.patch("component_export.export.np.save", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                self.component(writer)
        self.assertEqual([p.name for p in writer.root.iterdir()], ["component-0000"])
        with self.assertRaises(ValueError):
            writer.finish()
        self.assertFalse((writer.root / "receipt.json").exists())

    def test_nonfinite_supported_scores_and_wrong_grid_are_rejected(self):
        writer = self.writer()
        with self.assertRaises(ValueError):
            self.component(writer, np.full((2, 3), np.nan, np.float32))
        with self.assertRaises(ValueError):
            self.component(writer, np.zeros((3, 2), np.float32))
        self.assertEqual(list(writer.root.iterdir()), [])

    def test_new_directory_and_both_git_marker_forms_and_symlink(self):
        self.writer()
        with self.assertRaises(FileExistsError):
            self.writer()
        for name, marker_file in (("regular", False), ("linked", True)):
            worktree = self.base / name
            worktree.mkdir()
            marker = worktree / ".git"
            marker.write_text("gitdir: elsewhere") if marker_file else marker.mkdir()
            with self.assertRaises(ValueError):
                ComponentWriter(worktree / "out", run_metadata={}, expected_components=1)
        link = self.base / "worktree-link"
        link.symlink_to(self.base / "linked", target_is_directory=True)
        with self.assertRaises(ValueError):
            ComponentWriter(link / "out", run_metadata={}, expected_components=1)

    def test_checkpoint_hash_and_changes_before_receipt(self):
        path = self.base / "synthetic-checkpoint.bin"
        path.write_bytes(b"synthetic checkpoint: not model weights")
        writer = self.writer()
        record = writer.checkpoint_before_load(path)
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        sink = writer.model_sink(record, 0, {"model": "synthetic"})
        sink("forward", 0.0, np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32), sampling())
        path.write_bytes(b"changed synthetic checkpoint")
        with self.assertRaises(ValueError):
            writer.finish()
        self.assertFalse((writer.root / "receipt.json").exists())

    def test_provenance_declarations_are_preserved_without_inventing_hashes(self):
        path = self.base / "declared.json"
        values = {"normal_convention": "Synthetic +Z test normal", "plane_order": "Synthetic increasing Z",
                  "mesh_sha256": "1" * 64, "frame": {"description": "synthetic frame"}}
        path.write_text(json.dumps(values))
        declared, digest = read_provenance(path)
        self.assertEqual(declared["normal_convention"], values["normal_convention"])
        self.assertIsNone(declared["normal_sha256"])
        self.assertEqual(digest, sha256_file(path))
        path.write_text('{"normal_sha256": "unknown"}')
        with self.assertRaises(ValueError):
            read_provenance(path)


@unittest.skipUnless(HAS_CV2 and HAS_TORCH, "Existing inference OpenCV and Torch dependencies are needed")
class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        import cv2

        self.temp = tempfile.TemporaryDirectory(prefix="scrollprize-inference-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        z, y, x = np.indices((9, 13, 15))
        self.array = (10 + z * 12 + y * 3 + x * 2).astype(np.uint8)
        np.save(self.base / "source.npy", self.array)
        self.mask = np.ones((13, 15), np.uint8) * 255
        self.mask[:2] = 0
        self.mask[:, 7] = 0
        cv2.imwrite(str(self.base / "mask.png"), self.mask)
        self.checkpoints = [self.base / "a.bin", self.base / "b.bin"]
        for path in self.checkpoints:
            path.write_bytes(("synthetic fixture " + path.name).encode())
        self.models = []

    def args(self, extra=()):
        return infer.parse_args([
            str(self.base / "source.npy"), "-c", *map(str, self.checkpoints),
            "-o", str(self.base / "legacy.png"), "--input-um", "1",
            "--roi", "1", "2", "12", "14", "--mask", str(self.base / "mask.png"),
            "--z-shifts=-1,0,1", "--direction", "both", "--save-float", "--save-each",
            *extra,
        ])

    def load_synthetic(self, path, args):
        model = SyntheticPredictor(path, depth=3 if Path(path).name == "a.bin" else 4)
        self.models.append(model)
        return model

    def execute(self, module, args):
        with mock.patch.object(module, "load_model", side_effect=self.load_synthetic), \
             mock.patch.object(module.time, "time", return_value=1000.0):
            metadata = module.run(args)
        paths = [Path(p) for files in metadata["outputs"].values() for p in files]
        paths.append(Path(args.output).with_suffix(".json"))
        return metadata, {p.name: p.read_bytes() for p in paths}

    def records(self, output):
        receipt = json.loads((output / "receipt.json").read_text())
        return [(output / row["directory"], json.loads((output / row["directory"] / "component.json").read_text()))
                for row in receipt["components"]]

    def test_multiple_models_depths_directions_and_legacy_bytes(self):
        _, baseline = self.execute(infer, self.args())
        calls = [(m.calls, m.examples) for m in self.models]
        self.models.clear()
        output = self.base / "components"
        _, exported = self.execute(infer, self.args(["--export-components", str(output)]))
        self.assertEqual(baseline, exported)  # Includes PNG, float16 NPY and JSON with a fixed clock.
        self.assertEqual(calls, [(m.calls, m.examples) for m in self.models])
        records = self.records(output)
        self.assertEqual(len(records), 12)
        self.assertEqual({(r["model_index"], r["direction"], r["shift_model_planes"]) for _, r in records},
                         {(m, d, s) for m in (0, 1) for d in ("forward", "reverse") for s in (-1.0, 0.0, 1.0)})
        for folder, record in records:
            score = np.load(folder / "probability.npy")
            support = np.load(folder / "prediction_support.npy")
            self.assertEqual(score.dtype, np.float32)
            self.assertEqual(support.dtype, np.bool_)
            self.assertEqual(score.shape, (11, 12))
            np.testing.assert_array_equal(support, self.mask[1:12, 2:14] > 0)
            self.assertTrue(np.isnan(score[~support]).all())
            self.assertEqual(record["checkpoint_sha256"], sha256_file(self.checkpoints[record["model_index"]]))
            self.assertTrue(record["full_depth_context"])
            if record["model_index"] == 0:
                shift = int(record["shift_model_planes"])
                first, last = (3 + shift, 5 + shift)
                if record["direction"] == "reverse":
                    first, last = last, first
                expected = (0.2 + 0.45 * self.array[first].astype(np.float32) / 255
                            + 0.25 * self.array[last].astype(np.float32) / 255)[1:12, 2:14]
                np.testing.assert_allclose(score[support], expected[support], rtol=0, atol=2e-7)

    def test_unchanged_default_against_pre_change_git_blob(self):
        # Pinned pre-feature source, independent of the edited implementation.
        # Source archives may omit Git history; the other contract tests still run.
        if not shutil.which("git"):
            self.skipTest("Git history unavailable")
        result = subprocess.run(["git", "show", "891013ad0ae4f51dc5d1657f23ade30009ef65d1"],
                                cwd=PROJECT, capture_output=True, text=True)
        if result.returncode:
            self.skipTest("Pinned pre-feature source blob is unavailable in this source archive")
        original = types.ModuleType("synthetic_legacy_infer")
        original.__file__ = str(PROJECT / "ink" / "infer_ink.py")
        exec(compile(result.stdout, original.__file__, "exec"), original.__dict__)
        _, before = self.execute(original, self.args())
        _, after = self.execute(infer, self.args())
        self.assertEqual(before, after)

    def test_resampled_exact_roi_grid_support_and_effective_pitch(self):
        import cv2

        args = self.args(["--input-um", "1.2", "--target-um", "1", "--resample", "always",
                          "--export-components", str(self.base / "resampled")])
        self.execute(infer, args)
        for folder, record in self.records(self.base / "resampled"):
            geometry = record["sampling"]
            self.assertAlmostEqual(geometry["effective_depth_pitch_um"], 1.0)
            self.assertAlmostEqual(geometry["source_layers_per_model_plane"], 1 / 1.2)
            self.assertAlmostEqual(geometry["shift_um"], record["shift_model_planes"])
            self.assertEqual(geometry["output_shape_yx"], [11, 12])
            my0, mx0, my1, mx1 = geometry["model_roi_y0_x0_y1_x1"]
            mh, mw = geometry["model_grid_shape_yx"]
            native = cv2.resize(self.mask, (mw, mh), interpolation=cv2.INTER_NEAREST)[my0:my1, mx0:mx1] > 0
            expected = cv2.resize((~native).astype(np.float32), (12, 11), interpolation=cv2.INTER_LINEAR) == 0
            np.testing.assert_array_equal(np.load(folder / "prediction_support.npy"), expected)
            self.assertEqual(np.load(folder / "probability.npy").shape, (11, 12))

    def test_unknown_physical_pitch_is_not_promoted_to_a_measurement(self):
        output = self.base / "unknown-pitch"
        args = self.args(["--export-components", str(output)])
        args.input_um = None
        self.execute(infer, args)
        for _, record in self.records(output):
            geometry = record["sampling"]
            self.assertFalse(geometry["physical_pitch_known"])
            self.assertIsNone(geometry["xy_pitch_um"])
            self.assertIsNone(geometry["effective_depth_pitch_um"])
            self.assertIsNone(geometry["shift_um"])
            self.assertEqual(geometry["inference_effective_depth_pitch_um"], 1.0)

    def sweep_export(self, array, model, extra=(), name="sweep"):
        source = infer.ArraySource(array, "synthetic test array")
        args = self.args(["--direction", "forward", "--z-shifts", "0", "--no-mask", *extra])
        writer = ComponentWriter(self.base / name, run_metadata={"output_shape_yx": list(array.shape[1:])},
                                 expected_components=1)
        sink = writer.model_sink(writer.checkpoint_before_load(self.checkpoints[0]), 0, model.info)
        probs, _ = infer.sweep(source, model, args, (0, 0, array.shape[1], array.shape[2]), None,
                              (1.0, 1.0, True), component_sink=sink)
        writer.finish()
        return probs, self.records(writer.root)[0]

    def test_fractional_out_of_stack_depth_retains_score_but_not_complete_support(self):
        array = np.full((41, 48, 48), 120, np.uint8)
        _, (folder, record) = self.sweep_export(array, SyntheticPredictor(depth=16),
                                               ["--surface-layer", "19.5", "--z-shifts", "13.75"])
        self.assertEqual(record["sampling"]["model_plane_source_weight_masses"][-1], 0.25)
        self.assertFalse(record["full_depth_context"])
        self.assertTrue(np.load(folder / "prediction_support.npy").all())
        self.assertFalse(np.load(folder / "complete_depth_support.npy").any())
        self.assertTrue(np.isfinite(np.load(folder / "probability.npy")).all())

    def test_empty_tile_and_padded_xy_edges_have_distinct_semantics(self):
        array = np.full((7, 16, 16), 120, np.uint8)
        array[:, :8, :8] = 0
        probs, (folder, record) = self.sweep_export(array, SyntheticPredictor(), ["--stride", "8"])
        support = np.load(folder / "prediction_support.npy")
        self.assertFalse(support[:8, :8].any())
        self.assertTrue(support[8:, :].all())
        self.assertTrue(np.isnan(np.load(folder / "probability.npy")[:8, :8]).all())
        self.assertTrue((probs[("forward", 0.0)][:8, :8] == 0).all())
        self.assertFalse(record["sampling"]["xy_tile_padding_present"])
        small = np.full((3, 4, 5), 120, np.uint8)
        _, (folder, record) = self.sweep_export(small, SyntheticPredictor(), name="padded")
        self.assertTrue(record["sampling"]["xy_tile_padding_present"])
        self.assertEqual(record["sampling"]["xy_tile_zero_padding_bottom_right"], [4, 3])
        self.assertTrue(np.load(folder / "prediction_support.npy").all())
        self.assertTrue(np.load(folder / "complete_depth_support.npy").all())
        self.assertIn("unknown", record["acquisition_coverage_status"])


if __name__ == "__main__":
    unittest.main()
