# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Source-support regressions ported from ChatGPT-018 (generated volumes only, CPU-only).

No CT downloads or ink-model inference. Each test pins one ported bugfix or one
backward-compatibility guarantee for screen_scroll_rtx4060 / paris4_dual_energy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stream_snap_render as ssr  # noqa: E402
from test_stream_snap_render import _write_local_zarr_v2  # noqa: E402


def plane(z, x0=1.0, y0=1.0):
    x, y = np.meshgrid(np.arange(2, dtype=np.float32) + x0,
                       np.arange(2, dtype=np.float32) + y0, indexing="xy")
    return x, y, np.full_like(x, z)


class TestSourceSupport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def volume(self, name, values, chunks=(4, 4, 4), **kwargs):
        return _write_local_zarr_v2(self.root / name, values, chunks=chunks, **kwargs)

    # --- Positive-weight trilinear stencil (zero-fill vs 404 / missing chunk) ---

    def test_missing_chunk_corner_is_not_available_after_interpolation(self):
        p = self.volume("missing", np.full((8, 8, 8), 100, np.uint8))
        for iz in range(2):
            for iy in range(2):
                (p / "0" / str(iz) / str(iy) / "1").unlink()
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            values, support = reader.sample_trilinear(np.array([[3.0, 2, 2], [3.5, 2, 2], [4.0, 2, 2]]), return_support=True)
            legacy = reader.sample_trilinear(np.array([[3.5, 2, 2]]))
        np.testing.assert_array_equal(values, [100, 50, 0])
        np.testing.assert_array_equal(support, [True, False, False])
        self.assertIsInstance(legacy, np.ndarray)
        self.assertEqual(float(legacy[0]), 50.0)

    def test_read_box_support_marks_only_decoded_chunks(self):
        p = self.volume("box", np.zeros((8, 8, 8), np.uint8))
        (p / "0" / "1" / "1" / "1").unlink()
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            block, origin = reader.read_box([0, 0, 0], [8, 8, 8])
            _, _, support = reader.read_box([0, 0, 0], [8, 8, 8], return_support=True)
        self.assertEqual(origin, (0, 0, 0))
        self.assertEqual(block.shape, (8, 8, 8))
        self.assertFalse(support[4:, 4:, 4:].any())
        self.assertEqual(int(support.sum()), 512 - 64)

    def test_all_eight_positive_corners_and_exact_integer_edges(self):
        p = self.volume("eight", np.full((2, 2, 2), 100, np.uint8), chunks=(1, 1, 1))
        points = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [1e-12, 0, 0], [-1e-12, 0, 0], [np.nan, 0, 0]])
        for corner in np.ndindex(2, 2, 2):
            if corner != (0, 0, 0):
                (p / "0" / str(corner[0]) / str(corner[1]) / str(corner[2])).unlink()
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            _, support = reader.sample_trilinear(points, return_support=True)
        np.testing.assert_array_equal(support, [True, False, False, False, False, False])
        q = self.volume("edges", np.full((2, 2, 2), 100, np.uint8), chunks=(1, 1, 1))
        with ssr.RemoteZarrV2(q, workers=1) as reader:
            value, support = reader.sample_trilinear(np.array([[1., 1., 1.]]), return_support=True)
        self.assertEqual(value[0], 100)
        self.assertTrue(support[0])

    def test_each_positive_corner_is_required(self):
        for corner in np.ndindex(2, 2, 2):
            p = self.volume("corner" + "".join(map(str, corner)), np.full((2, 2, 2), 100, np.uint8), chunks=(1, 1, 1))
            (p / "0" / str(corner[0]) / str(corner[1]) / str(corner[2])).unlink()
            with ssr.RemoteZarrV2(p, workers=1) as reader:
                _, support = reader.sample_trilinear(np.array([[0.5, 0.5, 0.5]]), return_support=True)
            self.assertFalse(support[0], msg=str(corner))

    def test_nonfinite_zero_weight_neighbor_does_not_poison_integer(self):
        a = np.full((2, 2, 2), 17, np.float32)
        a[0, 0, 1] = np.nan
        p = self.volume("nan", a, chunks=(2, 2, 2))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            values, support = reader.sample_trilinear(np.array([[0., 0, 0], [0.5, 0, 0], [1., 0, 0]]), return_support=True)
        self.assertEqual(values[0], 17)
        np.testing.assert_array_equal(support, [True, False, False])

    def test_sparse_partition_and_dense_box_have_identical_support(self):
        rng = np.random.default_rng(9)
        p = self.volume("partition", rng.integers(0, 256, (9, 11, 13), dtype=np.uint8))
        (p / "0" / "1" / "1" / "1").unlink()
        points = rng.uniform([0, 0, 0], [12, 10, 8], size=(50, 3))
        points = np.vstack([points, [[0, 0, 0], [12, 10, 8], [3, 4, 4], [3.5, 4, 4]]])
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            dense = reader.sample_trilinear(points, return_support=True)
            grouped = reader.sample_trilinear(points, return_support=True, max_direct_box_voxels=0)
        np.testing.assert_array_equal(dense[0], grouped[0])
        np.testing.assert_array_equal(dense[1], grouped[1])

    # --- Renderer validity: out-of-bounds depth, stored zeros, missing fill ---

    def test_out_of_bounds_depth_cannot_be_hidden_by_another_nonzero_plane(self):
        p = self.volume("partial", np.full((8, 8, 8), 100, np.uint8))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            stack, valid, support = ssr.render_surface_stack(*plane(0.5), reader, num_slices=3, uv_stride=1, return_support=True)
        np.testing.assert_array_equal(stack[:, 0, 0], [0, 100, 100])
        self.assertFalse(valid.any())
        self.assertTrue(support["mesh_valid_yx"].all())
        self.assertFalse(support["full_source_available_yx"].any())
        self.assertFalse(support["source_available_lhw"][0].any())
        self.assertTrue(support["source_available_lhw"][1:].all())
        self.assertTrue(support["nonzero_intensity_yx"].all())

    def test_stored_zeros_excluded_by_default_and_available_on_opt_in(self):
        p = self.volume("stored_zeros", np.zeros((8, 8, 8), np.uint8))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            stack, legacy_valid, support = ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1, return_support=True)
            _, opt_in_valid = ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1, require_nonzero_intensity=False)
        self.assertFalse(legacy_valid.any())
        self.assertTrue(opt_in_valid.all())
        self.assertTrue(support["full_source_available_yx"].all())
        self.assertTrue(support["source_available_lhw"].all())
        self.assertFalse(support["nonzero_intensity_yx"].any())
        self.assertFalse(stack.any())

    def test_missing_nonzero_fill_never_becomes_source_support(self):
        p = self.volume("fill", np.zeros((8, 8, 8), np.uint8), sparse_zero_skip=True)
        metadata_path = p / "0" / ".zarray"
        meta = json.loads(metadata_path.read_text())
        meta["fill_value"] = 128
        metadata_path.write_text(json.dumps(meta))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            stack, valid, support = ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1, return_support=True)
            _, valid_opt_in = ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1, require_nonzero_intensity=False)
            x, y, z = plane(3)
            sx, sy, sz, _ = ssr.snap_tifxyz_to_m7(x, y, z, reader, passes=2, search_radius=2)
        self.assertTrue((stack == 128).all())
        self.assertFalse(valid.any())
        self.assertFalse(valid_opt_in.any())
        self.assertFalse(support["source_available_lhw"].any())
        for actual, expected in ((sx, x), (sy, y), (sz, z)):
            np.testing.assert_array_equal(actual, expected)

    def test_mesh_mask_requires_every_positive_weight_contributor(self):
        x, y = np.meshgrid(np.arange(4, dtype=np.float32) + 1, np.arange(4, dtype=np.float32) + 1, indexing="xy")
        z = np.full_like(x, 3.0)
        x[0, 0] = -1.0
        p = self.volume("mesh", np.full((8, 8, 8), 100, np.uint8))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            _, valid, support = ssr.render_surface_stack(
                x, y, z, reader, num_slices=1, scale=(0.0004, 1.0), return_support=True
            )
        # Pixel 2499 sits at u=0.9996: the invalid vertex keeps weight 0.0004 > 0, which
        # the legacy ``> 0.999`` threshold admitted.
        self.assertFalse(bool(support["mesh_valid_yx"][0, 2499]))
        self.assertFalse(bool(valid[0, 2499]))
        self.assertTrue(bool(support["mesh_valid_yx"][0, 2500]))
        self.assertTrue(bool(support["mesh_valid_yx"][1, 2499]))

    def test_uint16_keeps_legacy_scaling_and_strict_mode_rejects(self):
        p = self.volume("u16", np.full((8, 8, 8), 65535, np.uint16))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            stack, valid = ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1)
            self.assertTrue((stack == 255).all())
            self.assertTrue(valid.all())
            with self.assertRaisesRegex(ValueError, "validated intensity mapping"):
                ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1, strict_uint8=True)
        q = self.volume("f32", np.full((8, 8, 8), 100, np.float32))
        with ssr.RemoteZarrV2(q, workers=1) as reader:
            with self.assertRaisesRegex(ValueError, "validated intensity mapping"):
                ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1, strict_uint8=True)

    def test_default_render_signature_still_returns_two_arrays(self):
        p = self.volume("legacy", np.full((8, 8, 8), 100, np.uint8))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            result = ssr.render_surface_stack(*plane(3), reader, num_slices=3, uv_stride=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].dtype, np.uint8)
        self.assertEqual(result[1].dtype, np.bool_)

    # --- m7 snapping: no clipping, radius, unsupported rays, float32 acceptance ---

    def test_m7_out_of_bounds_center_is_not_clipped_into_volume(self):
        p = self.volume("no_peak", np.zeros((8, 8, 8), np.uint8))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            x, y, z = plane(100)
            _, _, snapped_z, metrics = ssr.snap_tifxyz_to_m7(x, y, z, reader, passes=2, search_radius=2)
        np.testing.assert_array_equal(snapped_z, z)
        self.assertEqual(metrics["median_disp_vx"], 0)
        self.assertEqual(metrics["m7_source_available_fraction_raw"], 0)

    def test_m7_search_grid_does_not_round_outside_declared_radius(self):
        a = np.zeros((8, 8, 8), np.uint8)
        a[5] = 255
        p = self.volume("radius", a)
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            x, y, z = plane(3)
            _, _, snapped_z, metrics = ssr.snap_tifxyz_to_m7(x, y, z, reader, passes=1, search_radius=1.9, step=1)
        np.testing.assert_array_equal(snapped_z, z)
        self.assertLessEqual(metrics["p90_disp_vx"], 1.9)

    def test_m7_ridge_behind_missing_chunk_is_ignored(self):
        a = np.zeros((8, 8, 8), np.uint8)
        a[5] = 255
        p = self.volume("m7_missing", a)
        for iy in range(2):
            for ix in range(2):
                (p / "0" / "1" / str(iy) / str(ix)).unlink()
        meta_path = p / "0" / ".zarray"
        meta = json.loads(meta_path.read_text())
        meta["fill_value"] = 255
        meta_path.write_text(json.dumps(meta))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            x, y, z = plane(2)
            _, _, snapped_z, metrics = ssr.snap_tifxyz_to_m7(x, y, z, reader, passes=2, search_radius=3)
        # The z>=4 chunk is absent: its 255 fill is not m7 evidence, so no move.
        np.testing.assert_array_equal(snapped_z, z)
        self.assertEqual(metrics["on_sheet_after_snap"], 0.0)

    def test_m7_supported_ridge_is_still_reached(self):
        a = np.zeros((8, 8, 8), np.uint8)
        a[5] = 255
        p = self.volume("m7_ok", a)
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            x, y, z = plane(3)
            _, _, snapped_z, metrics = ssr.snap_tifxyz_to_m7(x, y, z, reader, passes=2, search_radius=3)
        np.testing.assert_allclose(snapped_z, 5.0, atol=1e-5)
        self.assertEqual(metrics["on_sheet_after_snap"], 1.0)
        self.assertEqual(metrics["m7_source_available_fraction_after_snap"], 1.0)
        self.assertEqual(snapped_z.dtype, np.float32)

    def test_ridge_picker_ignores_nonfinite_neighbors_for_subvoxel_fit(self):
        offsets = np.array([-1.0, 0.0, 1.0])
        profile = np.array([-np.inf, 200.0, 100.0])
        self.assertEqual(ssr._pick_nearest_ridge_offset(profile, offsets, 51.0, 1.0), 0.0)

    def test_snap_rejects_nonfinite_parameters(self):
        p = self.volume("params", np.zeros((8, 8, 8), np.uint8))
        with ssr.RemoteZarrV2(p, workers=1) as reader:
            for kwargs in ({"search_radius": np.nan}, {"step": np.inf}, {"threshold": np.nan}):
                with self.assertRaises(ValueError):
                    ssr.snap_tifxyz_to_m7(*plane(3), reader, **kwargs)

    # --- Layer-alignment diagnostic ---

    def test_alignment_diagnostic_keys_and_legacy_alias(self):
        rng = np.random.default_rng(1)
        image = rng.integers(1, 255, (32, 32), dtype=np.uint8)
        stack = np.repeat(image[None], 5, axis=0)
        result = ssr.layer_coherence_score(stack, 2, patch_size=32, valid_mask=np.ones((32, 32), bool))
        self.assertEqual(result["diagnostic_kind"], "zero-shift-dominance-only")
        self.assertEqual(result["zero_shift_dominant_fraction"], 1)
        self.assertEqual(result["sheet_parallel_fraction"], result["zero_shift_dominant_fraction"])
        self.assertTrue(all("zero_shift_dominant" in p for p in result["patch_scores"]))
        legacy = ssr.layer_coherence_score(stack, 2, patch_size=32)
        self.assertEqual(legacy["patch_count"], 1)

    def test_alignment_uses_a_common_mask_for_every_shift(self):
        rng = np.random.default_rng(1)
        stack = np.repeat(rng.integers(1, 255, (32, 32), dtype=np.uint8)[None], 5, axis=0)
        mask = np.ones((32, 32), bool)
        mask[::3, ::3] = False
        result = ssr.layer_coherence_score(stack, 2, patch_size=32, valid_mask=mask)
        self.assertEqual(result["patch_count"], 0)
        self.assertEqual(result["sheet_parallel_fraction"], 0.0)

    # --- CLI privacy / provenance ---

    def test_private_guard_rejects_git_worktrees(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: elsewhere")
        with self.assertRaisesRegex(ValueError, "outside every Git"):
            ssr._private_output_path(repo / "sub" / "receipt.json")
        self.assertEqual(ssr._private_output_path(self.root / "out.npy").name, "out.npy")

    def test_private_guard_follows_symlinks(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        link = self.root / "link"
        try:
            link.symlink_to(repo, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
        with self.assertRaisesRegex(ValueError, "outside every Git"):
            ssr._private_output_path(link / "receipt.json")

    def _cli_args(self, *extra):
        mesh = self.root / "mesh"
        ssr.write_tifxyz(mesh, *plane(3), meta={"scale": [1, 1]})
        volume = self.volume("cli", np.full((8, 8, 8), 100, np.uint8))
        return ssr.build_arg_parser().parse_args([
            "--mesh-dir", str(mesh), "--ct-zarr", str(volume), "--num-slices", "3",
            "--out-npy", str(self.root / "stack.npy"), "--out-receipt", str(self.root / "receipt.json"),
            "--workers", "1", *extra,
        ])

    def test_cli_emits_separate_private_source_masks_and_file_hashes(self):
        support_path = self.root / "support.npz"
        receipt = ssr.run_pipeline(self._cli_args("--out-support-npz", str(support_path)))
        self.assertEqual(receipt["schema_version"], "1.1.0")
        self.assertEqual(receipt["stack_npy_sha256"], hashlib.sha256((self.root / "stack.npy").read_bytes()).hexdigest())
        self.assertEqual(receipt["support_npz_sha256"], hashlib.sha256(support_path.read_bytes()).hexdigest())
        with np.load(support_path, allow_pickle=False) as masks:
            self.assertEqual(
                set(masks.files),
                {"mesh_valid_yx", "source_available_lhw", "full_source_available_yx", "nonzero_intensity_yx", "full_stack_valid_yx"},
            )
            for name in masks.files:
                self.assertEqual(masks[name].dtype, np.bool_)
                self.assertTrue(masks[name].all())
        self.assertIn("legacy masked-zero exclusion", receipt["validity_definition"])
        self.assertEqual(receipt["params"]["depth_offsets_native_voxels"], [-1.0, 0.0, 1.0])

    def test_cli_rejects_nonzero_pyramid_levels(self):
        args = self._cli_args("--ct-level", "1")
        with self.assertRaisesRegex(ValueError, "level-0"):
            ssr.run_pipeline(args)


if __name__ == "__main__":
    unittest.main()
