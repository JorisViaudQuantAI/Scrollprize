import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from combine import REPO, combine, run


def provenance():
    value = dict(checkpoint_sha256="1"*64, mesh_sha256="2"*64, normal_sha256="3"*64,
                 frame=dict(acquisition="synthetic", axes="XYZ", units="native_voxels", level=0,
                            voxel_um_xyz=[9.36]*3, origin_native_xyz=[0, 0, 0]),
                 normal_convention="cross(dy,dx) in native XYZ", plane_order="16 planes in fixed descending order",
                 xy_pitch_um=[9.36, 9.36], depth_pitch_um=9.36,
                 inference=dict(precision="fp32", stride=48, blend="hann"),
                 preprocess=dict(clip_max=200, divide=255))
    return [copy.deepcopy(value) for _ in range(3)]


class CombinationTests(unittest.TestCase):
    def setUp(self):
        self.scores = [np.array([[.1, .2], [.3, .4]], np.float32),
                       np.array([[.2, .1], [.3, .3]], np.float32),
                       np.array([[.3, .4], [.1, .2]], np.float32)]
        self.masks = [np.ones((2, 2), bool) for _ in range(3)]
        self.offsets = [-1, 0, 1]

    def test_values_and_central_are_preserved_without_rescaling(self):
        out, report = combine(self.scores, self.masks, self.offsets, provenance())
        np.testing.assert_array_equal(out["maximum"], np.maximum.reduce(self.scores))
        np.testing.assert_array_equal(out["median"], np.median(np.stack(self.scores), axis=0))
        np.testing.assert_array_equal(out["central"], self.scores[1])
        self.assertEqual(report["common_valid_pixels"], 4)
        self.assertNotIn("candidate", report)

    def test_partial_coverage_intersects_and_does_not_fill_missing_depth(self):
        self.masks[0][0, 0] = False; self.scores[0][0, 0] = np.nan
        self.masks[2][1, 1] = False; self.scores[2][1, 1] = np.nan
        out, report = combine(self.scores, self.masks, self.offsets, provenance())
        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(out["common_valid"], expected)
        for name in ("maximum", "median", "central"):
            self.assertTrue(np.isnan(out[name][~expected]).all())
        self.assertTrue((out["winner_index"][~expected] == -1).all())
        self.assertEqual(report["excluded_pixels"], 2)

    def test_winner_indices_do_not_confuse_negative_offset_with_invalid(self):
        self.scores[0][0, 0] = .9
        out, report = combine(self.scores, self.masks, self.offsets, provenance())
        self.assertEqual(out["winner_index"][0, 0], 0)
        self.assertEqual(report["index_to_offset_planes"][0], -1)

    def test_ties_prefer_center_then_ascending_offset_and_are_recorded(self):
        self.scores[0][:] = .5; self.scores[1][:] = .5; self.scores[2][:] = .5
        out, report = combine(self.scores, self.masks, self.offsets, provenance())
        self.assertTrue((out["winner_index"] == 1).all())
        self.assertTrue((out["maximum_tie_count"] == 3).all())
        self.assertEqual(report["exact_maximum_tie_pixels"], 4)
        self.scores[1][:] = .1
        out, _ = combine(self.scores, self.masks, self.offsets, provenance())
        self.assertTrue((out["winner_index"] == 0).all())

    def test_pixelwise_max_accumulates_disjoint_noise_peaks(self):
        # Synthetic artifact-only maps: no individual depth contains this row.
        maps = [np.zeros((1, 3), np.float32) for _ in range(3)]
        for i in range(3):maps[i][0, i] = .9
        out, _ = combine(maps, [np.ones((1, 3), bool) for _ in range(3)], self.offsets, provenance())
        self.assertEqual(sum(int((a > .5).sum()) for a in maps), 3)
        self.assertTrue(all(int((a > .5).sum()) == 1 for a in maps))
        self.assertEqual(int((out["maximum"] > .5).sum()), 3)
        np.testing.assert_array_equal(out["winner_index"], [[0, 1, 2]])
        self.assertEqual(int((out["median"] > .5).sum()), 0)

    def test_wrong_masks_missing_center_duplicate_offsets_and_valid_nan_fail(self):
        for offsets in ([1, 2, 3], [-1, 0, 0], [-1, 0, float("nan")]):
            with self.assertRaises(ValueError):combine(self.scores, self.masks, offsets, provenance())
        with self.assertRaises(ValueError):combine(self.scores, [m.astype(np.uint8) for m in self.masks], self.offsets, provenance())
        self.scores[1][0, 0] = np.nan
        with self.assertRaises(ValueError):combine(self.scores, self.masks, self.offsets, provenance())

    def test_metadata_changes_in_model_geometry_order_or_inference_fail(self):
        for field, value in (("checkpoint_sha256", "a"*64), ("mesh_sha256", "b"*64),
                             ("normal_sha256", "c"*64), ("plane_order", "reversed"),
                             ("xy_pitch_um", [9.6, 9.6]), ("depth_pitch_um", 9.6),
                             ("inference", dict(precision="fp16")), ("preprocess", dict(divide=200))):
            prov = provenance();prov[2][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                combine(self.scores, self.masks, self.offsets, prov)

    def test_wrong_frame_and_no_common_coverage_fail(self):
        prov = provenance(); prov[1]["frame"]["axes"] = "ZYX"
        with self.assertRaises(ValueError):combine(self.scores, self.masks, self.offsets, prov)
        self.masks[0][:] = False
        with self.assertRaises(ValueError):combine(self.scores, self.masks, self.offsets, provenance())

    def test_integer_scores_and_wrong_grids_fail(self):
        maps = list(self.scores);maps[0] = maps[0].astype(np.uint8)
        with self.assertRaises(ValueError):combine(maps, self.masks, self.offsets, provenance())
        maps = list(self.scores);maps[0] = maps[0][:1]
        with self.assertRaises(ValueError):combine(maps, self.masks, self.offsets, provenance())

    def test_private_file_outputs_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp);rows=[];prov=provenance()
            for i in range(3):
                np.save(path/f"s{i}.npy", self.scores[i]);np.save(path/f"m{i}.npy", self.masks[i])
                rows.append(dict(offset_planes=self.offsets[i], map=f"s{i}.npy", mask=f"m{i}.npy", provenance=prov[i]))
            (path/"inputs.json").write_text(json.dumps(dict(maps=rows)))
            report=run(path/"inputs.json",path/"outputs")
            self.assertEqual(report["status"],"complete")
            self.assertEqual(len(report["output_sha256"]),6)
            self.assertTrue((path/"outputs/report.json").exists())
            with self.assertRaises(FileExistsError):run(path/"inputs.json",path/"outputs")
            with self.assertRaises(ValueError):run(path/"inputs.json",REPO/"must-not-exist-private-depth-output")

    def test_other_git_worktrees_rejected_for_git_directory_and_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for name, is_file in (("ordinary_clone", False), ("linked_worktree", True)):
                root = path/name; root.mkdir()
                if is_file:(root/".git").write_text("gitdir: /synthetic/unneeded/admin/path\n")
                else:(root/".git").mkdir()
                with self.subTest(worktree=name), self.assertRaises(ValueError):
                    run(path/"not-read.json", root/"private-depth-products")
                self.assertFalse((root/"private-depth-products").exists())


if __name__ == "__main__":
    unittest.main()
