"""Tests of physical geometry and mask semantics, not detector accuracy."""
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import numpy as np

from review import Config, main, outside_git, rank_map, write_packet


def one_line(spacing=35.0):
    mm = spacing / 1000
    h, w = round(12 / mm), round(24 / mm)
    yy, xx = np.mgrid[:h, :w]
    y, x = (yy + 0.5) * mm, (xx + 0.5) * mm
    p = np.full((h, w), 0.05, dtype=np.float32)
    for center in (3, 6, 9, 12, 15, 18):
        p[(abs(x - center) <= 0.4) & (abs(y - 6) <= 0.75)] = 0.9
    return p, np.ones(p.shape, bool)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(Config(), thresholds=(0.5,), angles_deg=(0.0, 90.0))

    def test_single_line_needs_no_second_row(self):
        p, m = one_line()
        r = rank_map(p, m, [35, 35], "model_score", self.config)
        self.assertTrue(r["review_required"])
        self.assertIsNone(r["automated_text_or_blank_verdict"])
        self.assertTrue(r["proposals"])
        best = r["proposals"][0]
        self.assertEqual(best["slot_count"], 6)
        self.assertEqual(best["angle_deg"], 0)
        self.assertGreater(best["length_mm"], 15)
        self.assertGreater(best["upper_flank_coverage"], 0.95)

    def test_resolution_and_anisotropic_sampling_preserve_geometry(self):
        p, m = one_line()
        original = rank_map(p, m, [35, 35], "model_score", self.config)
        fine = rank_map(np.repeat(np.repeat(p, 2, 0), 2, 1), np.repeat(np.repeat(m, 2, 0), 2, 1),
                        [17.5, 17.5], "model_score", self.config)
        anisotropic = rank_map(np.repeat(p, 2, 1), np.repeat(m, 2, 1), [35, 17.5], "model_score", self.config)
        for result in (fine, anisotropic):
            a, b = original["proposals"][0], result["proposals"][0]
            self.assertEqual(a["slot_count"], b["slot_count"])
            self.assertAlmostEqual(a["length_mm"], b["length_mm"], places=6)
            self.assertAlmostEqual(a["score"], b["score"], places=6)

    def test_missing_flanks_do_not_act_as_blank_background(self):
        p, mask = one_line()
        complete = rank_map(p, mask, [35, 35], "model_score", self.config)["proposals"][0]
        yy = (np.arange(p.shape[0]) + 0.5) * 0.035
        mask &= (abs(yy[:, None] - 6) <= 1)
        clipped = rank_map(p, mask, [35, 35], "model_score", self.config)["proposals"][0]
        self.assertIsNone(clipped["contrast"])
        self.assertIsNone(clipped["flank_hot_fraction"])
        self.assertIn("incomplete_flank_support_no_contrast_bonus", clipped["warnings"])
        self.assertLess(clipped["score"], complete["score"])

    def test_internal_gap_retains_physical_distance_and_support(self):
        p, mask = one_line()
        baseline = rank_map(p, mask, [35, 35], "model_score", self.config)["proposals"][0]
        x = (np.arange(p.shape[1]) + 0.5) * 0.035
        mask[:, (x >= 6.7) & (x <= 8.2)] = False
        p[~mask] = 1  # Ignored; it must not create another stroke or fill a gap.
        result = rank_map(p, mask, [35, 35], "model_score", self.config)["proposals"][0]
        self.assertAlmostEqual(result["length_mm"], baseline["length_mm"], places=6)
        self.assertEqual(result["slot_count"], baseline["slot_count"])
        self.assertLess(result["band_coverage"], baseline["band_coverage"])
        self.assertAlmostEqual(result["band_coverage"], 1 - 1.5 / result["length_mm"], delta=0.02)

    def test_invalid_values_cannot_create_proposals(self):
        p = np.zeros((400, 800), np.float32)
        m = np.zeros_like(p, dtype=bool)
        p[:] = np.nan
        r = rank_map(p, m, [35, 35], "model_score", self.config)
        self.assertEqual(r["proposals"], [])
        self.assertTrue(r["review_required"])
        self.assertEqual(r["valid_area_cm2"], 0)

    def test_blank_input_still_has_a_review_packet(self):
        p = np.full((200, 400), 0.1, np.float32)
        m = np.ones_like(p, dtype=bool)
        r = rank_map(p, m, [35, 35], "display_signal", self.config)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "packet"
            write_packet(p, m, r, out)
            self.assertTrue((out / "overview.png").exists())
            self.assertTrue((out / "review.html").exists())
            np.testing.assert_array_equal(np.load(out / "original_map.npy"), p)
            self.assertIn("zero proposals is not a negative", (out / "review.html").read_text())

    def test_original_crops_are_unchanged_and_existing_packet_not_overwritten(self):
        p, m = one_line()
        r = rank_map(p, m, [35, 35], "model_score", self.config)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "packet"
            write_packet(p, m, r, out)
            y0, x0, y1, x1 = r["proposals"][0]["bbox_yxyx"]
            np.testing.assert_array_equal(np.load(out / "proposal_001.npy"), p[y0:y1, x0:x1])
            with self.assertRaises(FileExistsError):
                write_packet(p, m, r, out)

    def test_git_worktree_guard_including_nested_paths(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".git").write_text("gitdir: elsewhere")
            with self.assertRaises(ValueError):
                outside_git(Path(d) / "nested" / "packet")

    def test_invalid_inputs_and_unbounded_config_fail(self):
        p, m = one_line()
        for args in [(p.astype(np.uint8), m, [35, 35], "model_score"),
                     (p, m.astype(np.uint8), [35, 35], "model_score"),
                     (p, m, [0, 35], "model_score"), (p, m, [35, 35], "unspecified")]:
            with self.assertRaises(ValueError):
                rank_map(*args, self.config)
        with self.assertRaises(ValueError):
            rank_map(p, m, [35, 35], "model_score", replace(self.config, thresholds=(0.5, 0.5)))
        with self.assertRaises(ValueError):
            rank_map(p, m, [35, 35], "model_score", replace(self.config, maximum_work_pixels=10))

    def test_search_truncation_is_explicit(self):
        p, m = one_line()
        r = rank_map(p, m, [35, 35], "model_score", replace(self.config, maximum_chains_per_threshold=1))
        self.assertTrue(r["search_truncated"])
        self.assertTrue(r["review_required"])

    def test_producer_replacing_map_or_mask_during_ranking_cannot_publish_packet(self):
        for changed in ("map", "mask"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                source_map, source_mask, output = root / "map.npy", root / "mask.npy", root / "packet"
                original = np.zeros((2, 2), dtype=np.float32)
                support = np.ones((2, 2), dtype=bool)
                np.save(source_map, original); np.save(source_mask, support)

                def replace_while_ranking(score, valid, *args):
                    np.testing.assert_array_equal(score, original)
                    np.testing.assert_array_equal(valid, support)
                    target = source_map if changed == "map" else source_mask
                    replacement = np.ones((2, 2), np.float32) if changed == "map" else np.zeros((2, 2), bool)
                    pending = root / "producer-new.npy"
                    np.save(pending, replacement)
                    pending.replace(target)  # Simulate an atomic producer handoff.
                    return dict(proposals=[], seconds=0.0, search_truncated=False)

                argv = ["review.py", "--map", str(source_map), "--mask", str(source_mask),
                        "--spacing-um", "35", "35", "--map-kind", "model_score", "--output-dir", str(output)]
                with mock.patch("sys.argv", argv), mock.patch("review.rank_map", side_effect=replace_while_ranking), mock.patch("review.write_packet") as writer:
                    with self.assertRaisesRegex(ValueError, "changed during review"):
                        main()
                    writer.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
