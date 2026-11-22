# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline integrity and schema checks for gemini_lane evidence artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
FORBIDDEN_PATTERNS = (
    re.compile(r"[A-Za-z]:\\\\"),
    re.compile(r"[A-Za-z]:/Users/", re.I),
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"\\Users\\[^\\]+\\"),
    re.compile(r"\b(?:gho|ghp)_[A-Za-z0-9]{10,}\b|\bhf_[A-Za-z0-9]{20,}\b"),
)


class TestGeminiLaneEvidence(unittest.TestCase):
    def assert_no_private_paths_or_tokens(self, text: str) -> None:
        for pat in FORBIDDEN_PATTERNS:
            self.assertIsNone(
                pat.search(text),
                msg=f"Found forbidden local/token pattern {pat.pattern!r} in evidence text",
            )

    def test_coverage_matrix_schema(self) -> None:
        path = EVIDENCE / "coverage_matrix.json"
        self.assertTrue(path.is_file(), f"Missing {path}")
        raw = path.read_text(encoding="utf-8")
        self.assert_no_private_paths_or_tokens(raw)
        data = json.loads(raw)

        self.assertEqual(data["summary"]["eligible_scroll_count"], 19)
        self.assertEqual(data["summary"]["scrolls_with_s3_m7"], 19)
        scrolls = data["scrolls"]
        self.assertEqual(len(scrolls), 19)
        names = [r["scroll"] for r in scrolls]
        self.assertEqual(len(set(names)), 19)
        for row in scrolls:
            self.assertIn(row["voxel_um"], (8.64, 9.362))
            self.assertTrue(row["s3_has_m7_and_normal_grids"])
            self.assertGreaterEqual(row["s3_volume_count"], 2)

    def test_pherc0306b_rtx4060_replay(self) -> None:
        path = EVIDENCE / "pherc0306b_rtx4060_replay.json"
        self.assertTrue(path.is_file(), f"Missing {path}")
        raw = path.read_text(encoding="utf-8")
        self.assert_no_private_paths_or_tokens(raw)
        data = json.loads(raw)

        self.assertTrue(data["all_pass"])
        self.assertEqual(data["runtime"]["device"], "cuda")
        self.assertEqual(len(data["results"]), 3)
        for idx, res in enumerate(data["results"]):
            self.assertEqual(res["seed"], idx)
            self.assertEqual(res["verdict"], "PASS")
            self.assertEqual(res["tiles"], 36)
            self.assertEqual(res["compared_px"], 1849)
            self.assertLessEqual(res["max_abs_diff"], 0.02)
            self.assertGreaterEqual(res["pearson_r"], 0.999)
            self.assertLess(res["peak_vram_allocated_mib"], 5500.0)


if __name__ == "__main__":
    unittest.main()
