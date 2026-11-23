# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline schema/privacy tests for the sanitized PHerc0125 + PHerc0191 on-sheet triage JSON."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_pherc0125_0191_onsheet_triage as B  # noqa: E402

EVIDENCE = HERE / "evidence" / "pherc0125_0191_onsheet_triage.json"
PRIVACY_PATTERNS = (
    r"(?<![A-Za-z])[A-Za-z]:\\{1,2}",
    r"C:/Users",
    r"/Users/",
    r"/home/",
    r"joris",
    r"Documents",
    r"Scrollprize_private",
    r"\.(png|tif|tiff|npy|npz|safetensors)\b",
)
BANNED_KEYS = ("detectors", "top_hotspots", "artifacts", "montage", "hotspots", "render_seconds", "prob", "ink")


def _summary(scroll: str, tag: str, f_par: float, snap: float, cm2: float) -> dict:
    return {
        "scroll": scroll,
        "mesh_tag": tag,
        "voxel_um": 9.362,
        "num_slices": 5,
        "snap_passes": 2,
        "uv_crop": None,
        "geometry_snap_metrics": {
            "valid_vertex_fraction": 0.5, "valid_vertex_count": 10, "total_vertex_count": 20,
            "on_sheet_raw": 0.3, "on_sheet_after_snap": snap, "median_disp_vx": 3.0,
            "p90_disp_vx": 9.0, "center_vs_outer_disp_diff": -0.3,
            "passes": 2, "search_radius": 12.0, "step": 1.0, "threshold": 51.0,
        },
        "onsheet_coherence": {
            "tiles": 5, "tile": 256, "step": 128, "r_t": 0.1, "s_t": 1.5, "hp_sigma": 6.0,
            "min_valid": 0.9, "f_parallel": f_par, "f_drift": 0.1, "f_still": 0.05,
            "response_median": 0.03, "shift_median_confident": 2.5, "layers": 5, "mid": 2, "dk": 2,
        },
        "valid_ct_cm2": cm2,
        "detectors": {"studio_desc": {"p995": 0.9}},
        "top_hotspots": [{"y": 1, "x": 2}],
        "artifacts": {"montage": "montage.png", "hotspots": "hotspots.png"},
        "render_seconds": 1.0,
        "m7_chunks_fetched": 1,
    }


def _all_summaries() -> dict[str, list[dict]]:
    return {
        scroll: [
            _summary(scroll, tag, 0.9 - 0.05 * i, 0.9, 2.0 + i)
            for i, tag in enumerate(B.EXPECTED_MESHES[scroll])
        ]
        for scroll in B.EXPECTED_MESHES
    }


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


class BuilderTests(unittest.TestCase):
    def test_combined_schema_and_sanitization(self):
        data = B.build_combined(_all_summaries())
        self.assertEqual(set(data), B.TOP_LEVEL_KEYS)
        self.assertEqual([s["scroll"] for s in data["scrolls"]], ["PHerc0125", "PHerc0191"])
        for scroll in data["scrolls"]:
            self.assertEqual(set(scroll), B.SCROLL_KEYS)
            self.assertEqual(set(scroll["source"]), B.SOURCE_KEYS)
            self.assertEqual([s["rank"] for s in scroll["sheets"]], list(range(1, len(scroll["sheets"]) + 1)))
            for sheet in scroll["sheets"]:
                self.assertEqual(set(sheet), B.SHEET_KEYS)
        banned = set(BANNED_KEYS) & set(_walk_keys(data))
        self.assertEqual(banned, set())
        raw = json.dumps(data)
        for pattern in PRIVACY_PATTERNS:
            self.assertIsNone(re.search(pattern, raw), pattern)

    def test_mesh_sets_are_exact(self):
        self.assertEqual(len(B.EXPECTED_MESHES["PHerc0125"]), 5)
        self.assertNotIn("z14144_w045", B.EXPECTED_MESHES["PHerc0125"])
        self.assertEqual(B.EXPECTED_MESHES["PHerc0191"], ("w020", "w030", "w040", "w050"))
        for scroll, source in B.SOURCES.items():
            self.assertRegex(source["mesh_revision"], r"^[0-9a-f]{40}$", scroll)
        self.assertTrue(B.SOURCES["PHerc0125"]["mesh_revision"].startswith("66000516"))

    def test_rejects_wrong_or_missing_meshes(self):
        data = _all_summaries()
        data["PHerc0191"].pop()
        with self.assertRaises(ValueError):
            B.build_combined(data)
        data = _all_summaries()
        data["PHerc0125"][0] = _summary("PHerc0191", "z10544_w035", 0.5, 0.5, 1.0)
        with self.assertRaises(ValueError):
            B.build_combined(data)
        with self.assertRaises(ValueError):
            B.build_scroll_entry("PHerc9999", [])

    def test_load_summaries_reads_only_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sheet = root / "PHerc0191_w020"
            sheet.mkdir()
            (sheet / "summary.json").write_text(json.dumps({"mesh_tag": "w020"}), encoding="utf-8")
            (sheet / "montage.png").write_bytes(b"not-an-image")
            (root / "PHerc0191_batch_summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(B.load_summaries(root), [{"mesh_tag": "w020"}])


class PublicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = EVIDENCE.read_text(encoding="utf-8")
        cls.data = json.loads(cls.raw)

    def test_no_local_paths_images_or_private_markers(self):
        for pattern in PRIVACY_PATTERNS:
            self.assertIsNone(re.search(pattern, self.raw), pattern)
        self.assertEqual(set(BANNED_KEYS) & set(_walk_keys(self.data)), set())

    def test_schema(self):
        data = self.data
        self.assertEqual(set(data), B.TOP_LEVEL_KEYS)
        self.assertEqual(data["schema_version"], B.SCHEMA_VERSION)
        self.assertEqual([s["scroll"] for s in data["scrolls"]], ["PHerc0125", "PHerc0191"])
        for scroll in data["scrolls"]:
            name = scroll["scroll"]
            self.assertEqual(set(scroll), B.SCROLL_KEYS)
            self.assertEqual(scroll["source"], B.SOURCES[name])
            self.assertEqual((scroll["num_slices"], scroll["snap_passes"]), (5, 2))
            self.assertAlmostEqual(scroll["voxel_um"], 9.362)
            ids = [s["mesh_id"] for s in scroll["sheets"]]
            self.assertEqual(sorted(ids), sorted(B.EXPECTED_MESHES[name]))
            self.assertEqual(set(scroll["rankings"]), {"by_onsheet_f_parallel", "by_on_sheet_after_snap", "by_valid_ct_cm2"})
            for order in scroll["rankings"].values():
                self.assertEqual(sorted(order), sorted(ids))
            for sheet in scroll["sheets"]:
                self.assertEqual(set(sheet), B.SHEET_KEYS)
                self.assertGreater(sheet["valid_ct_cm2"], 0)
                self.assertTrue(0 <= sheet["on_sheet_after_snap"] <= 1)
                self.assertTrue(0 <= sheet["onsheet_score"]["f_parallel"] <= 1)
                self.assertTrue(0 <= sheet["valid_vertex_fraction"] <= 1)
            scores = [s["onsheet_score"]["f_parallel"] for s in scroll["sheets"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertAlmostEqual(
                scroll["total_valid_ct_cm2"], sum(s["valid_ct_cm2"] for s in scroll["sheets"]), places=3
            )


if __name__ == "__main__":
    unittest.main()
