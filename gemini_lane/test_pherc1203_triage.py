# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline tests for the sanitized PHerc1203 on-sheet triage builder."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_pherc1203_onsheet_triage as B  # noqa: E402


def _summary(tag: str, f_par: float, snap: float, cm2: float) -> dict:
    return {
        "scroll": "PHerc1203",
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
    }


class TriageBuilderTests(unittest.TestCase):
    def test_ranking_and_sanitization(self):
        data = B.build_public_triage(
            [_summary("a", 0.5, 0.9, 3.0), _summary("b", 0.8, 0.8, 5.0), _summary("c", 0.7, 0.95, 4.0)]
        )
        self.assertEqual([s["mesh_id"] for s in data["sheets"]], ["b", "c", "a"])
        self.assertEqual([s["rank"] for s in data["sheets"]], [1, 2, 3])
        self.assertEqual(data["rankings"]["by_on_sheet_after_snap"], ["c", "a", "b"])
        self.assertEqual(data["rankings"]["by_valid_ct_cm2"], ["b", "c", "a"])
        self.assertEqual(data["total_valid_ct_cm2"], 12.0)
        text = repr(data)
        for banned in ("detectors", "top_hotspots", "artifacts", "montage", "hotspots", "render_seconds", "png"):
            self.assertNotIn(banned, text)

    def test_inconsistent_inputs_rejected(self):
        a, b = _summary("a", 0.5, 0.9, 3.0), copy.deepcopy(_summary("b", 0.8, 0.8, 5.0))
        b["snap_passes"] = 0
        with self.assertRaises(ValueError):
            B.build_public_triage([a, b])
        with self.assertRaises(ValueError):
            B.build_public_triage([])


if __name__ == "__main__":
    unittest.main()
