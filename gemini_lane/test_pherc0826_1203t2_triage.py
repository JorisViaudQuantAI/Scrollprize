# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline schema and privacy tests for pherc0826_1203t2_onsheet_triage.json."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "evidence" / "pherc0826_1203t2_onsheet_triage.json"

FORBIDDEN_PATTERNS = (
    re.compile(r"[A-Za-z]:\\\\"),
    re.compile(r"[A-Za-z]:/Users/", re.I),
    re.compile(r"[A-Za-z]:\\Users\\", re.I),
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"\\Users\\[^\\]+\\"),
    re.compile(r"Scrollprize_private", re.I),
    re.compile(r"\b(?:gho|ghp)_[A-Za-z0-9]{10,}\b|\bhf_[A-Za-z0-9]{20,}\b"),
)

BANNED_PUBLIC_KEYS = {
    "detectors",
    "top_hotspots",
    "hotspots",
    "artifacts",
    "montage",
    "studio_desc",
    "studio_asc",
    "ahn_s0_asc",
    "ahn_s0_desc",
    "ink_predictions",
    "prob",
    "render_seconds",
    "total_seconds",
}

ALLOWED_TARGET_KEYS = {
    "rank",
    "target_id",
    "scroll",
    "cohort",
    "mesh_id",
    "uv_crop",
    "valid_vertex_fraction",
    "valid_vertex_count",
    "total_vertex_count",
    "on_sheet_raw",
    "on_sheet_after_snap",
    "m7_source_available_fraction_raw",
    "m7_source_available_fraction_after_snap",
    "median_disp_vx",
    "p90_disp_vx",
    "center_vs_outer_disp_diff",
    "onsheet_score",
    "valid_ct_cm2",
}

REQUIRED_ONSHEET_KEYS = {
    "tiles",
    "f_parallel",
    "f_drift",
    "f_still",
    "response_median",
    "shift_median_confident",
    "layers",
    "mid",
    "dk",
}


def _collect_keys(obj: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_keys(item))
    return keys


class TestPherc0826And1203T2Triage(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(ARTIFACT.is_file(), f"Missing artifact: {ARTIFACT}")
        self.raw = ARTIFACT.read_text(encoding="utf-8")
        self.data = json.loads(self.raw)

    def test_privacy_zero_local_paths_tokens_or_predictions(self) -> None:
        for pat in FORBIDDEN_PATTERNS:
            self.assertIsNone(
                pat.search(self.raw),
                msg=f"Found forbidden local path or token pattern {pat.pattern!r}",
            )
        lowered = self.raw.lower()
        for banned_tok in (".png", "montage.png", "hotspots.png"):
            self.assertNotIn(banned_tok, lowered)
        all_keys = _collect_keys(self.data)
        self.assertTrue(
            all_keys.isdisjoint(BANNED_PUBLIC_KEYS),
            msg=f"Found banned private/detector keys: {sorted(all_keys & BANNED_PUBLIC_KEYS)}",
        )

    def test_renderer_semantics_and_metadata(self) -> None:
        self.assertEqual(self.data["renderer_commit"], "4f347c1")
        sem = self.data["renderer_semantics"]
        self.assertIn("chatgpt-018", sem["source_support_patch"].lower())
        self.assertTrue(sem["full_source_support_all_planes"])
        self.assertTrue(sem["require_nonzero_intensity"])
        self.assertTrue(sem["missing_chunk_and_fill_excluded_from_support"])
        self.assertTrue(sem["no_oob_m7_snap_clipping"])
        self.assertTrue(sem["float32_candidate_snap_validation"])
        self.assertTrue(sem["no_ink_detector_run"])
        self.assertEqual(self.data["voxel_um"], 9.362)
        self.assertEqual(self.data["num_slices"], 5)
        self.assertEqual(self.data["snap_passes"], 2)

    def test_expected_non_overlapping_targets_and_allowlist(self) -> None:
        targets = self.data["targets"]
        self.assertEqual(len(targets), 7)
        expected_ids = {
            "PHerc1203/auto_grown_20251005221856743_r0-253_c158-226",
            "PHerc1203/auto_grown_20251005221856743_r0-253_c0-62",
            "PHerc1203/auto_grown_20251005230830030_r0-165_c0-68",
            "PHerc0826/w020",
            "PHerc0826/w030",
            "PHerc0826/w040",
            "PHerc0826/w050",
        }
        actual_ids = {t["target_id"] for t in targets}
        self.assertEqual(actual_ids, expected_ids)
        # Ensure skipped PHerc0826 windings are absent from target list.
        mesh_ids = {t["mesh_id"] for t in targets}
        self.assertNotIn("w027", mesh_ids)
        self.assertNotIn("w051", mesh_ids)

        for idx, t in enumerate(targets, start=1):
            self.assertEqual(set(t.keys()), ALLOWED_TARGET_KEYS)
            self.assertEqual(t["rank"], idx)
            self.assertEqual(set(t["onsheet_score"].keys()), REQUIRED_ONSHEET_KEYS)
            self.assertEqual(t["onsheet_score"]["layers"], 5)
            self.assertEqual(t["onsheet_score"]["mid"], 2)
            self.assertEqual(t["onsheet_score"]["dk"], 2)
            self.assertGreater(t["onsheet_score"]["tiles"], 0)
            self.assertGreater(t["valid_ct_cm2"], 0.0)
            for frac_key in (
                "valid_vertex_fraction",
                "on_sheet_raw",
                "on_sheet_after_snap",
                "m7_source_available_fraction_raw",
                "m7_source_available_fraction_after_snap",
            ):
                self.assertGreaterEqual(t[frac_key], 0.0)
                self.assertLessEqual(t[frac_key], 1.0)
            coh = t["onsheet_score"]
            self.assertAlmostEqual(
                coh["f_parallel"] + coh["f_drift"] + coh["f_still"],
                1.0,
                places=3,
            )

    def test_rankings_and_area_totals_consistent(self) -> None:
        targets = self.data["targets"]
        # Targets list is strictly ordered by f_parallel descending, then on_sheet_after_snap descending.
        expected_order = [
            t["target_id"]
            for t in sorted(
                targets,
                key=lambda r: (-r["onsheet_score"]["f_parallel"], -r["on_sheet_after_snap"], r["target_id"]),
            )
        ]
        self.assertEqual([t["target_id"] for t in targets], expected_order)

        combined = self.data["rankings"]["combined"]
        self.assertEqual(combined["by_onsheet_f_parallel"], expected_order)
        self.assertEqual(
            combined["by_on_sheet_after_snap"],
            [
                t["target_id"]
                for t in sorted(
                    targets,
                    key=lambda r: (-r["on_sheet_after_snap"], -r["onsheet_score"]["f_parallel"], r["target_id"]),
                )
            ],
        )
        self.assertEqual(
            combined["by_valid_ct_cm2"],
            [t["target_id"] for t in sorted(targets, key=lambda r: (-r["valid_ct_cm2"], r["target_id"]))],
        )

        # Cohort totals and rankings.
        by_cohort = self.data["rankings"]["by_cohort"]
        self.assertEqual(set(by_cohort.keys()), {"PHerc1203_tier2", "PHerc0826_slice_8928_9728"})
        for cohort, block in by_cohort.items():
            sub = [t for t in targets if t["cohort"] == cohort]
            self.assertEqual(
                block["by_onsheet_f_parallel"],
                [
                    t["target_id"]
                    for t in sorted(
                        sub,
                        key=lambda r: (-r["onsheet_score"]["f_parallel"], -r["on_sheet_after_snap"], r["target_id"]),
                    )
                ],
            )
            self.assertAlmostEqual(
                self.data["cohort_totals_valid_ct_cm2"][cohort],
                round(sum(t["valid_ct_cm2"] for t in sub), 4),
                places=4,
            )

        self.assertAlmostEqual(
            self.data["total_valid_ct_cm2"],
            round(sum(t["valid_ct_cm2"] for t in targets), 4),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
