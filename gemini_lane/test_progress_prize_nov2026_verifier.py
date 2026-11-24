# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gemini_lane/progress_prize_nov2026_verifier.py."""
from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import progress_prize_nov2026_verifier as V  # noqa: E402

VERIFIER_PATH = HERE / "progress_prize_nov2026_verifier.py"
TEST_PATH = HERE / "test_progress_prize_nov2026_verifier.py"
DOSSIER_MD_PATH = HERE / "evidence" / "progress_prize_nov2026_dossier.md"
GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba.cuda", "pycuda")


class TestProgressPrizeNov2026Verifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = V.build_consolidated_dossier_summary(REPO_ROOT)

    def test_01_pillar1_direct_unwrapping_invariants(self) -> None:
        p1 = self.summary["pillar1_direct_unwrapping"]
        rgt = p1["exp_t9_seismic_rgt"]
        self.assertEqual(rgt["http_bytes"], 12239830)
        self.assertLessEqual(rgt["http_bytes"], rgt["budget_bytes"])
        self.assertFalse(rgt["persistent_ct_on_disk"])

        c306 = rgt["pherc0306b_control"]
        self.assertEqual(c306["sheet_count_rgt"], 4)
        self.assertEqual(c306["sheet_count_single_baseline"], 1)
        self.assertEqual(c306["crossing_fraction_rgt"], 0.0)
        self.assertAlmostEqual(c306["area_gain_ratio_vs_baseline"], 5.499828, places=6)
        self.assertAlmostEqual(c306["median_f_parallel_rgt"], 0.202, places=4)

        c1545 = rgt["pherc1545_crushed_unread"]
        self.assertEqual(c1545["sheet_count_rgt"], 5)
        self.assertEqual(c1545["sheet_count_single_baseline"], 1)
        self.assertEqual(c1545["crossing_fraction_rgt"], 0.0)
        self.assertAlmostEqual(c1545["area_gain_ratio_vs_baseline"], 14.871486, places=6)
        self.assertAlmostEqual(c1545["median_f_parallel_rgt"], 0.4167, places=4)
        self.assertAlmostEqual(c1545["baseline_f_parallel"], 0.2857, places=4)
        self.assertEqual(c1545["f_parallel_rel_gain_pct"], 45.9)

        t2 = p1["exp_t2_spiral_winding_resample"]
        self.assertEqual(t2["du_p95_over_p05_before"], 5.183)
        self.assertEqual(t2["du_p95_over_p05_after"], 1.019)
        self.assertLessEqual(t2["max_shear_deg_after"], 15.0)
        self.assertLess(abs(t2["triangulated_area_change_pct"]), 0.2)

        gm7 = p1["grow_m7_patch_pherc0846a_preflight"]
        self.assertEqual(gm7["scroll"], "PHerc0846A")
        self.assertEqual(gm7["connected_components_in_grid"], 1)
        self.assertAlmostEqual(gm7["on_sheet_raw"], 0.996237, places=6)
        self.assertAlmostEqual(gm7["on_sheet_after_snap"], 0.999059, places=6)
        self.assertFalse(gm7["persistent_ct_on_disk"])
        self.assertFalse(gm7["gate_passed"])

        t11 = p1["exp_t11_seismic_rgt_stitched"]
        self.assertEqual(t11["http_bytes"], 34512551)
        self.assertFalse(t11["persistent_ct_on_disk"])
        self.assertEqual(t11["total_sheet_count_rgt"], 34)
        self.assertAlmostEqual(t11["total_usable_cm2_rgt_across_scrolls"], 0.215204, places=6)
        self.assertAlmostEqual(t11["max_seam_phase_rmse_rad"], 0.102162, places=6)
        self.assertAlmostEqual(t11["pherc1545_area_gain"], 29.00924, places=5)

        t15 = p1["exp_t15_grid_quality_audit"]
        self.assertEqual(t15["http_bytes"], 19646463)
        self.assertFalse(t15["persistent_downloads_on_disk"])
        self.assertEqual(t15["pherc0211_shear_8cell_abs_p95_deg"], 2.889)
        self.assertEqual(t15["pherc0125_screened_pct_area_gt_20deg"], 0.124)
        self.assertEqual(t15["pherc0125_screened_area_inflation"], 1.0408)
        self.assertEqual(t15["pherc0191_hf_pct_area_gt_20deg"], 58.959)
        self.assertEqual(t15["exp_t11_rgt_synthetic_shear_p95_deg"], 0.32)

        t18 = p1["exp_t18_mit_arap_2d_flattening"]
        self.assertEqual(t18["windings_evaluated_0211"], 6)
        self.assertEqual(t18["mit_arap_gate_passed_0211"], 6)
        self.assertEqual(t18["claude_flatboi_gate_passed_0211"], 6)
        self.assertEqual(t18["mit_arap_median_sh8_deg"], 6.03)
        self.assertEqual(t18["claude_flatboi_median_sh8_deg"], 6.796)
        self.assertEqual(t18["mit_arap_median_area_inflation"], 0.9889)
        self.assertEqual(t18["pherc0826_windings_ready"], 7)
        self.assertEqual(t18["pherc0826_total_ready_valid_ct_cm2"], 37.4939)
        self.assertEqual(t18["pherc0211_w033_arap_valid_ct_cm2"], 5.4443)
        self.assertEqual(t18["pherc0211_w033_arap_verdict"], "POSITIVE_GATE_PASS")

    def test_02_pillar2_fp16_patches_and_streaming_invariants(self) -> None:
        patches = self.summary["artifacts"]["patches"]
        self.assertEqual(patches["studio_fp16"]["base_commit"], "bf66f753176792a966bca17822629fb8821f20c8")
        self.assertEqual(patches["hecate_fp16_main"]["base_commit"], "9cb86e500e944b11a06a7020403cde5dffb5bcb2")
        self.assertEqual(patches["hecate_fp16_pr3"]["base_commit"], "f4386b19e0639f1c3da79572309ecc3f4cec760e")
        for pinfo in patches.values():
            self.assertTrue(pinfo["lf_clean"])
            self.assertGreater(pinfo["hunk_count"], 0)

        p2 = self.summary["pillar2_fp16_and_streaming"]
        par = p2["studio_rtx4060_parity"]
        self.assertTrue(par["fp16_s48_parity_pass"])
        self.assertFalse(par["bf16_s48_parity_pass"])
        self.assertEqual(par["recommended_rtx4060_precision"], "fp16")
        self.assertEqual(par["fp16_desc_s48_b2"]["peak_allocated_mib"], 2164)
        self.assertEqual(par["fp16_desc_s48_b2"]["max_abs_diff"], 0.002526)
        self.assertEqual(par["fp16_asc_s48_b4"]["max_abs_diff"], 0.002113)
        self.assertEqual(par["bf16_over_fp16_max_diff_ratio_desc_b2"], 4.38)
        self.assertEqual(par["bf16_over_fp16_max_diff_ratio_asc_b4"], 29.22)

        gate = p2["pherc0306b_positive_gate_rtx4060"]
        self.assertEqual(gate["peak_allocated_mib"], 1608)
        self.assertEqual(gate["valid_ct_cm2"], 0.4228)
        self.assertEqual(gate["f_parallel"], 0.9375)
        self.assertEqual(gate["studio_asc_p995"], 0.93303)
        self.assertEqual(gate["studio_asc_max"], 0.95549)

    def test_03_pillar3_fp_suppression_invariance_and_falsification(self) -> None:
        p3 = self.summary["pillar3_fp_suppression_invariance_and_ledger"]
        q31 = p3["exp_t8_q31_cloud_masked_rescue"]
        self.assertEqual(q31["pipeline_b_masked_mad_all_levels_top1_hit_rate"], 1.0)
        self.assertEqual(q31["pipeline_b_masked_mad_all_levels_top3_recall"], 1.0)
        self.assertEqual(q31["pipeline_b_masked_mad_all_levels_cloud_top3_far"], 0.0)
        self.assertEqual(q31["pipeline_b_masked_mad_all_levels_cloud_outranks_fpr"], 0.0)
        self.assertEqual(q31["pipeline_a_global_z2_30pct"]["top3_true_text_recall"], 0.333333)

        q34 = p3["exp_t6_q34_stroke_bounded_tfce"]
        self.assertTrue(q34["pherc0306b_line1"]["passes_q34_bar"])
        self.assertEqual(q34["pherc0306b_line1"]["sb_retention_pct"], 75.8)
        self.assertFalse(q34["bilateral_clouds_2_4mm"]["passes_q34_bar"])
        self.assertEqual(q34["bilateral_clouds_2_4mm"]["suppression_pct"], 85.5)

        q22 = p3["exp_t4_q22_radiometric_audit"]
        self.assertEqual(q22["total_scrolls_audited"], 13)
        self.assertEqual(q22["pherc0358_papyrus_wall_p50_shift"], 24.0)
        self.assertTrue(q22["all_scrolls_unflagged_after_nyul"])

        t10 = p3["exp_t10_gpu_orientation_and_nyul"]
        self.assertEqual(t10["orientation_iou_ge_07_range"], [0.848127, 0.883746])
        self.assertEqual(t10["orientation_pearson_r_range"], [0.968067, 0.981438])
        self.assertEqual(t10["orientation_hit_retention_ge_07_range"], [0.922408, 0.94832])
        self.assertEqual(t10["shift_plus24_pearson_r"], 0.99474)
        self.assertEqual(t10["shift_plus24_hit_retention_ge_07"], 0.963019)

        nogo = p3["closed_nogo_zero_shot_2d_1d_filters"]
        self.assertEqual(nogo["exp_t3_pherc0139_overall"], "CLOSED NO-GO")
        self.assertEqual(nogo["exp_t5_pherc0306b_verdict"], "CLOSED NO-GO")
        self.assertEqual(nogo["exp_t7_overall_verdict"], "CLOSED NO-GO")
        self.assertEqual(nogo["exp_t7_pherc0841_2p403um_wst_slope_drop_vs_pm8"], -0.04112)
        self.assertEqual(nogo["exp_t12_overall_verdict"], "CLOSED NO-GO")
        self.assertEqual(nogo["exp_t12_bridge_post_alignment_iou"], 0.57853)
        self.assertEqual(nogo["exp_t12_fold_b_human_2p4um_auc"], 0.53408)

        t13 = p3["exp_t13_tiled_q34_ranker"]
        self.assertEqual(t13["upgraded_positive_pass_count"], 5)
        self.assertEqual(t13["upgraded_negative_pass_count"], 0)
        self.assertEqual(t13["pherc0343_argus_area_matched_retention_max"], 0.967727)

        t16 = p3["exp_t16_gpu_shear_and_0211"]
        self.assertEqual(t16["min_15deg_interior_ret_ge_07"], 0.939702)
        self.assertEqual(t16["min_30deg_interior_ret_ge_07"], 0.906106)
        self.assertTrue(t16["pherc0211_winner_letters_recovered"])
        self.assertFalse(t16["pherc0211_legacy_passes_q34_bar"])

        t17 = p3["exp_t17_gpu_0211_and_0306b_matrix"]
        self.assertTrue(t17["stride96_snap2_screening_gate_pass"])
        self.assertEqual(t17["stride96_snap2_ret_ge_07_vs_canon"], 0.869617)
        self.assertEqual(t17["snap2_on_sheet_after_snap"], 0.9953125)
        self.assertEqual(t17["snap2_studio_s48_dom_over_opp_ratio"], 8.5508)
        self.assertTrue(t17["snap2_studio_s48_top_tile_passes_q34_bar"])
        self.assertFalse(t17["snap2_studio_s48_global_upgraded_passes_q34_bar"])
        self.assertTrue(t17["pherc0306b_same_face_direction_between_families"])
        self.assertEqual(t17["pherc0306b_ahn_dom_over_opp_ratio"], 2.8031)
        self.assertEqual(len(self.summary["artifacts"]["receipts"]), 20)

    def test_04_rtx4060_ledger_sum_and_followup_consistency(self) -> None:
        led = self.summary["pillar3_fp_suppression_invariance_and_ledger"]["rtx4060_screen_ledger"]
        self.assertEqual(led["total_unique_valid_ct_cm2"], 162.5247)
        self.assertEqual(led["meshes_screened"], 34)
        self.assertEqual(led["two_family_followups"], 7)
        self.assertEqual(led["candidates_found"], 0)
        self.assertEqual(led["hardware_or_oom_incidents"], 0)
        self.assertAlmostEqual(sum(led["per_scroll_valid_ct_cm2"].values()), 162.5247, places=4)
        self.assertEqual(sum(led["per_scroll_mesh_counts"].values()), 34)

    def test_05_validate_dossier_markdown_passes_on_committed_file(self) -> None:
        self.assertTrue(DOSSIER_MD_PATH.is_file())
        V.validate_dossier_markdown(DOSSIER_MD_PATH, self.summary)

    def test_06_validate_dossier_markdown_rejects_privacy_leaks_and_missing_metrics(self) -> None:
        clean_md = DOSSIER_MD_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad_dossier.md"

            # 1. Privacy leaks (local paths, tokens, unread crop coordinates)
            for leak in (
                "C:\\Users\\someone\\scroll.zarr",
                "/Users/someone/scroll.zarr",
                "/home/someone/scroll.zarr",
                "/tmp/hecate_crops",
                "hf_" + "A" * 28,
                "top_hotspots",
                "r12-56",
                "c178-255",
            ):
                bad_path.write_text(clean_md + f"\nLeak: {leak}\n", encoding="utf-8")
                with self.assertRaises(ValueError, msg=f"Failed to reject leak {leak!r}"):
                    V.validate_dossier_markdown(bad_path, self.summary)

            # 2. Missing required numerical token
            corrupted = clean_md.replace("162.5247", "99.9999")
            bad_path.write_text(corrupted, encoding="utf-8")
            with self.assertRaises(ValueError):
                V.validate_dossier_markdown(bad_path, self.summary)

    def test_07_lf_patch_verifier_rejects_crlf_and_bad_base_commit(self) -> None:
        studio_patch = REPO_ROOT / "gemini_lane" / "progress_prize_fp16" / "0001-studio-upstream-fp16.patch"
        raw = studio_patch.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            crlf_patch = Path(tmp) / "crlf.patch"
            crlf_patch.write_bytes(raw.replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(ValueError, "CRLF"):
                V.verify_lf_clean_file(crlf_patch)

            good_copy = Path(tmp) / "good.patch"
            good_copy.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Base commit"):
                V.verify_unified_patch(
                    good_copy,
                    expected_base_commit="0" * 40,
                    expected_files={"README.md"},
                )

    def test_08_cli_main_round_trip_and_json_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "summary.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = V.main(["--repo-root", str(REPO_ROOT), "--json-out", str(out_json)])
            self.assertEqual(rc, 0)
            brief = json.loads(buf.getvalue())
            self.assertTrue(brief["cpu_only"])
            self.assertFalse(brief["torch_imported"])
            self.assertTrue(brief["dossier_markdown_valid"])
            self.assertEqual(brief["rtx4060_screened_cm2"], 162.5247)
            self.assertEqual(brief["receipts_verified"], 20)
            self.assertTrue(out_json.is_file())
            exported = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(exported["schema"], V.SCHEMA_VERSION)

    def test_09_ast_and_subprocess_zero_torch_or_gpu_imports(self) -> None:
        for path in (VERIFIER_PATH, TEST_PATH):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            for mod in imported:
                self.assertFalse(
                    any(mod == g or mod.startswith(g + ".") for g in GPU_MODULES),
                    f"{path.name} imports forbidden GPU module {mod!r}",
                )

        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import progress_prize_nov2026_verifier as v; "
            "s = v.build_consolidated_dossier_summary(); "
            f"print([m for m in {GPU_MODULES!r} if m in sys.modules])"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        proc = subprocess.run(
            [sys.executable, "-c", code, str(HERE)],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
            check=True,
        )
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "[]")


if __name__ == "__main__":
    unittest.main()
