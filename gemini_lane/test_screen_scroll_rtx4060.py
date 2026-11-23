# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline unit tests for screen_scroll_rtx4060.py safety gates and helpers."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

try:
    import tifffile as _tifffile
except ImportError:
    _tifffile = None

import screen_scroll_rtx4060 as S


def _write_f32_tiff(path: Path, arr: np.ndarray) -> None:
    arr32 = np.asarray(arr, dtype=np.float32)
    if _tifffile is not None:
        _tifffile.imwrite(path, arr32)
    else:
        Image.fromarray(arr32, mode="F").save(str(path), format="TIFF")


def _write_local_zarr_v2(root: Path, arr: np.ndarray, chunks: tuple[int, int, int]) -> None:
    level_dir = root / "0"
    level_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 2,
        "shape": list(arr.shape),
        "chunks": list(chunks),
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "dimension_separator": ".",
    }
    (level_dir / ".zarray").write_text(json.dumps(meta), encoding="utf-8")
    cz, cy, cx = chunks
    for iz in range(0, arr.shape[0], cz):
        for iy in range(0, arr.shape[1], cy):
            for ix in range(0, arr.shape[2], cx):
                blk = np.zeros(chunks, dtype=arr.dtype)
                sub = arr[iz : iz + cz, iy : iy + cy, ix : ix + cx]
                blk[: sub.shape[0], : sub.shape[1], : sub.shape[2]] = sub
                key = f"{iz // cz}.{iy // cy}.{ix // cx}"
                (level_dir / key).write_bytes(blk.tobytes(order="C"))


class TestScreenScrollSafety(unittest.TestCase):
    def test_mesh_dir_tag_strips_url_scheme_and_long_paths(self) -> None:
        url = "https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/abc/submission/mesh/PHerc0306B_w074_V5Rc"
        self.assertEqual(S._mesh_dir_tag(url), "PHerc0306B_w074_V5Rc")
        self.assertEqual(S._mesh_dir_tag("https://example.org/a/b/z10544_w035/"), "z10544_w035")

    def test_refuses_output_inside_git_repo(self) -> None:
        inside = S.REPO_ROOT / "gemini_lane" / "evidence" / "forbidden_private"
        with self.assertRaises(ValueError):
            S.ensure_outside_git_repo(inside)

    def test_allows_output_outside_git_repo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scrollprize-private-test-") as tmp:
            resolved = S.ensure_outside_git_repo(Path(tmp))
            self.assertTrue(resolved.exists())

    def test_find_top_hotspots_and_visuals(self) -> None:
        prob = np.full((600, 700), 0.05, dtype=np.float32)
        valid = np.ones((600, 700), dtype=bool)
        prob[40:300, 100:360] = 0.85
        hotspots = S.find_top_hotspots(prob, valid, win=256, top_k=2)
        self.assertGreaterEqual(len(hotspots), 1)
        self.assertGreater(hotspots[0]["hot_frac_ge045"], 0.4)

        with tempfile.TemporaryDirectory(prefix="scrollprize-vis-test-") as tmp:
            out_dir = Path(tmp)
            ct = np.full((600, 700), 120, dtype=np.uint8)
            mname, hname = S.write_visuals(
                out_dir, "TEST/m0", ct, valid, [("Studio fp16 desc", prob, valid)], hotspots, 9.362
            )
            self.assertTrue((out_dir / mname).is_file())
            self.assertIsNotNone(hname)
            self.assertTrue((out_dir / hname).is_file())

    def test_screen_one_mesh_end_to_end_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scrollprize-e2e-test-") as tmp:
            root = Path(tmp)
            ct_zarr = root / "ct.zarr"
            m7_zarr = root / "m7.zarr"
            mesh_dir = root / "mesh_dir"
            work_root = root / "work"
            out_dir = root / "out_private"
            work_root.mkdir()
            out_dir.mkdir()

            # Synthetic CT: z in [0..63], sheet plane at z = 32.0 + 0.01 * y
            zz, yy, xx = np.meshgrid(
                np.arange(64, dtype=np.float32),
                np.arange(320, dtype=np.float32),
                np.arange(320, dtype=np.float32),
                indexing="ij",
            )
            z_sheet = 32.0 + 0.01 * yy
            ct = np.clip(90.0 + 110.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.8) ** 2) + 8.0 * np.sin(xx * 0.35), 20, 240).astype(np.uint8)
            m7 = np.clip(255.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.2) ** 2), 0, 255).astype(np.uint8)
            _write_local_zarr_v2(ct_zarr, ct, chunks=(32, 160, 160))
            _write_local_zarr_v2(m7_zarr, m7, chunks=(32, 160, 160))

            # 27x27 mesh at scale 0.1 -> 261x261 upsampled stack (>= 256 tile for onsheet_score)
            gy, gx = np.meshgrid(
                np.linspace(16.0, 286.0, 27, dtype=np.float32),
                np.linspace(16.0, 286.0, 27, dtype=np.float32),
                indexing="ij",
            )
            gz = (30.5 + 0.01 * gy).astype(np.float32)  # Offset by ~1.5 vx to test m7 snap
            mesh_dir.mkdir()
            _write_f32_tiff(mesh_dir / "x.tif", gx)
            _write_f32_tiff(mesh_dir / "y.tif", gy)
            _write_f32_tiff(mesh_dir / "z.tif", gz)
            (mesh_dir / "meta.json").write_text(json.dumps({"scale": [0.1, 0.1]}), encoding="utf-8")

            def _fake_studio(
                stack_lhw,
                mesh_idx,
                staged,
                *,
                stride,
                batch_size,
                batch_pause_ms,
                max_gpu_temp_c,
            ):
                self.assertEqual(list(work_root.rglob("stack_lhw.npy")), [])
                h, w = stack_lhw.shape[1], stack_lhw.shape[2]
                prob = np.full((h, w), 0.12, dtype=np.float32)
                prob[20:100, 20:100] = 0.88
                val = np.ones((h, w), dtype=bool)
                return {
                    "desc": {"prob": prob, "valid": val, "seconds": 0.1, "peak_allocated_mib": 100, "peak_reserved_mib": 120},
                    "asc": {"prob": prob * 0.9, "valid": val, "seconds": 0.1, "peak_allocated_mib": 100, "peak_reserved_mib": 120},
                }

            def _fake_ahn(
                stack_lhw,
                mesh_idx,
                staged,
                *,
                batch_size,
                batch_pause_ms,
                max_gpu_temp_c,
            ):
                h, w = stack_lhw.shape[1], stack_lhw.shape[2]
                prob = np.full((h, w), 0.08, dtype=np.float32)
                prob[20:100, 20:100] = 0.82
                val = np.ones((h, w), dtype=bool)
                return {
                    "asc": {"prob": prob, "valid": val, "tiles": 4, "seconds": 0.1, "peak_allocated_mib": 90, "peak_reserved_mib": 110},
                    "desc": {"prob": prob * 0.85, "valid": val, "tiles": 4, "seconds": 0.1, "peak_allocated_mib": 90, "peak_reserved_mib": 110},
                }

            with (
                mock.patch.object(S, "run_studio_both_orders", side_effect=_fake_studio),
                mock.patch.object(S, "run_ahn_both_orders", side_effect=_fake_ahn),
            ):
                summary = S.screen_one_mesh(
                    scroll="TEST",
                    mesh_source=str(mesh_dir),
                    ct_zarr_url=str(ct_zarr),
                    m7_zarr_url=str(m7_zarr),
                    snap_passes=2,
                    num_slices=17,
                    uv_crop=None,
                    voxel_um=9.362,
                    detectors={"studio", "ahn"},
                    studio_stride=96,
                    studio_batch=1,
                    ahn_batch=1,
                    batch_pause_ms=150,
                    max_gpu_temp_c=75,
                    cache_mb=64,
                    staged={},
                    work_root=work_root,
                    out_private_dir=out_dir,
                )

            self.assertGreater(summary["m7_chunks_fetched"], 0)
            self.assertGreater(summary["ct_chunks_fetched"], 0)
            self.assertGreater(summary["geometry_snap_metrics"]["on_sheet_after_snap"], 0.9)
            self.assertIn("f_parallel", summary["onsheet_coherence"])
            self.assertIn("studio_desc", summary["detectors"])
            self.assertIn("ahn_s0_asc", summary["detectors"])
            fp = summary["fp_diagnostics"]
            self.assertIsInstance(fp, dict)
            self.assertIn(fp["dominant_face"], ("asc", "desc"))
            self.assertIsNotNone(fp["two_family"])
            self.assertIn("ink_like_candidate", fp)
            self.assertAlmostEqual(fp["clean_papyrus_area_frac"] + fp["crack_halo_area_frac"], 1.0, places=5)
            mesh_out = out_dir / "TEST_mesh_dir"
            self.assertTrue((mesh_out / "montage.png").is_file())
            self.assertTrue((mesh_out / "hotspots.png").is_file())
            self.assertTrue((mesh_out / "summary.json").is_file())
            on_disk = json.loads((mesh_out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["fp_diagnostics"], fp)
            self.assertEqual(list(work_root.iterdir()), [])

    def test_compute_fp_diagnostics_studio_only_flags_crack_halo(self) -> None:
        rng = np.random.default_rng(0)
        ct = (0.6 + 0.02 * rng.standard_normal((200, 200))).astype(np.float32)
        ct[90:110, 20:180] = 0.1  # delamination crack
        valid = np.ones((200, 200), dtype=bool)
        prob = np.full((200, 200), 0.05, dtype=np.float32)
        prob[85:115, 30:170] = 0.9  # crack-wall speck band, both depth orders
        studio = {o: {"prob": prob, "valid": valid} for o in ("asc", "desc")}
        fp = S.compute_fp_diagnostics(ct, valid, studio, None, 9.362)
        self.assertIsNone(fp["two_family"])
        self.assertGreater(fp["crack_halo_area_frac"], 0.0)
        self.assertTrue(fp["flags"]["crack_halo_dominated"] or fp["flags"]["no_hits"])
        self.assertFalse(fp["ink_like_candidate"])
        json.dumps(fp)

    def test_query_gpu_telemetry_raises_on_lost_gpu(self) -> None:
        fake_proc = mock.Mock(
            returncode=1,
            stdout="Unable to determine the device handle for GPU0: 0000:01:00.0: GPU is lost.  Reboot the system to recover this GPU\n",
            stderr="",
        )
        with mock.patch.object(S.subprocess, "run", return_value=fake_proc):
            with self.assertRaises(RuntimeError):
                S.query_gpu_telemetry()

        good_proc = mock.Mock(returncode=0, stdout="62, 12, 420, 8188\n", stderr="")
        with mock.patch.object(S.subprocess, "run", return_value=good_proc):
            info = S.query_gpu_telemetry()
            self.assertEqual(info, {"temp_c": 62, "util_pct": 12, "mem_used_mib": 420, "mem_total_mib": 8188})

    def test_enforce_gpu_guardrail_cooldown_and_timeout(self) -> None:
        # Below threshold returns immediately without sleeping.
        with mock.patch.object(S.time, "sleep") as sleep_mock:
            out = S.enforce_gpu_guardrail(
                max_temp_c=75,
                telemetry_fn=lambda: {"temp_c": 64, "util_pct": 10, "mem_used_mib": 300, "mem_total_mib": 8188},
            )
            self.assertEqual(out["temp_c"], 64)
            sleep_mock.assert_not_called()

        # Hits threshold -> waits in cooldown steps until cooled by >= 6C (<= 69C).
        readings = iter([
            {"temp_c": 76, "util_pct": 85, "mem_used_mib": 1200, "mem_total_mib": 8188},
            {"temp_c": 72, "util_pct": 0, "mem_used_mib": 1200, "mem_total_mib": 8188},
            {"temp_c": 68, "util_pct": 0, "mem_used_mib": 1200, "mem_total_mib": 8188},
        ])
        with mock.patch.object(S.time, "sleep") as sleep_mock:
            out = S.enforce_gpu_guardrail(
                max_temp_c=75,
                cooldown_s=4.0,
                max_wait_s=20.0,
                telemetry_fn=lambda: next(readings),
            )
            self.assertEqual(out["temp_c"], 68)
            self.assertEqual(sleep_mock.call_count, 2)

        # Remains hot past max_wait_s -> raises RuntimeError.
        with mock.patch.object(S.time, "sleep"):
            with self.assertRaises(RuntimeError):
                S.enforce_gpu_guardrail(
                    max_temp_c=75,
                    cooldown_s=4.0,
                    max_wait_s=8.0,
                    telemetry_fn=lambda: {"temp_c": 80, "util_pct": 90, "mem_used_mib": 1200, "mem_total_mib": 8188},
                )

    def test_cli_defaults_are_conservative_for_laptop_gpu(self) -> None:
        args = S.build_arg_parser().parse_args(
            [
                "--scroll",
                "PHerc0125",
                "--mesh",
                "https://example.org/m",
                "--ct-zarr",
                "https://example.org/ct.zarr",
                "--voxel-um",
                "9.362",
                "--out-private-dir",
                "out_priv",
            ]
        )
        self.assertEqual(args.vram_fraction, 0.45)
        self.assertEqual(args.studio_batch, 1)
        self.assertEqual(args.ahn_batch, 1)
        self.assertEqual(args.studio_stride, 96)
        self.assertEqual(args.batch_pause_ms, 150)
        self.assertEqual(args.max_gpu_temp_c, 75)
        args_none = S.build_arg_parser().parse_args(
            [
                "--scroll",
                "PHerc0125",
                "--mesh",
                "m",
                "--ct-zarr",
                "c",
                "--voxel-um",
                "9.362",
                "--detectors",
                "none",
                "--out-private-dir",
                "out_priv",
            ]
        )
        self.assertEqual(args_none.detectors, "none")

    def test_screen_one_mesh_uv_crop_cpu_only_dir_naming(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scrollprize-crop-test-") as tmp:
            root = Path(tmp)
            ct_zarr = root / "ct.zarr"
            mesh_dir = root / "mesh_dir"
            work_root = root / "work"
            out_dir = root / "out_private"
            work_root.mkdir()
            out_dir.mkdir()

            zz, yy, _ = np.meshgrid(
                np.arange(48, dtype=np.float32),
                np.arange(160, dtype=np.float32),
                np.arange(160, dtype=np.float32),
                indexing="ij",
            )
            z_sheet = 24.0 + 0.01 * yy
            ct = np.clip(100.0 + 90.0 * np.exp(-0.5 * ((zz - z_sheet) / 1.5) ** 2), 15, 235).astype(np.uint8)
            _write_local_zarr_v2(ct_zarr, ct, chunks=(24, 80, 80))

            gy, gx = np.meshgrid(
                np.linspace(12.0, 148.0, 20, dtype=np.float32),
                np.linspace(12.0, 148.0, 20, dtype=np.float32),
                indexing="ij",
            )
            gz = (24.0 + 0.01 * gy).astype(np.float32)
            mesh_dir.mkdir()
            _write_f32_tiff(mesh_dir / "x.tif", gx)
            _write_f32_tiff(mesh_dir / "y.tif", gy)
            _write_f32_tiff(mesh_dir / "z.tif", gz)
            (mesh_dir / "meta.json").write_text(json.dumps({"scale": [0.1, 0.1]}), encoding="utf-8")

            summary = S.screen_one_mesh(
                scroll="TEST",
                mesh_source=str(mesh_dir),
                ct_zarr_url=str(ct_zarr),
                m7_zarr_url=None,
                snap_passes=0,
                num_slices=5,
                uv_crop=(0, 15, 0, 15),
                voxel_um=9.362,
                detectors=set(),
                studio_stride=96,
                studio_batch=1,
                ahn_batch=1,
                cache_mb=32,
                staged={},
                work_root=work_root,
                out_private_dir=out_dir,
            )

            self.assertEqual(summary["uv_crop"], [0, 15, 0, 15])
            self.assertEqual(summary["detectors"], {})
            self.assertIsNone(summary["fp_diagnostics"])
            crop_out = out_dir / "TEST_mesh_dir_r0-15_c0-15"
            self.assertTrue((crop_out / "summary.json").is_file())
            self.assertTrue((crop_out / "montage.png").is_file())
            self.assertEqual(list(work_root.iterdir()), [])

    def test_rtx4060_screen_ledger_sanitized(self) -> None:
        ledger_path = Path(__file__).resolve().parent / "evidence" / "firstletters_rtx4060_screen_ledger.json"
        raw_text = ledger_path.read_text(encoding="utf-8")
        forbidden_substrings = (
            "top_hotspots",
            "uv_crop",
            "C:\\Users",
            "C:/Users",
            "Scrollprize_private",
            "montage.png",
            "hotspots.png",
            "r0-56",
            "r12-56",
            "r118-164",
            "r68-112",
            "r120-164",
            "r10-56",
        )
        for forbidden in forbidden_substrings:
            self.assertNotIn(forbidden, raw_text)

        data = json.loads(raw_text)
        total_cm2 = round(sum(m["valid_ct_cm2"] for m in data["meshes"]), 4)
        self.assertEqual(total_cm2, data["untouched_screening_summary"]["total_unique_valid_ct_cm2"])
        self.assertEqual(total_cm2, 64.899)
        self.assertEqual(data["untouched_screening_summary"]["candidates_found"], 0)
        self.assertEqual(data["untouched_screening_summary"]["two_family_followups"], 5)
        self.assertEqual(data["untouched_screening_summary"]["meshes_screened"], 12)
        self.assertEqual(data["untouched_screening_summary"]["meshes_screened"], len(data["meshes"]))


if __name__ == "__main__":
    unittest.main()
