"""Offline CPU-only unit tests for prepare_hecate_parity_crops.py."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
GEMINI_LANE = HERE.parent
for _p in (str(HERE), str(GEMINI_LANE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prepare_hecate_parity_crops as prep  # noqa: E402
import leak_guard  # noqa: E402  (shared public-repo privacy guard)
import stream_snap_render as ssr  # noqa: E402
import verify_upstream_gpu_burst as burst  # noqa: E402

FORBIDDEN_LOCAL_PATH = leak_guard.LOCAL_PATH   # always rejected, for every account name including "root"


def _write_local_zarr_v2(
    root: Path,
    arr: np.ndarray,
    *,
    chunks: tuple[int, int, int] = (32, 64, 64),
    level: str = "0",
    dimension_separator: str = "/",
    sparse_zero_skip: bool = True,
) -> Path:
    """Create a synthetic local Zarr v2 store for offline testing."""
    data = np.ascontiguousarray(arr)
    arr_dir = (root / level) if level else root
    arr_dir.mkdir(parents=True, exist_ok=True)
    if level:
        (root / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    meta = {
        "zarr_format": 2,
        "shape": list(data.shape),
        "chunks": list(chunks),
        "dtype": data.dtype.str,
        "compressor": None,
        "filters": None,
        "fill_value": 0,
        "order": "C",
        "dimension_separator": dimension_separator,
    }
    (arr_dir / ".zarray").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    sz, sy, sx = data.shape
    cz, cy, cx = chunks
    for iz in range((sz + cz - 1) // cz):
        for iy in range((sy + cy - 1) // cy):
            for ix in range((sx + cx - 1) // cx):
                block = np.zeros(chunks, dtype=data.dtype)
                z0, z1 = iz * cz, min((iz + 1) * cz, sz)
                y0, y1 = iy * cy, min((iy + 1) * cy, sy)
                x0, x1 = ix * cx, min((ix + 1) * cx, sx)
                block[: z1 - z0, : y1 - y0, : x1 - x0] = data[z0:z1, y0:y1, x0:x1]
                if sparse_zero_skip and not np.any(block):
                    continue
                if dimension_separator == "/":
                    cpath = arr_dir / str(iz) / str(iy) / str(ix)
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                else:
                    cpath = arr_dir / f"{iz}{dimension_separator}{iy}{dimension_separator}{ix}"
                cpath.write_bytes(block.tobytes(order="C"))
    return root


def _no_network(*_a, **_k):
    raise AssertionError("Offline tests must never perform real network calls")


class OfflineBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="prep-hecate-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p_url = mock.patch("urllib.request.urlopen", side_effect=_no_network)
        p_url.start()
        self.addCleanup(p_url.stop)
        if ssr.requests is not None:
            p_req = mock.patch.object(ssr.requests.Session, "get", side_effect=_no_network)
            p_req.start()
            self.addCleanup(p_req.stop)

    def _make_structured_volume(
        self,
        shape: tuple[int, int, int] = (96, 192, 192),
        *,
        seed: int = 0,
        zero_center: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vol = rng.integers(30, 220, size=shape, dtype=np.uint8)
        if zero_center:
            vol[:, 64:128, 64:128] = 0
        return vol


class RepoWriteGuardTests(OfflineBase):
    def test_refuses_current_repo_and_worktree_paths(self) -> None:
        self.assertTrue(prep.is_inside_git_repo(HERE))
        self.assertTrue(prep.is_inside_git_repo(HERE / "nested_uncreated_dir"))
        with self.assertRaisesRegex(ValueError, "strictly outside any git"):
            prep.ensure_outside_git_repo(HERE / "crops_out")

    def test_refuses_synthetic_git_repo_and_git_worktree_file(self) -> None:
        fake_repo = self.tmp / "fake_repo"
        (fake_repo / ".git").mkdir(parents=True)
        self.assertTrue(prep.is_inside_git_repo(fake_repo))
        self.assertTrue(prep.is_inside_git_repo(fake_repo / "sub" / "crops"))

        fake_wt = self.tmp / "fake_worktree"
        fake_wt.mkdir(parents=True)
        (fake_wt / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8")
        self.assertTrue(prep.is_inside_git_repo(fake_wt))
        self.assertTrue(prep.is_inside_git_repo(fake_wt / "out"))

        safe_dir = self.tmp / "external_crops"
        self.assertFalse(prep.is_inside_git_repo(safe_dir))
        self.assertEqual(prep.ensure_outside_git_repo(safe_dir), safe_dir.resolve())

    def test_cli_refuses_out_dir_inside_repo_even_on_dry_run(self) -> None:
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                prep.main(["--out-dir", str(HERE / "bad_out"), "--dry-run"])
        self.assertNotEqual(ctx.exception.code, 0)


class LocalZarrCroppingTests(OfflineBase):
    def test_local_zarr_cropping_and_burst_cli_compatibility(self) -> None:
        z96 = _write_local_zarr_v2(
            self.tmp / "vol_9p6.zarr",
            self._make_structured_volume((96, 192, 192), seed=1),
            chunks=(32, 64, 64),
        )
        z24 = _write_local_zarr_v2(
            self.tmp / "vol_2p4.zarr",
            self._make_structured_volume((96, 192, 192), seed=2),
            chunks=(32, 64, 64),
        )
        out_dir = self.tmp / "crops_out"
        receipt_path = self.tmp / "prep_receipt.json"

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = prep.main([
                "--out-dir", str(out_dir),
                "--source-9p6", str(z96),
                "--source-2p4", str(z24),
                "--depth", "64",
                "--size-yx", "128",
                "--min-nonzero-frac", "0.2",
                "--receipt", str(receipt_path),
            ])
        self.assertEqual(rc, 0)

        crop_96_path = out_dir / prep.CROP_9P6_NAME
        crop_24_path = out_dir / prep.CROP_2P4_NAME
        self.assertTrue(crop_96_path.is_file())
        self.assertTrue(crop_24_path.is_file())
        self.assertTrue(receipt_path.is_file())

        for key, path in (("9.6", crop_96_path), ("2.4", crop_24_path)):
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            self.assertEqual(arr.shape, (64, 128, 128))
            self.assertEqual(arr.dtype, np.uint8)
            self.assertGreater(float((arr > 0).mean()), 0.2)
            self.assertGreater(float(arr.std()), 0.0)
            del arr

        receipt_text = receipt_path.read_text(encoding="utf-8")
        self.assertEqual(receipt_text, stdout_buf.getvalue())
        receipt = json.loads(receipt_text)
        self.assertFalse(receipt["dry_run"])
        self.assertEqual(receipt["crop_shape"], [64, 128, 128])
        self.assertEqual(receipt["crops"]["9.6"]["source"], "vol_9p6.zarr")
        self.assertEqual(receipt["crops"]["2.4"]["source"], "vol_2p4.zarr")
        for key in ("9.6", "2.4"):
            self.assertNotIn("origin", receipt["crops"][key])
            self.assertNotIn("bbox", receipt["crops"][key])
            self.assertNotIn("coords", receipt["crops"][key])

        # Verify receipt sanitization: no local paths, no tmp dir, no username.
        self.assertNotIn(str(self.tmp), receipt_text)
        self.assertNotIn(str(out_dir), receipt_text)
        self.assertIsNone(FORBIDDEN_LOCAL_PATH.search(receipt_text))
        self.assertEqual(leak_guard.find_user_leaks(receipt_text, Path.home().name), [])
        # Same check under a generic "root" account: the word alone is fine, identifying contexts are not.
        self.assertEqual(leak_guard.find_user_leaks(receipt_text, "root"), [])

        # Verify direct compatibility with verify_upstream_gpu_burst.py CLI + dry-run.
        burst_args = burst.build_parser().parse_args([
            "--stages", "studio,hecate",
            "--hecate-input-9p6", str(crop_96_path),
            "--hecate-input-2p4", str(crop_24_path),
            "--output", str(self.tmp / "burst_receipt.json"),
            "--dry-run",
        ])
        self.assertEqual(burst.validate(burst_args), [])
        burst_out = io.StringIO()
        with redirect_stdout(burst_out):
            burst_rc = burst.main([
                "--stages", "studio,hecate",
                "--hecate-input-9p6", str(crop_96_path),
                "--hecate-input-2p4", str(crop_24_path),
                "--output", str(self.tmp / "burst_receipt.json"),
                "--dry-run",
            ])
        self.assertEqual(burst_rc, 0)
        plan = json.loads(burst_out.getvalue())
        hecate_tags = [r["tag"] for r in plan["runs"] if r["script"] == "hecate.py"]
        self.assertEqual(len(hecate_tags), 8)
        self.assertTrue(any("2p4_fwd_fp16" in t for t in hecate_tags))
        self.assertTrue(any("2p4_rev_fp16" in t for t in hecate_tags))
        self.assertTrue(any("9p6_fwd_fp16" in t for t in hecate_tags))
        self.assertTrue(any("9p6_rev_fp16" in t for t in hecate_tags))

    def test_nonzero_threshold_skips_empty_center_and_rejects_all_zero_volume(self) -> None:
        # Volume where only a non-center quadrant is populated.
        sparse = np.zeros((64, 192, 192), dtype=np.uint8)
        rng = np.random.default_rng(7)
        sparse[:, 128:192, 64:128] = rng.integers(50, 200, size=(64, 64, 64), dtype=np.uint8)
        z_sparse = _write_local_zarr_v2(self.tmp / "sparse.zarr", sparse, chunks=(32, 64, 64))
        with ssr.RemoteZarrV2(z_sparse, level="0", workers=1) as reader:
            crop, stats = prep.select_and_stream_crop(reader, (64, 64, 64), min_nonzero_frac=0.5)
        self.assertEqual(crop.shape, (64, 64, 64))
        self.assertGreater(stats["non_zero_frac"], 0.9)

        # All-zero volume must fail the non-zero structural verification.
        z_empty = _write_local_zarr_v2(
            self.tmp / "empty.zarr", np.zeros((64, 128, 128), dtype=np.uint8), chunks=(32, 64, 64)
        )
        with ssr.RemoteZarrV2(z_empty, level="0", workers=1) as reader:
            with self.assertRaisesRegex(RuntimeError, "non_zero_frac"):
                prep.select_and_stream_crop(reader, (64, 64, 64), min_nonzero_frac=0.2)

    def test_dry_run_makes_no_writes_and_sanitizes_urls(self) -> None:
        out_dir = self.tmp / "dry_out"
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            rc = prep.main([
                "--out-dir", str(out_dir),
                "--source-9p6", "https://user:secret@example.org/vol96.zarr?token=abc",
                "--source-2p4", str(self.tmp / "secret_dir" / "vol24.zarr"),
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        self.assertFalse(out_dir.exists())
        text = stdout_buf.getvalue()
        plan = json.loads(text)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["crops"]["9.6"]["source"], "https://example.org/vol96.zarr")
        self.assertEqual(plan["crops"]["2.4"]["source"], "vol24.zarr")
        self.assertNotIn("secret", text)
        self.assertNotIn("token", text)
        self.assertNotIn(str(self.tmp), text)


class RemoteZRangeTests(OfflineBase):
    def test_remote_uncompressed_z_range_reads_partial_chunks_under_budget(self) -> None:
        """Verify HTTP 206 partial Z-range reads cut transfer in half on uncompressed C-order Zarr."""
        chunk_shape = (128, 128, 128)
        meta = {
            "zarr_format": 2,
            "shape": [256, 256, 256],
            "chunks": list(chunk_shape),
            "dtype": "|u1",
            "compressor": None,
            "filters": None,
            "fill_value": 0,
            "order": "C",
            "dimension_separator": "/",
        }
        rng = np.random.default_rng(42)
        chunk_arr = rng.integers(20, 200, size=chunk_shape, dtype=np.uint8)
        chunk_raw = chunk_arr.tobytes(order="C")
        range_headers_seen: list[str] = []

        class _FakeResp:
            def __init__(self, status_code: int, content: bytes) -> None:
                self.status_code = status_code
                self.content = content

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise ssr.requests.HTTPError(f"HTTP {self.status_code}", response=self)

        def fake_get(_self, url: str, headers=None, timeout=None):
            if url.endswith(".zarray"):
                return _FakeResp(200, json.dumps(meta).encode("utf-8"))
            if headers and "Range" in headers:
                hdr = headers["Range"]
                range_headers_seen.append(hdr)
                span = hdr.split("=", 1)[1]
                s0, s1 = (int(v) for v in span.split("-"))
                return _FakeResp(206, chunk_raw[s0 : s1 + 1])
            return _FakeResp(200, chunk_raw)

        out_dir = self.tmp / "remote_out"
        if ssr.requests is None:
            self.skipTest("requests not installed")
        with mock.patch.object(ssr.requests.Session, "get", new=fake_get):
            summary = prep.prepare_crops(
                out_dir,
                source_9p6="https://example.org/scrollA.zarr",
                source_2p4="https://example.org/scrollB.zarr",
                crop_shape=(64, 256, 256),
                min_nonzero_frac=0.2,
                max_total_mb=15.0,
                workers=1,
            )

        self.assertTrue(range_headers_seen)
        # Each 64x256x256 crop uses 4-slice probe (64 KiB) + 4 x 64-slice half-chunks (4 MiB) = ~4.26 MB
        # Total across BOTH crops must be well under 15 MB.
        self.assertLess(summary["total_bytes_fetched"], 10_000_000)
        self.assertEqual(summary["crops"]["9.6"]["shape"], [64, 256, 256])
        self.assertEqual(summary["crops"]["2.4"]["shape"], [64, 256, 256])
        self.assertGreater(summary["crops"]["9.6"]["non_zero_frac"], 0.2)
        self.assertGreater(summary["crops"]["2.4"]["non_zero_frac"], 0.2)

    def test_remote_range_403_raises_and_budget_cap_is_enforced(self) -> None:
        if ssr.requests is None:
            self.skipTest("requests not installed")
        chunk_shape = (128, 128, 128)
        meta = {
            "zarr_format": 2,
            "shape": [256, 256, 256],
            "chunks": list(chunk_shape),
            "dtype": "|u1",
            "compressor": None,
            "filters": None,
            "fill_value": 0,
            "order": "C",
            "dimension_separator": "/",
        }

        class _FakeResp:
            def __init__(self, status_code: int, content: bytes = b"") -> None:
                self.status_code = status_code
                self.content = content

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise ssr.requests.HTTPError(f"HTTP {self.status_code}", response=self)

        def fake_403(_self, url: str, headers=None, timeout=None):
            if url.endswith(".zarray"):
                return _FakeResp(200, json.dumps(meta).encode("utf-8"))
            return _FakeResp(403, b"denied")

        with mock.patch.object(ssr.requests.Session, "get", new=fake_403):
            with self.assertRaises(ssr.requests.HTTPError):
                prep.prepare_crops(
                    self.tmp / "out_403",
                    source_9p6="https://example.org/a.zarr",
                    source_2p4="https://example.org/b.zarr",
                    crop_shape=(64, 256, 256),
                    workers=1,
                )

        # Verify tight transfer budget cleans up and raises.
        chunk_raw = np.full(chunk_shape, 100, dtype=np.uint8).tobytes(order="C")
        chunk_raw_varied = np.arange(int(np.prod(chunk_shape)), dtype=np.uint8).tobytes(order="C")

        def fake_206(_self, url: str, headers=None, timeout=None):
            if url.endswith(".zarray"):
                return _FakeResp(200, json.dumps(meta).encode("utf-8"))
            span = headers["Range"].split("=", 1)[1]
            s0, s1 = (int(v) for v in span.split("-"))
            return _FakeResp(206, chunk_raw_varied[s0 : s1 + 1])

        out_tight = self.tmp / "out_tight"
        with mock.patch.object(ssr.requests.Session, "get", new=fake_206):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                prep.prepare_crops(
                    out_tight,
                    source_9p6="https://example.org/a.zarr",
                    source_2p4="https://example.org/b.zarr",
                    crop_shape=(64, 256, 256),
                    max_total_mb=1.0,
                    workers=1,
                )
        self.assertFalse((out_tight / prep.CROP_9P6_NAME).exists())
        self.assertFalse((out_tight / prep.CROP_2P4_NAME).exists())
        del chunk_raw


class LeakGuardRootAccountTests(unittest.TestCase):
    """The public-repo guard must work on a machine whose account is "root" (an ordinary word in this kit)."""

    def test_kit_sources_and_dry_run_receipt_pass_for_root_account(self):
        texts = [p.read_text(encoding="utf-8") for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.md"))]
        self.assertTrue(any("hecate_root" in t or "repo_root" in t or "root" in t for t in texts))
        for text in texts:
            self.assertIsNone(FORBIDDEN_LOCAL_PATH.search(text))
            self.assertEqual(leak_guard.find_user_leaks(text, "root"), [])

    def test_root_account_still_rejects_paths_and_credentials(self):
        bad = ["/" + "root" + "/secret", "C:" + chr(92) + "Users" + chr(92) + "alice" + chr(92) + "x",
               "/" + "home/alice/x", "/" + "Users/alice/x"]
        for text in bad:
            with self.subTest(text=text):
                self.assertIsNotNone(FORBIDDEN_LOCAL_PATH.search(text))
        self.assertTrue(leak_guard.find_user_leaks("scp x root" + "@box:/srv", "root"))
        self.assertEqual(leak_guard.find_user_leaks("repo_root and hecate_root", "root"), [])


if __name__ == "__main__":
    unittest.main()
