# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Offline CPU-only unit tests for gemini_lane/stream_snap_render.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stream_snap_render as ssr  # noqa: E402

FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"[A-Za-z]:/"),
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"\\Users\\[^\\]+\\", re.I),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"file://", re.I),
)


def _write_local_zarr_v2(
    root: Path,
    arr: np.ndarray,
    *,
    chunks: tuple[int, int, int] = (16, 16, 16),
    level: str | None = "0",
    dimension_separator: str = "/",
    compressor: dict | None = None,
    sparse_zero_skip: bool = False,
) -> Path:
    """Create a local Zarr v2 store (uncompressed or zlib) for offline testing."""
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
        "compressor": compressor,
        "filters": None,
        "fill_value": 0,
        "order": "C",
        "dimension_separator": dimension_separator,
    }
    (arr_dir / ".zarray").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    sz, sy, sx = data.shape
    cz, cy, cx = chunks
    nz = (sz + cz - 1) // cz
    ny = (sy + cy - 1) // cy
    nx = (sx + cx - 1) // cx

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                block = np.zeros(chunks, dtype=data.dtype)
                z0, z1 = iz * cz, min((iz + 1) * cz, sz)
                y0, y1 = iy * cy, min((iy + 1) * cy, sy)
                x0, x1 = ix * cx, min((ix + 1) * cx, sx)
                block[: z1 - z0, : y1 - y0, : x1 - x0] = data[z0:z1, y0:y1, x0:x1]
                if sparse_zero_skip and not np.any(block):
                    continue
                raw = block.tobytes(order="C")
                if compressor is not None:
                    assert compressor.get("id") == "zlib"
                    raw = zlib.compress(raw, level=int(compressor.get("level", 1)))
                if dimension_separator == "/":
                    cpath = arr_dir / str(iz) / str(iy) / str(ix)
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                else:
                    cpath = arr_dir / f"{iz}{dimension_separator}{iy}{dimension_separator}{ix}"
                cpath.write_bytes(raw)
    return root


def _curved_surface_z_and_grad(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth curved synthetic sheet z(x, y) and analytic partial derivatives."""
    ax = (x - 16.0) / 7.5
    ay = (y - 16.0) / 8.5
    z = 32.0 + 2.4 * np.sin(ax) + 1.8 * np.cos(ay)
    dz_dx = (2.4 / 7.5) * np.cos(ax)
    dz_dy = -(1.8 / 8.5) * np.sin(ay)
    return z, dz_dx, dz_dy


def _build_curved_sheet_fixture(shape: tuple[int, int, int] = (64, 64, 64)):
    """Build curved m7 volume + initial mesh offset by -3.0 vx along true normal."""
    sz, sy, sx = shape
    zz, yy, xx = np.meshgrid(
        np.arange(sz, dtype=np.float64),
        np.arange(sy, dtype=np.float64),
        np.arange(sx, dtype=np.float64),
        indexing="ij",
    )
    z_surf, dz_dx, dz_dy = _curved_surface_z_and_grad(xx, yy)
    inv_norm = 1.0 / np.sqrt(1.0 + dz_dx ** 2 + dz_dy ** 2)
    d_perp = (zz - z_surf) * inv_norm
    m7 = np.clip(np.rint(255.0 * np.exp(-0.5 * (d_perp / 0.85) ** 2)), 0, 255).astype(np.uint8)

    # Initial mesh sits at -3.0 vx along true normal (so m7 sheet is +3.0 vx along +N).
    u = np.arange(14, 50, dtype=np.float64)
    v = np.arange(14, 50, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # (H=36, W=36)
    zt, gx, gy = _curved_surface_z_and_grad(uu, vv)
    nlen = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    # Unit normal pointing toward +z: (-dz/dx, -dz/dy, 1) / norm
    nx = -gx / nlen
    ny = -gy / nlen
    nz = 1.0 / nlen

    offset_vx = 3.0
    x_raw = (uu - offset_vx * nx).astype(np.float32)
    y_raw = (vv - offset_vx * ny).astype(np.float32)
    z_raw = (zt - offset_vx * nz).astype(np.float32)
    return m7, x_raw, y_raw, z_raw


class _FakeResponse:
    def __init__(self, url: str, status_code: int, content: bytes = b"") -> None:
        self.url = url
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ssr.requests.HTTPError(f"HTTP {self.status_code} for {self.url}", response=self)


class _FakeSession:
    def __init__(self, meta: dict, chunk_status: int = 200, chunk_bytes: bytes = b"") -> None:
        self.meta = meta
        self.chunk_status = chunk_status
        self.chunk_bytes = chunk_bytes
        self.chunk_requests = 0

    def get(self, url: str, **kwargs) -> _FakeResponse:
        if url.endswith(".zarray"):
            return _FakeResponse(url, 200, json.dumps(self.meta).encode("utf-8"))
        self.chunk_requests += 1
        return _FakeResponse(url, self.chunk_status, self.chunk_bytes)


class TestStreamSnapRender(unittest.TestCase):
    def test_1_curved_sheet_m7_snap_recovers_true_surface(self) -> None:
        m7_vol, x_raw, y_raw, z_raw = _build_curved_sheet_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            zarr_dir = _write_local_zarr_v2(
                Path(tmp) / "m7.zarr", m7_vol, chunks=(16, 16, 16), compressor={"id": "zlib", "level": 1}
            )
            with ssr.RemoteZarrV2(zarr_dir, level="0", cache_mb=16, workers=2) as reader:
                sx, sy, sz, metrics = ssr.snap_tifxyz_to_m7(
                    x_raw, y_raw, z_raw, reader, passes=2, search_radius=12, step=1.0, threshold=51
                )

        # Evaluate perpendicular distance of every snapped vertex to the true curved sheet.
        z_true, gx, gy = _curved_surface_z_and_grad(sx.astype(np.float64), sy.astype(np.float64))
        d_perp = np.abs(sz.astype(np.float64) - z_true) / np.sqrt(1.0 + gx ** 2 + gy ** 2)

        self.assertEqual(metrics["valid_vertex_fraction"], 1.0)
        self.assertEqual(metrics["on_sheet_raw"], 0.0)
        self.assertGreaterEqual(metrics["on_sheet_after_snap"], 0.95)
        self.assertLessEqual(float(np.max(d_perp)), 0.5)
        self.assertLessEqual(float(np.median(d_perp)), 0.25)
        self.assertAlmostEqual(metrics["median_disp_vx"], 3.0, delta=0.5)
        for key in (
            "valid_vertex_fraction",
            "on_sheet_raw",
            "on_sheet_after_snap",
            "median_disp_vx",
            "p90_disp_vx",
            "center_vs_outer_disp_diff",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

    def test_2_sentinel_safety_three_channel_and_neighbor_normals(self) -> None:
        # Planar sheet z = 20.0, normal strictly (0, 0, +1).
        h, w = 16, 16
        uu, vv = np.meshgrid(
            np.arange(10, 10 + w, dtype=np.float32),
            np.arange(10, 10 + h, dtype=np.float32),
            indexing="xy",
        )
        zz = np.full((h, w), 20.0, dtype=np.float32)

        x_bad = uu.copy()
        y_bad = vv.copy()
        z_bad = zz.copy()
        # Inject single-channel sentinels at isolated interior vertices.
        sentinels = [(4, 4), (4, 10), (10, 4), (10, 10)]
        x_bad[4, 4] = -1.0
        y_bad[4, 10] = np.nan
        z_bad[10, 4] = -0.25
        z_bad[10, 10] = np.inf

        cx, cy, cz, valid = ssr.validate_tifxyz_channels(x_bad, y_bad, z_bad)
        for r, c in sentinels:
            self.assertFalse(bool(valid[r, c]))
            self.assertEqual(float(cx[r, c]), -1.0)
            self.assertEqual(float(cy[r, c]), -1.0)
            self.assertEqual(float(cz[r, c]), -1.0)

        nrm, nok = ssr.compute_grid_normals(x_bad, y_bad, z_bad)
        for r, c in sentinels:
            self.assertFalse(bool(nok[r, c]))
            np.testing.assert_allclose(nrm[r, c], [0.0, 0.0, 0.0], atol=1e-6)
            # All 4 adjacent valid neighbors must remain exact (0, 0, +1) with zero poisoning.
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                self.assertTrue(bool(nok[nr, nc]))
                np.testing.assert_allclose(nrm[nr, nc], [0.0, 0.0, 1.0], atol=1e-6)

        # Also verify snap_tifxyz_to_m7 preserves -1 across all 3 channels for sentinels.
        m7 = np.zeros((40, 40, 40), dtype=np.uint8)
        m7[22, :, :] = 255  # sheet at z=22 (+2 vx from planar z=20)
        with tempfile.TemporaryDirectory() as tmp:
            zdir = _write_local_zarr_v2(Path(tmp) / "m7_plane.zarr", m7, chunks=(16, 16, 16))
            with ssr.RemoteZarrV2(zdir, level="0", cache_mb=8, workers=1) as reader:
                sx, sy, sz, metrics = ssr.snap_tifxyz_to_m7(
                    x_bad, y_bad, z_bad, reader, passes=2, search_radius=6, step=1.0, threshold=51
                )

        self.assertEqual(metrics["valid_vertex_count"], h * w - len(sentinels))
        for r, c in sentinels:
            self.assertEqual(float(sx[r, c]), -1.0)
            self.assertEqual(float(sy[r, c]), -1.0)
            self.assertEqual(float(sz[r, c]), -1.0)
        # Adjacent valid neighbors snapped cleanly to z=22.
        for r, c in sentinels:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self.assertAlmostEqual(float(sz[r + dr, c + dc]), 22.0, delta=0.1)

    def test_3_remote_zarr_http_errors_and_lru_eviction(self) -> None:
        meta = {
            "zarr_format": 2,
            "shape": [4, 4, 4],
            "chunks": [4, 4, 4],
            "dtype": "|u1",
            "compressor": None,
            "filters": None,
            "fill_value": 0,
            "order": "C",
            "dimension_separator": "/",
        }
        expected_chunk = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)

        # 200 OK -> exact data and fetch accounting
        s200 = _FakeSession(meta, chunk_status=200, chunk_bytes=expected_chunk.tobytes())
        with mock.patch.object(ssr.requests, "Session", return_value=s200):
            with ssr.RemoteZarrV2("https://example.invalid/ct.zarr", workers=1, retry_backoff=0.0) as rz:
                block, org = rz.read_box((0, 0, 0), (4, 4, 4))
                self.assertEqual(org, (0, 0, 0))
                np.testing.assert_array_equal(block, expected_chunk)
                self.assertEqual(rz.chunks_fetched, 1)
                self.assertEqual(rz.bytes_fetched, 64)

        # 404 Not Found -> zero fill without raising
        s404 = _FakeSession(meta, chunk_status=404)
        with mock.patch.object(ssr.requests, "Session", return_value=s404):
            with ssr.RemoteZarrV2("https://example.invalid/ct.zarr", workers=1, retry_backoff=0.0) as rz:
                block, _ = rz.read_box((0, 0, 0), (4, 4, 4))
                np.testing.assert_array_equal(block, np.zeros((4, 4, 4), dtype=np.uint8))
                self.assertEqual(s404.chunk_requests, 1)

        # 403 Forbidden -> must raise HTTPError (also OSError), never return zeros
        s403 = _FakeSession(meta, chunk_status=403, chunk_bytes=b"AccessDenied")
        with mock.patch.object(ssr.requests, "Session", return_value=s403):
            with ssr.RemoteZarrV2("https://example.invalid/ct.zarr", workers=1, retry_backoff=0.0) as rz:
                with self.assertRaises((ssr.requests.HTTPError, OSError)) as ctx:
                    rz.read_box((0, 0, 0), (4, 4, 4))
                self.assertEqual(ctx.exception.response.status_code, 403)
                # 403 must fail fast without retrying.
                self.assertEqual(s403.chunk_requests, 1)

        # 500 Internal Server Error -> must raise HTTPError (also OSError), never return zeros
        s500 = _FakeSession(meta, chunk_status=500, chunk_bytes=b"ServerError")
        with mock.patch.object(ssr.requests, "Session", return_value=s500):
            with ssr.RemoteZarrV2(
                "https://example.invalid/ct.zarr", workers=1, max_retries=2, retry_backoff=0.0
            ) as rz:
                with self.assertRaises((ssr.requests.HTTPError, OSError)) as ctx:
                    rz.read_box((0, 0, 0), (4, 4, 4))
                self.assertEqual(ctx.exception.response.status_code, 500)
                self.assertEqual(s500.chunk_requests, 2)

        # Strict LRU cache cap eviction test: 8 chunks of 4096 B each, cap = 3 chunks (12288 B)
        vol = np.arange(32 * 32 * 32, dtype=np.uint8).reshape(32, 32, 32)
        chunk_bytes = 16 * 16 * 16
        cap_mb = (3 * chunk_bytes) / (1024 * 1024)
        with tempfile.TemporaryDirectory() as tmp:
            zdir = _write_local_zarr_v2(Path(tmp) / "lru.zarr", vol, chunks=(16, 16, 16))
            with ssr.RemoteZarrV2(zdir, level="0", cache_mb=cap_mb, workers=1) as rz:
                self.assertEqual(rz.cache_max, 3 * chunk_bytes)
                rz.get_chunk((0, 0, 0))
                rz.get_chunk((0, 0, 1))
                rz.get_chunk((0, 1, 0))
                self.assertEqual(rz.cache_bytes, 3 * chunk_bytes)
                self.assertEqual(rz.chunks_fetched, 3)

                # Touch (0, 0, 0) to make it MRU; (0, 0, 1) is now LRU.
                rz.get_chunk((0, 0, 0))
                self.assertEqual(rz.chunks_fetched, 3)
                self.assertEqual(rz.cache_hits, 1)

                # Fetching a 4th chunk must evict (0, 0, 1) and keep cache_bytes <= cache_max.
                rz.get_chunk((0, 1, 1))
                self.assertLessEqual(rz.cache_bytes, rz.cache_max)
                self.assertIn((0, 0, 0), rz.cache)
                self.assertNotIn((0, 0, 1), rz.cache)
                self.assertGreaterEqual(rz.cache_evictions, 1)

                # Reading the entire 8-chunk volume in one call must still respect cache_bytes <= cache_max.
                full, _ = rz.read_box((0, 0, 0), (32, 32, 32))
                np.testing.assert_array_equal(full, vol)
                self.assertLessEqual(rz.cache_bytes, rz.cache_max)

    def test_4_layer_coherence_sheet_parallel_vs_oblique_cut(self) -> None:
        rng = np.random.default_rng(2026)
        # Rich bandlimited papyrus-like texture with strong zero-shift autocorrelation peak.
        base = rng.normal(0.0, 1.0, size=(160, 160))
        base = ndi.gaussian_filter(base, sigma=1.2)
        base = 128.0 + 42.0 * (base / (base.std() + 1e-6))

        depth = 31
        mesh_idx = 15
        # Sheet-parallel stack: same texture aligned across depth + mild per-slice noise.
        parallel_stack = np.empty((depth, 128, 128), dtype=np.uint8)
        for k in range(depth):
            noise = rng.normal(0.0, 5.0, size=(128, 128))
            atten = 1.0 - 0.015 * abs(k - mesh_idx)
            parallel_stack[k] = np.clip(np.rint(base[16:144, 16:144] * atten + noise), 1, 255).astype(np.uint8)

        # Oblique-cut stack: features translate laterally by 0.75 px per slice (3 px across delta=±2).
        oblique_stack = np.empty((depth, 128, 128), dtype=np.uint8)
        yy, xx = np.meshgrid(np.arange(16, 144, dtype=np.float64), np.arange(16, 144, dtype=np.float64), indexing="ij")
        for k in range(depth):
            shift_x = 0.75 * (k - mesh_idx)
            shifted = ndi.map_coordinates(base, [yy, xx + shift_x], order=1, mode="nearest")
            noise = rng.normal(0.0, 4.0, size=(128, 128))
            oblique_stack[k] = np.clip(np.rint(shifted + noise), 1, 255).astype(np.uint8)

        res_parallel = ssr.layer_coherence_score(
            parallel_stack, mesh_index=mesh_idx, delta=2, patch_size=64, max_shift=4
        )
        res_oblique = ssr.layer_coherence_score(
            oblique_stack, mesh_index=mesh_idx, delta=2, patch_size=64, max_shift=4
        )

        self.assertEqual(res_parallel["patch_count"], 4)
        self.assertGreaterEqual(res_parallel["sheet_parallel_fraction"], 0.80)
        self.assertGreater(res_parallel["mean_zero_shift_r"], res_parallel["mean_best_shifted_r"])

        self.assertEqual(res_oblique["patch_count"], 4)
        self.assertLessEqual(res_oblique["sheet_parallel_fraction"], 0.30)
        self.assertGreater(res_oblique["mean_best_shifted_r"], res_oblique["mean_zero_shift_r"])

    def test_5_cli_end_to_end_and_zero_local_paths_in_receipt(self) -> None:
        m7_vol, x_raw, y_raw, z_raw = _build_curved_sheet_fixture(shape=(64, 64, 64))
        # Build synthetic textured CT volume where voxels inside x in [12, 52], y in [12, 52]
        # carry non-zero sheet-parallel texture and outside stays 0 (testing CT support mask).
        rng = np.random.default_rng(7)
        tex = ndi.gaussian_filter(rng.normal(0.0, 1.0, size=(64, 64)), sigma=1.1)
        tex_u8 = np.clip(np.rint(140.0 + 40.0 * (tex / (tex.std() + 1e-6))), 20, 240).astype(np.uint8)
        ct_vol = np.zeros((64, 64, 64), dtype=np.uint8)
        ct_vol[:, 12:52, 12:52] = tex_u8[None, 12:52, 12:52]
        # Zero out a corner block wider than the +-15 vx normal cone to confirm valid_mask excludes zero-CT regions.
        ct_vol[:, :24, :24] = 0

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_dir = tmp_path / "synthetic_winding"
            ssr.write_tifxyz(
                mesh_dir,
                x_raw,
                y_raw,
                z_raw,
                meta={"scale": [1.0, 1.0], "uuid": "synthetic_winding"},
            )
            ct_dir = _write_local_zarr_v2(tmp_path / "ct.zarr", ct_vol, chunks=(16, 16, 16), sparse_zero_skip=True)
            m7_dir = _write_local_zarr_v2(tmp_path / "m7.zarr", m7_vol, chunks=(16, 16, 16), sparse_zero_skip=True)
            out_npy = tmp_path / "out" / "stack.npy"
            out_receipt = tmp_path / "out" / "receipt.json"

            cmd = [
                sys.executable,
                str(HERE / "stream_snap_render.py"),
                "--mesh-dir",
                str(mesh_dir),
                "--ct-zarr",
                str(ct_dir),
                "--m7-zarr",
                str(m7_dir),
                "--snap-passes",
                "2",
                "--num-slices",
                "31",
                "--cache-mb",
                "8",
                "--workers",
                "2",
                "--out-npy",
                str(out_npy),
                "--out-receipt",
                str(out_receipt),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, msg=f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            self.assertTrue(out_npy.is_file())
            self.assertTrue(out_receipt.is_file())

            stack = np.load(out_npy)
            self.assertEqual(stack.dtype, np.uint8)
            self.assertEqual(stack.shape, (31, x_raw.shape[0], x_raw.shape[1]))

            raw_receipt = out_receipt.read_text(encoding="utf-8")
            # Zero local filesystem paths in receipt.
            for pat in FORBIDDEN_PATH_PATTERNS:
                self.assertIsNone(
                    pat.search(raw_receipt),
                    msg=f"Receipt leaked forbidden local path pattern {pat.pattern!r}:\n{raw_receipt}",
                )
            self.assertNotIn(str(tmp_path), raw_receipt)
            self.assertNotIn(tmp_path.as_posix(), raw_receipt)

            receipt = json.loads(raw_receipt)
            expected_sha = hashlib.sha256(np.ascontiguousarray(stack).tobytes()).hexdigest()
            self.assertEqual(receipt["stack_sha256"], expected_sha)
            self.assertEqual(receipt["mesh_shape"], [int(x_raw.shape[0]), int(x_raw.shape[1])])
            self.assertEqual(receipt["stack_shape"], [31, int(x_raw.shape[0]), int(x_raw.shape[1])])
            self.assertGreater(receipt["chunks_fetched"], 0)
            self.assertGreater(receipt["bytes_fetched"], 0)
            # Some pixels were zeroed in CT -> valid_pixels strictly > 0 and < total_pixels.
            self.assertGreater(receipt["valid_pixels"], 0)
            self.assertLess(receipt["valid_pixels"], x_raw.size)
            self.assertGreaterEqual(receipt["geometry_metrics"]["on_sheet_after_snap"], 0.95)
            self.assertGreaterEqual(receipt["coherence"]["patch_count"], 1)
            self.assertGreaterEqual(receipt["coherence"]["sheet_parallel_fraction"], 0.80)

    def test_6_render_flip_normals_upsampling_and_ct_support_mask(self) -> None:
        # Depth-ramp CT volume: intensity equals 10 + 3*z inside x >= 10, and 0 for x < 10.
        zz = np.arange(48, dtype=np.uint8)[:, None, None]
        ct = np.broadcast_to((10 + 3 * zz).astype(np.uint8), (48, 32, 32)).copy()
        ct[:, :, :10] = 0

        # Planar grid at z = 24 spanning x in [6..20], y in [6..20] at 2-voxel grid spacing (scale=0.5).
        uu, vv = np.meshgrid(
            np.arange(6, 22, 2, dtype=np.float32),
            np.arange(6, 22, 2, dtype=np.float32),
            indexing="xy",
        )
        zz_mesh = np.full_like(uu, 24.0)

        with tempfile.TemporaryDirectory() as tmp:
            ct_dir = _write_local_zarr_v2(Path(tmp) / "ct_ramp.zarr", ct, chunks=(16, 16, 16))
            with ssr.RemoteZarrV2(ct_dir, level="0", cache_mb=8, workers=1) as reader:
                stack_fwd, mask_fwd = ssr.render_surface_stack(
                    uu, vv, zz_mesh, reader, num_slices=11, slice_step=1.0, flip_normals=False, scale=0.5
                )
                stack_rev, mask_rev = ssr.render_surface_stack(
                    uu, vv, zz_mesh, reader, num_slices=11, slice_step=1.0, flip_normals=True, scale=0.5
                )

        # (8 - 1) / 0.5 + 1 = 15 native pixels per side.
        self.assertEqual(stack_fwd.shape, (11, 15, 15))
        self.assertEqual(stack_rev.shape, (11, 15, 15))
        np.testing.assert_array_equal(mask_fwd, mask_rev)
        # Reversing normals reverses depth stack order identically on this planar surface.
        np.testing.assert_array_equal(stack_rev, stack_fwd[::-1])
        # Center slice (index 5) sits at z = 24 -> intensity 10 + 3*24 = 82 on valid CT region.
        self.assertEqual(int(stack_fwd[5, 7, 10]), 82)
        # x < 10 corresponds to output cols 0..3 (x = 6, 7, 8, 9) -> zero CT support -> mask False.
        self.assertFalse(bool(mask_fwd[:, :3].any()))
        self.assertTrue(bool(mask_fwd[:, 5:].all()))


if __name__ == "__main__":
    unittest.main()
