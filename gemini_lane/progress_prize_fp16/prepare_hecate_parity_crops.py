"""Stream two tiny public uint8 [Z, Y, X] CT crops for Hecate fp16 parity bursts.

Reuses ``RemoteZarrV2`` from ``gemini_lane/stream_snap_render.py`` to fetch two
small public Zarr v2 crops (default ``64 x 256 x 256``, ``< 15 MB`` total HTTP
transfer via Z-subchunk HTTP Range reads on uncompressed C-order S3 stores) into
a caller-specified temporary directory strictly outside any git repository or
worktree:

* ``hecate_9p6um_crop.npy``: published PHerc0306B positive control (8.640 um S3
  CT Zarr v2), auto-verified non-zero structural crop.
* ``hecate_2p4um_crop.npy``: public PHerc0846A 2.403 um S3 CT Zarr v2,
  auto-verified non-zero structural crop selected dynamically at runtime (no
  voxel coordinates hardcoded in source or logged in receipts).

Intended solely for numerical fp16-vs-fp32 parity checks with
``verify_upstream_gpu_burst.py`` (not an ink-accuracy evaluation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import urllib.error
import urllib.request

import numpy as np

# Ensure sibling gemini_lane modules are importable regardless of working directory.
_GEMINI_LANE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_GEMINI_LANE_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMINI_LANE_DIR))

import stream_snap_render as ssr  # noqa: E402
from stream_snap_render import RemoteZarrV2  # noqa: E402

CROP_9P6_NAME = "hecate_9p6um_crop.npy"
CROP_2P4_NAME = "hecate_2p4um_crop.npy"

DEFAULT_SOURCE_9P6 = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr"
)
DEFAULT_SOURCE_2P4 = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr"
)

DEFAULT_DEPTH = 64
DEFAULT_SIZE_YX = 256
DEFAULT_MIN_NONZERO_FRAC = 0.2
DEFAULT_MAX_TOTAL_MB = 15.0
PROBE_Z_SLICES = 4


# ---------------------------------------------------------------------------
# Path and privacy guardrails
# ---------------------------------------------------------------------------

def _has_git_marker(path: Path) -> bool:
    """Return True if path or any ancestor contains a .git directory or file."""
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return True
    return False


def is_inside_git_repo(path: str | os.PathLike[str]) -> bool:
    """Return True if path (resolved or lexical) sits inside any git repo/worktree."""
    raw = Path(path).expanduser()
    resolved = raw.resolve(strict=False)
    lexical = Path(os.path.abspath(str(raw)))
    for probe in (resolved, lexical):
        if _has_git_marker(probe):
            return True
        if probe == _REPO_ROOT or _REPO_ROOT in probe.parents:
            return True
    return False


def ensure_outside_git_repo(path: str | os.PathLike[str], *, label: str = "--out-dir") -> Path:
    """Validate that path is strictly outside any git repo or worktree."""
    if is_inside_git_repo(path):
        raise ValueError(
            f"{label} must be strictly outside any git repository or worktree "
            "(choose an external temporary directory)"
        )
    return Path(path).expanduser().resolve(strict=False)


def sanitize_source(source: str | os.PathLike[str]) -> str:
    """Return a sanitized public URL (no creds/query) or bare basename for local paths."""
    cleaned = ssr._sanitize_source_label(source)
    return cleaned if cleaned else "local"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# ---------------------------------------------------------------------------
# Streaming Zarr v2 reader with Z-subchunk HTTP Range support
# ---------------------------------------------------------------------------

def _http_get_range(
    reader: RemoteZarrV2, key: str, start_byte: int, end_byte: int
) -> tuple[bytes | None, int]:
    """Fetch inclusive HTTP byte range [start_byte, end_byte] with strict 404 handling.

    Returns ``(slice_bytes, wire_bytes_fetched)``. Missing chunks (HTTP 404)
    return ``(None, 0)``; non-404 4xx errors raise immediately.
    """
    expected_len = end_byte - start_byte + 1
    if expected_len <= 0:
        raise ValueError("Invalid byte range")
    url = f"{reader.root}/{key}"
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}

    if ssr.requests is not None:
        for attempt in range(reader.max_retries):
            try:
                resp = reader._session().get(url, headers=headers, timeout=reader.timeout)
                status = int(resp.status_code)
                if status == 404:
                    return None, 0
                if status == 206:
                    data = resp.content
                    if len(data) != expected_len:
                        raise OSError(
                            f"Short HTTP 206 read for {key}: expected {expected_len} bytes, got {len(data)}"
                        )
                    return data, len(data)
                if status == 200:
                    data = resp.content
                    if len(data) < end_byte + 1:
                        raise OSError(
                            f"Short HTTP 200 read for {key}: expected >= {end_byte + 1} bytes, got {len(data)}"
                        )
                    return data[start_byte : end_byte + 1], len(data)
                if 400 <= status < 500:
                    resp.raise_for_status()
                    raise ssr.requests.HTTPError(
                        f"HTTP {status} for range request on {key}", response=resp
                    )
                if attempt + 1 >= reader.max_retries:
                    resp.raise_for_status()
                    raise ssr.requests.HTTPError(
                        f"Unexpected HTTP {status} for range request on {key}", response=resp
                    )
            except ssr.requests.HTTPError:
                raise
            except ssr.requests.RequestException:
                if attempt + 1 >= reader.max_retries:
                    raise
            if reader.retry_backoff > 0.0:
                time.sleep(reader.retry_backoff * (attempt + 1))
        raise OSError(f"Exhausted retries fetching range for {key}")  # pragma: no cover

    for attempt in range(reader.max_retries):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=reader.timeout) as resp:
                status = int(getattr(resp, "status", 200))
                data = resp.read()
                if status == 206:
                    if len(data) != expected_len:
                        raise OSError(
                            f"Short HTTP 206 read for {key}: expected {expected_len} bytes, got {len(data)}"
                        )
                    return data, len(data)
                if status == 200:
                    if len(data) < end_byte + 1:
                        raise OSError(
                            f"Short HTTP 200 read for {key}: expected >= {end_byte + 1} bytes, got {len(data)}"
                        )
                    return data[start_byte : end_byte + 1], len(data)
                raise urllib.error.HTTPError(
                    url, status, f"Unexpected HTTP {status}", resp.headers, None
                )
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, 0
            if 400 <= exc.code < 500 or attempt + 1 >= reader.max_retries:
                raise
        except OSError:
            if attempt + 1 >= reader.max_retries:
                raise
        if reader.retry_backoff > 0.0:
            time.sleep(reader.retry_backoff * (attempt + 1))
    raise OSError(f"Exhausted retries fetching range for {key}")  # pragma: no cover


def read_box_streaming(
    reader: RemoteZarrV2,
    lo_zyx: Sequence[int],
    hi_zyx: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Read dense ``[lo_zyx, hi_zyx)`` block + chunk-availability support mask.

    Uses HTTP byte-range Z-subchunk reads when ``reader`` targets a remote
    uncompressed C-order Zarr v2 array and each overlapping chunk only needs a
    strict Z-subinterval; otherwise delegates directly to ``reader.read_box``.
    """
    lo = [max(0, int(v)) for v in lo_zyx]
    hi = [min(s, int(v)) for s, v in zip(reader.shape, hi_zyx)]
    if any(h <= l for l, h in zip(lo, hi)):
        raise ValueError("Requested crop box is empty after clipping to volume bounds")

    cz, cy, cx = reader.chunks
    c0 = [l // c for l, c in zip(lo, reader.chunks)]
    c1 = [(h - 1) // c for h, c in zip(hi, reader.chunks)]

    # Determine whether every overlapping chunk uses a strict Z sub-slice.
    can_z_range = (
        (not reader.local)
        and (reader._decode is None)
        and (reader.order == "C")
    )
    if can_z_range:
        max_lz_span = 0
        for iz in range(c0[0], c1[0] + 1):
            a0 = max(lo[0], iz * cz)
            a1 = min(hi[0], (iz + 1) * cz, reader.shape[0])
            max_lz_span = max(max_lz_span, a1 - a0)
        can_z_range = 0 < max_lz_span < cz

    if not can_z_range:
        block, _origin, support = reader.read_box(lo, hi, return_support=True)
        return block, support

    idxs = [
        (iz, iy, ix)
        for iz in range(c0[0], c1[0] + 1)
        for iy in range(c0[1], c1[1] + 1)
        for ix in range(c0[2], c1[2] + 1)
    ]
    slice_bytes = cy * cx * int(reader.dtype.itemsize)

    def _fetch_subchunk(idx: tuple[int, int, int]) -> tuple[tuple[int, int, int], int, np.ndarray | None]:
        iz = idx[0]
        sz0 = iz * cz
        a0 = max(lo[0], sz0)
        a1 = min(hi[0], sz0 + cz, reader.shape[0])
        lz0 = a0 - sz0
        lz1 = a1 - sz0
        raw, wire_len = _http_get_range(
            reader, reader._chunk_key(idx), lz0 * slice_bytes, lz1 * slice_bytes - 1
        )
        if raw is None:
            return idx, lz0, None
        arr = np.frombuffer(raw, dtype=reader.dtype).reshape((lz1 - lz0, cy, cx)).copy()
        with reader._lock:
            reader.chunks_fetched += 1
            reader.bytes_fetched += wire_len
        return idx, lz0, arr

    if reader._pool is not None and len(idxs) > 1:
        results = list(reader._pool.map(_fetch_subchunk, idxs))
    else:
        results = [_fetch_subchunk(idx) for idx in idxs]

    block = np.full([h - l for l, h in zip(lo, hi)], reader.fill_value, dtype=reader.dtype)
    support = np.zeros(block.shape, dtype=bool)
    for idx, lz0, sub in results:
        if sub is None:
            continue
        s = [i * c for i, c in zip(idx, reader.chunks)]
        a0 = [max(l, x) for l, x in zip(lo, s)]
        a1 = [min(h, x + c, sh) for h, x, c, sh in zip(hi, s, reader.chunks, reader.shape)]
        if any(y <= x for x, y in zip(a0, a1)):
            continue
        src = sub[
            (a0[0] - s[0]) - lz0 : (a1[0] - s[0]) - lz0,
            a0[1] - s[1] : a1[1] - s[1],
            a0[2] - s[2] : a1[2] - s[2],
        ]
        dst_slices = (
            slice(a0[0] - lo[0], a1[0] - lo[0]),
            slice(a0[1] - lo[1], a1[1] - lo[1]),
            slice(a0[2] - lo[2], a1[2] - lo[2]),
        )
        block[dst_slices] = src
        support[dst_slices] = np.isfinite(src)
    return block, support


# ---------------------------------------------------------------------------
# Automatic non-zero structural crop selection (no hardcoded coordinates)
# ---------------------------------------------------------------------------

def candidate_crop_boxes(
    vol_shape: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Generate deterministic chunk-aligned boxes centered on the volume geometry."""
    if any(vs < cs for vs, cs in zip(vol_shape, crop_shape)):
        raise ValueError(
            f"Volume shape {vol_shape} is smaller than requested crop shape {crop_shape}"
        )

    base = []
    for vs, ch, cs in zip(vol_shape, chunk_shape, crop_shape):
        center = (vs - cs) // 2
        aligned = (center // ch) * ch
        aligned = max(0, min(aligned, vs - cs))
        base.append(aligned)

    chunk_offsets = [
        (0, 0, 0),
        (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
        (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
        (2, 0, 0), (-2, 0, 0),
    ]

    boxes: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    seen: set[tuple[int, int, int]] = set()
    for dz, dy, dx in chunk_offsets:
        lo = (
            max(0, min(base[0] + dz * chunk_shape[0], vol_shape[0] - crop_shape[0])),
            max(0, min(base[1] + dy * chunk_shape[1], vol_shape[1] - crop_shape[1])),
            max(0, min(base[2] + dx * chunk_shape[2], vol_shape[2] - crop_shape[2])),
        )
        if lo in seen:
            continue
        seen.add(lo)
        hi = (lo[0] + crop_shape[0], lo[1] + crop_shape[1], lo[2] + crop_shape[2])
        boxes.append((lo, hi))

    if (0, 0, 0) not in seen:
        boxes.append(((0, 0, 0), crop_shape))
    return boxes


def select_and_stream_crop(
    reader: RemoteZarrV2,
    crop_shape: tuple[int, int, int],
    *,
    min_nonzero_frac: float = DEFAULT_MIN_NONZERO_FRAC,
    max_bytes: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stream and verify a uint8 structural crop automatically from ``reader``.

    Does not include raw voxel coordinates in the returned statistics dict.
    """
    if reader.dtype != np.dtype(np.uint8):
        raise ValueError(f"Expected uint8 Zarr v2 volume, got dtype={reader.dtype}")

    candidates = candidate_crop_boxes(reader.shape, reader.chunks, crop_shape)
    fallback_candidates: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    for lo, hi in candidates:
        if max_bytes is not None and reader.bytes_fetched > max_bytes:
            raise RuntimeError(
                f"HTTP transfer budget exceeded ({reader.bytes_fetched} > {max_bytes} bytes)"
            )

        # Lightweight central-Z probe on remote uncompressed stores to avoid
        # streaming full empty/masked-out boxes.
        if (not reader.local) and (reader._decode is None) and (reader.order == "C"):
            mid_z = lo[0] + max(0, (crop_shape[0] - PROBE_Z_SLICES) // 2)
            probe_lo = (mid_z, lo[1], lo[2])
            probe_hi = (
                min(hi[0], mid_z + PROBE_Z_SLICES),
                min(hi[1], lo[1] + reader.chunks[1]),
                min(hi[2], lo[2] + reader.chunks[2]),
            )
            probe_block, _ = read_box_streaming(reader, probe_lo, probe_hi)
            if not np.any(probe_block):
                fallback_candidates.append((lo, hi))
                continue

        block, support = read_box_streaming(reader, lo, hi)
        if max_bytes is not None and reader.bytes_fetched > max_bytes:
            raise RuntimeError(
                f"HTTP transfer budget exceeded ({reader.bytes_fetched} > {max_bytes} bytes)"
            )
        if tuple(block.shape) != tuple(crop_shape) or block.dtype != np.uint8:
            continue
        nonzero_frac = float((block > 0).mean())
        std_u8 = float(block.std())
        if nonzero_frac > min_nonzero_frac and std_u8 > 0.0:
            stats = {
                "shape": [int(v) for v in block.shape],
                "dtype": "uint8",
                "non_zero_frac": round(nonzero_frac, 6),
                "source_support_frac": round(float(support.mean()), 6),
                "mean_u8": round(float(block.mean()), 4),
                "std_u8": round(std_u8, 4),
                "min_u8": int(block.min()),
                "max_u8": int(block.max()),
                "bytes_fetched": int(reader.bytes_fetched),
                "chunks_fetched": int(reader.chunks_fetched),
                "selection_strategy": "auto_center_chunk_aligned",
            }
            return np.ascontiguousarray(block, dtype=np.uint8), stats

    raise RuntimeError(
        f"Failed to locate a structural uint8 crop with non_zero_frac > {min_nonzero_frac} "
        f"and std > 0 across {len(candidates)} candidate boxes"
    )


def atomic_save_npy(path: Path, arr: np.ndarray) -> str:
    """Atomically save a contiguous uint8 numpy array and return its SHA-256 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.stem}.part.npy")
    try:
        np.save(part, np.ascontiguousarray(arr, dtype=np.uint8), allow_pickle=False)
        part.replace(path)
    except Exception:
        if part.exists():
            part.unlink()
        raise
    return sha256_file(path)


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="External output directory strictly outside any git repo or worktree",
    )
    parser.add_argument(
        "--source-9p6",
        default=DEFAULT_SOURCE_9P6,
        help="Zarr v2 URL or local directory for the 9.6 um parity crop (default: PHerc0306B 8.640 um)",
    )
    parser.add_argument(
        "--source-2p4",
        default=DEFAULT_SOURCE_2P4,
        help="Zarr v2 URL or local directory for the 2.4 um parity crop (default: PHerc0846A 2.403 um)",
    )
    parser.add_argument("--level", default="0", help="Multiscale Zarr v2 level (default: 0)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Crop Z depth (default: 64)")
    parser.add_argument("--size-yx", type=int, default=DEFAULT_SIZE_YX, help="Crop Y and X size (default: 256)")
    parser.add_argument(
        "--min-nonzero-frac",
        type=float,
        default=DEFAULT_MIN_NONZERO_FRAC,
        help="Minimum required fraction of non-zero uint8 voxels (default: 0.2)",
    )
    parser.add_argument(
        "--max-total-mb",
        type=float,
        default=DEFAULT_MAX_TOTAL_MB,
        help="Hard ceiling on combined HTTP/source bytes fetched in MB (default: 15.0)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Chunk fetch worker threads (default: 4)")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP request timeout in seconds")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional sanitized JSON receipt path (must not exist; contains no local paths or voxel coordinates)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print sanitized execution plan without network or disk writes",
    )
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    try:
        ensure_outside_git_repo(args.out_dir, label="--out-dir")
    except ValueError as exc:
        errors.append(str(exc))
    if args.depth < 16:
        errors.append("--depth must be >= 16 (64 recommended for 2.4 um Hecate)")
    if args.size_yx < 64:
        errors.append("--size-yx must be >= 64 (256 recommended for 2.4 um Hecate)")
    if not 0.0 <= args.min_nonzero_frac < 1.0:
        errors.append("--min-nonzero-frac must be in [0.0, 1.0)")
    if args.max_total_mb <= 0.0:
        errors.append("--max-total-mb must be > 0")
    if args.workers < 1:
        errors.append("--workers must be >= 1")
    if args.receipt is not None and args.receipt.exists():
        errors.append("--receipt already exists; choose a new file")
    return errors


def build_next_command_template() -> str:
    return (
        "python gemini_lane/progress_prize_fp16/verify_upstream_gpu_burst.py "
        "--stages studio,hecate "
        f"--hecate-input-9p6 <out-dir>/{CROP_9P6_NAME} "
        f"--hecate-input-2p4 <out-dir>/{CROP_2P4_NAME} "
        "--output <out-dir>/burst_receipt.json"
    )


def prepare_crops(
    out_dir: str | os.PathLike[str],
    *,
    source_9p6: str | os.PathLike[str] = DEFAULT_SOURCE_9P6,
    source_2p4: str | os.PathLike[str] = DEFAULT_SOURCE_2P4,
    level: str | None = "0",
    crop_shape: tuple[int, int, int] = (DEFAULT_DEPTH, DEFAULT_SIZE_YX, DEFAULT_SIZE_YX),
    min_nonzero_frac: float = DEFAULT_MIN_NONZERO_FRAC,
    max_total_mb: float = DEFAULT_MAX_TOTAL_MB,
    workers: int = 4,
    timeout: float = 60.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stream both Hecate parity crops into an external directory and return a sanitized receipt."""
    safe_out_dir = ensure_outside_git_repo(out_dir, label="--out-dir")
    crop_specs = (
        ("9.6", CROP_9P6_NAME, source_9p6, "9.6um_target_from_8.640um_control"),
        ("2.4", CROP_2P4_NAME, source_2p4, "2.4um_target_from_2.403um_public_ct"),
    )

    summary: dict[str, Any] = {
        "kind": "hecate fp16 parity crop preparation (numerical parity only, not an accuracy claim)",
        "dry_run": bool(dry_run),
        "crop_shape": [int(v) for v in crop_shape],
        "min_nonzero_frac": float(min_nonzero_frac),
        "max_total_mb": float(max_total_mb),
        "crops": {},
        "next_command_template": build_next_command_template(),
    }

    if dry_run:
        for spacing_key, filename, source, spacing_label in crop_specs:
            summary["crops"][spacing_key] = {
                "filename": filename,
                "source": sanitize_source(source),
                "spacing_label": spacing_label,
                "shape": [int(v) for v in crop_shape],
                "dtype": "uint8",
                "selection_strategy": "auto_center_chunk_aligned",
            }
        summary["total_bytes_fetched"] = 0
        summary["total_mb_fetched"] = 0.0
        return summary

    max_total_bytes = int(round(float(max_total_mb) * 1_000_000))
    total_bytes = 0
    written_paths: list[Path] = []
    safe_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for spacing_key, filename, source, spacing_label in crop_specs:
            remaining = max(0, max_total_bytes - total_bytes)
            with RemoteZarrV2(
                source,
                level=level,
                cache_mb=32.0,
                workers=workers,
                timeout=timeout,
            ) as reader:
                crop, stats = select_and_stream_crop(
                    reader,
                    crop_shape,
                    min_nonzero_frac=min_nonzero_frac,
                    max_bytes=remaining,
                )
            total_bytes += int(stats["bytes_fetched"])
            if total_bytes > max_total_bytes:
                raise RuntimeError(
                    f"Combined transfer budget exceeded ({total_bytes} > {max_total_bytes} bytes)"
                )
            out_path = safe_out_dir / filename
            digest = atomic_save_npy(out_path, crop)
            written_paths.append(out_path)
            summary["crops"][spacing_key] = {
                "filename": filename,
                "source": sanitize_source(source),
                "spacing_label": spacing_label,
                "sha256": digest,
                **stats,
            }
    except Exception:
        for p in written_paths:
            if p.exists():
                p.unlink()
        raise

    summary["total_bytes_fetched"] = int(total_bytes)
    summary["total_mb_fetched"] = round(total_bytes / 1_000_000.0, 4)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if errors := validate_args(args):
        parser.error("; ".join(errors))

    summary = prepare_crops(
        args.out_dir,
        source_9p6=args.source_9p6,
        source_2p4=args.source_2p4,
        level=args.level,
        crop_shape=(args.depth, args.size_yx, args.size_yx),
        min_nonzero_frac=args.min_nonzero_frac,
        max_total_mb=args.max_total_mb,
        workers=args.workers,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    payload = json.dumps(summary, indent=2) + "\n"
    if args.receipt is not None and not args.dry_run:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
