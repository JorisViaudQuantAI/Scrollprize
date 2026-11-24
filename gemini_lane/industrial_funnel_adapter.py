# Copyright (c) 2026 Scrollprize Contributors (MIT License)
"""Cross-platform R1 Industrial Search Funnel adapter, semantic validator,
S0 19-scroll inventory joiner, and normal-correspondence dedup helper.

CPU-only (zero torch/CUDA imports). Compatible with both Windows (NTFS) and
POSIX hosts while preserving the byte-integrity, atomic no-overwrite, and
privacy invariants of RFC-001 / AMENDMENT-001 (``funnel.file_receipt.v1``).

Key responsibilities:
1. Cross-platform ``CrossPlatformCAS`` and ``CrossPlatformFileReceipts`` that
   avoid POSIX-only ``st_mode & 0o077`` checks and directory ``os.open``/``fsync``
   failures on Windows NTFS while preserving them strictly on POSIX.
2. Dual-mode inference receipts supporting BOTH:
   - ``paired_both_orders`` (reference / deep S4 / ambiguous normals), and
   - ``single_order_audit_15pct`` (AMENDMENT-001 B8/S2 experimental coarse mode),
   resolving the schema contradiction in the Linux-only reference ``file_receipts.py``.
3. Gemini domain semantic contracts from Exp T15, Exp T16-GPU, Exp T17-GPU, Q22,
   and Q33:
   - 2D shear quarantine gate (``>30 deg`` p95 or ``pct_area_gt_20deg > 20%``),
     explicitly separating 1D Exp T2 arc-length resampling from 2D flattening;
   - explicit pixel-center frame (``p * scale`` vs ``(p + 0.5) * scale``),
     exact signed slice lists, and non-padded ``source_support`` area accounting;
   - non-gating Q34 / neural bleed-through diagnostic (``2-3x`` floor on real text),
     Q33 mixed-region recall reporting, GPU/VRAM/thermal telemetry,
     ``worker_stopped`` attestation, and ``leak_guard`` privacy verification.
4. S0 + Coverage + Geometry + Radiometry 19-scroll inventory joiner (public
   scroll-level aggregates only; zero unread coordinates or selected z-bands).
5. CPU-only 64-voxel cell occupancy vs <=3-voxel normal-correspondence dedup
   evaluator for ``tifxyz`` surfaces.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    from . import leak_guard as LG
except ImportError:  # pragma: no cover - direct module import fallback
    import leak_guard as LG  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Constants & Canonical Schemas
# ---------------------------------------------------------------------------

SCHEMA_FILE_RECEIPT = "funnel.file_receipt.v1"
SCHEMA_ADAPTER_METADATA = "gemini.r1_adapter_metadata.v1"
SCHEMA_S0_JOIN = "gemini.s0_19scroll_inventory.v1"

PINNED_S0_GIT_BLOB_SHA1 = "333735fa42e032171d0fc8ebd3df4f4a0da9f1b2"
S0_LEVELS: tuple[float, ...] = (0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92)
S0_COLUMNS: tuple[str, ...] = (
    "scroll",
    "status",
    "z_fraction",
    "z_slice",
    "slabs",
    "clean_slabs",
    "edge_rise_um_q75",
    "merged_frac",
    "coherence",
    "air_frac",
)

LANES: tuple[str, ...] = ("exploit", "repair", "explore")
RESOURCES: tuple[str, ...] = ("cpu_ms", "gpu_ms", "human_ms", "io_bytes")
PREDECESSOR: dict[str, str | None] = {
    "REGISTER": None,
    "CONTROL": None,
    "GEOMETRY": "REGISTER",
    "COARSE": "GEOMETRY",
    "RANK": "COARSE",
    "DEEP": "RANK",
    "REVIEW": "DEEP",
}
INFERENCE_STAGES = frozenset({"COARSE", "DEEP"})
FAILURES = frozenset(
    {
        "TIMEOUT",
        "OOM",
        "NETWORK",
        "MISSING_SUPPORT",
        "BAD_GEOMETRY",
        "BAD_CONTROL",
        "BAD_INPUT",
        "REVIEW_INCOMPLETE",
        "WORKER_STOPPED",
        "RESOURCE_OVERRUN",
        "SESSION_LOST",
        "ACCOUNT_LIMIT",
        "GPU_UNAVAILABLE",
        "THERMAL_ABORT",
        "SHEAR_QUARANTINED",
    }
)
REVIEW_STATUSES = frozenset(
    {
        "REVIEWED_NO_QUALIFYING_READING",
        "INCONCLUSIVE",
        "PRIVATE_REVIEW_REQUIRED",
    }
)
ORDER_MODES = frozenset({"paired_both_orders", "single_order_audit_15pct"})
PIXEL_CENTER_FRAMES = frozenset(
    {
        "vertex_aligned_p_times_scale",
        "pixel_center_p_plus_0p5_times_scale",
    }
)
PARAMETERIZATION_METHODS = frozenset(
    {
        "flatboi",
        "metric_flatten",
        "raw_fit_grid",
        "1d_arc_length_resample_t2",
        "growpatch_native",
        "rgt_stitched",
    }
)
METHODS_WITH_2D_FLATTENING = frozenset({"flatboi", "metric_flatten"})

# Exact signed slice lists on a 31-layer render (mesh_index=15, slice_step=1.0)
STUDIO_DESC_SLICES_31: list[int] = list(range(23, 7, -1))  # [23..8]
STUDIO_ASC_SLICES_31: list[int] = list(range(7, 23))       # [7..22]
AHN_S0_ASC_SLICES_31: list[int] = list(range(3, 27))       # [3..26]
AHN_S0_DESC_SLICES_31: list[int] = list(range(26, 2, -1))  # [26..3]

# Shear quarantine thresholds grounded in Exp T15 + Exp T16-GPU
SHEAR_P95_QUARANTINE_DEG = 30.0
SHEAR_PCT_AREA_GT_20DEG_QUARANTINE = 20.0
SINGLE_ORDER_MIN_AUDIT_RATE = 0.15

# Privacy banlist for public aggregate exports
PRIVATE_FORBIDDEN_KEYS = frozenset(
    {
        "uv_crop",
        "top_hotspots",
        "hotspots",
        "private_dir",
        "out_private_dir",
        "mesh_path",
        "local_path",
        "selected_z_slices",
        "proposed_first_two",
        "patch_scores",
    }
)
TOKEN_PATTERN = re.compile(
    r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
SCROLL_RE = re.compile(r"PHerc(?:[0-9]{3,4}[AB]?|Paris4)\Z")


# ---------------------------------------------------------------------------
# Canonical Hashing & Basic Validators (compatible with control_plane.py)
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to canonical UTF-8 JSON (sorted keys, compact, finite)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_obj(obj: Any) -> str:
    """Return SHA-256 hex digest of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def require_sha256(value: Any) -> str:
    """Verify ``value`` is a 64-char lowercase SHA-256 hex string."""
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Expected 64-char lowercase SHA-256 hex string, got {value!r}")
    return value


def file_sha256(path: Path) -> str:
    """Stream SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """Compute Git's ``blob <len>\\0<data>`` SHA-1 digest."""
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def validate_resource_vector(vec: Any) -> dict[str, int]:
    """Validate integer resource vector ``{cpu_ms, gpu_ms, human_ms, io_bytes}``."""
    if (
        not isinstance(vec, Mapping)
        or set(vec.keys()) != set(RESOURCES)
        or any(type(v) is not int or v < 0 for v in vec.values())
    ):
        raise ValueError("Resource vector must have nonnegative int cpu_ms, gpu_ms, human_ms, io_bytes")
    return {k: int(vec[k]) for k in RESOURCES}


def resource_within(actual: Mapping[str, int], grant: Mapping[str, int]) -> bool:
    """Return True iff every resource component in ``actual`` <= ``grant``."""
    return all(actual[k] <= grant[k] for k in RESOURCES)


def validate_hashes_map(obj: Any) -> dict[str, str]:
    """Validate named mapping of role -> SHA-256 hex digest."""
    if not isinstance(obj, Mapping) or any(not isinstance(k, str) or not k for k in obj):
        raise ValueError("Expected non-empty string keys for artifact hashes map")
    return {str(k): require_sha256(v) for k, v in obj.items()}


def normalize_scroll_code(name: str) -> str:
    """Normalize scroll strings like ``PHerc. 0125`` -> ``PHerc0125``."""
    return str(name).replace("PHerc. ", "PHerc").replace(" ", "").strip()


def validate_job_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an immutable job spec compatible with ``control_plane.validate_spec``."""
    required = {"scroll", "unit", "frame", "stage", "recipe", "inputs", "deps", "calibration"}
    if not isinstance(spec, Mapping) or set(spec.keys()) != required:
        raise ValueError("Job spec schema mismatch")
    scroll = spec["scroll"]
    if not isinstance(scroll, str) or not SCROLL_RE.fullmatch(scroll):
        raise ValueError(f"Invalid scroll code: {scroll!r}")
    unit = spec["unit"]
    if not isinstance(unit, str) or not unit or len(unit) > 128:
        raise ValueError("Invalid private unit identifier")
    require_sha256(spec["frame"])
    require_sha256(spec["recipe"])
    stage = spec["stage"]
    if stage not in PREDECESSOR:
        raise ValueError(f"Invalid stage: {stage!r}")
    inputs = validate_hashes_map(spec["inputs"])
    deps = spec["deps"]
    if not isinstance(deps, list) or len(deps) != len(set(deps)):
        raise ValueError("Invalid dependencies list")
    for dep in deps:
        require_sha256(dep)
    cal = spec["calibration"]
    if stage in INFERENCE_STAGES | {"CONTROL"}:
        require_sha256(cal)
    elif cal is not None:
        raise ValueError("Unexpected calibration key for non-inference/non-control stage")
    return {
        "scroll": scroll,
        "unit": unit,
        "frame": spec["frame"],
        "stage": stage,
        "recipe": spec["recipe"],
        "inputs": inputs,
        "deps": sorted(deps),
        "calibration": cal,
    }


def job_key(spec: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 job fingerprint."""
    return digest_obj(validate_job_spec(spec))


# ---------------------------------------------------------------------------
# 1. Cross-Platform Private Root, CAS & R1 FileReceipts (Windows + POSIX)
# ---------------------------------------------------------------------------

def private_root_cross_platform(path: str | os.PathLike[str]) -> Path:
    """Create and validate a private root directory outside every Git worktree.

    - Always rejects any path inside a Git worktree (``os.path.lexists(a / '.git')``).
    - Always rejects symlink roots.
    - On POSIX: enforces strict ``mode 0700`` (``(st_mode & 0o077) == 0``).
    - On Windows (NTFS): avoids the POSIX-only ``st_mode & 0o077`` false failure
      because NTFS always reports ``0o40777`` in ``os.stat()`` regardless of ACLs.
    """
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("Private root cannot be a symlink")
    resolved = raw.resolve()
    for ancestor in (resolved, *resolved.parents):
        if os.path.lexists(ancestor / ".git"):
            raise ValueError("Private state cannot be inside a Git worktree")
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("Private root must be a real directory")
    if os.name == "posix" and (resolved.stat().st_mode & 0o077):
        raise ValueError("Private root must have mode 0700 on POSIX")
    return resolved


def _fsync_directory_if_supported(directory: Path) -> None:
    """Fsync a directory descriptor on POSIX; safe no-op on Windows."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(directory), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_no_overwrite(tmp_path: Path, dst_path: Path, expected_body: bytes) -> None:
    """Atomically publish ``tmp_path`` to ``dst_path`` without overwriting differing content.

    Uses ``os.link`` where supported (POSIX + Windows NTFS/ReFS), falling back to
    ``os.O_CREAT | os.O_EXCL`` if the filesystem does not support hard links.
    """
    try:
        os.link(str(tmp_path), str(dst_path))
        return
    except FileExistsError:
        if dst_path.is_symlink() or not dst_path.is_file() or dst_path.read_bytes() != expected_body:
            raise ValueError("Conflicting immutable attempt receipt") from None
        return
    except OSError as exc:
        # Filesystem without hard-link support (e.g. FAT32/exFAT): use O_CREAT|O_EXCL.
        if exc.errno not in (errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
            raise
    try:
        fd = os.open(str(dst_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if dst_path.is_symlink() or not dst_path.is_file() or dst_path.read_bytes() != expected_body:
            raise ValueError("Conflicting immutable attempt receipt") from None
        return
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(expected_body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if dst_path.exists():
            dst_path.unlink(missing_ok=True)
        raise


class CrossPlatformCAS:
    """Content-addressed local private artifact store (Windows NTFS + POSIX)."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = private_root_cross_platform(root)

    def verify(self, sha: str) -> Path:
        digest = require_sha256(sha)
        path = self.root / digest
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Missing or corrupt CAS artifact: {digest}")
        return path

    def put(self, body: bytes) -> str:
        if not isinstance(body, (bytes, bytearray)):
            raise TypeError("CAS payload must be bytes")
        data = bytes(body)
        digest = hashlib.sha256(data).hexdigest()
        target = self.root / digest
        if target.exists() or target.is_symlink():
            self.verify(digest)
            return digest
        fd, tmp_name = tempfile.mkstemp(prefix=".cas-", dir=str(self.root))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(tmp_path), str(target))
            _fsync_directory_if_supported(self.root)
            self.verify(digest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return digest

    def put_json(self, obj: Any) -> str:
        return self.put((canonical_json(obj) + "\n").encode("utf-8"))

    def read_bytes(self, sha: str) -> bytes:
        return self.verify(sha).read_bytes()

    def read_json(self, sha: str) -> Any:
        return json.loads(self.read_bytes(sha).decode("utf-8"))


# ---------------------------------------------------------------------------
# 2. Domain Semantic Validators (Exp T15 / T16-GPU / T17-GPU / Q22 / Q33)
# ---------------------------------------------------------------------------

def expected_slice_indices(
    family: str,
    order: str,
    *,
    num_slices: int = 31,
    mesh_index: int | None = None,
    shift: int = 0,
) -> list[int]:
    """Return exact expected signed layer index list for Studio or A-hn."""
    if num_slices < 16:
        raise ValueError(f"num_slices must be >= 16, got {num_slices}")
    mid = (num_slices - 1) // 2 if mesh_index is None else int(mesh_index)
    c = mid + int(shift)
    fam = family.strip().lower()
    ord_ = order.strip().lower()
    if ord_ not in ("asc", "desc"):
        raise ValueError(f"Unsupported depth order: {order!r}")
    if fam == "studio":
        ks = list(range(c + 8, c - 8, -1)) if ord_ == "desc" else list(range(c - 8, c + 8))
    elif fam in ("ahn", "ahn_s0"):
        if num_slices < 24:
            raise ValueError(f"A-hn requires >= 24 slices, got {num_slices}")
        ks = list(range(c - 12, c + 12)) if ord_ == "asc" else list(range(c + 11, c - 13, -1))
    else:
        raise ValueError(f"Unsupported detector family: {family!r}")
    if min(ks) < 0 or max(ks) >= num_slices:
        raise ValueError(f"Slice indices {ks[0]}..{ks[-1]} outside [0, {num_slices - 1}]")
    return ks


def validate_geometry_contract(geom: Mapping[str, Any]) -> dict[str, Any]:
    """Validate 2D shear quarantine & parameterization lineage (Exp T15 + T16-GPU).

    Distinguishes 1D Exp T2 arc-length resampling from true 2D flattening
    (``flatboi`` / ``metric_flatten``). High-shear grids (``abs_p95_deg > 30`` or
    ``pct_area_gt_20deg > 20``) are quarantined for negative claims rather than
    silently accepted as blank papyrus.
    """
    method = geom.get("parameterization_method")
    if method not in PARAMETERIZATION_METHODS:
        raise ValueError(f"Unknown parameterization_method: {method!r}")

    abs_p95 = float(geom["shear_abs_p95_deg"])
    pct_gt20 = float(geom["pct_area_gt_20deg"])
    area_infl = float(geom.get("area_inflation_true_over_nominal", 1.0))
    snap_passes = int(geom.get("snap_passes", 0))
    if not all(math.isfinite(v) for v in (abs_p95, pct_gt20, area_infl)):
        raise ValueError("Geometry shear/area metrics must be finite")
    if abs_p95 < 0.0 or not (0.0 <= pct_gt20 <= 100.0) or area_infl <= 0.0 or snap_passes < 0:
        raise ValueError("Out-of-range geometry shear/area/snap parameters")

    is_2d_flattened = method in METHODS_WITH_2D_FLATTENING
    is_1d_t2_resampled = method == "1d_arc_length_resample_t2"
    quarantined = bool(
        abs_p95 > SHEAR_P95_QUARANTINE_DEG
        or pct_gt20 > SHEAR_PCT_AREA_GT_20DEG_QUARANTINE
    )

    if is_1d_t2_resampled and geom.get("claims_2d_flattening") is True:
        raise ValueError(
            "1D Exp T2 arc-length resampling cannot claim 2D flattening equivalence; "
            "use flatboi or metric_flatten for 2D distortion removal"
        )

    reasons: list[str] = []
    if abs_p95 > SHEAR_P95_QUARANTINE_DEG:
        reasons.append(f"shear_abs_p95_deg>{SHEAR_P95_QUARANTINE_DEG}")
    if pct_gt20 > SHEAR_PCT_AREA_GT_20DEG_QUARANTINE:
        reasons.append(f"pct_area_gt_20deg>{SHEAR_PCT_AREA_GT_20DEG_QUARANTINE}")

    return {
        "parameterization_method": method,
        "is_2d_flattened": is_2d_flattened,
        "is_1d_t2_resampled": is_1d_t2_resampled,
        "shear_abs_p95_deg": round(abs_p95, 4),
        "pct_area_gt_20deg": round(pct_gt20, 4),
        "area_inflation_true_over_nominal": round(area_infl, 4),
        "snap_passes": snap_passes,
        "snap_is_identity_certificate": False,
        "shear_quarantined": quarantined,
        "admissible_for_negative_claim": not quarantined,
        "quarantine_reasons": reasons,
        "recommended_remediation": (
            "run_2d_flattening_flatboi_or_metric_flatten" if quarantined else "none"
        ),
    }


def validate_sampling_and_detector_contract(
    sampling: Mapping[str, Any],
    detectors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate pixel-center frame, exact signed slice lists, and non-padded support."""
    frame = sampling.get("pixel_center_frame")
    if frame not in PIXEL_CENTER_FRAMES:
        raise ValueError(f"Explicit pixel_center_frame required from {sorted(PIXEL_CENTER_FRAMES)}, got {frame!r}")

    voxel_um = float(sampling["voxel_um"])
    num_slices = int(sampling["num_slices"])
    mesh_index = int(sampling.get("mesh_index", (num_slices - 1) // 2))
    slice_step = float(sampling.get("slice_step_vx", 1.0))
    if voxel_um <= 0.0 or num_slices < 16 or slice_step <= 0.0:
        raise ValueError("Invalid voxel_um / num_slices / slice_step_vx")

    support_px = int(sampling["source_support_pixels"])
    nominal_px = int(sampling["nominal_pixels"])
    valid_cm2 = float(sampling["valid_ct_cm2"])
    if support_px < 0 or nominal_px < support_px:
        raise ValueError("source_support_pixels must satisfy 0 <= source_support_pixels <= nominal_pixels")
    if sampling.get("padded_pixels_excluded_from_area") is not True:
        raise ValueError("Padded output pixels must be explicitly excluded from observed area")
    if sampling.get("missing_chunk_fill_excluded") is not True:
        raise ValueError("Missing Zarr chunks / fill values must be explicitly excluded from source support")

    expected_cm2 = support_px * ((voxel_um / 10000.0) ** 2)
    if abs(valid_cm2 - expected_cm2) > max(1e-3, 0.005 * expected_cm2):
        raise ValueError(
            f"valid_ct_cm2 ({valid_cm2:.4f}) disagrees with non-padded source_support_pixels "
            f"({support_px} -> {expected_cm2:.4f} cm^2)"
        )

    validated_slices: dict[str, list[int]] = {}
    if not detectors:
        raise ValueError("At least one detector slice specification is required")
    for det_key, cfg in detectors.items():
        fam = str(cfg["family"]).lower()
        ord_ = str(cfg["order"]).lower()
        shift = int(cfg.get("shift", 0))
        given = cfg.get("slice_indices")
        if not isinstance(given, list) or not all(type(i) is int for i in given):
            raise ValueError(f"Detector {det_key!r} must supply exact integer slice_indices list")
        exp = expected_slice_indices(fam, ord_, num_slices=num_slices, mesh_index=mesh_index, shift=shift)
        if given != exp:
            raise ValueError(
                f"Slice list mismatch for {det_key!r} ({fam}/{ord_}): got {given}, expected {exp}"
            )
        validated_slices[det_key] = given

    return {
        "pixel_center_frame": frame,
        "half_pixel_offset_applied": frame == "pixel_center_p_plus_0p5_times_scale",
        "voxel_um": voxel_um,
        "num_slices": num_slices,
        "mesh_index": mesh_index,
        "slice_step_vx": slice_step,
        "source_support_pixels": support_px,
        "nominal_pixels": nominal_px,
        "padded_pixels_excluded_from_area": True,
        "missing_chunk_fill_excluded": True,
        "valid_ct_cm2": round(expected_cm2, 4),
        "validated_slice_lists": validated_slices,
    }


def validate_diagnostics_and_telemetry(
    diagnostics: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Enforce non-gating Q34 (2-3x bleed-through floor), Q33 recall field, and GPU telemetry."""
    if diagnostics.get("q34_used_as_hard_gate") is not False:
        raise ValueError(
            "Q34 / TFCE asymmetry MUST be non-gating (q34_used_as_hard_gate=False): "
            "Exp T17-GPU measured a 2-3x neural bleed-through floor on real text (PHerc0211/0306B)"
        )
    if diagnostics.get("fp_filter_used_as_sole_gate") is True:
        raise ValueError("FP filter cannot be used as sole admission gate without montage review")

    q33 = diagnostics.get("q33_mixed_region_recall")
    if not isinstance(q33, Mapping) or "reported" not in q33:
        raise ValueError("Diagnostics must include q33_mixed_region_recall reporting field")
    if q33["reported"] is True:
        rec = float(q33["recall_at_budget"])
        if not (0.0 <= rec <= 1.0):
            raise ValueError("q33_mixed_region_recall.recall_at_budget must be in [0, 1]")

    # Telemetry validation
    required_tel = (
        "gpu_name",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "vram_cap_fraction",
        "max_temp_c",
        "post_run_temp_c",
        "worker_stopped",
    )
    missing = [k for k in required_tel if k not in telemetry]
    if missing:
        raise ValueError(f"Missing telemetry fields: {missing}")
    if telemetry["worker_stopped"] is not True:
        raise ValueError("Telemetry must attest worker_stopped=True")
    vram_frac = float(telemetry["vram_cap_fraction"])
    if not (0.0 < vram_frac <= 1.0):
        raise ValueError("vram_cap_fraction must be in (0, 1]")
    peak_alloc = int(telemetry["peak_allocated_mib"])
    peak_res = int(telemetry["peak_reserved_mib"])
    max_temp = int(telemetry["max_temp_c"])
    post_temp = int(telemetry["post_run_temp_c"])
    if peak_alloc < 0 or peak_res < peak_alloc:
        raise ValueError("peak_reserved_mib must be >= peak_allocated_mib >= 0")
    if post_temp > max_temp + 5:
        raise ValueError(f"post_run_temp_c ({post_temp}C) breached max_temp_c ({max_temp}C) tolerance")

    return {
        "q34_used_as_hard_gate": False,
        "q34_bleed_through_floor_acknowledged": True,
        "q34_diagnostic_score": diagnostics.get("q34_diagnostic_score"),
        "q33_mixed_region_recall": dict(q33),
        "telemetry": {
            "gpu_name": str(telemetry["gpu_name"]),
            "peak_allocated_mib": peak_alloc,
            "peak_reserved_mib": peak_res,
            "vram_cap_fraction": vram_frac,
            "max_temp_c": max_temp,
            "post_run_temp_c": post_temp,
            "batch_pause_ms": int(telemetry.get("batch_pause_ms", 0)),
            "hardware_or_oom_incidents": int(telemetry.get("hardware_or_oom_incidents", 0)),
            "worker_stopped": True,
        },
    }


def find_privacy_violations(obj: Any, *, user: str | None = None) -> list[str]:
    """Scan a JSON-serializable object or string for local paths, user leaks, tokens, and banned keys."""
    violations: list[str] = []

    def _walk(node: Any, path: str = "$") -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                ks = str(k)
                if ks in PRIVATE_FORBIDDEN_KEYS:
                    violations.append(f"forbidden_key:{path}.{ks}")
                _walk(v, f"{path}.{ks}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")

    if not isinstance(obj, str):
        _walk(obj)
        text = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    else:
        text = obj

    for hit in LG.find_leaks(text, user=user):
        violations.append(f"leak_guard:{hit}")
    for m in TOKEN_PATTERN.finditer(text):
        violations.append(f"token:{m.group(0)[:10]}...")
    return violations


def assert_public_export_clean(obj: Any, *, user: str | None = None) -> None:
    """Raise ``ValueError`` if ``obj`` violates ``leak_guard`` or private-key policy."""
    hits = find_privacy_violations(obj, user=user)
    if hits:
        raise ValueError(f"Privacy hygiene check failed: {hits[:8]}")


# ---------------------------------------------------------------------------
# 3. Receipt Validation (Dual-Order + A01 Single-Order-15%-Audit Modes)
# ---------------------------------------------------------------------------

def validate_r1_receipt(
    spec: Mapping[str, Any],
    attempt: int,
    receipt: Mapping[str, Any],
    store: CrossPlatformCAS,
) -> dict[str, Any]:
    """Validate a ``funnel.file_receipt.v1`` receipt supporting both order modes.

    Backward-compatible with ChatGPT's ``file_receipts.validate`` while adding
    AMENDMENT-001 B8/S2 ``single_order_audit_15pct`` support for ``COARSE``.
    """
    s = validate_job_spec(spec)
    j = job_key(s)
    if type(attempt) is not int or attempt < 1:
        raise ValueError("Invalid attempt number")

    base = {"job", "attempt", "outcome", "worker_stopped", "grant", "actual", "outputs", "failure"}
    outcome = receipt.get("outcome")
    if outcome not in ("COMPLETE", "FAILED", "PARTIAL"):
        raise ValueError(f"Invalid outcome: {outcome!r}")

    allowed_extra: set[str] = set()
    required_extra: set[str] = set()
    if outcome == "COMPLETE":
        if s["stage"] == "CONTROL":
            required_extra = {"calibration", "control_pass", "valid_until"}
        elif s["stage"] in INFERENCE_STAGES:
            required_extra = {"orders"}
            allowed_extra = {"order_mode", "audit_policy"}
        elif s["stage"] == "REVIEW":
            required_extra = {"review_status"}

    keys = set(receipt.keys())
    if not (base | required_extra <= keys) or not (keys <= base | required_extra | allowed_extra):
        raise ValueError(f"Receipt schema keys mismatch: got {sorted(keys)}")
    if receipt["job"] != j or receipt["attempt"] != attempt:
        raise ValueError("Receipt job or attempt mismatch")
    if receipt["worker_stopped"] is not True:
        raise ValueError("Worker not confirmed stopped")

    grant = validate_resource_vector(receipt["grant"])
    actual = validate_resource_vector(receipt["actual"])
    outputs = validate_hashes_map(receipt["outputs"])
    for h in outputs.values():
        store.verify(h)

    if outcome == "COMPLETE":
        if not resource_within(actual, grant):
            raise ValueError("Resource overrun cannot be marked COMPLETE")
        if not outputs or receipt["failure"] is not None:
            raise ValueError("COMPLETE receipt requires outputs and failure=None")
        for h in s["inputs"].values():
            store.verify(h)

        if s["stage"] in INFERENCE_STAGES:
            order_mode = receipt.get("order_mode", "paired_both_orders")
            if order_mode not in ORDER_MODES:
                raise ValueError(f"Invalid order_mode: {order_mode!r}")
            orders = receipt["orders"]
            if order_mode == "paired_both_orders":
                if orders != ["asc", "desc"] or not {
                    "asc_map",
                    "desc_map",
                    "asc_support",
                    "desc_support",
                    "metadata",
                } <= set(outputs.keys()):
                    raise ValueError("Both-order manifest incomplete")
            else:
                # AMENDMENT-001 B8/S2 experimental coarse single-order + 15% audit mode
                if s["stage"] != "COARSE":
                    raise ValueError("single_order_audit_15pct is permitted only on COARSE, never DEEP")
                audit = receipt.get("audit_policy")
                if not isinstance(audit, Mapping):
                    raise ValueError("single_order_audit_15pct requires explicit audit_policy dict")
                rate = float(audit.get("audit_rate", 0.0))
                if rate < SINGLE_ORDER_MIN_AUDIT_RATE or rate > 1.0:
                    raise ValueError("audit_rate must be in [0.15, 1.0]")
                require_sha256(audit.get("paired_calibration_key"))
                ambiguous = bool(audit.get("normal_orientation_ambiguous", True))
                audit_selected = bool(audit.get("audit_selected", False))
                if ambiguous or audit_selected:
                    # B8 rule: ambiguous normals or audit-selected units MUST run BOTH orders!
                    if orders != ["asc", "desc"] or not {
                        "asc_map",
                        "desc_map",
                        "asc_support",
                        "desc_support",
                        "metadata",
                    } <= set(outputs.keys()):
                        raise ValueError(
                            "Ambiguous normal or 15%-audit-selected unit must supply both asc and desc orders"
                        )
                else:
                    if not isinstance(orders, list) or len(orders) != 1 or orders[0] not in ("asc", "desc"):
                        raise ValueError("Single-order COARSE receipt must specify orders=['asc'] or ['desc']")
                    sel = orders[0]
                    if audit.get("selected_order") != sel:
                        raise ValueError("audit_policy.selected_order must match orders[0]")
                    needed = {f"{sel}_map", f"{sel}_support", "metadata"}
                    if not needed <= set(outputs.keys()):
                        raise ValueError(f"Single-order COARSE manifest missing roles: {sorted(needed - set(outputs))}")

        if s["stage"] == "CONTROL":
            if (
                receipt["calibration"] != s["calibration"]
                or type(receipt["control_pass"]) is not bool
                or type(receipt["valid_until"]) is not int
            ):
                raise ValueError("Control receipt key/verdict/valid_until invalid")
        if s["stage"] == "REVIEW" and receipt["review_status"] not in REVIEW_STATUSES:
            raise ValueError("Invalid scoped review_status")
    else:
        if receipt["failure"] not in FAILURES:
            raise ValueError(f"Failed/partial receipt requires enumerated failure, got {receipt['failure']!r}")

    return s


class CrossPlatformFileReceipts:
    """Atomic no-overwrite R1 file-receipt repository compatible with Windows & POSIX."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = private_root_cross_platform(root)
        self.store = CrossPlatformCAS(self.root / "cas")
        self.receipts = private_root_cross_platform(self.root / "receipts")

    def _path(self, spec: Mapping[str, Any], attempt: int) -> Path:
        return self.receipts / f"{job_key(spec)}.{attempt}.json"

    def publish(self, spec: Mapping[str, Any], attempt: int, receipt: Mapping[str, Any]) -> str:
        s = validate_r1_receipt(spec, attempt, receipt, self.store)
        envelope = {"schema": SCHEMA_FILE_RECEIPT, "spec": s, "receipt": dict(receipt)}
        obj = {**envelope, "sha256": digest_obj(envelope)}
        body = (canonical_json(obj) + "\n").encode("utf-8")
        dst = self._path(s, attempt)
        fd, tmp_name = tempfile.mkstemp(prefix=".partial-", dir=str(self.receipts))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_no_overwrite(tmp_path, dst, body)
            _fsync_directory_if_supported(self.receipts)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return obj["sha256"]

    def read(self, spec: Mapping[str, Any], attempt: int) -> dict[str, Any]:
        path = self._path(spec, attempt)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing completed file receipt")
        obj = json.loads(path.read_text(encoding="utf-8"))
        if set(obj.keys()) != {"schema", "spec", "receipt", "sha256"} or obj["schema"] != SCHEMA_FILE_RECEIPT:
            raise ValueError("Envelope schema mismatch")
        if (
            digest_obj({k: v for k, v in obj.items() if k != "sha256"}) != obj["sha256"]
            or obj["spec"] != validate_job_spec(spec)
        ):
            raise ValueError("Receipt hash or spec mismatch")
        validate_r1_receipt(spec, attempt, obj["receipt"], self.store)
        return obj

    def resume(self, spec: Mapping[str, Any], now: int | None = None) -> dict[str, Any]:
        ts = int(time.time()) if now is None else int(now)
        j = job_key(spec)
        rows: list[dict[str, Any]] = []
        for path in sorted(self.receipts.glob(f"{j}.*.json")):
            parts = path.name.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                raise ValueError(f"Malformed attempt filename: {path.name}")
            rows.append(self.read(spec, int(parts[1])))
        good: list[dict[str, Any]] = []
        for item in rows:
            r = item["receipt"]
            if r["outcome"] != "COMPLETE":
                continue
            if spec["stage"] == "CONTROL" and (not r["control_pass"] or r["valid_until"] <= ts):
                continue
            good.append(item)
        if len(good) > 1:
            raise ValueError("Multiple complete attempts require explicit operator reconciliation")
        return {
            "attempts": len(rows),
            "completed": len(good) == 1,
            "receipt": good[0] if good else None,
            "next_attempt": max((q["receipt"]["attempt"] for q in rows), default=0) + 1,
        }


# ---------------------------------------------------------------------------
# 4. Bridge from screen_scroll_rtx4060.py / stream_snap_render.py -> R1 Receipt
# ---------------------------------------------------------------------------

def emit_r1_receipt_from_rtx4060_summary(
    repo: CrossPlatformFileReceipts,
    *,
    summary: Mapping[str, Any],
    stage: str,
    unit_private_id: str,
    frame_hash: str,
    recipe_hash: str,
    calibration_key: str,
    deps: list[str],
    inputs: Mapping[str, str],
    artifact_payloads: Mapping[str, bytes],
    geometry_audit: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    grant: Mapping[str, int],
    attempt: int = 1,
    order_mode: str = "paired_both_orders",
    audit_policy: Mapping[str, Any] | None = None,
    pixel_center_frame: str = "vertex_aligned_p_times_scale",
    q33_mixed_region_recall: Mapping[str, Any] | None = None,
    q34_used_as_hard_gate: bool = False,
) -> dict[str, Any]:
    """Bridge an RTX 4060 ``summary.json`` into a semantically validated R1 receipt."""
    scroll = normalize_scroll_code(str(summary["scroll"]))
    num_slices = int(summary.get("num_slices", 31))
    voxel_um = float(summary["voxel_um"])
    support_px = int(summary["valid_ct_pixels"])
    nominal_px = int(summary.get("nominal_pixels", support_px))
    valid_cm2 = float(summary["valid_ct_cm2"])

    # Validate geometry contract (2D shear quarantine)
    geom_checked = validate_geometry_contract(
        {
            **geometry_audit,
            "snap_passes": int(summary.get("snap_passes", geometry_audit.get("snap_passes", 0))),
        }
    )

    # Derive exact signed slice specs from detectors present in summary
    det_map = summary.get("detectors") or {}
    det_specs: dict[str, dict[str, Any]] = {}
    orders_seen: set[str] = set()
    for det_key in sorted(det_map.keys()):
        if det_key.startswith("studio_"):
            ord_ = det_key.split("_", 1)[1]
            fam = "studio"
        elif det_key.startswith("ahn_s0_"):
            ord_ = det_key.split("_", 2)[2]
            fam = "ahn_s0"
        else:
            continue
        orders_seen.add(ord_)
        det_specs[det_key] = {
            "family": fam,
            "order": ord_,
            "slice_indices": expected_slice_indices(fam, ord_, num_slices=num_slices),
        }

    sampling_checked = validate_sampling_and_detector_contract(
        {
            "pixel_center_frame": pixel_center_frame,
            "voxel_um": voxel_um,
            "num_slices": num_slices,
            "mesh_index": (num_slices - 1) // 2,
            "slice_step_vx": float(summary.get("slice_step_vx", 1.0)),
            "source_support_pixels": support_px,
            "nominal_pixels": nominal_px,
            "valid_ct_cm2": valid_cm2,
            "padded_pixels_excluded_from_area": True,
            "missing_chunk_fill_excluded": True,
        },
        det_specs,
    )

    diag_checked = validate_diagnostics_and_telemetry(
        {
            "q34_used_as_hard_gate": q34_used_as_hard_gate,
            "fp_filter_used_as_sole_gate": False,
            "q34_diagnostic_score": (summary.get("tfce_q34") or {}).get("q34_score"),
            "q33_mixed_region_recall": dict(
                q33_mixed_region_recall
                or {
                    "reported": False,
                    "stratum": "unlabeled_unread_or_control",
                    "recall_at_budget": None,
                    "note": "No mixed-region ground truth on unread papyrus; tracked on labeled controls",
                }
            ),
        },
        telemetry,
    )

    # Store raw artifact payloads in CAS
    outputs: dict[str, str] = {}
    for role, payload in artifact_payloads.items():
        outputs[role] = repo.store.put(payload)

    # Compute actual resource vector from summary timings and bytes
    total_ms = int(round(float(summary.get("total_seconds", 0.0)) * 1000.0))
    gpu_s = sum(float(v.get("seconds", 0.0)) for v in det_map.values() if isinstance(v, Mapping))
    gpu_ms = int(round(gpu_s * 1000.0))
    cpu_ms = max(0, total_ms - gpu_ms)
    io_bytes = int(
        round(
            (float(summary.get("ct_mib_fetched", 0.0)) + float(summary.get("m7_mib_fetched", 0.0)))
            * 1024.0
            * 1024.0
        )
    )
    actual = {
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "human_ms": int(summary.get("human_ms", 0)),
        "io_bytes": io_bytes,
    }

    metadata_doc = {
        "schema": SCHEMA_ADAPTER_METADATA,
        "producer": "gemini_lane/industrial_funnel_adapter.py",
        "scroll": scroll,
        "geometry_contract": geom_checked,
        "sampling_contract": sampling_checked,
        "diagnostics_contract": diag_checked,
        "order_mode": order_mode,
        "orders_executed": sorted(orders_seen),
        "detector_summaries": {
            k: {
                key: v[key]
                for key in ("valid_pixels", "valid_cm2", "p99", "p995", "max", "frac_gt_05", "frac_gt_07", "seconds")
                if key in v
            }
            for k, v in det_map.items()
            if isinstance(v, Mapping)
        },
    }
    outputs["metadata"] = repo.store.put_json(metadata_doc)

    spec = validate_job_spec(
        {
            "scroll": scroll,
            "unit": unit_private_id,
            "frame": frame_hash,
            "stage": stage,
            "recipe": recipe_hash,
            "inputs": dict(inputs),
            "deps": list(deps),
            "calibration": calibration_key,
        }
    )

    # Determine outcome: if resource overrun, emit FAILED(RESOURCE_OVERRUN); never silent COMPLETE
    grant_vec = validate_resource_vector(grant)
    overrun = not resource_within(actual, grant_vec)
    receipt: dict[str, Any] = {
        "job": job_key(spec),
        "attempt": int(attempt),
        "outcome": "FAILED" if overrun else "COMPLETE",
        "worker_stopped": True,
        "grant": grant_vec,
        "actual": actual,
        "outputs": outputs,
        "failure": "RESOURCE_OVERRUN" if overrun else None,
    }
    if not overrun and stage in INFERENCE_STAGES:
        receipt["orders"] = sorted(orders_seen)
        receipt["order_mode"] = order_mode
        if audit_policy is not None:
            receipt["audit_policy"] = dict(audit_policy)

    receipt_sha = repo.publish(spec, attempt, receipt)
    return {
        "job_key": job_key(spec),
        "receipt_sha256": receipt_sha,
        "outcome": receipt["outcome"],
        "shear_quarantined": geom_checked["shear_quarantined"],
        "admissible_for_negative_claim": geom_checked["admissible_for_negative_claim"],
        "metadata_sha256": outputs["metadata"],
        "spec": spec,
        "receipt": receipt,
    }


def sanitize_receipt_for_public_aggregate(
    envelope: Mapping[str, Any],
    metadata_doc: Mapping[str, Any],
) -> dict[str, Any]:
    """Export strictly allowlisted scroll-level aggregate metrics (zero private IDs/coords)."""
    spec = envelope["spec"]
    rcpt = envelope["receipt"]
    geom = metadata_doc["geometry_contract"]
    samp = metadata_doc["sampling_contract"]
    diag = metadata_doc["diagnostics_contract"]
    public_row = {
        "scroll": spec["scroll"],
        "stage": spec["stage"],
        "outcome": rcpt["outcome"],
        "failure": rcpt["failure"],
        "order_mode": metadata_doc.get("order_mode", "paired_both_orders"),
        "orders_executed": metadata_doc.get("orders_executed", []),
        "valid_ct_cm2": samp["valid_ct_cm2"],
        "voxel_um": samp["voxel_um"],
        "pixel_center_frame": samp["pixel_center_frame"],
        "parameterization_method": geom["parameterization_method"],
        "shear_quarantined": geom["shear_quarantined"],
        "admissible_for_negative_claim": geom["admissible_for_negative_claim"],
        "q34_used_as_hard_gate": diag["q34_used_as_hard_gate"],
        "actual_resources": rcpt["actual"],
        "gpu_post_run_temp_c": diag["telemetry"]["post_run_temp_c"],
        "gpu_peak_allocated_mib": diag["telemetry"]["peak_allocated_mib"],
    }
    assert_public_export_clean(public_row)
    return public_row


# ---------------------------------------------------------------------------
# 5. S0 + Coverage + Geometry + Radiometry 19-Scroll Inventory Joiner
# ---------------------------------------------------------------------------

def parse_s0_priors_csv(
    csv_path: str | os.PathLike[str],
    *,
    verify_pinned_blob: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Parse and validate Claude's ``S0_priors_by_zband.csv`` (pinned ``67bd872``)."""
    path = Path(csv_path)
    raw = path.read_bytes()
    blob_sha1 = git_blob_sha1(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    if verify_pinned_blob and blob_sha1 != PINNED_S0_GIT_BLOB_SHA1:
        raise ValueError(
            f"Pinned S0 CSV blob SHA1 mismatch: got {blob_sha1}, expected {PINNED_S0_GIT_BLOB_SHA1}"
        )

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != S0_COLUMNS:
        raise ValueError(f"S0 CSV column mismatch: {reader.fieldnames}")

    groups: dict[str, dict[str, Any]] = {}
    row_count = 0
    for row in reader:
        row_count += 1
        scroll = normalize_scroll_code(row["scroll"])
        frac = round(float(row["z_fraction"]), 2)
        z_slice = int(row["z_slice"])
        slabs = int(row["slabs"])
        clean = int(row["clean_slabs"])
        rise = float(row["edge_rise_um_q75"])
        merged = float(row["merged_frac"])
        coh = float(row["coherence"])
        air = float(row["air_frac"])
        if (
            frac not in S0_LEVELS
            or z_slice < 0
            or not (1 <= slabs <= 4)
            or not (0 <= clean <= slabs)
            or not all(math.isfinite(v) for v in (rise, merged, coh, air))
            or rise < 0.0
            or any(not (0.0 <= v <= 1.0) for v in (merged, coh, air))
        ):
            raise ValueError(f"Invalid S0 row for {scroll} at z_fraction={frac}")
        grp = groups.setdefault(scroll, {"status": row["status"], "bands": {}})
        if frac in grp["bands"] or grp["status"] != row["status"]:
            raise ValueError(f"Duplicate z_fraction or status conflict for {scroll}")
        grp["bands"][frac] = {
            "z_fraction": frac,
            "z_slice": z_slice,
            "slabs": slabs,
            "clean_slabs": clean,
            "edge_rise_um_q75": rise,
            "merged_frac": merged,
            "coherence": coh,
            "air_frac": air,
        }

    meta = {
        "rows": row_count,
        "git_blob_sha1": blob_sha1,
        "sha256": sha256,
        "scrolls_total": len(groups),
    }
    return groups, meta


def build_s0_19scroll_inventory(
    *,
    s0_csv_path: str | os.PathLike[str],
    coverage_matrix_path: str | os.PathLike[str],
    screen_ledger_path: str | os.PathLike[str],
    grid_quality_path: str | os.PathLike[str],
    q22_audit_path: str | os.PathLike[str],
    triage_paths: Sequence[str | os.PathLike[str]] = (),
    verify_pinned_s0_blob: bool = True,
) -> dict[str, Any]:
    """Join Claude's S0 anatomy priors with Gemini's coverage, geometry, radiometry & triage.

    Outputs ONLY sanitized scroll-level aggregates (zero unread coordinates,
    zero ``uv_crop``, zero selected private ``z_slice`` values).
    """
    s0_groups, s0_meta = parse_s0_priors_csv(s0_csv_path, verify_pinned_blob=verify_pinned_s0_blob)
    cov = json.loads(Path(coverage_matrix_path).read_text(encoding="utf-8"))
    ledger = json.loads(Path(screen_ledger_path).read_text(encoding="utf-8"))
    gq = json.loads(Path(grid_quality_path).read_text(encoding="utf-8"))
    q22 = json.loads(Path(q22_audit_path).read_text(encoding="utf-8"))

    # Aggregate RTX 4060 ledger strictly to scroll-level totals
    ledger_by_scroll: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"meshes_screened": 0, "valid_ct_cm2": 0.0, "two_family_followups": 0, "coarse_seconds": 0.0}
    )
    for item in ledger.get("meshes", []):
        sc = normalize_scroll_code(item["scroll"])
        rec = ledger_by_scroll[sc]
        rec["meshes_screened"] += 1
        rec["valid_ct_cm2"] += float(item.get("valid_ct_cm2", 0.0))
        rec["coarse_seconds"] += float((item.get("coarse_pass") or {}).get("total_seconds", 0.0))
        if item.get("two_family_followup"):
            rec["two_family_followups"] += 1

    # Aggregate Exp T15 grid quality sets by scroll
    gq_by_scroll: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for st in gq.get("sets", []):
        sid = str(st.get("set_id", ""))
        pooled = st.get("pooled")
        if not pooled:
            continue
        m = re.match(r"pherc(\d+[ab]?)_", sid, flags=re.I)
        if not m:
            continue
        sc = "PHerc" + m.group(1).upper()
        s1 = pooled["shear_1cell"]
        gq_by_scroll[sc].append(
            {
                "set_id": sid,
                "meshes": int(pooled["meshes"]),
                "shear_1cell_abs_median_deg": float(s1["abs_median_deg"]),
                "shear_1cell_abs_p95_deg": float(s1["abs_p95_deg"]),
                "pct_area_gt_20deg": float(s1["pct_area_gt_20deg"]),
                "area_inflation_true_over_nominal": float(pooled["area_inflation_true_over_nominal"]),
            }
        )

    # Index Q22 radiometric audit by scroll
    q22_by_scroll: dict[str, dict[str, Any]] = {}
    for row in q22.get("scrolls", []):
        sc = normalize_scroll_code(row["scroll_id"])
        aff = row.get("affine_to_ref_control", {})
        chk = row.get("studio_envelope_check", {})
        post = row.get("post_nyul_udupa_validation", {})
        q22_by_scroll[sc] = {
            "flagged_outside_studio_envelope": bool(chk.get("flagged", False)),
            "flag_reasons": list(chk.get("flag_reasons", [])),
            "affine_gain_vs_ref": round(float(aff.get("gain", 1.0)), 4),
            "affine_bias_vs_ref": round(float(aff.get("bias", 0.0)), 2),
            "nonzero_ct_p50": float((row.get("nonzero_ct") or {}).get("p50", 0.0)),
            "clip_200_frac": round(float((row.get("nonzero_ct") or {}).get("frac_ge_200", 0.0)), 5),
            "flagged_after_nyul_udupa": bool(post.get("flagged_after_nyul", False)),
        }

    # Aggregate CPU on-sheet triage JSONs strictly to scroll-level medians/totals (no mesh IDs/coords)
    triage_by_scroll: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"f_parallel": [], "on_sheet_after_snap": [], "valid_ct_cm2": []}
    )
    for tpath in triage_paths:
        tdata = json.loads(Path(tpath).read_text(encoding="utf-8"))
        if "scrolls" in tdata:
            for sc_block in tdata["scrolls"]:
                sc = normalize_scroll_code(sc_block["scroll"])
                for sh in sc_block.get("sheets", []):
                    triage_by_scroll[sc]["f_parallel"].append(float(sh["onsheet_score"]["f_parallel"]))
                    triage_by_scroll[sc]["on_sheet_after_snap"].append(float(sh["on_sheet_after_snap"]))
                    triage_by_scroll[sc]["valid_ct_cm2"].append(float(sh["valid_ct_cm2"]))
        elif "targets" in tdata:
            for tgt in tdata["targets"]:
                sc = normalize_scroll_code(tgt["scroll"])
                triage_by_scroll[sc]["f_parallel"].append(float(tgt["onsheet_score"]["f_parallel"]))
                triage_by_scroll[sc]["on_sheet_after_snap"].append(float(tgt["on_sheet_after_snap"]))
                triage_by_scroll[sc]["valid_ct_cm2"].append(float(tgt["valid_ct_cm2"]))

    eligible_rows: list[dict[str, Any]] = []
    for entry in sorted(cov["scrolls"], key=lambda x: x["scroll"]):
        sc = normalize_scroll_code(entry["scroll"])
        if sc not in s0_groups:
            raise ValueError(f"Eligible scroll {sc} missing from S0 CSV")
        s0 = s0_groups[sc]
        bands = list(s0["bands"].values())
        slabs = sum(b["slabs"] for b in bands)
        clean = sum(b["clean_slabs"] for b in bands)
        missing_levels = len(S0_LEVELS) - len(bands)
        clean_frac = clean / slabs if slabs else 0.0
        s0_route = "GEOMETRY_FIRST" if clean == 0 else "PRIORITIZE_WITHIN_SCROLL"

        gq_sets = gq_by_scroll.get(sc, [])
        max_shear_p95 = max((x["shear_1cell_abs_p95_deg"] for x in gq_sets), default=None)
        max_gt20_area = max((x["pct_area_gt_20deg"] for x in gq_sets), default=None)
        shear_quarantined = bool(
            (max_shear_p95 is not None and max_shear_p95 > SHEAR_P95_QUARANTINE_DEG)
            or (max_gt20_area is not None and max_gt20_area > SHEAR_PCT_AREA_GT_20DEG_QUARANTINE)
        )

        rad = q22_by_scroll.get(sc)
        led = ledger_by_scroll.get(sc)
        tri = triage_by_scroll.get(sc)

        has_public_meshes = bool(
            int(entry.get("pscamillo_meshes", 0)) > 0
            or int(entry.get("s3_segments_count", 0)) > 0
            or int(entry.get("hf_spiral_fit_runs", 0)) > 0
            or int(entry.get("iyandopico_verified_patches", 0)) > 0
        )

        if s0_route == "GEOMETRY_FIRST":
            next_stage = "S0_GEOMETRY_FIRST_REPAIR"
        elif shear_quarantined:
            next_stage = "S2_2D_FLATTEN_BEFORE_SCREEN"
        elif not has_public_meshes:
            next_stage = "S1_CPU_GROWPATCH_WINDOWS"
        else:
            next_stage = "S2_COARSE_SCREEN_EXISTING_CHARTS"

        eligible_rows.append(
            {
                "scroll": sc,
                "voxel_um": float(entry["voxel_um"]),
                "has_2_4um_rescan": bool(entry.get("has_2_4um_rescan", False)),
                "has_s3_m7": bool(entry.get("s3_has_m7_and_normal_grids", False)),
                "has_public_meshes": has_public_meshes,
                "public_mesh_counts": {
                    "s3_segments": int(entry.get("s3_segments_count", 0)),
                    "pscamillo_meshes": int(entry.get("pscamillo_meshes", 0)),
                    "hf_spiral_fit_runs": int(entry.get("hf_spiral_fit_runs", 0)),
                    "iyandopico_patches": int(entry.get("iyandopico_verified_patches", 0)),
                },
                "gemini_priority": str(entry.get("gemini_priority", "")),
                "s0_anatomy": {
                    "sampled_levels": len(bands),
                    "missing_levels": missing_levels,
                    "slabs": slabs,
                    "clean_slabs": clean,
                    "clean_fraction": round(clean_frac, 4),
                    "zero_clean_levels": sum(1 for b in bands if b["clean_slabs"] == 0),
                    "median_edge_rise_um_q75": round(statistics.median(b["edge_rise_um_q75"] for b in bands), 2),
                    "median_merged_frac": round(statistics.median(b["merged_frac"] for b in bands), 4),
                    "median_coherence": round(statistics.median(b["coherence"] for b in bands), 4),
                    "median_air_frac": round(statistics.median(b["air_frac"] for b in bands), 4),
                    "route": s0_route,
                    "unsampled_band_survey_required": missing_levels > 0,
                },
                "geometry_audit_t15": {
                    "audited_sets": len(gq_sets),
                    "max_shear_1cell_abs_p95_deg": round(max_shear_p95, 3) if max_shear_p95 is not None else None,
                    "max_pct_area_gt_20deg": round(max_gt20_area, 3) if max_gt20_area is not None else None,
                    "shear_quarantined": shear_quarantined,
                },
                "radiometry_q22": rad,
                "cpu_triage_aggregate": (
                    {
                        "triaged_targets": len(tri["f_parallel"]),
                        "median_f_parallel": round(statistics.median(tri["f_parallel"]), 4),
                        "median_on_sheet_after_snap": round(statistics.median(tri["on_sheet_after_snap"]), 4),
                        "triaged_valid_ct_cm2": round(sum(tri["valid_ct_cm2"]), 4),
                    }
                    if tri and tri["f_parallel"]
                    else None
                ),
                "rtx4060_ledger_aggregate": (
                    {
                        "meshes_screened": int(led["meshes_screened"]),
                        "valid_ct_cm2": round(led["valid_ct_cm2"], 4),
                        "two_family_followups": int(led["two_family_followups"]),
                        "coarse_seconds": round(led["coarse_seconds"], 2),
                    }
                    if led and led["meshes_screened"] > 0
                    else {"meshes_screened": 0, "valid_ct_cm2": 0.0, "two_family_followups": 0, "coarse_seconds": 0.0}
                ),
                "recommended_next_stage": next_stage,
            }
        )

    controls_summary: list[dict[str, Any]] = []
    for sc, grp in sorted(s0_groups.items()):
        if grp["status"] == "eligible, unread":
            continue
        bands = list(grp["bands"].values())
        slabs = sum(b["slabs"] for b in bands)
        clean = sum(b["clean_slabs"] for b in bands)
        controls_summary.append(
            {
                "scroll": sc,
                "status": grp["status"],
                "sampled_levels": len(bands),
                "slabs": slabs,
                "clean_slabs": clean,
                "clean_fraction": round(clean / slabs if slabs else 0.0, 4),
                "radiometry_q22_flagged": (
                    q22_by_scroll[sc]["flagged_outside_studio_envelope"] if sc in q22_by_scroll else None
                ),
            }
        )

    joined = {
        "schema": SCHEMA_S0_JOIN,
        "s0_source": s0_meta,
        "scope_note": (
            "Sanitized scroll-level aggregates only (zero unread coordinates, zero crop bounds, "
            "zero selected axial band indices). S0 clean_fraction orders bands WITHIN a scroll only; "
            "it is not a cross-scroll ink probability."
        ),
        "summary": {
            "eligible_scrolls": len(eligible_rows),
            "read_controls": len(controls_summary),
            "geometry_first_scrolls": [r["scroll"] for r in eligible_rows if r["s0_anatomy"]["route"] == "GEOMETRY_FIRST"],
            "shear_quarantined_scrolls": [r["scroll"] for r in eligible_rows if r["geometry_audit_t15"]["shear_quarantined"]],
            "q22_flagged_eligible_scrolls": [
                r["scroll"]
                for r in eligible_rows
                if r["radiometry_q22"] and r["radiometry_q22"]["flagged_outside_studio_envelope"]
            ],
            "zero_public_mesh_scrolls": [r["scroll"] for r in eligible_rows if not r["has_public_meshes"]],
            "total_rtx4060_screened_cm2": round(sum(r["rtx4060_ledger_aggregate"]["valid_ct_cm2"] for r in eligible_rows), 4),
            "next_stage_counts": {
                stage_name: sum(1 for r in eligible_rows if r["recommended_next_stage"] == stage_name)
                for stage_name in sorted({r["recommended_next_stage"] for r in eligible_rows})
            },
        },
        "eligible_scrolls": eligible_rows,
        "read_controls": controls_summary,
    }
    assert_public_export_clean(joined)
    return joined


# ---------------------------------------------------------------------------
# 6. CPU-Only 64-vx Cell Occupancy vs <=3-vx Normal Correspondence Dedup
# ---------------------------------------------------------------------------

def _compute_grid_normals_simple(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit normals over valid 2D tifxyz grids without sentinel contamination."""
    h, w = valid.shape
    pts = np.zeros((h, w, 3), dtype=np.float64)
    pts[valid, 0] = x[valid]
    pts[valid, 1] = y[valid]
    pts[valid, 2] = z[valid]

    def _diff(axis: int) -> tuple[np.ndarray, np.ndarray]:
        d = np.zeros((h, w, 3), dtype=np.float64)
        vf = np.zeros((h, w), dtype=bool)
        vb = np.zeros((h, w), dtype=bool)
        fwd = np.zeros((h, w, 3), dtype=np.float64)
        bwd = np.zeros((h, w, 3), dtype=np.float64)
        if axis == 1 and w >= 2:
            vf[:, :-1] = valid[:, :-1] & valid[:, 1:]
            vb[:, 1:] = valid[:, 1:] & valid[:, :-1]
            fwd[:, :-1] = pts[:, 1:] - pts[:, :-1]
            bwd[:, 1:] = pts[:, 1:] - pts[:, :-1]
        elif axis == 0 and h >= 2:
            vf[:-1, :] = valid[:-1, :] & valid[1:, :]
            vb[1:, :] = valid[1:, :] & valid[:-1, :]
            fwd[:-1, :] = pts[1:, :] - pts[:-1, :]
            bwd[1:, :] = pts[1:, :] - pts[:-1, :]
        both = vf & vb
        d[both] = 0.5 * (fwd[both] + bwd[both])
        d[vf & ~vb] = fwd[vf & ~vb]
        d[vb & ~vf] = bwd[vb & ~vf]
        return d, (vf | vb)

    du, ok_u = _diff(1)
    dv, ok_v = _diff(0)
    nrm = np.cross(du, dv)
    ln = np.linalg.norm(nrm, axis=-1)
    ok = valid & ok_u & ok_v & (ln > 1e-9)
    nrm[ok] /= ln[ok, None]
    nrm[~ok] = 0.0
    return nrm, ok


def _upsample_valid_quads(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray, factor: int = 4
) -> np.ndarray:
    """Bilinearly upsample valid 2D tifxyz quads by ``factor`` to reduce lateral grid-step bias."""
    h, w = valid.shape
    if h < 2 or w < 2:
        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float64)
        return pts
    qok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    if not qok.any():
        return np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float64)
    oh = (h - 1) * factor + 1
    ow = (w - 1) * factor + 1
    gv = np.arange(oh, dtype=np.float64) / factor
    gu = np.arange(ow, dtype=np.float64) / factor
    Gv, Gu = np.meshgrid(gv, gu, indexing="ij")
    iv = np.minimum(np.floor(Gv).astype(np.int64), h - 2)
    iu = np.minimum(np.floor(Gu).astype(np.int64), w - 2)
    mask = qok[iv, iu]
    pts = np.stack([x, y, z], axis=-1).astype(np.float64)
    fv = (Gv - iv)[..., None]
    fu = (Gu - iu)[..., None]
    interp = (
        (1.0 - fv) * (1.0 - fu) * pts[iv, iu]
        + (1.0 - fv) * fu * pts[iv, iu + 1]
        + fv * (1.0 - fu) * pts[iv + 1, iu]
        + fv * fu * pts[iv + 1, iu + 1]
    )
    return interp[mask]


def evaluate_normal_correspondence_dedup(
    surf_a: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    surf_b: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    cell_side: int = 64,
    max_normal_vox: float = 3.0,
    max_lateral_vox: float = 4.0,
    min_abs_cos: float = 0.7,
    upsample_factor: int = 4,
) -> dict[str, Any]:
    """Compare 64-voxel cell occupancy vs <=3-voxel normal correspondence from A -> B.

    Each surface is ``(x, y, z, valid)`` of matching 2D arrays in native voxel
    coordinates. Uses ``upsample_factor`` bilinear quad sampling on surface B so
    20-vx coarse mesh grids do not artificially inflate lateral offsets.
    """
    xa, ya, za, va = surf_a
    xb, yb, zb, vb = surf_b
    na, ok_a = _compute_grid_normals_simple(xa, ya, za, va)
    nb, ok_b = _compute_grid_normals_simple(xb, yb, zb, vb)
    pts_a = np.stack([xa[ok_a], ya[ok_a], za[ok_a]], axis=-1).astype(np.float64)
    nrm_a = na[ok_a]
    pts_b = np.stack([xb[ok_b], yb[ok_b], zb[ok_b]], axis=-1).astype(np.float64)
    nrm_b = nb[ok_b]
    if len(pts_a) == 0 or len(pts_b) == 0:
        raise ValueError("Both surfaces must contain usable vertices with finite normals")

    dense_a = _upsample_valid_quads(xa, ya, za, va, factor=upsample_factor)
    dense_b = _upsample_valid_quads(xb, yb, zb, vb, factor=upsample_factor)
    cells_a = set(map(tuple, np.floor(dense_a / cell_side).astype(np.int64).tolist()))
    cells_b = set(map(tuple, np.floor(dense_b / cell_side).astype(np.int64).tolist()))
    a_vcells = np.floor(pts_a / cell_side).astype(np.int64)
    same_cell_64 = np.array([tuple(c) in cells_b for c in a_vcells], dtype=bool)
    cell_jaccard = len(cells_a & cells_b) / max(1, len(cells_a | cells_b))

    tree_dense_b = cKDTree(dense_b)
    d_dense, idx_dense = tree_dense_b.query(pts_a, k=1)
    diff = dense_b[idx_dense] - pts_a
    dn = np.einsum("ij,ij->i", diff, nrm_a)
    abs_dn = np.abs(dn)
    dlat = np.sqrt(np.maximum(0.0, d_dense ** 2 - dn ** 2))

    tree_vert_b = cKDTree(pts_b)
    _, idx_vert = tree_vert_b.query(pts_a, k=1)
    cos_ang = np.abs(np.einsum("ij,ij->i", nrm_a, nrm_b[idx_vert]))

    euclid_match = d_dense <= max_normal_vox
    normal_match = (abs_dn <= max_normal_vox) & (dlat <= max_lateral_vox) & (cos_ang >= min_abs_cos)
    cell_and_normal = same_cell_64 & normal_match

    return {
        "n_vertices_a": int(len(pts_a)),
        "n_vertices_b": int(len(pts_b)),
        "cell_side_vox": int(cell_side),
        "cell_64vx_same_vert_frac": round(float(same_cell_64.mean()), 4),
        "cell_64vx_jaccard": round(float(cell_jaccard), 4),
        "surface_dist_median_vox": round(float(np.median(d_dense)), 3),
        "surface_dist_p10_vox": round(float(np.percentile(d_dense, 10)), 3),
        "euclid_le_threshold_frac": round(float(euclid_match.mean()), 4),
        "normal_correspondence_frac": round(float(normal_match.mean()), 4),
        "cell64_and_normal_correspondence_frac": round(float(cell_and_normal.mean()), 4),
        "thresholds": {
            "max_normal_vox": float(max_normal_vox),
            "max_lateral_vox": float(max_lateral_vox),
            "min_abs_cos": float(min_abs_cos),
            "upsample_factor": int(upsample_factor),
        },
    }
