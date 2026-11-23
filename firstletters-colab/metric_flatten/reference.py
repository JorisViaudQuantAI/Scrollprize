"""Bounded, pinned source preparation for the public PHerc. 306B control.

Only ``prepare(cache)`` performs network access. The experiment runner must freeze
its protocol before calling it, with a previously nonexistent cache directory
outside every Git checkout. Importing this module never fetches or reads arrays.

Public APIs:
    prepare(cache) -> acquisition receipt dict
    reconstruct_touchup(cache) -> (float64 XYZ, neutral metadata)
    load_published_xyz(cache) -> float64 XYZ
    load_arap_functions(cache) -> {"tris_of", "arap", "procrustes"}

Invalid XYZ nodes are NaN, matching the publisher's ``p1geom.load``. The touch-up
adapter preserves native input dtypes, operation order, SciPy defaults, and the
float32 TIFF roundtrip. Target comparison concerns decoded values, not TIFF file
serialization. No CT, ink model, annotation or historical local path is used.

The touch-up arithmetic and extracted ARAP functions are from
SirProdigle/pherc0306b-first-letters at the revision below, under the MIT license.
Preserve the upstream copyright and permission notice in LICENSE.upstream.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from types import FunctionType
from typing import Any
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import splu


REVISION = "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc"
REPOSITORY = "SirProdigle/pherc0306b-first-letters"
MANIFEST_SHA256 = "087464bd4c3163ce7314779aa05d917dcacac5791a00280b6bc57073473f1ba2"
MAX_BODY_BYTES = 16_000_000
MAX_REQUESTS = 20
SOURCE = "submission/code/surface"
INPUTS = f"{SOURCE}/inputs"
TARGET = "submission/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc"
_MANIFEST = Path(__file__).with_name("source_manifest.json")
_RECEIPT = "prepared.json"


class SourceError(RuntimeError):
    """A source integrity, bounded acquisition or format check failed."""


def _manifest() -> dict[str, Any]:
    body = _MANIFEST.read_bytes()
    if hashlib.sha256(body).hexdigest() != MANIFEST_SHA256:
        raise SourceError("source_manifest.json differs from its frozen hash")
    manifest = json.loads(body)
    if manifest["revision"] != REVISION or manifest["repository"] != REPOSITORY:
        raise SourceError("unexpected source repository or revision")
    files = manifest["files"]
    if len(files) != 17 or len(files) > MAX_REQUESTS:
        raise SourceError("unexpected source file count")
    paths = set()
    for entry in files:
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or str(path) != entry["path"]:
            raise SourceError("unsafe manifest path")
        if entry["path"] in paths:
            raise SourceError("duplicate manifest path")
        paths.add(entry["path"])
        expected_url = (
            f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{path}"
        )
        if entry["url"] != expected_url or not 0 < entry["bytes"] <= MAX_BODY_BYTES:
            raise SourceError("unexpected source URL or byte count")
    expected_total = sum(entry["bytes"] for entry in files)
    if expected_total != manifest["total_advertised_bytes"] or expected_total > MAX_BODY_BYTES:
        raise SourceError("source manifest exceeds the total body budget")
    return manifest


def _outside_git(cache: str | Path) -> Path:
    path = Path(cache).expanduser().absolute()
    if path.is_symlink():
        raise SourceError("cache must not be a symbolic link")
    path = path.resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise SourceError("source cache must be outside Git checkouts")
    return path


def _check_body(body: bytes, entry: dict[str, Any]) -> dict[str, Any]:
    if len(body) != entry["bytes"]:
        raise SourceError(f"unexpected byte count: {entry['path']}")
    sha256 = hashlib.sha256(body).hexdigest()
    blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    if blob != entry["git_blob_sha1"]:
        raise SourceError(f"Git blob hash mismatch: {entry['path']}")
    if "sha256" in entry and sha256 != entry["sha256"]:
        raise SourceError(f"SHA-256 mismatch: {entry['path']}")
    return {"path": entry["path"], "bytes": len(body), "sha256": sha256, "git_blob_sha1": blob}


def _verified_bytes(cache: str | Path, relative_path: str) -> bytes:
    """Read at most the expected length plus one, then verify before any parse."""
    manifest = _manifest()
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if relative_path not in entries:
        raise SourceError("requested path is not a pinned asset")
    root = _outside_git(cache)
    path = root / relative_path
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SourceError("cached asset escapes its directory")
    entry = entries[relative_path]
    with path.open("rb") as handle:
        body = handle.read(entry["bytes"] + 1)
    _check_body(body, entry)
    return body


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SourceError("redirects are forbidden for pinned source acquisition")


def prepare(cache: str | Path) -> dict[str, Any]:
    """Acquire all 17 assets once, after protocol freeze, into a fresh directory.

    At most 16,000,000 total body bytes and 20 HTTP requests are allowed. There
    are no redirects, retries or fallback URLs. On failure the incomplete fresh
    directory remains for diagnosis; it is never reused by this function.
    Every file is checked against its Git blob hash and any published SHA-256
    before being saved. No downloaded array or source program is parsed here.
    """
    manifest = _manifest()
    root = _outside_git(cache)
    if root.exists() or not root.parent.is_dir():
        raise SourceError("cache must be new and its parent directory must exist")
    root.mkdir(mode=0o700)
    opener = urllib.request.build_opener(_NoRedirect())
    request_count = 0
    body_bytes = 0
    verified = []
    try:
        for entry in manifest["files"]:
            if request_count >= MAX_REQUESTS:
                raise SourceError("request budget exhausted")
            request = urllib.request.Request(
                entry["url"],
                headers={"Accept-Encoding": "identity", "User-Agent": "Scrollprize-public-control/1"},
                method="GET",
            )
            request_count += 1
            with opener.open(request, timeout=30) as response:
                if response.status != 200 or response.geturl() != entry["url"]:
                    raise SourceError("unexpected source response")
                encoding = response.headers.get("Content-Encoding", "identity").lower()
                if encoding != "identity":
                    raise SourceError("encoded response bodies are not allowed")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != entry["bytes"]:
                    raise SourceError(f"unexpected response length: {entry['path']}")
                chunks = []
                count = 0
                while True:
                    remaining = MAX_BODY_BYTES - body_bytes
                    if remaining <= 0:
                        raise SourceError("total body budget exhausted")
                    # One extra byte detects overlong responses within the total budget.
                    limit = min(65_536, entry["bytes"] + 1 - count, remaining)
                    chunk = response.read(limit)
                    if not chunk:
                        break
                    count += len(chunk)
                    body_bytes += len(chunk)
                    if count > entry["bytes"]:
                        raise SourceError(f"overlong response: {entry['path']}")
                    chunks.append(chunk)
                body = b"".join(chunks)
            verified.append(_check_body(body, entry))
            target = root / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(body)
        receipt = {
            "status": "complete",
            "revision": REVISION,
            "manifest_sha256": MANIFEST_SHA256,
            "requests": request_count,
            "body_bytes": body_bytes,
            "retries": 0,
            "redirects": 0,
            "files": verified,
        }
        with (root / _RECEIPT).open("x") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
        return receipt
    except Exception as error:
        with (root / "prepare_failed.json").open("x") as handle:
            json.dump({"status": "failed", "requests": request_count, "body_bytes": body_bytes,
                       "error_type": type(error).__name__}, handle, indent=2)
            handle.write("\n")
        raise


def _npz(cache: str | Path, name: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    body = _verified_bytes(cache, f"{INPUTS}/{name}.npz")
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        if not set(keys).issubset(archive.files):
            raise SourceError(f"missing input keys in {name}.npz")
        arrays = {key: archive[key] for key in keys}
    for key, value in arrays.items():
        if value.dtype.kind not in "biuf":
            raise SourceError(f"non-numeric input: {name}.{key}")
    return arrays


def _publisher_load_policy(P: np.ndarray) -> np.ndarray:
    """Exact invalid-node policy from the pinned p1geom.load function."""
    P = P.astype(np.float64)
    bad = (P[..., 0] == -1) | (P[..., 2] <= 0) | ~np.isfinite(P).all(-1)
    P[bad] = np.nan
    return P


def reconstruct_touchup(cache: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Recreate the publisher's pre-ARAP surface; never fetch or execute scripts.

    The original calculation is reproduced below with native input dtypes.
    It includes the published centroid ROI, cached third lane, SciPy defaults,
    and float32 output rounding. Diagnostic images and historical metadata I/O
    are omitted. Return XYZ has the exact subsequent p1geom.load invalid policy.
    """
    _verified_bytes(cache, f"{SOURCE}/touchup.py")
    _verified_bytes(cache, f"{SOURCE}/p1geom.py")
    C = _npz(cache, "carrier", ("P", "n", "ok", "u", "v", "CELL", "x_z84_w074", "x_z84o_w074"))
    T = _npz(cache, "track", ("V0_orig",))
    T3 = _npz(cache, "track_m7", ("V2_trackP",))
    l3 = np.load(io.BytesIO(_verified_bytes(cache, f"{INPUTS}/x_z99_w074.npy")), allow_pickle=False)
    P, n, ok, U, V = C["P"], C["n"], C["ok"], C["u"], C["v"]
    cell = float(C["CELL"])
    v0 = T["V0_orig"]
    v3 = T3["V2_trackP"]
    l1, l2 = C["x_z84_w074"], C["x_z84o_w074"]
    if P.ndim != 3 or P.shape[-1] != 3 or n.shape != P.shape or min(P.shape[:2]) < 2:
        raise SourceError("unexpected carrier XYZ or normal shape")
    shape = P.shape[:2]
    if any(a.shape != shape for a in (ok, U, V, v0, v3, l1, l2, l3)):
        raise SourceError("inconsistent carrier, mask or offset shapes")
    if ok.dtype.kind != "b" or l3.dtype.kind not in "iuf":
        raise SourceError("unexpected mask or third-lane dtype")
    if not np.isfinite(cell) or not np.isclose(cell, 20 / (1000.0 / 8.64), rtol=0, atol=1e-7):
        raise SourceError("unexpected carrier pitch")

    # Published numerical block, with the cached l3 supplied above. In particular,
    # preserve NaN propagation in w*v3 + (1-w)*v0, even where a weight is zero.
    ONLY = (12.0, 30.0, -4.0, 4.0)
    agree_lanes = np.isfinite(l1) & np.isfinite(l2) & (np.abs(l1 - l2) <= 4)
    lane = np.where(agree_lanes, (l1 + l2) / 2, np.nan)
    cons = np.where(np.isfinite(lane), lane, np.nan)
    only99 = ~np.isfinite(cons) & np.isfinite(l3)
    cons = np.where(only99, l3, cons)
    m = np.isfinite(cons) & (np.abs(v3 - cons) <= 4) & (np.abs(v0 - cons) > 4) & ok
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_dilation(m, iterations=1) & ok
    lab, nl = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, nl + 1)) * cell ** 2
    keep = np.isin(lab, 1 + np.nonzero(sizes >= 0.5)[0])
    m = keep
    lab2, n2 = ndi.label(m)
    cen = ndi.center_of_mass(m, lab2, range(1, n2 + 1))
    ids = [k + 1 for k, (ci, cj) in enumerate(cen)
           if ONLY[0] <= -36 + cj * cell <= ONLY[1] and ONLY[2] <= -21 + ci * cell <= ONLY[3]]
    m = np.isin(lab2, ids)
    w = np.clip(ndi.gaussian_filter(m.astype(float), 0.35 / cell / 2) * 2, 0, 1)
    w = np.maximum(w, m)
    f = w * v3 + (1 - w) * v0
    Q = P + f[..., None] * n
    Q[~ok] = -1

    # Equivalent decoded values to per-channel float32 TIFF write, then load.
    result = _publisher_load_policy(Q.astype(np.float32))
    valid = np.isfinite(result).all(-1)
    if not valid.any():
        raise SourceError("touch-up produced no valid XYZ nodes")
    meta = {
        "format": "tifxyz",
        "scale": [0.05, 0.05],
        "source_voxel_um": 8.64,
        "carrier_step_voxels": 20,
        "carrier_step_mm": cell,
        "carrier_origin_uv_mm": [-36.0, -21.0],
        "coordinate_order": "xyz",
        "invalid_return_value": "NaN in all three coordinates",
        "bbox": [result[valid].min(0).tolist(), result[valid].max(0).tolist()],
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_manifest_sha256": MANIFEST_SHA256,
        "numeric_adapter": "published touch-up arithmetic; float32 roundtrip; p1geom.load invalid policy",
        "metadata_adapter": "neutral metadata; historical source metadata not required",
        "touchup": {"centroid_region_uv_mm": list(ONLY), "mask_nodes": int(m.sum()),
                    "mask_area_mm2": float(m.sum() * cell ** 2), "kept_components": len(ids)},
    }
    return result, meta


def load_published_xyz(cache: str | Path) -> np.ndarray:
    """Read the verified published full XYZ with Pillow and upstream validity.

    The returned float64 values are decoded from the source float32 TIFFs. This
    permits numerical comparison independent of the adapter's TIFF serializer.
    """
    _verified_bytes(cache, f"{SOURCE}/p1geom.py")
    channels = []
    for channel in "xyz":
        body = _verified_bytes(cache, f"{TARGET}/{channel}.tif")
        with Image.open(io.BytesIO(body)) as image:
            if image.format != "TIFF" or getattr(image, "n_frames", 1) != 1 or image.mode != "F":
                raise SourceError("expected one float32 TIFF page per XYZ channel")
            values = np.array(image, copy=True)
        if values.ndim != 2 or values.dtype != np.float32:
            raise SourceError("expected a two-dimensional float32 XYZ channel")
        channels.append(values)
    if any(channel.shape != channels[0].shape for channel in channels):
        raise SourceError("published XYZ channel shapes differ")
    return _publisher_load_policy(np.stack(channels, -1))


def load_arap_functions(cache: str | Path) -> dict[str, FunctionType]:
    """Compile only three exact function ASTs from verified, pinned source.

    No upstream import, assignment, helper module, CLI or main function executes.
    The globals contain only NumPy, SciPy sparse/splu and the small builtin set
    used by these three published functions. NumPy's C-level ndarray methods can
    request lazy imports through the caller's builtins; those imports are limited
    to NumPy and SciPy. Function arithmetic is unchanged. This is a pinned-source
    loader, not a sandbox for arbitrary untrusted Python.
    """
    body = _verified_bytes(cache, f"{SOURCE}/reflatten.py")
    tree = ast.parse(body.decode("utf-8"), filename="pinned_306b_reflatten.py")
    names = ("tris_of", "arap", "procrustes")
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
        raise SourceError("missing or duplicate published ARAP function")
    for node in nodes:
        if node.decorator_list or any(isinstance(part, (ast.Import, ast.ImportFrom)) for part in ast.walk(node)):
            raise SourceError("unexpected decorator or nested import in ARAP source")
    def numeric_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name.split(".", 1)[0] not in {"numpy", "scipy"}:
            raise SourceError("unexpected import requested by numerical kernel")
        return __import__(name, globals, locals, fromlist, level)

    namespace = {
        "__builtins__": {"int": int, "len": len, "range": range, "__import__": numeric_import},
        "np": np,
        "sparse": sparse,
        "splu": splu,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, "pinned_306b_reflatten.py", "exec"), namespace)
    return {name: namespace[name] for name in names}
