"""Pinned, bounded acquisition of the author PHerc0841 holdout geometry.

No CT or checkpoint is downloaded here.  The five required geometry members
occur in the first 8 MiB of a 1.377 GB gzip archive.  Every source and extracted
member has a prospective hard-coded SHA-256; extraction never trusts tar paths.
Raw author records remain in the external cache.  Returned summaries contain
public provenance and scientific parameters, not author-specific absolute paths.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen


DATASET_ID = "841-expanded-layer-03-approved-r000001"
HF_REVISION = "ad919d288ede1c4daf92725288bb6fe29c558591"
HF_ROOT = (
    "https://huggingface.co/datasets/paranke/"
    "vesuvius-814-841-9b-studio-training-data/resolve/" + HF_REVISION + "/"
)
PREPARED_ROOT = HF_ROOT + "donor-studio-resunet-scratch-20260926/prepared/" + DATASET_ID + "/"
ARCHIVE_URL = HF_ROOT + "backup/remote-artifacts.tar.gz"
ARCHIVE_BYTES = 1376603369
ARCHIVE_SHA256 = "7e1098556ed3229f2dc0d99565039c19a299790af0bb51b2ab1c5c8cf5e63dd2"
PREFIX_BYTES = 8388608
PREFIX_SHA256 = "b5d6f3cd285e5e168fccddc546659a2db6ef7d1be1f8c8fe53954bf36e454c69"
PREFIX_FILE = "remote-artifacts_prefix_8m.gz.part"
MAX_UNCOMPRESSED_PREFIX_BYTES = 32 * 1024**2

SOURCES = {
    "selection": {
        "url": PREPARED_ROOT + "selection.json",
        "file": "841_selection.json", "bytes": 17271,
        "sha256": "1d52f2feb76c53b408612ff5b0b3c5cca6a7a8ffeba27e60c79047a77b9fe83b",
    },
    "render": {
        "url": PREPARED_ROOT + "render.json",
        "file": "841_render.json", "bytes": 1433,
        "sha256": "bc2bdfc1280ea8fb837815454c10d5f2125b4c2f45d0ee313bf85a4ac330329e",
    },
    "file_index": {
        "url": HF_ROOT + "backup/file-index.json",
        "file": "backup_file-index.json", "bytes": 12431639,
        "sha256": "e81e3d6971e3bd4a52f8872e7d931c8112c503ded0b3ca1fda25e01c22bca40b",
    },
}
_MESH_ARCHIVE_ROOT = (
    "remote/home/ubuntu/donor-labels-20260906/expansion/approved-batches/"
    "evening-20260908/" + DATASET_ID + "/mesh/"
)
# Archive-member paths are an internal read whitelist, never output destinations.
MEMBERS = {
    "meta.json": {
        "member": _MESH_ARCHIVE_ROOT + "meta.json", "bytes": 4636,
        "sha256": "656832c5361c4bebb690f5317bc5f8cb75248225301a7885d85612806e29dfc5",
    },
    "x.tif": {
        "member": _MESH_ARCHIVE_ROOT + "x.tif", "bytes": 69904,
        "sha256": "7ca2e15f6fba2e462e255c263d8ae53e2d3bcbc6e03ddac72d86d06f833a7f57",
    },
    "y.tif": {
        "member": _MESH_ARCHIVE_ROOT + "y.tif", "bytes": 69904,
        "sha256": "50d693df00e09cd111005ae58f2a5c63fb767ecce771f771fd6e8df33bb792da",
    },
    "z.tif": {
        "member": _MESH_ARCHIVE_ROOT + "z.tif", "bytes": 69904,
        "sha256": "bb4c33ebfc52cdf76730fc54e92644a7bb6243be44b6689de5035cc5814ef5b6",
    },
    "transform.json": {
        "member": "remote/home/ubuntu/donor-continued-training-20260910/841-transform/transform.json",
        "bytes": 1352,
        "sha256": "700d59da195761ca5877d7a1aed6804ae7e1b14a29dc0c2d9f6b86580c87719e",
    },
}
EXPECTED_MATRIX = [
    [1.0254361365022744, 0.0035657098422584277, 0.0004988027718623516, -81.50512802792237],
    [-0.0028396559131153793, 1.0257641060055425, -0.0006003501869395713, -61.145824039353144],
    [-2.9385715359188414e-05, 0.0006015680556569281, 1.025831056140028, 12788.96485394228],
]
EXPECTED_OFFSETS_UM = [
    -74.928, -65.562, -56.196, -46.83, -37.464, -28.098, -18.732, -9.366,
    0.0, 9.366, 18.732, 28.098, 37.464, 46.83, 56.196, 65.562,
]
NATIVE_CT_URL = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/volumes/"
    "20250821151531-9.366um-1.2m-113keV-masked.zarr/0/"
)


def _verify(data: bytes, size: int, sha256: str, description: str) -> None:
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Pinned size/hash mismatch: {description}")


def _read_cached(path: Path, size: int, sha256: str) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Unexpected cached asset size/type: {path.name}")
    data = path.read_bytes()
    _verify(data, size, sha256, path.name)
    return data


def _download(url: str, size: int, *, range_end: int | None = None) -> tuple[bytes, dict]:
    headers = {"Accept-Encoding": "identity", "User-Agent": "Scrollprize-public-geometry/1.0"}
    if range_end is not None:
        headers["Range"] = f"bytes=0-{range_end}"
    with urlopen(Request(url, headers=headers), timeout=45) as response:
        expected_status = 206 if range_end is not None else 200
        if response.status != expected_status:
            raise RuntimeError(f"Expected HTTP {expected_status}, got {response.status}")
        if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
            raise RuntimeError("Unexpected HTTP content encoding")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length != str(size):
            raise RuntimeError("Unexpected Content-Length")
        if range_end is not None:
            expected_range = f"bytes 0-{range_end}/{ARCHIVE_BYTES}"
            if response.headers.get("Content-Range") != expected_range:
                raise RuntimeError("Unexpected Content-Range")
            if content_length != str(size):
                raise RuntimeError("Range response lacks exact Content-Length")
        data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Response body differs from pinned length")
        observed = {key: response.headers.get(key) for key in
            ("ETag", "Content-Length", "Content-Range", "Content-Type")}
    return data, observed


def _source(cache: Path, spec: dict) -> tuple[bytes, dict]:
    path = cache / spec["file"]
    data = _read_cached(path, spec["bytes"], spec["sha256"])
    record = {key: spec[key] for key in ("url", "bytes", "sha256")}
    record["cache_hit"] = data is not None
    if data is None:
        data, record["headers"] = _download(spec["url"], spec["bytes"])
        _verify(data, spec["bytes"], spec["sha256"], spec["file"])
        path.write_bytes(data)
    return data, record


def _extract(prefix: bytes, mesh_dir: Path) -> None:
    _verify(prefix, PREFIX_BYTES, PREFIX_SHA256, "archive prefix")
    by_member = {spec["member"]: (name, spec) for name, spec in MEMBERS.items()}
    found = set()
    visited_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(prefix), mode="r|gz") as archive:
        for member in archive:
            visited_bytes += member.size
            if visited_bytes > MAX_UNCOMPRESSED_PREFIX_BYTES:
                raise RuntimeError("Decompressed prefix budget exceeded")
            if member.name not in by_member:
                continue
            name, spec = by_member[member.name]
            if name in found:
                raise ValueError("Duplicate whitelisted archive member")
            if not member.isfile() or member.size != spec["bytes"]:
                raise ValueError("Unexpected whitelisted member type/size")
            data = archive.extractfile(member).read(spec["bytes"] + 1)
            _verify(data, spec["bytes"], spec["sha256"], name)
            existing = _read_cached(mesh_dir/name, spec["bytes"], spec["sha256"])
            if existing is None:
                (mesh_dir/name).write_bytes(data)
            found.add(name)
            if found == set(MEMBERS):
                break
    if found != set(MEMBERS):
        raise RuntimeError("Pinned archive prefix lacks required geometry")


def _recipe(selection: dict, render: dict, meta: dict, transform: dict) -> dict:
    if selection["id"] != DATASET_ID or selection["role"] != "validation":
        raise ValueError("Unexpected holdout identity/role")
    geometry = selection["geometry"]
    baseline = geometry["baseline"]
    source_meta = baseline["source_meta"]
    recipe = geometry["recipe"]
    expected = {
        "kind": "native_low_resolution_mesh_derived_affine",
        "volume_id": "20250821151531", "source_volume": "20260319124803",
        "url": NATIVE_CT_URL, "level": 0, "xy_pitch_um": 9.36, "z_pitch_um": 9.366,
        "normal_sign": 1, "matrix": EXPECTED_MATRIX, "offsets_um": EXPECTED_OFFSETS_UM,
    }
    if geometry["kind"] != "original" or geometry["runtime_reverse"] is not False:
        raise ValueError("Unexpected geometry kind/depth order")
    if baseline["runtime_reverse"] is not False or baseline["recipe"] != recipe:
        raise ValueError("Inconsistent baseline recipe")
    if render["recipe"] != recipe or render["status"] != "complete":
        raise ValueError("Inconsistent completed render recipe")
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected native sampling parameters")
    if (source_meta["geometry_um"] != 9.612
            or source_meta["geometry_shape"] != [3781, 7618, 7618]
            or source_meta["matrix_to_hr"] != [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0]]
            or source_meta["hr_um"] != 2.403
            or selection["shape"] != [2773, 2609]
            or geometry["offset_yx"] != [0, 0]
            or geometry["pitch"] != 9.36
            or meta["scale"] != [0.05, 0.05]
            or meta["fit_config"]["voxel_size_um"] != 9.612):
        raise ValueError("Unexpected source mesh/label grid parameters")
    # This verifies provenance only.  The returned matrix already includes the
    # mesh-to-HR factor four; callers must not apply transform.json a second time.
    from_hr = transform["matrix"]
    combined = [[4 * row[0], 4 * row[1], 4 * row[2], row[3]] for row in from_hr]
    if combined != EXPECTED_MATRIX:
        raise ValueError("Transform provenance does not match native recipe")
    result = copy.deepcopy({k: recipe[k] for k in (
        "kind", "volume_id", "source_volume", "url", "level", "xy_pitch_um", "z_pitch_um",
        "matrix", "normal_sign", "offsets_um", "pixel_centers", "limitation")})
    result.update({
        "dataset_id": DATASET_ID, "shape": [2773, 2609],
        "mesh_geometry_um": 9.612, "mesh_geometry_shape": [3781, 7618, 7618],
        "mesh_scale_yx": [0.05, 0.05], "runtime_reverse": False, "offset_yx": [0, 0],
        "normal_basis": baseline["orientation"]["normal_basis"],
    })
    return result


def acquire_geometry(cache_dir: str | Path) -> tuple[Path, dict, dict]:
    """Return ``(mesh_dir, recipe, provenance)`` using verified external cache.

    ``recipe['matrix']`` maps stored XYZ mesh coordinates directly to native
    level-0 XYZ.  ``transform.json`` is a provenance receipt only.  Returned
    provenance is sanitized for publication; cached raw JSON is not.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    records, decoded = {}, {}
    for key, spec in SOURCES.items():
        data, records[key] = _source(cache, spec)
        decoded[key] = json.loads(data)
    index = decoded["file_index"]["archives"]["remote-artifacts.tar.gz"]
    if index["archive_bytes"] != ARCHIVE_BYTES or index["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Unexpected archive identity in pinned index")
    for name, spec in MEMBERS.items():
        indexed = index["files"][spec["member"]]
        if indexed != {"bytes": spec["bytes"], "sha256": spec["sha256"]}:
            raise ValueError(f"Pinned file-index disagreement: {name}")

    mesh_dir = cache / "841_geometry"
    mesh_dir.mkdir(exist_ok=True)
    missing = [name for name, spec in MEMBERS.items()
               if _read_cached(mesh_dir/name, spec["bytes"], spec["sha256"]) is None]
    range_record = {
        "url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES,
        "archive_expected_sha256": ARCHIVE_SHA256, "whole_archive_hash_verified": False,
        "range_start": 0, "range_end": PREFIX_BYTES - 1,
        "range_bytes": PREFIX_BYTES, "range_sha256": PREFIX_SHA256,
        "needed": bool(missing), "network_requests": 0, "downloaded_bytes": 0,
    }
    if missing:
        prefix = _read_cached(cache/PREFIX_FILE, PREFIX_BYTES, PREFIX_SHA256)
        range_record["cache_hit"] = prefix is not None
        if prefix is None:
            prefix, range_record["headers"] = _download(
                ARCHIVE_URL, PREFIX_BYTES, range_end=PREFIX_BYTES-1)
            _verify(prefix, PREFIX_BYTES, PREFIX_SHA256, "archive prefix")
            (cache/PREFIX_FILE).write_bytes(prefix)
            range_record.update(network_requests=1, downloaded_bytes=PREFIX_BYTES)
        _extract(prefix, mesh_dir)
    final = {}
    for name, spec in MEMBERS.items():
        data = _read_cached(mesh_dir/name, spec["bytes"], spec["sha256"])
        if data is None:
            raise RuntimeError("Required geometry member is absent after acquisition")
        if name.endswith(".json"):
            final[name] = json.loads(data)
    recipe = _recipe(decoded["selection"], decoded["render"], final["meta.json"], final["transform.json"])
    provenance = {
        "schema": "studio.oblique.geometry_assets.v1", "dataset_id": DATASET_ID,
        "hf_revision": HF_REVISION,
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "asset_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": records, "archive_range": range_record,
        "geometry_files": {name: {k: spec[k] for k in ("bytes", "sha256")}
                           for name, spec in MEMBERS.items()},
        "transform_file_usage": "Receipt only; recipe.matrix already maps stored mesh coordinates to native L0.",
        "recipe": recipe,
        "network_requests": sum(not r["cache_hit"] for r in records.values()) + range_record["network_requests"],
        "downloaded_bytes": sum(r["bytes"] for r in records.values() if not r["cache_hit"])
                            + range_record["downloaded_bytes"],
    }
    (cache/"geometry_assets_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    return mesh_dir, recipe, provenance
