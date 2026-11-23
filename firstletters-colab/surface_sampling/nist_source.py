#!/usr/bin/env python3
"""Read a frozen, bounded subset of the public NIST modern lead-ink phantom.

No network activity occurs on import or with --inspect. Acquisition requires a
freeze document, its SHA-256, and a new private cache outside any Git checkout.
The freeze document must contain the SHA-256 of nist_manifest.json. The caller
must also verify its own implementation against its public freeze commit.

This is a physical sampling/geometry control, not ancient carbon-ink validation.
There are no optical ink annotations in this acquisition. A declared whole-ZIP
SHA-256 is retained as provenance but is NOT verified by partial acquisition.

API: stack, receipt = acquire(private_cache, freeze_path=..., freeze_sha256=...)
Only NumPy, Pillow, and Python's standard library are required. The reader makes
exactly three identity requests and two requests per selected member, in order,
without retries or redirects. All downloaded bytes stay in the private cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import urllib.error
import urllib.request
import zlib

import numpy as np
from PIL import Image


DEFAULT_MANIFEST = Path(__file__).with_name("nist_manifest.json")
_ROOT_NAME = "recon_binned_9x_down_sampling/"
_ARCHIVE_BYTES = 293_236_353
_MAX_BYTES = 96 * 1024 * 1024
_MAX_REQUESTS = 300
_FIRST, _STOP = 246, 374


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _safe_member_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
            or not name.startswith(_ROOT_NAME)):
        raise ValueError("Unsafe or unexpected ZIP member name")


def _parse_directory(body):
    """Parse classic ZIP central-directory records, with no payload access."""
    position, entries, seen = 0, [], set()
    while position < len(body):
        if len(body) - position < 46:
            raise ValueError("Truncated ZIP central-directory header")
        values = struct.unpack_from("<4s6H3I5H2I", body, position)
        (magic, made_by, needed, flags, method, mtime, mdate, crc,
         compressed, decoded, name_len, extra_len, comment_len, disk,
         internal, external, offset) = values
        if magic != b"PK\x01\x02" or disk != 0:
            raise ValueError("Unsupported central directory / split ZIP")
        if needed >= 45 or max(compressed, decoded, offset) == 0xFFFFFFFF:
            raise ValueError("ZIP64 is outside the frozen source contract")
        end = position + 46 + name_len + extra_len + comment_len
        if end > len(body):
            raise ValueError("Truncated central-directory variable fields")
        name_bytes = body[position + 46:position + 46 + name_len]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        _safe_member_name(name)
        if name in seen:
            raise ValueError("Duplicate ZIP member name")
        seen.add(name)
        entries.append({
            "name": name, "local_header_offset": offset,
            "compressed_bytes": compressed, "decoded_file_bytes": decoded,
            "crc32": crc, "compression_method": method,
            "general_purpose_flags": flags, "version_needed": needed,
            "central_filename_bytes": name_len,
            "central_extra_bytes": extra_len,
        })
        position = end
    return entries


def _selected_entries(directory):
    entries = _parse_directory(directory)
    if len(entries) != 621:
        raise ValueError("Expected one directory plus 620 TIFF members")
    files = sorted((x for x in entries if not x["name"].endswith("/")),
                   key=lambda x: x["name"])
    if len(files) != 620:
        raise ValueError("Unexpected released slice count")
    offsets = sorted(x["local_header_offset"] for x in entries)
    if len(set(offsets)) != len(offsets):
        raise ValueError("Overlapping ZIP local-header offsets")
    next_offset = dict(zip(offsets, offsets[1:] + [293_158_731]))
    selected = []
    for index, entry in enumerate(files):
        expected_name = _ROOT_NAME + f"reconstruciton_{9 * index:06d}.tif"
        if entry["name"] != expected_name:
            raise ValueError("Released filename / index contract changed")
        if entry["compression_method"] != 8 or entry["general_purpose_flags"] & ~0x808:
            raise ValueError("Unsupported ZIP compression/encryption/flags")
        if entry["decoded_file_bytes"] != 509_930:
            raise ValueError("Unexpected decoded TIFF file size")
        if not 0 < entry["compressed_bytes"] <= 474_465:
            raise ValueError("Unexpected compressed TIFF size")
        if _FIRST <= index < _STOP:
            selected.append(dict(entry, released_index=index,
                                 source_filename_index=9 * index,
                                 next_local_header_offset=next_offset[entry["local_header_offset"]]))
    return selected


def _load_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    archive = manifest["archive"]
    selection = manifest["selection"]
    if archive["bytes"] != _ARCHIVE_BYTES:
        raise ValueError("Archive byte length differs from source contract")
    if (selection["released_index_start"] != _FIRST
            or selection["released_index_stop_exclusive"] != _STOP
            or len(manifest["members"]) != 128):
        raise ValueError("Selection differs from fixed middle 128 slices")
    limits = manifest["limits"]
    if limits["actual_response_body_bytes"] != _MAX_BYTES or limits["charged_requests"] != _MAX_REQUESTS:
        raise ValueError("Frozen transfer limits differ")
    if limits["new_acquisition_requests"] != 259:
        raise ValueError("Unexpected new-acquisition request limit")
    for entry in manifest["members"]:
        _safe_member_name(entry["name"])
    return manifest, _sha(raw)


def _freeze_digest(freeze_path, expected, manifest_digest):
    if not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
        raise ValueError("Supply the exact SHA-256 of the completed freeze document")
    freeze = Path(freeze_path).read_bytes()
    if len(freeze) > 1024 * 1024 or _sha(freeze) != expected:
        raise ValueError("Freeze document hash/size mismatch")
    if manifest_digest not in freeze.decode("utf-8"):
        raise ValueError("Freeze document does not pin this exact source manifest")
    return expected


def _new_private_cache(path):
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ValueError("Private cache must be an absolute path")
    cache = requested.resolve()
    if any((ancestor / ".git").exists() for ancestor in (cache, *cache.parents)):
        raise ValueError("Raw source cache must be outside all Git checkouts")
    if not cache.parent.is_dir():
        raise ValueError("Private cache parent must already exist")
    cache.mkdir(mode=0o700, exist_ok=False)
    (cache / "members").mkdir(mode=0o700)
    return cache


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _Transfer:
    """Ledger for application response bodies, requests and early failures.

    Prior metadata bodies are exact. Historical automatic redirect hops were
    not instrumented: the manifest records that limitation and an explicit
    request allowance, not a fabricated historical count. Every new call here
    is instrumented, including an unsuccessful call, and redirects are refused.
    """

    def __init__(self, manifest, cache, receipt):
        self.manifest, self.cache, self.receipt = manifest, cache, receipt
        self.opener = urllib.request.build_opener(_NoRedirect())

    def flush(self):
        _write_json(self.cache / "receipt.json", self.receipt)

    def fetch(self, url, length, *, start=None, expected_sha256=None, etag=None):
        ledger = self.receipt["transfer"]
        if ledger["charged_requests"] + 1 > _MAX_REQUESTS:
            raise RuntimeError("Total charged request budget exhausted")
        if ledger["new_requests"] + 1 > 259:
            raise RuntimeError("New-acquisition request budget exhausted")
        # Reserve one byte to detect an overlong response without exceeding cap.
        if length < 0 or ledger["charged_body_bytes"] + length + 1 > _MAX_BYTES:
            raise RuntimeError("Response-body byte budget exhausted")
        headers = {"Accept-Encoding": "identity", "User-Agent": "Scrollprize-NIST-bounded-reader/1.0"}
        if start is not None:
            if start < 0 or start + length > _ARCHIVE_BYTES:
                raise ValueError("Range outside archive")
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        if etag is not None:
            headers["If-Match"] = etag
        event = {"url": url, "range_start": start, "expected_body_bytes": length,
                 "received_body_bytes": 0, "started_utc": _now(), "state": "started"}
        self.receipt["requests"].append(event)
        ledger["new_requests"] += 1
        ledger["charged_requests"] += 1
        self.flush()
        try:
            request = urllib.request.Request(url, headers=headers)
            with self.opener.open(request, timeout=30) as response:
                event["status"] = response.status
                event["etag"] = response.headers.get("ETag")
                event["content_range"] = response.headers.get("Content-Range")
                if response.geturl() != url or response.status != (206 if start is not None else 200):
                    raise ValueError("Redirect or unexpected HTTP status")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected HTTP content encoding")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != length:
                    raise ValueError("HTTP Content-Length differs from exact request")
                if start is not None:
                    expected_range = f"bytes {start}-{start + length - 1}/{_ARCHIVE_BYTES}"
                    if event["content_range"] != expected_range:
                        raise ValueError("HTTP Content-Range mismatch")
                if etag is not None and event["etag"] != etag:
                    raise ValueError("Archive ETag changed or missing")
                pieces, received = [], 0
                while received <= length:
                    part = response.read(min(65_536, length + 1 - received))
                    if not part:
                        break
                    received += len(part)
                    ledger["new_body_bytes"] += len(part)
                    ledger["charged_body_bytes"] += len(part)
                    event["received_body_bytes"] = received
                    pieces.append(part)
                    self.flush()
                body = b"".join(pieces)
                if len(body) != length:
                    raise ValueError("Truncated or overlong HTTP response")
                event["sha256"] = _sha(body)
                if expected_sha256 is not None and event["sha256"] != expected_sha256:
                    raise ValueError("Pinned metadata SHA-256 mismatch")
                event["state"] = "complete"
                return body
        except urllib.error.HTTPError as error:
            event["status"] = error.code
            event["state"] = "failed"
            event["failure"] = "HTTP error; no redirect followed and error body not read"
            error.close()
            raise
        except Exception as error:
            event["state"] = "failed"
            event["failure"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            event["ended_utc"] = _now()
            self.flush()


def _local_header(header, entry):
    if len(header) != 30:
        raise ValueError("Wrong local ZIP header length")
    magic, needed, flags, method, mtime, mdate, crc, compressed, decoded, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    if (magic != b"PK\x03\x04" or method != 8
            or flags != entry["general_purpose_flags"]
            or needed != entry["version_needed"]
            or name_len != entry["central_filename_bytes"]):
        raise ValueError("Local ZIP header disagrees with pinned central entry")
    for actual, expected in ((crc, entry["crc32"]),
                             (compressed, entry["compressed_bytes"]),
                             (decoded, entry["decoded_file_bytes"])):
        if actual != expected and not (flags & 8 and actual == 0):
            raise ValueError("Local ZIP sizes/CRC disagree with central entry")
    end = entry["local_header_offset"] + 30 + name_len + extra_len + entry["compressed_bytes"]
    if end > entry["next_local_header_offset"]:
        raise ValueError("ZIP payload crosses next member or central directory")
    return name_len, extra_len


def _inflate(payload, entry):
    decoder = zlib.decompressobj(wbits=-15)
    expected = entry["decoded_file_bytes"]
    decoded = decoder.decompress(payload, expected + 1)
    if (len(decoded) != expected or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError("Deflate stream incomplete, oversized or has trailing data")
    if (zlib.crc32(decoded) & 0xFFFFFFFF) != entry["crc32"]:
        raise ValueError("Decoded TIFF CRC32 differs from pinned ZIP directory")
    return decoded


def _decode_tiff(body, expected_shape=None):
    with Image.open(io.BytesIO(body)) as image:
        if image.format != "TIFF" or image.mode != "F" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Expected a single-channel, single-frame floating TIFF")
        width, height = image.size
        if min(width, height) <= 0 or width * height * 4 > len(body):
            raise ValueError("TIFF geometry violates uncompressed file-size bound")
        bits = image.tag_v2.get(258)
        samples = image.tag_v2.get(277, 1)
        sample_format = image.tag_v2.get(339)
        if bits not in (32, (32,)) or samples != 1 or sample_format not in (3, (3,)):
            raise ValueError("TIFF tags do not declare one float32 sample")
        array = np.array(image)
    if array.ndim != 2 or array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ValueError("TIFF did not decode to a float32 2D plane")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError("Selected TIFFs do not share one shape")
    if not np.isfinite(array).all():
        raise ValueError("TIFF contains non-finite attenuation values")
    return array.astype(np.float32, copy=False)


def acquire(private_cache, *, freeze_path, freeze_sha256, manifest_path=DEFAULT_MANIFEST):
    """Return (stack[128,H,W], receipt); all network bytes go to a new private cache.

    H and W are unknown until the first frozen member is decoded. A failed run
    retains a private receipt and raises. It is never retried or resumed here.
    Creating a different cache does not authorize a second scientific run.
    """
    manifest, manifest_digest = _load_manifest(manifest_path)
    _freeze_digest(freeze_path, freeze_sha256, manifest_digest)
    cache = _new_private_cache(private_cache)
    baseline = manifest["accounting"]["budget_baseline"]
    receipt = {
        "schema_version": 1, "state": "started", "started_utc": _now(),
        "source_manifest_sha256": manifest_digest,
        "freeze_document_sha256": freeze_sha256,
        "reader_sha256": _sha(Path(__file__).read_bytes()),
        "purpose": manifest["scientific_scope"],
        "whole_archive_sha256_verification": "NOT VERIFIED; partial acquisition only",
        "historical_request_count_certified": False,
        "historical_request_count_caveat": manifest["accounting"]["historical_request_count_caveat"],
        "transfer": {"new_requests": 0, "new_body_bytes": 0,
                     "charged_requests": baseline["charged_requests"],
                     "charged_body_bytes": baseline["body_bytes"],
                     "historical_unknown_hop_allowance": baseline["unknown_hop_allowance"]},
        "requests": [], "members": [],
    }
    transfer = _Transfer(manifest, cache, receipt)
    transfer.flush()
    try:
        archive = manifest["archive"]
        metadata = manifest["source_metadata"]
        body = transfer.fetch(metadata["url"], metadata["bytes"], expected_sha256=metadata["sha256"])
        (cache / "source_metadata.json").write_bytes(body)
        for key in ("eocd", "central_directory"):
            descriptor = archive[key]
            body = transfer.fetch(archive["canonical_url"], descriptor["bytes"],
                                  start=descriptor["offset"], expected_sha256=descriptor["sha256"],
                                  etag=archive["observed_etag"])
            (cache / (key + ".bin")).write_bytes(body)
            if key == "eocd":
                fields = struct.unpack("<4s4H2IH", body)
                if fields != (b"PK\x05\x06", 0, 0, 621, 621, 77600, 293158731, 0):
                    raise ValueError("End-of-central-directory contract differs")
            elif _selected_entries(body) != manifest["members"]:
                raise ValueError("Selected central-directory entries differ from manifest")

        stack = None
        for output_index, entry in enumerate(manifest["members"]):
            offset = entry["local_header_offset"]
            header = transfer.fetch(archive["canonical_url"], 30, start=offset, etag=archive["observed_etag"])
            name_len, extra_len = _local_header(header, entry)
            remaining = name_len + extra_len + entry["compressed_bytes"]
            body = transfer.fetch(archive["canonical_url"], remaining, start=offset + 30,
                                  etag=archive["observed_etag"])
            expected_name = entry["name"].encode("utf-8" if entry["general_purpose_flags"] & 0x800 else "cp437")
            if body[:name_len] != expected_name:
                raise ValueError("Local ZIP filename differs from pinned central entry")
            # Preserve the exact acquisition before decoding, including on a
            # later CRC/TIFF failure; never use a remote filename as a path.
            acquired = header + body
            (cache / "members" / f"released_{entry['released_index']:06d}.zipmember").write_bytes(acquired)
            compressed = body[name_len + extra_len:]
            decoded = _inflate(compressed, entry)
            plane = _decode_tiff(decoded, None if stack is None else stack.shape[1:])
            if stack is None:
                stack = np.empty((128, *plane.shape), dtype=np.float32)
            stack[output_index] = plane
            receipt["members"].append({
                "released_index": entry["released_index"], "name": entry["name"],
                "acquired_local_member_sha256": _sha(acquired),
                "compressed_payload_sha256": _sha(compressed),
                "decoded_tiff_sha256": _sha(decoded), "verified_crc32": entry["crc32"],
                "decoded_shape_yx": list(plane.shape), "dtype": "float32",
                "finite": True, "local_filename_bytes": name_len,
                "local_extra_bytes": extra_len,
            })
            transfer.flush()
        if stack is None or stack.shape[0] != 128:
            raise ValueError("Incomplete source stack")
        receipt.update(state="complete", ended_utc=_now(), stack_shape_zyx=list(stack.shape),
                       stack_dtype="float32", stack_sha256_little_endian_c=_sha(stack.astype("<f4", copy=False).tobytes(order="C")))
        transfer.flush()
        return stack, receipt
    except Exception as error:
        receipt.update(state="failed", ended_utc=_now(), failure=f"{type(error).__name__}: {error}")
        transfer.flush()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Print pinned selection/budgets; no network")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private-cache", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    if args.inspect:
        manifest, digest = _load_manifest(args.manifest)
        print(json.dumps({"manifest_sha256": digest, "selection": manifest["selection"],
                          "limits": manifest["limits"], "accounting": manifest["accounting"],
                          "scientific_scope": manifest["scientific_scope"]}, indent=2))
        return
    if any(x is None for x in (args.private_cache, args.freeze, args.freeze_sha256)):
        parser.error("acquisition requires --private-cache, --freeze and --freeze-sha256")
    stack, receipt = acquire(args.private_cache, freeze_path=args.freeze,
                             freeze_sha256=args.freeze_sha256, manifest_path=args.manifest)
    np.save(args.private_cache / "stack.npy", stack, allow_pickle=False)
    print(json.dumps({"state": receipt["state"], "shape_zyx": list(stack.shape),
                      "transfer": receipt["transfer"]}, indent=2))


if __name__ == "__main__":
    main()
