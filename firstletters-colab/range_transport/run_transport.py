#!/usr/bin/env python3
"""Frozen, bounded byte-range check on one already acquired public CT chunk.

No labels, inference, unread scan, interpolation, or automatic fallback.
HTTP representation lengths and application-read bytes are reported separately;
these are not a socket/TLS wire-byte bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "frozen_protocol.json"


class TransportError(RuntimeError):
    """Controlled failure codes never include redirected URLs or response bodies."""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    return digest(Path(path).read_bytes())


def outside_git(path):
    path = Path(path).resolve()
    if any(os.path.lexists(parent / ".git") for parent in (path, *path.parents)):
        raise ValueError("Detailed run output must be outside every Git worktree")
    return path


def header(headers, key):
    values = headers.get_all(key) if hasattr(headers, "get_all") else [
        value for name, value in headers.items() if name.lower() == key.lower()
    ]
    if not values:
        return None
    if len(values) != 1:
        raise TransportError("duplicate_response_header:" + key.lower())
    return values[0].strip()


def strong_etag(headers):
    value = header(headers, "ETag")
    if value is None or not re.fullmatch(r'"[\x21\x23-\x7e]+"', value):
        raise TransportError("missing_or_weak_etag")
    return value


def validate_headers(status, headers, *, object_size, byte_range=None, etag=None):
    expected_status = 200 if byte_range is None else 206
    if status != expected_status:
        raise TransportError("unexpected_http_status:" + str(status))
    encoding = header(headers, "Content-Encoding")
    if encoding is not None and encoding.lower() != "identity":
        raise TransportError("content_encoding_not_identity")
    transfer = header(headers, "Transfer-Encoding")
    if transfer is not None and transfer.lower() != "identity":
        raise TransportError("transfer_encoding_not_identity")
    length = header(headers, "Content-Length")
    if length is None or not re.fullmatch(r"[0-9]+", length):
        raise TransportError("missing_or_invalid_content_length")
    expected_length = object_size if byte_range is None else byte_range[1] - byte_range[0] + 1
    if int(length) != expected_length:
        raise TransportError("content_length_mismatch")
    content_range = header(headers, "Content-Range")
    if byte_range is None:
        if content_range is not None:
            raise TransportError("unexpected_content_range_on_whole_object")
    else:
        expected = f"bytes {byte_range[0]}-{byte_range[1]}/{object_size}"
        if content_range != expected:
            raise TransportError("content_range_mismatch")
    actual_etag = strong_etag(headers)
    if etag is not None and actual_etag != etag:
        raise TransportError("etag_changed")
    return expected_length, actual_etag


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RangeClient:
    """Serial HTTPS only, counted redirects, no retries and no full-GET fallback."""

    def __init__(self, *, body_cap=5 * 2**20, request_cap=24, opener=None, progress=False):
        self.body_cap, self.request_cap = int(body_cap), int(request_cap)
        self.body_bytes_read = 0
        self.requests = []
        self.opener = opener or urllib.request.build_opener(NoRedirect)
        self.progress = progress

    @staticmethod
    def checked_url(url):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise TransportError("nonpublic_https_url")
        return url

    def get(self, url, *, object_size, byte_range=None, etag=None):
        if byte_range is not None:
            if etag is None or not (0 <= byte_range[0] <= byte_range[1] < object_size):
                raise TransportError("invalid_range_or_missing_version")
        headers = {"Accept-Encoding": "identity", "User-Agent": "Scrollprize-public-range-control/1"}
        if byte_range is not None:
            headers.update({"Range": f"bytes={byte_range[0]}-{byte_range[1]}", "If-Match": etag})
        url = self.checked_url(url)
        for hop in range(4):
            if len(self.requests) >= self.request_cap:
                raise TransportError("request_cap_exceeded")
            entry = {
                "method": "GET", "host": urllib.parse.urlsplit(url).hostname,
                "range": None if byte_range is None else list(byte_range),
                "if_match_sent": etag is not None, "accept_encoding": "identity",
                "requested_representation_bytes": object_size if byte_range is None else byte_range[1] - byte_range[0] + 1,
                "application_body_bytes_read": 0, "redirect_hop": hop,
            }
            self.requests.append(entry)
            start = time.monotonic()
            try:
                response = self.opener.open(urllib.request.Request(url, headers=headers), timeout=25)
            except urllib.error.HTTPError as exc:
                response = exc
            except Exception as exc:
                entry["error_type"] = type(exc).__name__
                raise TransportError("transport_open_failed") from None
            try:
                status = int(response.status if hasattr(response, "status") else response.code)
                entry["status"] = status
                declared = header(response.headers, "Content-Length")
                entry["server_declared_body_bytes"] = int(declared) if declared and declared.isdecimal() else None
                if status in (301, 302, 303, 307, 308):
                    if hop == 3:
                        raise TransportError("redirect_cap_exceeded")
                    location = header(response.headers, "Location")
                    if location is None:
                        raise TransportError("redirect_without_location")
                    url = self.checked_url(urllib.parse.urljoin(url, location))
                    entry["body_not_read_reason"] = "redirect"
                    continue
                length, actual_etag = validate_headers(
                    status, response.headers, object_size=object_size, byte_range=byte_range, etag=etag,
                )
                if self.body_bytes_read + length + 1 > self.body_cap:
                    raise TransportError("application_body_cap_exceeded")
                raw = response.read(length + 1)
                self.body_bytes_read += len(raw)
                entry["application_body_bytes_read"] = len(raw)
                if len(raw) != length:
                    raise TransportError("response_body_length_mismatch")
                entry["etag_sha256"] = digest(actual_etag.encode("ascii"))
                entry["body_sha256"] = digest(raw)
                return raw, actual_etag, url
            finally:
                response.close()
                entry["elapsed_seconds"] = round(time.monotonic() - start, 6)
                if self.progress:
                    print(json.dumps({"requests": len(self.requests),
                                      "application_body_bytes_read": self.body_bytes_read,
                                      "last_status": entry.get("status")}), flush=True)
        raise TransportError("redirect_cap_exceeded")


def plan_box(box, shape):
    """C-order uint8 box -> minimal contiguous byte spans, in output order."""
    if len(box) != 6 or any(type(v) is not int for v in box):
        raise ValueError("Box must contain six integer ZYX bounds")
    z0, y0, x0, z1, y1, x1 = box
    if not (0 <= z0 < z1 <= shape[0] and 0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
        raise ValueError("Box outside the fixed chunk")
    spans = []
    for z in range(z0, z1):
        for y in range(y0, y1):
            start = (z * shape[1] + y) * shape[2] + x0
            end = start + x1 - x0 - 1
            if spans and spans[-1][1] + 1 == start:
                spans[-1][1] = end
            else:
                spans.append([start, end])
    return spans


def verify_freeze(reference):
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ValueError("A full published freeze commit is required")
    repo = HERE.parents[1]
    for path in (PROTOCOL, Path(__file__).resolve()):
        relative = path.relative_to(repo).as_posix()
        frozen = subprocess.check_output(["git", "show", f"{reference}:{relative}"], cwd=repo)
        if frozen != path.read_bytes():
            raise ValueError("Runner or protocol differs from the freeze commit")


def local_reference(protocol, receipt_path, cache_dir):
    """Verify the old receipt, fixed lexical selection, metadata, and cached bytes."""
    receipt_path, cache_dir = Path(receipt_path), Path(cache_dir)
    if file_digest(receipt_path) != protocol["prior_acquisition_receipt_sha256"]:
        raise ValueError("Prior acquisition receipt differs")
    old = json.loads(receipt_path.read_text())
    source = protocol["source"]
    prefix = source["array_url"]
    choices = sorted(
        (a for a in old["completed_assets"]
         if a["url"].startswith(prefix) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", a["url"][len(prefix):])),
        key=lambda a: a["url"],
    )
    if not choices or choices[0] != {k: source[k] for k in ("url", "bytes", "sha256")}:
        raise ValueError("Frozen object is not the first lexicographic old chunk")
    payloads = {}
    for name, url, expected in (
        ("chunk", source["url"], source["sha256"]),
        ("metadata", prefix + ".zarray", source["metadata_sha256"]),
    ):
        path = cache_dir / digest(url.encode())
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError("Cached reference identity mismatch: " + name)
        payloads[name] = raw
    meta = json.loads(payloads["metadata"])
    if meta["chunks"] != [65, 128, 128] or meta["dtype"] != "|u1" or meta["order"] != "C" or meta["compressor"] is not None or meta["filters"] is not None:
        raise ValueError("Only the frozen uncompressed C-order uint8 source is supported")
    if len(payloads["chunk"]) != source["bytes"]:
        raise ValueError("Cached reference size mismatch")
    return payloads["chunk"]


def run(protocol, reference, client):
    source = protocol["source"]
    shape = protocol["chunk_shape_zyx"]
    array = np.frombuffer(reference, np.uint8).reshape(shape)
    cases = []
    planned = []
    for case in protocol["cases"]:
        spans = plan_box(case["box_zyx"], shape)
        z0, y0, x0, z1, y1, x1 = case["box_zyx"]
        expected = array[z0:z1, y0:y1, x0:x1].tobytes()
        if digest(expected) != case["expected_sha256"] or len(expected) != case["requested_bytes"]:
            raise ValueError("Frozen slice reference differs")
        planned.append((case, spans, expected))
    if sum(len(spans) for _, spans, _ in planned) != protocol["expected_range_requests"]:
        raise ValueError("Range request plan differs from freeze")
    started = time.monotonic()
    whole, etag, resolved_url = client.get(source["url"], object_size=source["bytes"])
    if digest(whole) != source["sha256"] or whole != reference:
        raise TransportError("whole_object_no_longer_matches_frozen_cache")
    whole_seconds = time.monotonic() - started
    whole_array = np.frombuffer(whole, np.uint8).reshape(shape)
    range_started = time.monotonic()
    for case, spans, expected in planned:
        bodies = []
        start = time.monotonic()
        for span in spans:
            raw, _, _ = client.get(resolved_url, object_size=source["bytes"], byte_range=span, etag=etag)
            bodies.append(raw)
        actual = b"".join(bodies)
        z0, y0, x0, z1, y1, x1 = case["box_zyx"]
        baseline = whole_array[z0:z1, y0:y1, x0:x1].tobytes()
        if actual != expected or actual != baseline:
            raise TransportError("range_voxels_differ_from_whole_or_cache")
        cases.append({"id": case["id"], "shape_zyx": [z1-z0, y1-y0, x1-x0],
                      "range_requests": len(spans), "requested_bytes": len(expected),
                      "actual_sha256": digest(actual), "equal_to_whole": True,
                      "equal_to_prior_cache": True, "voxel_mismatches": 0,
                      "elapsed_seconds": round(time.monotonic()-start, 6)})
    range_seconds = time.monotonic() - range_started
    return {
        "status": "pass", "conclusion": "Exact byte ranges verified for this one public HF object and observed transport path only.",
        "source": source, "cases": cases, "whole_unique_object_bytes": len(whole),
        "ranges_total_requested_bytes": sum(c["requested_bytes"] for c in cases),
        "range_to_unique_whole_body_fraction": sum(c["requested_bytes"] for c in cases) / len(whole),
        "whole_elapsed_seconds": round(whole_seconds, 6), "ranges_elapsed_seconds": round(range_seconds, 6),
        "latency_interpretation": "One fixed whole-first sequence; no throughput or cold-cache speed claim.",
        "s3_or_paris4_support_established": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-reference", required=True)
    p.add_argument("--prior-receipt", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    verify_freeze(args.freeze_reference)
    protocol = json.loads(PROTOCOL.read_text())
    reference = local_reference(protocol, args.prior_receipt, args.cache_dir)
    out = outside_git(args.output_dir)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; prior failures are retained")
    out.mkdir(parents=True)
    client = RangeClient(body_cap=protocol["maximum_application_body_bytes_read"],
                         request_cap=protocol["maximum_requests_including_redirects"], progress=True)
    base = {
        "schema": "scrollprize.public-range-transport.v1", "freeze_reference": args.freeze_reference,
        "protocol_sha256": file_digest(PROTOCOL), "runner_sha256": file_digest(__file__),
        "generated_data": False, "new_region_acquired": False, "model_forward_calls": 0,
        "source_selection": protocol["selection"],
        "body_budget_semantics": "Application-read body bytes only; HTTP headers, redirects without body reads, socket buffering and TLS wire bytes are not measured.",
    }
    (out / "start.json").write_text(json.dumps(base, indent=2) + "\n")
    code = 0
    try:
        base.update(run(protocol, reference, client))
    except Exception as exc:
        code = 1
        base.update({"status": "failed", "error_type": type(exc).__name__,
                     "reason": str(exc) if isinstance(exc, TransportError) else "local_or_transport_validation_failed",
                     "s3_or_paris4_support_established": False})
    finally:
        base["requests"] = client.requests
        base["request_count_including_redirects"] = len(client.requests)
        base["application_body_bytes_read"] = client.body_bytes_read
        base["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (out / "result.json").write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": base["status"], "requests": len(client.requests),
                      "application_body_bytes_read": client.body_bytes_read,
                      "reason": base.get("reason")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
