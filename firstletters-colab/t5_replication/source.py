"""Bounded acquisition of the one public T5 control (see source_manifest.json).

The reference is a model prediction proxy, not optical ground truth. Construct a
Budget only after the caller has frozen its protocol; authorize four regions
after its deterministic label-only selection. No source arrays are fetched on
import. Only urllib, NumPy and Pillow are used; every failure is terminal.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import warnings

import numpy as np
from PIL import Image

MANIFEST = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
MIB = 1024 * 1024
SHAPE = tuple(MANIFEST["zarr"]["metadata"]["shape"])
CHUNKS = tuple(MANIFEST["zarr"]["metadata"]["chunks"])
CHUNK_BYTES = MANIFEST["zarr"]["full_chunk_bytes"]
KNOWN_ROI = tuple(MANIFEST["known_gemini_roi_y0_y1_x0_x1"])


class SourceError(RuntimeError):
    """A source, cache, region, or acquisition contract was violated."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise SourceError("HTTP redirects are forbidden")


class Budget:
    """Single-process, sequential HTTP budget, including the tiny preflight.

    protocol_sha256 identifies the already-frozen protocol supplied by the
    runner. Cache reuse is verified and separately recorded; it consumes no
    additional HTTP bodies or requests. Do not share this object across threads.
    """

    def __init__(self, protocol_sha256, max_bytes=176 * MIB, max_requests=100):
        if not re.fullmatch(r"[0-9a-f]{64}", str(protocol_sha256)):
            raise SourceError("A frozen protocol SHA-256 is required")
        if not 207 <= max_bytes <= 176 * MIB or not 2 <= max_requests <= 100:
            raise SourceError("Budget may not exceed 176 MiB or 100 requests")
        self.protocol_sha256 = protocol_sha256
        self.max_bytes, self.max_requests = int(max_bytes), int(max_requests)
        self.body_bytes, self.requests = 207, 2
        self.events, self.assets = [], {}
        self.allowed_regions = None
        self.reference_ready = False
        self.failed = False

    def authorize_regions(self, regions):
        """Freeze exactly four disjoint label-selected 512x640 regions."""
        if self.failed or not self.reference_ready or self.allowed_regions is not None:
            raise SourceError("Regions require a verified reference and one-time authorization")
        regions = tuple(_bounds(b) for b in regions)
        if len(regions) != 4 or KNOWN_ROI not in regions:
            raise SourceError("Exactly four regions including the known Gemini ROI are required")
        for i, (y0, y1, x0, x1) in enumerate(regions):
            for a0, a1, b0, b1 in regions[:i]:
                if max(y0, a0) < min(y1, a1) and max(x0, b0) < min(x1, b1):
                    raise SourceError("Selected regions overlap")
        self.allowed_regions = frozenset(regions)

    def check(self, size, requests=1):
        if self.failed:
            raise SourceError("Acquisition previously failed; retries are forbidden")
        if self.body_bytes + size > self.max_bytes or self.requests + requests > self.max_requests:
            self.failed = True
            raise SourceError("Acquisition budget exceeded before request")

    def summary(self):
        return {
            "protocol_sha256": self.protocol_sha256,
            "total_response_body_bytes_including_preflight": self.body_bytes,
            "total_requests_including_preflight": self.requests,
            "max_response_body_bytes": self.max_bytes, "max_requests": self.max_requests,
            "metadata_preflight": MANIFEST["metadata_preflight"],
            "authorized_regions_y0_y1_x0_x1": sorted(self.allowed_regions or ()),
            "failed": self.failed, "network_events": self.events,
            "whole_asset_receipts": list(self.assets.values()),
        }


def _bounds(bounds):
    b = tuple(bounds)
    if len(b) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in b):
        raise SourceError("Bounds must be four integers: y0,y1,x0,x1")
    y0, y1, x0, x1 = map(int, b)
    if not (0 <= y0 < y1 <= SHAPE[1] and 0 <= x0 < x1 <= SHAPE[2]):
        raise SourceError("Region is outside the pinned array")
    if (y1 - y0, x1 - x0) != (512, 640) or any(v % 128 for v in b):
        raise SourceError("Each region must be chunk-aligned and exactly 512x640")
    return y0, y1, x0, x1


def _cache_root(cache):
    root = Path(cache).expanduser().resolve()
    if any((p / ".git").exists() for p in (root, *root.parents)):
        raise SourceError("The private acquisition cache must be outside Git working trees")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    return root


def _get(cache, budget, url, size, etag=None, sha256=None):
    """Read a whole pinned asset, with no redirects, retries or silent fallback."""
    if budget.failed:
        raise SourceError("Acquisition previously failed")
    root = _cache_root(cache)
    key = hashlib.sha256(url.encode()).hexdigest()
    body_path, receipt_path = root / (key + ".bin"), root / (key + ".json")
    event = None
    try:
        if body_path.exists() or receipt_path.exists():
            if not body_path.is_file() or not receipt_path.is_file():
                raise SourceError("Incomplete cached asset; no automatic replacement")
            receipt = json.loads(receipt_path.read_text())
            if body_path.stat().st_size != size:
                raise SourceError("Cached asset has the wrong length")
            raw = body_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if (receipt.get("url") != url or receipt.get("body_bytes") != size
                    or receipt.get("sha256") != digest
                    or (etag and receipt.get("etag") != etag)
                    or (sha256 and digest != sha256)):
                raise SourceError("Cached asset identity failed validation")
            budget.assets[url] = {**receipt, "used_from_cache": True}
            return raw
        budget.check(size)
        budget.requests += 1
        event = {"url": url, "method": "GET", "body_bytes": 0, "complete": False}
        budget.events.append(event)
        headers = {"Accept-Encoding": "identity", "User-Agent": "t5-frozen-replication/1"}
        if etag:
            headers["If-Match"] = etag
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=45) as response:
            event["status"] = response.status
            actual_etag = response.headers.get("ETag")
            if (response.status != 200 or response.geturl() != url
                    or response.headers.get("Content-Length") != str(size)
                    or response.headers.get("Content-Encoding", "identity") != "identity"
                    or not actual_etag or (etag and actual_etag != etag)):
                raise SourceError("HTTP status, location, encoding, size or ETag mismatch")
            data = bytearray()
            while len(data) < size:
                part = response.read(min(65536, size - len(data)))
                budget.body_bytes += len(part)
                event["body_bytes"] += len(part)
                if not part:
                    raise SourceError("Short response body")
                data.extend(part)
            raw = bytes(data)
        digest = hashlib.sha256(raw).hexdigest()
        if sha256 and digest != sha256:
            raise SourceError("Downloaded asset SHA-256 mismatch")
        receipt = {"url": url, "body_bytes": len(raw), "etag": actual_etag,
                   "sha256": digest, "protocol_sha256": budget.protocol_sha256}
        # Directory is private; partial or unpaired cache files fail closed next time.
        with body_path.open("xb") as stream:
            os.chmod(body_path, 0o600)
            stream.write(raw)
        with receipt_path.open("x") as stream:
            os.chmod(receipt_path, 0o600)
            json.dump(receipt, stream, indent=2)
        event.update(complete=True, sha256=digest, etag=actual_etag)
        budget.assets[url] = {**receipt, "used_from_cache": False}
        return raw
    except Exception as exc:
        budget.failed = True
        if event is not None:
            event["error_type"] = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            if event is not None:
                event["status"] = exc.code
            exc.close()
        raise


def _metadata(cache, budget):
    spec = MANIFEST["zarr"]
    raw = _get(cache, budget, spec["metadata_url"], spec["content_length"],
               spec["etag"], spec["sha256"])
    if json.loads(raw) != spec["metadata"]:
        budget.failed = True
        raise SourceError("Zarr metadata does not exactly match the frozen manifest")


def fetch_reference(cache, budget):
    """Fetch/validate metadata and the entire model TIFF; return its uint8 map."""
    _metadata(cache, budget)
    spec = MANIFEST["reference"]
    raw = _get(cache, budget, spec["url"], spec["content_length"], spec["etag"])
    try:
        # Its pinned 145M-pixel geometry is above Pillow's generic warning limit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if (image.format != "TIFF" or image.mode != spec["expected_pillow_mode"]
                        or image.size != tuple(reversed(spec["expected_shape_yx"]))
                        or image.n_frames != spec["expected_frames"]):
                    raise SourceError("Reference TIFF geometry, mode or frames mismatch")
                array = np.asarray(image)
        if array.dtype != np.uint8 or array.shape != tuple(spec["expected_shape_yx"]):
            raise SourceError("Decoded reference is not the exact expected uint8 array")
        budget.reference_ready = True
        return array
    except Exception:
        budget.failed = True
        raise


def fetch_region(cache, budget, bounds):
    """Return the authorized region as (109,512,640) uint8, in pinned C order."""
    bounds = _bounds(bounds)
    if not budget.reference_ready or budget.allowed_regions is None or bounds not in budget.allowed_regions:
        raise SourceError("Region is not in the frozen label-only selection")
    _metadata(cache, budget)
    y0, y1, x0, x1 = bounds
    result = np.empty((SHAPE[0], y1 - y0, x1 - x0), dtype=np.uint8)
    for cy in range(y0 // 128, y1 // 128):
        for cx in range(x0 // 128, x1 // 128):
            url = f'{MANIFEST["zarr"]["array_url"]}/0/{cy}/{cx}'
            raw = _get(cache, budget, url, CHUNK_BYTES)
            chunk = np.frombuffer(raw, dtype=np.uint8).reshape(CHUNKS, order="C")
            yy, xx = cy * 128 - y0, cx * 128 - x0
            result[:, yy:yy + 128, xx:xx + 128] = chunk
    return result
