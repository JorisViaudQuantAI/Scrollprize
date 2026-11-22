#!/usr/bin/env python3
"""Frozen Frag5 source-contract and unlabeled native-pair anatomy experiment.

No ink labels, photographs, neural models, learned registration or best-shift
selection. This program makes network/pixel requests only with --run and an
explicit 40-character published freeze reference. --check-plan is local only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CPU_START = time.process_time()

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "4"

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata

CHUNK_SHAPE = (128, 128, 128)
CHUNK_BYTES = 128 ** 3 * 2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def box_chunks(lo_xyz, hi_xyz, shape_zyx):
    """All Cartesian chunk cells intersecting an inclusive integer XYZ box."""
    lo, hi = np.asarray(lo_xyz, dtype=np.int64), np.asarray(hi_xyz, dtype=np.int64)
    shape = np.asarray(shape_zyx[::-1], dtype=np.int64)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo < 0) or np.any(hi >= shape) or np.any(hi < lo):
        raise ValueError("Invalid complete source support; coordinates must not be clipped")
    return sorted(tuple(int(v) for v in xyz[::-1]) for xyz in itertools.product(
        *[range(int(a // 128), int(b // 128) + 1) for a, b in zip(lo, hi)]))


def bounds(points_xyz):
    values = np.asarray(points_xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite scanner XYZ points")
    return np.floor(values.min(0)).astype(np.int64), np.ceil(values.max(0)).astype(np.int64)


def corners(lo, hi):
    return np.asarray(list(itertools.product(*zip(lo, hi))), dtype=np.float64)


def expected_geometry(plan):
    """Recompute support and grids from pinned geometry; no intensity access."""
    transform = np.asarray(plan["affine_xyz"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError("Expected finite homogeneous column-XYZ affine")
    if not np.isfinite(transform).all():
        raise ValueError("Nonfinite affine")
    A, t = transform[:3, :3], transform[:3, 3]
    inverse = np.linalg.inv(A)
    fine_keys, coarse_keys, identity_keys = set(), set(), set()
    computed = []
    for item in plan["centers"]:
        p = np.asarray(item["fine_xyz"], dtype=np.float64)
        n = np.asarray(item["stored_normal_xyz"], dtype=np.float64)
        if p.shape != (3,) or n.shape != (3,) or not np.isfinite([p, n]).all() or abs(np.linalg.norm(n) - 1) > 1e-6:
            raise ValueError("Invalid preselected PPM record; no normalization or replacement")
        center = np.rint(A @ p + t).astype(np.int64)
        # Nominal cube integration uses midpoint quadrature with maximum |u|=.4.
        fine_box = corners(center - 16.4, center + 16.4)
        fine_points = (fine_box - t) @ inverse.T
        ray = p + np.arange(-32, 33, dtype=np.float64)[:, None] * n
        ray_lo, ray_hi = bounds(ray)
        fine_lo, fine_hi = bounds(np.vstack([fine_points, ray]))
        coarse_lo, coarse_hi = center - 17, center + 17
        ik = box_chunks(ray_lo, ray_hi, plan["volumes"]["fine"]["metadata"]["shape"])
        fk = box_chunks(fine_lo, fine_hi, plan["volumes"]["fine"]["metadata"]["shape"])
        ck = box_chunks(coarse_lo, coarse_hi, plan["volumes"]["coarse"]["metadata"]["shape"])
        identity_keys.update(ik); fine_keys.update(fk); coarse_keys.update(ck)
        computed.append({"id": item["id"], "coarse_center_xyz": center.tolist(),
                         "identity_fine_box_xyz": [ray_lo.tolist(), ray_hi.tolist()],
                         "anatomy_fine_box_xyz": [fine_lo.tolist(), fine_hi.tolist()],
                         "anatomy_coarse_box_xyz": [coarse_lo.tolist(), coarse_hi.tolist()]})
    count = len(fine_keys) + len(coarse_keys)
    if count > 64:
        raise ValueError(f"Prospective support requires {count} decoded chunks, exceeding 64")
    return {"centers": computed,
            "fine_chunk_keys_zyx": [list(k) for k in sorted(fine_keys)],
            "coarse_chunk_keys_zyx": [list(k) for k in sorted(coarse_keys)],
            "identity_fine_chunk_keys_zyx": [list(k) for k in sorted(identity_keys)],
            "decoded_chunks": count, "decoded_bytes_upper_bound": count * CHUNK_BYTES}


def check_plan(plan):
    if plan["schema"] != "frag5-native-pair-source-anatomy-v1":
        raise ValueError("Unexpected protocol schema")
    expected = {"source_chunk_body_bytes": 268435456, "tiff_metadata_body_bytes": 2097152,
                "decoded_chunks": 64, "cpu_seconds": 3600, "threads": 4}
    if plan["caps"] != expected:
        raise ValueError("Resource caps changed")
    if len(plan["centers"]) != 4 or len({x["id"] for x in plan["centers"]}) != 4:
        raise ValueError("Exactly four distinct frozen geometry centers required")
    if plan["algorithm"] != {"coarse_cube_side": 33, "midpoint_quadrature_per_axis": 5,
                              "shift_xyz_values": [-1, 0, 1], "tiff_depth_indices": list(range(65)),
                              "profile_normalization": "none", "profile_phase": "k-32",
                              "identity_tolerance_uint16": 1, "reverse_identity_sensitivity": True,
                              "anatomy_requires_unique_identity_match": True,
                              "label_stage_automatic_go": False}:
        raise ValueError("Unexpected experimental algorithm")
    for name in ("fine", "coarse"):
        volume = plan["volumes"][name]
        m = volume["metadata"]
        if (m["chunks"] != [128, 128, 128] or m["dtype"] != "<u2" or m["order"] != "C"
                or m["zarr_format"] != 2 or m.get("filters") not in (None, [])
                or m.get("dimension_separator") != "/" or m["compressor"]["id"] != "blosc"):
            raise ValueError("Unsupported raw Zarr contract")
    actual = expected_geometry(plan)
    if actual != plan["support_plan"]:
        raise ValueError("Frozen chunk/support plan does not reproduce")
    return actual


class Acquisition:
    def __init__(self, out, caps, started_cpu):
        self.out, self.caps, self.started_cpu = out, caps, started_cpu
        self.lock = threading.RLock()
        self.heads, self.head_receipts, self.receipts = {}, [], []
        self.bytes = {"source_chunk": 0, "tiff_metadata": 0}

    def check_cpu(self):
        if time.process_time() - self.started_cpu > self.caps["cpu_seconds"]:
            raise RuntimeError("Frozen CPU budget exhausted")

    def save(self):
        with self.lock:
            write_json(self.out / "acquisition.json", {"heads": self.heads, "head_requests": self.head_receipts,
                       "responses": self.receipts,
                       "body_bytes": self.bytes, "cpu_seconds": time.process_time() - self.started_cpu})

    @staticmethod
    def safe_url(url):
        u = urlparse(url)
        if u.scheme != "https" or u.hostname != "dl.ash2txt.org" or u.query or u.fragment or u.username or u.password:
            raise ValueError("Only canonical public dl.ash2txt.org HTTPS objects are permitted")

    def head(self, url):
        self.safe_url(url)
        self.check_cpu()
        receipt = {"url": url, "status": "started"}
        with self.lock:
            self.head_receipts.append(receipt)
        try:
            with urlopen(Request(url, method="HEAD"), timeout=45) as response:
                if response.status != 200 or response.geturl() != url:
                    raise ValueError("Unexpected HEAD status or redirect")
                length, etag = int(response.headers["Content-Length"]), response.headers.get("ETag")
                if length < 1 or not etag:
                    raise ValueError("HEAD must declare positive length and ETag")
                value = {"length": length, "etag": etag, "last_modified": response.headers.get("Last-Modified")}
            with self.lock:
                self.heads[url] = value
                receipt.update({"status": "complete", "http_status": 200})
        except Exception as exc:
            with self.lock:
                receipt.update({"status": "failed", "error_type": type(exc).__name__,
                                "http_status": getattr(exc, "code", None)})
            raise
        finally:
            self.save()

    def fetch(self, url, category, byte_range=None):
        self.safe_url(url)
        self.check_cpu()
        head = self.heads[url]
        start, end = byte_range if byte_range is not None else (0, head["length"] - 1)
        expected = end - start + 1
        if start < 0 or expected < 1 or end >= head["length"]:
            raise ValueError("Requested body outside frozen HEAD bounds")
        maximum = self.caps[category + "_body_bytes"]
        with self.lock:
            if self.bytes[category] + expected > maximum:
                raise RuntimeError("Frozen body budget exhausted before request")
            # Charge full expected bytes prospectively, including any interrupted response.
            self.bytes[category] += expected
        request_headers = {"If-Match": head["etag"]}
        if byte_range is not None:
            request_headers["Range"] = f"bytes={start}-{end}"
        receipt = {"url": url, "category": category, "range": list(byte_range) if byte_range else None,
                   "expected_bytes": expected, "body_bytes_read": 0, "status": "started"}
        with self.lock:
            self.receipts.append(receipt)
        self.save()
        try:
            with urlopen(Request(url, headers=request_headers), timeout=45) as response:
                status = 206 if byte_range is not None else 200
                if response.status != status or response.geturl() != url:
                    raise ValueError("Unexpected response status/redirect; body not read")
                if response.headers.get("ETag") != head["etag"] or int(response.headers["Content-Length"]) != expected:
                    raise ValueError("Object changed or response length is wrong; body not read")
                if byte_range is not None and response.headers.get("Content-Range") != f"bytes {start}-{end}/{head['length']}":
                    raise ValueError("Incorrect Content-Range; body not read")
                pieces = []
                remaining = expected
                while remaining:
                    self.check_cpu()
                    piece = response.read(min(65536, remaining))
                    if not piece:
                        raise ValueError("Truncated response")
                    pieces.append(piece)
                    receipt["body_bytes_read"] += len(piece)
                    remaining -= len(piece)
                data = b"".join(pieces)
                if response.read(1):
                    raise ValueError("Unexpected excess body")
            key = sha256((url + str(byte_range)).encode())
            target = self.out / "objects" / (key + ".bin")
            target.write_bytes(data)
            with self.lock:
                receipt.update({"status": "complete", "sha256": sha256(data), "cache_file": str(target.relative_to(self.out))})
            return data
        except Exception as exc:
            with self.lock:
                receipt.update({"status": "failed", "error_type": type(exc).__name__,
                                "http_status": getattr(exc, "code", None)})
            raise
        finally:
            self.save()


class Volume:
    def __init__(self, specification, acquisition):
        self.spec, self.acquisition, self.chunks = specification, acquisition, {}
        import numcodecs
        numcodecs.blosc.set_nthreads(4)
        self.codec = numcodecs.get_codec(specification["metadata"]["compressor"])

    def url(self, key):
        return self.spec["base_url"].rstrip("/") + "/" + "/".join(map(str, key))

    def load(self, keys):
        missing = [tuple(k) for k in keys if tuple(k) not in self.chunks]
        def one(key):
            raw = self.acquisition.fetch(self.url(key), "source_chunk")
            return key, raw
        with ThreadPoolExecutor(max_workers=4) as pool:
            acquired = list(pool.map(one, missing))
        # Finish the IO pool before starting the four-thread decoder.
        for key, raw in acquired:
            if len(raw) < 16:
                raise ValueError("Truncated Blosc header")
            header = struct.unpack("<4B3I", raw[:16])
            if header[4] != CHUNK_BYTES or header[6] != len(raw):
                raise ValueError("Blosc declares an unexpected decoded size or object length")
            decoded = self.codec.decode(raw)
            if len(decoded) != CHUNK_BYTES:
                raise ValueError("Decoded raw chunk is not exactly 128³ uint16")
            self.chunks[key] = np.frombuffer(decoded, dtype="<u2").reshape(CHUNK_SHAPE)
        self.acquisition.check_cpu()

    def region(self, box_xyz):
        lo, hi = (np.asarray(v, dtype=np.int64) for v in box_xyz)
        keys = box_chunks(lo, hi, self.spec["metadata"]["shape"])
        output = np.empty(tuple((hi - lo + 1)[::-1]), dtype=np.uint16)
        for key in keys:
            chunk = self.chunks[key]  # Missing data raises; never zero-fill.
            begin_xyz = np.asarray(key[::-1]) * 128
            a, b = np.maximum(lo, begin_xyz), np.minimum(hi + 1, begin_xyz + 128)
            output[tuple(slice(int(x), int(y)) for x, y in zip((a-lo)[::-1], (b-lo)[::-1]))] = chunk[
                tuple(slice(int(x), int(y)) for x, y in zip((a-begin_xyz)[::-1], (b-begin_xyz)[::-1]))]
        return Region(output, lo)


class Region:
    def __init__(self, values_zyx, origin_xyz):
        self.values = np.asarray(values_zyx, dtype=np.float64)
        self.origin = np.asarray(origin_xyz, dtype=np.float64)

    def sample(self, points_xyz):
        q = np.asarray(points_xyz, dtype=np.float64) - self.origin
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError("Invalid sample coordinates")
        if np.any(np.floor(q) < 0) or np.any(np.ceil(q) >= np.asarray(self.values.shape[::-1])):
            raise ValueError("Sample lacks complete trilinear support")
        return map_coordinates(self.values, q.T[::-1], order=1, mode="constant", cval=np.nan, prefilter=False)


def check_tiff_header(acquisition, plan, depth):
    url = plan["surface_base_url"] + f"/surface_volume/{depth:02}.tif"
    h = acquisition.fetch(url, "tiff_metadata", (0, 7))
    if h[:4] != b"II*\x00":
        raise ValueError("Expected little-endian classic TIFF")
    ifd = struct.unpack("<I", h[4:])[0]
    if ifd != 66658088:
        raise ValueError("TIFF IFD pointer differs from the metadata preflight")
    # All source files must independently satisfy the declared 12-tag contract.
    raw = acquisition.fetch(url, "tiff_metadata", (ifd, ifd + 149))
    count = struct.unpack("<H", raw[:2])[0]
    if count != 12 or struct.unpack("<I", raw[146:150])[0] != 0:
        raise ValueError("Unexpected TIFF IFD structure")
    tags = {}
    for j in range(count):
        tag, dtype, length = struct.unpack("<HHI", raw[2+12*j:10+12*j])
        payload = raw[10+12*j:14+12*j]
        if dtype in (3, 4) and length == 1:
            tags[tag] = struct.unpack("<H" if dtype == 3 else "<I", payload[:2] if dtype == 3 else payload)[0]
    expected = {256:4560,257:7309,258:16,259:1,262:1,273:8,277:1,278:7309,279:66658080,284:1,339:1}
    if any(tags.get(k) != v for k, v in expected.items()):
        raise ValueError("TIFF raster contract changed")
    if acquisition.heads[url]["length"] != plan["tiff_object_bytes"]:
        raise ValueError("TIFF object length changed")
    return {"depth": depth, "ifd_offset": ifd, "tags": tags}


def tiff_profile(acquisition, plan, depth):
    url = plan["surface_base_url"] + f"/surface_volume/{depth:02}.tif"
    values = []
    for item in plan["centers"]:
        y, x = item["surface_yx"]
        start = 8 + 2 * (y * 4560 + x)
        values.append(struct.unpack("<H", acquisition.fetch(url, "tiff_metadata", (start, start+1)))[0])
    return depth, values


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid metric inputs")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"n": int(a.size), "pearson": None, "spearman": None, "status": "constant_unassessable"}
    return {"n": int(a.size), "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), "status": "assessable"}


def laplacian(values):
    return (values[2:,1:-1,1:-1] + values[:-2,1:-1,1:-1] + values[1:-1,2:,1:-1] +
            values[1:-1,:-2,1:-1] + values[1:-1,1:-1,2:] + values[1:-1,1:-1,:-2]
            - 6 * values[1:-1,1:-1,1:-1])


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "minimum": float(a.min()), "maximum": float(a.max()),
            "mean": float(a.mean()), "sd": float(a.std()), "nonzero": int(np.count_nonzero(a)),
            "zeros": int(np.count_nonzero(a == 0)), "at_65535": int(np.count_nonzero(a == 65535))}


def classify_identity(rows):
    primary = all(r["nonconstant_nonzero"] and r["primary_max_abs_error_uint16"] <= 1 for r in rows)
    reverse = all(r["nonconstant_nonzero"] and r["reverse_max_abs_error_uint16"] <= 1 for r in rows)
    classification = ("both_ambiguous" if primary and reverse else "primary_matched" if primary
                      else "reverse_only" if reverse else "neither")
    return {"classification": classification, "primary_all_four_pass": primary,
            "reverse_all_four_pass": reverse, "anatomy_allowed": bool(primary != reverse)}


def run(plan, output, freeze_reference, started_cpu):
    started_wall = time.monotonic()
    acquisition = Acquisition(output, plan["caps"], started_cpu)
    try:
        fine, coarse = (Volume(plan["volumes"][n], acquisition) for n in ("fine", "coarse"))
        s = plan["support_plan"]
        raw_urls = ([fine.url(k) for k in s["fine_chunk_keys_zyx"]] +
                    [coarse.url(k) for k in s["coarse_chunk_keys_zyx"]])
        tiff_urls = [plan["surface_base_url"] + f"/surface_volume/{k:02}.tif" for k in range(65)]
        metadata_urls = [plan["volumes"][n]["metadata_url"] for n in ("fine", "coarse")]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(acquisition.head, raw_urls + tiff_urls + metadata_urls))
        planned_bytes = sum(acquisition.heads[u]["length"] for u in raw_urls)
        if planned_bytes > plan["caps"]["source_chunk_body_bytes"]:
            raise RuntimeError("HEAD-declared raw source payload exceeds 256 MiB")
        for u in tiff_urls:
            if acquisition.heads[u]["length"] != plan["tiff_object_bytes"]:
                raise ValueError("TIFF HEAD length differs from frozen raster")
        write_json(output / "acquisition_heads_freeze.json", {
            "freeze_reference": freeze_reference, "heads": acquisition.heads,
            "planned_source_chunk_bytes": planned_bytes, "decoded_chunks": s["decoded_chunks"],
            "no_CT_or_TIFF_pixel_body_access_before_this_receipt": True})
        for name in ("fine", "coarse"):
            spec = plan["volumes"][name]
            metadata = acquisition.fetch(spec["metadata_url"], "tiff_metadata")
            if sha256(metadata) != spec["metadata_sha256"] or json.loads(metadata) != spec["metadata"]:
                raise ValueError("Pinned raw Zarr metadata changed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            header_results = list(pool.map(lambda k: check_tiff_header(acquisition, plan, k), range(65)))
        write_json(output / "all_tiff_headers.json", {"headers": header_results,
                   "all_65_headers_checked_before_any_CT_or_TIFF_pixel_body": True})
        fine.load(s["identity_fine_chunk_keys_zyx"])
        measured = np.zeros((4, 65), dtype=np.uint16)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for depth, values in pool.map(lambda k: tiff_profile(acquisition, plan, k), range(65)):
                measured[:, depth] = values
        profiles, rows = [], []
        for item, geometry, observed in zip(plan["centers"], s["centers"], measured):
            p, n = np.asarray(item["fine_xyz"]), np.asarray(item["stored_normal_xyz"])
            predicted = fine.region(geometry["identity_fine_box_xyz"]).sample(p + np.arange(-32,33)[:,None] * n)
            rounded = np.rint(predicted).astype(np.uint16)
            direct = np.abs(rounded.astype(np.int64) - observed.astype(np.int64))
            reverse = np.abs(rounded[::-1].astype(np.int64) - observed.astype(np.int64))
            direct_unrounded = np.abs(predicted - observed.astype(np.float64))
            reverse_unrounded = np.abs(predicted[::-1] - observed.astype(np.float64))
            informative = bool(np.ptp(rounded.astype(np.int64)) > 0 and np.ptp(observed.astype(np.int64)) > 0
                               and np.count_nonzero(rounded) > 0 and np.count_nonzero(observed) > 0)
            rows.append({"id": item["id"], "nonconstant_nonzero": informative,
                         "predicted_nonzero": int(np.count_nonzero(rounded)),
                         "tiff_nonzero": int(np.count_nonzero(observed)),
                         "primary_max_abs_error_uint16": int(direct.max()),
                         "reverse_max_abs_error_uint16": int(reverse.max()),
                         "primary_unrounded_max_abs_error": float(direct_unrounded.max()),
                         "reverse_unrounded_max_abs_error": float(reverse_unrounded.max()),
                         "primary_absolute_errors": direct.tolist(), "reverse_absolute_errors": reverse.tolist(),
                         "primary_unrounded_absolute_errors": direct_unrounded.tolist(),
                         "reverse_unrounded_absolute_errors": reverse_unrounded.tolist(),
                         "tiff_values": observed.tolist(), "fine_trilinear_values": predicted.tolist()})
            profiles.append(predicted)
        identity = {**classify_identity(rows), "profiles": rows,
                    "meaning": "Consistency with one declared source/order/phase/interpolation contract at four geometry-only points; not physical face or general rendering provenance"}
        write_json(output / "identity_result.json", identity)
        print(json.dumps({"stage": "identity_complete", "classification": identity["classification"],
                          "identity_sha256": sha256((output/"identity_result.json").read_bytes())}), flush=True)
        result = {"freeze_reference": freeze_reference, "identity": identity, "anatomy": [],
                  "labels_or_photographs_accessed": False, "registration_fitted": False,
                  "best_shift_selected": False, "automatic_label_stage_go": False}
        if identity["anatomy_allowed"]:
            fine.load(s["fine_chunk_keys_zyx"]); coarse.load(s["coarse_chunk_keys_zyx"])
            A = np.asarray(plan["affine_xyz"], dtype=np.float64)
            inverse, translation = np.linalg.inv(A[:3,:3]), A[:3,3]
            z, y, x = np.meshgrid(np.arange(-16,17), np.arange(-16,17), np.arange(-16,17), indexing="ij")
            offsets = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            quadrature = (np.arange(5, dtype=np.float64) + .5) / 5 - .5
            for item, geometry in zip(plan["centers"], s["centers"]):
                acquisition.check_cpu()
                center = np.asarray(geometry["coarse_center_xyz"])
                q = center + offsets
                fine_region = fine.region(geometry["anatomy_fine_box_xyz"])
                fine_point = fine_region.sample((q - translation) @ inverse.T).reshape(33,33,33)
                integrated = np.zeros(len(q), dtype=np.float64)
                for u in itertools.product(quadrature, repeat=3):
                    acquisition.check_cpu()
                    integrated += fine_region.sample((q + u - translation) @ inverse.T)
                integrated = (integrated / 125).reshape(33,33,33)
                coarse_region = coarse.region(geometry["anatomy_coarse_box_xyz"])
                measures = []
                for shift in itertools.product((-1,0,1), repeat=3):
                    native = coarse_region.sample(q + shift).reshape(33,33,33)
                    for arm, simulated in (("fine_point", fine_point), ("fine_nominal_coarse_footprint", integrated)):
                        measures.append({"shift_xyz_coarse_voxels": list(shift), "arm": arm,
                                         "original_transform_primary": shift == (0,0,0),
                                         "intensity": correlation(simulated, native),
                                         "laplacian_interior": correlation(laplacian(simulated), laplacian(native))})
                arrays = output / f"anatomy_{item['id']}.npz"
                np.savez_compressed(arrays, fine_point=fine_point, fine_nominal_coarse_footprint=integrated,
                                    coarse_neighborhood=coarse_region.values,
                                    coarse_center_xyz=center, coarse_neighborhood_origin_xyz=coarse_region.origin)
                one = {"id": item["id"], "descriptive_statistics": {
                    "fine_point": describe(fine_point), "fine_nominal_coarse_footprint": describe(integrated),
                    "native_coarse_zero_shift": describe(coarse_region.sample(q))},
                    "within_fine_point_vs_footprint": correlation(fine_point, integrated),
                    "all_shifts": measures, "arrays_file": arrays.name, "arrays_sha256": sha256(arrays.read_bytes())}
                result["anatomy"].append(one)
                write_json(output / "progress_result.json", result)
                print(json.dumps({"stage": "anatomy_center_complete", "id": item["id"]}), flush=True)
            result["outcome"] = "completed_unlabeled_source_and_anatomy_comparison"
        else:
            result["outcome"] = ("source_contract_ambiguous_anatomy_not_run" if identity["classification"] == "both_ambiguous"
                                 else "declared_source_contract_failed_anatomy_not_run")
        result["resources"] = {"cpu_seconds": time.process_time()-started_cpu,
                               "wall_seconds": time.monotonic()-started_wall,
                               "body_bytes_prospectively_charged": acquisition.bytes,
                               "decoded_chunk_count": len(fine.chunks)+len(coarse.chunks)}
        acquisition.check_cpu()
        write_json(output / "result.json", result)
        acquisition.save()
        return result
    except Exception as exc:
        acquisition.save()
        write_json(output / "error.json", {"error_type": type(exc).__name__,
                   "message": "Execution stopped; inspect fixed protocol and acquisition receipt. No automatic retry or protocol change.",
                   "cpu_seconds": time.process_time()-started_cpu,
                   "wall_seconds": time.monotonic()-started_wall})
        raise


def main():
    started_cpu = CPU_START
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check-plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-reference")
    args = parser.parse_args()
    if args.check_plan == args.run:
        parser.error("Choose exactly one of --check-plan and --run")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    check_plan(plan)
    if args.check_plan:
        print(json.dumps({"plan_sha256": sha256(plan_bytes), "support_plan": plan["support_plan"]}, indent=2))
        return
    if not args.out or not args.freeze_reference or not re.fullmatch("[0-9a-f]{40}", args.freeze_reference):
        parser.error("--run requires a new --out directory and full published --freeze-reference")
    if sha256(Path(__file__).read_bytes()) != plan["runner_sha256"]:
        raise ValueError("Runner differs from the frozen plan")
    if args.out.exists():
        raise FileExistsError("Preserve prior outcomes; the execution directory must be new")
    args.out.mkdir(parents=True); (args.out / "objects").mkdir()
    import scipy
    import numcodecs
    write_json(args.out / "start.json", {"freeze_reference": args.freeze_reference,
               "plan_sha256": sha256(plan_bytes), "runner_sha256": plan["runner_sha256"],
               "caps": plan["caps"], "numpy_version": np.__version__, "python_version": sys.version,
               "scipy_version": scipy.__version__, "numcodecs_version": numcodecs.__version__,
               "no_label_or_photo_assets_in_protocol": True})
    result = run(plan, args.out, args.freeze_reference, started_cpu)
    print(json.dumps({"outcome": result["outcome"], "resources": result["resources"],
                      "result_sha256": sha256((args.out/"result.json").read_bytes())}), flush=True)


if __name__ == "__main__":
    main()
