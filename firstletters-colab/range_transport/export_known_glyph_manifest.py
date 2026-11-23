#!/usr/bin/env python3
"""Offline, retrospective selected-byte provenance for the published glyph.

Usage: python export_known_glyph_manifest.py --run-dir COMPLETED_CONTROL_DIR
       python export_known_glyph_manifest.py --self-test

No HTTP client or detector is used. Existing frozen acquisition protocols remain
unchanged. Only the already-published control and its exact retained cache are
accepted. A future reader must pin the emitted manifest before any request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np


LAB = Path(__file__).resolve().parents[1]
FREEZE = "cada2a5497a47868d8dfde68f04e1a333a7723fb"
SOURCE_BASE = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHercParis4/segments/20231210121321/surface-volumes/"
    "2.4um-0.22m-78keV-volume-20260411134726.zarr"
)
FROZEN_FILES = {
    "public_glyph_control/protocol.json": "00a46e858b009858fa5c17c42d5fec9f3d1e25451d28f4af81d74e51a01e3e01",
    "public_glyph_control/selection.json": "d69fc05cde575d147fcca23645ca80db84eb86eb93ab4eee3a7dc1cf1921899e",
    "public_glyph_control/run.py": "cc3fb7ae83ee891c7d749ff3ad4603bac8ea2e7e5d7adc12c4aad7a9039dd113",
}
# These completed-run identities were fixed by the independent full-glyph review.
# They bind every expected cache hash before this retrospective exporter reads CT.
COMPLETED_INPUTS = {
    "result.json": "4785d83c49d5c0a1e0309b40dc5226154c3a2619120e85e4d51d528fc25efbee",
    "source_receipt.json": "0792908247c6a372ef5ff8ab7f6fa964330e5a91c3db2f4585c87c70a337bed2",
    "acquisition_plan.json": "5e724a4305ac670372b3013f086f51eded57da91b8b0447614ed21233acb6f8d",
}
MANIFEST_NAME = "paris4-known-glyph-selected-manifest.json"
REVIEW_NAME = "paris4-known-glyph-manifest-review.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def verified_body(path, expected_bytes, expected_sha256, label):
    """Reject missing, truncated, extended or same-size-corrupt source objects."""
    require(path.is_file(), "Missing source object: " + label)
    require(path.stat().st_size == expected_bytes, "Source size mismatch: " + label)
    with path.open("rb") as stream:
        body = stream.read(expected_bytes + 1)
    require(len(body) == expected_bytes, "Source changed length: " + label)
    require(hashlib.sha256(body).hexdigest() == expected_sha256,
            "Source SHA256 mismatch: " + label)
    return body


def rejection_checks():
    """Tiny generated fixtures only; never mutate the retained source cache."""
    body = bytes(range(16))
    expected = hashlib.sha256(body).hexdigest()
    checks = {}
    with tempfile.TemporaryDirectory(prefix="selected-manifest-check-") as tmp:
        path = Path(tmp) / "fixture"
        path.write_bytes(body)
        require(verified_body(path, len(body), expected, "generated") == body,
                "Valid generated fixture failed")
        checks["valid_fixture_accepted"] = True
        for name, candidate, payload in (
            ("missing_rejected", Path(tmp) / "absent", None),
            ("truncated_rejected", path, body[:-1]),
            ("same_size_corrupt_rejected", path, b"X" + body[1:]),
        ):
            if payload is not None:
                candidate.write_bytes(payload)
            try:
                verified_body(candidate, len(body), expected, "generated")
            except ValueError:
                checks[name] = True
            else:
                raise ValueError("Generated rejection check failed: " + name)
    return checks


def independent_geometry(selection, source):
    require(selection["source_base"] == SOURCE_BASE, "Only the published control is allowed")
    require(source["chunks_zyx"] == [109, 128, 128]
            and source["shape_zyx"] == [109, 51000, 39980], "Native geometry changed")
    require(source["retained_source_z_halfopen"] == [22, 87]
            and source["retained_shape_zyx"] == [65, 2048, 2048], "Selected depth changed")
    require(source["dtype"] == "|u1" and source["order"] == "C"
            and source["compressor"] is None and source["filters"] is None
            and source["dimension_separator"] == "/", "Source is not raw native uint8")
    y0, x0 = selection["context"]["origin_yx"]
    h, w = selection["context"]["shape_yx"]
    y1, x1 = y0 + h, x0 + w
    require((h, w) == (2048, 2048)
            and selection["context"]["end_yx_exclusive"] == [y1, x1]
            and 0 <= y0 < y1 <= 51000 and 0 <= x0 < x1 <= 39980,
            "Context extent changed")
    plan = []
    for iy in range(y0 // 128, (y1 + 127) // 128):
        for ix in range(x0 // 128, (x1 + 127) // 128):
            top, left = iy * 128, ix * 128
            ay, by = max(y0, top), min(y1, top + 128)
            ax, bx = max(x0, left), min(x1, left + 128)
            plan.append({"key": f"0/{iy}/{ix}",
                         "source_yx": [ay-top, by-top, ax-left, bx-left],
                         "target_yx": [ay-y0, by-y0, ax-x0, bx-x0]})
    require(len(plan) == source["complete_chunk_count"] == 289, "Incomplete chunk geometry")
    return plan


def prior_range_check(chunks, range_interval, selected_bytes):
    base = LAB / "range_transport"
    whole_protocol = read_json(base / "paris4_native_range_protocol.json")
    sparse_protocol = read_json(base / "paris4_native_sparse_protocol.json")
    whole = read_json(base / "evidence/paris4-native-public-v1.json")["result"]
    sparse_evidence = read_json(base / "evidence/paris4-native-sparse-public-v1.json")
    sparse = sparse_evidence["runner_result"]
    url = whole_protocol["chunk_url"]
    require(url == sparse_protocol["chunk_url"] and url.startswith(SOURCE_BASE + "/"),
            "Prior Range source changed")
    relative = url[len(SOURCE_BASE) + 1:]
    row = next((x for x in chunks if x["relative_source_path"] == relative), None)
    require(row is not None, "Prior Range chunk absent from complete control")
    require(whole["passed"] and sparse["passed"] and sparse["pinned_reference_match"]
            and whole["mismatch_count"] == 0, "Prior Range evidence did not pass")
    require(whole_protocol["range_inclusive"] == sparse_protocol["range_inclusive"] == range_interval
            and whole_protocol["selected_bytes"] == sparse_protocol["selected_bytes"] == selected_bytes
            and whole["selected_bytes"] == selected_bytes, "Prior Range interval differs")
    require(row["full_sha256"] == whole["full_sha256"], "Prior whole-chunk SHA256 differs")
    require(row["selected_sha256"] == whole["selected_sha256"] == sparse["selected_sha256"]
            == sparse_protocol["pinned_selected_sha256"] == sparse_evidence["prior_selected_sha256"],
            "Prior selected-body SHA256 differs")
    return {"relative_source_path": relative, "full_sha256": row["full_sha256"],
            "selected_sha256": row["selected_sha256"], "both_prior_receipts_match": True}


def export(run_dir, output_dir):
    started = time.monotonic()
    checks = rejection_checks()
    require(not (output_dir / MANIFEST_NAME).exists()
            and not (output_dir / REVIEW_NAME).exists(), "Refusing to overwrite existing evidence")
    for name, expected in FROZEN_FILES.items():
        require(digest_file(LAB / name) == expected, "Frozen source changed: " + name)
    protocol = read_json(LAB / "public_glyph_control/protocol.json")
    selection = read_json(LAB / "public_glyph_control/selection.json")
    source = protocol["source"]
    plan = independent_geometry(selection, source)
    completed = {}
    for name, expected_sha in COMPLETED_INPUTS.items():
        path = run_dir / name
        require(path.is_file(), "Missing completed metadata: " + name)
        size = path.stat().st_size
        require(0 < size <= 1024 * 1024, "Unexpected completed metadata size: " + name)
        completed[name] = json.loads(verified_body(path, size, expected_sha, name))
    result = completed["result.json"]
    require(result["status"] == "complete" and result["freeze_reference"] == FREEZE,
            "A completed frozen public glyph run is required")
    receipt = completed["source_receipt.json"]
    saved_plan = completed["acquisition_plan.json"]
    require(saved_plan["chunk_plan"] == plan and saved_plan["chunk_count"] == 289
            and saved_plan["source_base"] == SOURCE_BASE
            and saved_plan["retained_source_z_halfopen"] == [22, 87]
            and saved_plan["complete_chunk_reads"] and saved_plan["missing_objects_are_fatal"]
            and saved_plan["no_resampling"], "Saved acquisition plan differs")
    require(not receipt["failed_assets"], "Source receipt contains failed acquisitions")
    rows = receipt["completed_assets"]
    by_url = {row["url"]: row for row in rows}
    expected_rel = {"0/" + item["key"] for item in plan}
    expected_rel.update(item["suffix"] for item in source["metadata"])
    require(len(rows) == len(by_url) == 291
            and set(by_url) == {SOURCE_BASE + "/" + s for s in expected_rel},
            "Missing, duplicate or unexpected source identities")
    require(receipt["requests_including_redirects"] == result["requests_including_redirects"] == 291
            and protocol["limits"]["requests_including_redirects"] == 400,
            "Historical request accounting or frozen cap differs")

    def consume(relative, size, expected_sha=None):
        url = SOURCE_BASE + "/" + relative
        row = by_url[url]
        require(row["bytes"] == size and row["reused_old_cache"] is False,
                "Source size or historical reuse differs: " + relative)
        require(expected_sha is None or row["sha256"] == expected_sha,
                "Pinned source SHA256 differs: " + relative)
        path = run_dir / "source_cache" / hashlib.sha256(url.encode()).hexdigest()
        return verified_body(path, size, row["sha256"], relative), row["sha256"]

    metadata, metadata_manifest = {}, []
    for item in source["metadata"]:
        body, sha = consume(item["suffix"], item["bytes"], item["sha256"])
        metadata[item["suffix"]] = json.loads(body)
        metadata_manifest.append({"relative_source_path": item["suffix"],
                                  "expected_bytes": len(body), "sha256": sha})
    zarray, attrs = metadata["0/.zarray"], metadata[".zattrs"]
    require(zarray["zarr_format"] == 2, "Unexpected Zarr version")
    for key, value in {"shape": "shape_zyx", "chunks": "chunks_zyx", "dtype": "dtype",
                       "order": "order", "compressor": "compressor", "filters": "filters",
                       "fill_value": "fill_value", "dimension_separator": "dimension_separator"}.items():
        require(zarray[key] == source[value], "Decoded native metadata differs: " + key)
    require(attrs["num_slices"] == 109 and len(attrs["multiscales"]) == 1,
            "Unexpected source depth/multiscale")
    ms = attrs["multiscales"][0]
    require([a["name"] for a in ms["axes"]] == ["z", "y", "x"]
            and all(a["unit"] == "micrometer" for a in ms["axes"]), "Source axes differ")
    native = [x["coordinateTransformations"] for x in ms["datasets"] if x["path"] == "0"]
    require(native == [[{"type": "scale", "scale": [2.4, 2.4, 2.4]},
                        {"type": "translation", "translation": [0.0, 0.0, 0.0]}]],
            "Native source pitch differs")

    full_bytes = 109 * 128 * 128
    start, stop = 22 * 128 * 128, 87 * 128 * 128
    selected_bytes = stop - start
    interval = [start, stop - 1]
    rebuilt = np.empty((65, 2048, 2048), dtype=np.uint8)
    coverage = np.zeros((2048, 2048), dtype=np.uint8)
    chunks = []
    for item in plan:
        relative = "0/" + item["key"]
        body, full_sha = consume(relative, full_bytes)
        selected = memoryview(body)[start:stop]
        selected_sha = hashlib.sha256(selected).hexdigest()
        slab = np.frombuffer(selected, dtype=np.uint8).reshape(65, 128, 128)
        sy0, sy1, sx0, sx1 = item["source_yx"]
        ty0, ty1, tx0, tx1 = item["target_yx"]
        require(not coverage[ty0:ty1, tx0:tx1].any(), "Reconstructed chunk footprints overlap")
        rebuilt[:, ty0:ty1, tx0:tx1] = slab[:, sy0:sy1, sx0:sx1]
        coverage[ty0:ty1, tx0:tx1] = 1
        chunks.append({"relative_source_path": relative, "full_sha256": full_sha,
                       "selected_sha256": selected_sha, "source_yx": item["source_yx"],
                       "target_yx": item["target_yx"]})
    require(np.all(coverage == 1), "Incomplete native union coverage")
    volume_path = run_dir / "source_union65_uint8.npy"
    union_npy_sha = digest_file(volume_path)
    require(union_npy_sha == result["source_union_file_sha256"], "Retained union file SHA256 differs")
    retained = np.load(volume_path, mmap_mode="r", allow_pickle=False)
    require(retained.shape == rebuilt.shape and retained.dtype == np.uint8
            and retained.flags.c_contiguous, "Unexpected retained union representation")
    mismatches = sum(int(np.count_nonzero(rebuilt[z] != retained[z])) for z in range(65))
    require(mismatches == 0, "Selected-byte union differs from retained native volume")
    raw_sha = hashlib.sha256(memoryview(rebuilt)).hexdigest()
    require(raw_sha == hashlib.sha256(memoryview(retained)).hexdigest()
            and digest_file(volume_path) == union_npy_sha, "Native union changed during validation")
    prior = prior_range_check(chunks, interval, selected_bytes)
    metadata_bytes = sum(x["expected_bytes"] for x in metadata_manifest)
    historical_bytes = 289 * full_bytes + metadata_bytes
    require(historical_bytes == source["metadata_and_chunk_body_bytes"]
            == receipt["new_application_body_bytes"] == result["new_application_body_bytes"],
            "Historical whole-body accounting differs")
    projection = {
        "status": "Future geometry-derived feasibility only; not measured transfer",
        "chunk_range_GET_requests": 289, "metadata_GET_requests": 2,
        "chunk_HEAD_requests": 0, "total_requests": 291,
        "selected_chunk_application_body_bytes": 289 * selected_bytes,
        "metadata_application_body_bytes": metadata_bytes,
        "total_application_body_bytes": 289 * selected_bytes + metadata_bytes,
        "unchanged_original_request_cap": 400,
        "new_network_reader_implemented": False,
        "requires_separate_future_protocol": True,
    }
    manifest = {
        "schema": "scrollprize.paris4-known-glyph-selected-manifest.v1",
        "scope": "Retrospective offline provenance transform of an already-published control; not a preregistered detector experiment",
        "original_glyph_freeze": FREEZE, "source_base_url": SOURCE_BASE,
        "source_layout": {"dtype": "|u1", "order": "C", "compressor": None,
                          "chunk_shape_zyx": [109, 128, 128], "full_chunk_bytes": full_bytes},
        "selection": {"source_z_halfopen": [22, 87], "range_inclusive": interval,
                      "selected_chunk_shape_zyx": [65, 128, 128],
                      "selected_chunk_bytes": selected_bytes,
                      "context_origin_yx": selection["context"]["origin_yx"],
                      "context_shape_zyx": [65, 2048, 2048]},
        "metadata": metadata_manifest, "chunks": chunks,
        "reconstructed_native_union_raw_sha256": raw_sha,
        "retained_native_union_npy_sha256": union_npy_sha,
        "future_geometry_projection": projection,
        "future_response_requirements": [
            "Pin this manifest SHA256 or its immutable repository revision before requesting any bytes",
            "Require HTTP206, exact Content-Range including the pinned full-object length, expected selected-body length and identity encoding; no retries or whole-object fallback unless separately authorised",
            "Verify every selected response against its prior per-chunk selected_sha256 before assembly, replacing a per-object HEAD with content identity validation",
            "Verify both complete metadata objects against their pinned sizes and SHA256 values",
            "Abort on any missing or mismatched object; never fill an absent chunk with zeros",
        ],
        "limitations": [
            "No full-context Range retrieval or speedup was measured by this exporter",
            "Per-object content identities do not authenticate an atomic snapshot of the mutable remote bucket",
            "Selected-byte hashes identify only the prescribed intervals, not future bytes elsewhere in those source objects",
            "No detector, new manuscript inference, image interpretation or prize claim",
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    references = dict(FROZEN_FILES)
    for name in ("paris4_native_range_protocol.json", "paris4_native_sparse_protocol.json",
                 "evidence/paris4-native-public-v1.json", "evidence/paris4-native-sparse-public-v1.json"):
        references["range_transport/" + name] = digest_file(LAB / "range_transport" / name)
    for name, expected_sha in COMPLETED_INPUTS.items():
        require(digest_file(run_dir / name) == expected_sha, "Completed metadata changed: " + name)
        references["completed_public_control/" + name] = expected_sha
    review = {
        "schema": "scrollprize.paris4-known-glyph-manifest-review.v1", "status": "complete",
        "scope": "Measured offline source validation and provenance transform; not an independent reviewer of this exporter",
        "exporter_sha256": digest_file(Path(__file__)), "input_sha256": references,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "metadata_objects_verified": 2, "full_native_chunks_verified": len(chunks),
        "full_native_chunk_bytes_verified": 289 * full_bytes,
        "selected_intervals_hashed": len(chunks), "selected_interval_bytes_hashed": 289 * selected_bytes,
        "original_native_union_bytes_compared": int(rebuilt.nbytes), "mismatch_count": mismatches,
        "native_union_coverage_complete": True, "native_union_raw_sha256": raw_sha,
        "retained_native_union_npy_sha256": union_npy_sha,
        "prior_one_chunk_range_comparison": prior, "generated_rejection_checks": checks,
        "historical_whole_acquisition_body_bytes": historical_bytes,
        "future_geometry_projection": projection, "atomic_remote_snapshot_established": False,
        "new_network_requests": 0, "new_downloaded_body_bytes": 0, "new_detector_forwards": 0,
        "frozen_protocols_changed": False, "images_opened": 0,
        "wall_seconds_through_validation": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_NAME).write_bytes(manifest_bytes)
    (output_dir / REVIEW_NAME).write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=LAB / "range_transport/evidence")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(rejection_checks(), sort_keys=True))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    review = export(args.run_dir, args.output_dir)
    print(json.dumps({k: review[k] for k in ("status", "full_native_chunks_verified",
          "original_native_union_bytes_compared", "mismatch_count", "manifest_sha256",
          "wall_seconds_through_validation", "peak_rss_kib")}, sort_keys=True))


if __name__ == "__main__":
    main()
