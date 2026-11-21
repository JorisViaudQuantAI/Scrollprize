#!/usr/bin/env python3
"""Replay the frozen audit from pinned public geometry; no CT or ink access.

Only aggregate fields are serialized. Original-ray peaks are compared before
and after the along-original-normal displacement. New-ray peak indices are
never treated as corresponding crest identities.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from audit import audit_points, finite

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMMIT = "1ed75191bbef1c0fa7834c1249f4b0bb8c0eaeaf"
BASE = "registration-checks/atlas-0813/"
FILES = ("paired/geometry.json", "paired/sites.csv", "paired/summary.json",
         "paired/verify.py", "paired/README.md", "result.json", "check.py", "protocol.txt")
PROTOCOL_SHA256 = "c826b7ff84b451786765b59d82b54f53774dca585975ff45a6faac8610e660cf"
UNIQUE_ATOL_VOX = 1e-9


def digest(data):
    return hashlib.sha256(data).hexdigest()


def nearest(values):
    minimum = min(values)
    # Exactly the source check.py's rtol=0, atol=1e-9 convention.
    return minimum, [i for i, value in enumerate(values)
                     if abs(value - minimum) <= UNIQUE_ATOL_VOX]


def adapt_record(record):
    """Return a legacy-schema row and classification; never fill a tied identity."""
    try:
        raw = record["rays"]["raw"]
        peaks = raw["peak_offsets_vox"]
        support = raw["material_center_fraction"]
        displacement = record["along_raw_normal_vox"]
    except (KeyError, TypeError):
        return {}, "unknown"
    if (not isinstance(peaks, list) or not all(finite(p) for p in peaks)
            or not finite(support) or not 0 <= support <= 1
            or not finite(displacement)
            or any(a >= b for a, b in zip(peaks, peaks[1:]))):
        return {}, "unknown"
    row = dict(material_fraction=support, n_peaks=len(peaks))
    if support < 0.8 or not peaks:
        return row, "unsupported"
    before, near0 = nearest([abs(p) for p in peaks])
    after, near1 = nearest([abs(p - displacement) for p in peaks])
    row.update(raw_nearest=before, snapped_nearest=after,
               unique_raw=len(near0) == 1, unique_snapped=len(near1) == 1)
    if len(near0) == 1:
        row["after_same_peak"] = abs(peaks[near0[0]] - displacement)
        if len(near1) == 1:
            row["nearest_peak_changed"] = near0[0] != near1[0]
    # Verify redundant published legacy fields instead of trusting a changed
    # flag that the original offline verifier does not itself compare directly.
    for key, value, indices in (("legacy_raw_nearest", before, near0),
                                ("legacy_snap_nearest", after, near1)):
        if key in record:
            stored = record[key]
            if (stored["indices"] != indices or not math.isclose(
                    stored["minimum"], value, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("Published legacy nearest fields disagree with the fixed original ray")
    changed = row.get("nearest_peak_changed")
    if "legacy_nearest_peak_changed" in record and record["legacy_nearest_peak_changed"] is not changed:
        raise ValueError("Published switch flag disagrees with unique fixed-ray identity")
    if len(near0) == 1:
        for key, value in (("fixed_landmark_raw_offset_vox", peaks[near0[0]]),
                           ("legacy_same_peak_after_vox", row["after_same_peak"])):
            if key in record and (not finite(record[key]) or not math.isclose(
                    record[key], value, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("Published fixed landmark disagrees with original-ray reconstruction")
    elif record.get("fixed_landmark_raw_offset_vox") is not None:
        raise ValueError("An initially tied crest cannot supply a unique fixed landmark")
    return row, ("ambiguous" if changed is None else "switched" if changed else "unchanged")


def summary(values):
    known = [v for v in values if finite(v)]
    return dict(n=len(known), unknown=len(values)-len(known),
                median=statistics.median(known) if known else None,
                maximum=max(known) if known else None)


def grouped_geometry(records, classifications, pitch):
    result = {}
    for group in ("switched", "unchanged", "ambiguous", "unknown", "unsupported"):
        selected = [r for r, c in zip(records, classifications) if c == group]
        if not selected:
            continue
        out = dict(sites=len(selected), normal_change_deg=summary(
            [r.get("normal_angle_deg") for r in selected]))
        for frame, name in (("snapped_uv", "fixed_reference_transverse_miss_same_uv_um"),
                            ("projected_raw", "fixed_reference_transverse_miss_projected_raw_um")):
            values = []
            for record in selected:
                value = record.get("rays", {}).get(frame, {}).get("fixed_raw_landmark", {}).get("ray_miss_vox")
                values.append(value * pitch if finite(value) else None)
            out[name] = summary(values)
        result[group] = out
    return result


def main():
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    if digest(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("Frozen audit protocol changed; this replay refuses to run")
    inputs = {name: subprocess.check_output(
        ["git", "show", COMMIT + ":" + BASE + name], cwd=REPO) for name in FILES}
    # The pinned verifier was reviewed: it only reads these public files and
    # calculates identities. It neither fetches CT nor writes artifacts.
    with tempfile.TemporaryDirectory(prefix="scrollprize-paired-verify-") as temporary:
        root = Path(temporary)
        for name, data in inputs.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        verification = subprocess.run([sys.executable, str(root / "paired/verify.py")],
                                      capture_output=True, text=True, timeout=60)
        if verification.returncode:
            raise ValueError("Pinned offline coordinate verifier failed; no audit result written")
    doc = json.loads(inputs["paired/geometry.json"])
    protocol = json.loads(protocol_bytes)
    if doc["source_protocol_sha256"] != protocol["source_protocol_sha256"]:
        raise ValueError("Published source protocol is not the frozen source protocol")
    pitch = doc["pitch_um"]
    if not finite(pitch) or pitch <= 0:
        raise ValueError("Missing physical voxel pitch")
    records = doc["records"]
    if any(r.get("pitch_um") != pitch for r in records):
        raise ValueError("Inconsistent physical voxel pitches")
    adapted, groups = zip(*(adapt_record(r) for r in records))
    report = audit_points(list(adapted))
    report.update(source_commit=COMMIT,
                  source_files_sha256={BASE + name: digest(data) for name, data in inputs.items()},
                  protocol_sha256=digest(protocol_bytes), adapter_sha256=digest(Path(__file__).read_bytes()),
                  audit_code_sha256=digest((HERE / "audit.py").read_bytes()),
                  source_coordinate_verifier="passed", native_pitch_um=pitch,
                  nearest_uniqueness_absolute_tolerance_vox=UNIQUE_ATOL_VOX,
                  maximum_same_crest_distance_increase_vox=protocol["point_audit"]["maximum_same_crest_distance_increase_vox"],
                  classifications={g: groups.count(g) for g in sorted(set(groups))},
                  initial_ties=sum(r.get("unique_raw") is False for r in adapted),
                  adjusted_fixed_ray_ties=sum(r.get("unique_snapped") is False for r in adapted),
                  grouped_geometry=grouped_geometry(records, groups, pitch),
                  missing_evidence_interpretation="The legacy schema cannot give after_same_peak or changed identity for an initially tied crest; that complete-but-ambiguous source record remains an abstention, not a guessed identity.",
                  transverse_miss_definition="Distance from the initial unique original-ray crest point to the indicated changed-pose central ray; source ray_miss_vox times native pitch. It is not a sheet-identity error bound.",
                  limitations=["Retrospective geometry-only audit on previously summarized sites; no fresh validation or new CT acquisition.",
                               "The switched/unchanged split is exploratory and has no calibrated significance test.",
                               "Unchanged indices refer only to the same fixed original-ray peak list, never indices across newly sampled poses.",
                               "No physical sheet, ink, reading, or whole-winding correctness is established.",
                               "No empirical crest-location uncertainty bound; the optional interval gate remains unevaluated."])
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
