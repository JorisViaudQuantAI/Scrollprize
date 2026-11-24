#!/usr/bin/env python3
"""Audit the completed 2026-11-24 visual campaign without running a model.

Reads the original evidence bundle, rebases historical paths within that bundle,
and emits a compact native-adapter receipt. This is an evidence importer, not a
worker, scheduler, scientific calibration, or authorization to discard a chart.
The full source bundle is needed: the small public publication contains selected
numeric crops, not all original model inputs. No download or GPU is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from native_evidence import validate_native, assess_pair


SCHEMA = "funnel.native_evidence.v1"
SEGMENT_C = "20251028220955-auto_grown_20251028220955262"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rebase(value, root):
    """Relocate only paths rooted in the original bundle, never arbitrary files."""
    p = Path(value)
    if p.is_absolute():
        parts = p.parts
        if "discovery-private" not in parts:
            raise ValueError("Historical path is outside the documented bundle")
        p = Path(*parts[parts.index("discovery-private") + 1:])
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Artifact escaped the evidence bundle")
    if not resolved.is_file():
        raise ValueError(f"Missing source artifact: {p}")
    return resolved


def descriptor(path, root, expected_sha, axes):
    actual = sha(path)
    if actual != expected_sha:
        raise ValueError(f"Recorded source hash mismatch: {path.name}")
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"path": path.relative_to(root).as_posix(), "sha256": actual,
            "dtype": str(a.dtype), "shape": list(a.shape), "axes": axes}


def frame_for(name, job, root):
    if name.startswith("0125_"):
        variant = "snapped" if "snapped" in name else "raw"
        roi = job.get("full_canvas_roi_y0_y1_x0_x1", [0, 801, 0, 1061])
        selection = root / "render-audit/0125-w025-v2/selection_before_ct.json"
        preparation = json.loads((selection.parent / "preparation_receipt.json").read_text())
        chart_evidence = {"selection_sha256": sha(selection), "variant": variant,
                          "prepared_stack31_sha256": preparation["arms"][variant]["stack31_sha256"],
                          "renderer_sha256": preparation["renderer_sha256"]}
        prepared_id = hashlib.sha256(canonical(chart_evidence).encode()).hexdigest()
        return dict(scroll="PHerc0125", source_volume="20250821151825-9.362um-1.2m-113keV-masked",
                    chart_id=f"z14144_w025/prepared-{prepared_id}/{variant}", origin_yx=[roi[0], roi[2]],
                    chart_evidence=chart_evidence,
                    registration_status="PREPARED_CHART_COORDINATES_ONLY")
    if name.startswith("0800_"):
        if "center512" in name:
            origin = [794, 1194]
        elif "left_context96" in name:
            origin = [650, 804]
        elif "left512" in name:
            origin = [650, 900]
        else:
            raise ValueError("Unregistered PHerc0800 crop")
        return dict(scroll="PHerc0800", source_volume="20250521135224-8.640um-1.2m-116keV-masked",
                    chart_id=SEGMENT_C, origin_yx=origin,
                    registration_status="PUBLIC_SURFACE_CROP_LOCAL_CT_REPLAY_ONLY")
    if name.startswith("author841_"):
        return dict(scroll="PHerc0841", source_volume="public-author-prepared-donor",
                    chart_id="841-expanded-layer-03-approved-r000001",
                    origin_yx=[512, 1280 if "mosaic" in name else 1536],
                    registration_status="PREPARED_DONOR_NO_NATIVE_XYZ_REGISTRATION")
    raise ValueError(f"Unregistered campaign job: {name}")


def manifest_for(record, root):
    receipt_path = rebase(record["receipt"], root)
    if sha(receipt_path) != record["receipt_sha256"]:
        raise ValueError("Campaign receipt fingerprint changed")
    r = json.loads(receipt_path.read_text())
    j = r["job"]
    if j["name"] != record["name"]:
        raise ValueError("Campaign name differs from original receipt")
    expected = {"shape": r["input_shape"], "stride": r["stride"], "blend": r["blend"],
                "windows": r["statistics"]["windows"], "sentinel": 1,
                "forwards": r["new_model_forwards_including_sentinel"]}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Campaign counts/recipe differ from original receipt")
    if expected["forwards"] != expected["windows"] + expected["sentinel"]:
        raise ValueError("Historical forward count is inconsistent")
    frame = frame_for(j["name"], j, root)
    arrays = {"input": descriptor(rebase(j["input"], root), root, r["input_sha256"], "LYX")}
    if j.get("source_support"):
        arrays["source_depth"] = descriptor(rebase(j["source_support"], root), root,
                                             r["source_support_sha256"], "LYX")
        support_basis = "EXPLICIT_PREPARED_MASK"
    else:
        if r["source_support_sha256"] is not None:
            raise ValueError("Inconsistent historical implicit support")
        arrays["source_depth"] = dict(derivation="all_true_as_executed",
                                       evidence_sha256=sha(receipt_path),
                                       evidence_path=receipt_path.relative_to(root).as_posix(), dtype="bool",
                                       shape=r["input_shape"], axes="LYX")
        support_basis = "IMPLICIT_PREPARED_PRESENCE"
    for role, original in [("probability", "prob"), ("output_valid", "valid")]:
        item = r["outputs"][original]
        arrays[role] = descriptor(receipt_path.parent / item["file"], root, item["sha256"], "YX")
    for role in ("source_valid", "context_valid"):
        item = r["extra_masks"][role]
        arrays[role] = descriptor(rebase(item["file"], root), root, item["sha256"], "YX")
    samples = r["depth_order"] if isinstance(r["depth_order"], list) else None
    direction = "as_recorded" if samples is None else ("asc" if samples[0] < samples[-1] else "desc")
    stride = r["stride"]
    recipe = dict(model_sha256=r["weights_sha256"], pipeline_sha256=r["pipeline_sha256"],
                  runner_sha256=r["runner_sha256"], model_code_sha256=r["model_source"]["model_py_sha256"],
                  normalization=r["normalization"], precision="fp32", stride=stride, blend=r["blend"],
                  input_window=128, output_window=96, halo=16,
                  tiling_policy="studio_v1_end_anchored", source_domain=frame["source_volume"],
                  origin_yx=frame["origin_yx"], phase_yx=[x % stride for x in frame["origin_yx"]],
                  direction=direction)
    return dict(schema=SCHEMA, name=j["name"], frame=frame, recipe=recipe,
                depth_samples=samples, depth_units="source_plane_index", signed_offsets_um=None,
                depth_unknown_reason=None if samples is not None else r["depth_order"], arrays=arrays,
                support_basis=support_basis,
                cost=dict(map_elapsed_seconds=r["statistics"]["seconds"], cpu_ms=None, gpu_ms=0,
                          io_bytes=None, model_windows=r["statistics"]["windows"], sentinel_forwards=1),
                calibration=dict(identity_sentinel_pass=r["sentinel_bitwise_equal"],
                                 semantic_control_status="NOT_MATCHED",
                                 training_exposure="known_public_possible_train" if j["name"].startswith("author841") else "unknown"),
                scientific_status="known_public_control" if j["name"].startswith("author841") else "unconfirmed")


def execution_key(manifest):
    """Input-only identity. New origin, depth sequence, support or recipe is new work."""
    value = {k: manifest[k] for k in ("schema", "frame", "recipe", "depth_samples", "depth_units", "signed_offsets_um", "support_basis")}
    value["arrays"] = {k: {a: v for a, v in manifest["arrays"][k].items()
                           if a not in ("path", "evidence_path", "evidence_sha256")}
                       for k in ("input", "source_depth")}
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def chart_accounting(manifests, root):
    """Exact pixel union within a named unchanged chart, never physical-sheet area."""
    groups = {}
    for m in manifests:
        f = m["frame"]
        groups.setdefault((f["scroll"], f["source_volume"], f["chart_id"]), []).append(m)
    rows = []
    for (scroll, volume, chart), items in sorted(groups.items()):
        boxes = []
        for m in items:
            y, x = m["frame"]["origin_yx"]
            h, w = m["arrays"]["probability"]["shape"]
            boxes.append((y, x, y + h, x + w))
        y0, x0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        y1, x1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        row = dict(scroll=scroll, chart_id=chart, source_volume=volume, maps=len(items),
                   distinct_input_recipes=len({execution_key(m) for m in items}),
                   physical_area_cm2=None, physical_sheet_identity="NOT_ESTABLISHED")
        for role in ("source_valid", "output_valid", "context_valid"):
            union = np.zeros((y1 - y0, x1 - x0), bool)
            gross = 0
            for m, b in zip(items, boxes):
                a = np.load(root / m["arrays"][role]["path"], allow_pickle=False)
                union[b[0]-y0:b[2]-y0, b[1]-x0:b[3]-x0] |= a
                gross += int(a.sum())
            row[role] = dict(gross_map_pixels=gross, union_chart_pixels=int(union.sum()),
                             repeated_chart_pixels=gross-int(union.sum()))
        rows.append(row)
    return rows


def aligned_comparison(a, b, root):
    """Exact chart-translation comparison; no image registration or interpolation."""
    fa, fb = a["frame"], b["frame"]
    for k in ("scroll", "source_volume", "chart_id"):
        if fa[k] != fb[k]:
            raise ValueError("Cannot align different chart identities")
    ay, ax = fa["origin_yx"]; by, bx = fb["origin_yx"]
    ah, aw = a["arrays"]["probability"]["shape"]; bh, bw = b["arrays"]["probability"]["shape"]
    y0, x0, y1, x1 = max(ay, by), max(ax, bx), min(ay+ah, by+bh), min(ax+aw, bx+bw)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("No aligned overlap")
    sa = (slice(y0-ay, y1-ay), slice(x0-ax, x1-ax)); sb = (slice(y0-by, y1-by), slice(x0-bx, x1-bx))
    def arr(m, role):
        item = m["arrays"][role]
        if item.get("derivation") == "all_true_as_executed":
            return np.ones(item["shape"], dtype=bool)
        return np.load(root / item["path"], mmap_mode="r", allow_pickle=False)
    inputs_equal = np.array_equal(arr(a, "input")[(slice(None), *sa)], arr(b, "input")[(slice(None), *sb)])
    support_equal = np.array_equal(arr(a, "source_depth")[(slice(None), *sa)], arr(b, "source_depth")[(slice(None), *sb)])
    ca, cb = arr(a, "context_valid")[sa], arr(b, "context_valid")[sb]
    common = ca & cb; gained = cb & ~ca; lost = ca & ~cb
    pa, pb = arr(a, "probability")[sa], arr(b, "probability")[sb]
    return dict(a=a["name"], b=b["name"], overlap_pixels=int(pa.size), input_overlap_bitwise_equal=inputs_equal,
                source_support_overlap_bitwise_equal=support_equal,
                common_strict_context_pixels=int(common.sum()), newly_strict_context_pixels=int(gained.sum()),
                lost_strict_context_pixels=int(lost.sum()),
                probabilities_bitwise_equal_on_common_context=bool(np.array_equal(pa[common], pb[common])),
                mean_absolute_difference_on_common_context=float(np.abs(pa[common]-pb[common]).mean()) if common.any() else None,
                changed_recipe_fields=[k for k in sorted(set(a["recipe"]) | set(b["recipe"])) if a["recipe"].get(k) != b["recipe"].get(k)],
                controlled_single_factor_experiment=False,
                implication="Context/tiling robustness only; no independent glyph replication or recall/cost gate")


def verify_declaration(job, manifest, root):
    """An identically named rerun request must still describe the executed inputs."""
    if sha(rebase(job["input"], root)) != manifest["arrays"]["input"]["sha256"]:
        raise ValueError("Conflicting repeated input declaration")
    if job.get("stride", 96) != manifest["recipe"]["stride"]:
        raise ValueError("Conflicting repeated stride declaration")
    expected_depth = manifest["depth_samples"]
    if expected_depth is None:
        expected_depth = manifest["depth_unknown_reason"]
    if job.get("depth_order") != expected_depth:
        raise ValueError("Conflicting repeated depth declaration")
    if frame_for(job["name"], job, root) != manifest["frame"]:
        raise ValueError("Conflicting repeated frame declaration")
    y, x = manifest["frame"]["origin_yx"]
    h, w = manifest["arrays"]["input"]["shape"][1:]
    bounds = [y, y+h, x, x+w]
    for key in ("input_global_y0_y1_x0_x1", "full_canvas_roi_y0_y1_x0_x1"):
        if key in job and job[key] != bounds:
            raise ValueError(f"Conflicting repeated crop extent: {key}")
    support = manifest["arrays"]["source_depth"]
    if support.get("derivation") == "all_true_as_executed":
        if job.get("source_support"):
            raise ValueError("New explicit support differs from historical assumption")
    elif not job.get("source_support") or sha(rebase(job["source_support"], root)) != support["sha256"]:
        raise ValueError("Conflicting repeated support declaration")


def audit(root):
    started = time.perf_counter()
    campaign_path = root / "deliverables/campaign_summary.json"
    campaign = json.loads(campaign_path.read_text())
    names = [r["name"] for r in campaign["records"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate completed-map names require reconciliation")
    if campaign["unique_jobs"] != len(names) or campaign["completed_maps"] != len(names) or campaign["pending_jobs"]:
        raise ValueError("Campaign completion counts do not match the actual records")
    manifests = [manifest_for(r, root) for r in campaign["records"]]
    if len({execution_key(m) for m in manifests}) != len(manifests):
        raise ValueError("Repeated completed input recipes require explicit attempt reconciliation")
    rows = []
    for m in manifests:
        result = validate_native(m, root)
        if not result["valid"]:
            raise ValueError(f"Native evidence failed for {m['name']}: {result['errors']}")
        rows.append(dict(name=m["name"], execution_key=execution_key(m), validation=result,
                         array_hashes={k: v.get("sha256") for k, v in m["arrays"].items()},
                         support_basis=m["support_basis"], cost=m["cost"]))
    lookup = {m["name"]: m for m in manifests}
    pairs = []
    for names in [("0125_raw_desc16_s96", "0125_raw_same_band_reverse16_s96"),
                  ("0125_snapped_desc16_s96", "0125_snapped_same_band_reverse16_s96"),
                  ("0800_C_center512_k23_to_k08_s96", "0800_C_center512_k08_to_k23_s96"),
                  ("0800_C_left512_k23_to_k08_s96", "0800_C_left512_k08_to_k23_s96")]:
        pairs.append(dict(names=list(names), result=assess_pair(lookup[names[0]], lookup[names[1]], root)))
    comparisons = [aligned_comparison(lookup[a], lookup[b], root) for a, b in [
        ("0800_C_left512_k08_to_k23_s96", "0800_C_left_context96_k08_to_k23_s96"),
        ("0125_raw_same_band_reverse16_s96", "0125_raw_reverse_roi_s48"),
        ("0125_snapped_same_band_reverse16_s96", "0125_snapped_reverse_roi_s48"),
        ("author841_positive_s96", "author841_mosaic_positive_s96"),
    ]]
    declarations = []
    for name in campaign["manifests"]:
        declarations.extend(json.loads((root / name).read_text()))
    if set(j["name"] for j in declarations) != set(lookup):
        raise ValueError("Job declarations differ from completed campaign")
    for j in declarations:
        verify_declaration(j, lookup[j["name"]], root)
    total_windows = sum(m["cost"]["model_windows"] for m in manifests)
    if (campaign["prediction_windows"], campaign["identity_sentinels"], campaign["actual_model_forwards"]) != (total_windows, len(manifests), total_windows + len(manifests)):
        raise ValueError("Campaign headline forward counts disagree with verified receipts")
    result = dict(schema="funnel.visual_campaign_audit.v1", status="COMPUTATIONAL_EVIDENCE_VERIFIED",
                  campaign_source_sha256=sha(campaign_path),
                  tool_sha256=sha(__file__), native_validator_sha256=sha(Path(__file__).with_name("native_evidence.py")),
                  python=platform.python_version(), numpy=np.__version__,
                  execution=dict(read_only=True, new_model_forwards=0, new_gpu_work=False,
                                 audit_elapsed_seconds=round(time.perf_counter()-started, 6)),
                  job_declarations=len(declarations), unique_completed_maps=len(manifests),
                  repeated_declarations=len(declarations)-len(manifests),
                  historical_prediction_windows=total_windows,
                  historical_identity_sentinels=len(manifests),
                  historical_model_forwards=total_windows+len(manifests),
                  sum_historical_map_elapsed_seconds=round(sum(m["cost"]["map_elapsed_seconds"] for m in manifests), 2),
                  measured_cpu_ms=None, measured_gpu_savings=None,
                  maps=rows, exact_reverse_pairs=pairs,
                  maps_without_exact_same_configuration_reverse=sorted(set(lookup)-{n for p in pairs for n in p["names"]}),
                  registered_chart_pixel_accounting=chart_accounting(manifests, root),
                  aligned_robustness_comparisons=comparisons,
                  scientific_status="NO_CONFIRMED_NEW_READING",
                  production_promotion=False,
                  remaining_gates=["matched semantic calibration", "independent physical sheet/metric evidence",
                                   "frozen native coarse/fine recall-and-cost benchmark", "actual CPU/IO/human metering"],
                  limitations=["Pixel unions only within the declared unchanged chart; raw and snapped charts kept separate",
                               "Known public PHerc0841 control is not an independent holdout",
                               "CPU map durations are elapsed intervals, not process CPU time or wall-time sum",
                               "Repeated declarations are not proof of computation saved",
                               "Imported results remain diagnostic; no queue completion or blank verdict"])
    return manifests, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="New compact JSON receipt, no absolute source paths")
    ap.add_argument("--native-manifests", type=Path, help="Optional new directory of local native descriptors")
    args = ap.parse_args()
    manifests, result = audit(args.campaign_root.resolve())
    if args.native_manifests:
        args.native_manifests.mkdir(parents=True, exist_ok=False)
        for m in manifests:
            (args.native_manifests / (m["name"] + ".json")).write_text(json.dumps(m, indent=2, allow_nan=False)+"\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "unique_completed_maps", "repeated_declarations", "historical_model_forwards", "production_promotion")}))


if __name__ == "__main__":
    main()
