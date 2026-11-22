#!/usr/bin/env python3
"""A fixed, bounded optical-height pilot; never a CT ink-detection claim.

Read profilometer_protocol.json before running. Publish its hash before any image
access. --prepare-only downloads and verifies bytes without decoding images.
--run decodes the verified, fixed subset and fits the five registered heads.
Both modes require the externally frozen protocol hash and keep artifacts
outside the repository. --self-check uses generated arrays only, with no fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
import urllib.parse
import urllib.request
import warnings

# This experiment has one CPU thread, independently of the caller's defaults.
for _thread_env in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

CODE = Path(__file__).resolve()
REPO = CODE.parents[2]
DEFAULT_PROTOCOL = CODE.with_name("profilometer_protocol.json")
FEATURE_NAMES = [
    "height", "height_minus_gaussian8", "std_gaussian2", "std_gaussian8",
    "gradient_magnitude_sigma1", "absolute_gaussian_laplacian_sigma1",
    "log_std2_over_std8", "tensor_anisotropy_sigma1_gaussian4",
]


class PilotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def protocol_read(path: Path, expected_hash: str | None) -> tuple[dict, str]:
    actual = sha256_file(path)
    if expected_hash is not None and actual != expected_hash.lower():
        raise PilotError("Protocol hash differs from the externally frozen hash.")
    protocol = json.loads(path.read_text())
    if protocol["schema"] != "vesuvius.profilometer_pilot.v1":
        raise PilotError("Unsupported protocol schema.")
    if protocol["features"]["names"] != FEATURE_NAMES:
        raise PilotError("Feature recipe differs from this fixed implementation.")
    if protocol["degradation"]["factors"] != [1, 2, 4]:
        raise PilotError("This pilot has exactly three input conditions.")
    if protocol["models"]["C"] != 1 or protocol["models"]["max_iter"] != 500:
        raise PilotError("Model parameters differ from the fixed implementation.")
    train = protocol["split"]["train_case_ids"]
    heldout = protocol["split"]["heldout_case_ids"]
    if len(train) != 2 or len(heldout) != 2 or set(train) & set(heldout):
        raise PilotError("The registered subset must have disjoint two-case roles.")
    records = protocol["records"]
    if len(records) != 4 or {r["case_id"] for r in records} != set(train + heldout):
        raise PilotError("Source record IDs do not match the registered split.")
    if sum(f["bytes"] for r in records for f in r["files"]) != protocol["limits"]["expected_image_bytes"]:
        raise PilotError("Registered source byte total does not match its records.")
    for record in records:
        expected_role = "train" if record["case_id"] in train else "val"
        if record["fold5_role"] != expected_role or len(record["files"]) != 2:
            raise PilotError("Record role or file count mismatch.")
    return protocol, actual


def private_workdir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise PilotError("Image, model and prediction artifacts must be outside the repository.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def source_destination(work: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise PilotError("Unsafe source path in protocol.")
    return work / "source" / rel


def verify_file(path: Path, record: dict, hash_key: str) -> str:
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise PilotError(f"Source size mismatch: {record['path']}")
    digest = sha256_file(path)
    if digest != record[hash_key]:
        raise PilotError(f"Source checksum mismatch: {record['path']}")
    return digest


def download_verified(work: Path, protocol: dict, record: dict, hash_key: str) -> dict:
    destination = source_destination(work, record["path"])
    base = protocol["source"]["url"]
    revision = protocol["source"]["revision"]
    url = f"{base}/resolve/{revision}/{urllib.parse.quote(record['path'], safe='/')}"
    start = time.monotonic()
    reused = destination.is_file()
    if not reused:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        byte_count = 0
        request = urllib.request.Request(url, headers={"User-Agent": "vesuvius-bounded-profilometer-pilot/1"})
        # Public source only; no token, cookie, signed redirect URL or credential
        # is stored in the receipt. There are no automatic retries.
        with urllib.request.urlopen(request, timeout=protocol["limits"]["request_timeout_seconds"]) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > record["bytes"]:
                        raise PilotError(f"Source exceeds registered size: {record['path']}")
                    digest.update(chunk)
                    output.write(chunk)
        if byte_count != record["bytes"] or digest.hexdigest() != record[hash_key]:
            raise PilotError(f"Downloaded source failed integrity check: {record['path']}")
        temporary.replace(destination)
    digest_text = verify_file(destination, record, hash_key)
    return {"path": record["path"], "canonical_source_url": url,
            "bytes": record["bytes"], "sha256": digest_text,
            "reused_verified_file": reused, "wall_seconds": time.monotonic() - start}


def verify_official_split(work: Path, protocol: dict) -> None:
    split_path = source_destination(work, "folds/splits_final.json")
    if sha256_file(split_path) != protocol["split"]["split_source_sha256"]:
        raise PilotError("Official split hash mismatch.")
    fold = json.loads(split_path.read_text())[protocol["split"]["official_fold_index"]]
    for role, official_role in (("train_case_ids", "train"), ("heldout_case_ids", "val")):
        if not set(protocol["split"][role]).issubset(fold[official_role]):
            raise PilotError("Registered subset does not belong to the official fold.")
    # The stronger unit split is by physical specimen, not randomly held pixels.
    train_specimens = {case.split("_")[0] for case in protocol["split"]["train_case_ids"]}
    test_specimens = {case.split("_")[0] for case in protocol["split"]["heldout_case_ids"]}
    if train_specimens & test_specimens:
        raise PilotError("A specimen occurs in both roles.")


def prepare(work: Path, protocol: dict, protocol_hash: str) -> None:
    start = time.monotonic()
    cpu_start = time.process_time()
    receipt = {"schema": "vesuvius.profilometer_download_receipt.v1", "status": "preparing",
               "protocol_sha256": protocol_hash, "code_sha256": sha256_file(CODE),
               "image_decoding_performed": False, "model_fitting_performed": False,
               "records": []}
    receipt_path = work / "prepare_receipt.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get("protocol_sha256") != protocol_hash or old.get("code_sha256") != receipt["code_sha256"]:
            raise PilotError("Existing preparation belongs to different frozen source or protocol.")
    json_write(receipt_path, receipt)
    try:
        for record in protocol["metadata_files"]:
            receipt["records"].append(download_verified(work, protocol, record, "sha256"))
        verify_official_split(work, protocol)
        for case in protocol["records"]:
            for record in case["files"]:
                receipt["records"].append(download_verified(work, protocol, record, "sha256_lfs"))
            print(json.dumps({"stage": "verified_bytes", "case_id": case["case_id"]}), flush=True)
        receipt["status"] = "prepared_verified_bytes_no_decoding"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error_type"] = type(exc).__name__
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - start
        receipt["cpu_seconds"] = time.process_time() - cpu_start
        json_write(receipt_path, receipt)


def block_mean(array, factor: int):
    import numpy as np
    h, w = array.shape
    cropped = array[:h - h % factor, :w - w % factor]
    if min(cropped.shape) == 0:
        raise PilotError("Image is smaller than a required block.")
    return cropped.reshape(cropped.shape[0] // factor, factor,
                           cropped.shape[1] // factor, factor).mean(axis=(1, 3), dtype=np.float64)


def degrade(height, factor: int):
    import numpy as np
    from PIL import Image
    if factor == 1:
        return height
    h, w = height.shape
    padded = np.pad(height, ((0, (-h) % factor), (0, (-w) % factor)), mode="edge")
    coarse = block_mean(padded, factor).astype(np.float32)
    resized = Image.fromarray(coarse).resize((padded.shape[1], padded.shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)[:h, :w]


def features(height):
    import numpy as np
    from scipy import ndimage as ndi
    height = np.asarray(height, dtype=np.float64)
    opts = {"mode": "reflect", "truncate": 4.0}
    smooth1 = ndi.gaussian_filter(height, 1, **opts)
    smooth2 = ndi.gaussian_filter(height, 2, **opts)
    smooth8 = ndi.gaussian_filter(height, 8, **opts)
    squared = height * height
    std2 = np.sqrt(np.maximum(ndi.gaussian_filter(squared, 2, **opts) - smooth2 * smooth2, 0))
    std8 = np.sqrt(np.maximum(ndi.gaussian_filter(squared, 8, **opts) - smooth8 * smooth8, 0))
    gy, gx = np.gradient(smooth1)
    jxx = ndi.gaussian_filter(gx * gx, 4, **opts)
    jyy = ndi.gaussian_filter(gy * gy, 4, **opts)
    jxy = ndi.gaussian_filter(gx * gy, 4, **opts)
    anisotropy = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-12)
    result = np.stack([
        height, height - smooth8, std2, std8, np.hypot(gx, gy),
        np.abs(ndi.gaussian_laplace(height, 1, **opts)),
        np.log((std2 + 1e-4) / (std8 + 1e-4)), anisotropy,
    ], axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise PilotError("A fixed feature is nonfinite.")
    return result


def load_case(work: Path, protocol: dict, record: dict):
    import numpy as np
    from PIL import Image
    height_record = next(f for f in record["files"] if f["path"].endswith("_z.tif"))
    label_record = next(f for f in record["files"] if f["path"].endswith("_label.png"))
    for entry in record["files"]:
        verify_file(source_destination(work, entry["path"]), entry, "sha256_lfs")
    with Image.open(source_destination(work, height_record["path"])) as image:
        raw_height = np.asarray(image)
    with Image.open(source_destination(work, label_record["path"])) as image:
        binary = np.asarray(image.convert("L")) != 0
    native_shape = (record["height_px"], record["width_px"])
    if raw_height.dtype != np.uint16 or raw_height.shape != native_shape or binary.shape != native_shape:
        raise PilotError(f"Unexpected dtype or native shape for {record['case_id']}.")
    factor = protocol["working_grid"]["native_blockmean_factor"]
    height = block_mean(raw_height, factor) / 65535.0
    labels = block_mean(binary, factor) >= 0.5
    border = protocol["working_grid"]["boundary_exclusion_working_pixels"]
    mask = np.zeros(labels.shape, dtype=bool)
    if min(labels.shape) <= 2 * border:
        raise PilotError("Working grid has no interior after registered exclusion.")
    mask[border:-border, border:-border] = True
    counts = np.bincount(labels[mask].astype(np.uint8), minlength=2)
    if not np.all(counts > 0):
        raise PilotError(f"Both label classes are required in {record['case_id']}.")
    summary = {"case_id": record["case_id"], "native_shape_hw": list(native_shape),
               "working_shape_hw": list(height.shape), "scored_pixels": int(mask.sum()),
               "scored_background_pixels": int(counts[0]), "scored_ink_pixels": int(counts[1]),
               "ink_prevalence": float(counts[1] / counts.sum()),
               "score_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
               "working_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()}
    return height, labels, mask, summary


def scored_metrics(labels, scores, top_fraction: float) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise PilotError("A prediction is nonfinite.")
    k = max(1, math.ceil(top_fraction * len(labels)))
    # Stable ties preserve the row-major order of the fixed scoring mask.
    top = np.argsort(-scores, kind="stable")[:k]
    positives = int(labels.sum())
    selected_positive = int(labels[top].sum())
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "ink_prevalence": float(labels.mean()), "scored_pixels": len(labels),
            "ink_pixels": positives, "top_1_percent_count": k,
            "precision_at_top_1_percent": selected_positive / k,
            "recall_at_top_1_percent": selected_positive / positives}


def artifact_record(path: Path, work: Path) -> dict:
    return {"path_relative_to_private_workdir": str(path.relative_to(work)),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def cpu_deadline(seconds: int):
    import resource
    start = time.process_time()

    def over_limit(signum=None, frame=None):
        raise PilotError("Registered whole-run CPU budget exhausted; no retry.")

    signal.signal(signal.SIGXCPU, over_limit)
    current = math.ceil(time.process_time())
    _, existing_hard = resource.getrlimit(resource.RLIMIT_CPU)
    hard = current + seconds + 5
    if existing_hard != resource.RLIM_INFINITY:
        hard = min(hard, existing_hard)
    resource.setrlimit(resource.RLIMIT_CPU, (min(current + seconds, hard), hard))

    def check():
        if time.process_time() - start > seconds:
            over_limit()
    return check


def execute(work: Path, protocol: dict, protocol_hash: str) -> None:
    start = time.monotonic()
    cpu_start = time.process_time()
    check_budget = cpu_deadline(protocol["limits"]["whole_run_cpu_seconds"])
    receipt = json.loads((work / "prepare_receipt.json").read_text())
    code_hash = sha256_file(CODE)
    if receipt["status"] != "prepared_verified_bytes_no_decoding":
        raise PilotError("Complete byte-only preparation is required before a run.")
    if receipt["protocol_sha256"] != protocol_hash or receipt["code_sha256"] != code_hash:
        raise PilotError("Code or protocol changed after preparation.")
    for metadata in protocol["metadata_files"]:
        verify_file(source_destination(work, metadata["path"]), metadata, "sha256")
    verify_official_split(work, protocol)
    if (work / "run_receipt.json").exists():
        raise PilotError("This work directory already contains a run; registered pilot has no retries.")

    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits, threadpool_info

    audit = {"schema": "vesuvius.profilometer_run.v1", "status": "running",
             "protocol_sha256": protocol_hash, "code_sha256": code_hash,
             "source_revision": protocol["source"]["revision"],
             "source_receipt_sha256": sha256_file(work / "prepare_receipt.json"),
             "split_source_sha256": protocol["split"]["split_source_sha256"],
             "python": platform.python_version(),
             "versions": {name: importlib.metadata.version(name) for name in
                          ["numpy", "scipy", "scikit-learn", "Pillow", "joblib", "threadpoolctl"]},
             "cases": [], "models": [], "artifacts": [], "results": [],
             "limitations": protocol["limitations"]}
    json_write(work / "run_receipt.json", audit)
    records = {r["case_id"]: r for r in protocol["records"]}
    try:
        with threadpool_limits(limits=1):
            audit["threadpools"] = [{"internal_api": t["internal_api"], "num_threads": t["num_threads"]}
                                    for t in threadpool_info()]
            training_x = {factor: [] for factor in protocol["degradation"]["factors"]}
            training_y = []
            generator = np.random.default_rng(protocol["sampling"]["seed"])
            for case_id in protocol["split"]["train_case_ids"]:
                height, labels, mask, summary = load_case(work, protocol, records[case_id])
                summary["role"] = "train"
                audit["cases"].append(summary)
                chosen = []
                eligible_by_class = {}
                for label in protocol["sampling"]["classes"]:
                    eligible_by_class[label] = np.flatnonzero((mask & (labels == label)).ravel())
                count = min(protocol["sampling"]["max_pixels_per_class_per_training_case"],
                            *(len(indices) for indices in eligible_by_class.values()))
                for label in protocol["sampling"]["classes"]:
                    eligible = eligible_by_class[label]
                    chosen.append(generator.choice(eligible, size=count, replace=False))
                indices = generator.permutation(np.concatenate(chosen)).astype(np.int64)
                index_path = work / "training_indices" / f"{case_id}.npy"
                index_path.parent.mkdir(exist_ok=True)
                np.save(index_path, indices, allow_pickle=False)
                audit["artifacts"].append(artifact_record(index_path, work))
                summary["sampled_class_counts"] = np.bincount(labels.ravel()[indices].astype(np.uint8), minlength=2).tolist()
                training_y.append(labels.ravel()[indices].astype(np.uint8))
                for factor in protocol["degradation"]["factors"]:
                    matrix = features(degrade(height, factor))
                    training_x[factor].append(matrix.reshape(-1, len(FEATURE_NAMES))[indices].copy())
                    del matrix
                    check_budget()
                print(json.dumps({"stage": "fixed_training_features", "case_id": case_id}), flush=True)
            train_labels = np.concatenate(training_y)
            training_x = {factor: np.concatenate(parts) for factor, parts in training_x.items()}
            models = {}
            for spec in protocol["models"]["fits"]:
                check_budget()
                model_start = time.monotonic()
                estimator = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, max_iter=500, solver="lbfgs", class_weight=None,
                    penalty="l2", tol=1e-4, random_state=protocol["sampling"]["seed"],
                ))
                x_fit = training_x[spec["train_factor"]][:, spec["columns"]]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(x_fit, train_labels)
                model_path = work / "models" / f"{spec['id']}.joblib"
                model_path.parent.mkdir(exist_ok=True)
                joblib.dump(estimator, model_path, compress=3)
                audit["models"].append({"id": spec["id"], "train_factor": spec["train_factor"],
                    "columns": spec["columns"], "n_training_pixels": len(train_labels),
                    "n_iter": estimator[-1].n_iter_.tolist(), "wall_seconds": time.monotonic() - model_start,
                    "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
                    "artifact": artifact_record(model_path, work)})
                models[spec["id"]] = estimator
                print(json.dumps({"stage": "fixed_model_fitted", "model": spec["id"]}), flush=True)
            del training_x, training_y, train_labels

            # Held-out image/label decoding occurs only after all five heads are frozen.
            for case_id in protocol["split"]["heldout_case_ids"]:
                height, labels, mask, summary = load_case(work, protocol, records[case_id])
                summary["role"] = "heldout"
                audit["cases"].append(summary)
                case_dir = work / "predictions" / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                for name, array in (("scoring_mask", mask), ("working_labels", labels)):
                    path = case_dir / f"{name}.npy"
                    np.save(path, array, allow_pickle=False)
                    audit["artifacts"].append(artifact_record(path, work))
                mask_indices = np.flatnonzero(mask.ravel())
                truth = labels.ravel()[mask_indices]
                for factor in protocol["degradation"]["factors"]:
                    matrix = features(degrade(height, factor)).reshape(-1, len(FEATURE_NAMES))
                    for spec in protocol["models"]["fits"]:
                        if factor not in spec["test_factors"]:
                            continue
                        check_budget()
                        score = np.empty(len(mask_indices), dtype=np.float32)
                        for offset in range(0, len(mask_indices), 65536):
                            positions = mask_indices[offset:offset + 65536]
                            x_eval = matrix[positions][:, spec["columns"]]
                            score[offset:offset + len(positions)] = models[spec["id"]].predict_proba(x_eval)[:, 1]
                        row = {"case_id": case_id, "model": spec["id"],
                               "training_degradation_factor": spec["train_factor"],
                               "test_degradation_factor": factor, "native_to_working_factor": 4,
                               **scored_metrics(truth, score, protocol["evaluation"]["top_fraction"])}
                        full_map = np.full(mask.shape, np.nan, dtype=np.float32)
                        full_map.ravel()[mask_indices] = score
                        prediction_path = case_dir / f"{spec['id']}__test_f{factor}.npy"
                        np.save(prediction_path, full_map, allow_pickle=False)
                        audit["artifacts"].append(artifact_record(prediction_path, work))
                        audit["results"].append(row)
                    del matrix
                    print(json.dumps({"stage": "heldout_condition_scored", "case_id": case_id,
                                      "degradation_factor": factor}), flush=True)
                json_write(work / "run_receipt.json", audit)
            if len(audit["results"]) != 22:
                raise PilotError("Not all registered result rows were produced.")
            audit["status"] = "completed_fixed_protocol"
    except Exception as exc:
        audit["status"] = "aborted_no_parameter_retry"
        audit["error_type"] = type(exc).__name__
        raise
    finally:
        audit["wall_seconds"] = time.monotonic() - start
        audit["cpu_seconds"] = time.process_time() - cpu_start
        json_write(work / "run_receipt.json", audit)
    sanitized = {"status": audit["status"], "protocol_sha256": protocol_hash,
                 "code_sha256": code_hash, "source_revision": audit["source_revision"],
                 "split_source_sha256": audit["split_source_sha256"],
                 "experiment": protocol["objective"], "split_design": protocol["split"]["design"],
                 "source_url": protocol["source"]["url"], "paper_url": protocol["source"]["paper_url"],
                 "feature_names": FEATURE_NAMES, "results": audit["results"],
                 "models": [{k: m[k] for k in ("id", "train_factor", "columns", "n_training_pixels", "n_iter", "warnings")}
                            for m in audit["models"]],
                 "wall_seconds": audit["wall_seconds"], "cpu_seconds": audit["cpu_seconds"],
                 "limitations": protocol["limitations"]}
    json_write(work / "sanitized_results.json", sanitized)
    with (work / "sanitized_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit["results"][0]))
        writer.writeheader()
        writer.writerows(audit["results"])
    print(json.dumps({"status": audit["status"], "result_rows": len(audit["results"]),
                      "cpu_seconds": audit["cpu_seconds"], "wall_seconds": audit["wall_seconds"]}), flush=True)


def self_check() -> None:
    """Generated numerical invariants only: no source images and no estimator fit."""
    import numpy as np
    array = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)
    small = block_mean(array, 4)
    assert small.shape == (2, 3)
    assert small[0, 0] == np.mean(array[:4, :4])
    constant = np.full((101, 103), 0.5, dtype=np.float64)
    for factor in (1, 2, 4):
        out = degrade(constant, factor)
        assert out.shape == constant.shape
        assert np.max(np.abs(out - 0.5)) < 1e-7
    yy, xx = np.mgrid[:101, :103]
    field = 0.5 + 0.1 * np.sin(xx / 13) + 0.01 * yy / 101
    feat = features(field)
    assert feat.shape == (101, 103, 8) and feat.dtype == np.float32
    assert np.isfinite(feat).all()
    assert np.min(feat[..., 7]) >= 0 and np.max(feat[..., 7]) <= 1.000001
    truth = np.array([1, 0, 1, 0], dtype=np.uint8)
    equal = scored_metrics(truth, np.ones(4, dtype=np.float32), 0.25)
    assert equal["precision_at_top_1_percent"] == 1.0
    assert equal["recall_at_top_1_percent"] == 0.5
    print(json.dumps({"self_check": "passed", "source_images_accessed": False, "model_fits": 0}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--frozen-protocol-sha256", help="Hash published before image access.")
    parser.add_argument("--work-dir", type=Path, help="Private directory outside this repository.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    protocol, protocol_hash = protocol_read(args.protocol, args.frozen_protocol_sha256)
    if args.self_check:
        self_check()
        return 0
    if not args.frozen_protocol_sha256 or args.work_dir is None:
        parser.error("Download/run requires --frozen-protocol-sha256 and --work-dir.")
    work = private_workdir(args.work_dir)
    if args.prepare_only:
        prepare(work, protocol, protocol_hash)
    else:
        execute(work, protocol, protocol_hash)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Do not print redirected download URLs or any external credential data.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "detail": str(error) if isinstance(error, PilotError) else "See private audit for execution stage."}),
              file=sys.stderr)
        raise SystemExit(1)
