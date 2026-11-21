"""Measure complete native Villa CLI processes on the public W058 fixture on Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import tifffile


def save_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def measure(checkout, surface, weights, output):
    environment = os.environ.copy()
    sources = [str(checkout / "vesuvius/src"), environment.get("PYTHONPATH")]
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, sources))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    command = [
        sys.executable, "-m", "vesuvius.ink_detection.inference.infer",
        str(surface), str(weights), str(output), "--roi", "3072", "3200", "1920", "2048",
        "--direction", "reverse", "--resolution", "0", "--device", "cpu",
        "--no-compile", "--num-workers", "0", "--batch-size", "1",
        "--blend-mode", "hann", "--overlap", "0.5",
    ]
    log = output.with_suffix(".log")
    with log.open("wb") as stream:
        start = perf_counter()
        with subprocess.Popen(command, env=environment, cwd=output.parent,
                              stdout=stream, stderr=subprocess.STDOUT) as process:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        elapsed = perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"Villa exited {process.returncode}. See {log}.")
    image = tifffile.imread(output)
    if image.shape != (128, 128) or image.dtype != np.uint8:
        raise ValueError(f"Unexpected TIFF geometry in {output}")
    return {
        "wall_s": elapsed, "user_s": usage.ru_utime, "system_s": usage.ru_stime,
        "cpu_s": usage.ru_utime + usage.ru_stime, "peak_rss_mib": usage.ru_maxrss / 1024,
        "decoded_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "sidecar": json.loads(output.with_suffix(".tif.roi.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "patched", "surface", "weights", "output"):
        parser.add_argument(name, type=Path)
    paths = {key: path.resolve() for key, path in vars(parser.parse_args()).items()}
    if paths["baseline"] == paths["patched"]:
        parser.error("supply distinct baseline and patched Villa checkouts")
    dest = paths["output"]
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        parser.error("use an empty trial directory to keep runs separate")
    order = [("baseline", "patched"), ("patched", "baseline")] * 2
    with paths["weights"].open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    plan = {
        "roi_yx": [3072, 3200, 1920, 2048], "source": "PHerc.1447 W058, 21 native layers, local Zarr",
        "order": order, "weight_sha256": weight_hash,
        "scope": "Fresh complete command, shared OS cache, one math thread, CPU",
        "inference": {"batch": 1, "workers": 0, "tile_overlap": 0.5, "blend": "hann", "direction": "reverse"},
    }
    save_json(dest / "plan.json", plan)
    trials = []
    for pair_number, versions in enumerate(order, 1):
        for version in versions:
            label = f"pair{pair_number}-{version}"
            trial = measure(paths[version], paths["surface"], paths["weights"], dest / f"{label}.tif")
            if trials and any(trial[key] != trials[0][key] for key in ("decoded_sha256", "sidecar")):
                raise ValueError(f"Prediction or geometry changed in {label}")
            trials.append({"pair": pair_number, "version": version, **trial})
            save_json(dest / "trials.json", trials)
            print(label, trial["wall_s"], trial["cpu_s"], trial["peak_rss_mib"], flush=True)

    metrics = ("wall_s", "cpu_s", "peak_rss_mib")
    by_version = {name: [row for row in trials if row["version"] == name] for name in order[0]}
    medians = {name: {metric: statistics.median(row[metric] for row in rows) for metric in metrics}
               for name, rows in by_version.items()}
    gains = [{metric: before[metric] - after[metric] for metric in metrics}
             for before, after in zip(by_version["baseline"], by_version["patched"])]
    summary = {
        "plan": plan, "decoded_sha256": trials[0]["decoded_sha256"], "same_pixels": 128 * 128,
        "repeats_per_version": len(order), "versions": medians, "paired_gains": gains,
    }
    save_json(dest / "summary.json", summary)


if __name__ == "__main__":
    main()
