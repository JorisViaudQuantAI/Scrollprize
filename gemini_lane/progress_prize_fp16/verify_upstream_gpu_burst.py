"""One short GPU burst for the parity checks still open in the PR drafts, run on the patched upstream code.

Stages (each one optional, both by default):

* studio: the patched release CLI (`infer_release.py`, which calls `hf/inference.py`) on the public 169 x 169
  PHerc. 0306B smoke crop (mesh layer 12 of 24, planes 20 down to 5 in the release's `desc` order, stride 48
  Hann).
  CPU fp32 is the reference; CUDA `--precision fp16` and `--precision auto` are compared with it, plus
  `--precision bf16` on compute capability >= 8.0 to identify what `auto` resolved to.
* hecate: the patched `hecate.py`, fp16 against fp32 on CUDA, for each checkpoint given an input
  (`hecate_2.4um.pth`, `hecate_9.6um.pth`), with `--output` (2D PNG) and `--output-3d` (Zarr) in the same run,
  without and with `--reverse`.

Laptop safety defaults, also fine on a Colab T4: `--vram-fraction 0.35` (applied with
`torch.cuda.set_per_process_memory_fraction` inside every CLI process, including the one `infer_release.py`
spawns), `--batch 1`, `--max-gpu-temp-c 68`. A watchdog polls `nvidia-smi` every second, stops the burst when the
GPU passes the temperature limit or the VRAM grows past the cap plus a context allowance, and waits for the GPU to
cool down before each run.

Everything (clones, weights, inputs, outputs) lives in a temporary directory that is deleted at the end. The JSON
receipt holds file names, hashes, metrics and GPU readings, no local paths. `--dry-run` prints the plan only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - only the child processes need torch
    torch = None

try:
    import zarr
except ImportError:  # pragma: no cover - only needed to read Hecate --output-3d
    zarr = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only needed to read Hecate --output
    Image = None

HERE = Path(__file__).resolve().parent
STUDIO_PATCH = HERE / "0001-studio-upstream-fp16.patch"
HECATE_PATCHES = {"main": HERE / "0002-hecate-upstream-fp16.patch",
                  "pr3": HERE / "0002b-hecate-upstream-fp16-after-pr3.patch"}
STUDIO_REPO = "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt.git"
HECATE_REPO = "https://huggingface.co/scrollprize/hecate"
STUDIO_WEIGHTS_URL = ("https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/"
                      "studio-softloss-20261023/studio-softloss-20261023-ema.safetensors")
STUDIO_WEIGHTS_SHA = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"
CROP_URL = ("https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/"
            "f435998b55269762942a979052044309b29650ac/docker/build/smoke_crop.npz")
CROP_SHA = "4a81e193e253e8b6234eaa4c1b274e33c08b214183951ed8e573c9ae3f8f639d"
CROP_MESH_INDEX = 12
HECATE_CHECKPOINTS = {"2.4": "hecate_2.4um.pth", "9.6": "hecate_9.6um.pth"}

STUDIO_FP16_MAX_ABS = 0.01        # measured 0.0021-0.0031 on RTX 4060 and L4
STUDIO_FP16_MIN_R = 0.9999
HECATE_MAX_LEVEL_DIFF = 3         # uint8 levels; measured 1 on a T4 (9.6 um, 2D)
HECATE_MAX_MEAN_LEVEL_DIFF = 0.25
VRAM_CONTEXT_ALLOWANCE_MIB = 1024  # CUDA context and cuDNN workspaces are outside the allocator cap


# ------------------------------------------------------------------ arguments
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="JSON receipt (must not exist)")
    p.add_argument("--stages", default="studio,hecate", help="comma list of studio, hecate")
    p.add_argument("--vram-fraction", type=float, default=0.35)
    p.add_argument("--batch", type=int, default=1, help="--batch-size given to both CLIs")
    p.add_argument("--max-gpu-temp-c", type=float, default=68.0)
    p.add_argument("--cooldown-margin-c", type=float, default=5.0,
                   help="before each run, wait until the GPU is this far below --max-gpu-temp-c")
    p.add_argument("--cooldown-timeout-s", type=float, default=600.0)
    p.add_argument("--studio-stride", type=int, default=48)
    p.add_argument("--hecate-patch", choices=tuple(HECATE_PATCHES), default="main",
                   help="main: 0002 on main; pr3: 0002b on refs/pr/3 (--resample)")
    p.add_argument("--hecate-input-2p4", type=Path, help="uint8 Z,Y,X .npy already rendered at 2.4 um")
    p.add_argument("--hecate-input-9p6", type=Path, help="uint8 Z,Y,X .npy already rendered at 9.6 um")
    p.add_argument("--hecate-crop-yx", type=int, default=512, help="central Y,X crop of each input; 0 = whole")
    p.add_argument("--dry-run", action="store_true", help="print the plan; no clone, download or GPU use")
    return p


def validate(args) -> list[str]:
    errors = []
    if not 0.05 <= args.vram_fraction <= 0.9:
        errors.append("--vram-fraction must be in [0.05, 0.9]")
    if not 1 <= args.batch <= 16:
        errors.append("--batch must be in [1, 16]")
    if not 40 <= args.max_gpu_temp_c <= 85:
        errors.append("--max-gpu-temp-c must be in [40, 85]")
    if not 0 <= args.cooldown_margin_c < args.max_gpu_temp_c - 30:
        errors.append("--cooldown-margin-c out of range")
    stages = set(args.stages.split(","))
    if not stages or not stages <= {"studio", "hecate"}:
        errors.append("--stages takes studio and/or hecate")
    if "hecate" in stages and not hecate_inputs(args):
        errors.append("the hecate stage needs --hecate-input-2p4 and/or --hecate-input-9p6")
    for path in hecate_inputs(args).values():
        if not path.is_file():
            errors.append(f"missing Hecate input: {path.name}")
    if args.output.exists():
        errors.append("--output already exists; choose a new file")
    return errors


def hecate_inputs(args) -> dict[str, Path]:
    pairs = {"2.4": args.hecate_input_2p4, "9.6": args.hecate_input_9p6}
    return {um: path for um, path in pairs.items() if path is not None}


# ------------------------------------------------------------------ plan
@dataclass
class Run:
    tag: str
    stage: str
    script: str            # relative to the stage's checkout
    argv: list[str]
    gpu: bool
    outputs: dict = field(default_factory=dict)


def studio_runs(batch: int, stride: int, compute_capability: float | None) -> list[Run]:
    def run(tag, device, precision, b):
        out = f"studio_{tag}.npy"
        return Run(tag=f"studio_{tag}", stage="studio", script="infer_release.py", gpu=device == "cuda",
                   argv=["--input", "{work}/studio_stack.npy", "--output", "{work}/" + out,
                         "--checkpoint", "{work}/studio-softloss-20261023-ema.safetensors",
                         "--device", device, "--batch-size", str(b), "--precision", precision,
                         "--stride", str(stride)],
                   outputs={"map": out})
    runs = [run("cpu_fp32", "cpu", "fp32", 1), run("cuda_fp16", "cuda", "fp16", batch),
            run("cuda_auto", "cuda", "auto", batch)]
    if compute_capability is None or compute_capability >= 8.0:
        runs.append(run("cuda_bf16", "cuda", "bf16", batch))
    return runs


def hecate_runs(checkpoints: list[str], batch: int) -> list[Run]:
    runs = []
    for um in checkpoints:
        for reverse in (False, True):
            for precision in ("fp32", "fp16"):
                tag = f"hecate_{um.replace('.', 'p')}_{'rev' if reverse else 'fwd'}_{precision}"
                argv = ["--checkpoint", "{work}/" + HECATE_CHECKPOINTS[um],
                        "--input", "{work}/hecate_input_" + um.replace(".", "p") + ".npy",
                        "--spacing-um", um, "--device", "cuda", "--batch-size", str(batch),
                        "--precision", precision,
                        "--output", "{work}/" + tag + ".png", "--output-3d", "{work}/" + tag + ".zarr"]
                if reverse:
                    argv.append("--reverse")
                runs.append(Run(tag=tag, stage="hecate", script="hecate.py", argv=argv, gpu=True,
                                outputs={"png": tag + ".png", "zarr": tag + ".zarr"}))
    return runs


# ------------------------------------------------------------------ GPU readings and guards
@dataclass
class GpuReading:
    name: str
    temp_c: float
    used_mib: float
    total_mib: float
    compute_capability: float | None


def parse_nvidia_smi(text: str) -> GpuReading:
    """First GPU of `nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,compute_cap
    --format=csv,noheader,nounits`; compute_cap may be missing on old drivers."""
    fields = [f.strip() for f in text.strip().splitlines()[0].split(",")]
    if len(fields) < 4:
        raise ValueError(f"unexpected nvidia-smi output: {text!r}")
    try:
        cc = float(fields[4]) if len(fields) > 4 else None
    except ValueError:
        cc = None
    return GpuReading(fields[0], float(fields[1]), float(fields[2]), float(fields[3]), cc)


def query_gpu() -> GpuReading:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,compute_cap",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
                             check=True)
    return parse_nvidia_smi(out.stdout)


def guard_violation(reading: GpuReading, baseline_used_mib: float, max_temp_c: float,
                    vram_fraction: float) -> str | None:
    if reading.temp_c > max_temp_c:
        return f"GPU at {reading.temp_c:.0f} C > {max_temp_c:.0f} C"
    budget = reading.total_mib * vram_fraction + VRAM_CONTEXT_ALLOWANCE_MIB
    if reading.used_mib - baseline_used_mib > budget:
        return f"VRAM grew by {reading.used_mib - baseline_used_mib:.0f} MiB > {budget:.0f} MiB"
    return None


def wait_for_cooldown(read, target_c: float, timeout_s: float, sleep=time.sleep, clock=time.monotonic) -> float:
    start = clock()
    while True:
        temp = read().temp_c
        if temp <= target_c:
            return temp
        if clock() - start > timeout_s:
            raise RuntimeError(f"GPU still at {temp:.0f} C after {timeout_s:.0f} s; stopping the burst")
        sleep(5)


class Watchdog(threading.Thread):
    """Polls the GPU while a child runs; terminates it on the first guard violation."""

    def __init__(self, proc, read, baseline_used_mib, max_temp_c, vram_fraction, interval_s=1.0):
        super().__init__(daemon=True)
        self.proc, self.read, self.interval_s = proc, read, interval_s
        self.limits = (baseline_used_mib, max_temp_c, vram_fraction)
        self.violation, self.peak_temp_c, self.peak_used_mib = None, 0.0, 0.0
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.is_set() and self.proc.poll() is None:
            try:
                reading = self.read()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                reading, violation = None, f"nvidia-smi failed during the run: {error}"
            else:
                self.peak_temp_c = max(self.peak_temp_c, reading.temp_c)
                self.peak_used_mib = max(self.peak_used_mib, reading.used_mib)
                violation = guard_violation(reading, *self.limits)
            if violation:
                self.violation = violation
                self.proc.terminate()
                return
            self.stopped.wait(self.interval_s)


# ------------------------------------------------------------------ child launcher
def child_command(fraction: float, script: Path, argv: list[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--child", str(fraction), str(script), *argv]


def run_child(fraction: float, script: str, argv: list[str]):
    """Run an upstream CLI in this process with the VRAM cap applied; Python grandchildren get it too."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    original = subprocess.run

    def capped_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[0] == sys.executable and str(cmd[1]).endswith(".py"):
            cmd = child_command(fraction, Path(cmd[1]), [str(c) for c in cmd[2:]])
        return original(cmd, *a, **kw)

    subprocess.run = capped_run
    sys.argv = [script, *argv]
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")


# ------------------------------------------------------------------ data
def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(url: str, path: Path, expected_sha: str) -> None:
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "progress-prize-fp16-burst"})
    with urllib.request.urlopen(request, timeout=300) as src, part.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    digest = sha256_file(part)
    if digest != expected_sha:
        part.unlink()
        raise ValueError(f"SHA-256 mismatch for {path.name}: {digest} != {expected_sha}")
    part.replace(path)


def lfs_oid(pointer_text: str) -> str:
    match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer_text, re.M)
    if not match or "git-lfs" not in pointer_text:
        raise ValueError("not a git-lfs pointer")
    return match.group(1)


def studio_stack(st_hwl: np.ndarray, mesh_index: int = CROP_MESH_INDEX) -> np.ndarray:
    """[H,W,L] smoke-crop stack -> [16,H,W] in the release order (`desc`: planes c+8 down to c-7)."""
    planes = list(range(mesh_index + 8, mesh_index - 8, -1))
    if min(planes) < 0 or max(planes) >= st_hwl.shape[2]:
        raise ValueError("the crop does not hold planes c-7..c+8")
    return np.ascontiguousarray(st_hwl.transpose(2, 0, 1)[planes], dtype=np.uint8)


def center_crop_yx(volume: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return np.ascontiguousarray(volume)
    y0 = max(0, (volume.shape[1] - size) // 2)
    x0 = max(0, (volume.shape[2] - size) // 2)
    return np.ascontiguousarray(volume[:, y0:y0 + size, x0:x0 + size])


def compare_float(candidate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict:
    a, b = candidate[valid].astype(np.float64), reference[valid].astype(np.float64)
    diff = np.abs(a - b)
    r = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    return {"max_abs_diff": round(float(diff.max()), 6), "mean_abs_diff": round(float(diff.mean()), 6),
            "pearson_r": round(r, 6), "pixels": int(valid.sum())}


def compare_levels(candidate: np.ndarray, reference: np.ndarray) -> dict:
    a, b = candidate.astype(np.int16), reference.astype(np.int16)
    diff = np.abs(a - b)
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    r = float(np.corrcoef(af, bf)[0, 1]) if af.std() > 0 and bf.std() > 0 else float("nan")
    return {"max_level_diff": int(diff.max()), "mean_level_diff": round(float(diff.mean()), 6),
            "pearson_r": round(r, 6), "voxels": int(diff.size)}


def read_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def read_zarr(path: Path) -> np.ndarray:
    store = zarr.open(str(path), mode="r")
    return np.asarray((store if hasattr(store, "shape") else store["0"])[...])


def studio_verdict(metrics: dict) -> bool:
    return metrics["max_abs_diff"] <= STUDIO_FP16_MAX_ABS and metrics["pearson_r"] >= STUDIO_FP16_MIN_R


def hecate_verdict(metrics: dict) -> bool:
    return (metrics["max_level_diff"] <= HECATE_MAX_LEVEL_DIFF
            and metrics["mean_level_diff"] <= HECATE_MAX_MEAN_LEVEL_DIFF)


def expected_auto(compute_capability: float | None) -> str | None:
    if compute_capability is None:
        return None
    return "bf16" if compute_capability >= 8.0 else "fp16"


# ------------------------------------------------------------------ checkout
def git(*args, cwd=None, env=None):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True)


def base_commit(patch: Path) -> str:
    head = patch.read_text(encoding="utf-8").split("diff --git ", 1)[0]
    return re.search(r"^Base: .*?\b([0-9a-f]{40})\b", head, re.M).group(1)


def checkout_patched(repo: str, patch: Path, dest: Path, extra_ref: str | None = None) -> str:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    git("clone", "-q", repo, str(dest), env=env)
    if extra_ref:
        git("fetch", "-q", "origin", extra_ref, cwd=dest, env=env)
    base = base_commit(patch)
    git("checkout", "-q", base, cwd=dest, env=env)
    git("apply", "--check", str(patch), cwd=dest)
    git("apply", str(patch), cwd=dest)
    return base


# ------------------------------------------------------------------ burst
def execute(run: Run, root: Path, work: Path, args, log: list) -> dict:
    argv = [a.replace("{work}", str(work)) for a in run.argv]
    script = root / run.script
    env = dict(os.environ)
    if not run.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    record = {"tag": run.tag, "argv": [a.replace("{work}", "<work>") for a in run.argv]}
    if run.gpu:
        record["cooldown_start_c"] = wait_for_cooldown(query_gpu, args.max_gpu_temp_c - args.cooldown_margin_c,
                                                       args.cooldown_timeout_s)
        baseline = query_gpu().used_mib
    start = time.monotonic()
    proc = subprocess.Popen(child_command(args.vram_fraction, script, argv), cwd=root, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dog = None
    if run.gpu:
        dog = Watchdog(proc, query_gpu, baseline, args.max_gpu_temp_c, args.vram_fraction)
        dog.start()
    output, _ = proc.communicate()
    if dog:
        dog.stopped.set()
        dog.join()
        record.update(peak_temp_c=dog.peak_temp_c, peak_used_mib_delta=round(dog.peak_used_mib - baseline, 1))
    record.update(seconds=round(time.monotonic() - start, 2), returncode=proc.returncode)
    if dog and dog.violation:
        record["aborted"] = dog.violation
    if proc.returncode != 0:
        record["log_tail"] = output.replace(str(work), "<work>").replace(str(root), "<repo>")[-2000:]
    log.append(record)
    print(json.dumps({k: v for k, v in record.items() if k != "log_tail"}), flush=True)
    if dog and dog.violation:
        raise RuntimeError(f"{run.tag}: {dog.violation}; burst stopped")
    if proc.returncode != 0:
        raise RuntimeError(f"{run.tag} failed:\n{record['log_tail']}")
    return record


def studio_stage(args, tmp: Path, gpu: GpuReading, receipt: dict) -> None:
    root = tmp / "studio"
    receipt["studio"] = stage = {"base": checkout_patched(STUDIO_REPO, STUDIO_PATCH, root), "runs": []}
    fetch(STUDIO_WEIGHTS_URL, tmp / "studio-softloss-20261023-ema.safetensors", STUDIO_WEIGHTS_SHA)
    fetch(CROP_URL, tmp / "smoke_crop.npz", CROP_SHA)
    with np.load(tmp / "smoke_crop.npz") as z:
        stack = studio_stack(np.asarray(z["st"]))
    np.save(tmp / "studio_stack.npy", stack, allow_pickle=False)
    stage["input"] = {"crop_sha256": CROP_SHA, "shape": list(stack.shape), "order": "desc",
                      "mesh_index": CROP_MESH_INDEX}
    maps = {}
    for run in studio_runs(args.batch, args.studio_stride, gpu.compute_capability):
        execute(run, root, tmp, args, stage["runs"])
        maps[run.tag] = np.load(tmp / run.outputs["map"])
    valid = stack.any(0)
    ref = maps["studio_cpu_fp32"]
    parity = {tag: compare_float(m, ref, valid) for tag, m in maps.items() if tag != "studio_cpu_fp32"}
    stage["parity_vs_cpu_fp32"] = parity
    candidates = {p: compare_float(maps["studio_cuda_auto"], maps[f"studio_cuda_{p}"], valid)["max_abs_diff"]
                  for p in ("fp16", "bf16") if f"studio_cuda_{p}" in maps}
    stage["auto_resolved"] = min(candidates, key=candidates.get)
    stage["auto_expected"] = expected_auto(gpu.compute_capability) or stage["auto_resolved"]
    stage["pass"] = studio_verdict(parity["studio_cuda_fp16"]) and stage["auto_resolved"] == stage["auto_expected"]


def hecate_stage(args, tmp: Path, receipt: dict) -> None:
    root = tmp / "hecate"
    patch = HECATE_PATCHES[args.hecate_patch]
    base = checkout_patched(HECATE_REPO, patch, root, "refs/pr/3" if args.hecate_patch == "pr3" else None)
    receipt["hecate"] = stage = {"patch": patch.name, "base": base, "inputs": {}, "runs": [], "parity": {}}
    inputs = hecate_inputs(args)
    for um, path in inputs.items():
        name = HECATE_CHECKPOINTS[um]
        oid = lfs_oid((root / name).read_text(encoding="utf-8"))
        fetch(f"{HECATE_REPO}/resolve/{base}/{name}", tmp / name, oid)
        volume = center_crop_yx(np.load(path, mmap_mode="r", allow_pickle=False), args.hecate_crop_yx)
        if volume.ndim != 3 or volume.dtype != np.uint8:
            raise ValueError(f"{path.name}: expected a uint8 Z,Y,X volume")
        np.save(tmp / f"hecate_input_{um.replace('.', 'p')}.npy", volume, allow_pickle=False)
        stage["inputs"][um] = {"file": path.name, "sha256": sha256_file(path), "shape_used": list(volume.shape),
                               "checkpoint_sha256": oid}
    runs = hecate_runs(list(inputs), args.batch)
    for run in runs:
        execute(run, root, tmp, args, stage["runs"])
    for run in runs:
        if not run.tag.endswith("_fp16"):
            continue
        ref = run.tag[:-5] + "_fp32"
        png = compare_levels(read_png(tmp / run.outputs["png"]), read_png(tmp / f"{ref}.png"))
        vol = compare_levels(read_zarr(tmp / run.outputs["zarr"]), read_zarr(tmp / f"{ref}.zarr"))
        stage["parity"][run.tag] = {"2d": png, "3d": vol, "pass": hecate_verdict(png) and hecate_verdict(vol)}
    stage["pass"] = all(p["pass"] for p in stage["parity"].values())


def burst(args) -> dict:
    if shutil.which("nvidia-smi") is None or torch is None:
        raise RuntimeError("needs an NVIDIA GPU with nvidia-smi and torch; use --dry-run elsewhere")
    stages = set(args.stages.split(","))
    if "hecate" in stages and (zarr is None or Image is None):
        raise RuntimeError("the hecate stage needs zarr and Pillow installed")
    gpu = query_gpu()
    receipt = {
        "kind": "upstream fp16 parity burst (numerical parity only, not an accuracy claim)",
        "gpu": {"name": gpu.name, "total_mib": gpu.total_mib, "compute_capability": gpu.compute_capability},
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "safety": {"vram_fraction": args.vram_fraction, "batch": args.batch, "max_gpu_temp_c": args.max_gpu_temp_c,
                   "cooldown_margin_c": args.cooldown_margin_c},
        "thresholds": {"studio_fp16_max_abs": STUDIO_FP16_MAX_ABS, "studio_fp16_min_r": STUDIO_FP16_MIN_R,
                       "hecate_max_level_diff": HECATE_MAX_LEVEL_DIFF,
                       "hecate_max_mean_level_diff": HECATE_MAX_MEAN_LEVEL_DIFF},
    }
    with tempfile.TemporaryDirectory(prefix="fp16-burst-") as d:
        tmp = Path(d)
        try:
            if "studio" in stages:
                studio_stage(args, tmp, gpu, receipt)
            if "hecate" in stages:
                hecate_stage(args, tmp, receipt)
        except Exception as error:  # noqa: BLE001 - the receipt records why the burst stopped
            receipt["stopped"] = str(error).replace(str(tmp), "<work>")[-2000:]
    receipt["pass"] = "stopped" not in receipt and all(receipt[s]["pass"] for s in stages)
    return receipt


def plan(args) -> list[dict]:
    stages = set(args.stages.split(","))
    runs = studio_runs(args.batch, args.studio_stride, None) if "studio" in stages else []
    runs += hecate_runs(list(hecate_inputs(args)), args.batch) if "hecate" in stages else []
    return [{"tag": r.tag, "script": r.script, "gpu": r.gpu, "argv": [a.replace("{work}", "<work>") for a in r.argv]}
            for r in runs]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--child"]:
        run_child(float(argv[1]), argv[2], argv[3:])
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if errors := validate(args):
        parser.error("; ".join(errors))
    if args.dry_run:
        print(json.dumps({"safety": {"vram_fraction": args.vram_fraction, "batch": args.batch,
                                     "max_gpu_temp_c": args.max_gpu_temp_c}, "runs": plan(args)}, indent=2))
        return 0
    receipt = burst(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": receipt["pass"], "stopped": receipt.get("stopped")}))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
