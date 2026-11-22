#!/usr/bin/env python3
"""Run the released PHerc0211 Studio ink detector on one rendered surface, in fp16 on a T4.

The released CLI offers bf16 (refused on GPUs without bf16) or fp32 (11.35 GB at batch 1 on a T4).
This tool runs the same network and the same tiling with fp16 autocast on CUDA.

Input: a rendered surface stack, uint8 [L,H,W] (.npy, a directory of per-layer TIFFs NN.tif as written by
vc_render_tifxyz, or a zarr), the mesh layer index and the plane order:
    desc : planes c+8, c+7, ..., c-7   (documented order: 23 down to 8 on a 31-layer render, mesh at 15)
    asc  : planes c-8, c-7, ..., c+7   (the same physical planes on a render with the opposite normal sign)
with c = mesh index + shift.  --prepared takes a [16,H,W] .npy already in model order (e.g. from
../studio_inputs/prepare.py) instead.

Normalisation as the release: float32, clip(raw, 0, 200) / 255.  Tiling as the release: 128 px windows,
central 96 px kept; stride 48 with Hann blending or stride 96 without blending; windows with no CT data in
their central 96 px are skipped; pixels with all 16 planes zero are written as 0.

The network definition is imported from a checkout of the release repository (--release-dir); its model.py
must match the pinned SHA-256.  Weights are never downloaded without --fetch-weights, and are always
checked against the SHA-256 pinned by the release's infer_release.py.

Outputs (never overwritten): PREFIX.prob.npy (float32 [H,W]), PREFIX.valid.npy (bool [H,W]),
PREFIX.preview.png (8-bit), PREFIX.receipt.json.

Tiling and precision code (_crop, hann96, window_starts, predict) adapted from the release's
studio-multiscroll-cpt-20261023/hf/inference.py, MIT License, Copyright (c) 2026 Priyank Aranke.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

RELEASE_REPO = "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt"
RELEASE_COMMIT = "bf66f753176792a966bca17822629fb8821f20c8"
MODEL_PY = "studio-multiscroll-cpt-20261023/hf/model.py"
MODEL_PY_SHA256 = "efbb390b6208c480fa8e8681137690b7df897c3e406956b9e804fa66645316d7"   # model.py at RELEASE_COMMIT
WEIGHTS_SHA256 = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"    # infer_release.py constant
WEIGHTS_BYTES = 893443724
WEIGHTS_URL = (RELEASE_REPO + "/releases/download/studio-softloss-20261023/"
               "studio-softloss-20261023-ema.safetensors")
OUTPUT_SUFFIXES = {"prob": ".prob.npy", "valid": ".valid.npy", "preview": ".preview.png", "receipt": ".receipt.json"}
SCHEMA = "studio-t4-receipt-v1"


# ------------------------------------------------------------------ hashing
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ------------------------------------------------------------------ planes
def plane_indices(n_layers: int, mesh_index: int, order: str, shift: int = 0) -> list[int]:
    """Source layer indices of the 16 model planes, in model order."""
    if order not in ("desc", "asc"):
        raise ValueError("order must be 'desc' or 'asc'")
    if not 0 <= mesh_index < n_layers:
        raise ValueError(f"mesh index {mesh_index} is outside the stack (0..{n_layers - 1})")
    c = mesh_index + shift
    ks = list(range(c + 8, c - 8, -1)) if order == "desc" else list(range(c - 8, c + 8))
    if min(ks) < 0 or max(ks) >= n_layers:
        raise ValueError(f"planes {ks[0]}..{ks[-1]} are outside the stack (0..{n_layers - 1}); "
                         "re-render with more layers, do not pad")
    return ks


# ------------------------------------------------------------------ inputs
class Render:
    """Read-only access to the layers of a rendered stack, uint8 [L,H,W]."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_file() and self.path.suffix == ".npy":
            self.kind = "npy"
            self.arr = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if self.arr.ndim != 3:
                raise ValueError(f"expected a [L,H,W] array, got shape {self.arr.shape}")
            self._check_dtype(self.arr.dtype)
            self.n_layers, self.shape = self.arr.shape[0], tuple(self.arr.shape[1:])
        elif self.path.is_dir() and any((self.path / m).exists() for m in (".zarray", ".zgroup", "zarr.json")):
            self.kind = "zarr"
            import zarr  # optional dependency
            node = zarr.open(str(self.path), mode="r")
            self.arr = node["0"] if not hasattr(node, "shape") else node
            if len(self.arr.shape) != 3:
                raise ValueError(f"expected a [L,H,W] zarr array, got shape {self.arr.shape}")
            self._check_dtype(self.arr.dtype)
            self.n_layers, self.shape = self.arr.shape[0], tuple(self.arr.shape[1:])
        elif self.path.is_dir():
            self.kind = "tiff-dir"
            self.files = self._tiff_layers(self.path)
            import tifffile
            self._tifffile = tifffile
            first = tifffile.imread(self.files[0])
            if first.ndim != 2:
                raise ValueError("each layer TIFF must be a 2-D image")
            self._check_dtype(first.dtype)
            self.n_layers, self.shape = len(self.files), tuple(first.shape)
        else:
            raise ValueError("input must be a .npy file, a zarr store or a directory of per-layer TIFFs")

    @staticmethod
    def _check_dtype(dtype):
        if np.dtype(dtype) != np.uint8:
            raise ValueError(f"input must be uint8 (got {dtype}); no implicit intensity conversion")

    @staticmethod
    def _tiff_layers(path: Path) -> list[Path]:
        for d in (path / "layers", path):
            found = {}
            for f in d.glob("*.tif*") if d.is_dir() else []:
                m = re.fullmatch(r"(\d+)\.tiff?", f.name)
                if m:
                    found[int(m.group(1))] = f
            if found:
                if sorted(found) != list(range(len(found))):
                    raise ValueError("layer TIFFs must be numbered 0..L-1 without gaps")
                return [found[k] for k in range(len(found))]
        raise ValueError("no per-layer TIFFs (00.tif, 01.tif, ...) found")

    def layer(self, k: int) -> np.ndarray:
        if not 0 <= k < self.n_layers:
            raise IndexError(f"layer {k} outside 0..{self.n_layers - 1}")
        if self.kind == "tiff-dir":
            a = self._tifffile.imread(self.files[k])
            self._check_dtype(a.dtype)
            if a.shape != self.shape:
                raise ValueError(f"layer {k} has shape {a.shape}, expected {self.shape}")
        else:
            a = np.asarray(self.arr[k])
        return np.ascontiguousarray(a)

    def stack(self, ks) -> np.ndarray:
        return np.stack([self.layer(k) for k in ks])

    def input_sha256(self):
        """SHA-256 of the input file; for a TIFF directory, of the 'name sha256' manifest of all layers."""
        if self.kind == "npy":
            return sha256_file(self.path)
        if self.kind == "tiff-dir":
            manifest = "".join(f"{f.name} {sha256_file(f)}\n" for f in self.files)
            return hashlib.sha256(manifest.encode()).hexdigest()
        return None   # zarr: only the selected stack is hashed (stack_sha256)


def check_planes_nonzero(stack: np.ndarray, ks):
    zero = [int(k) for k, p in zip(ks, stack) if not p.any()]
    if zero:
        raise ValueError(f"selected plane(s) {zero} are entirely zero; re-render, do not pad")


# ------------------------------------------------------------------ weights and model
def fetch_weights(path: Path):
    """Download the release asset to path (via a .part file) and verify it. Only called with --fetch-weights."""
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as response, open(part, "wb") as stream:
        while block := response.read(1 << 22):
            stream.write(block)
    digest = sha256_file(part)
    if digest != WEIGHTS_SHA256:
        part.unlink()
        raise ValueError(f"downloaded weights SHA-256 {digest} does not match the release ({WEIGHTS_SHA256})")
    part.replace(path)


def verify_weights(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"weights file not found; download it with --fetch-weights or from {WEIGHTS_URL}")
    digest = sha256_file(path)
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"weights SHA-256 {digest} does not match the release ({WEIGHTS_SHA256}); "
                         "no inference was run")
    return digest


def release_info(release_dir: Path) -> dict:
    model_py = release_dir / MODEL_PY
    if not model_py.is_file():
        raise ValueError(f"{MODEL_PY} not found under --release-dir; clone {RELEASE_REPO}")
    digest = sha256_file(model_py)
    if digest != MODEL_PY_SHA256:
        raise ValueError(f"release {MODEL_PY} SHA-256 {digest} differs from the pinned file of commit "
                         f"{RELEASE_COMMIT}; check out that commit")
    try:
        head = subprocess.run(["git", "-C", str(release_dir), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return dict(model_py_sha256=digest, checkout_head=head, pinned_commit=RELEASE_COMMIT)


def import_release_model(release_dir: Path):
    release_info(release_dir)
    spec = importlib.util.spec_from_file_location("studio_release_model", release_dir / MODEL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_release_model(release_dir: Path, weights: Path, device):
    from safetensors.torch import load_file
    module = import_release_model(release_dir)
    model = module.DeepResidualUNet(deeper=False)
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return model.eval().to(device)


# ------------------------------------------------------------------ tiling (the measured wrapper's functions)
def _crop(array, y, x):
    h, w = array.shape[-2:]
    tile = array[:, max(0, y):min(h, y + 128), max(0, x):min(w, x + 128)]
    padding = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    return np.pad(tile, padding, mode="reflect") if any(a or b for a, b in padding) else tile


def hann96():
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    return (w[:, None] * w[None, :]).astype(np.float32)


def window_starts(h, w, stride, blend):
    sy = sorted(set(list(range(0, h, stride)) + ([max(h - 96, 0)] if blend == "hann" else [])))
    sx = sorted(set(list(range(0, w, stride)) + ([max(w - 96, 0)] if blend == "hann" else [])))
    return sy, sx


def predict(model, volume, device="cuda", batch_size=8, amp="fp16", stride=48, blend="hann", skip_empty=True,
            stats=None):
    """Same tiling as hf/inference.py predict(), in memory; amp in {fp32, fp16, bf16}."""
    import torch
    assert volume.ndim == 3 and volume.shape[0] == 16 and volume.dtype == np.uint8
    h, w = volume.shape[-2:]
    result = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32) if blend == "hann" else None
    kernel = hann96()
    empty = (volume == 0).all(0)
    sy, sx = window_starts(h, w, stride, blend)
    coords = [(y, x) for y in sy for x in sx if not (skip_empty and empty[y:y + 96, x:x + 96].all())]
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i + batch_size]
            raw = np.stack([_crop(volume, y - 16, x - 16) for y, x in batch])
            xb = torch.from_numpy(raw[:, None]).to(device=device, dtype=torch.float32)
            xb.clamp_(0, 200).div_(255)
            if str(device).startswith("cuda") and amp in ("fp16", "bf16"):
                ctx = torch.autocast("cuda", dtype=torch.float16 if amp == "fp16" else torch.bfloat16)
            else:
                ctx = nullcontext()
            with ctx:
                logits = model(xb)
            prob = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
            for (y, x), p in zip(batch, prob):
                hh, ww = min(96, h - y), min(96, w - x)
                if blend == "hann":
                    result[y:y + hh, x:x + ww] += p[:hh, :ww] * kernel[:hh, :ww]
                    weight[y:y + hh, x:x + ww] += kernel[:hh, :ww]
                else:
                    result[y:y + hh, x:x + ww] = p[:hh, :ww]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    if blend == "hann":
        result = np.where(weight > 0, result / np.maximum(weight, 1e-12), 0).astype(np.float32)
    result[empty] = 0
    if stats is not None:
        stats.update(windows=len(coords), seconds=round(dt, 2), mpix=round(h * w / 1e6, 3),
                     mpix_data=round(float((~empty).sum()) / 1e6, 3),
                     s_per_mpix_data=round(dt / max(float((~empty).sum()) / 1e6, 1e-9), 2),
                     ms_per_window=round(1000 * dt / max(len(coords), 1), 1))
    return result


def covered_mask(volume, stride, blend, skip_empty=True):
    """Pixels inside the central 96 px of at least one window that predict() runs."""
    h, w = volume.shape[-2:]
    empty = (volume == 0).all(0)
    covered = np.zeros((h, w), bool)
    sy, sx = window_starts(h, w, stride, blend)
    for y in sy:
        for x in sx:
            if not (skip_empty and empty[y:y + 96, x:x + 96].all()):
                covered[y:y + 96, x:x + 96] = True
    return covered


def validity_mask(volume, stride, blend):
    """Not all 16 selected planes zero, and covered by at least one window."""
    return ~(volume == 0).all(0) & covered_mask(volume, stride, blend)


def to_u8(p):
    return np.clip(np.round(p * 255), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ devices
def resolve_device_and_precision(device: str, precision: str):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu")
    native_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    if precision == "auto":
        precision = "fp32" if device == "cpu" else ("bf16" if native_bf16 else "fp16")
    if device == "cpu" and precision != "fp32":
        raise ValueError("on CPU only --precision fp32 is supported")
    if precision == "bf16" and not native_bf16:
        raise ValueError("this GPU has no native bf16 (compute capability < 8.0); use --precision fp16")
    return device, precision


def device_record(device):
    import torch
    rec = dict(device=device, torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version())
    if device == "cuda":
        rec["device_name"] = torch.cuda.get_device_name(0)
        rec["compute_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        rec["device_name"] = platform.processor() or platform.machine()
        rec["cpu_threads"] = torch.get_num_threads()
    return rec


def peak_rss_kib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # noqa: BLE001 - not available on every platform
        return None


# ------------------------------------------------------------------ outputs
def output_paths(prefix) -> dict:
    return {k: Path(str(prefix) + s) for k, s in OUTPUT_SUFFIXES.items()}


def refuse_existing(paths: dict):
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(f"output(s) already exist: {', '.join(existing)}; choose a new --out-prefix")


def write_outputs(paths, prob, valid, receipt):
    """Write the four outputs with exclusive creation; the receipt (with output hashes) last."""
    from PIL import Image
    paths["prob"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["prob"], "xb") as f:
        np.save(f, prob.astype(np.float32), allow_pickle=False)
    with open(paths["valid"], "xb") as f:
        np.save(f, valid.astype(bool), allow_pickle=False)
    with open(paths["preview"], "xb") as f:
        Image.fromarray(np.where(valid, to_u8(prob), 0).astype(np.uint8)).save(f, format="PNG")
    receipt["outputs"] = {k: dict(file=paths[k].name, sha256=sha256_file(paths[k])) for k in ("prob", "valid", "preview")}
    with open(paths["receipt"], "x") as f:
        json.dump(receipt, f, indent=1)
        f.write("\n")


# ------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="uint8 [L,H,W] .npy, a directory of per-layer TIFFs (NN.tif, or layers/NN.tif), or a zarr")
    p.add_argument("--mesh-index", type=int, help="layer index of the mesh surface in the render (e.g. 20 of 41)")
    p.add_argument("--order", choices=("desc", "asc"), help="desc: c+8 down to c-7; asc: c-8 up to c+7")
    p.add_argument("--shift", type=int, default=0, help="centre shift in layers (default 0)")
    p.add_argument("--prepared", action="store_true",
                   help="input is already a [16,H,W] .npy in model order; no plane selection")
    p.add_argument("--stride", type=int, choices=(48, 96), default=48,
                   help="48 = Hann blending, no tile pattern (default); 96 = no overlap, no blending, ~4x faster")
    p.add_argument("--batch-size", type=int, help="windows per forward (default 16 on CUDA, 1 on CPU)")
    p.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="auto",
                   help="auto: fp16 on CUDA without native bf16, bf16 with it, fp32 on CPU")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--threads", type=int, help="CPU threads for torch")
    p.add_argument("--weights", type=Path, required=True, help="studio-softloss-20261023-ema.safetensors")
    p.add_argument("--fetch-weights", action="store_true",
                   help="download the release asset to --weights if it is missing (SHA-256 always checked)")
    p.add_argument("--release-dir", type=Path, required=True,
                   help=f"checkout of {RELEASE_REPO} (commit {RELEASE_COMMIT[:7]}); model.py is imported from it")
    p.add_argument("--out-prefix", type=Path, required=True, help="writes PREFIX.prob.npy, .valid.npy, "
                   ".preview.png, .receipt.json; refuses to overwrite")
    return p


def run(argv=None, model_loader=None):
    """Run the CLI. model_loader(release_dir, weights, device) -> model; tests pass a stand-in."""
    parser = build_parser()
    a = parser.parse_args(argv)
    t_start = time.time()
    paths = output_paths(a.out_prefix)
    refuse_existing(paths)
    if a.batch_size is not None and a.batch_size < 1:
        parser.error("--batch-size must be positive")
    if a.prepared and (a.mesh_index is not None or a.order is not None or a.shift):
        parser.error("--prepared takes no --mesh-index, --order or --shift")
    if not a.prepared and (a.mesh_index is None or a.order is None):
        parser.error("--mesh-index and --order are required (or --prepared)")
    blend = "hann" if a.stride == 48 else "none"

    # input first (cheap errors before the weights are hashed)
    render = Render(a.input)
    if a.prepared:
        if render.kind != "npy" or render.n_layers != 16:
            raise ValueError("--prepared expects a [16,H,W] uint8 .npy in model order")
        ks = list(range(16))
    else:
        ks = plane_indices(render.n_layers, a.mesh_index, a.order, a.shift)
    stack = render.stack(ks)
    check_planes_nonzero(stack, ks)

    release = release_info(a.release_dir)
    if a.fetch_weights and not a.weights.exists():
        fetch_weights(a.weights)
    t0 = time.time()
    weights_sha = verify_weights(a.weights)
    t_verify = time.time() - t0

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    device, precision = resolve_device_and_precision(a.device, a.precision)
    batch = a.batch_size or (16 if device == "cuda" else 1)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    t0 = time.time()
    model = (model_loader or load_release_model)(a.release_dir, a.weights, device)
    t_load = time.time() - t0
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    stats = {}
    prob = predict(model, stack, device=device, batch_size=batch, amp=precision, stride=a.stride, blend=blend,
                   stats=stats)
    memory = dict(peak_allocated_mib=None, peak_reserved_mib=None)
    if device == "cuda":
        memory = dict(peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                      peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20))
    memory["process_peak_rss_kib"] = peak_rss_kib()
    valid = validity_mask(stack, a.stride, blend)

    receipt = dict(
        schema=SCHEMA,
        tool_sha256=sha256_file(__file__),
        release=dict(repository=RELEASE_REPO, **release),
        weights=dict(sha256=weights_sha, expected_sha256=WEIGHTS_SHA256, licence="CC BY-NC 4.0"),
        input=dict(kind=render.kind, sha256=render.input_sha256(), shape_lhw=[render.n_layers, *render.shape],
                   stack_sha256=sha256_array(stack), prepared=a.prepared),
        planes=dict(mesh_index=a.mesh_index, order=a.order if not a.prepared else "as prepared",
                    shift=a.shift, source_layers=ks),
        normalisation="float32, clip(raw, 0, 200) / 255",
        tiling=dict(window=128, kept=96, stride=a.stride, blend=blend, skip_empty_windows=True),
        precision=precision, batch_size=batch,
        device=device_record(device),
        memory=memory,
        seconds=dict(weights_verify=round(t_verify, 2), model_load=round(t_load, 2), inference=stats["seconds"],
                     total=None),
        windows=stats["windows"], ms_per_window=stats["ms_per_window"],
        pixels=dict(height=int(stack.shape[1]), width=int(stack.shape[2]), total=int(stack.shape[1] * stack.shape[2]),
                    valid=int(valid.sum())),
    )
    receipt["seconds"]["total"] = round(time.time() - t_start, 2)
    write_outputs(paths, prob, valid, receipt)
    return receipt


def main(argv=None):
    try:
        receipt = run(argv)
    except (ValueError, FileExistsError) as exc:
        print(f"studio_t4: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(precision=receipt["precision"], device=receipt["device"]["device_name"],
                          seconds=receipt["seconds"], memory=receipt["memory"], pixels=receipt["pixels"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
