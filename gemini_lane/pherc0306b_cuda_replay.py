# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""CUDA/CPU parity replay of the three published PHerc. 306B A-hn finalists.

Modelled directly on Codex's ``public-controls/pherc0306b-replay.py`` and
SirProdigle's pinned ``submission/code/smoke_v7.py`` + ``submission/code/ink_v8.py``,
using the exact same pinned commits, SHA-256 verification manifest, 36-tile grid,
and 1,849 interior pixel comparisons on the public smoke crop.

Resource guardrails for consumer 8 GB GPUs (e.g. RTX 4060 Laptop):
  * Enforces ``torch.cuda.set_per_process_memory_fraction`` (default 0.65).
  * Defaults to ``--batch 2`` and processes seeds strictly sequentially.
  * Deletes each downloaded checkpoint and reference map immediately after
    scoring that seed, and uses a temporary directory outside the repository.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request

import numpy as np

RELEASE = "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc"
DATA = "f435998b55269762942a979052044309b29650ac"
WEIGHTS = "e57c4d541c239e06450a98c1cf2015d3de8d887e"
REPO = f"https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/{RELEASE}/"
ORIGINS = {
    "repo": REPO,
    "hf-dataset": f"https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/{DATA}/",
    "hf-model": f"https://huggingface.co/Prodigle/pherc0306b-ink-ahn/resolve/{WEIGHTS}/",
}
MANIFEST_SHA = "6e56c61594cfce44ab24d9566da2f6d88ad14e3a6edbafea1b1f465cfdd97770"
PUBLIC_FILES = [
    "code/claim/ink_v8.py",
    "code/claim/smoke_v7.py",
    "v8in/config.json",
    "v8in/ink8um/__init__.py",
    "v8in/ink8um/inference.py",
    "v8in/ink8um/modeling.py",
    "v8in/model.safetensors",
    "data/smoke/crop.npz",
]
S, TILE = 21, 64


def fetch(url: str, path: Path, expected: str) -> str:
    """Download ``url`` to ``path`` with retry and strict SHA-256 verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "gemini-pherc306b-cuda-parity"}
            )
            with urllib.request.urlopen(req, timeout=120) as src, tmp.open("wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
            with tmp.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if expected != "-" and digest != expected:
                raise ValueError(f"Bad published checksum for {path.name}: {digest} != {expected}")
            tmp.replace(path)
            return digest
        except OSError:
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def run_replay(
    outdir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 2,
    vram_fraction: float = 0.65,
    tta: int = 1,
) -> dict:
    """Execute pinned 306B smoke parity check across seeds 0, 1, and 2."""
    import torch
    from safetensors.torch import load_file

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(vram_fraction))

    with tempfile.TemporaryDirectory(prefix="public-pherc306b-cuda-") as work:
        root = Path(work)
        index = root / "context.tsv"
        fetch(REPO + "docker/context.tsv", index, MANIFEST_SHA)
        with index.open(encoding="utf-8") as handle:
            records = {
                path: (src, sha)
                for path, src, sha in csv.reader(
                    (line for line in handle if line.strip() and not line.startswith("#")),
                    delimiter="\t",
                )
            }

        sources: dict[str, str] = {}

        def materialize(rel: str) -> Path:
            locator, expected = records[rel]
            kind, remote = locator.split(":", 1)
            url = remote if kind == "url" else ORIGINS[kind] + remote
            sources[rel] = fetch(url, root / rel, expected)
            return root / rel

        for rel in PUBLIC_FILES:
            materialize(rel)

        os.environ["V8_DIR"] = str(root / "v8in")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["INK_DEVICE"] = device
        claim_dir = str(root / "code" / "claim")
        if claim_dir not in sys.path:
            sys.path.insert(0, claim_dir)

        import ink_v8 as K  # noqa: E402

        if K.sha256(f"{K.V8}/model.safetensors") != K.V8_SHA:
            raise SystemExit("v8-in base weights sha256 mismatch")

        crop_path = root / "data" / "smoke" / "crop.npz"
        with np.load(crop_path) as z:
            st = np.array(z["st"])
            y0, x0, n = int(z["y0"]), int(z["x0"]), int(z["n"])

        tops = np.arange(0, S * n + 1, S)
        pts = np.array([(a, b) for a in tops for b in tops])
        flips = ["id"] if tta == 1 else list(K.FLIPS)
        c0, c1 = TILE - 1, S * n + 1
        height, width = st.shape[:2]

        device_info = {
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if device == "cuda" else None,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if device == "cuda" else False,
            "autocast_dtype": "float16" if device == "cuda" else "float32",
            "batch_size": int(batch_size),
            "vram_fraction_cap": float(vram_fraction) if device == "cuda" else None,
        }

        outputs = []
        for seed in (0, 1, 2):
            weights_name = f"finalist/hn4_s{seed}.safetensors"
            ref_name = f"data/members/fwd_hn4_s{seed}.npy"
            weights_path = materialize(weights_name)
            ref_path = materialize(ref_name)

            model = K.InkDetector.from_pretrained(K.V8)
            model.load_state_dict(load_file(str(weights_path)), strict=True)
            model = model.to(device).eval()

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            num = np.zeros((height, width), np.float32)
            den = np.zeros((height, width), np.float32)
            for fl in flips:
                yb, nu, de = K.part_sums(
                    model, st, pts, fl, batch=batch_size, device=device, nsig=3
                )
                num[yb : yb + nu.shape[0]] += nu
                den[yb : yb + de.shape[0]] += de
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            peak_alloc_mib = (
                round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
                if device == "cuda"
                else 0.0
            )
            peak_res_mib = (
                round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
                if device == "cuda"
                else 0.0
            )

            pred = num / np.maximum(den, 1e-9)
            mine = pred[c0:c1, c0:c1]

            ref_mmap = np.load(ref_path, mmap_mode="r")
            ref = np.array(ref_mmap[y0 + c0 : y0 + c1, x0 + c0 : x0 + c1], dtype=np.float32)
            if hasattr(ref_mmap, "_mmap") and ref_mmap._mmap is not None:
                ref_mmap._mmap.close()
            del ref_mmap

            ok = np.isfinite(ref)
            diff = np.abs(mine - ref)[ok]
            r = float(np.corrcoef(mine[ok], ref[ok])[0, 1]) if ok.sum() > 2 else float("nan")
            max_abs = round(float(diff.max()), 5)

            seed_res = {
                "seed": seed,
                "weights_sha256": sources[weights_name],
                "tiles": int(len(pts)),
                "flips": flips,
                "compared_px": int(ok.sum()),
                "window_render_px": [y0 + c0, y0 + c1, x0 + c0, x0 + c1],
                "max_abs_diff": max_abs,
                "mean_abs_diff": round(float(diff.mean()), 6),
                "pearson_r": round(r, 6),
                "map_range": [round(float(ref[ok].min()), 4), round(float(ref[ok].max()), 4)],
                "inference_seconds": round(elapsed, 3),
                "peak_vram_allocated_mib": peak_alloc_mib,
                "peak_vram_reserved_mib": peak_res_mib,
                "verdict": "PASS" if max_abs <= 0.02 and r >= 0.999 else "FAIL",
            }
            outputs.append(seed_res)
            print(json.dumps(seed_res), flush=True)

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            weights_path.unlink()
            ref_path.unlink()

    record = {
        "project": "RTX 4060 CUDA parity replay on the three published PHerc. 306B A-hn finalist weights",
        "method_repo_commit": RELEASE,
        "dataset_revision": DATA,
        "finalist_revision": WEIGHTS,
        "manifest_sha256": MANIFEST_SHA,
        "authors": "Liam Warner and Nicola Bodill; see the published credits for the underlying models and data.",
        "reference_cpu_replay": "origin/codex/public-306b-cpu-replay:public-controls/pherc0306b-replay.json",
        "runtime": device_info,
        "input_hashes_sha256": sources,
        "results": outputs,
        "all_pass": all(r["verdict"] == "PASS" for r in outputs),
        "scope": (
            "Numerical parity reproduction on the original published PHerc. 306B CT smoke field, "
            "using the complete author-selected 36-window grid per seed, one pose (tta=1). "
            "This verifies local CUDA arithmetic parity and VRAM safety; it does not measure transfer to an unknown scroll."
        ),
    }

    outdir.parent.mkdir(parents=True, exist_ok=True)
    outdir.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="Output JSON receipt path")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch", type=int, default=2, help="Tile batch size (default 2 for 8 GB VRAM safety)")
    ap.add_argument("--vram-fraction", type=float, default=0.65, help="Per-process CUDA VRAM cap fraction")
    ap.add_argument("--tta", type=int, choices=(1, 4), default=1)
    args = ap.parse_args()
    res = run_replay(
        args.output,
        device=args.device,
        batch_size=args.batch,
        vram_fraction=args.vram_fraction,
        tta=args.tta,
    )
    if not res["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
