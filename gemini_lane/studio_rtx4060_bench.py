# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""VRAM-safe RTX 4060 parity & throughput benchmark for Studio (PHerc0211 detector).

Exercises ``firstletters-colab/studio_t4/studio_t4.py`` on the public 169 x 169
PHerc. 306B smoke crop (mesh layer index 12 of the 24-layer slice) in:
  * CPU fp32 reference (batch 1, desc & asc)
  * CUDA bf16 (batch 1, 2, 4 at stride 48 Hann; plus batch 4 asc and stride 96)
  * CUDA fp16 (batch 4 at stride 48 Hann, desc & asc)

Enforces ``torch.cuda.set_per_process_memory_fraction(0.65)`` (~5.2 GiB cap)
and streams all downloaded weights/crops inside a TemporaryDirectory outside
the repository so persistent local disk footprint stays zero.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIO_T4_DIR = REPO_ROOT / "firstletters-colab" / "studio_t4"

MODEL_PY_URL = (
    "https://raw.githubusercontent.com/priyankaranke/vesuvius-studio-multiscroll-cpt/"
    "bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/model.py"
)
MODEL_PY_SHA = "efbb390b6208c480fa8e8681137690b7df897c3e406956b9e804fa66645316d7"
WEIGHTS_URL = (
    "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/"
    "studio-softloss-20261023/studio-softloss-20261023-ema.safetensors"
)
WEIGHTS_SHA = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"
DATA_REV = "f435998b55269762942a979052044309b29650ac"
CROP_URL = f"https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/{DATA_REV}/docker/build/smoke_crop.npz"
CROP_SHA = "4a81e193e253e8b6234eaa4c1b274e33c08b214183951ed8e573c9ae3f8f639d"
AHN_S0_URL = f"https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/{DATA_REV}/results/member_maps/fwd_hn4_s0.npy"
AHN_S0_SHA = "3ca5a625959d6bb6a287ea56ff469ea33a7717bdcd9558c31da95fa003542a22"
CROP_Y0, CROP_X0 = 2110, 4495
CORE_SLICE = slice(63, 106)


def fetch(url: str, path: Path, expected_sha: str) -> str:
    """Download ``url`` to ``path`` with retries and exact SHA-256 check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gemini-studio-rtx4060-bench"})
            with urllib.request.urlopen(req, timeout=180) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            with tmp.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {path.name}: {digest} != {expected_sha}")
            tmp.replace(path)
            return digest
        except OSError:
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def compare_maps(cand: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict:
    """Compute diff and Pearson correlation over valid pixels."""
    a = cand[valid].astype(np.float64)
    b = ref[valid].astype(np.float64)
    diff = np.abs(a - b)
    r = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    return {
        "max_abs_diff": round(float(diff.max()), 6),
        "mean_abs_diff": round(float(diff.mean()), 6),
        "pearson_r": round(r, 6),
    }


def run_benchmark(output_path: Path, vram_fraction: float = 0.65) -> dict:
    """Execute Studio CPU fp32 reference and RTX 4060 bf16/fp16 parity suite."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for studio_rtx4060_bench.py")

    if str(STUDIO_T4_DIR) not in sys.path:
        sys.path.insert(0, str(STUDIO_T4_DIR))
    import studio_t4  # noqa: E402

    torch.cuda.set_per_process_memory_fraction(float(vram_fraction))

    with tempfile.TemporaryDirectory(prefix="studio-rtx4060-bench-") as tmp:
        work = Path(tmp)
        release_dir = work / "release"
        model_py_path = release_dir / "studio-multiscroll-cpt-20261023" / "hf" / "model.py"
        weights_path = work / "studio-softloss-20261023-ema.safetensors"
        crop_npz_path = work / "crop.npz"
        ahn_s0_path = work / "fwd_hn4_s0.npy"
        stack_npy_path = work / "crop_lhw.npy"
        runs_dir = work / "runs"

        fetch(MODEL_PY_URL, model_py_path, MODEL_PY_SHA)
        fetch(WEIGHTS_URL, weights_path, WEIGHTS_SHA)
        fetch(CROP_URL, crop_npz_path, CROP_SHA)
        fetch(AHN_S0_URL, ahn_s0_path, AHN_S0_SHA)

        with np.load(crop_npz_path) as z:
            st = np.array(z["st"], dtype=np.uint8)
        stack_lhw = np.ascontiguousarray(st.transpose(2, 0, 1))
        np.save(stack_npy_path, stack_lhw, allow_pickle=False)

        ahn_mmap = np.load(ahn_s0_path, mmap_mode="r")
        ahn_core = np.array(
            ahn_mmap[CROP_Y0 + 63 : CROP_Y0 + 106, CROP_X0 + 63 : CROP_X0 + 106],
            dtype=np.float32,
        )
        if hasattr(ahn_mmap, "_mmap") and ahn_mmap._mmap is not None:
            ahn_mmap._mmap.close()
        del ahn_mmap
        ahn_s0_path.unlink()

        configs = [
            ("cpu_fp32_desc_s48_b1", "cpu", "fp32", "desc", 48, 1),
            ("cpu_fp32_asc_s48_b1", "cpu", "fp32", "asc", 48, 1),
            ("cuda_bf16_desc_s48_b1", "cuda", "bf16", "desc", 48, 1),
            ("cuda_bf16_desc_s48_b2", "cuda", "bf16", "desc", 48, 2),
            ("cuda_bf16_desc_s48_b4", "cuda", "bf16", "desc", 48, 4),
            ("cuda_bf16_asc_s48_b4", "cuda", "bf16", "asc", 48, 4),
            ("cuda_bf16_desc_s96_b4", "cuda", "bf16", "desc", 96, 4),
            ("cuda_fp16_desc_s48_b2", "cuda", "fp16", "desc", 48, 2),
            ("cuda_fp16_desc_s48_b4", "cuda", "fp16", "desc", 48, 4),
            ("cuda_fp16_asc_s48_b4", "cuda", "fp16", "asc", 48, 4),
        ]

        prob_arrays: dict[str, np.ndarray] = {}
        valid_arrays: dict[str, np.ndarray] = {}
        run_records: list[dict] = []

        for tag, dev, prec, order, stride, batch in configs:
            out_prefix = runs_dir / tag
            receipt = studio_t4.run(
                [
                    "--input",
                    str(stack_npy_path),
                    "--mesh-index",
                    "12",
                    "--order",
                    order,
                    "--stride",
                    str(stride),
                    "--batch-size",
                    str(batch),
                    "--precision",
                    prec,
                    "--device",
                    dev,
                    "--weights",
                    str(weights_path),
                    "--release-dir",
                    str(release_dir),
                    "--out-prefix",
                    str(out_prefix),
                ]
            )
            prob = np.load(out_prefix.with_name(out_prefix.name + ".prob.npy"))
            valid = np.load(out_prefix.with_name(out_prefix.name + ".valid.npy"))
            prob_arrays[tag] = prob
            valid_arrays[tag] = valid

            rec = {
                "tag": tag,
                "device": dev,
                "precision": prec,
                "order": order,
                "stride": stride,
                "blend": receipt["tiling"]["blend"],
                "batch_size": batch,
                "windows": receipt["windows"],
                "inference_seconds": receipt["seconds"]["inference"],
                "ms_per_window": receipt["ms_per_window"],
                "peak_allocated_mib": receipt["memory"]["peak_allocated_mib"],
                "peak_reserved_mib": receipt["memory"]["peak_reserved_mib"],
                "valid_pixels": receipt["pixels"]["valid"],
                "mean_prob": round(float(prob[valid].mean()), 6),
                "p95_prob": round(float(np.percentile(prob[valid], 95)), 6),
                "max_prob": round(float(prob[valid].max()), 6),
            }
            if dev == "cuda":
                ref_tag = f"cpu_fp32_{order}_s48_b1"
                if stride == 48:
                    rec["parity_vs_cpu_fp32"] = compare_maps(prob, prob_arrays[ref_tag], valid)
                else:
                    rec["parity_vs_cpu_fp32_s48_note"] = "stride 96 non-overlapping vs stride 48 Hann"

            studio_core = prob[CORE_SLICE, CORE_SLICE]
            ok_core = np.isfinite(ahn_core)
            core_r = float(np.corrcoef(studio_core[ok_core], ahn_core[ok_core])[0, 1])
            rec["pearson_r_vs_ahn_s0_core_1849px"] = round(core_r, 6)
            run_records.append(rec)
            print(json.dumps(rec), flush=True)

            gc.collect()
            torch.cuda.empty_cache()

        runtime = {
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "vram_fraction_cap": float(vram_fraction),
        }

    bf16_desc = next(r for r in run_records if r["tag"] == "cuda_bf16_desc_s48_b4")
    bf16_asc = next(r for r in run_records if r["tag"] == "cuda_bf16_asc_s48_b4")
    fp16_desc = next(r for r in run_records if r["tag"] == "cuda_fp16_desc_s48_b4")
    fp16_asc = next(r for r in run_records if r["tag"] == "cuda_fp16_asc_s48_b4")

    fp16_pass = all(
        r["parity_vs_cpu_fp32"]["pearson_r"] >= 0.999 and r["parity_vs_cpu_fp32"]["max_abs_diff"] <= 0.02
        for r in (fp16_desc, fp16_asc)
    )
    bf16_pass = all(
        r["parity_vs_cpu_fp32"]["pearson_r"] >= 0.999 and r["parity_vs_cpu_fp32"]["max_abs_diff"] <= 0.02
        for r in (bf16_desc, bf16_asc)
    )
    rec_precision = "fp16" if fp16_pass and not bf16_pass else ("bf16" if bf16_pass else "cpu_fp32")

    summary = {
        "project": "RTX 4060 BF16/FP16 parity and VRAM profile for Studio (PHerc0211 detector) via studio_t4.py",
        "release_repo": studio_t4.RELEASE_REPO,
        "release_commit": studio_t4.RELEASE_COMMIT,
        "model_py_sha256": MODEL_PY_SHA,
        "weights_sha256": WEIGHTS_SHA,
        "crop_sha256": CROP_SHA,
        "crop_shape_lhw": [24, 169, 169],
        "mesh_index": 12,
        "runtime": runtime,
        "runs": run_records,
        "fp16_s48_parity_pass": fp16_pass,
        "bf16_s48_parity_pass": bf16_pass,
        "recommended_rtx4060_precision": rec_precision,
        "scope": (
            f"Parity and VRAM footprint benchmark on the 169 x 169 PHerc. 306B public smoke crop "
            f"({fp16_desc['windows']} windows at stride 48). Verifies native FP16 and BF16 numerical parity vs CPU FP32 "
            "and VRAM safety on an 8 GB RTX 4060 Laptop GPU; the 43 x 43 interior core (1,849 px, 0.37 x 0.37 mm) is "
            "sub-letter scale and not a letter-level cross-family metric or unread-scroll transfer proof."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vram-fraction", type=float, default=0.65)
    args = ap.parse_args()
    res = run_benchmark(args.output, vram_fraction=args.vram_fraction)
    if not res["fp16_s48_parity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
