# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Serialized, VRAM-capped, zero-disk-bloat RTX 4060 First Letters screener.

Workflow per surface mesh:
  1. Fetch tifxyz files (local path or raw HTTPS directory) into temp workspace.
  2. Optionally snap vertices onto S3 m7 predictions (2 passes, sentinel-safe)
     and stream-render a 31-layer CT stack into RAM via ``stream_snap_render.py``.
  3. Compute ``onsheet_score.score_stack`` layer-coherence metrics.
  4. Run Studio (``studio_t4.py --precision fp16``) and/or PHerc. 306B A-hn
     (``hn4_s0``, ``fp16`` autocast, ``nsig=3``, ``tta=1``) in BOTH depth orders,
     capped at ``--vram-fraction 0.65`` (~5.2 GiB) with CUDA cache clears between passes.
  5. Write compact visual montage + 1:1 native hotspot crops + JSON summary ONLY
     to ``--out-private-dir`` (which is strictly verified to lie outside the git repo),
     then immediately discard the 31-layer CT stack before the next mesh.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
GEMINI_DIR = Path(__file__).resolve().parent
STUDIO_T4_DIR = REPO_ROOT / "firstletters-colab" / "studio_t4"
ONSHEET_DIR = REPO_ROOT / "firstletters-colab" / "onsheet_score"
LINE_REVIEW_DIR = REPO_ROOT / "firstletters-colab" / "line_review"

for _p in (str(GEMINI_DIR), str(STUDIO_T4_DIR), str(ONSHEET_DIR), str(LINE_REVIEW_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import screening_fp_filter as FPF  # noqa: E402
import stream_snap_render as SSR  # noqa: E402

# Pinned model sources (identical to parity scripts)
STUDIO_MODEL_URL = (
    "https://raw.githubusercontent.com/priyankaranke/vesuvius-studio-multiscroll-cpt/"
    "bf66f753176792a966bca17822629fb8821f20c8/studio-multiscroll-cpt-20261023/hf/model.py"
)
STUDIO_MODEL_SHA = "efbb390b6208c480fa8e8681137690b7df897c3e406956b9e804fa66645316d7"
STUDIO_WEIGHTS_URL = (
    "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/"
    "studio-softloss-20261023/studio-softloss-20261023-ema.safetensors"
)
STUDIO_WEIGHTS_SHA = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"

AHN_RELEASE = "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc"
AHN_WEIGHTS_REV = "e57c4d541c239e06450a98c1cf2015d3de8d887e"
AHN_DATA_REV = "f435998b55269762942a979052044309b29650ac"
AHN_REPO = f"https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/{AHN_RELEASE}/"
AHN_ORIGINS = {
    "repo": AHN_REPO,
    "hf-dataset": f"https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/{AHN_DATA_REV}/",
    "hf-model": f"https://huggingface.co/Prodigle/pherc0306b-ink-ahn/resolve/{AHN_WEIGHTS_REV}/",
}
AHN_MANIFEST_SHA = "6e56c61594cfce44ab24d9566da2f6d88ad14e3a6edbafea1b1f465cfdd97770"
AHN_BASE_FILES = [
    "code/claim/ink_v8.py",
    "v8in/config.json",
    "v8in/ink8um/__init__.py",
    "v8in/ink8um/inference.py",
    "v8in/ink8um/modeling.py",
    "v8in/model.safetensors",
    "finalist/hn4_s0.safetensors",
]


def ensure_outside_git_repo(out_dir: Path) -> Path:
    """Refuse to write unread outputs inside the public git repository."""
    resolved_out = out_dir.resolve()
    resolved_repo = REPO_ROOT.resolve()
    try:
        resolved_out.relative_to(resolved_repo)
        raise ValueError(
            f"Refusing to write private screening outputs inside public repo {resolved_repo}: {resolved_out}"
        )
    except ValueError as exc:
        if "Refusing to write" in str(exc):
            raise
    return resolved_out


def query_gpu_telemetry() -> dict[str, int] | None:
    """Return GPU temperature/utilization/memory via ``nvidia-smi``, or raise if GPU is lost."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return None

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
    lowered = combined.lower()
    if "gpu is lost" in lowered or "no devices were found" in lowered or (
        proc.returncode != 0 and any(tok in lowered for tok in ("unable to determine", "lost", "error"))
    ):
        raise RuntimeError(f"GPU is lost or unavailable; aborting before CUDA call: {combined}")
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed (rc={proc.returncode}): {combined}")

    first_line = (proc.stdout or "").strip().splitlines()[0]
    parts = [int(float(tok.strip())) for tok in first_line.split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi CSV output: {first_line!r}")
    return {
        "temp_c": parts[0],
        "util_pct": parts[1],
        "mem_used_mib": parts[2],
        "mem_total_mib": parts[3],
    }


def enforce_gpu_guardrail(
    *,
    max_temp_c: int,
    cooldown_s: float = 4.0,
    max_wait_s: float = 120.0,
    telemetry_fn=None,
) -> dict[str, int] | None:
    """Verify GPU health and pause in ``cooldown_s`` steps if temperature hits ``max_temp_c``."""
    probe = telemetry_fn or query_gpu_telemetry
    stats = probe()
    if stats is None or stats["temp_c"] < max_temp_c:
        return stats

    cool_target = max_temp_c - 6
    waited = 0.0
    step = max(0.0, float(cooldown_s))
    while stats["temp_c"] > cool_target:
        if waited >= max_wait_s:
            raise RuntimeError(
                f"GPU stayed too hot ({stats['temp_c']}C > target {cool_target}C) after {waited:.1f}s cooldown"
            )
        wait_step = min(step, max(0.0, max_wait_s - waited))
        if wait_step <= 0.0:
            raise RuntimeError(
                f"GPU stayed too hot ({stats['temp_c']}C > target {cool_target}C) after {waited:.1f}s cooldown"
            )
        time.sleep(wait_step)
        waited += wait_step
        stats = probe()
        if stats is None:
            return None
    return stats


def fetch_verified(url: str, path: Path, expected_sha: str | None = None) -> str:
    """Download ``url`` to ``path`` with retry and optional SHA-256 verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gemini-rtx4060-screener"})
            with urllib.request.urlopen(req, timeout=180) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            with tmp.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if expected_sha and expected_sha != "-" and digest != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {path.name}: {digest} != {expected_sha}")
            tmp.replace(path)
            return digest
        except urllib.error.HTTPError as exc:
            if tmp.exists():
                tmp.unlink()
            if exc.code in (400, 401, 403, 404, 410) or attempt == 3:
                raise
            time.sleep(2 ** attempt)
        except OSError:
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_mesh_dir(source: str, dest_dir: Path) -> Path:
    """Materialize a tifxyz mesh directory from local path or HTTPS directory URL."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source.lower().startswith(("http://", "https://")):
        base = source.rstrip("/") + "/"
        for name in ("x.tif", "y.tif", "z.tif"):
            fetch_verified(base + name, dest_dir / name)
        for opt in ("meta.json", "mask.tif"):
            try:
                fetch_verified(base + opt, dest_dir / opt)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        return dest_dir

    src_path = Path(source)
    if not src_path.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {src_path}")
    for name in ("x.tif", "y.tif", "z.tif", "meta.json", "mask.tif"):
        cand = src_path / name
        if cand.is_file():
            shutil.copy2(cand, dest_dir / name)
    return dest_dir


def parse_uv_crop(spec: str | None) -> tuple[int, int, int, int] | None:
    """Parse ``r0:r1,c0:c1`` mesh-grid crop specification."""
    if not spec:
        return None
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected r0:r1,c0:c1 for --uv-crop, got {spec!r}")
    r0, r1 = (int(v) for v in parts[0].split(":"))
    c0, c1 = (int(v) for v in parts[1].split(":"))
    if r0 < 0 or c0 < 0 or r1 <= r0 or c1 <= c0:
        raise ValueError(f"Invalid --uv-crop bounds: {spec!r}")
    return r0, r1, c0, c1


def stage_models(work_root: Path, detectors: set[str]) -> dict[str, Path]:
    """Download and SHA-256 verify required models once into temporary ``work_root``."""
    staged: dict[str, Path] = {}
    if "studio" in detectors:
        release_dir = work_root / "studio_release"
        model_py = release_dir / "studio-multiscroll-cpt-20261023" / "hf" / "model.py"
        weights = work_root / "studio-softloss-20261023-ema.safetensors"
        fetch_verified(STUDIO_MODEL_URL, model_py, STUDIO_MODEL_SHA)
        fetch_verified(STUDIO_WEIGHTS_URL, weights, STUDIO_WEIGHTS_SHA)
        staged["studio_release_dir"] = release_dir
        staged["studio_weights"] = weights

    if "ahn" in detectors:
        ahn_root = work_root / "ahn_root"
        index = ahn_root / "context.tsv"
        fetch_verified(AHN_REPO + "docker/context.tsv", index, AHN_MANIFEST_SHA)
        with index.open(encoding="utf-8") as handle:
            records = {
                path: (src, sha)
                for path, src, sha in csv.reader(
                    (line for line in handle if line.strip() and not line.startswith("#")),
                    delimiter="\t",
                )
            }
        for rel in AHN_BASE_FILES:
            locator, expected = records[rel]
            kind, remote = locator.split(":", 1)
            url = remote if kind == "url" else AHN_ORIGINS[kind] + remote
            fetch_verified(url, ahn_root / rel, expected)
        staged["ahn_root"] = ahn_root

    return staged


def run_studio_both_orders(
    stack_lhw: np.ndarray,
    mesh_index: int,
    staged: dict[str, Path],
    *,
    stride: int,
    batch_size: int,
    batch_pause_ms: int = 150,
    max_gpu_temp_c: int = 75,
) -> dict[str, dict]:
    """Run Studio fp16 in RAM across ``desc`` and ``asc`` orders with strict GPU cooldowns."""
    import torch
    import studio_t4

    studio_t4.verify_weights(staged["studio_weights"])
    torch.backends.cudnn.benchmark = False
    model = studio_t4.load_release_model(staged["studio_release_dir"], staged["studio_weights"], "cuda")

    blend = "hann" if stride == 48 else "none"
    kernel = studio_t4.hann96()
    results: dict[str, dict] = {}
    try:
        for order in ("desc", "asc"):
            ks = studio_t4.plane_indices(stack_lhw.shape[0], mesh_index, order, 0)
            vol = np.ascontiguousarray(stack_lhw[ks])
            studio_t4.check_planes_nonzero(vol, ks)

            h, w = vol.shape[-2:]
            result = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32) if blend == "hann" else None
            empty = (vol == 0).all(0)
            sy, sx = studio_t4.window_starts(h, w, stride, blend)
            coords = [
                (y, x)
                for y in sy
                for x in sx
                if not empty[y : y + 96, x : x + 96].all()
            ]

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                for b_idx, i in enumerate(range(0, len(coords), batch_size), start=1):
                    batch = coords[i : i + batch_size]
                    raw = np.stack([studio_t4._crop(vol, y - 16, x - 16) for y, x in batch])
                    xb = torch.from_numpy(raw[:, None]).to(device="cuda", dtype=torch.float32)
                    xb.clamp_(0, 200).div_(255)
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(xb)
                    prob_b = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
                    for (y, x), p in zip(batch, prob_b):
                        hh, ww = min(96, h - y), min(96, w - x)
                        if blend == "hann":
                            result[y : y + hh, x : x + ww] += p[:hh, :ww] * kernel[:hh, :ww]
                            weight[y : y + hh, x : x + ww] += kernel[:hh, :ww]
                        else:
                            result[y : y + hh, x : x + ww] = p[:hh, :ww]
                    torch.cuda.synchronize()
                    if batch_pause_ms > 0:
                        time.sleep(batch_pause_ms / 1000.0)
                    if b_idx % 2 == 0:
                        enforce_gpu_guardrail(max_temp_c=max_gpu_temp_c)

            if blend == "hann":
                result = np.where(weight > 0, result / np.maximum(weight, 1e-12), 0.0).astype(np.float32)
            result[empty] = 0.0
            valid = studio_t4.validity_mask(vol, stride, blend)
            result = np.where(valid, result, 0.0).astype(np.float32)
            results[order] = {
                "prob": result,
                "valid": valid,
                "seconds": round(time.perf_counter() - t0, 2),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            }
            del vol
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def run_ahn_both_orders(
    stack_lhw: np.ndarray,
    mesh_index: int,
    staged: dict[str, Path],
    *,
    batch_size: int,
    batch_pause_ms: int = 150,
    max_gpu_temp_c: int = 75,
) -> dict[str, dict]:
    """Run PHerc. 306B A-hn (hn4_s0) in fp16 CUDA with chunked pauses and thermal checks."""
    import torch
    from safetensors.torch import load_file

    ahn_root = staged["ahn_root"]
    os.environ["V8_DIR"] = str(ahn_root / "v8in")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["INK_DEVICE"] = "cuda"
    claim_dir = str(ahn_root / "code" / "claim")
    if claim_dir not in sys.path:
        sys.path.insert(0, claim_dir)
    import ink_v8 as K  # noqa: E402

    c0 = mesh_index - 12
    c1 = mesh_index + 12
    if c0 < 0 or c1 > stack_lhw.shape[0]:
        raise ValueError(f"A-hn requires 24 layers [{c0}:{c1}] inside stack depth {stack_lhw.shape[0]}")

    slab = stack_lhw[c0:c1]  # [24, H, W]
    weights_path = ahn_root / "finalist" / "hn4_s0.safetensors"
    model = K.InkDetector.from_pretrained(K.V8)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    model = model.to("cuda").eval()

    results: dict[str, dict] = {}
    height, width = slab.shape[1], slab.shape[2]
    chunk_size = max(int(batch_size) * 4, 4)
    try:
        for order in ("asc", "desc"):
            sub = slab if order == "asc" else slab[::-1]
            st = np.ascontiguousarray(np.minimum(sub, 200).transpose(1, 2, 0)).astype(np.uint8)
            cov = sub[12] > 0
            pts = K.positions(cov, min_cov=0.25, stride=21)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            prob = np.zeros((height, width), np.float32)
            valid = np.zeros((height, width), dtype=bool)
            if len(pts) > 0:
                pts = pts[np.argsort(pts[:, 0], kind="stable")]
                num_full = np.zeros((height, width), np.float32)
                den_full = np.zeros((height, width), np.float32)
                for chunk_idx, ci in enumerate(range(0, len(pts), chunk_size), start=1):
                    sub_pts = pts[ci : ci + chunk_size]
                    yb, nu, de = K.part_sums(
                        model, st, sub_pts, "id", batch=batch_size, device="cuda", nsig=3
                    )
                    torch.cuda.synchronize()
                    num_full[yb : yb + nu.shape[0]] += nu
                    den_full[yb : yb + de.shape[0]] += de
                    if batch_pause_ms > 0:
                        time.sleep(batch_pause_ms / 1000.0)
                    enforce_gpu_guardrail(max_temp_c=max_gpu_temp_c)
                valid = (den_full > 1e-9) & cov
                prob[valid] = num_full[valid] / den_full[valid]

            elapsed = time.perf_counter() - t0
            results[order] = {
                "prob": prob,
                "valid": valid,
                "tiles": int(len(pts)),
                "seconds": round(elapsed, 2),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            }
            del st
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def summarize_map(prob: np.ndarray, valid: np.ndarray, voxel_um: float) -> dict:
    """Compute compact summary statistics over valid pixels."""
    n = int(valid.sum())
    area_cm2 = round(float(n * ((voxel_um / 10000.0) ** 2)), 4)
    if n == 0:
        return {"valid_pixels": 0, "valid_cm2": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0, "p995": 0.0, "max": 0.0, "frac_gt_05": 0.0, "frac_gt_07": 0.0}
    vals = prob[valid]
    return {
        "valid_pixels": n,
        "valid_cm2": area_cm2,
        "mean": round(float(vals.mean()), 5),
        "std": round(float(vals.std()), 5),
        "p95": round(float(np.percentile(vals, 95)), 5),
        "p99": round(float(np.percentile(vals, 99)), 5),
        "p995": round(float(np.percentile(vals, 99.5)), 5),
        "max": round(float(vals.max()), 5),
        "frac_gt_05": round(float((vals > 0.5).mean()), 6),
        "frac_gt_07": round(float((vals > 0.7).mean()), 6),
    }


def compute_fp_diagnostics(
    ct_mid: np.ndarray,
    valid_ct: np.ndarray,
    studio: dict[str, dict],
    ahn: dict[str, dict] | None,
    voxel_um: float,
) -> dict:
    """CPU-only crack/void-halo + probability-map FP diagnostics for a Studio ``asc``/``desc`` pair.

    ``studio`` / ``ahn`` map depth order -> ``{"prob", "valid"}``; A-hn adds two-family agreement.
    """
    valid = np.asarray(valid_ct, dtype=bool) & studio["asc"]["valid"] & studio["desc"]["valid"]
    halo = FPF.build_crack_void_halo_mask(ct_mid, valid)
    ahn_maps = {o: ahn[o]["prob"] for o in ("asc", "desc")} if ahn else {"asc": None, "desc": None}
    return FPF.score_probability_map_diagnostics(
        studio["asc"]["prob"],
        studio["desc"]["prob"],
        valid,
        halo,
        ahn_maps["asc"],
        ahn_maps["desc"],
        voxel_um,
    )


def compute_line_first_summary(
    st_res: dict[str, dict],
    valid_ct: np.ndarray,
    voxel_um: float,
    *,
    top_k: int = 4,
) -> dict:
    """Run CPU MIT ``line_review.rank_map`` across ``desc`` and ``asc`` Studio probability maps.

    Explicitly records ``status='ABSTAIN'`` with ``top_score=None`` when no chain passes
    the line-finder threshold (never conflating abstention with score ``0.0``).
    """
    try:
        import review as line_review
    except Exception as exc:
        return {"available": False, "error": str(exc), "orders": {}}

    spacing = [float(voxel_um), float(voxel_um)]
    orders_out: dict[str, dict] = {}
    for order in ("desc", "asc"):
        if order not in st_res:
            continue
        r = st_res[order]
        valid = np.asarray(valid_ct, dtype=bool) & np.asarray(r["valid"], dtype=bool)
        prob = np.where(valid, np.clip(r["prob"], 0.0, 1.0), 0.0).astype(np.float32)
        if not valid.any():
            orders_out[order] = {
                "status": "ABSTAIN",
                "top_score": None,
                "proposal_count": 0,
                "seconds": 0.0,
                "top_proposals": [],
            }
            continue
        rep = line_review.rank_map(prob, valid, spacing, "probability")
        props = rep.get("proposals", [])
        orders_out[order] = {
            "status": "RANKED" if props else "ABSTAIN",
            "top_score": round(float(props[0]["score"]), 5) if props else None,
            "proposal_count": int(len(props)),
            "seconds": round(float(rep.get("seconds", 0.0)), 3),
            "top_proposals": [
                {
                    "rank": idx,
                    "score": round(float(p["score"]), 5),
                    "slot_count": int(p["slot_count"]),
                    "length_mm": round(float(p["length_mm"]), 2),
                    "angle_deg": round(float(p["angle_deg"]), 1),
                    "contrast": round(float(p["contrast"]), 4) if p.get("contrast") is not None else None,
                    "bbox_yxyx": [int(v) for v in p["bbox_yxyx"]],
                }
                for idx, p in enumerate(props[:top_k], start=1)
            ],
        }
    return {"available": True, "schema": "line_first_s3_v1", "orders": orders_out}


def find_top_hotspots(
    prob: np.ndarray,
    valid: np.ndarray,
    *,
    win: int = 512,
    top_k: int = 4,
    min_valid_frac: float = 0.5,
) -> list[dict]:
    """Find non-overlapping top-K ``win x win`` windows ranked by high-probability fraction."""
    h, w = prob.shape
    wy, wx = min(win, h), min(win, w)
    step = max(64, wy // 2)
    hot = ((prob >= 0.45) & valid).astype(np.int64)
    p_sum = np.where(valid, prob, 0.0).astype(np.float64)

    int_val = np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    int_hot = np.pad(hot.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    int_sum = np.pad(p_sum.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    cands = []
    for y in range(0, h - wy + 1, step):
        for x in range(0, w - wx + 1, step):
            v = int_val[y + wy, x + wx] - int_val[y, x + wx] - int_val[y + wy, x] + int_val[y, x]
            if v < min_valid_frac * wy * wx:
                continue
            hs = int_hot[y + wy, x + wx] - int_hot[y, x + wx] - int_hot[y + wy, x] + int_hot[y, x]
            ps = int_sum[y + wy, x + wx] - int_sum[y, x + wx] - int_sum[y + wy, x] + int_sum[y, x]
            score = (hs / max(v, 1)) * 0.7 + (ps / max(v, 1)) * 0.3
            cands.append((score, y, x, v, hs, ps / max(v, 1)))

    cands.sort(reverse=True)
    picked: list[dict] = []
    for score, y, x, v, hs, mean_p in cands:
        if any(abs(y - p["y"]) < wy * 0.65 and abs(x - p["x"]) < wx * 0.65 for p in picked):
            continue
        picked.append(
            {
                "rank": len(picked) + 1,
                "y": int(y),
                "x": int(x),
                "h": int(wy),
                "w": int(wx),
                "score": round(float(score), 5),
                "hot_frac_ge045": round(float(hs / max(v, 1)), 5),
                "mean_prob": round(float(mean_p), 5),
            }
        )
        if len(picked) >= top_k:
            break
    return picked


def _bbox_iou_yxyx(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    iy0 = max(box_a[0], box_b[0])
    ix0 = max(box_a[1], box_b[1])
    iy1 = min(box_a[2], box_b[2])
    ix1 = min(box_a[3], box_b[3])
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    inter = float((iy1 - iy0) * (ix1 - ix0))
    area_a = float(max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1]))
    area_b = float(max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _dedupe_density_hotspots(
    st_res: dict[str, dict],
    *,
    win: int = 512,
    top_k: int = 4,
) -> list[dict]:
    """Merge ``find_top_hotspots`` across ``desc``/``asc`` with spatial deduplication."""
    merged: list[dict] = []
    present_orders = [o for o in ("desc", "asc") if o in st_res]
    for order in present_orders:
        for item in find_top_hotspots(st_res[order]["prob"], st_res[order]["valid"], win=win, top_k=top_k):
            merged.append({**item, "order": order, "source": "density"})
    merged.sort(
        key=lambda h: (float(h["score"]), float(h["hot_frac_ge045"]), float(h["mean_prob"])),
        reverse=True,
    )

    picked: list[dict] = []
    for cand in merged:
        cy, cx, ch, cw = cand["y"], cand["x"], cand["h"], cand["w"]
        if any(abs(cy - p["y"]) < ch * 0.65 and abs(cx - p["x"]) < cw * 0.65 for p in picked):
            continue
        if len(picked) == top_k - 1 and len(present_orders) == 2:
            seen_orders = {p["order"] for p in picked}
            missing = next((o for o in present_orders if o not in seen_orders), None)
            if missing is not None and cand["order"] != missing:
                alt = next(
                    (
                        c
                        for c in merged
                        if c["order"] == missing
                        and not any(abs(c["y"] - p["y"]) < c["h"] * 0.65 and abs(c["x"] - p["x"]) < c["w"] * 0.65 for p in picked)
                    ),
                    None,
                )
                if alt is not None:
                    picked.append(alt)
                    break
        picked.append(cand)
        if len(picked) >= top_k:
            break
    return picked


def select_line_rank_hotspots(
    st_res: dict[str, dict],
    line_rank: dict | None,
    existing_hotspots: list[dict],
    *,
    crop_px: int = 512,
    min_score: float = 2.5,
    max_iou: float = 0.1,
    min_peak_prob: float = 0.65,
    min_hot_frac: float = 0.05,
    max_crops: int = 2,
) -> list[dict]:
    """Select uncovered ``line_rank`` peak sub-crops so chains outside density hotspots are rendered."""
    if not line_rank or not line_rank.get("available"):
        return []

    proposals: list[tuple[float, str, dict]] = []
    orders = line_rank.get("orders", {})
    for order in ("desc", "asc"):
        if order not in st_res:
            continue
        for prop in orders.get(order, {}).get("top_proposals", []):
            proposals.append((float(prop.get("score", 0.0)), order, prop))
    proposals.sort(key=lambda item: item[0], reverse=True)

    taken_boxes: list[tuple[int, int, int, int]] = [
        (int(h["y"]), int(h["x"]), int(h["y"] + h["h"]), int(h["x"] + h["w"]))
        for h in existing_hotspots
    ]
    integrals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    selected: list[dict] = []

    for lr_score, order, prop in proposals:
        if lr_score < min_score:
            continue
        prob = st_res[order]["prob"]
        valid = np.asarray(st_res[order]["valid"], dtype=bool)
        h_img, w_img = prob.shape
        ch, cw = min(crop_px, h_img), min(crop_px, w_img)
        if ch <= 0 or cw <= 0:
            continue

        ry0, rx0, ry1, rx1 = (int(v) for v in prop["bbox_yxyx"])
        ry0 = max(0, min(h_img, ry0))
        rx0 = max(0, min(w_img, rx0))
        ry1 = max(0, min(h_img, ry1))
        rx1 = max(0, min(w_img, rx1))
        if ry1 <= ry0 or rx1 <= rx0:
            continue

        prop_box = (ry0, rx0, ry1, rx1)

        if order not in integrals:
            hot_arr = ((prob >= 0.45) & valid).astype(np.int64)
            p_arr = np.where(valid, prob, 0.0).astype(np.float64)
            integrals[order] = (
                np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))),
                np.pad(hot_arr.cumsum(0).cumsum(1), ((1, 0), (1, 0))),
                np.pad(p_arr.cumsum(0).cumsum(1), ((1, 0), (1, 0))),
            )
        int_val, int_hot, int_sum = integrals[order]

        if ry1 - ry0 <= ch:
            ys = [max(0, min(h_img - ch, (ry0 + ry1 - ch) // 2))]
        else:
            y_end = max(0, min(h_img - ch, ry1 - ch))
            ys = list(range(max(0, min(y_end, ry0)), y_end + 1, 64))
            if not ys or ys[-1] != y_end:
                ys.append(y_end)
        if rx1 - rx0 <= cw:
            xs = [max(0, min(w_img - cw, (rx0 + rx1 - cw) // 2))]
        else:
            x_end = max(0, min(w_img - cw, rx1 - cw))
            xs = list(range(max(0, min(x_end, rx0)), x_end + 1, 64))
            if not xs or xs[-1] != x_end:
                xs.append(x_end)

        best = None
        fallback = None
        for cy in ys:
            for cx in xs:
                crop_box = (cy, cx, cy + ch, cx + cw)
                if any(
                    _bbox_iou_yxyx(crop_box, tb) >= max_iou
                    or (abs(cy - tb[0]) < ch * 0.65 and abs(cx - tb[1]) < cw * 0.65)
                    for tb in taken_boxes
                ):
                    continue
                v_cnt = int(int_val[cy + ch, cx + cw] - int_val[cy, cx + cw] - int_val[cy + ch, cx] + int_val[cy, cx])
                if v_cnt <= 0:
                    continue
                hs_cnt = int(int_hot[cy + ch, cx + cw] - int_hot[cy, cx + cw] - int_hot[cy + ch, cx] + int_hot[cy, cx])
                ps_sum = float(int_sum[cy + ch, cx + cw] - int_sum[cy, cx + cw] - int_sum[cy + ch, cx] + int_sum[cy, cx])
                sc = (hs_cnt / v_cnt) * 0.7 + (ps_sum / v_cnt) * 0.3
                cand_tuple = (sc, v_cnt, cy, cx)
                if fallback is None or cand_tuple > fallback:
                    fallback = cand_tuple
                if v_cnt >= 0.25 * ch * cw and (best is None or cand_tuple > best):
                    best = cand_tuple
        chosen = best or fallback
        if chosen is None:
            continue
        _, _, cy, cx = chosen
        crop_box = (cy, cx, cy + ch, cx + cw)

        p_crop = prob[cy : cy + ch, cx : cx + cw]
        v_crop = valid[cy : cy + ch, cx : cx + cw]
        if not v_crop.any():
            continue
        peak_prob = float(p_crop[v_crop].max())
        hot_frac = float(((p_crop >= 0.45) & v_crop).mean())
        mean_prob = float(p_crop[v_crop].mean())
        if peak_prob < min_peak_prob or hot_frac < min_hot_frac:
            continue

        selected.append(
            {
                "y": int(cy),
                "x": int(cx),
                "h": int(ch),
                "w": int(cw),
                "score": round(float(hot_frac * 0.7 + mean_prob * 0.3), 5),
                "hot_frac_ge045": round(hot_frac, 5),
                "mean_prob": round(mean_prob, 5),
                "peak_prob": round(peak_prob, 5),
                "order": order,
                "source": "line_rank",
                "lr_rank": int(prop.get("rank", 0)),
                "lr_score": round(float(lr_score), 5),
                "lr_bbox_yxyx": [ry0, rx0, ry1, rx1],
            }
        )
        taken_boxes.append(crop_box)
        if len(selected) >= max_crops:
            break
    return selected


def _to_u8_stretch(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [1.0, 99.7])
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.where(valid, np.round(norm * 255.0), 0.0).astype(np.uint8)


def write_visuals(
    out_dir: Path,
    mesh_tag: str,
    ct_mid: np.ndarray,
    valid_ct: np.ndarray,
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    hotspots: list[dict],
    voxel_um: float,
) -> tuple[str, str | None]:
    """Write a labeled montage PNG with 1 cm scale bar + native 1:1 hotspot crop PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = ct_mid.shape
    target_w = min(w, 1600)
    scale = target_w / max(w, 1)
    target_h = max(32, int(round(h * scale)))

    all_panels = [("CT mid", _to_u8_stretch(ct_mid, valid_ct))] + [
        (name, np.where(v, np.clip(np.round(p * 255.0), 0, 255), 0).astype(np.uint8))
        for name, p, v in panels
    ]

    pad = 28
    canvas_h = len(all_panels) * (target_h + pad) + 12
    canvas = Image.new("L", (target_w, canvas_h), color=20)
    draw = ImageDraw.Draw(canvas)

    cm_px_native = 10000.0 / voxel_um
    cm_px_scaled = int(round(cm_px_native * scale))
    for idx, (label, img_u8) in enumerate(all_panels):
        y0 = idx * (target_h + pad) + 4
        draw.text((8, y0), f"{mesh_tag} | {label}", fill=235)
        tile = Image.fromarray(img_u8, mode="L").resize((target_w, target_h), Image.Resampling.BILINEAR)
        canvas.paste(tile, (0, y0 + 20))
        if idx == 0 and 10 <= cm_px_scaled <= target_w - 20:
            by = y0 + 20 + target_h - 10
            draw.line([(16, by), (16 + cm_px_scaled, by)], fill=255, width=3)
            draw.text((16, by - 14), "1 cm", fill=255)

    montage_name = "montage.png"
    canvas.save(out_dir / montage_name, format="PNG")

    hotspot_name = None
    if hotspots and panels:
        n_cols = len(panels) + 1
        cw = hotspots[0]["w"]
        ch = hotspots[0]["h"]
        h_canvas = Image.new("L", (n_cols * cw + (n_cols - 1) * 4, len(hotspots) * (ch + 24)), color=18)
        h_draw = ImageDraw.Draw(h_canvas)
        for row_i, hs in enumerate(hotspots):
            ry0 = row_i * (ch + 24)
            yy, xx = hs["y"], hs["x"]
            if hs.get("source") == "line_rank":
                row_label = (
                    f"#{hs['rank']} LR-{hs.get('order', '')}#{hs.get('lr_rank', '?')} "
                    f"s={float(hs.get('lr_score', 0.0)):.2f} "
                    f"(y={yy}, x={xx}, hot={hs['hot_frac_ge045']:.3f})"
                )
            else:
                order_tag = f" {hs['order']}" if "order" in hs else ""
                row_label = f"#{hs['rank']}{order_tag} (y={yy}, x={xx}, hot={hs['hot_frac_ge045']:.3f})"
            h_draw.text(
                (4, ry0 + 2),
                row_label,
                fill=240,
            )
            ct_patch = _to_u8_stretch(ct_mid[yy : yy + ch, xx : xx + cw], valid_ct[yy : yy + ch, xx : xx + cw])
            h_canvas.paste(Image.fromarray(ct_patch, mode="L"), (0, ry0 + 20))
            for col_i, (_, p, v) in enumerate(panels, start=1):
                u8 = np.where(
                    v[yy : yy + ch, xx : xx + cw],
                    np.clip(np.round(p[yy : yy + ch, xx : xx + cw] * 255.0), 0, 255),
                    0,
                ).astype(np.uint8)
                h_canvas.paste(Image.fromarray(u8, mode="L"), (col_i * (cw + 4), ry0 + 20))
        hotspot_name = "hotspots.png"
        h_canvas.save(out_dir / hotspot_name, format="PNG")

    return montage_name, hotspot_name


def _mesh_dir_tag(mesh_source: str) -> str:
    """Return a short Windows/POSIX-safe directory label for a mesh path or URL."""
    tail = mesh_source.rstrip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", tail).strip("._")
    return (clean[:72] or "mesh")


def render_mesh_ct_stack(
    mesh_source: str,
    work_dir: Path,
    *,
    ct_zarr_url: str,
    m7_zarr_url: str | None,
    snap_passes: int,
    num_slices: int,
    uv_crop: tuple[int, int, int, int] | None,
    cache_mb: int,
    tile_size: int = 256,
    ct_reader_cls=SSR.RemoteZarrV2,
) -> dict:
    """Fetch a tifxyz mesh into ``work_dir``, optionally ``m7``-snap it, and stream-render its CT stack into RAM.

    ``ct_reader_cls`` builds the CT reader (e.g. a byte-capped ``RemoteZarrV2`` subclass). Returns
    ``stack_lhw``, ``valid_ct``, ``geom_metrics`` and the ``m7`` / CT fetch counters.
    """
    fetch_mesh_dir(mesh_source, work_dir / "mesh")
    x, y, z, valid, meta = SSR.read_tifxyz(work_dir / "mesh")
    if uv_crop is not None:
        r0, r1, c0, c1 = uv_crop
        x = x[r0:r1, c0:c1]
        y = y[r0:r1, c0:c1]
        z = z[r0:r1, c0:c1]
        valid = valid[r0:r1, c0:c1]

    scale = meta.get("scale", [0.05, 0.05])
    geom_metrics = None
    m7_bytes = 0
    m7_chunks = 0
    if snap_passes > 0:
        if not m7_zarr_url:
            raise ValueError("--m7-zarr is required when --snap-passes > 0")
        m7_reader = SSR.RemoteZarrV2(
            m7_zarr_url, cache_mb=max(128, cache_mb // 2), workers=8
        )
        try:
            x, y, z, geom_metrics = SSR.snap_tifxyz_to_m7(
                x, y, z, m7_reader, passes=snap_passes
            )
            m7_bytes = int(m7_reader.bytes_fetched)
            m7_chunks = int(m7_reader.chunks_fetched)
        finally:
            m7_reader.close()

    ct_reader = ct_reader_cls(ct_zarr_url, cache_mb=cache_mb, workers=8)
    t_render = time.perf_counter()
    try:
        stack_lhw, valid_ct = SSR.render_surface_stack(
            x, y, z, ct_reader, num_slices=num_slices, scale=scale, tile_size=tile_size
        )
        ct_bytes = int(ct_reader.bytes_fetched)
        ct_chunks = int(ct_reader.chunks_fetched)
    finally:
        ct_reader.close()
    render_sec = round(time.perf_counter() - t_render, 2)
    return {
        "stack_lhw": stack_lhw,
        "valid_ct": valid_ct,
        "geom_metrics": geom_metrics,
        "m7_bytes": m7_bytes,
        "m7_chunks": m7_chunks,
        "ct_bytes": ct_bytes,
        "ct_chunks": ct_chunks,
        "render_sec": render_sec,
    }


def screen_one_mesh(
    *,
    scroll: str,
    mesh_source: str,
    ct_zarr_url: str,
    m7_zarr_url: str | None,
    snap_passes: int,
    num_slices: int,
    uv_crop: tuple[int, int, int, int] | None,
    voxel_um: float,
    detectors: set[str],
    studio_stride: int,
    studio_batch: int,
    ahn_batch: int,
    batch_pause_ms: int = 150,
    max_gpu_temp_c: int = 75,
    cache_mb: int,
    staged: dict[str, Path],
    work_root: Path,
    out_private_dir: Path,
    save_prob_u8: bool = False,
) -> dict:
    """Screen one mesh end-to-end with zero persistent CT retention."""
    import onsheet_score

    mesh_tag = _mesh_dir_tag(mesh_source)
    crop_suffix = (
        f"_r{uv_crop[0]}-{uv_crop[1]}_c{uv_crop[2]}-{uv_crop[3]}"
        if uv_crop is not None
        else ""
    )
    mesh_out_dir = out_private_dir / f"{scroll}_{mesh_tag}{crop_suffix}"
    t_total = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="mesh-", dir=str(work_root), ignore_cleanup_errors=True
    ) as mtmp:
        rendered = render_mesh_ct_stack(
            mesh_source,
            Path(mtmp),
            ct_zarr_url=ct_zarr_url,
            m7_zarr_url=m7_zarr_url,
            snap_passes=snap_passes,
            num_slices=num_slices,
            uv_crop=uv_crop,
            cache_mb=cache_mb,
        )
        stack_lhw, valid_ct = rendered.pop("stack_lhw"), rendered.pop("valid_ct")
        geom_metrics, render_sec = rendered["geom_metrics"], rendered["render_sec"]
        m7_bytes, m7_chunks = rendered["m7_bytes"], rendered["m7_chunks"]
        ct_bytes, ct_chunks = rendered["ct_bytes"], rendered["ct_chunks"]

        t_on = time.perf_counter()
        mesh_idx = (num_slices - 1) // 2
        try:
            coherence = onsheet_score.score_stack(stack_lhw, mid=mesh_idx, dk=2)
        except ModuleNotFoundError:
            fb = SSR.layer_coherence_score(
                stack_lhw,
                mesh_index=mesh_idx,
                delta=min(2, mesh_idx),
                valid_mask=valid_ct,
            )
            coherence = {
                "f_parallel": fb["sheet_parallel_fraction"],
                "n_tiles": fb["patch_count"],
                "mean_zero_shift_r": fb["mean_zero_shift_r"],
                "mean_best_shifted_r": fb["mean_best_shifted_r"],
                "backend": "stream_snap_render_fallback",
            }
        onsheet_sec = round(time.perf_counter() - t_on, 2)

        detector_stats: dict[str, dict] = {}
        panels: list[tuple[str, np.ndarray, np.ndarray]] = []
        primary_map = None
        primary_valid = None
        st_res = None
        ahn_res = None

        if "studio" in detectors and int(valid_ct.sum()) >= 10000:
            st_res = run_studio_both_orders(
                stack_lhw,
                mesh_idx,
                staged,
                stride=studio_stride,
                batch_size=studio_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
            )
            for order in ("desc", "asc"):
                r = st_res[order]
                stats = summarize_map(r["prob"], r["valid"], voxel_um)
                stats.update(
                    seconds=r["seconds"],
                    peak_allocated_mib=r["peak_allocated_mib"],
                    peak_reserved_mib=r["peak_reserved_mib"],
                )
                detector_stats[f"studio_{order}"] = stats
                panels.append((f"Studio fp16 {order}", r["prob"], r["valid"]))
            best_st_ord = max(("desc", "asc"), key=lambda o: detector_stats[f"studio_{o}"]["p995"])
            primary_map = st_res[best_st_ord]["prob"]
            primary_valid = st_res[best_st_ord]["valid"]
            print(
                f"STUDIO_DONE {scroll}/{mesh_tag} desc={detector_stats['studio_desc']['seconds']}s asc={detector_stats['studio_asc']['seconds']}s",
                flush=True,
            )

        if "ahn" in detectors and int(valid_ct.sum()) >= 10000:
            ahn_res = run_ahn_both_orders(
                stack_lhw,
                mesh_idx,
                staged,
                batch_size=ahn_batch,
                batch_pause_ms=batch_pause_ms,
                max_gpu_temp_c=max_gpu_temp_c,
            )
            for order in ("asc", "desc"):
                r = ahn_res[order]
                stats = summarize_map(r["prob"], r["valid"], voxel_um)
                stats.update(
                    tiles=r["tiles"],
                    seconds=r["seconds"],
                    peak_allocated_mib=r["peak_allocated_mib"],
                    peak_reserved_mib=r["peak_reserved_mib"],
                )
                detector_stats[f"ahn_s0_{order}"] = stats
                panels.append((f"A-hn fp16 {order}", r["prob"], r["valid"]))
            if primary_map is None:
                best_ah_ord = max(("asc", "desc"), key=lambda o: detector_stats[f"ahn_s0_{o}"]["p995"])
                primary_map = ahn_res[best_ah_ord]["prob"]
                primary_valid = ahn_res[best_ah_ord]["valid"]
            print(
                f"AHN_DONE {scroll}/{mesh_tag} asc={detector_stats['ahn_s0_asc']['seconds']}s desc={detector_stats['ahn_s0_desc']['seconds']}s",
                flush=True,
            )

        t_fp = time.perf_counter()
        fp_diagnostics = (
            compute_fp_diagnostics(stack_lhw[mesh_idx], valid_ct, st_res, ahn_res, voxel_um)
            if st_res is not None
            else None
        )
        fp_diag_sec = round(time.perf_counter() - t_fp, 2)

        t_lr = time.perf_counter()
        line_rank = (
            compute_line_first_summary(st_res, valid_ct, voxel_um)
            if st_res is not None and int(valid_ct.sum()) >= 10000
            else None
        )
        line_rank_sec = round(time.perf_counter() - t_lr, 2)

        if st_res is not None:
            density_hs = _dedupe_density_hotspots(st_res, win=512, top_k=4)
            lr_hs = select_line_rank_hotspots(st_res, line_rank, density_hs)
            hotspots = [
                {**item, "rank": idx}
                for idx, item in enumerate(density_hs + lr_hs, start=1)
            ]
        else:
            hotspots = (
                find_top_hotspots(primary_map, primary_valid)
                if primary_map is not None and primary_valid is not None
                else []
            )

        t_vis = time.perf_counter()
        montage_file, hotspot_file = write_visuals(
            mesh_out_dir,
            f"{scroll}/{mesh_tag}",
            stack_lhw[mesh_idx],
            valid_ct,
            panels,
            hotspots,
            voxel_um,
        )
        prob_u8_artifacts = None
        if save_prob_u8 and st_res is not None:
            ensure_outside_git_repo(mesh_out_dir)
            prob_u8_artifacts = {}
            for order in ("desc", "asc"):
                if order in st_res:
                    p_arr = np.asarray(st_res[order]["prob"], dtype=np.float32)
                    v_arr = valid_ct & np.asarray(st_res[order]["valid"], dtype=bool)
                    u8_arr = np.where(v_arr, np.clip(np.round(p_arr * 255.0), 0, 255), 0).astype(np.uint8)
                    fname = f"studio_{order}_u8.png"
                    Image.fromarray(u8_arr, mode="L").save(mesh_out_dir / fname, format="PNG")
                    prob_u8_artifacts[f"studio_{order}_u8"] = fname
            Image.fromarray((valid_ct.astype(np.uint8) * 255), mode="L").save(
                mesh_out_dir / "valid_ct_mask.png", format="PNG"
            )
            prob_u8_artifacts["valid_ct_mask"] = "valid_ct_mask.png"
        visuals_sec = round(time.perf_counter() - t_vis, 2)

        del stack_lhw
        gc.collect()

    artifacts_dict: dict[str, object] = {"montage": montage_file, "hotspots": hotspot_file}
    if prob_u8_artifacts is not None:
        artifacts_dict["prob_maps_u8"] = prob_u8_artifacts

    summary = {
        "scroll": scroll,
        "mesh_tag": mesh_tag,
        "voxel_um": voxel_um,
        "num_slices": num_slices,
        "snap_passes": snap_passes,
        "uv_crop": list(uv_crop) if uv_crop else None,
        "geometry_snap_metrics": geom_metrics,
        "onsheet_coherence": coherence,
        "render_seconds": render_sec,
        "m7_chunks_fetched": m7_chunks,
        "m7_mib_fetched": round(m7_bytes / (1024 ** 2), 2),
        "ct_chunks_fetched": ct_chunks,
        "ct_mib_fetched": round(ct_bytes / (1024 ** 2), 2),
        "valid_ct_pixels": int(valid_ct.sum()),
        "valid_ct_cm2": round(float(int(valid_ct.sum()) * ((voxel_um / 10000.0) ** 2)), 4),
        "detectors": detector_stats,
        "fp_diagnostics": fp_diagnostics,
        "line_rank": line_rank,
        "top_hotspots": hotspots,
        "artifacts": artifacts_dict,
        "timing_breakdown": {
            "render_seconds": render_sec,
            "onsheet_seconds": onsheet_sec,
            "fp_diagnostics_seconds": fp_diag_sec,
            "line_rank_seconds": line_rank_sec,
            "visuals_seconds": visuals_sec,
        },
        "total_seconds": round(time.perf_counter() - t_total, 2),
    }
    (mesh_out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETED {scroll}/{mesh_tag} in {summary['total_seconds']}s -> {mesh_out_dir}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser with conservative laptop VRAM and thermal defaults."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--mesh", action="append", required=True, help="Mesh local dir or raw HTTPS URL prefix (repeatable)")
    ap.add_argument("--ct-zarr", required=True, help="S3 or local CT Zarr root URL")
    ap.add_argument("--m7-zarr", default="", help="S3 or local m7 Zarr root URL")
    ap.add_argument("--snap-passes", type=int, default=0)
    ap.add_argument("--num-slices", type=int, default=31)
    ap.add_argument("--uv-crop", default="", help="Optional r0:r1,c0:c1 grid crop")
    ap.add_argument("--voxel-um", type=float, required=True)
    ap.add_argument("--detectors", default="studio,ahn", help="Comma-separated detectors (studio,ahn or none for CPU preflight)")
    ap.add_argument("--studio-stride", type=int, default=96)
    ap.add_argument("--studio-batch", type=int, default=1)
    ap.add_argument("--ahn-batch", type=int, default=1)
    ap.add_argument(
        "--vram-fraction",
        type=float,
        default=0.45,
        help="Per-process CUDA VRAM cap fraction (default 0.45 ~= 3.6 GiB on 8 GB GPUs)",
    )
    ap.add_argument(
        "--batch-pause-ms",
        type=int,
        default=150,
        help="Duty-cycle cooldown sleep per batch to keep GPU utilization/thermals controlled",
    )
    ap.add_argument(
        "--max-gpu-temp-c",
        type=int,
        default=75,
        help="Pause inference to cool down if GPU reaches this temperature",
    )
    ap.add_argument("--cache-mb", type=int, default=512)
    ap.add_argument("--out-private-dir", type=Path, required=True, help="Private output dir strictly outside repo")
    ap.add_argument(
        "--save-prob-u8",
        action="store_true",
        help="Persist compact uint8 prob maps and valid mask PNGs inside private output dir only",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = ensure_outside_git_repo(args.out_private_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detectors = {
        d.strip().lower()
        for d in args.detectors.split(",")
        if d.strip() and d.strip().lower() != "none"
    }
    uv_crop = parse_uv_crop(args.uv_crop or None)

    if detectors:
        enforce_gpu_guardrail(max_temp_c=args.max_gpu_temp_c)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when GPU detectors are enabled")
        torch.cuda.set_per_process_memory_fraction(float(args.vram_fraction))

    with tempfile.TemporaryDirectory(
        prefix="rtx4060-screen-work-", ignore_cleanup_errors=True
    ) as tmp:
        work = Path(tmp)
        staged = stage_models(work, detectors)
        summaries = []
        for mesh_src in args.mesh:
            s = screen_one_mesh(
                scroll=args.scroll,
                mesh_source=mesh_src,
                ct_zarr_url=args.ct_zarr,
                m7_zarr_url=args.m7_zarr or None,
                snap_passes=args.snap_passes,
                num_slices=args.num_slices,
                uv_crop=uv_crop,
                voxel_um=args.voxel_um,
                detectors=detectors,
                studio_stride=args.studio_stride,
                studio_batch=args.studio_batch,
                ahn_batch=args.ahn_batch,
                batch_pause_ms=args.batch_pause_ms,
                max_gpu_temp_c=args.max_gpu_temp_c,
                cache_mb=args.cache_mb,
                staged=staged,
                work_root=work,
                out_private_dir=out_dir,
                save_prob_u8=bool(args.save_prob_u8),
            )
            summaries.append(s)

    batch_path = out_dir / f"{args.scroll}_batch_summary.json"
    batch_path.write_text(json.dumps({"scroll": args.scroll, "meshes": summaries}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
