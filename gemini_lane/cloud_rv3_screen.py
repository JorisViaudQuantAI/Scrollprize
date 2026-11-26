# SPDX-License-Identifier: MIT
"""Auditable Cloud GPU Reader v3 Fast Screener for First Letters volumes.

Uses ``stream_snap_render.py`` (chunk-cell partitioned trilinear sampling,
thread-local HTTP sessions, and signed initial-normal snap metrics) paired with
upstream MIT ``Reader v3`` (``domenicor046/reader-v3``, 17 layers, FP16) via
``ScrollPrize/villa`` and captures native ``float32`` ``probability.zarr /
weight.zarr`` ratios before cleanup.

All GPU/network dependencies (``torch``, ``zarr``, ``huggingface_hub``, ``villa``)
are imported lazily inside runtime functions so pure-NumPy metrics, unit tests,
and CPU-only audits run without CUDA or extra packages.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stream_snap_render as SSR  # noqa: E402

VOXEL_UM = 9.362
PIN_PS = "66000516c97e043b696ce874f145e29638109c91"
PS_BASE = f"https://raw.githubusercontent.com/pscamillo/vesuvius-eligible-meshes/{PIN_PS}/meshes/PHerc0125"
CT_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr"
M7_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr"

RV3_REPO = "domenicor046/reader-v3"
RV3_REV = "39f626e595304c35c6ef574ffc0b6cc1e0ed44cb"
RV3_FILE = "reader-v3.pth"
RV3_SHA = "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a"

DEFAULT_TARGETS: tuple[str, ...] = (
    "z11136_w035",
    "z9936_w025",
    "z7536_w035",
    "z11136_w025",
    "z10544_w015",
    "z14736_w045",
    "z15344_w045",
    "z10544_w030",
    "z11744_w035",
    "z14144_w030",
)


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file in streaming 4 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blk := f.read(1 << 22):
            h.update(blk)
    return h.hexdigest()


def summarize_prob_map(prob: np.ndarray, valid: np.ndarray, voxel_um: float = VOXEL_UM) -> dict[str, Any]:
    """Compute standardized tail statistics on a valid boolean mask."""
    valid_b = np.asarray(valid, dtype=bool)
    n = int(valid_b.sum())
    cm2 = round(float(n * ((float(voxel_um) / 10000.0) ** 2)), 4)
    if n == 0:
        return {"valid_pixels": 0, "valid_cm2": 0.0}
    v = np.asarray(prob, dtype=np.float32)[valid_b]
    return {
        "valid_pixels": n,
        "valid_cm2": cm2,
        "mean": round(float(v.mean()), 5),
        "std": round(float(v.std()), 5),
        "p95": round(float(np.percentile(v, 95)), 5),
        "p99": round(float(np.percentile(v, 99)), 5),
        "p995": round(float(np.percentile(v, 99.5)), 5),
        "max": round(float(v.max()), 5),
        "frac_gt_05": round(float((v > 0.5).mean()), 6),
        "frac_gt_07": round(float((v > 0.7).mean()), 6),
    }


def compute_direction_comparison(
    pf: np.ndarray,
    pr: np.ndarray,
    common: np.ndarray,
    *,
    tile_size: int = 512,
    prob_thresh: float = 0.7,
) -> dict[str, Any]:
    """Compare forward vs reverse Reader v3 hit asymmetry and regional clustering."""
    common_b = np.asarray(common, dtype=bool)
    pf_a = np.asarray(pf, dtype=np.float32)
    pr_a = np.asarray(pr, dtype=np.float32)

    mf = (pf_a >= prob_thresh) & common_b
    mr = (pr_a >= prob_thresh) & common_b
    hf = int(mf.sum())
    hr = int(mr.sum())
    bil = int((mf & mr).sum())

    dom = "forward" if hf >= hr else "reverse"
    dom_hits, opp_hits = (hf, hr) if dom == "forward" else (hr, hf)
    asym = round(dom_hits / max(opp_hits, 1), 4) if dom_hits > 0 else 0.0
    bil_frac = round(bil / max(dom_hits, 1), 6) if dom_hits > 0 else 0.0

    dom_m = mf if dom == "forward" else mr
    opp_m = mr if dom == "forward" else mf
    max_reg_asym, max_reg_hits = 0.0, 0
    h, w = common_b.shape
    ts = max(1, int(tile_size))
    for y0 in range(0, h, ts):
        for x0 in range(0, w, ts):
            dh = int(dom_m[y0 : y0 + ts, x0 : x0 + ts].sum())
            oh = int(opp_m[y0 : y0 + ts, x0 : x0 + ts].sum())
            ra = dh / max(oh, 1)
            if dh >= 500 and ra > max_reg_asym:
                max_reg_asym, max_reg_hits = round(ra, 3), dh

    n_common = max(int(common_b.sum()), 1)
    dom_frac_07 = dom_hits / n_common
    alert = bool(
        (asym >= 3.0 and dom_frac_07 >= 0.015 and bil_frac < 0.03)
        or (max_reg_asym >= 5.0 and max_reg_hits >= 1500 and bil_frac < 0.05)
    )
    return {
        "dominant_direction": dom,
        "hits_ge_07_forward": hf,
        "hits_ge_07_reverse": hr,
        "bilateral_ge_07_px": bil,
        "asymmetry_ratio": asym,
        "bilateral_overlap_frac": bil_frac,
        "max_regional_asymmetry": max_reg_asym,
        "max_regional_dom_hits": max_reg_hits,
        "hit_alert": alert,
    }


def stretch_u8(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-stretch (1%..99.7%) an array over valid pixels to uint8."""
    valid_b = np.asarray(valid, dtype=bool)
    if not valid_b.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid_b], [1.0, 99.7])
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.where(valid_b, np.round(norm * 255.0), 0.0).astype(np.uint8)


def save_visual_panel(
    path: Path,
    title: str,
    ct_mid: np.ndarray,
    valid: np.ndarray,
    fwd: np.ndarray,
    rev: np.ndarray,
    *,
    voxel_um: float = VOXEL_UM,
) -> None:
    """Render a 5-strip diagnostic visual_panel.png outside any git repo."""
    from PIL import Image, ImageDraw

    h, w = ct_mid.shape
    tw = min(w, 1400)
    sc = tw / max(w, 1)
    th = max(48, int(round(h * sc)))
    panels = [
        (f"{title} | CT mid", stretch_u8(ct_mid, valid)),
        ("Reader v3 forward (raw 0..1)", np.where(valid, np.clip(np.rint(fwd * 255.0), 0, 255), 0).astype(np.uint8)),
        ("Reader v3 forward (stretched 1-99.7%)", stretch_u8(fwd, valid)),
        ("Reader v3 reverse (raw 0..1)", np.where(valid, np.clip(np.rint(rev * 255.0), 0, 255), 0).astype(np.uint8)),
        ("Reader v3 reverse (stretched 1-99.7%)", stretch_u8(rev, valid)),
    ]
    pad = 24
    canv = Image.new("L", (tw, len(panels) * (th + pad) + 8), color=18)
    dr = ImageDraw.Draw(canv)
    cm_px = int(round((10000.0 / float(voxel_um)) * sc))
    for i, (lbl, u8) in enumerate(panels):
        y0 = i * (th + pad) + 4
        dr.text((8, y0), lbl, fill=240)
        canv.paste(Image.fromarray(u8).resize((tw, th), Image.Resampling.BILINEAR), (0, y0 + 18))
        if i == 0 and 10 <= cm_px <= tw - 20:
            by = y0 + 18 + th - 10
            dr.line([(16, by), (16 + cm_px, by)], fill=255, width=3)
            dr.text((16, by - 14), "1 cm", fill=255)
    canv.save(path, format="PNG")


def fetch_mesh(seg_url: str, dest: Path) -> None:
    """Fetch raw tifxyz files for one segment over HTTPS."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    for fn in ("x.tif", "y.tif", "z.tif", "meta.json"):
        r = requests.get(f"{seg_url.rstrip('/')}/{fn}", timeout=60)
        if fn == "meta.json" and r.status_code == 404:
            continue
        r.raise_for_status()
        (dest / fn).write_bytes(r.content)


def run_rv3_both(stack_17: np.ndarray, valid_ct: np.ndarray, ckpt: Path, tmp_dir: Path, tag: str) -> dict[str, dict[str, Any]]:
    """Run upstream MIT Reader v3 forward + reverse via villa_infer and capture native float32 ratios."""
    import tifffile
    import torch
    import zarr
    from vesuvius.ink_detection.inference import infer as villa_infer

    surf_zarr = tmp_dir / f"{tag}_s17.zarr"
    mask_tif = tmp_dir / f"{tag}_mask.tif"
    if surf_zarr.exists():
        shutil.rmtree(surf_zarr)
    zarr.open_array(str(surf_zarr), mode="w", shape=stack_17.shape, chunks=(17, 256, 256), dtype=np.uint8, zarr_format=2)[:] = stack_17
    tifffile.imwrite(str(mask_tif), valid_ct.astype(np.uint8) * 255)
    out: dict[str, dict[str, Any]] = {}
    try:
        for direction in ("forward", "reverse"):
            pred_dir = tmp_dir / f"{tag}_{direction}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_tif = pred_dir / "pred.tif"
            for bs in (8, 4, 2, 1):
                captured: dict[str, Any] = {"prob_f32": None, "valid_f32": None, "source": "u8_fallback"}
                real_rmtree = shutil.rmtree

                def hooked_rmtree(path, *args, **kwargs):
                    p = Path(path)
                    if p.name.startswith("ink_flat_infer_") and p.is_dir():
                        pz, wz = p / "probability.zarr", p / "weight.zarr"
                        if pz.exists() and wz.exists():
                            pa = np.asarray(zarr.open(str(pz), mode="r"), dtype=np.float32)
                            wa = np.asarray(zarr.open(str(wz), mode="r"), dtype=np.float32)
                            vw = wa > 1e-6
                            captured["prob_f32"] = np.where(vw, pa / np.maximum(wa, 1e-6), np.nan).astype(np.float32)
                            captured["valid_f32"] = vw
                            captured["source"] = "native_zarr_ratio"
                    return real_rmtree(path, *args, **kwargs)

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                shutil.rmtree = hooked_rmtree
                try:
                    villa_infer.main([
                        str(surf_zarr),
                        str(ckpt),
                        str(pred_tif),
                        "--mask-path",
                        str(mask_tif),
                        "--overlap",
                        "0.5",
                        "--blend-mode",
                        "hann",
                        "--direction",
                        direction,
                        "--batch-size",
                        str(bs),
                        "--num-workers",
                        "0",
                        "--no-compile",
                    ])
                except torch.cuda.OutOfMemoryError:
                    shutil.rmtree = real_rmtree
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                finally:
                    shutil.rmtree = real_rmtree

                torch.cuda.synchronize()
                secs = round(time.perf_counter() - t0, 2)
                if captured["prob_f32"] is None:
                    u8 = tifffile.imread(str(pred_tif))
                    prob = u8.astype(np.float32) / 255.0
                    val = valid_ct.copy()
                else:
                    prob = captured["prob_f32"]
                    val = np.asarray(captured["valid_f32"], dtype=bool) & valid_ct & np.isfinite(prob)
                prob = np.where(val, np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0), 0.0).astype(np.float32)
                out[direction] = {"prob": prob, "valid": val, "seconds": secs, "batch_size": bs, "source": captured["source"]}
                break
    finally:
        shutil.rmtree(surf_zarr, ignore_errors=True)
        mask_tif.unlink(missing_ok=True)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auditable Cloud GPU Reader v3 Fast Screener")
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS), help="Comma-separated mesh segment tags.")
    p.add_argument("--out-dir", default="cloud_rv3_out", help="Output directory (keep outside git repo).")
    p.add_argument("--mesh-base-url", default=PS_BASE, help="Base HTTPS URL holding segment directories.")
    p.add_argument("--ct-url", default=CT_URL, help="Remote or local CT OME-Zarr v2 path/URL.")
    p.add_argument("--m7-url", default=M7_URL, help="Remote or local m7 surface Zarr v2 path/URL.")
    p.add_argument("--voxel-um", type=float, default=VOXEL_UM, help="Voxel size in micrometers.")
    p.add_argument("--reuse-zarr-cache", action="store_true", help="Keep CT & m7 LRU caches warm across targets.")
    p.add_argument("--no-reuse-zarr-cache", dest="reuse_zarr_cache", action="store_false")
    p.set_defaults(reuse_zarr_cache=True)
    p.add_argument("--m7-cache-mb", type=float, default=1024.0, help="In-memory m7 chunk cache MiB.")
    p.add_argument("--ct-cache-mb", type=float, default=2048.0, help="In-memory CT chunk cache MiB.")
    p.add_argument("--workers", type=int, default=16, help="HTTP chunk fetch worker threads.")
    p.add_argument("--snap-passes", type=int, default=2, help="Two-pass m7 ridge-snap iterations.")
    p.add_argument("--skip-visuals", action="store_true", help="Skip saving visual_panel.png outputs.")
    p.add_argument("--pack-tar", action="store_true", help="Write out_batch.tar.gz inside parent of out-dir.")
    return p


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Execute screening batch on GPU and return structured index payload."""
    from huggingface_hub import hf_hub_download
    import torch

    t_all = time.perf_counter()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = Path(hf_hub_download(RV3_REPO, RV3_FILE, revision=RV3_REV))
    if sha256_file(ckpt) != RV3_SHA:
        raise RuntimeError("Reader v3 checkpoint SHA-256 mismatch")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    index_rows: list[dict[str, Any]] = []

    shared_m7 = SSR.RemoteZarrV2(args.m7_url, "0", cache_mb=args.m7_cache_mb, workers=args.workers) if args.reuse_zarr_cache else None
    shared_ct = SSR.RemoteZarrV2(args.ct_url, "0", cache_mb=args.ct_cache_mb, workers=args.workers) if args.reuse_zarr_cache else None

    try:
        with tempfile.TemporaryDirectory(prefix="rv3_cloud_work_", ignore_cleanup_errors=True) as tmp:
            wdir = Path(tmp)
            for seg in targets:
                seg_out = out_root / f"PHerc0125_{seg}"
                sum_path = seg_out / "summary.json"
                if sum_path.is_file():
                    row = json.loads(sum_path.read_text(encoding="utf-8"))
                    index_rows.append(row)
                    continue

                t_seg = time.perf_counter()
                seg_out.mkdir(parents=True, exist_ok=True)
                mdir = wdir / f"mesh_{seg}"
                fetch_mesh(f"{args.mesh_base_url.rstrip('/')}/{seg}", mdir)
                x, y, z, _valid, meta = SSR.read_tifxyz(mdir)

                m7 = shared_m7 or SSR.RemoteZarrV2(args.m7_url, "0", cache_mb=args.m7_cache_mb, workers=args.workers)
                b_m7_0, c_m7_0 = int(m7.bytes_fetched), int(m7.chunks_fetched)
                t_snap = time.perf_counter()
                try:
                    sx, sy, sz, g_metrics = SSR.snap_tifxyz_to_m7(x, y, z, m7, passes=args.snap_passes)
                    m7_mib = round((int(m7.bytes_fetched) - b_m7_0) / (1024 ** 2), 2)
                    m7_chunks = int(m7.chunks_fetched) - c_m7_0
                finally:
                    if not args.reuse_zarr_cache:
                        m7.close()
                snap_s = round(time.perf_counter() - t_snap, 2)

                ct = shared_ct or SSR.RemoteZarrV2(args.ct_url, "0", cache_mb=args.ct_cache_mb, workers=args.workers)
                b_ct_0, c_ct_0 = int(ct.bytes_fetched), int(ct.chunks_fetched)
                t_r = time.perf_counter()
                try:
                    stack_17, val_ct = SSR.render_surface_stack(
                        sx,
                        sy,
                        sz,
                        ct,
                        num_slices=17,
                        slice_step=1.0,
                        flip_normals=False,
                        scale=meta.get("scale", [0.05, 0.05]),
                        tile_size=256,
                    )
                    ct_mib = round((int(ct.bytes_fetched) - b_ct_0) / (1024 ** 2), 2)
                    ct_chunks = int(ct.chunks_fetched) - c_ct_0
                finally:
                    if not args.reuse_zarr_cache:
                        ct.close()
                rend_s = round(time.perf_counter() - t_r, 2)

                rv3 = run_rv3_both(stack_17, val_ct, ckpt, wdir, seg)
                common = val_ct & rv3["forward"]["valid"] & rv3["reverse"]["valid"]
                pf, pr = rv3["forward"]["prob"], rv3["reverse"]["prob"]
                st_f = summarize_prob_map(pf, common, voxel_um=args.voxel_um)
                st_r = summarize_prob_map(pr, common, voxel_um=args.voxel_um)
                comp = compute_direction_comparison(pf, pr, common)

                if not args.skip_visuals:
                    save_visual_panel(
                        seg_out / "visual_panel.png",
                        f"PHerc0125/{seg}",
                        stack_17[8],
                        common,
                        pf,
                        pr,
                        voxel_um=args.voxel_um,
                    )

                summary = {
                    "scroll": "PHerc0125",
                    "mesh_tag": seg,
                    "gpu": gpu_name,
                    "voxel_um": float(args.voxel_um),
                    "valid_ct_cm2": st_f.get("valid_cm2", 0.0),
                    "geometry_snap_metrics": g_metrics,
                    "snap_seconds": snap_s,
                    "render_seconds": rend_s,
                    "ct_mib": ct_mib,
                    "ct_chunks": ct_chunks,
                    "m7_mib": m7_mib,
                    "m7_chunks": m7_chunks,
                    "reuse_zarr_cache": bool(args.reuse_zarr_cache),
                    "rv3_forward": {**st_f, "seconds": rv3["forward"]["seconds"], "batch_size": rv3["forward"]["batch_size"], "source": rv3["forward"]["source"]},
                    "rv3_reverse": {**st_r, "seconds": rv3["reverse"]["seconds"], "batch_size": rv3["reverse"]["batch_size"], "source": rv3["reverse"]["source"]},
                    "comparison": comp,
                    "total_seconds": round(time.perf_counter() - t_seg, 2),
                }
                sum_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                index_rows.append(summary)
                del stack_17, val_ct, pf, pr, rv3
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if shared_m7 is not None:
            shared_m7.close()
        if shared_ct is not None:
            shared_ct.close()

    idx = {
        "schema": "batch_c1_readerv3_colab_v1",
        "gpu": gpu_name,
        "reuse_zarr_cache": bool(args.reuse_zarr_cache),
        "meshes_completed": len(index_rows),
        "total_cm2": round(sum(r["valid_ct_cm2"] for r in index_rows), 4),
        "total_wall_seconds": round(time.perf_counter() - t_all, 2),
        "alerts": [r["mesh_tag"] for r in index_rows if r["comparison"]["hit_alert"]],
        "rows": index_rows,
    }
    (out_root / "batch_index.json").write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
    if args.pack_tar:
        tgz = out_root.parent / "out_batch.tar.gz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(out_root, arcname=out_root.name)
    return idx


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    idx = run_batch(args)
    print(
        f"CLOUD_RV3_COMPLETE meshes={idx['meshes_completed']} cm2={idx['total_cm2']} "
        f"alerts={idx['alerts']} wall={idx['total_wall_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
