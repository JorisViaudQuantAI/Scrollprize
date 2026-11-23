# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Bounded CPU-only real-CT validation of ``screening_fp_filter.build_crack_void_halo_mask`` void modes.

Streams small PUBLIC CT sub-windows into RAM (no ink detector, no ink prediction, no CT written to
disk; tifxyz geometry lives only in a temporary directory) and measures how much valid area the
crack/void halo mask removes under BOTH the default ``"robust_sigma"`` void rule and the opt-in
physical ``"physical"`` / ``"hybrid"`` rules (``dark_frac_of_median = 0.5``, fixed a priori):

* ``PHerc0306B`` published touched-up ``w074`` positive-control strip (intact woven papyrus), inside
  the published ``r=95:120, c=190:250`` window -> clean-papyrus retention (target ``>= 0.85``);
* ``PHerc0191`` public spiral fits ``w050`` and ``w030`` (delaminated), centred ``50x50`` mesh-grid
  windows -> crack/void halo fraction.

Each strip is rendered on TWO same-size sub-windows chosen from geometry only (never CT content):
``fewest_chunks`` (flattest, cheapest) and ``max_curvature`` (largest median second-difference of
the mesh, i.e. the most folded / wrinkled region, where delamination seams concentrate), so the
benchmark never samples only maximally flat sheet. Both windows of a strip share one in-RAM chunk
cache. One global byte budget (``STREAM_CAP_MIB``) reserves a full decoded chunk before every CT or
``m7`` fetch and hard-aborts past the cap; mesh geometry bytes are charged to it too.

Library defaults are reported as-is (never tuned on these strips); ``dark_sigma`` and
``dark_frac_of_median`` sweeps are reported for threshold rationale only. Writes ONE sanitized
aggregate JSON receipt (``--output``): no local paths, no images, no unread coordinates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import screen_scroll_rtx4060 as S  # noqa: E402
import screening_fp_filter as F  # noqa: E402

SCHEMA_VERSION = 2
MIB_BUDGET = 120.0
STREAM_CAP_MIB = 118.0
RETENTION_TARGET = 0.85
CHUNKS_PER_WINDOW = 7
WINDOWS_PER_STRIP = 2
RENDER_TILE = 32
WINDOW_FACTORS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2)
MIN_VALID_VERTEX_FRAC = 0.9
SIGMA_SWEEP = (3.0, 4.0, 5.0)
FRAC_SWEEP = (0.35, 0.5, 0.65)
MODES = F.VOID_MODES
CLEAN_ROLE, CRACK_ROLE = "intact_clean_papyrus", "delaminated_crack_void"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HF_0191 = (
    "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-spiral-fits/resolve/"
    "63c641ba1dc794073208c24f7f88fa59c4f77621/2026-09-15_PHerc0191_slice-11600-12400_0-patch_w90/meshes/fitted/"
)
CAVEATS = (
    "Streaming budget limits each sub-window to <= 7 CT chunks (~0.02-0.04 cm2); results are per-window "
    "aggregates, not whole-strip coverage.",
    "Sub-windows are chosen from mesh geometry only (fewest chunks / max curvature); presence of a dark seam "
    "in a window is measured afterwards (ct_dark_px_frac_below_half_median), never used for selection.",
    "clean_papyrus_retention_frac counts BOTH void halo and mesh-border erosion halo; void_halo_frac and "
    "void_detection_fired isolate the dark-void rule.",
    "Rendered pixels with zero CT intensity (masked-out air) are invalid, so fully empty gaps appear as "
    "border halo; the void rule can only fire on partial-volume dark seams.",
    "The physical rule is fixed a priori (dark_frac_of_median = 0.5: majority-air under linear partial-volume "
    "mixing); sweeps are reported, never used to tune.",
)
METHOD = (
    "CPU-only stream-render of the mid CT slice on geometry-selected sub-windows + build_crack_void_halo_mask "
    "under robust_sigma (default), physical and hybrid void modes; no ink detector run, no ink prediction read, "
    "zero persistent CT on disk"
)
_STRIP_0191 = {
    "role": CRACK_ROLE,
    "scroll": "PHerc0191",
    "ct_volume": f"{S3}/PHerc0191/volumes/20250821151635-9.362um-1.2m-113keV-masked.zarr",
    "mesh_source": "rodriguescarson/eligible-scroll-spiral-fits",
    "mesh_revision": "63c641ba1dc794073208c24f7f88fa59c4f77621",
    "outer_window": None,
    "outer_window_rule": "centred 50x50 mesh-grid window",
    "publish_window": False,
    "m7_zarr": f"{S3}/PHerc0191/representations/predictions/surfaces/"
    "20250821151635-surface-20260413222639-surface-m7-L0-th0.2.zarr",
    "snap_passes": 2,
    "num_slices": 5,
    "voxel_um": 9.362,
}
STRIPS = (
    {
        "name": "PHerc0306B_w074_positive_control",
        "role": CLEAN_ROLE,
        "scroll": "PHerc0306B",
        "ct_volume": f"{S3}/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr",
        "mesh_source": "SirProdigle/pherc0306b-first-letters",
        "mesh_revision": "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc",
        "mesh_id": "PHerc0306B_20250521133212_w074own_touchup_V5Rc",
        "mesh_url": (
            "https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/"
            "feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/"
            "PHerc0306B_20250521133212_w074own_touchup_V5Rc"
        ),
        "outer_window": (95, 120, 190, 250),
        "outer_window_rule": "published positive-control window r=95:120, c=190:250",
        "publish_window": True,
        "m7_zarr": None,
        "snap_passes": 0,
        "num_slices": 5,
        "voxel_um": 8.64,
    },
    {**_STRIP_0191, "name": "PHerc0191_w050_delaminated", "mesh_id": "w050", "mesh_url": HF_0191 + "w050"},
    {**_STRIP_0191, "name": "PHerc0191_w030_delaminated", "mesh_id": "w030", "mesh_url": HF_0191 + "w030"},
)
PUBLIC_STRIP_KEYS = (
    "name", "role", "scroll", "ct_volume", "mesh_source", "mesh_revision", "mesh_id", "outer_window_rule",
    "snap_passes", "num_slices", "voxel_um",
)


class BudgetExceeded(RuntimeError):
    """Raised before a fetch that would exceed the streaming cap."""


class StreamBudget:
    """Thread-safe global byte budget: reserve an upper bound before a fetch, settle to actual bytes after."""

    def __init__(self, max_mib: float) -> None:
        self.cap = int(max_mib * 1024 ** 2)
        self.used = 0
        self._lock = threading.Lock()

    def reserve(self, nbytes: int) -> None:
        with self._lock:
            if self.used + nbytes > self.cap:
                raise BudgetExceeded(f"fetch would exceed the {self.cap / 1024 ** 2:.0f} MiB streaming cap")
            self.used += nbytes

    def settle(self, reserved: int, actual: int) -> None:
        with self._lock:
            self.used += actual - reserved

    def charge(self, nbytes: int) -> None:
        """Account bytes already fetched outside a reader (mesh geometry); raises if the cap is passed."""
        with self._lock:
            self.used += int(nbytes)
            if self.used > self.cap:
                raise BudgetExceeded(f"streamed bytes exceed the {self.cap / 1024 ** 2:.0f} MiB streaming cap")


class BudgetedZarrV2(S.SSR.RemoteZarrV2):
    """``RemoteZarrV2`` that reserves one full decoded chunk per fetch against a (shared) ``StreamBudget``."""

    def __init__(self, *args, max_mib: float | None = None, budget: StreamBudget | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if budget is None:
            if max_mib is None:
                raise ValueError("supply max_mib or budget")
            budget = StreamBudget(max_mib)
        self.budget = budget

    def _get_bytes(self, key, *, allow_404=True):
        raw = super()._get_bytes(key, allow_404=allow_404)
        self._tls.last_len = 0 if raw is None else len(raw)
        return raw

    def _load_chunk(self, idx):
        need = self._expected_chunk_bytes
        self.budget.reserve(need)
        self._tls.last_len = 0
        try:
            arr = super()._load_chunk(idx)
        except BaseException:
            self.budget.settle(need, getattr(self._tls, "last_len", 0))
            raise
        self.budget.settle(need, getattr(self._tls, "last_len", 0))
        return arr


class ChunkRecorder(S.SSR.RemoteZarrV2):
    """Payload-free reader: records requested chunk indices and returns fill (fetches metadata only)."""

    def _load_chunk(self, idx):
        self.seen.add(idx)
        return None

    def footprint(self, x, y, z, num_slices: int, scale) -> int:
        """Exact number of CT chunks ``render_surface_stack`` would fetch for this geometry."""
        self.seen, self.cache, self.cache_bytes = set(), type(self.cache)(), 0
        S.SSR.render_surface_stack(x, y, z, self, num_slices=num_slices, scale=scale, tile_size=RENDER_TILE)
        return len(self.seen)


def centered_crop(shape: tuple[int, int], size: int = 50) -> tuple[int, int, int, int]:
    """Return the ``r0, r1, c0, c1`` centred ``size x size`` window of a mesh grid."""
    h, w = shape
    if h < size or w < size:
        raise ValueError(f"mesh grid {shape} smaller than {size}x{size}")
    r0, c0 = (h - size) // 2, (w - size) // 2
    return r0, r0 + size, c0, c0 + size


def geometry_roughness(x, y, z, valid) -> float:
    """Median norm (voxels) of the mesh grid's second differences, pooled over both grid axes on fully
    valid stencils (0 if none). Pooling means a singly-curved fold scores half its along-fold value when
    both axes contribute equally; it is only used to rank same-size candidate windows."""
    p = np.stack([x, y, z], axis=-1).astype(np.float64)
    v = np.asarray(valid, dtype=bool)
    vals = []
    for ax in (0, 1):
        n = p.shape[ax]
        if n < 3:
            continue
        a, b, c = (np.take(p, range(k, n - 2 + k), axis=ax) for k in (0, 1, 2))
        va, vb, vc = (np.take(v, range(k, n - 2 + k), axis=ax) for k in (0, 1, 2))
        ok = va & vb & vc
        vals.append(np.linalg.norm(a - 2 * b + c, axis=-1)[ok])
    allv = np.concatenate(vals) if vals else np.zeros(0)
    return float(np.median(allv)) if allv.size else 0.0


def _overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def select_subwindows(x, y, z, valid, outer, footprint, chunk_cap: int = CHUNKS_PER_WINDOW) -> tuple[list[dict], float]:
    """Geometry-only choice of up to ``WINDOWS_PER_STRIP`` same-size sub-windows of ``outer``.

    Per shrink factor (largest first), candidates at quarter-window strides with at least
    ``MIN_VALID_VERTEX_FRAC`` valid vertices and ``footprint(x, y, z) <= chunk_cap`` CT chunks are
    collected. ``fewest_chunks`` is the cheapest (then most valid) candidate; ``max_curvature`` is the
    candidate not overlapping it with the largest :func:`geometry_roughness`. The first factor yielding
    both wins; if none does, the largest factor with any fit returns ``fewest_chunks`` alone.
    Returns ``([{"rule", "window", "dry_run_ct_chunks", "roughness_vx", "valid_vertex_frac"}], factor)``.
    """
    r0, r1, c0, c1 = outer
    fallback = None
    for f in WINDOW_FACTORS:
        h, w = max(4, round((r1 - r0) * f)), max(4, round((c1 - c0) * f))
        fits = []
        for rr in range(r0, r1 - h + 1, max(1, h // 4)):
            for cc in range(c0, c1 - w + 1, max(1, w // 4)):
                sl = (slice(rr, rr + h), slice(cc, cc + w))
                vfrac = float(valid[sl].mean())
                if vfrac < MIN_VALID_VERTEX_FRAC:
                    continue
                n = footprint(x[sl], y[sl], z[sl])
                if n <= chunk_cap:
                    fits.append({
                        "window": (rr, rr + h, cc, cc + w),
                        "dry_run_ct_chunks": int(n),
                        "roughness_vx": round(geometry_roughness(x[sl], y[sl], z[sl], valid[sl]), 4),
                        "valid_vertex_frac": round(vfrac, 4),
                    })
        if not fits:
            continue
        flat = min(fits, key=lambda c: (c["dry_run_ct_chunks"], -c["valid_vertex_frac"]))
        flat = {"rule": "fewest_chunks", **flat}
        rest = [c for c in fits if not _overlaps(c["window"], flat["window"])]
        if rest and WINDOWS_PER_STRIP > 1:
            rough = max(rest, key=lambda c: (c["roughness_vx"], -c["dry_run_ct_chunks"]))
            return [flat, {"rule": "max_curvature", **rough}], f
        if fallback is None:
            fallback = ([flat], f)
    if fallback is None:
        raise RuntimeError(f"no sub-window of {outer} fits {chunk_cap} CT chunks")
    return fallback


def _mode_stats(ct_mid, valid, n_valid, **kw) -> dict:
    frac = lambda m: round(int(m.sum()) / n_valid, 6)  # noqa: E731
    halo = F.build_crack_void_halo_mask(ct_mid, valid, **kw)
    void_only = F.build_crack_void_halo_mask(ct_mid, valid, border_halo_radius=0, **kw)
    return {
        "clean_papyrus_retention_frac": round(1.0 - frac(halo), 6),
        "crack_halo_frac": frac(halo),
        "void_halo_frac": frac(void_only),
        "border_halo_only_frac": frac(halo & ~void_only),
        "void_detection_fired": bool(void_only.any()),
    }


def summarize_halo(ct_mid: np.ndarray, valid: np.ndarray, voxel_um: float) -> dict:
    """Aggregate crack/void halo statistics per void mode for one mid CT slice (no coordinates, no images)."""
    valid = np.asarray(valid, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("no valid CT pixels rendered")
    vals = np.asarray(ct_mid, dtype=np.float32)[valid]
    med = float(np.median(vals))
    robust_sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    smooth_vals = F._masked_smooth(np.asarray(ct_mid, dtype=np.float32), valid, 1.0)[valid]
    modes = {}
    for mode in MODES:
        entry = {"threshold": round(F.void_threshold(smooth_vals, void_mode=mode), 3)}
        entry.update(_mode_stats(ct_mid, valid, n_valid, void_mode=mode))
        modes[mode] = entry
    return {
        "valid_px": n_valid,
        "valid_cm2": round(n_valid * (voxel_um / 10000.0) ** 2, 6),
        "valid_frame_frac": round(n_valid / valid.size, 6),
        "ct_median": round(med, 3),
        "ct_robust_sigma": round(robust_sigma, 3),
        "ct_p01": round(float(np.percentile(vals, 1)), 3),
        "ct_p05": round(float(np.percentile(vals, 5)), 3),
        "ct_dark_px_frac_below_half_median": round(float((vals < 0.5 * med).mean()), 6),
        "modes": modes,
        "retention_by_dark_sigma": {
            f"{s:g}": _mode_stats(ct_mid, valid, n_valid, dark_sigma=s)["clean_papyrus_retention_frac"]
            for s in SIGMA_SWEEP
        },
        "physical_by_dark_frac": {
            f"{fr:g}": {
                k: v for k, v in _mode_stats(
                    ct_mid, valid, n_valid, void_mode="physical", dark_frac_of_median=fr
                ).items() if k in ("clean_papyrus_retention_frac", "void_halo_frac")
            }
            for fr in FRAC_SWEEP
        },
    }


def run_strip(strip: dict, work_root: Path, cache_mb: int, budget: StreamBudget) -> dict:
    """Stream one public strip's sub-windows into RAM, measure the halo mask, and return its sanitized entry."""
    used0 = budget.used
    with tempfile.TemporaryDirectory(prefix="fpbench-", dir=str(work_root)) as tmp:
        src = Path(tmp) / "src"
        S.fetch_mesh_dir(strip["mesh_url"], src)
        mesh_bytes = sum(p.stat().st_size for p in src.iterdir())
        budget.charge(mesh_bytes)
        x, y, z, valid, meta = S.SSR.read_tifxyz(src)
    scale = meta.get("scale", [0.05, 0.05])
    recorder = ChunkRecorder(strip["ct_volume"], cache_mb=0, workers=1)
    choices, factor = select_subwindows(
        x, y, z, valid,
        strip["outer_window"] or centered_crop(valid.shape),
        lambda a, b, c: recorder.footprint(a, b, c, strip["num_slices"], scale),
    )
    recorder.close()
    t0 = time.perf_counter()
    ct_reader = BudgetedZarrV2(strip["ct_volume"], cache_mb=cache_mb, workers=8, budget=budget)
    m7_reader = (
        BudgetedZarrV2(strip["m7_zarr"], cache_mb=cache_mb, workers=8, budget=budget)
        if strip["snap_passes"] > 0 else None
    )
    windows = []
    try:
        for ch in choices:
            r0, r1, c0, c1 = ch["window"]
            wx, wy, wz = (a[r0:r1, c0:c1] for a in (x, y, z))
            snap = None
            if m7_reader is not None:
                wx, wy, wz, g = S.SSR.snap_tifxyz_to_m7(wx, wy, wz, m7_reader, passes=strip["snap_passes"])
                snap = {k: round(float(g[k]), 4) for k in ("on_sheet_raw", "on_sheet_after_snap", "median_disp_vx")}
            stack, valid_ct = S.SSR.render_surface_stack(
                wx, wy, wz, ct_reader, num_slices=strip["num_slices"], scale=scale, tile_size=RENDER_TILE
            )
            stats = summarize_halo(stack[(strip["num_slices"] - 1) // 2], valid_ct, strip["voxel_um"])
            del stack
            windows.append({
                "rule": ch["rule"],
                "grid_shape": [r1 - r0, c1 - c0],
                "window": list(ch["window"]) if strip["publish_window"] else None,
                "dry_run_ct_chunks": ch["dry_run_ct_chunks"],
                "roughness_vx": ch["roughness_vx"],
                "m7_snap": snap,
                "halo": stats,
            })
        ct_mib, ct_chunks = ct_reader.bytes_fetched / 1024 ** 2, ct_reader.chunks_fetched
        m7_mib = 0.0 if m7_reader is None else m7_reader.bytes_fetched / 1024 ** 2
    finally:
        ct_reader.close()
        if m7_reader is not None:
            m7_reader.close()
    entry = {k: strip[k] for k in PUBLIC_STRIP_KEYS}
    entry["factor_of_outer"] = factor
    entry["streamed_mib"] = {"ct": round(ct_mib, 2), "m7": round(m7_mib, 2), "mesh_geometry": round(mesh_bytes / 1024 ** 2, 2)}
    entry["streamed_mib"]["total"] = round(sum(entry["streamed_mib"].values()), 2)
    entry["budget_accounted_mib"] = round((budget.used - used0) / 1024 ** 2, 2)
    entry["ct_chunks_fetched"] = int(ct_chunks)
    entry["seconds"] = round(time.perf_counter() - t0, 2)
    entry["windows"] = windows
    return entry


def build_receipt(entries: list[dict]) -> dict:
    """Assemble the public receipt and per-mode verdicts from per-strip entries."""
    halos = {role: [w["halo"] for e in entries if e["role"] == role for w in e["windows"]] for role in (CLEAN_ROLE, CRACK_ROLE)}
    if not halos[CLEAN_ROLE] or not halos[CRACK_ROLE]:
        raise ValueError("need at least one clean and one delaminated window")
    total_mib = round(sum(e["streamed_mib"]["total"] for e in entries), 2)
    by_mode = {}
    for mode in MODES:
        clean = [h["modes"][mode] for h in halos[CLEAN_ROLE]]
        crack = [h["modes"][mode] for h in halos[CRACK_ROLE]]
        min_ret = min(c["clean_papyrus_retention_frac"] for c in clean)
        mean = lambda xs, k: round(float(np.mean([x[k] for x in xs])), 6)  # noqa: E731
        by_mode[mode] = {
            "clean_retention_min": min_ret,
            "clean_retention_mean": mean(clean, "clean_papyrus_retention_frac"),
            "clean_retention_meets_target": min_ret >= RETENTION_TARGET,
            "clean_void_halo_frac_mean": mean(clean, "void_halo_frac"),
            "clean_void_detection_fired_windows": sum(c["void_detection_fired"] for c in clean),
            "delaminated_crack_halo_frac_mean": mean(crack, "crack_halo_frac"),
            "delaminated_void_halo_frac_mean": mean(crack, "void_halo_frac"),
            "delaminated_void_halo_frac_max": max(c["void_halo_frac"] for c in crack),
            "delaminated_void_detection_fired_windows": sum(c["void_detection_fired"] for c in crack),
            "delaminated_minus_clean_void_halo_frac": round(
                mean(crack, "void_halo_frac") - mean(clean, "void_halo_frac"), 6
            ),
        }
    interpretation = []
    for mode, vm in by_mode.items():
        sep = vm["delaminated_minus_clean_void_halo_frac"]
        interpretation.append(
            f"{mode}: clean retention min {vm['clean_retention_min']:.3f} "
            f"({'meets' if vm['clean_retention_meets_target'] else 'FAILS'} >= {RETENTION_TARGET}); void fired on "
            f"{vm['delaminated_void_detection_fired_windows']}/{len(halos[CRACK_ROLE])} delaminated and "
            f"{vm['clean_void_detection_fired_windows']}/{len(halos[CLEAN_ROLE])} clean windows; delaminated minus "
            f"clean void halo {sep:+.3f} ({'separates' if sep > 0 and vm['clean_retention_meets_target'] else 'no usable separation'})"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "filter_parameters": (
            "screening_fp_filter.build_crack_void_halo_mask defaults per void_mode "
            f"(dark_sigma=4, dark_frac_of_median={F.DEFAULT_DARK_FRAC_OF_MEDIAN}); untuned on these strips"
        ),
        "mib_budget": MIB_BUDGET,
        "stream_cap_mib": STREAM_CAP_MIB,
        "chunks_per_window": CHUNKS_PER_WINDOW,
        "render_tile": RENDER_TILE,
        "total_streamed_mib": total_mib,
        "within_budget": total_mib < MIB_BUDGET,
        "n_windows": {role: len(h) for role, h in halos.items()},
        "strips": entries,
        "caveats": list(CAVEATS),
        "verdict": {
            "clean_retention_target": RETENTION_TARGET,
            "default_mode": "robust_sigma",
            "by_mode": by_mode,
            "interpretation": interpretation,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=HERE / "evidence" / "fp_filter_real_ct_benchmark.json")
    ap.add_argument("--cache-mb", type=int, default=128)
    ap.add_argument("--rebuild-from", type=Path, default=None,
                    help="re-assemble the receipt from a previous receipt's strip entries (no streaming)")
    args = ap.parse_args()
    if args.rebuild_from is not None:
        receipt = build_receipt(json.loads(args.rebuild_from.read_text(encoding="utf-8"))["strips"])
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt["verdict"], indent=2))
        return
    budget = StreamBudget(STREAM_CAP_MIB)
    entries = []
    with tempfile.TemporaryDirectory(prefix="fpbench-root-") as root:
        for strip in STRIPS:
            entries.append(run_strip(strip, Path(root), args.cache_mb, budget))
            print(f"{strip['name']}: {entries[-1]['streamed_mib']} budget_used={budget.used / 1024 ** 2:.2f} MiB",
                  flush=True)
    receipt = build_receipt(entries)
    if not receipt["within_budget"]:
        raise RuntimeError(f"streamed {receipt['total_streamed_mib']} MiB >= {MIB_BUDGET} MiB budget")
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["verdict"], indent=2))


if __name__ == "__main__":
    main()
