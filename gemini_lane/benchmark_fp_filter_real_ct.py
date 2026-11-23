# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Bounded CPU-only real-CT validation of ``screening_fp_filter.build_crack_void_halo_mask``.

Streams two small PUBLIC CT strips into RAM (no ink detector, no ink prediction, no CT written to
disk; the tifxyz geometry lives only in a temporary directory) and measures how much valid area the
crack/void halo mask removes:

* ``PHerc0306B`` published touched-up ``w074`` positive-control strip (intact woven papyrus) ->
  clean-papyrus retention fraction (target ``>= 0.85``);
* ``PHerc0191`` public spiral-fit ``w050`` (delaminated), centred ``50x50`` mesh-grid window ->
  crack/void halo fraction.

Both CT volumes are uncompressed ``128^3`` uint8 Zarr v2 (2 MiB / chunk), so the full outer windows
touch ~90-140 chunks each. To stay strictly under the streaming budget, each strip is rendered on the
largest same-aspect sub-window of its outer window whose exact CT chunk footprint (a dry run of the
renderer against a payload-free reader: geometry only, never CT content) fits ``CHUNKS_PER_STRIP``;
a byte-capped reader then hard-aborts any fetch past ``CT_MIB_CAP_PER_STRIP``.

The default filter parameters are reported as-is (never tuned on these strips); a fixed
``dark_sigma`` sweep is reported alongside for threshold rationale only. Writes ONE sanitized
aggregate JSON receipt (``--output``).
"""
from __future__ import annotations

import argparse
import functools
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

SCHEMA_VERSION = 1
MIB_BUDGET = 80.0
RETENTION_TARGET = 0.85
CHUNKS_PER_STRIP = 16
CT_MIB_CAP_PER_STRIP = 36.0
RENDER_TILE = 32
WINDOW_FACTORS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25)
MIN_VALID_VERTEX_FRAC = 0.9
SIGMA_SWEEP = (3.0, 4.0, 5.0)
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
CAVEATS = (
    "Streaming budget limits each strip to a ~0.06-0.07 cm2 sub-window; fewest-chunk selection favours "
    "flat geometry and can under-sample delamination.",
    "clean_papyrus_retention_frac counts BOTH void halo and mesh-border erosion halo; void_detection_fired "
    "separates them.",
    "raw_dark_threshold_est (median - 4 * robust sigma, unsmoothed) below the CT floor means the default "
    "robust rule cannot flag voids on that strip; retention there is not evidence of weave-vs-void "
    "discrimination.",
    "Parameters are the library defaults; the dark_sigma sweep is reported, never used to tune.",
)
METHOD = (
    "CPU-only stream-render of the mid CT slice + build_crack_void_halo_mask (default parameters); "
    "no ink detector run, no ink prediction read, zero persistent CT on disk"
)
STRIPS = (
    {
        "name": "PHerc0306B_w074_positive_control",
        "role": "intact_clean_papyrus",
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
    {
        "name": "PHerc0191_w050_delaminated",
        "role": "delaminated_crack_void",
        "scroll": "PHerc0191",
        "ct_volume": f"{S3}/PHerc0191/volumes/20250821151635-9.362um-1.2m-113keV-masked.zarr",
        "mesh_source": "rodriguescarson/eligible-scroll-spiral-fits",
        "mesh_revision": "63c641ba1dc794073208c24f7f88fa59c4f77621",
        "mesh_id": "w050",
        "mesh_url": (
            "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-spiral-fits/resolve/"
            "63c641ba1dc794073208c24f7f88fa59c4f77621/"
            "2026-09-15_PHerc0191_slice-11600-12400_0-patch_w90/meshes/fitted/w050"
        ),
        "outer_window": None,
        "outer_window_rule": "centred 50x50 mesh-grid window",
        "publish_window": False,
        "m7_zarr": f"{S3}/PHerc0191/representations/predictions/surfaces/"
        "20250821151635-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "snap_passes": 2,
        "num_slices": 5,
        "voxel_um": 9.362,
    },
)
PUBLIC_STRIP_KEYS = (
    "name", "role", "scroll", "ct_volume", "mesh_source", "mesh_revision", "mesh_id", "outer_window_rule",
    "snap_passes", "num_slices", "voxel_um",
)


class BudgetExceeded(RuntimeError):
    """Raised before a CT chunk fetch that would exceed the streaming cap."""


class BudgetedZarrV2(S.SSR.RemoteZarrV2):
    """``RemoteZarrV2`` that reserves one full decoded chunk per fetch and refuses fetches past ``max_mib``."""

    def __init__(self, *args, max_mib: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cap = int(max_mib * 1024 ** 2)
        self._reserved = 0
        self._budget_lock = threading.Lock()

    def _load_chunk(self, idx):
        with self._budget_lock:
            if self._reserved + self._expected_chunk_bytes > self._cap:
                raise BudgetExceeded(f"CT fetch would exceed the {self._cap / 1024 ** 2:.0f} MiB cap")
            self._reserved += self._expected_chunk_bytes
        return super()._load_chunk(idx)


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


def select_subwindow(x, y, z, valid, outer, footprint) -> tuple[tuple[int, int, int, int], float, int]:
    """Largest same-aspect sub-window of ``outer`` whose CT chunk ``footprint`` fits ``CHUNKS_PER_STRIP``.

    Geometry-only: per shrink factor, candidates at quarter-window strides with at least
    ``MIN_VALID_VERTEX_FRAC`` valid vertices are ranked by fewest chunks, then most valid vertices.
    ``footprint(x, y, z)`` returns a candidate's chunk count. Returns ``(window, factor, n_chunks)``.
    """
    r0, r1, c0, c1 = outer
    for f in WINDOW_FACTORS:
        h, w = max(4, round((r1 - r0) * f)), max(4, round((c1 - c0) * f))
        best = None
        for rr in range(r0, r1 - h + 1, max(1, h // 4)):
            for cc in range(c0, c1 - w + 1, max(1, w // 4)):
                sl = (slice(rr, rr + h), slice(cc, cc + w))
                vfrac = float(valid[sl].mean())
                if vfrac < MIN_VALID_VERTEX_FRAC:
                    continue
                n = footprint(x[sl], y[sl], z[sl])
                if n <= CHUNKS_PER_STRIP and (best is None or (n, -vfrac) < best[0]):
                    best = ((n, -vfrac), (rr, rr + h, cc, cc + w))
        if best is not None:
            return best[1], f, best[0][0]
    raise RuntimeError(f"no sub-window of {outer} fits {CHUNKS_PER_STRIP} CT chunks")


def summarize_halo(ct_mid: np.ndarray, valid: np.ndarray, voxel_um: float) -> dict:
    """Aggregate crack/void halo statistics for one mid CT slice (no coordinates, no images)."""
    valid = np.asarray(valid, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("no valid CT pixels rendered")
    frac = lambda m: round(int(m.sum()) / n_valid, 6)  # noqa: E731
    halo = F.build_crack_void_halo_mask(ct_mid, valid)
    void_only = F.build_crack_void_halo_mask(ct_mid, valid, border_halo_radius=0)
    sweep = {
        f"{s:g}": round(1.0 - frac(F.build_crack_void_halo_mask(ct_mid, valid, dark_sigma=s)), 6)
        for s in SIGMA_SWEEP
    }
    vals = np.asarray(ct_mid, dtype=np.float32)[valid]
    med = float(np.median(vals))
    robust_sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    return {
        "valid_px": n_valid,
        "valid_cm2": round(n_valid * (voxel_um / 10000.0) ** 2, 6),
        "valid_frame_frac": round(n_valid / valid.size, 6),
        "ct_median": round(med, 3),
        "ct_robust_sigma": round(robust_sigma, 3),
        "ct_p01": round(float(np.percentile(vals, 1)), 3),
        "raw_dark_threshold_est": round(med - 4.0 * robust_sigma, 3),
        "clean_papyrus_retention_frac": round(1.0 - frac(halo), 6),
        "crack_halo_frac": frac(halo),
        "void_halo_frac": frac(void_only),
        "border_halo_only_frac": frac(halo & ~void_only),
        "void_detection_fired": bool(void_only.any()),
        "retention_by_dark_sigma": sweep,
    }


def run_strip(strip: dict, work_root: Path, cache_mb: int) -> dict:
    """Stream one public strip into RAM, measure the halo mask, and return its sanitized entry."""
    with tempfile.TemporaryDirectory(prefix="fpbench-", dir=str(work_root)) as tmp:
        work = Path(tmp)
        src = work / "src"
        S.fetch_mesh_dir(strip["mesh_url"], src)
        mesh_mib = sum(p.stat().st_size for p in src.iterdir()) / 1024 ** 2
        x, y, z, valid, meta = S.SSR.read_tifxyz(src)
        scale = meta.get("scale", [0.05, 0.05])
        recorder = ChunkRecorder(strip["ct_volume"], cache_mb=0, workers=1)
        window, factor, n_chunks = select_subwindow(
            x, y, z, valid,
            strip["outer_window"] or centered_crop(valid.shape),
            lambda a, b, c: recorder.footprint(a, b, c, strip["num_slices"], scale),
        )
        t0 = time.perf_counter()
        rendered = S.render_mesh_ct_stack(
            str(src),
            work,
            ct_zarr_url=strip["ct_volume"],
            m7_zarr_url=strip["m7_zarr"],
            snap_passes=strip["snap_passes"],
            num_slices=strip["num_slices"],
            uv_crop=window,
            cache_mb=cache_mb,
            tile_size=RENDER_TILE,
            ct_reader_cls=functools.partial(BudgetedZarrV2, max_mib=CT_MIB_CAP_PER_STRIP),
        )
    stack = rendered.pop("stack_lhw")
    stats = summarize_halo(stack[(strip["num_slices"] - 1) // 2], rendered.pop("valid_ct"), strip["voxel_um"])
    del stack
    entry = {k: strip[k] for k in PUBLIC_STRIP_KEYS}
    entry["subwindow"] = {
        "factor_of_outer": factor,
        "grid_shape": [window[1] - window[0], window[3] - window[2]],
        "dry_run_ct_chunks": n_chunks,
        "window": list(window) if strip["publish_window"] else None,
    }
    g = rendered["geom_metrics"]
    entry["m7_snap"] = (
        None if g is None
        else {k: round(float(g[k]), 4) for k in ("on_sheet_raw", "on_sheet_after_snap", "median_disp_vx")}
    )
    entry["streamed_mib"] = {
        "ct": round(rendered["ct_bytes"] / 1024 ** 2, 2),
        "m7": round(rendered["m7_bytes"] / 1024 ** 2, 2),
        "mesh_geometry": round(mesh_mib, 2),
    }
    entry["streamed_mib"]["total"] = round(sum(entry["streamed_mib"].values()), 2)
    entry["ct_chunks_fetched"] = rendered["ct_chunks"]
    entry["seconds"] = round(time.perf_counter() - t0, 2)
    entry["halo"] = stats
    return entry


def build_receipt(entries: list[dict]) -> dict:
    """Assemble the public receipt and its verdicts from per-strip entries."""
    by_role = {e["role"]: e["halo"] for e in entries}
    clean, cracked = by_role["intact_clean_papyrus"], by_role["delaminated_crack_void"]
    total_mib = round(sum(e["streamed_mib"]["total"] for e in entries), 2)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "filter_parameters": "screening_fp_filter.build_crack_void_halo_mask defaults (untuned on these strips)",
        "mib_budget": MIB_BUDGET,
        "chunks_per_strip": CHUNKS_PER_STRIP,
        "render_tile": RENDER_TILE,
        "total_streamed_mib": total_mib,
        "within_budget": total_mib < MIB_BUDGET,
        "strips": entries,
        "caveats": list(CAVEATS),
        "verdict": {
            "clean_retention_target": RETENTION_TARGET,
            "clean_retention_observed": clean["clean_papyrus_retention_frac"],
            "clean_retention_meets_target": clean["clean_papyrus_retention_frac"] >= RETENTION_TARGET,
            "delaminated_crack_halo_frac": cracked["crack_halo_frac"],
            "delaminated_void_detection_fired": cracked["void_detection_fired"],
            "clean_void_detection_fired": clean["void_detection_fired"],
            "delaminated_minus_clean_halo_frac": round(cracked["crack_halo_frac"] - clean["crack_halo_frac"], 6),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=HERE / "evidence" / "fp_filter_real_ct_benchmark.json")
    ap.add_argument("--cache-mb", type=int, default=128)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="fpbench-root-") as root:
        entries = [run_strip(strip, Path(root), args.cache_mb) for strip in STRIPS]
    receipt = build_receipt(entries)
    if not receipt["within_budget"]:
        raise RuntimeError(f"streamed {receipt['total_streamed_mib']} MiB >= {MIB_BUDGET} MiB budget")
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["verdict"], indent=2))


if __name__ == "__main__":
    main()
