# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Merge private ``PHerc0125`` / ``PHerc0191`` CPU triage summaries into ONE sanitized public JSON.

Reuses the ``PHerc1203`` allowlist (geometry / ``m7``-snap / ``onsheet_score`` / ``valid_ct_cm2``). Never copies
local paths, detector maps, hotspots, images, ink predictions or artifact names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_pherc1203_onsheet_triage import build_public_triage

SCHEMA_VERSION = 1
METHOD = "CPU-only m7 2x snap + onsheet_score (5 slices); no ink detector run, no ink prediction read"

SOURCES = {
    "PHerc0125": {
        "mesh_source": "pscamillo/vesuvius-eligible-meshes",
        "mesh_revision": "66000516c97e043b696ce874f145e29638109c91",
        "mesh_selection": "Tier-1 untouched windings; z14144_w045 skipped (already screened)",
        "mesh_license_note": None,
    },
    "PHerc0191": {
        "mesh_source": "rodriguescarson/eligible-scroll-spiral-fits",
        "mesh_revision": "63c641ba1dc794073208c24f7f88fa59c4f77621",
        "mesh_selection": "run 2026-09-15 slice-11600-12400 (w90), fitted windings w020/w030/w040/w050",
        "mesh_license_note": "CC BY-NC 4.0 (Carson Rodrigues); author publishes these meshes as unvalidated geometry",
    },
}
EXPECTED_MESHES = {
    "PHerc0125": ("z10544_w035", "z14144_w035", "z8736_w035", "z6944_w030", "z15936_w030"),
    "PHerc0191": ("w020", "w030", "w040", "w050"),
}
SUMMARY_GLOB = "*/summary.json"
TOP_LEVEL_KEYS = frozenset({"schema_version", "method", "scrolls"})
SCROLL_KEYS = frozenset(
    {
        "scroll", "voxel_um", "num_slices", "snap_passes", "method", "total_valid_ct_cm2", "sheets", "rankings",
        "source",
    }
)
SOURCE_KEYS = frozenset({"mesh_source", "mesh_revision", "mesh_selection", "mesh_license_note"})
SHEET_KEYS = frozenset(
    {
        "rank", "mesh_id", "uv_crop", "valid_vertex_fraction", "valid_vertex_count", "total_vertex_count",
        "on_sheet_raw", "on_sheet_after_snap", "median_disp_vx", "p90_disp_vx", "center_vs_outer_disp_diff",
        "onsheet_score", "valid_ct_cm2",
    }
)


def build_scroll_entry(scroll: str, summaries: list[dict]) -> dict:
    """Sanitize one scroll's private summaries and attach its pinned public provenance."""
    if scroll not in SOURCES:
        raise ValueError(f"unknown scroll {scroll!r}")
    wrong = [s["scroll"] for s in summaries if s["scroll"] != scroll]
    if wrong:
        raise ValueError(f"summaries for {wrong!r} passed as {scroll!r}")
    got = {s["mesh_tag"] for s in summaries}
    expected = set(EXPECTED_MESHES[scroll])
    if got != expected:
        raise ValueError(f"{scroll}: meshes {sorted(got)} != expected {sorted(expected)}")
    entry = build_public_triage(summaries)
    entry["source"] = dict(SOURCES[scroll])
    return entry


def build_combined(per_scroll: dict[str, list[dict]]) -> dict:
    """Return the single public JSON covering every requested scroll (stable scroll order)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scrolls": [build_scroll_entry(scroll, per_scroll[scroll]) for scroll in sorted(per_scroll)],
    }


def load_summaries(private_dir: Path) -> list[dict]:
    """Read per-mesh ``summary.json`` files (never images or other artifacts)."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(private_dir.glob(SUMMARY_GLOB))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--private-dir-0125", type=Path, required=True)
    ap.add_argument("--private-dir-0191", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    per_scroll = {
        "PHerc0125": load_summaries(args.private_dir_0125),
        "PHerc0191": load_summaries(args.private_dir_0191),
    }
    data = build_combined(per_scroll)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output.name}: " + ", ".join(f"{k}={len(v)}" for k, v in per_scroll.items()))


if __name__ == "__main__":
    main()
