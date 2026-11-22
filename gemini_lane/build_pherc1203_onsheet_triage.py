# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Merge private CPU triage ``summary.json`` files into ONE sanitized public JSON.

Keeps only geometry / ``m7``-snap / ``onsheet_score`` / ``valid_ct_cm2`` fields. Never copies local
paths, detector maps, hotspots, images or artifact names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GEOMETRY_KEYS = (
    "valid_vertex_fraction",
    "valid_vertex_count",
    "total_vertex_count",
    "on_sheet_raw",
    "on_sheet_after_snap",
    "median_disp_vx",
    "p90_disp_vx",
    "center_vs_outer_disp_diff",
)
ONSHEET_KEYS = (
    "tiles",
    "f_parallel",
    "f_drift",
    "f_still",
    "response_median",
    "shift_median_confident",
    "layers",
    "mid",
    "dk",
)


def sanitize_sheet(summary: dict) -> dict:
    """Project one private screener summary onto the public allowlist."""
    geom = summary["geometry_snap_metrics"]
    coh = summary["onsheet_coherence"]
    sheet = {
        "mesh_id": summary["mesh_tag"],
        "uv_crop": summary["uv_crop"],
        **{k: geom[k] for k in GEOMETRY_KEYS},
        "onsheet_score": {k: coh[k] for k in ONSHEET_KEYS},
        "valid_ct_cm2": summary["valid_ct_cm2"],
    }
    return sheet


def build_public_triage(summaries: list[dict]) -> dict:
    """Rank sanitized sheets by ``f_parallel`` (desc) and add per-metric rankings."""
    if not summaries:
        raise ValueError("no summaries to merge")
    first = summaries[0]
    for s in summaries:
        for key in ("scroll", "voxel_um", "num_slices", "snap_passes"):
            if s[key] != first[key]:
                raise ValueError(f"inconsistent {key}: {s[key]!r} != {first[key]!r}")
    sheets = sorted(
        (sanitize_sheet(s) for s in summaries),
        key=lambda sh: (-sh["onsheet_score"]["f_parallel"], -sh["on_sheet_after_snap"], sh["mesh_id"]),
    )
    for i, sheet in enumerate(sheets, start=1):
        sheet["rank"] = i
    sheets = [{"rank": sh["rank"], **{k: v for k, v in sh.items() if k != "rank"}} for sh in sheets]

    def order(key) -> list[str]:
        return [sh["mesh_id"] for sh in sorted(sheets, key=key)]

    return {
        "scroll": first["scroll"],
        "voxel_um": first["voxel_um"],
        "num_slices": first["num_slices"],
        "snap_passes": first["snap_passes"],
        "method": "CPU-only m7 snap + onsheet_score; no ink detector run",
        "total_valid_ct_cm2": round(sum(sh["valid_ct_cm2"] for sh in sheets), 4),
        "sheets": sheets,
        "rankings": {
            "by_onsheet_f_parallel": order(lambda sh: -sh["onsheet_score"]["f_parallel"]),
            "by_on_sheet_after_snap": order(lambda sh: -sh["on_sheet_after_snap"]),
            "by_valid_ct_cm2": order(lambda sh: -sh["valid_ct_cm2"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--private-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.private_dir.glob("PHerc1203_auto_grown_*/summary.json"))
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    args.output.write_text(json.dumps(build_public_triage(summaries), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output.name} from {len(summaries)} summaries")


if __name__ == "__main__":
    main()
