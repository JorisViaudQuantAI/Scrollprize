# Copyright (c) 2026 JorisViaudQuantAI (MIT License)
"""Build a live-verified 19-scroll First Letters coverage & public asset matrix.

Queries only official public endpoints (S3 open-data bucket, public GitHub
mesh repositories, and Hugging Face public dataset manifests) using the Python
standard library and merges recorded team coverage from ``collab/`` notes.

No unread-scroll predictions, private coordinates, or local filesystem paths
are written to the output JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
S3_BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

# Canonical 19 eligible First Letters scrolls from https://scrollprize.org/prizes (checked 2026-11-22)
ELIGIBLE_SCROLLS = [
    "PHerc0125",
    "PHerc0175A",
    "PHerc0175B",
    "PHerc0191",
    "PHerc0257",
    "PHerc0268",
    "PHerc0358",
    "PHerc0483A",
    "PHerc0483B",
    "PHerc0490A",
    "PHerc0490B",
    "PHerc0800",
    "PHerc0813",
    "PHerc0826",
    "PHerc0846A",
    "PHerc0846B",
    "PHerc1203",
    "PHerc1218",
    "PHerc1545",
]

# Recorded public team activity up to chatgpt-017 / claude-010 / codex-010 (2026-11-22)
RECORDED_TEAM_STATUS = {
    "PHerc0125": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": (
            "Claude: 1 own window N4 (3.49 cm2, rejected/negative) + 62 old atlas windings queued on L4; "
            "Codex: claimed 2 of 278 new pscamillo windings (>4 cm2 scored privately, codex-008..010)"
        ),
        "remaining_gap": "276 of 278 new pscamillo windings unscreened",
        "gemini_priority": "P1-G2",
    },
    "PHerc0175A": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: 3 negative GrowPatch/m7 windows (N1-N3: 7.45, 3.91, 2.18 cm2; N2 overlaps N1)",
        "remaining_gap": "No public spiral atlas; other axial z-bands require GrowPatch/spiral fit",
        "gemini_priority": "P3-after-claude011",
    },
    "PHerc0175B": {
        "owners": [],
        "recorded_work": "No recorded screening in collab notes through claude-010/codex-010",
        "remaining_gap": "Untouched scroll (no public meshes; S3 m7 available)",
        "gemini_priority": "P2-G4-pending-claude011",
    },
    "PHerc0191": {
        "owners": ["Codex (mentioned in codex-001, no published coverage count)"],
        "recorded_work": "Zero screening recorded on the 90-winding HF spiral fit (415.63 cm2 raw surface)",
        "remaining_gap": "90 HF spiral windings completely unscreened + umbilicus + S3 m7 available",
        "gemini_priority": "P0-G1",
    },
    "PHerc0257": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: 6 atlas windings queued on L4 (claude-010); Codex mentioned early screening (codex-001)",
        "remaining_gap": "Outside the 6 atlas windings (23 cm2), rest of scroll unmeshed",
        "gemini_priority": "P3-G4",
    },
    "PHerc0268": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: 6 atlas windings queued on L4 (claude-010; layer-coherence 0.31)",
        "remaining_gap": "Low sheet-parallel quality in 6-winding atlas; new spiral/GrowPatch needed",
        "gemini_priority": "P3-G4",
    },
    "PHerc0358": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: 3 atlas windings queued + 7k-step T4 spiral lane stopped (0.286 within 2 vx of m7); Codex: spiral replay fixture",
        "remaining_gap": "Only 3 atlas windings (11 cm2) rendered; full spiral needs deeper fit or GrowPatch",
        "gemini_priority": "P3-G4",
    },
    "PHerc0483A": {
        "owners": [],
        "recorded_work": "No recorded screening in collab notes through claude-010/codex-010",
        "remaining_gap": "Untouched scroll (no public meshes; S3 m7 available)",
        "gemini_priority": "P2-G4-pending-claude011",
    },
    "PHerc0483B": {
        "owners": ["Codex (mentioned in codex-001)"],
        "recorded_work": "Mentioned in codex-001; no published mesh/coverage table in collab-001..010",
        "remaining_gap": "No public meshes; S3 m7 available",
        "gemini_priority": "P2-G4-pending-claude011",
    },
    "PHerc0490A": {
        "owners": ["Claude Code"],
        "recorded_work": "Included in Claude pre-protocol 30 cm2 search (claude-001); no public meshes",
        "remaining_gap": "Most axial volume unmeshed; S3 m7 available",
        "gemini_priority": "P3-after-claude011",
    },
    "PHerc0490B": {
        "owners": [],
        "recorded_work": "No recorded screening in collab notes through claude-010/codex-010",
        "remaining_gap": "Untouched scroll (no public meshes; S3 m7 available)",
        "gemini_priority": "P2-G4-pending-claude011",
    },
    "PHerc0800": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: all 95 atlas windings screened with reader-v3 (477 cm2 valid, claude-008; Studio interrupted)",
        "remaining_gap": "6 separate S3 auto_grown segments not recorded as screened; Studio re-pass pending 306B row gate",
        "gemini_priority": "P1-G3-s3-segments",
    },
    "PHerc0813": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: all 75 atlas windings screened with reader-v3 (407 cm2 valid) + Studio on top 5 (29 cm2); Codex: 24-profile m7 paired geometry audit",
        "remaining_gap": "Studio on remaining atlas pending 306B row gate + paired physical-anchor check",
        "gemini_priority": "P3-conditional",
    },
    "PHerc0826": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: 1 small own window N5 (1.16 cm2) + 3 atlas windings (14 cm2) queued on L4",
        "remaining_gap": "3 complete spiral-fit bands (270 windings total raw) on HF eligible-scroll-spiral-fits unscreened",
        "gemini_priority": "P0-G1",
    },
    "PHerc0846A": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Registration only (2.4 um -> 9.36 um transform + Codex annular residual check); no ink screening recorded",
        "remaining_gap": "Zero ink screening recorded; has BOTH 9.362 um and 2.403 um rescan volumes",
        "gemini_priority": "P2-G4-pending-claude011",
    },
    "PHerc0846B": {
        "owners": [],
        "recorded_work": "No recorded screening; Codex noted no workable local CT/mesh pair (codex-008)",
        "remaining_gap": "Untouched scroll (no public meshes; S3 m7 available)",
        "gemini_priority": "P2-G4-pending-claude011",
    },
    "PHerc1203": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Codex: windows A/B/C (N13-N15) + 2.4 um cached blocks; Claude: N6-N10 at 9.36 um & 2.4 um; N9 unresolved",
        "remaining_gap": "Other S3 segments/raw/ meshes (~20 remaining) + resolve N9 at 9.36 um",
        "gemini_priority": "P2-G3",
    },
    "PHerc1218": {
        "owners": ["Codex"],
        "recorded_work": "Codex: staged S3 CT + m7 near 25 IyanDopico verified patches on CPU (codex/dls-public-depth-fixture); zero GPU ink runs recorded",
        "remaining_gap": "25 verified patches ready for GPU ink inference (Studio BF16 + A-hn)",
        "gemini_priority": "P1-G3",
    },
    "PHerc1545": {
        "owners": ["Codex"],
        "recorded_work": "Codex: packaged 3 uniformly sampled anatomy cubes (codex/pherc1545-public-anatomy-grid); zero ink screening recorded",
        "remaining_gap": "Untouched for ink screening (no public meshes; S3 m7 + umbilicus JSON available)",
        "gemini_priority": "P2-G4-pending-claude011",
    },
}


def fetch_text(url: str) -> str:
    """GET ``url`` as UTF-8 text with simple retry."""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gemini-coverage-inventory"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                return resp.read().decode("utf-8", "replace")
        except OSError:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def s3_prefixes(prefix: str) -> list[str]:
    """List immediate S3 CommonPrefixes under ``prefix``."""
    xml_txt = fetch_text(f"{S3_BUCKET}/?list-type=2&delimiter=/&prefix={prefix}")
    root = ET.fromstring(xml_txt)
    return [el.text for el in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS) if el.text]


def inspect_scroll_s3(scroll: str) -> dict:
    """Query S3 public volumes, surface predictions, and segments for one scroll."""
    top = [p.replace(f"{scroll}/", "").rstrip("/") for p in s3_prefixes(f"{scroll}/")]
    vols = [p.split("/")[-2] for p in s3_prefixes(f"{scroll}/volumes/")] if "volumes" in top else []
    native_vols = [v for v in vols if not v.endswith(".vcd3d.zarr") and "fast-q1" not in v]
    has_24um = any("2.40" in v or "2.4um" in v for v in vols)
    voxel_um = None
    for v in native_vols:
        m = re.search(r"-(\d+\.\d+)um-", v)
        if m and ("9.362" in m.group(1) or "8.640" in m.group(1)):
            voxel_um = float(m.group(1))
            break

    preds = (
        [p.split("/")[-2] for p in s3_prefixes(f"{scroll}/representations/predictions/surfaces/")]
        if "representations" in top
        else []
    )
    has_m7 = any("m7" in p.lower() for p in preds)

    seg_count = 0
    if "segments" in top:
        seg_top = s3_prefixes(f"{scroll}/segments/")
        if any(p.endswith("raw/") for p in seg_top):
            seg_count = len(s3_prefixes(f"{scroll}/segments/raw/"))
        else:
            seg_count = len(seg_top)

    return {
        "scroll": scroll,
        "voxel_um": voxel_um,
        "has_2_4um_rescan": has_24um,
        "s3_volume_count": len(vols),
        "s3_native_volumes": native_vols,
        "s3_has_m7_and_normal_grids": has_m7,
        "s3_segments_count": seg_count,
    }


def build_inventory() -> dict:
    """Collect live public metadata across all 19 eligible First Letters scrolls."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        s3_rows = list(pool.map(inspect_scroll_s3, ELIGIBLE_SCROLLS))

    # 1. Hugging Face atlas manifest (340 pre-Nov17 atlas renders)
    manifest_csv = fetch_text(
        "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-atlas-renders/resolve/main/manifest.csv"
    )
    atlas_counts = Counter(
        line.split(",")[0] for line in manifest_csv.strip().splitlines()[1:] if line.strip()
    )

    # 2. pscamillo/vesuvius-eligible-meshes (includes 278 new PHerc0125 meshes)
    pscamillo_dirs = json.loads(
        fetch_text("https://api.github.com/repos/pscamillo/vesuvius-eligible-meshes/contents/meshes")
    )
    pscamillo_counts: dict[str, int] = {}
    for entry in pscamillo_dirs:
        name = entry["name"]
        if name in ELIGIBLE_SCROLLS:
            items = json.loads(
                fetch_text(
                    f"https://api.github.com/repos/pscamillo/vesuvius-eligible-meshes/contents/meshes/{name}"
                )
            )
            pscamillo_counts[name] = len(items)

    # 3. rodriguescarson/eligible-scroll-spiral-fits (PHerc0191 + PHerc0826)
    sf_tree = json.loads(
        fetch_text("https://huggingface.co/api/datasets/rodriguescarson/eligible-scroll-spiral-fits/tree/main")
    )
    spiral_fit_runs: dict[str, list[str]] = {}
    for item in sf_tree:
        if item.get("type") == "directory":
            p = item.get("path", "")
            for s in ELIGIBLE_SCROLLS:
                if s in p:
                    spiral_fit_runs.setdefault(s, []).append(p)

    # 4. IyanDopico/vesuvius-sheet-tools verified PHerc1218 patches
    d1218 = json.loads(
        fetch_text(
            "https://api.github.com/repos/IyanDopico/vesuvius-sheet-tools/contents/data/spiral_input_pherc1218/verified_patches"
        )
    )
    patches_1218 = sum(1 for item in d1218 if item.get("type") == "dir")

    # 5. AlexeyDrobkovStrikesBack/herculaneum-umbilici
    umb = json.loads(
        fetch_text("https://api.github.com/repos/AlexeyDrobkovStrikesBack/herculaneum-umbilici/contents")
    )
    umb_scrolls = {item["name"].replace("_umbilicus.json", "") for item in umb if item["name"].endswith(".json")}

    rows = []
    for base in s3_rows:
        s = base["scroll"]
        team = RECORDED_TEAM_STATUS[s]
        hf_spiral = spiral_fit_runs.get(s, [])
        row = {
            **base,
            "has_umbilicus_json": s in umb_scrolls,
            "pscamillo_meshes": pscamillo_counts.get(s, 0),
            "hf_atlas_renders": atlas_counts.get(s, 0),
            "hf_spiral_fit_runs": len(hf_spiral),
            "hf_spiral_fit_names": hf_spiral,
            "iyandopico_verified_patches": patches_1218 if s == "PHerc1218" else 0,
            "team_owners": team["owners"],
            "recorded_work": team["recorded_work"],
            "remaining_gap": team["remaining_gap"],
            "gemini_priority": team["gemini_priority"],
        }
        rows.append(row)

    summary = {
        "eligible_scroll_count": len(rows),
        "scrolls_with_s3_m7": sum(1 for r in rows if r["s3_has_m7_and_normal_grids"]),
        "scrolls_with_2_4um_rescan": [r["scroll"] for r in rows if r["has_2_4um_rescan"]],
        "scrolls_with_zero_public_meshes": [
            r["scroll"]
            for r in rows
            if r["pscamillo_meshes"] == 0
            and r["hf_spiral_fit_runs"] == 0
            and r["iyandopico_verified_patches"] == 0
            and r["s3_segments_count"] == 0
        ],
        "untouched_pscamillo_0125_new_meshes": max(
            0, pscamillo_counts.get("PHerc0125", 0) - atlas_counts.get("PHerc0125", 0) - 2
        ),
        "total_hf_spiral_fit_runs": sum(r["hf_spiral_fit_runs"] for r in rows),
    }

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_pages": {
            "prizes": "https://scrollprize.org/prizes",
            "s3_bucket": S3_BUCKET,
            "pscamillo": "https://github.com/pscamillo/vesuvius-eligible-meshes",
            "hf_atlas": "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-atlas-renders",
            "hf_spiral_fits": "https://huggingface.co/datasets/rodriguescarson/eligible-scroll-spiral-fits",
            "iyandopico": "https://github.com/IyanDopico/vesuvius-sheet-tools",
            "umbilici": "https://github.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici",
        },
        "caveat": (
            "Raw mesh/winding counts are NOT unique on-sheet papyrus area. "
            "Spiral fits overlap across bands and can drift off-sheet; verify m7 support and layer coherence before inference."
        ),
        "summary": summary,
        "scrolls": rows,
        "paris4_title_summary": {
            "prize_usd": 50000,
            "deadline": "2027-06-25",
            "recorded_work": {
                "Claude": "N11 agraphon right of last column (~140 cm2) + N12 unmeshed core + control-then-agraphon GPU run queued",
                "Codex": "codex/paris4-energy-coverage: 78 keV vs 53 keV mesh overlap & RLE coverage comparison",
            },
            "unexplored_title_zones": [
                "Below the final prose column (8 of 106 surviving Herculaneum end titles per Ranocchia / chatgpt-001)",
                "Outermost wraps / initial title / detached outer flakes",
                "Physical 2.4 um CT attenuation & crust anomalies targeting the organisers' 'possibly a different ink' note",
            ],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = ap.parse_args()
    data = build_inventory()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({data['summary']['eligible_scroll_count']} scrolls)")
    print("Summary:", json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
