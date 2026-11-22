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

# Recorded public team activity up to chatgpt-017 / claude-011 / codex-010 (2026-11-22)
RECORDED_TEAM_STATUS = {
    "PHerc0125": {
        "owners": ["Claude Code", "Codex", "Gemini (G2)"],
        "recorded_work": (
            "Claude: 62 older atlas windings finished in reader-v3 (364 cm2 valid / ~315 cm2 sheet-parallel, claude-011) "
            "+ Studio running on 15 windings (100 cm2) + N4 (3.49 cm2); "
            "Codex: claimed 2 of 278 new pscamillo windings (>4 cm2 scored privately, codex-008..010)"
        ),
        "remaining_gap": "276 of 278 new pscamillo windings unscreened (17 Tier-1 >=0.85, 51 Tier-2 [0.80,0.85) after 2x m7 snap; skip z17136 & w080-w100)",
        "gemini_priority": "P0-G2",
    },
    "PHerc0175A": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: N1-N3 (7.45, 3.91, 2.18 cm2) + 4 own GrowPatch windows (z 4800-9500; 3 inked >=4 cm2 core, 1 not inked, claude-011)",
        "remaining_gap": "No public spiral atlas; outside Claude z 4800-9500 tranche",
        "gemini_priority": "P3-claude-owned",
    },
    "PHerc0175B": {
        "owners": ["Gemini (G4)"],
        "recorded_work": "Zero screening recorded by Claude/Codex (confirmed untouched in claude-011)",
        "remaining_gap": "100% untouched scroll (no public meshes; S3 m7-L0 available, 8.640 um)",
        "gemini_priority": "P1-G4",
    },
    "PHerc0191": {
        "owners": ["Claude Code", "Gemini (G1)"],
        "recorded_work": "Claude: 5 own GrowPatch windows (z 7700-15400; 1 core >=4 cm2, 3 fragmented, 1 not inked, claude-011); 90-winding HF spiral run unscreened",
        "remaining_gap": "90 HF spiral windings (z 11600-12400, 415.63 cm2 raw) unscreened -> Gemini G1",
        "gemini_priority": "P1-G1",
    },
    "PHerc0257": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: all 6 atlas windings have reader-v3 + Studio maps in both orders (review closing, claude-011); Codex early screen",
        "remaining_gap": "Outside the 6 atlas windings (23 cm2), rest of scroll unmeshed",
        "gemini_priority": "P3-deferred",
    },
    "PHerc0268": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: reader-v3 maps exist on 6 atlas windings, but onsheet_score 0.15-0.49 (cut-through, treat as unscreened per claude-011)",
        "remaining_gap": "Effective zero on-sheet coverage; needs new GrowPatch/spiral fit",
        "gemini_priority": "P2-unscreened-effectively",
    },
    "PHerc0358": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: 3 atlas windings (reader-v3 + Studio both orders, review closing) + 4 own windows (z 3800-12800; 2 core >=4 cm2, 2 fragmented) + stopped 7k-step T4 fit; Codex: spiral replay",
        "remaining_gap": "Full spiral beyond Claude z-bands unscreened",
        "gemini_priority": "P3-deferred",
    },
    "PHerc0483A": {
        "owners": ["Gemini (G4)"],
        "recorded_work": "Zero screening recorded by Claude/Codex (confirmed untouched in claude-011)",
        "remaining_gap": "100% untouched scroll (no public meshes; S3 m7-L0 available, 8.640 um)",
        "gemini_priority": "P1-G4",
    },
    "PHerc0483B": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: 4 own GrowPatch windows (z 7300-10300; 2 core >=4 cm2, 2 not inked, claude-011); Codex mentioned in codex-001",
        "remaining_gap": "Bands outside z 7300-10300 unmeshed; S3 m7 available",
        "gemini_priority": "P3-claude-owned",
    },
    "PHerc0490A": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: pre-protocol search (claude-001) + 3 own GrowPatch windows (z 5500-9300; 1 fragmented, 2 not inked, claude-011)",
        "remaining_gap": "Bands outside z 5500-9300 unmeshed; S3 m7 available",
        "gemini_priority": "P3-claude-owned",
    },
    "PHerc0490B": {
        "owners": ["Gemini (G4)"],
        "recorded_work": "Zero screening recorded by Claude/Codex (confirmed untouched in claude-011)",
        "remaining_gap": "100% untouched scroll (no public meshes; S3 m7-L0 available, 8.640 um)",
        "gemini_priority": "P1-G4",
    },
    "PHerc0800": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: all 95 atlas windings screened with reader-v3 (477 cm2 valid, claude-008); all 6 S3 segments (9.93 cm2) already ink-detected on S3 incl. argus-v3 (2026-11-17)",
        "remaining_gap": "Studio re-pass dropped after 306B row test (claude-011); S3 segments already screened",
        "gemini_priority": "P4-exclude",
    },
    "PHerc0813": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Claude: all 75 atlas windings screened with reader-v3 (407 cm2 valid) + Studio on top 5 (29 cm2); Codex: 24-profile m7 paired geometry audit",
        "remaining_gap": "Studio re-pass dropped after 306B row test (claude-011); A-hn optional on top-ranked windings",
        "gemini_priority": "P4-low",
    },
    "PHerc0826": {
        "owners": ["Claude Code", "Gemini (G1)"],
        "recorded_work": "Claude: 3 atlas windings (reader-v3 + Studio both orders, review closing) + N5 (1.16 cm2) + 2 own windows (z 5400-8200; 2 fragmented, claude-011)",
        "remaining_gap": "3 HF spiral-fit bands (slice-8928-9728 & slice-9328-10128 unrepresented in pscamillo) -> Gemini G1",
        "gemini_priority": "P1-G1",
    },
    "PHerc0846A": {
        "owners": ["Gemini (G4)", "Claude Code (reg)", "Codex (reg)"],
        "recorded_work": "Registration only (2.4 um -> 9.36 um transform + Codex annular check); zero ink screening recorded (confirmed untouched in claude-011)",
        "remaining_gap": "100% untouched for ink; rare 2.403 um rescan + m7-L2 + normal-grids on S3 -> Gemini G4 top zero-mesh target",
        "gemini_priority": "P1-G4-rescan",
    },
    "PHerc0846B": {
        "owners": ["Claude Code"],
        "recorded_work": "Claude: 4 own GrowPatch windows (z 9300-12200; 1 fragmented, 3 not inked, claude-011)",
        "remaining_gap": "Bands outside z 9300-12200 unmeshed; S3 m7 available",
        "gemini_priority": "P3-claude-owned",
    },
    "PHerc1203": {
        "owners": ["Claude Code", "Codex", "Gemini (G3)"],
        "recorded_work": "Codex: A/B/C on 20251005230830031 + 2.4 um blocks; Claude: N6-N10 + 4 own windows (z 11000-14000; 2 core >=4 cm2, 2 fragmented, claude-011)",
        "remaining_gap": "9 large Oct/Sep auto_grown S3 segments (87.35 cm2, z 5403-10150, zero S3 ink maps) + 13 smaller segments (42.79 cm2) after UV deduplication -> Gemini G3",
        "gemini_priority": "P0-G3",
    },
    "PHerc1218": {
        "owners": ["Claude Code", "Codex", "Gemini (G3)"],
        "recorded_work": "Codex: staged S3 CT + m7 near 25 IyanDopico anchor seeds (3.64 cm2 total); Claude: 4 own windows (z 2400-7700; 1 fragmented, 3 not inked, claude-011)",
        "remaining_gap": "6 full-PASS spiral-fit windows on Kaggle pscamillo/pherc1218-spiral-fit-sweep-surfaces ready for RTX 4060 GPU screening -> Gemini G3",
        "gemini_priority": "P1-G3",
    },
    "PHerc1545": {
        "owners": ["Claude Code", "Codex"],
        "recorded_work": "Codex: 3 anatomy cubes; Claude: 5 own GrowPatch windows (z 2700-15500; 4 core >=4 cm2, 1 not inked, claude-011)",
        "remaining_gap": "Only zero-mesh scroll with both m7-L0 and lasagna on S3; bands outside Claude 5 windows",
        "gemini_priority": "P2-G4-secondary",
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
                "Claude": "N11 agraphon + N12 core + ~29 cm2 below final column (negative; ghosting warning) + right-of-column Hecate pass (claude-011)",
                "Codex": "codex/paris4-energy-coverage: 78 keV vs 137 keV (bridged via 54 keV DLS) mesh overlap & RLE coverage",
                "Gemini": "Owns different-ink dual-energy CT lane (54/78/137 keV + 1.129 um L0) + outer wraps w130-w135 + 20260701 extended height",
            },
            "unexplored_title_zones": [
                "Agraphon recto dual-energy (78 vs 137 keV spanning Pb 88.0 / Hg 83.1 keV K-edges; 54 vs 78 keV 96.8% overlap) & native 1.129 um crust",
                "Extended axial height in 20260701183124-w010-027 (z_78 = 11,953-27,964 and 68,063-72,051)",
                "Outermost wraps w130-w135 (20260926044226-spiral-output-generation-19) & detached outer flakes",
                "Innermost unmeshed wraps w001-w009 inside 1.129 um FOV & true verso boundary",
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
