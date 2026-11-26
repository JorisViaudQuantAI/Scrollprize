#!/usr/bin/env python3
"""Retrieve only the pinned public geometry required by run.py (under 16 MiB)."""
import argparse
import json
from pathlib import Path
import urllib.request

from run import CLAUDE, STATES, WINDS, read_git, sha, upstream


def prepare(repo, out):
    _, _, labels = upstream(repo)
    receipts = []
    for state in STATES:
        for winding in WINDS:
            records = labels["states"][state]["windings"][winding]["files"]
            for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
                rel = f"firstletters-colab/pherc0211_fit_windings/{state}/{winding}/{name}"
                dst = out / "meshes" / state / winding / name
                body = read_git(repo, CLAUDE, rel)
                q = records[name]
                if len(body) != q["bytes"] or sha(body) != q["sha256"]:
                    raise ValueError("Pinned Git geometry differs from reference: " + rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(body)
                receipts.append({"path": dst.relative_to(out).as_posix(), "bytes": len(body), "sha256": sha(body)})
    for name, q in labels["winner"]["files"].items():
        dst = out / "winner" / name
        if dst.is_file() and sha(dst.read_bytes()) == q["sha256"]:
            body = dst.read_bytes()
        else:
            url = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + q["source"]
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read(q["bytes"] + 1)
        if len(body) != q["bytes"] or sha(body) != q["sha256"]:
            raise ValueError("Public winner geometry integrity failure: " + name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(body)
        receipts.append({"path": dst.relative_to(out).as_posix(), "bytes": len(body), "sha256": sha(body)})
    total = sum(q["bytes"] for q in receipts)
    if total > 16 * 1024**2:
        raise ValueError("Geometry byte budget exceeded")
    result = {"files": receipts, "total_bytes": total, "ct_downloaded": False}
    (out / "prepared_geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(receipts), "total_bytes": total, "ct_downloaded": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    prepare(args.repo, args.out)
