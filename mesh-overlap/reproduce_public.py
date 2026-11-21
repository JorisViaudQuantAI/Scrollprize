"""Replay the two public examples on pinned files and compare all reported counts."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from surface_patch_mask import run


def acquire(root, asset, offline):
    local = root / asset["path"]
    if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == asset["sha256"]:
        return
    if offline:
        raise RuntimeError(f"Missing or changed cached file {local}")
    with urlopen(asset["url"], timeout=45) as response:
        data = response.read()
    if len(data) != asset["size"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
        raise ValueError(f"Published checksum differs for {local.name}")
    local.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=local.parent, delete=False) as tmp:
        path = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            path.replace(local)
        finally:
            path.unlink(missing_ok=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work", required=True, type=Path, help="Directory for verified public inputs")
    cli.add_argument("--offline", action="store_true", help="Use an existing verified cache")
    cli.add_argument("--case", choices=["all", "139", "343"], default="all")
    args = cli.parse_args()
    manifest = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    checks = 0
    for case in ("139", "343"):
        if args.case not in ("all", case):
            continue
        record = manifest[case]
        for asset in record["assets"]:
            acquire(args.work, asset, args.offline)
        base = args.work / f"fixture{case}"
        for sample in record["checks"]:
            _, result = run(base/sample["target"], [base/p for p in sample["references"]],
                            record["frame"], sample["radius"], (16, 16), (8, 8))
            measured = [result["valid_vertices"], result["excluded_vertices"],
                        *[result["windows"][key] for key in ("candidate", "center_only", "center_only_with_overlap", "retained")]]
            if measured != sample["counts"]:
                raise AssertionError((case, sample, measured))
            checks += 1
            print(case, sample["target"], "+".join(sample["references"]), sample["radius"], measured)
    print(f"{checks} comparisons reproduced")


if __name__ == "__main__":
    main()
