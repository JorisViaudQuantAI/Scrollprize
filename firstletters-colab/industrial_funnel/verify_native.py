#!/usr/bin/env python3
"""Execute local reference/native contract tests, without model or network work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import time
import unittest

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.glob("*.py"))}
    receipt = dict(schema="funnel.native_contract_tests.v1", tests_run=result.testsRun,
                   passed=result.wasSuccessful(), failures=[str(t) for t, _ in result.failures],
                   errors=[str(t) for t, _ in result.errors], skipped=[str(t) for t, _ in result.skipped],
                   source_sha256=sources, python=platform.python_version(), numpy=np.__version__, sqlite=sqlite3.sqlite_version,
                   elapsed_seconds=round(time.perf_counter()-started, 6), new_model_forwards=0,
                   scope="Synthetic reference and native-array/import contracts; not scientific promotion or production worker enforcement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("tests_run", "passed", "failures", "errors", "skipped")}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
