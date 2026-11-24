"""Record the actual synthetic reference-control-plane CI result."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import time
import unittest

HERE=Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    start=time.monotonic()
    tests=unittest.defaultTestLoader.discover(str(HERE),pattern="test_control_plane.py")
    r=unittest.TextTestRunner(verbosity=2).run(tests)
    report={"schema":"industrial_funnel.reference_ci.v1",
            "scope":"synthetic transaction, integrity, budget and survey contracts; no production workers or scientific screening",
            "sources":{p:sha(HERE/p) for p in ("RFC-001.md","control_plane.py","survey.py","test_control_plane.py","verify.py")},
            "python":platform.python_version(),"sqlite":sqlite3.sqlite_version,
            "tests_run":r.testsRun,"failures":[{"test":str(t),"traceback":x} for t,x in r.failures],
            "errors":[{"test":str(t),"traceback":x} for t,x in r.errors],
            "skipped":[{"test":str(t),"reason":x} for t,x in r.skipped],
            "passed":r.wasSuccessful() and not r.skipped,"wall_seconds":time.monotonic()-start}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("RESULT_JSON="+json.dumps(report,separators=(",",":")),flush=True)
    raise SystemExit(0 if report["passed"] else 1)
