"""Run A03 regression and pinned PR7 native compatibility suite."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument("--old-probe",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();t=time.monotonic()
    old=json.loads(a.old_probe.read_text())
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_*.py")
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    q=dict(schema="funnel.amendment075.v1",old_revision="9e66591b309eeb41561c173eb52932527370977b",
           native_revision="9ac322fd2f2cb9cfcb82f22904105ea08d64d3a4",
           old_probe=old,tests_run=r.testsRun,failures=[{"test":str(x),"traceback":y} for x,y in r.failures],
           errors=[{"test":str(x),"traceback":y} for x,y in r.errors],skipped=len(r.skipped),
           source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(HERE.glob("*.py"))},
           python=platform.python_version(),wall_seconds=time.monotonic()-t,
           scope="Synthetic fault injection and native array regressions only. No scans, model execution, or production dispatch.")
    q["pass"]=old["reproduced"] and r.wasSuccessful() and not r.skipped
    a.output.write_text(json.dumps(q,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(q,separators=(",",":")),flush=True)
    sys.exit(0 if q["pass"] else 1)
if __name__=="__main__":main()
