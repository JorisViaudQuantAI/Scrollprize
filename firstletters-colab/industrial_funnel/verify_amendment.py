"""Run old and new synthetic reference tests; record exact source hashes."""
import hashlib,json,platform,sqlite3,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
start=time.monotonic()
suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_*.py")
r=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={"schema":"funnel.amendment073.tests.v1","scope":"synthetic local file and prior coordinator/survey tests",
         "tests_run":r.testsRun,"failures":[{"test":str(t),"traceback":x} for t,x in r.failures],
         "errors":[{"test":str(t),"traceback":x} for t,x in r.errors],"skipped":len(r.skipped),
         "pass":r.wasSuccessful() and not r.skipped,"python":platform.python_version(),"sqlite":sqlite3.sqlite_version,
         "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"control_plane.py",HERE/"survey.py",HERE/"file_receipts.py",HERE/"test_control_plane.py",HERE/"test_file_receipts.py",Path(__file__))},
         "wall_seconds":time.monotonic()-start}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+"\n")
print("TEST_RESULT_JSON="+json.dumps(receipt,separators=(",",":")),flush=True)
raise SystemExit(0 if receipt["pass"] else 1)
