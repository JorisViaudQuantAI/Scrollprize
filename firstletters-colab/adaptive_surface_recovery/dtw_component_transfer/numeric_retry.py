#!/usr/bin/env python3
"""Assertion-only retry of Adaptive 059. Preserve all real-data gates."""
import argparse,hashlib,importlib.util,json,pathlib,sys
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("component059",HERE/"run.py")
old=importlib.util.module_from_spec(sp);sys.modules[sp.name]=old;sp.loader.exec_module(old)
strict=old.self_test
def corrected():
    a=strict()
    p=np.array([[0,0,30],[31,0,30],[0,31,30],[31,31,30]],float)
    q=np.vstack([p,p+[0,0,12]]);f=np.array([[0,1,3],[0,3,2],[4,5,7],[4,7,6]])
    d,v,rows,counts,n=old.project_components(q,f,[0,1],2,[0,0],0,f[0])
    order=np.argsort(d[:,16,16]);d=d[order];v=v[order]
    error=float(np.max(np.abs(d-np.array([30.,42.])[:,None,None]))) if n==2 and v.all() and np.isfinite(d).all() else None
    gates=dict(a["gates"]);gates["separate_exact_depths"]=error is not None and error<=1e-10
    return {**a,"strict059_gates":a["gates"],"measured_plane_max_abs_error_vox":error,
        "assertion_tolerance_vox":1e-10,"gates":gates,"pass":all(gates.values())}
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("output","gemini","prior"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    a=p.parse_args()
    old.self_test=corrected
    old.run(a)
    r=json.loads(a.output.read_text())
    r.update(schema="adaptive.dtw_component_transfer.numeric_retry.v1",retry_script_sha256=digest(__file__),
       retry_protocol_sha256=digest(HERE/"numeric_retry.md"),strict059_receipt_sha256=digest(HERE/"result.json"))
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RETRY_RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
