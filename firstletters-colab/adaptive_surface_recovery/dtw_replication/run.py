#!/usr/bin/env python3
"""Frozen replication of the exact 055 implementation; only population changes."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time,resource
import numpy as np
import requests
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("hybrid055",HERE.parent/"dtw_order_hybrid"/"run.py")
hy=importlib.util.module_from_spec(sp);sys.modules[sp.name]=hy;sp.loader.exec_module(hy)
CUBES=("00064_02664_04304","01024_02152_04048","08656_04304_04048","10192_04048_03280")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main(a):
    t=time.monotonic()
    r={"schema":"adaptive.dtw_replication.v1","script_sha256":sha(__file__),"protocol_sha256":sha(HERE/"protocol.md"),
       "hybrid055_sha256":sha(HERE.parent/"dtw_order_hybrid"/"run.py"),"sources":{},"source_bytes":0,"cases":[],"pass":False}
    hy.data.CAP=180*1024**2
    a.cache.mkdir(parents=True,exist_ok=False)
    try:
        r["synthetic"]=hy.self_test()
        if not r["synthetic"]["pass"]:raise ValueError("Inherited 055 synthetic gate failed")
        with requests.Session() as session:
            for cube in CUBES:
                rec={"cube":cube,"sections":[]}
                try:
                    ct,gt=hy.data.read_cube(session,a.cache,cube,r)
                    deriv=[float(np.mean(ndi.gaussian_filter(ct.astype(np.float32),1,order=tuple(int(i==j) for i in range(3)))**2)) for j in range(3)]
                    normal=int(np.argmax(deriv));rec.update(dominant_CT_normal_axis=normal,gradient_energy=deriv)
                    for axis in range(3):
                        if axis==normal:continue
                        remain=[j for j in range(3) if j!=axis];naxis=remain.index(normal)
                        for index in (24,48,72):
                            plane=np.take(ct,index,axis=axis);labels=np.take(gt,index,axis=axis)
                            if naxis==1:plane=plane.T;labels=labels.T
                            try:row=hy.section(plane,labels,axis,index)
                            except Exception as e:row={"fixed_axis":axis,"slice":index,"evaluable":False,"failure":f"{type(e).__name__}: {str(e)[:300]}"}
                            rec["sections"].append(row)
                    rec["aggregate"]=hy.aggregate(rec["sections"])
                except Exception as e:rec["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
                r["cases"].append(rec);print("CASE_JSON="+json.dumps(rec,separators=(",",":"),allow_nan=False),flush=True)
        complete=len(r["cases"])==len(CUBES) and all("failure" not in x and len(x["sections"])==6 and not any("failure" in q for q in x["sections"]) for x in r["cases"])
        r["complete"]=complete
        ag=[x["aggregate"] for x in r["cases"] if "aggregate" in x and x["aggregate"]["population_pass"]]
        r["assessable_cubes"]=len(ag)
        r["at_least_one_gain_ge03"]=any(x["H_within2_delta_W"]>=.03 for x in ag)
        r["pass"]=complete and len(ag)==len(CUBES) and r["at_least_one_gain_ge03"] and all(all(x["H_gates"].values()) for x in ag)
        r["equal_cube_summary"]={}
        for k in hy.KEYS:
            v=[x["arms"][k] for x in ag]
            if v:r["equal_cube_summary"][k]={m:float(np.mean([q[m] for q in v])) for m in ("within2_rate","correct_id_rate","wrong_id_rate","MAE_vox","gap_fraction")}
        r["pooled_summary"]={}
        for k in hy.KEYS:
            vals=[x["arms"][k] for x in ag];n=sum(x["eligible"] for x in vals)
            if n:r["pooled_summary"][k]={"eligible":n,"within2_rate":sum(x["near2"] for x in vals)/n,"correct_id_rate":sum(x["correct_id"] for x in vals)/n,"wrong_id_rate":sum(x["other_id"] for x in vals)/n,"MAE_vox":sum(x["absolute_error_sum"] for x in vals)/n,"gap_violations":sum(x["gap_violations"] for x in vals),"adjacent_pairs":sum(x["adjacent_pairs"] for x in vals)}
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    main(p.parse_args())
