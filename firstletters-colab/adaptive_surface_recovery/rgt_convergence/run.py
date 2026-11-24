#!/usr/bin/env python3
"""Same upstream RGT equations: Jacobi CG250/2000 versus symmetric AMG-PCG250."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="2"
import argparse,importlib.util,inspect,json,pathlib,resource,sys,time
import numpy as np
import requests,pyamg
from scipy import sparse
from scipy.sparse import linalg as spla
HERE=pathlib.Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("identity064",HERE.parent/"rgt_identity_bridge"/"run.py")
r64=importlib.util.module_from_spec(sp);sys.modules[sp.name]=r64;sp.loader.exec_module(r64)
co=r64.co;b=r64.b;old=r64.old;data=r64.data
KEYS=("B","J","A")
def relres(A,x,rhs):return float(np.linalg.norm(A@x-rhs)/max(np.linalg.norm(rhs),1e-30))
def quadratic(A,x,rhs):return float(.5*x@(A@x)-rhs@x)
def solve_all(A,rhs,x0,pre,callback=None):
    if A.nnz>2_000_000:raise ValueError("Sparse nnz cap")
    asym=(A-A.T).tocsr();max_asym=float(np.max(np.abs(asym.data),initial=0.))
    if max_asym>1e-10 or not np.isfinite(A.data).all() or not np.isfinite(rhs).all() or np.min(A.diagonal())<=0:raise ValueError("Matrix symmetry/finite/diagonal gate")
    ident={"shape":list(A.shape),"nnz":int(A.nnz),"data_sha256":old.arrsha(A.data),"indices_sha256":old.arrsha(A.indices),"indptr_sha256":old.arrsha(A.indptr),
        "rhs_sha256":old.arrsha(rhs),"initial_sha256":old.arrsha(x0),"max_asymmetry":max_asym,"diagonal_min":float(A.diagonal().min()),"rhs_norm":float(np.linalg.norm(rhs))}
    rows={};sols={}
    def one(key,P,limit,setup=0.,meta=None):
        n=[0]
        def cb(x):
            n[0]+=1
            if key=="A" and callback is not None:callback(x)
        t=time.monotonic();x,info=spla.cg(A,rhs,x0=x0.copy(),rtol=1e-5,atol=0.,maxiter=limit,M=P,callback=cb);elapsed=time.monotonic()-t
        if not np.isfinite(x).all():raise ValueError("Nonfinite CG result")
        sols[key]=x;rows[key]={"iterations":n[0],"cg_info":int(info),"relative_residual":relres(A,x,rhs),"quadratic":quadratic(A,x,rhs),
          "solve_seconds":elapsed,"setup_seconds":setup,"total_seconds":elapsed+setup,**(meta or {})}
    one("B",pre,250);one("J",pre,2000)
    np.random.seed(6601);t=time.monotonic()
    ml=pyamg.smoothed_aggregation_solver(A.copy(),symmetry="symmetric",strength="symmetric",
        presmoother=("gauss_seidel",{"sweep":"symmetric"}),postsmoother=("gauss_seidel",{"sweep":"symmetric"}),max_levels=10,max_coarse=100)
    setup=time.monotonic()-t
    one("A",ml.aspreconditioner(cycle="V"),250,setup,{"levels":len(ml.levels),"operator_complexity":float(ml.operator_complexity())})
    if old.arrsha(A.data)!=ident["data_sha256"] or old.arrsha(A.indices)!=ident["indices_sha256"] or old.arrsha(A.indptr)!=ident["indptr_sha256"] or old.arrsha(rhs)!=ident["rhs_sha256"]:raise ValueError("Matrix/rhs mutated")
    return sols,rows,ident
def project(x,shape,axis,eps):
    p=np.moveaxis(x.reshape(shape),axis,0);ramp=eps*np.arange(p.shape[0],dtype=np.float64).reshape(-1,1,1)
    proj=np.maximum.accumulate(p-ramp,axis=0)+ramp
    return np.ascontiguousarray(np.moveaxis(proj,0,axis)),float(np.mean(proj>p+1e-12))
def instrument(rgt,capture):
    src=inspect.getsource(rgt.solve_dip_guided_rgt_3d)
    line="    phi, cg_info = spla.cg(M, rhs, x0=phi0, rtol=float(tol), maxiter=int(max_iter), M=pre, callback=_cb)"
    if src.count(line)!=1:raise ValueError("Upstream CG call changed")
    def hook(A,rhs,x0,pre,cb):
        sols,rows,meta=solve_all(A,rhs,x0,pre,cb);capture.update(sols=sols,solvers=rows,matrix=meta)
        return sols["A"],rows["A"]["cg_info"]
    src=src.replace(line,"    phi, cg_info = compare_solvers(M, rhs, phi0, pre, _cb)")
    scope=dict(vars(rgt));scope["compare_solvers"]=hook;exec(compile(src,"<same upstream RGT matrix>","exec"),scope)
    return scope["solve_dip_guided_rgt_3d"]
def synthetic(rgt):
    A=sparse.diags([-np.ones(15),np.full(16,2.2),-np.ones(15)],[-1,0,1],format="csr")
    rhs=np.sin(np.arange(16));x0=np.zeros(16);pre=spla.LinearOperator(A.shape,matvec=lambda x:x/A.diagonal(),dtype=np.float64)
    sols,rows,_=solve_all(A,rhs,x0,pre);exact=np.linalg.solve(A.toarray(),rhs)
    z=np.arange(24)[:,None,None];y,x=np.indices((8,8));v=np.broadcast_to(100+25*np.cos(2*np.pi*(z+.1*y)/8),(24,8,8)).astype(np.float32)
    n,pl,_=rgt.estimate_structure_tensor_dip_3d(v);n=n*np.where(n[...,0]<0,-1.,1.)[...,None]
    k,_=rgt.estimate_local_wrap_wavenumber_3d(v,0)
    orig=rgt.solve_dip_guided_rgt_3d(v,n,pl,k,normal_axis=0)
    cap={};new=instrument(rgt,cap)(v,n,pl,k,normal_axis=0)
    proj,_=project(cap["sols"]["B"],v.shape,0,1e-3)
    pnew,_=project(cap["sols"]["A"],v.shape,0,1e-3)
    gates={"small_SPD":all(np.max(np.abs(q-exact))<1e-4 for q in sols.values()),"original_projection_exact":np.array_equal(proj,orig["phi"]),
        "AMG_projection_exact":np.array_equal(pnew,new["phi"]),"synthetic_AMG_converged":cap["solvers"]["A"]["cg_info"]==0 and cap["solvers"]["A"]["relative_residual"]<=1.05e-5}
    return {"gates":{k:bool(v) for k,v in gates.items()},"pass":all(gates.values()),"dense_max_error":{k:float(np.max(np.abs(q-exact))) for k,q in sols.items()},"wave_solvers":cap["solvers"]}
def case(ct,gt,prior61,prior64,rgt):
    normal=int(prior61["normal_axis"])
    v=np.moveaxis(ct,normal,0)[:,32:64,32:64].copy();g=np.moveaxis(gt,normal,0)[:,32:64,32:64].copy()
    if old.arrsha(v)!=prior61["CT_sha256"] or old.arrsha(g)!=prior61["label_sha256"]:raise ValueError("061 input hash mismatch")
    a=old.anchor_ids(g[:,co.CENTER[0],co.CENTER[1]]);seed=np.array([z for _,z in a],np.int64);ids=[int(i) for i,_ in a]
    extra=[];anchors=[[(*co.CENTER,int(z))] for z in seed]
    for y,x in co.EXTRA:
        for i,(gid,_) in enumerate(a):
            z=co.run_anchor(g[:,y,x],gid)
            if z is not None:extra.append((i,y,x,z-b.DLO));anchors[i].append((y,x,z))
    allowed=r64.allowed_mask();truth=b.truth_for(g,a,allowed)
    C=b.unary(v,seed,co.CENTER)
    n,pl,_=rgt.estimate_structure_tensor_dip_3d(v);n=n*np.where(n[...,0]<0,-1.,1.)[...,None]
    k,_=rgt.estimate_local_wrap_wavenumber_3d(v,0)
    cap={};_ = instrument(rgt,cap)(v,n,pl,k,normal_axis=0)
    arms={};preds={};projected={};pts,_,_=r64.observations(g,allowed,ids);z,y,x,gtid=pts.T
    for key in KEYS:
        phi,moved=project(cap["sols"][key],v.shape,0,1e-3);projected[key]=phi
        levels=np.array([np.median([phi[z,y,x] for y,x,z in loc]) for loc in anchors])
        hh=np.stack([rgt._heights_at_phase(phi,float(q)) for q in levels])
        d=np.abs(hh[...,None]-np.arange(b.DLO,b.DHI,dtype=np.float64));pen=np.rint(150*np.where(np.isfinite(d),np.minimum(d,12),0)).astype(np.int64)
        p,diag=co.ac.cut3d(C+pen,seed-b.DLO,co.CENTER,lam=100,slope=4,gap=2,extra=extra);p=p+b.DLO;preds[key]=p
        rr=r64.score_height(g,a,hh,truth);rc=r64.score_height(g,a,p,truth)
        if key=="B":
            if rr["prediction_sha256"]!=prior64["arms"]["R5"]["prediction_sha256"] or rc["prediction_sha256"]!=prior64["arms"]["RC"]["prediction_sha256"]:raise ValueError("064 R5/RC baseline hash mismatch")
            if rr["near2"]!=prior64["arms"]["R5"]["near2"] or rc["near2"]!=prior64["arms"]["RC"]["near2"]:raise ValueError("064 baseline metric mismatch")
        arms[key]={"projection_fraction":moved,"levels_ordered":bool((np.diff(levels)>0).all()),"R5":rr,"RC":rc,
            "assignment":r64.assignment_metrics(np.abs(p[:,y,x]-z[None]),ids,gtid),"cut":diag,
            "anchors_exact":bool(all(p[i,y0,x0]==z0+b.DLO for i,y0,x0,z0 in extra))}
    rc=arms["A"]["RC"];oldrc=arms["B"]["RC"];M=prior64["arms"]["M"];jm=prior64["arms"]["JM"]
    gB={"within2":rc["within2_rate"]>=oldrc["within2_rate"]-.02,"identity":rc["correct_id_rate"]>=oldrc["correct_id_rate"]-.03,
         "wrong":rc["wrong_id_rate"]<=oldrc["wrong_id_rate"]+.03,"constraints":rc["gap_violations"]==0 and rc["slope_violations_axes"]==[0,0] and arms["A"]["anchors_exact"]}
    gM={"within_M":rc["within2_rate"]>=M["within2_rate"]-.02,"within_JM":rc["within2_rate"]>=jm["within2_rate"]-.02,
         "id_M":rc["correct_id_rate"]>=M["correct_id_rate"]-.03,"wrong_M":rc["wrong_id_rate"]<=M["wrong_id_rate"]+.03,
         "R5_coverage":arms["A"]["R5"]["coverage"]>=.95,"ordered":arms["A"]["levels_ordered"],"constraints":gB["constraints"]}
    return {"complete":True,"matrix":cap["matrix"],"solvers":cap["solvers"],"arms":arms,
        "phase_difference":{"A_vs_B_abs_p50_p95_max":np.quantile(np.abs(projected["A"]-projected["B"]),[.5,.95,1]).tolist(),
           "A_vs_J_abs_p50_p95_max":np.quantile(np.abs(projected["A"]-projected["J"]),[.5,.95,1]).tolist(),
           "RC_A_vs_B_changed_fraction":float((preds["A"]!=preds["B"]).mean())},
        "B_nonharm":gB,"M_gates":gM,"RC_gain_M":rc["within2_rate"]-M["within2_rate"]}
def run(a):
    t=time.monotonic();r={"schema":"adaptive.rgt_convergence.v1","script_sha256":b.digest(__file__),"protocol_sha256":b.digest(HERE/"protocol.md"),
        "prior061_sha256":b.digest(a.prior061),"prior064_sha256":b.digest(a.prior064),"sources":{},"source_bytes":0,"cases":[],"pass":False}
    sys.path.insert(0,str(a.gemini));import seismic_rgt_unwrapper as rgt
    r["upstream_sha256"]=b.digest(a.gemini/"seismic_rgt_unwrapper.py")
    try:
        r["synthetic"]=synthetic(rgt)
        if not r["synthetic"]["pass"]:raise ValueError("Frozen synthetic contract failed")
        p61=json.loads(a.prior061.read_text());p64=json.loads(a.prior064.read_text());data.CAP=120*1024**2;a.cache.mkdir(parents=True,exist_ok=False)
        with requests.Session() as session:
            for cube in co.CUBES:
                try:
                    ct,gt=data.read_cube(session,a.cache,cube,r)
                    if any(r["sources"][cube][n]["sha256"]!=p61["sources"][cube][n]["sha256"] for n in ("volume","mask")):raise ValueError("Source mismatch")
                    row=case(ct,gt,next(q for q in p61["cases"] if q["cube"]==cube),next(q for q in p64["cases"] if q["cube"]==cube),rgt)
                except Exception as e:row={"complete":False,"failure":f"{type(e).__name__}: {str(e)[:400]}"}
                r["cases"].append({"cube":cube,**row});print("CASE_JSON="+json.dumps(r["cases"][-1],separators=(",",":"),allow_nan=False),flush=True)
        ok=[q for q in r["cases"] if q["complete"]]
        def converged(q,key):s=q["solvers"][key];return s["cg_info"]==0 and s["relative_residual"]<=1.05e-5
        r["numerical_pass"]=len(ok)==4 and all(converged(q,"A") for q in ok)
        both=[q for q in ok if converged(q,"A") and converged(q,"J")]
        r["speed_pass"]=len(both)==4 and all(q["solvers"]["A"]["total_seconds"]<q["solvers"]["J"]["total_seconds"] for q in both)
        r["physical_nonharm"]=len(ok)==4 and all(all(q["B_nonharm"].values()) for q in ok)
        r["physical_promotion"]=r["numerical_pass"] and len(ok)==4 and all(all(q["M_gates"].values()) for q in ok) and any(q["RC_gain_M"]>=.03 for q in ok)
        r["pass"]=r["numerical_pass"] and r["physical_nonharm"]
        r["macro"]={key:{m:float(np.mean([q["arms"][key]["RC"][m] for q in ok])) for m in ("within2_rate","correct_id_rate","wrong_id_rate","MAE_vox_conditional")} for key in KEYS} if ok else {}
    except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    r["wall_seconds"]=time.monotonic()-t;r["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","output","prior061","prior064","gemini"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    run(p.parse_args())
