#!/usr/bin/env python3
"""Support-aware arithmetic for the original spiral on-check profiles."""
import argparse,ast,hashlib,importlib.util,json,pathlib,sys,time,warnings
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
HERE=pathlib.Path(__file__).resolve().parent
OFFS=np.arange(-16,16.01,.5)
M7_URL="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2.zarr"

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def legacy_function(path):
    tree=ast.parse(pathlib.Path(path).read_text())
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="nearest_peak"]
    if len(defs)!=1:raise ValueError("Pinned nearest_peak not found")
    ns={"np":np,"ndimage":ndimage,"find_peaks":find_peaks,"offs":OFFS}
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),str(path),"exec"),ns)
    return ns["nearest_peak"]

def profile_peaks(profiles,observed,prominence):
    """Require complete measured profiles. Preserve tied nearest distances."""
    p=np.asarray(profiles,float);o=np.asarray(observed)
    if p.ndim!=2 or p.shape[1]!=65 or o.shape!=p.shape or o.dtype!=np.bool_:
        raise ValueError("Expected N x 65 profiles and bool source support")
    complete=(o&np.isfinite(p)).all(1)
    dist=np.full(len(p),np.nan);signed=dist.copy();tie=np.zeros(len(p),bool)
    for i in np.flatnonzero(complete):
        sm=ndimage.gaussian_filter1d(p[i],1.)
        pk,props=find_peaks(sm,prominence=prominence,plateau_size=True)
        if not len(pk):continue
        centers=(OFFS[props["left_edges"]]+OFFS[props["right_edges"]])/2
        d=np.abs(centers);m=d.min();nearest=np.isclose(d,m,rtol=0,atol=1e-12)
        dist[i]=m;tie[i]=nearest.sum()>1
        if not tie[i]:signed[i]=centers[nearest][0]
    return complete,dist,signed,tie

def safe_div(a,b):return float(a/b) if b else None
def summarize(p,o,prom,old):
    complete,d,sign,tie=profile_peaks(p,o,prom)
    finite=np.isfinite(d);near=finite&(d<=2);n=len(p);k=int(complete.sum());f=int(finite.sum());m=int(near.sum())
    # Emulate legacy missingness using NaN in explicitly unobserved positions.
    q=np.where(o,p,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        legacy=np.array([old(row,prom) for row in q])
    lf=np.isfinite(legacy);ln=lf&(np.abs(legacy)<=2)
    c=o[:,32]&np.isfinite(p[:,32])
    center_on=c&(p[:,32]>127)
    z=(p[complete,32]-p[complete].mean(1))/(p[complete].std(1)+1e-3)
    return {"sampled":n,"complete":k,"unsupported":n-k,"found":f,"tied_signs":int(tie.sum()),"near2":m,
            "near_among_found":safe_div(m,f),"near_among_complete":safe_div(m,k),
            "near_lower_all":safe_div(m,n),"near_upper_all":safe_div(m+n-k,n),
            "center_measured":int(c.sum()),"center_on_m7":int(center_on.sum()),
            "center_on_among_measured":safe_div(int(center_on.sum()),int(c.sum())),
            "complete_CT_z_mean":float(z.mean()) if len(z) else None,
            "legacy_found":int(lf.sum()),"legacy_near2":int(ln.sum()),
            "legacy_near_among_found":safe_div(int(ln.sum()),int(lf.sum()))}

def reversal(p,o,prom):
    a=profile_peaks(p,o,prom);b=profile_peaks(p[:,::-1],o[:,::-1],prom)
    return bool(np.array_equal(a[0],b[0]) and np.allclose(a[1],b[1],equal_nan=True,atol=1e-9,rtol=0)
                and np.allclose(a[2],-b[2],equal_nan=True,atol=1e-9,rtol=0) and np.array_equal(a[3],b[3]))
def synthetic(old):
    x=OFFS
    unique=30+150*np.exp(-((x-3)/1.2)**2)
    twin=20+180*np.exp(-((x-4)/.9)**2)+180*np.exp(-((x+4)/.9)**2)
    obs=np.ones((1,65),bool)
    u=profile_peaks(unique[None],obs,8)
    t=profile_peaks(twin[None],obs,8)
    rise=20+5*(x+16);cens=(x<=1)[None]
    incomplete=summarize(rise[None],cens,8,old)
    den=np.zeros((100,65),float);den[0]=10+200*np.exp(-(x/1.1)**2);mask=np.zeros_like(den,bool);mask[:10]=True
    summary=summarize(den,mask,8,old)
    rng=np.random.default_rng(4801);random=rng.uniform(0,200,(64,65));ro=np.ones_like(random,bool)
    tests={
      "unique_peak_reflection":bool(u[2][0]==3 and reversal(unique[None],obs,8)),
      "equal_distance_tie":bool(t[1][0]==4 and t[3][0] and np.isnan(t[2][0]) and old(twin,8)==old(twin[::-1],8) and reversal(twin[None],obs,8)),
      "incomplete_artificial_peak":bool(incomplete["complete"]==0 and incomplete["found"]==0 and incomplete["legacy_found"]==1 and incomplete["legacy_near2"]==1),
      "explicit_denominators":bool(summary["near_lower_all"]==.01 and summary["near_among_complete"]==.1 and summary["near_among_found"]==1 and summary["near_upper_all"]==.91),
      "observed_zero_not_missing":bool(summary["center_measured"]==10 and summary["complete"]==10 and summary["unsupported"]==90),
      "random_reversal":reversal(random,ro,8) and reversal(random,ro,60)}
    return {"gates":tests,"incomplete_fixture":incomplete,"denominator_fixture":summary,
            "legacy_tie_offsets":[float(old(twin,8)),float(old(twin[::-1],8))],"pass":all(tests.values())}

def real(a,old):
    bridge=module("adaptive_bridge",HERE.parent/"shared_renderer_bridge"/"run.py")
    surf=module("shared_surface",a.shared/"surface.py")
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    chart=bridge.ChartBridge(bridge.load_atlas(a.atlas,bridge.PIN),0,surf.OriginalSurface)
    budget=gm.ByteBudget(80*1024**2);rng=np.random.default_rng(4801);rows=[]
    with gm.BudgetedRemoteZarrV2(bridge.CT_URL,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=15) as ct,gm.BudgetedRemoteZarrV2(M7_URL,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=15) as m7:
        if tuple(ct.shape)!=tuple(m7.shape) or ct.dtype!=np.uint8 or m7.dtype!=np.uint8:raise ValueError("Native CT/M7 frames differ")
        for i,(y,x) in enumerate(((1774,2773),(2173,4558))):
            p=chart.sample_grid(((x+.5)/20,(y+.5)/20),(.4,.4),(8,8))
            xy=p["xyz"].reshape(-1,3);nv=p["normals"].reshape(-1,3);valid=p["valid"].ravel()
            if valid.sum()<48:raise ValueError("Insufficient valid original-face point samples")
            ctrl=rng.uniform(-8,8,len(xy))
            row={"index":i,"geometry_points":int(valid.sum()),"conditions":[]}
            for name,shift in (("fit",np.zeros(len(xy))),("control",ctrl)):
                coords=xy[:,None]+(OFFS[None,:]+shift[:,None])[:,:,None]*nv[:,None]
                q={"condition":name,"sources":{}}
                for kind,reader,prom in (("CT",ct,8.),("M7",m7,60.)):
                    vals,ok=gm.precise_sample_trilinear(reader,coords,return_support=True);ok&=valid[:,None]
                    r=summarize(vals,ok,prom,old)
                    r["reflection_exact"]=reversal(vals,ok,prom)
                    r["profile_sha256"]=hashlib.sha256(vals.tobytes()).hexdigest()
                    r["support_sha256"]=hashlib.sha256(ok.tobytes()).hexdigest()
                    if name=="fit":
                        cens=ok&(OFFS[None,:]<=1)
                        r["designed_tail_censoring"]=summarize(vals,cens,prom,old)
                    q["sources"][kind]=r
                row["conditions"].append(q)
            rows.append(row);print("CASE_JSON="+json.dumps(row,separators=(",",":")),flush=True)
    allq=[r for row in rows for c in row["conditions"] for r in c["sources"].values()]
    return {"cases":rows,"native_CT_M7_bytes":budget.used,"requests":budget.requests,"pass":len(rows)==2 and all(q["complete"]>=48 and q["reflection_exact"] and q.get("designed_tail_censoring",{}).get("complete",0)==0 for q in allq)}

def main():
    p=argparse.ArgumentParser()
    for k in ("legacy","atlas","shared","gemini","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    a=p.parse_args();t=time.monotonic();old=legacy_function(a.legacy);s=synthetic(old)
    r={"schema":"adaptive.profile_gate_audit.v1","script_sha256":sha(__file__),"protocol_sha256":sha(HERE/"protocol.md"),"legacy_sha256":sha(a.legacy),"synthetic":s,"pass":False}
    print("SYNTHETIC_JSON="+json.dumps(s,separators=(",",":")),flush=True)
    if s["pass"]:
        try:r["real"]=real(a,old);r["pass"]=r["real"]["pass"]
        except Exception as e:r["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    else:r["stop"]="synthetic gate failed"
    r["wall_seconds"]=time.monotonic()-t
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");print("RESULT_JSON="+json.dumps(r,separators=(",",":")),flush=True)
if __name__=="__main__":main()
