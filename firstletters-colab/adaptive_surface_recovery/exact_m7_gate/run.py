#!/usr/bin/env python3
"""Exact native-pixel M7 gate; shared legacy rule, explicit observed support."""
import argparse,hashlib,importlib.util,json,math,pathlib,sys,time
from collections import deque
import numpy as np
from scipy import ndimage as ndi
HERE=pathlib.Path(__file__).resolve().parent
M7_URL="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2.zarr"
DEPTHS=np.arange(-20,21,dtype=float)

def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def gate_exact(R,M,observed,um,block_px=None):
    """No decimation; same block lattice as old q=4 by default.

    observed: explicit boolean geometry AND full CT/M7 source support.
    CT centre >0 is the separate historical tissue proxy. A zero is not
    automatically missing; the support mask must come from the producer.
    """
    R=np.asarray(R);M=np.asarray(M);observed=np.asarray(observed)
    if R.shape!=M.shape or R.ndim!=3 or R.shape[0]!=41 or observed.dtype!=np.bool_ or observed.shape!=R.shape[1:]:
        raise ValueError("Expected matched 41-plane arrays and explicit boolean observed mask")
    if R.dtype!=np.uint8 or M.dtype!=np.uint8 or not np.isfinite(um) or um<=0:raise ValueError("Expected uint8 and positive native spacing")
    h,w=observed.shape
    if not h or not w:raise ValueError("empty raster")
    b=int(block_px) if block_px is not None else 4*max(1,round(1000/um/4))
    if b<1:raise ValueError("invalid block lattice")
    valid=observed&(R[20]>0)
    cls=np.zeros((h,w),np.uint8)
    # Bound the depth-channel temporary allocation by slicing rows.
    for y in range(0,h,256):
        stop=min(h,y+256);sl=slice(y,stop)
        z=M[11:30,sl]>127
        on=z[6:13].any(0)
        runs=z[0].astype(np.uint8)+((~z[:-1])&z[1:]).sum(0)
        cls[sl]=np.where(valid[sl],np.where(on,np.where(runs<=1,3,2),1),0)
    on=cls>=2;clean=cls==3
    starts_y=np.arange(0,h,b);starts_x=np.arange(0,w,b)
    def block_count(m):
        return np.add.reduceat(np.add.reduceat(m.astype(np.int64),starts_y,axis=0),starts_x,axis=1)
    v=block_count(valid);o=block_count(on);c=block_count(clean)
    # Integer comparison gives exact 60%; no floating cutoff error.
    ob=(v>0)&(5*o>=3*v);cb=(v>0)&(5*c>=3*v)
    mask=ob[np.arange(h)//b][:,np.arange(w)//b]&valid
    labels,n=ndi.label(mask)
    areas=np.bincount(labels.ravel());core=int(areas[1:].max(initial=0))
    va=int(valid.sum());oa=int((v*ob).sum());ca=int((v*cb).sum())
    if int(mask.sum())!=oa:raise ArithmeticError("block area accounting mismatch")
    px=(um*1e-4)**2
    return {"block_native_px":b,"observed_pixels":int(observed.sum()),"valid_pixels":va,
            "class_counts":np.bincount(cls.ravel(),minlength=4).tolist(),
            "raw_on_fraction":int(on.sum())/max(va,1),"raw_clean_fraction":int(clean.sum())/max(va,1),
            "on_sheet_pixels":oa,"clean_block_pixels":ca,"core_pixels":core,
            "valid_cm2":va*px,"on_sheet_fraction":oa/max(va,1),"clean_fraction":ca/max(va,1),
            "on_sheet_cm2":oa*px,"core_cm2":core*px,"gate":"ink" if core*px>=4 else "uninformative"}

def oracle(R,M,observed,um,b):
    """Independent pixel loops, block loops, BFS; deliberately slow reference."""
    h,w=observed.shape;cls=np.zeros((h,w),np.uint8)
    for y in range(h):
        for x in range(w):
            if not observed[y,x] or R[20,y,x]==0:continue
            near=any(int(M[k,y,x])>127 for k in range(17,24))
            if not near:cls[y,x]=1;continue
            prev=False;runs=0
            for k in range(11,30):
                now=int(M[k,y,x])>127
                if now and not prev:runs+=1
                prev=now
            cls[y,x]=3 if runs<=1 else 2
    mask=np.zeros((h,w),bool);oa=ca=0
    for y in range(0,h,b):
        for x in range(0,w,b):
            z=cls[y:y+b,x:x+b];v=int(np.count_nonzero(z))
            o=int(np.count_nonzero(z>=2));c=int(np.count_nonzero(z==3))
            if v and o/v>=.6:
                mask[y:y+b,x:x+b]=z>0;oa+=v
            if v and c/v>=.6:ca+=v
    core=0
    for y,x in zip(*np.nonzero(mask.copy())):
        if not mask[y,x]:continue
        queue=deque([(int(y),int(x))]);mask[y,x]=False;n=0
        while queue:
            yy,xx=queue.popleft();n+=1
            for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
                a,c=yy+dy,xx+dx
                if 0<=a<h and 0<=c<w and mask[a,c]:mask[a,c]=False;queue.append((a,c))
        core=max(core,n)
    return {"class_counts":np.bincount(cls.ravel(),minlength=4).tolist(),"valid_pixels":int(np.count_nonzero(cls)),
            "on_sheet_pixels":oa,"clean_block_pixels":ca,"core_pixels":core}

def compare(R,M,observed,um,legacy,b=None):
    ex=gate_exact(R,M,observed,um,b)
    ref=oracle(R,M,observed,um,ex["block_native_px"])
    exact=all(ex[k]==v for k,v in ref.items())
    # Legacy has no support argument. Mask both arrays identically.
    R2=R.copy();M2=M.copy();R2[:,~observed]=0;M2[:,~observed]=0
    lg=legacy.gate(R2,M2,um)
    rev=gate_exact(R[::-1],M[::-1],observed,um,b)
    return {"exact":ex,"legacy":lg,"oracle":ref,"oracle_match":bool(exact),"depth_reversal_match":bool(ex==rev)}

def synthetic(legacy):
    rng=np.random.default_rng(4701);rows=[]
    for i in range(12):
        h=int(rng.integers(13,65));w=int(rng.integers(13,65))
        R=rng.integers(0,256,(41,h,w),dtype=np.uint8);M=np.zeros_like(R)
        for _ in range(5):
            dep=int(rng.integers(11,30));M[dep]=np.where(rng.random((h,w))<.45,255,0)
        obs=rng.random((h,w))>.13
        q=compare(R,M,obs,100.,legacy)
        rows.append({"kind":"random","shape":[h,w],"oracle_match":q["oracle_match"],"depth_reversal_match":q["depth_reversal_match"]})
    fixtures=[]
    for name,h,w in (("partial_edge",199,199),("sampled_only",200,200),("sampled_missing",200,200),("missing_support",200,200),("nondivisible_mask",121,119)):
        R=np.full((41,h,w),150,np.uint8);M=np.zeros_like(R);obs=np.ones((h,w),bool)
        M[20]=255
        if name=="sampled_only":
            M[20]=0;M[20,::4,::4]=255
        if name=="sampled_missing":M[20,::4,::4]=0
        if name=="missing_support":obs[:,:7]=False
        if name=="nondivisible_mask":
            R[:]=0;R[:,::4,::4]=150;M[20]=0;M[20,::4,::4]=255
        r=compare(R,M,obs,100.,legacy);r["kind"]=name;fixtures.append(r)
    f={r["kind"]:r for r in fixtures}
    gates={
      "oracle_and_reversal":all(r["oracle_match"] and r["depth_reversal_match"] for r in rows+fixtures),
      "edge_overcount":f["partial_edge"]["legacy"]["gate"]=="ink" and f["partial_edge"]["exact"]["gate"]=="uninformative",
      "alias_false_accept":f["sampled_only"]["legacy"]["gate"]=="ink" and f["sampled_only"]["exact"]["gate"]=="uninformative" and f["sampled_only"]["exact"]["raw_on_fraction"]==.0625,
      "alias_false_reject":f["sampled_missing"]["legacy"]["gate"]=="uninformative" and f["sampled_missing"]["exact"]["gate"]=="ink" and f["sampled_missing"]["exact"]["raw_on_fraction"]==.9375,
      "support_not_filled":f["missing_support"]["exact"]["valid_pixels"]==200*193 and f["missing_support"]["exact"]["gate"]=="uninformative",
      "different_lattices":f["nondivisible_mask"]["legacy"]["on_sheet_fraction"]!=f["nondivisible_mask"]["exact"]["on_sheet_fraction"]}
    return {"random":rows,"fixtures":fixtures,"gates":gates,"pass":all(gates.values())}

def real(a,legacy):
    bridge=mod("adaptive_bridge",HERE.parent/"shared_renderer_bridge"/"run.py")
    surf=mod("shared_surface",a.shared/"surface.py")
    sys.path.insert(0,str(a.gemini));import grow_m7_patch as gm
    chart=bridge.ChartBridge(bridge.load_atlas(a.atlas,bridge.PIN),0,surf.OriginalSurface)
    budget=gm.ByteBudget(80*1024**2);rows=[]
    with gm.BudgetedRemoteZarrV2(bridge.CT_URL,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=15) as ct, gm.BudgetedRemoteZarrV2(M7_URL,level="0",budget=budget,workers=1,cache_mb=64,max_retries=1,timeout=15) as m7:
        if tuple(ct.shape)!=tuple(m7.shape) or ct.dtype!=np.uint8 or m7.dtype!=np.uint8:
            raise ValueError("CT/M7 native shapes or types differ")
        for i,(y,x) in enumerate(((1774,2773),(2173,4558))):
            origin=((x+.5)/20,(y+.5)/20);shape=(69,67);step=(.05,.05)
            fp=chart.footprint_support(origin,step,shape);p=chart.sample_grid(origin,step,shape)
            geom=fp["valid"]&p["valid"]
            R=np.zeros((41,*shape),np.uint8);M=np.zeros_like(R);obs=geom.copy()
            for j,z in enumerate(DEPTHS):
                coords=p["xyz"]+z*p["normals"]
                cv,cs=gm.precise_sample_trilinear(ct,coords,return_support=True)
                mv,ms=gm.precise_sample_trilinear(m7,coords,return_support=True)
                obs&=cs&ms
                R[j]=np.clip(np.rint(cv),0,255).astype(np.uint8)
                M[j]=np.clip(np.rint(mv),0,255).astype(np.uint8)
            r=compare(R,M,obs,8.64,legacy);r.update(index=i,shape=list(shape),geometry_pixels=int(geom.sum()),
                CT_sha256=hashlib.sha256(R.tobytes()).hexdigest(),M7_sha256=hashlib.sha256(M.tobytes()).hexdigest(),
                observation_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),minimum_eligible_pass=r["exact"]["valid_pixels"]>=1000)
            rows.append(r);print("CASE_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    return {"cases":rows,"native_CT_M7_bytes":budget.used,"native_requests":budget.requests,"cap_bytes":budget.cap,
            "source_CT":bridge.CT_URL,"source_M7":M7_URL,"shape_zyx":list(ct.shape),
            "pass":len(rows)==2 and all(r["oracle_match"] and r["depth_reversal_match"] and r["minimum_eligible_pass"] for r in rows)}

def main():
    p=argparse.ArgumentParser()
    for k in ("atlas","shared","gemini","legacy","output"):p.add_argument("--"+k,type=pathlib.Path,required=True)
    a=p.parse_args();t=time.monotonic()
    legacy=mod("legacy_gate",a.legacy);s=synthetic(legacy)
    result={"schema":"adaptive.exact_m7_gate.v1","script_sha256":sha(__file__),"protocol_sha256":sha(HERE/"protocol.md"),"legacy_sha256":sha(a.legacy),"synthetic":s,"pass":False}
    print("SYNTHETIC_JSON="+json.dumps(s,separators=(",",":")),flush=True)
    if s["pass"]:
        try:result["real"]=real(a,legacy);result["pass"]=result["real"]["pass"]
        except Exception as e:result["failure"]=f"{type(e).__name__}: {str(e)[:500]}"
    else:result["stop"]="synthetic gate failed"
    result["wall_seconds"]=time.monotonic()-t
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":main()
