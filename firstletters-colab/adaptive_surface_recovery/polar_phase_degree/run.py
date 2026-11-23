#!/usr/bin/env python3
"""Q12 polar phase degree: integer winding of radial analytic signal."""
import argparse, hashlib, importlib.util, json, math, pathlib, time
import numpy as np
from scipy import ndimage as nd, fft

HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("polar044",HERE.parent/"polar_sense"/"run.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
BANDS=((4.,24.),(6.,32.))
N=2048
OFFSETS=((0,0),(0,64),(0,-64),(64,0),(-64,0))

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def rays(img,valid,center):
    cy,cx=center;h,w=img.shape
    R=int(math.ceil(np.hypot(max(cx,w-1-cx),max(cy,h-1-cy))))+1
    a=np.arange(N)*2*np.pi/N
    r=np.arange(R,dtype=np.float32)
    yy=cy+r[:,None]*np.sin(a);xx=cx+r[:,None]*np.cos(a)
    raw=nd.map_coordinates(img.astype(np.float32),[yy,xx],order=1,mode="constant",cval=0)
    support=nd.map_coordinates((valid&(img>0)).astype(np.float32),[yy,xx],order=1,mode="constant",cval=0)>1-1e-6
    good=nd.minimum_filter1d(support.all(1).astype(np.uint8),size=65,mode="constant",cval=0)>0
    return raw,good

def degree(raw,good,band):
    R,A=raw.shape
    pad=64;L=R+2*pad
    f=fft.fftfreq(L)
    flo,fhi=1/band[1],1/band[0]
    # Cosine transition outside the passband, relative width 20%.
    w=np.zeros(L,np.float32)
    mid=(f>=flo)&(f<=fhi);w[mid]=2
    low=(f>.8*flo)&(f<flo)
    w[low]=1-np.cos(np.pi*(f[low]-.8*flo)/(.2*flo))
    high=(f>fhi)&(f<1.2*fhi)
    w[high]=1+np.cos(np.pi*(f[high]-fhi)/(.2*fhi))
    ids=np.flatnonzero(good)
    counts={"all_radii":R,"support_radii":len(ids),"amplitude_rejected":0,"phase_step_rejected":0,"integer_rejected":0}
    if not len(ids):
        return {"band":list(band),**counts,"accepted_radii":0,"degrees":{},"sign":None,"halves":[]}
    # Only retain analytic signals at complete supported radii. Angle blocks
    # keep memory bounded; FFT along radius is never mixed across angles.
    signal=np.empty((len(ids),A),np.complex64)
    for j in range(0,A,128):
        z=np.pad(raw[:,j:j+128],((pad,pad),(0,0)),mode="reflect")
        g=fft.ifft(fft.fft(z,axis=0)*w[:,None],axis=0)
        signal[:,j:j+128]=g[pad+ids]
    amp=np.abs(signal)
    amp_ok=np.isfinite(amp).all(1)&(amp.min(1)>=1)
    # complex products give wrapped increments, including the closing edge.
    d=np.angle(np.roll(signal,-1,axis=1)*np.conjugate(signal)).astype(np.float64)
    step_ok=np.isfinite(d).all(1)&(np.abs(d).max(1)<=np.pi/2)
    v=d.sum(1)/(2*np.pi);iv=np.rint(v)
    int_ok=np.isfinite(v)&(np.abs(v-iv)<=1e-5)
    ok=amp_ok&step_ok&int_ok
    counts.update(amplitude_rejected=int((~amp_ok).sum()),
                  phase_step_rejected=int((amp_ok&~step_ok).sum()),
                  integer_rejected=int((amp_ok&step_ok&~int_ok).sum()))
    iv=iv[ok].astype(int);rr=ids[ok]
    def stat(a,n):
        vals,cts=np.unique(a,return_counts=True)
        hist={str(int(v)):int(c) for v,c in zip(vals,cts)}
        z=len(a)
        pos=hist.get("-1",0)/max(z,1);neg=hist.get("1",0)/max(z,1)
        s=1 if z>=n and pos>=.6 and neg<=.1 else (-1 if z>=n and neg>=.6 and pos<=.1 else None)
        return {"accepted_radii":z,"degrees":hist,"CW_share":pos,"ACW_share":neg,"sign":s}
    tot=stat(iv,30)
    if len(rr):
        mid=(ids[0]+ids[-1])/2
        halves=[stat(iv[rr<=mid],15),stat(iv[rr>mid],15)]
    else:halves=[]
    return {"band":list(band),**counts,**tot,"halves":halves}

def evaluate(img,valid,center):
    raw,good=rays(img,valid,center)
    bands=[degree(raw,good,b) for b in BANDS]
    s=bands[0]["sign"]
    return {"bands":bands,"sign":s if s is not None and bands[1]["sign"]==s else None}

def synthetic():
    y,x=np.indices((640,640),dtype=np.float32);x-=319.5;y-=319.5
    rng=np.random.default_rng(451)
    cases=[]
    for ell in (1.,1.25):
        r=np.sqrt((x/ell)**2+y*y);theta=np.arctan2(y,x/ell)
        valid=r<232
        for pitch,period in ((-12,12),(12,12),(-16,16),(16,16),(0,12),(0,16)):
            for phase in (.37,1.19):
                intensity=128+(35+7*np.sin(theta+.2)+4*np.cos(3*theta))*np.cos(2*np.pi*r/period-np.sign(pitch)*theta+phase)+rng.normal(0,1,r.shape)
                img=np.where(valid,np.clip(np.rint(intensity),1,255),0).astype(np.uint8)
                for disp in ((0,0),(12,-12)):
                    center=(319.5+disp[0],319.5+disp[1])
                    got=evaluate(img,np.ones_like(valid),center)
                    def ok(d,s):
                        if s:return d["sign"]==s
                        return d["sign"] is None and all(b["accepted_radii"]>=30 and b["degrees"].get("0",0)/b["accepted_radii"]>=.6 for b in d["bands"])
                    expected=int(np.sign(pitch))
                    cases.append({"ell":ell,"pitch":pitch,"phase":phase,"offset":list(disp),"result":got,"pass":bool(ok(got,expected))})
                if phase==.37 and pitch!=0:
                    for name,im,sg in (("mirror",img[:,::-1].copy(),-int(np.sign(pitch))),("rotation90",np.rot90(img).copy(),int(np.sign(pitch)))):
                        got=evaluate(im,np.ones_like(valid),(319.5,319.5))
                        cases.append({"ell":ell,"pitch":pitch,"phase":phase,"transform":name,"result":got,"pass":bool(ok(got,sg))})
    return cases

def real_case(img,valid,center,official=None):
    cy,cx=center
    rows=[evaluate(img,valid,(cy+dy,cx+dx)) for dy,dx in OFFSETS]
    m=evaluate(img[:,::-1].copy(),valid[:,::-1].copy(),(cy,img.shape[1]-1-cx))
    s=rows[0]["sign"]
    center_ok=s is not None and all(r["sign"]==s for r in rows)
    mirror_ok=s is not None and m["sign"]==-s
    radial_ok=s is not None and all(len(b["halves"])==2 and all(q["sign"]==s for q in b["halves"]) for b in rows[0]["bands"])
    d={"centers":rows,"mirrored":m,"center_ok":bool(center_ok),"mirror_ok":bool(mirror_ok),"radial_ok":bool(radial_ok)}
    official_ok=True
    if official is not None:
        d["official"]=evaluate(img,valid,official);official_ok=d["official"]["sign"]==s
    d["official_ok"]=bool(official_ok)
    d["proxy_sign"]=s if center_ok and mirror_ok and radial_ok and official_ok else None
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=pathlib.Path,required=True);p.add_argument("--official-center",type=pathlib.Path,required=True);a=p.parse_args()
    start=time.monotonic();budget=old.Budget()
    tests=synthetic()
    r={"schema":"adaptive.polar_phase_degree.v1","script_sha256":sha(__file__),"source_reader_sha256":sha(HERE.parent/"polar_sense"/"run.py"),"synthetic":tests,"synthetic_pass":all(x["pass"] for x in tests),"cases":[],"pass":False}
    print("SYNTHETIC_COUNTS="+json.dumps({"pass":sum(x["pass"] for x in tests),"total":len(tests),"failed":[{k:v for k,v in x.items() if k!="result"} for x in tests if not x["pass"]]}),flush=True)
    if r["synthetic_pass"]:
        for case in old.CASES:
            row={"scroll":case[0],"source_volume":case[1]}
            try:
                img,valid,rec=old.read_plane(case,budget);row["source"]=rec
                center=old.center_of(img,valid);official=None
                if case[0]=="PHerc1203":
                    b=a.official_center.read_bytes();blob=hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
                    if blob!="2496c8ebf6dcd8d8d56e71cabc86360916d19c7f":raise ValueError("official center checksum")
                    cp=json.loads(b)["control_points"];z=np.array([p["z"] for p in cp])
                    official=(np.interp(case[2],z,[p["y"] for p in cp])/2,np.interp(case[2],z,[p["x"] for p in cp])/2)
                row["diagnostic"]=real_case(img,valid,center,official)
            except Exception as e:row["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
            r["cases"].append(row);print("CASE_JSON="+json.dumps(row,separators=(",",":"),allow_nan=False),flush=True)
        control=r["cases"][0].get("diagnostic",{}).get("proxy_sign")==-1
        r["control_pass"]=control
        for row in r["cases"]:
            s=row.get("diagnostic",{}).get("proxy_sign")
            row["calibrated_proxy_verdict"]=("CW" if s==1 else "ACW") if control and s in (-1,1) else "INCONCLUSIVE"
        r["pass"]=bool(control and all(row["calibrated_proxy_verdict"]!="INCONCLUSIVE" for row in r["cases"]))
    else:r["stop"]="synthetic gate failed before real CT"
    r.update(charged_bytes=budget.bytes,requests=budget.requests,wall_seconds=time.monotonic()-start)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    # Full receipt is retained as artifact; log only compact synthetic outcomes
    # and complete aggregate real results to fit connector log limit.
    print("RESULT_JSON="+json.dumps({**r,"synthetic":[{k:v for k,v in x.items() if k!="result"}|{"band_histograms":[b["degrees"] for b in x["result"]["bands"]],"band_accepted":[b["accepted_radii"] for b in x["result"]["bands"]],"estimated_sign":x["result"]["sign"]} for x in tests]},separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":main()
