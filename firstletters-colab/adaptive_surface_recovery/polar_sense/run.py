#!/usr/bin/env python3
"""Single-slice polar ridge-flow diagnostic; no ink model or anatomical claim."""
import argparse, concurrent.futures as cf, hashlib, json, math, os, pathlib, re, threading, time
import numpy as np
from scipy import ndimage as nd
import requests

BASE="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
CASES=[
 ("PHerc0175A","20250521115057-8.640um-1.2m-116keV-masked.zarr",7650),
 ("PHerc1203","20250820131727-9.362um-1.2m-113keV-masked.zarr",9700),
 ("PHerc0846A","20250728152254-9.362um-1.2m-113keV-masked.zarr",3100)]
N=2048
MAX_BYTES=160*1024**2
MAX_REQUESTS=4500
_tls=threading.local()

class Budget:
 def __init__(self):
  self.bytes=0;self.requests=0;self.lock=threading.Lock()
 def charge(self,n=0,request=False):
  with self.lock:
   self.bytes+=n;self.requests+=int(request)
   if self.bytes>MAX_BYTES or self.requests>MAX_REQUESTS:raise RuntimeError("budget exceeded")

def get(url,budget,range_=None,allow404=False):
 budget.charge(request=True)
 if not hasattr(_tls,"session"):_tls.session=requests.Session()
 h={"Accept-Encoding":"identity"}
 if range_ is not None:h["Range"]="bytes=%d-%d"%range_
 with _tls.session.get(url,headers=h,timeout=(8,25),stream=True) as r:
  if allow404 and r.status_code==404:return None
  expected=206 if range_ is not None else 200
  if r.status_code!=expected:raise RuntimeError(f"HTTP {r.status_code}, expected {expected} for source")
  if range_ is not None:
   a,b=range_;cr=r.headers.get("Content-Range","")
   if not re.fullmatch(fr"bytes {a}-{b}/[0-9]+",cr):raise RuntimeError("wrong Content-Range")
   cap=b-a+1
  else:cap=2**20
  data=bytearray()
  for part in r.iter_content(65536):
   budget.charge(len(part));data.extend(part)
   if len(data)>cap:raise RuntimeError("response exceeded expected size")
  if range_ is not None and len(data)!=cap:raise RuntimeError("short range")
  return bytes(data)

def read_plane(case,budget):
 scroll,vol,z=case;root=f"{BASE}/{scroll}/volumes/{vol}"
 attr=get(root+"/.zattrs",budget);meta=get(root+"/1/.zarray",budget)
 a=json.loads(attr);m=json.loads(meta)
 ms=a["multiscales"][0];axes=ms.get("axes")
 names=[x["name"] if isinstance(x,dict) else x for x in axes] if axes else []
 if names!=["z","y","x"]:raise RuntimeError(f"unsupported axis order: {names}")
 ds=[d for d in ms["datasets"] if str(d["path"])=="1"]
 if len(ds)!=1:raise RuntimeError("missing level 1 transform")
 tr=ds[0].get("coordinateTransformations",[])
 scales=[t["scale"] for t in tr if t.get("type")=="scale"]
 base=[d for d in ms["datasets"] if str(d["path"])=="0"][0]
 s0=[t["scale"] for t in base.get("coordinateTransformations",[]) if t.get("type")=="scale"]
 if len(scales)!=1 or len(s0)!=1 or not np.allclose(np.array(scales[0])/np.array(s0[0]),2):raise RuntimeError("not an isotropic factor-2 level")
 if any(t.get("type")!="scale" for t in tr+base.get("coordinateTransformations",[])):raise RuntimeError("unhandled translation or transform")
 if m.get("compressor") is not None or m.get("filters") not in (None,[]) or m.get("order")!="C" or np.dtype(m["dtype"])!=np.dtype("uint8"):raise RuntimeError("not uncompressed uint8 C-order")
 Z,Y,X=map(int,m["shape"]);cz,cy,cx=map(int,m["chunks"]);sep=m.get("dimension_separator",".")
 zz=z//2
 if zz>=Z:raise RuntimeError("z outside volume")
 kz,iz=divmod(zz,cz);a0=iz*cy*cx;b0=a0+cy*cx-1
 out=np.zeros((Y,X),np.uint8);valid=np.zeros((Y,X),bool)
 tasks=[(ky,kx) for ky in range(math.ceil(Y/cy)) for kx in range(math.ceil(X/cx))]
 before=(budget.bytes,budget.requests)
 def task(kykx):
  ky,kx=kykx;key=sep.join(map(str,(kz,ky,kx)))
  raw=get(root+"/1/"+key,budget,(a0,b0),True)
  return ky,kx,raw
 with cf.ThreadPoolExecutor(max_workers=12) as ex:
  for ky,kx,raw in ex.map(task,tasks):
   if raw is None:continue
   yy=ky*cy;xx=kx*cx;h=min(cy,Y-yy);w=min(cx,X-xx)
   out[yy:yy+h,xx:xx+w]=np.frombuffer(raw,np.uint8).reshape(cy,cx)[:h,:w]
   valid[yy:yy+h,xx:xx+w]=True
 receipt={"shape_yx":[Y,X],"level":1,"z_native":z,"z_level":zz,"chunk_shape":list(m["chunks"]),"available_fraction":float(valid.mean()),"nonzero_fraction":float((out>0).mean()),"data_sha256":hashlib.sha256(out.tobytes()).hexdigest(),"availability_sha256":hashlib.sha256(valid.tobytes()).hexdigest(),"zarray_sha256":hashlib.sha256(meta).hexdigest(),"zattrs_sha256":hashlib.sha256(attr).hexdigest(),"slice_body_bytes":budget.bytes-before[0],"slice_requests":budget.requests-before[1]}
 return out,valid,receipt

def center_of(img,valid):
 mask=valid&(img>0)
 # Remove isolated tiny pieces and components touching frame edges only if
 # an interior component remains. Never use ink or target sense.
 labels,n=nd.label(mask)
 if not n:raise ValueError("no tissue")
 edges=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
 sizes=np.bincount(labels.ravel());sizes[0]=0
 keep=np.ones(len(sizes),bool);keep[edges]=False;keep[0]=False
 if sizes[keep].sum()>=.5*sizes.sum():mask=mask&keep[labels]
 y,x=nd.center_of_mass(mask)
 if not np.isfinite([y,x]).all():raise ValueError("center undefined")
 return y,x

def max_cyclic_gap(ok):
 if not ok.any():return len(ok)
 a=np.r_[ok,ok];inds=np.flatnonzero(a)
 return int(np.diff(inds).max(initial=1)-1)

def flow(img,valid,center):
 cy,cx=center;h,w=img.shape
 R=int(math.ceil(np.hypot(max(cx,w-1-cx),max(cy,h-1-cy))))
 # Clip to physical image; ignore unmeasured and non-tissue interpolation.
 ang=np.arange(N)*2*np.pi/N
 r=np.arange(R,dtype=np.float32)
 y=cy+r[:,None]*np.sin(ang);x=cx+r[:,None]*np.cos(ang)
 raw=nd.map_coordinates(img.astype(np.float32),[y,x],order=1,mode="constant",cval=0)
 support=nd.map_coordinates((valid&(img>0)).astype(np.float32),[y,x],order=1,mode="constant",cval=0)>1-1e-6
 extent=np.max(np.where(support,np.arange(R)[:,None],0),axis=0)
 ext=float(np.median(extent));del y,x
 if ext<150:raise ValueError("insufficient tissue radius")
 hp=raw-nd.gaussian_filter1d(raw,3,axis=0,mode="nearest")
 shifted=np.roll(hp,-1,axis=1);sv=np.roll(support,-1,axis=1)
 centers=np.arange(max(40,math.ceil(.15*ext)),min(R-40,math.floor(.8*ext))+1,32)
 estimates=np.full((len(centers),N),np.nan,np.float32)
 for k,c in enumerate(centers):
  c=int(c);a=hp[c-32:c+32];v=support[c-32:c+32]
  da=a-a.mean(axis=0);sa=np.sqrt(np.mean(da*da,axis=0))
  scores=[]
  for d in range(-8,9):
   b=shifted[c-32+d:c+32+d];vb=sv[c-32+d:c+32+d]
   db=b-b.mean(axis=0);sb=np.sqrt(np.mean(db*db,axis=0))
   corr=np.divide(np.mean(da*db,axis=0),sa*sb,out=np.full(N,-2,np.float32),where=(sa*sb)>1e-8)
   corr[~(v.all(axis=0)&vb.all(axis=0)&(sa>=2)&(sb>=2))]=-2
   scores.append(corr)
  scores=np.stack(scores);best=np.argmax(scores,axis=0);cols=np.arange(N);peak=scores[best,cols]
  sep=np.abs(np.arange(17)[:,None]-best)>=2
  alt=np.where(sep,scores,-2).max(axis=0)
  ok=(best>0)&(best<16)&(peak>=.55)&(alt<peak-.03)
  b=np.clip(best,1,15);l=scores[b-1,cols];m=scores[b,cols];rr=scores[b+1,cols];den=l-2*m+rr
  sub=np.divide(.5*(l-rr),den,out=np.zeros(N,np.float32),where=np.abs(den)>1e-8)
  val=best-8+np.clip(sub,-.5,.5)
  estimates[k,ok]=val[ok]
 def combine(rows,min_n):
  if not len(rows):return {"pitch_level1":None,"accepted_fraction":0.,"max_missing_gap":N}
  n=np.isfinite(rows).sum(axis=0);ok=n>=min_n
  med=np.zeros(N)
  if ok.any():med[ok]=np.nanmedian(rows[:,ok],axis=0)
  gap=max_cyclic_gap(ok);coverage=float(ok.mean());p=None
  if coverage>=.95 and gap<=8:
   ids=np.flatnonzero(ok);miss=np.flatnonzero(~ok)
   med[miss]=np.interp(miss,np.r_[ids-N,ids,ids+N],np.tile(med[ids],3))
   p=float(med.sum())
  return {"pitch_level1":p,"accepted_fraction":coverage,"max_missing_gap":gap,"median_valid_windows":float(np.median(n))}
 total=combine(estimates,3)
 total["radial_bands"]=[combine(a,2) for a in np.array_split(estimates,4)]
 total["radial_window_count"]=len(centers)
 return total

def signed(x):
 return 0 if x is None or abs(x)<2.5 or abs(x)>30 else int(np.sign(x))
def summary(img,valid,center,official=None):
 y,x=center;offsets=[(0,0),(0,64),(0,-64),(64,0),(-64,0)]
 rows=[]
 for dy,dx in offsets:
  rows.append(flow(img,valid,(y+dy,x+dx)))
 mir=flow(img[:,::-1].copy(),valid[:,::-1].copy(),(y,img.shape[1]-1-x))
 p=rows[0]["pitch_level1"];mp=mir["pitch_level1"];s=signed(p)
 band=[signed(b["pitch_level1"]) for b in rows[0]["radial_bands"]]
 mirror_ok=s!=0 and signed(mp)==-s and abs(p+mp)<=max(2,.2*abs(p))
 center_ok=s!=0 and all(signed(r["pitch_level1"])==s for r in rows)
 band_ok=band.count(s)>=2 and -s not in band if s else False
 result={"center_and_displacements":rows,"mirrored":mir,"mirror_ok":bool(mirror_ok),"center_ok":bool(center_ok),"radial_band_ok":bool(band_ok),"local_proxy_sign":s if mirror_ok and center_ok and band_ok else None}
 if official is not None:result["official_center_result"]=flow(img,valid,official)
 return result

def self_test():
 # An analytic continuous spiral phase with elliptic cross-sections; the
 # seam is periodic because pitch is the radial period. 
 y,x=np.indices((768,768),dtype=np.float32);x-=383.5;y-=383.5
 rows=[]
 for ell in [1.,1.25]:
  r=np.sqrt((x/ell)**2+y*y);theta=np.arctan2(y,x/ell);valid=(r<290)
  for pitch in [-10,0,10]:
   phase=2*np.pi*r/10 - (0 if pitch==0 else np.sign(pitch))*theta
   img=np.where(valid,128+45*np.cos(phase),0).astype(np.uint8)
   for center in [(383.5,383.5),(395.5,371.5)]:
    p=flow(img,np.ones_like(valid),center)["pitch_level1"]
    ok=p is not None and (abs(p)<=2 if pitch==0 else np.sign(p)==np.sign(pitch) and abs(p-pitch)<=4)
    rows.append({"ellipticity":ell,"expected":pitch,"displaced":center[0]!=383.5,"estimated":p,"pass":bool(ok)})
   if pitch:
    m=flow(img[:,::-1].copy(),np.ones_like(valid),(383.5,383.5))["pitch_level1"]
    ok=m is not None and np.sign(m)==-np.sign(pitch) and abs(m+pitch)<=4
    rows.append({"ellipticity":ell,"expected":-pitch,"mirror":True,"estimated":m,"pass":bool(ok)})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=pathlib.Path,required=True);p.add_argument("--official-center",type=pathlib.Path);p.add_argument("--synthetic-only",action="store_true");a=p.parse_args()
 start=time.monotonic();budget=Budget()
 report={"schema":"adaptive.polar_sense.v1","script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),"synthetic":self_test(),"cases":[],"pass":False}
 report["synthetic_pass"]=all(r["pass"] for r in report["synthetic"])
 print("SYNTHETIC_JSON="+json.dumps(report["synthetic"],separators=(",",":")),flush=True)
 if report["synthetic_pass"] and not a.synthetic_only:
  for case in CASES:
   row={"scroll":case[0],"source_volume":case[1]}
   try:
    img,valid,rec=read_plane(case,budget);row["source"]=rec
    center=center_of(img,valid);official=None
    if case[0]=="PHerc1203" and a.official_center:
     b=a.official_center.read_bytes();blob=hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
     if blob!="2496c8ebf6dcd8d8d56e71cabc86360916d19c7f":raise ValueError("official center checksum")
     cp=json.loads(b)["control_points"];z=np.array([p["z"] for p in cp])
     official=(np.interp(case[2],z,[p["y"] for p in cp])/2,np.interp(case[2],z,[p["x"] for p in cp])/2)
    row["diagnostic"]=summary(img,valid,center,official)
   except Exception as e:row["failure"]=f"{type(e).__name__}: {str(e)[:400]}"
   report["cases"].append(row);print("CASE_JSON="+json.dumps(row,separators=(",",":"),allow_nan=False),flush=True)
  control=report["cases"][0].get("diagnostic",{}).get("local_proxy_sign")==-1
  report["control_pass"]=control
  for row in report["cases"]:
   d=row.get("diagnostic",{});s=d.get("local_proxy_sign")
   if "official_center_result" in d and signed(d["official_center_result"]["pitch_level1"])!=s:s=None
   row["calibrated_proxy_verdict"]=("CW" if s==1 else "ACW") if control and s in (-1,1) else "INCONCLUSIVE"
  report["pass"]=bool(control and all(r["calibrated_proxy_verdict"]!="INCONCLUSIVE" for r in report["cases"]))
 else:report["stop"]="synthetic gate failed before real CT" if not report["synthetic_pass"] else "synthetic only"
 report["charged_bytes"]=budget.bytes;report["requests"]=budget.requests;report["wall_seconds"]=time.monotonic()-start
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print("RESULT_JSON="+json.dumps(report,separators=(",",":"),allow_nan=False),flush=True)
if __name__=="__main__":main()
