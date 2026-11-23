"""Bounded acquisition of native CT on public anatomical atlases. Contains no detector."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from xml.etree import ElementTree

import numpy as np
import requests
from scipy.ndimage import map_coordinates
from atlas125 import NAMES, SIDE, DEPTH, chart, choose_chart, dump, fixed_survey, node_normals, summarize, xyz_from, surface_area

BUCKET="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
CT="PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr"
M7="PHerc0125/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr"
ATLAS="https://raw.githubusercontent.com/pscamillo/vesuvius-eligible-meshes/66000516c97e043b696ce874f145e29638109c91/meshes/PHerc0125"
UPSTREAM="https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/e8370f1f88a211166056df62ea6fcfffe4a1063e/mesh/snap_meshes.py"
UPSTREAM_SHA="253e48b31072be8c95b890715b80e90e4efbb34b5b0687a63b7d4a0a7a87857d"
CACHE=Path("acq-cache")
OFF=np.arange(-24,24.001,.5)
MAX_BLOCKS={"ct":190, "m7":650}


def fetch(url, key=None, byte=None):
    headers={} if byte is None else {"Range":f"bytes={byte}-{byte}","Accept-Encoding":"identity"}
    for attempt in range(4):
        try:
            response=requests.get(url,headers=headers,timeout=(8,32))
            if key and response.status_code==404:
                xml=ElementTree.fromstring(response.content)
                if xml.findtext("Code")=="NoSuchKey" and xml.findtext("Key")==key:
                    return b""
            response.raise_for_status()
            if byte is not None and (response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {byte}-{byte}/{128**3}"):
                raise ValueError("Unrecognized native-byte Range response")
            return response.content
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(1+attempt)


def materialize(folder,name):
    folder.mkdir(parents=True,exist_ok=True)
    for f in ("meta.json","x.tif","y.tif","z.tif"):
        p=folder/f
        if not p.exists():
            p.write_bytes(fetch(f"{ATLAS}/{name}/{f}"))


def zarr_meta(kind):
    out=CACHE/kind/"0"
    out.mkdir(parents=True,exist_ok=True)
    base=CT if kind=="ct" else M7
    body=fetch(BUCKET+base+"/0/.zarray")
    spec=json.loads(body)
    chunk=128 if kind=="ct" else 192
    if spec["dtype"]!="|u1" or spec["chunks"]!=[chunk]*3 or spec["order"]!="C" or spec["fill_value"]!=0 or spec["dimension_separator"]!="/":
        raise ValueError("Unexpected native scalar layout")
    if kind=="ct" and spec["compressor"] is not None:raise ValueError("Unexpected CT compression")
    (out/".zarray").write_bytes(body)
    (out.parent/".zgroup").write_text('{"zarr_format":2}')
    return spec


def keys_in_box(lo,hi,step):
    low=np.floor(lo).astype(int)//step
    high=np.ceil(hi).astype(int)//step
    return set(product(*(range(i,j+1) for i,j in zip(low,high))))


class Blocks:
    """Coordinates of keys are XYZ in memory, ZYX in the official zarr."""
    def __init__(self,kind):
        self.kind=kind
        self.key=CT if kind=="ct" else M7
        self.spec=zarr_meta(kind)
        self.seen={}

    def path(self,xyz):
        return (CACHE/self.kind/"0").joinpath(*(str(q) for q in xyz[::-1]))

    def prefetch(self, keys):
        keys=sorted(keys)
        if len(set(keys)|set(self.seen))>MAX_BLOCKS[self.kind]:
            raise ValueError(f"{self.kind} acquisition would exceed the declared block bound")
        def one(index):
            path=self.path(index)
            empty=path.with_suffix(".fill")
            if not (path.is_file() or empty.is_file()):
                key=self.key+"/0/"+"/".join(map(str,index[::-1]))
                data=fetch(BUCKET+key,key)
                path.parent.mkdir(parents=True,exist_ok=True)
                if not data:empty.touch()
                else:
                    if self.kind=="ct" and len(data)!=128**3:raise ValueError("Short CT chunk")
                    path.write_bytes(data)
            exists=path.is_file()
            data=path.read_bytes() if exists else b""
            return index,[len(data),sha256(data).hexdigest(),not exists]
        with ThreadPoolExecutor(8) as pool:
            self.seen.update(pool.map(one,[key for key in keys if key not in self.seen]))
        if self.kind=="m7" and sum(x[0] for x in self.seen.values())>250*1024**2:
            raise ValueError("M7 byte limit exceeded")

    def raw_box(self, keys):
        ix=np.array(sorted(keys))
        low=ix.min(0); high=ix.max(0)
        shape=tuple((high-low+1)[::-1]*128)
        if np.prod(shape)>250_000_000:raise ValueError("Oversized local scanner box")
        raw=np.zeros(shape,"u1")
        self.prefetch(keys)
        for q in keys:
            path=self.path(q)
            if not path.is_file():continue
            z,y,x=((np.asarray(q)-low)*128)[::-1]
            raw[z:z+128,y:y+128,x:x+128]=np.fromfile(path,"u1").reshape((128,)*3)
        return raw,low*128

    def sample(self, points):
        keys=keys_in_box(np.min(points.reshape(-1,3),0),np.max(points.reshape(-1,3),0),128)
        raw,origin=self.raw_box(keys)
        return sample(raw,origin,points)


def sample(raw,origin,pos):
    coords=(pos-origin)[...,::-1]
    if coords.min()<-.001 or np.any(coords>np.asarray(raw.shape)-1+.001):raise ValueError("Sample outside the loaded volume")
    return map_coordinates(raw,np.moveaxis(coords,-1,0),order=1,output="f4",prefilter=False)


def chart_keys(p,n):
    keys=set()
    for y in range(0,SIDE,32):
        for x in range(0,SIDE,32):
            c=p[y:y+32,x:x+32]; d=n[y:y+32,x:x+32]*32
            keys|=keys_in_box(np.nanmin(np.minimum(c-d,c+d),(0,1)),np.nanmax(np.maximum(c-d,c+d),(0,1)),128)
    return keys


def render(source,p,n):
    if not np.isfinite(p+n).all():raise ValueError("Incomplete geometric chart")
    raw,origin=source.raw_box(chart_keys(p,n))
    vol=np.stack([np.floor(sample(raw,origin,p+t*n)+.5).astype("u1") for t in DEPTH])
    probes=[]
    for row,col,depth in ((48,77,-13),(128,272,7),(257,88,0),(323,342,16)):
        q=np.rint(p[row,col]+depth*n[row,col]).astype(int)
        key_index=q//128; z,y,x=(q%128)[::-1];byte=int(z*128**2+y*128+x)
        key=CT+"/0/"+"/".join(map(str,key_index[::-1]))
        direct=fetch(BUCKET+key,key,byte)
        expected=raw[tuple((q-origin)[::-1])]
        if (direct[0] if direct else 0)!=expected:raise ValueError("Independent native CT byte disagrees")
        probes.append({"row":row,"column":col,"depth":depth,"verified_value":int(expected)})
    return vol,probes


def survey(source,p,sites):
    normals=node_normals(p)
    output=[]
    for y,x in sites:
        normal=normals[y,x]
        along=p[y,x+1]-p[y,x-1]
        along-=normal*(along@normal);along/=np.linalg.norm(along)
        across=np.cross(normal,along)
        offsets=np.array([np.zeros(3),2*along,-2*along,2*across,-2*across])
        rays=p[y,x]+offsets[:,None,:]+OFF[None,:,None]*normal
        output.append(source.sample(rays))
    return np.array(output)


def snap_twice(name,p,stage):
    folder=CACHE/"m7"
    blocks=Blocks("m7")
    mask=np.isfinite(p).all(-1)
    centres=p[mask]
    # Both passes search +/-15 native voxels. A radius of 32 covers either neighbourhood.
    lo=np.floor((centres-32)/192).astype(int);hi=np.floor((centres+32)/192).astype(int)
    keys=set()
    for ab in np.unique(np.c_[lo,hi],axis=0):
        keys.update(product(*(range(i,j+1) for i,j in zip(ab[:3],ab[3:]))))
    blocks.prefetch(keys)
    source=fetch(UPSTREAM)
    if sha256(source).hexdigest()!=UPSTREAM_SHA:raise ValueError("Upstream snap source changed")
    path=stage/"snap_meshes.py";path.write_bytes(source)
    wind=stage/"windings.txt";wind.write_text(name+"\n")
    start=stage/"original"
    passes=[]
    for target in [stage/"one",stage/"two"]:
        env=dict(os.environ,SURF_ZARR=str(folder.resolve()),RANK_PROCS="1",BLOSC_NTHREADS="1",OMP_NUM_THREADS="1")
        proc=subprocess.run([sys.executable,str(path),str(start),"PHerc0125",str(target),"--windings",str(wind),"--procs","1"],env=env,capture_output=True,text=True)
        if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
        passes.append((target/"snap_report.tsv").read_text())
        start=target
    return start/name,passes,blocks


def execute(name,certificate):
    if name not in NAMES:raise ValueError("Unknown public mesh")
    stage=Path("work")/name
    atlas=stage/"original"/name
    materialize(atlas,name)
    p=xyz_from(atlas);origin,g=choose_chart(p);sites=fixed_survey(p)
    raw=Blocks("ct")
    centre,n=chart(p,origin)
    before,checks1=render(raw,centre,n)
    rays=survey(raw,p,sites)
    snapped,passes,m7=snap_twice(name,p,stage)
    s=xyz_from(snapped); c,nn=chart(s,origin)
    after,checks2=render(raw,c,nn)
    rays2=survey(raw,s,sites)
    out=stage/"raw.npz"
    np.savez_compressed(out,raw=before,snapped=after,raw_profiles=rays,snapped_profiles=rays2)
    stats={"source_scan":CT,"m7":M7,"mesh":name,"before":summarize(atlas),"after":summarize(snapped),
           "chart_uv_origin_fixed":origin.tolist(),"survey_index_rc":sites.tolist(),
           "survey_xyz_original":p[tuple(sites.T)].tolist(),"survey_normals_original":node_normals(p)[tuple(sites.T)].tolist(),
           "survey_xyz_after":s[tuple(sites.T)].tolist(),"survey_normals_after":node_normals(s)[tuple(sites.T)].tolist(),
           "profile_offsets":OFF.tolist(),"after_chart_cm2_fixed_uv":surface_area(c)["measured_cm2"],
           "two_upstream_passes":passes,"snap_source_sha256":UPSTREAM_SHA,"byte_checks":[checks1,checks2],
           "inputs":{k:{"/".join(map(str,idx[::-1])):v for idx,v in o.seen.items()} for k,o in [("ct",raw),("m7",m7)]},
           "raw_npz_sha256":sha256(out.read_bytes()).hexdigest()}
    detail=stage/"provenance.json"
    dump(detail,stats)
    bundle=stage/"anatomical-ct.tar"
    with tarfile.open(bundle,"w") as tar:
        tar.add(out,arcname="raw.npz")
        tar.add(detail,arcname="provenance.json")
    package=Path("delivery")/name;package.mkdir(parents=True,exist_ok=True)
    cipher=package/"raw.der"
    subprocess.run(["openssl","cms","-encrypt","-binary","-outform","DER","-aes-256-gcm","-in",str(bundle),"-out",str(cipher),str(certificate)],check=True)
    if cipher.stat().st_size>24*1024**2:raise ValueError("Private attachment is too large")
    dump(package/"receipt.json",{"mesh":name,"scan":CT,"renderer":"native trilinear CT, fixed mesh UV geometric centre",
         "cipher_sha256":sha256(cipher.read_bytes()).hexdigest(),"payload_sha256":sha256(bundle.read_bytes()).hexdigest()})
    print(name, "verified anatomical receipt",len(raw.seen),"CT blocks",len(m7.seen),"M7 blocks",flush=True)


if __name__=="__main__":
    execute(sys.argv[1],Path(sys.argv[2]))
