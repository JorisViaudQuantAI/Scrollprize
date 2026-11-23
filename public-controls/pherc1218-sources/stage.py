"""Archive original S3 objects near every documented public PHerc. 1218 geometric seed.

Inputs consist only of two fixed public datasets. No private target list,
orientation label, intensity score, inference model or local study is used.
"""
import concurrent.futures
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as XML
import numpy as np
import requests
import tifffile

PIN="cc1fb1429d62670f7dc00943932e90d0e6d31c8a"
MESH_REPO="IyanDopico/vesuvius-sheet-tools"
PREFIX="data/spiral_input_pherc1218/verified_patches/"
BUCKET="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
ORIGINS={"ct":("PHerc1218/volumes/20250521120456-8.640um-1.2m-116keV-masked.zarr/",128),
         "m7":("PHerc1218/representations/predictions/surfaces/20250521120456-surface-20260413222639-surface-m7-L0-th0.2.zarr/",192)}
WORK=Path(os.environ["RUNNER_TEMP"])/"public-1218-native"
SHARD=20*1024*1024


def fetch(url):
    for attempt in range(4):
        try:
            r=requests.get(url, timeout=(10,40))
            if r.status_code not in (200,404):r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(attempt+1)


def prepare():
    WORK.mkdir(exist_ok=True)
    url="https://codeload.github.com/"+MESH_REPO+"/tar.gz/"+PIN
    data=fetch(url);data.raise_for_status()
    count=0
    with tarfile.open(fileobj=io.BytesIO(data.content),mode="r:gz") as tar:
        for member in tar:
            match=re.fullmatch(r"[^/]+/"+PREFIX+r"(seed-z[0-9]+-pherc1218)/(x.tif|y.tif|z.tif|meta.json)",member.name)
            if not match:continue
            sid,f=match.groups();directory=WORK/"mesh"/sid;directory.mkdir(parents=True,exist_ok=True)
            (directory/f).write_bytes(tar.extractfile(member).read());count+=1
    assert count==100
    seeds=sorted(f.name for f in (WORK/"mesh").iterdir());assert len(seeds)==25
    (WORK/"names.json").write_text(json.dumps(seeds))


def bbox_set(lo,hi,dim):
    first=np.floor(lo).astype(int)//dim;last=(np.floor(hi).astype(int)+1)//dim
    return [tuple(reversed(v)) for v in itertools.product(*(range(a,b+1) for a,b in zip(first,last)))]


def source_keys(sid):
    g=np.stack([tifffile.imread(WORK/"mesh"/sid/(k+".tif")) for k in "xyz"],axis=-1).astype("f8")
    valid=np.all(np.isfinite(g)&(g>=0),axis=-1)
    cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    keys={"ct":set(),"m7":set()}
    for y,x in zip(*np.where(cells)):
        p=g[y:y+2,x:x+2];center=p.mean(axis=(0,1))
        for which,(prefix,dim) in ORIGINS.items():
            # Universal cube at every public quad center. The CT slab can be reconstructed
            # locally at any pixel of every quad. Store exactly the original public bytes.
            keys[which].update(bbox_set(center-56,center+56,dim))
            if which=="ct":keys[which].update(bbox_set(p.min((0,1))-15,p.max((0,1))+15,dim))
    return [(kind,key) for kind in keys for key in sorted(keys[kind])]


def acquire(job):
    kind,key=job;prefix,dim=ORIGINS[kind];name=prefix+"0/"+"/".join(map(str,key))
    r=fetch(BUCKET+name)
    if r.status_code==404:
        xml=XML.fromstring(r.content)
        if xml.findtext("Key")!=name or xml.findtext("Code")!="NoSuchKey":raise RuntimeError("Unconfirmed absent public key")
        return kind,key,{"key":name,"bytes":0,"public_fill":True,"sha256":None},b""
    r.raise_for_status();raw=r.content
    if kind=="ct" and len(raw)!=128**3:raise ValueError("Unexpected original CT byte count")
    assert raw
    receipt={"key":name,"bytes":len(raw),"public_fill":False,"sha256":hashlib.sha256(raw).hexdigest()}
    etag=r.headers.get("ETag","").strip('"')
    if re.fullmatch(r"[0-9a-f]{32}",etag):
        assert hashlib.md5(raw).hexdigest()==etag
        receipt["s3_single_etag"]=etag
    return kind,key,receipt,raw


def put(tar,name,raw):
    info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0
    tar.addfile(info,io.BytesIO(raw))


def stage(idx,certificate):
    names=json.loads((WORK/"names.json").read_text());sid=names[idx]
    out=WORK/"send";out.mkdir(exist_ok=True)
    for f in out.iterdir():f.unlink()
    with tempfile.TemporaryDirectory(dir=WORK) as temp:
        temp=Path(temp);pakket=temp/"public.tar.gz";cipher=temp/"public.cms"
        receipt={"public_geometry_revision":PIN,"catalog_entry":sid,"data":"Original byte-for-byte public Zarr objects and documented public fills only.","records":[]}
        with tarfile.open(pakket,mode="w:gz",compresslevel=1) as tar, concurrent.futures.ThreadPoolExecutor(6) as pool:
            for kind,key,meta,raw in pool.map(acquire,source_keys(sid)):
                receipt["records"].append(meta)
                if raw:put(tar,kind+"/"+"-".join(map(str,key))+".raw",raw)
            put(tar,"sources.json",json.dumps(receipt).encode())
        subprocess.run(["openssl","cms","-encrypt","-binary","-outform","DER","-aes-256-gcm","-in",str(pakket),"-out",str(cipher),str(certificate)],check=True)
        digest=hashlib.sha256();parts=[]
        with cipher.open("rb") as f:
            while buf:=f.read(SHARD):
                digest.update(buf);fn="part-"+str(len(parts)).zfill(2)+".bin";(out/fn).write_bytes(buf)
                parts.append({"file":fn,"size":len(buf),"sha256":hashlib.sha256(buf).hexdigest()})
        outer={"public_id":sid,"source_revision":PIN,"parts":parts,"cms_sha256":digest.hexdigest()}
        (out/"receipt.json").write_text(json.dumps(outer,indent=2))
    print("Public catalogue entry",idx+1,"of",len(names),"ready in",len(parts),"encrypted parts.",flush=True)


if __name__=="__main__":
    if sys.argv[1]=="prepare":prepare()
    else:stage(int(sys.argv[1]),Path(sys.argv[2]))
