#!/usr/bin/env python3
"""Acquisition-only retry; original 050 numerical code is imported unchanged."""
import argparse,hashlib,importlib.util,json,pathlib,sys,urllib.parse
import requests
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("winner050",HERE/"run.py")
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)

def acquire_layout(cache):
    cache.mkdir(parents=True,exist_ok=False)
    allfiles=[];dirs=[];receipts=[];bydir={};queue=[(old.PREFIX,0)]
    with requests.Session() as s:
        while queue and len(receipts)<7:
            prefix,depth=queue.pop(0)
            files,children,r=old.listing(s,prefix)
            bydir[prefix]=files;allfiles.extend(files);dirs.extend(children);receipts.append(r)
            if depth<3:
                # Only geometry-related paths, including children of a named geometry directory.
                for d in children:
                    name=d.rstrip("/").split("/")[-1].lower()
                    if any(k in name for k in ("mesh","tifxyz","geometry","surface")) and "volume" not in name:
                        queue.append((d,depth+1))
        inventory={"listing":receipts,"public_file_inventory":allfiles,"public_directory_inventory":dirs}
        print("LAYOUT_JSON="+json.dumps(inventory,separators=(",",":")),flush=True)
        candidates=[]
        for prefix,files in bydir.items():
            lookup={x["key"].split("/")[-1]:x for x in files}
            axes={}
            for n in "xyz":
                opts=[lookup[k] for k in (n+".tif",n+".tiff") if k in lookup]
                if len(opts)==1:axes[n+".tif"]=opts[0]
            if len(axes)!=3:continue
            meta=lookup.get("meta.json")
            if meta is None:
                parent=prefix.rstrip("/").rsplit("/",1)[0]+"/"
                mm=[x for x in bydir.get(parent,[]) if x["key"].split("/")[-1]=="meta.json"]
                if len(mm)==1:meta=mm[0]
            if meta is not None:candidates.append({**axes,"meta.json":meta})
        if len(candidates)!=1:raise ValueError(f"Found {len(candidates)} complete unique mesh layouts")
        mesh=candidates[0]
        if sum(x["bytes"] for x in mesh.values())>old.CAP:raise ValueError("Mesh exceeds cap")
        objs={}
        for n,r in mesh.items():
            b=old.fetch(s,old.BUCKET+"/"+urllib.parse.quote(r["key"],safe="/"),old.CAP,expected=r)
            (cache/n).write_bytes(b);objs[n]={**r,"sha256":hashlib.sha256(b).hexdigest()}
    return {**inventory,"objects":objs,"mesh_bytes":sum(x["bytes"] for x in objs.values())}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",type=pathlib.Path,required=True);p.add_argument("--output",type=pathlib.Path,required=True);a=p.parse_args()
    old.acquire=acquire_layout
    old.run(a)
