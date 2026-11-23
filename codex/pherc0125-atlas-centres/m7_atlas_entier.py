"""Retrieve the compressed public M7 blocks covering every vertex of the published winding."""
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import subprocess
import sys
import tarfile

import numpy as np
from atlas125 import xyz_from, dump
from stage125 import Blocks, CACHE, M7, materialize

MESH = "z14144_w045"
GROUPS = 7


def main(certificate):
    atlas=Path("public-mesh")/MESH
    materialize(atlas,MESH)
    coords=xyz_from(atlas);centres=coords[np.isfinite(coords).all(-1)]
    # The source atlas defined this whole-winding neighbourhood in the earlier calculation.
    lo=np.floor((centres-32)/192).astype(int)
    hi=np.floor((centres+32)/192).astype(int)
    keys=set()
    for bounds in np.unique(np.c_[lo,hi],axis=0):
        keys.update(product(*(range(a,b+1) for a,b in zip(bounds[:3],bounds[3:]))))
    if len(keys)!=344:raise ValueError("Published whole-winding neighbourhood changed")
    store=Blocks("m7");store.prefetch(keys)
    n_bytes=sum(x[0] for x in store.seen.values())
    if n_bytes!=129441323 or n_bytes>250000000:raise ValueError("M7 release size changed")
    groups=[[] for _ in range(GROUPS)];sizes=[0]*GROUPS
    for key,info in sorted(store.seen.items(),key=lambda p:(-p[1][0],p[0])):
        group=int(np.argmin(sizes));groups[group].append(key);sizes[group]+=info[0]
    for i,group in enumerate(groups):
        target=Path("delivery")/f"group-{i:02d}";target.mkdir(parents=True,exist_ok=True)
        part=Path(f"whole-winding-{i:02d}.tar");manifest=Path(f"manifest-{i:02d}.json")
        keys_json={"/".join(str(q) for q in key[::-1]):store.seen[key] for key in sorted(group)}
        dump(manifest,dict(mesh=MESH,m7=M7,group=i,groups=GROUPS,n_bytes_remote=n_bytes,keys=keys_json,
                           zarray=json.loads((CACHE/"m7/0/.zarray").read_text())))
        with tarfile.open(part,"w") as archive:
            archive.add(manifest,arcname="manifest.json")
            for coord in group:
                source=store.path(coord)
                if source.is_file():archive.add(source,arcname="blocks/"+"/".join(str(q) for q in coord[::-1]))
        encrypted=target/"geometry.der"
        subprocess.run(["openssl","cms","-encrypt","-binary","-outform","DER","-aes-256-gcm",
                        "-in",str(part),"-out",str(encrypted),str(certificate)],check=True)
        if encrypted.stat().st_size>24*1024**2:raise ValueError("Archive exceeds transfer budget")
        dump(target/"receipt.json",dict(public_geometry=MESH,part=i,parts=GROUPS,
             tar_sha256=sha256(part.read_bytes()).hexdigest(),encrypted_sha256=sha256(encrypted.read_bytes()).hexdigest()))
    print("Public whole-winding M7 blocks",len(keys),"bytes",n_bytes,"encrypted archive parts",GROUPS)


if __name__=="__main__":main(Path(sys.argv[1]))
