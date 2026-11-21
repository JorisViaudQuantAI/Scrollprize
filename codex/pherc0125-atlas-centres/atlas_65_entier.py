"""Render the same complete public PHerc. 125 atlas over 65 scanner-native depth positions."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import numpy as np
from atlas125 import xyz_from, dump
from full_atlas125 import grid_rect
from stage125 import Blocks, CT, M7, MAX_BLOCKS, UPSTREAM_SHA, fetch, keys_in_box, materialize, sample, snap_twice

MESH="z14144_w045"
HALF=32
TILE_H,TILE_W=384,512


def tasks(mesh):
    h,w=(np.array(mesh.shape[:2])-1)*20
    result={}
    for row in range(0,h,TILE_H):
        for col in range(0,w,TILE_W):
            subs=[]
            for y in range(row,min(h,row+TILE_H),32):
                for x in range(col,min(w,col+TILE_W),64):
                    yi=min(y+32,h,row+TILE_H);xi=min(x+64,w,col+TILE_W)
                    p,n=grid_rect(mesh,y,yi,x,xi);valid=np.isfinite(p+n).all(-1)
                    if not valid.any():continue
                    a,b=p[valid],HALF*n[valid]
                    ids=keys_in_box(np.minimum(a-b,a+b).min(0),np.maximum(a-b,a+b).max(0),128)
                    subs.append((y,yi,x,xi,ids))
            result[(row,col)]=subs
    return result,int(h),int(w)


def main(cert):
    stage=Path("work")/MESH;atlas=stage/"original"/MESH
    materialize(atlas,MESH);initial=xyz_from(atlas)
    snapped,_,m7=snap_twice(MESH,initial,stage);mesh=xyz_from(snapped)
    measure,h,w=tasks(mesh);indices=set().union(*(q[-1] for subs in measure.values() for q in subs))
    m7bytes=sum(q[0] for q in m7.seen.values());native=len(indices)*128**3
    if native+m7bytes>2*1024**3 or m7bytes!=129441323:raise ValueError("Public native geometry source budget exceeded")
    print("Full public geometry extent",h,w,"native blocks",len(indices),"CT + M7 bytes upper bound",native+m7bytes,flush=True)
    MAX_BLOCKS["ct"]=len(indices);ct=Blocks("ct");ct.prefetch(indices)
    meta=dict(source=CT,m7=M7,mesh=MESH,native_voxel_um=9.362,snap_script_sha256=UPSTREAM_SHA,
              all_tile_rows_cols=[(h+TILE_H-1)//TILE_H,(w+TILE_W-1)//TILE_W],tile_shape=[TILE_H,TILE_W],
              chart_shape=[65,h,w],mesh_xyz_f32_sha256=sha256(np.asarray(mesh,dtype="f4").tobytes()).hexdigest(),
              remote_ct_bytes=sum(v[0] for v in ct.seen.values()),remote_m7_bytes=m7bytes,n_source_ct_blocks=len(ct.seen))
    for (row,col),subs in measure.items():
        hh,ww=min(h-row,TILE_H),min(w-col,TILE_W);vol=np.zeros((65,hh,ww),"u1");keep=np.zeros((hh,ww),bool)
        for y,yi,x,xi,ids in subs:
            p,n=grid_rect(mesh,y,yi,x,xi);valid=np.isfinite(p+n).all(-1)
            a,b=p[valid],n[valid];raw,origin=ct.raw_box(ids);rr,cc=np.nonzero(valid);rr+=y-row;cc+=x-col;keep[rr,cc]=True
            for j,d in enumerate(range(-HALF,HALF+1)):
                vol[j,rr,cc]=np.floor(sample(raw,origin,a+d*b)+.5).astype("u1")
        r,c=row//TILE_H,col//TILE_W;code=f"r{r:02d}c{c:02d}";root=Path("delivery")/code;root.mkdir(parents=True,exist_ok=True)
        check=None
        if keep.any():
            loc=np.argwhere(keep);v,u=loc[len(loc)//2];d=((r*11+c*7)%65)-32
            p,n=grid_rect(mesh,row+v,row+v+1,col+u,col+u+1);point=np.rint(p[0,0]+d*n[0,0]).astype(int)
            key=tuple(point//128);iz,iy,ix=(point%128)[::-1];off=int(iz*128**2+iy*128+ix)
            remote=CT+"/0/"+"/".join(str(q) for q in key[::-1]);data=fetch("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"+remote,remote,off)
            stored=ct.path(key);wanted=stored.read_bytes()[off] if stored.exists() else 0
            if (data[0] if data else 0)!=wanted:raise ValueError("Independent source byte disagreement")
            check={"tile_row_col":[int(v),int(u)],"depth":d,"source_xyz":point.tolist(),"native_byte":int(wanted)}
        plain=stage/f"{code}.npz";np.savez_compressed(plain,volume=vol,mask=keep)
        cipher=root/"raw.der";subprocess.run(["openssl","cms","-encrypt","-binary","-outform","DER","-aes-256-gcm","-in",str(plain),"-out",str(cipher),str(cert)],check=True)
        if cipher.stat().st_size>=22*1024**2:raise ValueError("A uniform archive exceeds the transfer bound")
        dump(root/"receipt.json",dict(public_atlas=meta,tile=code,uv0=[row,col],shape=list(vol.shape),n_valid=int(keep.sum()),
             ct_source_check=check,sha256=sha256(plain.read_bytes()).hexdigest(),encrypted_sha256=sha256(cipher.read_bytes()).hexdigest()))
        plain.unlink()
    print("Whole-winding raw CT tiles exported",len(measure),flush=True)


if __name__=="__main__":main(Path(sys.argv[1]))
