"""Bounded native CT of a whole previously specified public PHerc. 125 anatomical atlas."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from atlas125 import STEP, dump, surface_area, xyz_from
from stage125 import Blocks, CACHE, CT, M7, MAX_BLOCKS, UPSTREAM_SHA, fetch, keys_in_box, materialize, sample, snap_twice

MESH="z14144_w045"
DEPTHS=np.arange(-13,14)
BAND=100
LIMIT=2*1024**3


def grid_rect(p,y0,y1,x0,x1):
    yy,xx=np.mgrid[y0:y1,x0:x1]
    yf=yy/STEP;xf=xx/STEP
    y=(yf-np.floor(yf))[...,None];x=(xf-np.floor(xf))[...,None]
    row=yy//STEP;col=xx//STEP
    a,b,c,d=p[row,col],p[row,col+1],p[row+1,col],p[row+1,col+1]
    pos=(1-y)*((1-x)*a+x*b)+y*((1-x)*c+x*d)
    du=(1-y)*(b-a)+y*(d-c);dv=(1-x)*(c-a)+x*(d-b)
    n=np.cross(du,dv);ln=np.linalg.norm(n,axis=-1,keepdims=True)
    return pos,n/np.where(ln>1e-7,ln,np.nan)


def tiles(mesh,h,left,right):
    out=[]
    for band in range(0,h,BAND):
        stop=min(h,band+BAND)
        for y in range(band,stop,25):
            end=min(stop,y+25)
            for x in range(left,right,100):
                final=min(right,x+100)
                p,n=grid_rect(mesh,y,end,x,final);valid=np.isfinite(p+n).all(-1)
                if not valid.any():continue
                a,b=p[valid],13*n[valid]
                indices=keys_in_box(np.minimum(a-b,a+b).min(0),np.maximum(a-b,a+b).max(0),128)
                out.append((band,y,end,x,final,indices))
    return out


def main(certificate):
    tick=time.monotonic()
    work=Path("work")/MESH;atlas=work/"original"/MESH
    materialize(atlas,MESH)
    source=xyz_from(atlas)
    snapped,reports,m7=snap_twice(MESH,source,work)
    mesh=xyz_from(snapped)
    h,w=(np.array(mesh.shape[:2])-1)*STEP
    m7bytes=sum(t[0] for t in m7.seen.values())
    for fraction in [1,.95,.9,.85,.8,.75,.7,.65,.6,.55,.5]:
        trim=round((1-fraction)*w/(2*STEP))*STEP
        left,right=trim,w-trim
        tasks=tiles(mesh,h,left,right)
        allkeys=set().union(*(job[-1] for job in tasks))
        projected=len(allkeys)*128**3+m7bytes+int(28*h*(right-left))
        if projected<LIMIT:break
    else:raise ValueError("Even the predefined central half is over the permitted budget")
    print("Public geometry preflight",len(allkeys),"CT blocks;",m7bytes,"M7 bytes;",projected,"total conservative bytes",flush=True)
    MAX_BLOCKS["ct"]=len(allkeys)
    ct=Blocks("ct")
    ct.prefetch(allkeys)
    count=(h+BAND-1)//BAND;wout=right-left
    globalmeta=dict(mesh=MESH,scan=CT,geometry_source=M7,snap_source_sha256=UPSTREAM_SHA,snap_reports=reports,
         triangle_area=surface_area(mesh),grid_shape_rc=list(mesh.shape[:2]),shape_zyx=[len(DEPTHS),int(h),int(wout)],
         half_open_full_uv=[0,int(h),int(left),int(right)],band_count=int(count),projected_bytes=int(projected),
         input_m7_bytes=m7bytes,input_ct_bytes=sum(t[0] for t in ct.seen.values()),n_ct_blocks=len(ct.seen),n_m7_blocks=len(m7.seen),
         ct_checksums={"/".join(map(str,idx[::-1])):v for idx,v in ct.seen.items()})
    for band in range(0,h,BAND):
        end=min(h,band+BAND);out=np.zeros((len(DEPTHS),end-band,wout),"u1");mask=np.zeros((end-band,wout),bool)
        for _,y,ye,x,xe,indices in [job for job in tasks if job[0]==band]:
            p,n=grid_rect(mesh,y,ye,x,xe);valid=np.isfinite(p+n).all(-1)
            a,b=p[valid],n[valid]
            raw,origin=ct.raw_box(indices)
            rows,cols=np.nonzero(valid);rr=rows+y-band;cc=cols+x-left
            mask[rr,cc]=True
            for di,depth in enumerate(DEPTHS):
                out[di,rr,cc]=np.floor(sample(raw,origin,a+depth*b)+.5).astype("u1")
        i=band//BAND;folder=Path("delivery")/f"strip-{i:02d}";folder.mkdir(parents=True,exist_ok=True)
        check=None
        if mask.any():
            coords=np.argwhere(mask);r,c=coords[len(coords)//2];y0,x0=int(band+r),int(left+c)
            p,n=grid_rect(mesh,y0,y0+1,x0,x0+1);depth=int(DEPTHS[(i*7)%len(DEPTHS)])
            xyz=np.rint(p[0,0]+depth*n[0,0]).astype(int);key=tuple(xyz//128);z,j,k=(xyz%128)[::-1];by=int(z*128**2+j*128+k)
            remote=CT+"/0/"+"/".join(map(str,key[::-1]));b=fetch("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"+remote,remote,by)
            native=ct.path(key);expect=native.read_bytes()[by] if native.is_file() else 0
            if (b[0] if b else 0)!=expect:raise ValueError("Native source byte mismatch")
            check=[y0,x0,depth,int(expect)]
        extra={"mesh_xyz":mesh.astype("f4")} if i==0 else {}
        path=work/f"strip-{i:02d}.npz";np.savez_compressed(path,volume=out,mask=mask,**extra)
        cipher=folder/"raw.der"
        subprocess.run(["openssl","cms","-encrypt","-binary","-outform","DER","-aes-256-gcm","-in",str(path),"-out",str(cipher),str(certificate)],check=True)
        if cipher.stat().st_size>24*1024**2:raise ValueError("Strip exceeds the recipient transfer limit")
        receipt={"mesh":MESH,"index":i,"bands":int(count),"uv_row":int(band),"uv_column":int(left),"shape":list(out.shape),
              "n_valid":int(mask.sum()),"independent_native_byte_check":check is not None,
              "sha256":sha256(path.read_bytes()).hexdigest(),"cipher_sha256":sha256(cipher.read_bytes()).hexdigest()}
        if i==0:receipt["geometry"]=globalmeta
        dump(folder/"receipt.json",receipt)
        print("Public atlas native CT strip",i,int(mask.sum()),round(time.monotonic()-tick,1),"seconds",flush=True)


if __name__=="__main__":main(Path(sys.argv[1]))
