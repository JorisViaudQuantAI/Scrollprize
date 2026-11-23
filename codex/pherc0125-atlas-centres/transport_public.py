"""Encode every regular native CT band of a complete named public anatomical atlas."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from atlas125 import STEP, xyz_from, surface_area
from full_atlas125 import grid_rect, tiles
from stage125 import BUCKET, CT, CACHE, Blocks, MAX_BLOCKS, fetch, materialize, sample

BAND_WIDTH=400
DEPTHS=tuple(range(-13,14))


def encrypt(name,source):
    folder=Path('delivery')/name
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'native.der'
    subprocess.run(['openssl','cms','-encrypt','-binary','-outform','DER','-aes-256-gcm','-in',str(source),'-out',str(target),'recipient.pem'],check=True)
    if target.stat().st_size>24*1024**2:
        raise ValueError('A transport piece exceeds its cap')
    info={'package':name,'sha256_input':sha256(source.read_bytes()).hexdigest(),'sha256_encrypted':sha256(target.read_bytes()).hexdigest()}
    (folder/'receipt.json').write_text(json.dumps(info))
    return info


def checks(raw,pos,mask,left,band_id):
    indices=np.argwhere(mask);selected=[]
    for i in (1,2):
        r,c=indices[(len(indices)*i)//3];delta=DEPTHS[(band_id*5+i*11)%len(DEPTHS)]
        p,n=grid_rect(pos,int(r),int(r+1),int(left+c),int(left+c+1));w=(p+delta*n)[0,0]
        low=np.floor(w).astype(int);f=w-low;sampled=[]
        for dz,dy,dx in product((0,1),repeat=3):
            xyz=low+[dx,dy,dz];k=xyz//128;z,y,x=(xyz%128)[::-1];off=int(z*128**2+y*128+x)
            key=CT+'/0/'+'/'.join(map(str,k[::-1]))
            sampled.append((key,off))
        def item(index):
            k,off=index;body=fetch(BUCKET+k,k,off)
            return body[0] if body else 0
        with ThreadPoolExecutor(4) as workers:values=list(workers.map(item,sampled))
        att=0.
        for d,val in zip(product((0,1),repeat=3),values):
            att+=np.prod(np.where(d[::-1],f,1-f))*val
        obs=int(raw[delta+13,r,c])
        if int(np.floor(att+.5))!=obs:raise ValueError('Native byte interpolation mismatch')
        selected.append({'public_band':band_id,'uv':[int(r),int(left+c)],'depth':delta,'source_xyz':w.tolist(),'corners_zyx':values,'rounded_from_range':obs})
    return selected


def main(name):
    if name not in ('z14144_w045','z11136_w035'):
        raise ValueError('Use a member of the previously published two-atlas geometric survey')
    folder=Path('work')/'public-atlas';materialize(folder,name)
    mesh=xyz_from(folder);h,w=(np.array(mesh.shape[:2])-1)*STEP
    if h*w*27>=256*1024**2:raise ValueError('Atlas output too large')
    results={'mesh':name,'level0':CT,'original':True,'pitch_um':9.362,'source_area':surface_area(mesh),'shape':[27,int(h),int(w)],
             'band_width':BAND_WIDTH,'bands':[],'source_chunks':{},'independent_byte_checks':[]}
    for band_id,left in enumerate(range(0,w,BAND_WIDTH)):
        right=min(int(w),left+BAND_WIDTH);jobs=tiles(mesh,h,left,right)
        needed=set().union(*(j[-1] for j in jobs)) if jobs else set()
        if len(needed)*128**3>400*1024**2:raise ValueError('Raw cache exceeds the band budget')
        MAX_BLOCKS['ct']=len(needed)
        scanner=Blocks('ct');scanner.prefetch(needed)
        raw=np.zeros((27,h,right-left),'u1');mask=np.zeros((h,right-left),bool)
        for _,y,end,x,stop,keys in jobs:
            p,n=grid_rect(mesh,y,end,x,stop);ok=np.isfinite(p+n).all(-1);rows,cols=np.nonzero(ok)
            a,norm=p[ok],n[ok];box,origin=scanner.raw_box(keys)
            for z,d in enumerate(DEPTHS):
                raw[z,y+rows,x+cols-left]=np.floor(sample(box,origin,a+d*norm)+.5).astype('u1')
            mask[y+rows,x+cols-left]=True
        if mask.any():results['independent_byte_checks']+=checks(raw,mesh,mask,left,band_id)
        digest={'/'.join(map(str,key[::-1])):value for key,value in scanner.seen.items()}
        for key,val in digest.items():
            if key in results['source_chunks'] and results['source_chunks'][key]!=val:
                raise ValueError('Official source changed across regular bands')
            results['source_chunks'][key]=val
        out=Path('work')/'band.npz'
        np.savez_compressed(out,material=raw,mask=mask,uv_extent=np.array([0,h,left,right]))
        code=f'band-{band_id:02d}'
        info=encrypt(code,out)
        results['bands'].append({'id':code,'measured':int(mask.sum()),'uv_columns':[left,right],'source_chunks':sorted(digest),'payload_hash':info['sha256_input']})
        out.unlink();shutil.rmtree(CACHE,ignore_errors=True)
        print('Completed regular public band',band_id,flush=True)
    source=Path('work')/'manifest.json';source.write_text(json.dumps(results))
    encrypt('verification',source);source.unlink()


if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception as error:
        print('Public input job failed with',type(error).__name__,flush=True);sys.exit(1)
