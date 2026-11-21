"""Return uniformly sampled native public anatomy along a fixed grid of relative heights."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5,sha256
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
import argparse,json,subprocess,time
import numpy as np
from numcodecs import get_codec
import requests

BASE='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1545/'
AXIS='https://raw.githubusercontent.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici/de7642ac9b38266e95192489e22fb13b7e2d13b6/PHerc1545_umbilicus.json'
CT='volumes/20250821151648-9.362um-1.2m-113keV-masked.zarr/0/'
M7='representations/predictions/surfaces/20250821151648-surface-20260413222639-surface-m7-L0-th0.2.zarr/0/'

def get(url):
    for i in range(4):
        try:
            r=requests.get(url,timeout=(6,45))
            if r.status_code==404:
                m=ET.fromstring(r.content)
                if m.findtext('Code')=='NoSuchKey' and url.endswith(m.findtext('Key') or 'INVALID'):return None,None
            r.raise_for_status();data=r.content;tag=r.headers.get('ETag','').strip('"')
            if len(tag)==32 and md5(data).hexdigest()!=tag:raise ValueError('Object digest mismatch')
            return data,tag
        except (ValueError,requests.RequestException):
            if i==3:raise
            time.sleep(1+i)

def save(folder,kind,slot,part,volume,lo,records,cert):
    out=folder/f'{kind}-{part:02d}';out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as temp:
        raw=Path(temp)/'subset.npz';np.savez_compressed(raw,data=volume,origin_zyx=lo)
        dst=out/'native.der';subprocess.run(['openssl','cms','-encrypt','-binary','-aes-256-gcm','-outform','DER','-in',str(raw),'-out',str(dst),cert],check=True)
        d={'public_grid_slot':slot,'kind':kind,'part':part,'shape':list(volume.shape),'source_records':records,'plain_sha256':sha256(raw.read_bytes()).hexdigest(),'data_c_order_sha256':sha256(np.ascontiguousarray(volume)).hexdigest(),'encrypted_sha256':sha256(dst.read_bytes()).hexdigest()};(out/'receipt.json').write_text(json.dumps(d,indent=2)+'\n')

def main(slot,cert):
    f=[.60,.65,.70][slot];z=round(f*(20961-1));axis=sorted(json.loads(get(AXIS)[0])['control_points'],key=lambda p:p['z']);y=round(np.interp(z,[p['z'] for p in axis],[p['y'] for p in axis]));x=round(np.interp(z,[p['z'] for p in axis],[p['x'] for p in axis]))+1024
    for kind,base,rad in [('m7',M7,192),('ct',CT,208)]:
        layout=json.loads(get(BASE+base+'.zarray')[0]);d=layout['chunks'][0];assert layout['chunks']==[d]*3 and layout['dtype']=='|u1';codec=get_codec(layout['compressor']) if layout['compressor'] else None
        lower=np.array([z,y,x])-rad;high=np.array([z,y,x])+rad+1;keys=list(product(*(range(int(a)//d,(int(b)-1)//d+1) for a,b in zip(lower,high))));vol=np.zeros((2*rad+1,)*3,np.uint8)
        def one(key):
            path='/'.join(str(t) for t in key);raw,etag=get(BASE+base+path)
            source={'key_zyx':list(key),'fill':raw is None,'object_sha256':sha256(raw).hexdigest() if raw is not None else None,'etag':etag};start=np.array(key)*d;l=np.maximum(lower,start);h=np.minimum(high,start+d)
            if raw is not None:
                a=np.frombuffer(codec.decode(raw) if codec else raw,'u1').reshape((d,)*3);vol[tuple(slice(p,q) for p,q in zip(l-lower,h-lower))]=a[tuple(slice(p,q) for p,q in zip(l-start,h-start))]
            return source
        with ThreadPoolExecutor(7) as ex:records=list(ex.map(one,keys))
        length=len(vol) if kind=='m7' else 80
        for i,start in enumerate(range(0,len(vol),length)):
            out=vol[start:start+length];save(Path('delivery'),kind,slot,i,out,lower+np.array([start,0,0]),records,cert)
        print('Public grid element',slot,kind,'complete',len(keys),'source objects',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('slot',type=int,choices=range(3));parser.add_argument('certificate');args=parser.parse_args();main(args.slot,args.certificate)
