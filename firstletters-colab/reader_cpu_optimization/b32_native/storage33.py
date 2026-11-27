"""B33: exact random-access storage and CPU query timing."""
import argparse,hashlib,json,os,time,zlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
from panels import load,sha
SHAPE=(15,21,384,384)
def digest(b):return hashlib.sha256(b).hexdigest()
def build(raw,path):
    index={};t=time.perf_counter()
    with path.open('wb') as f:
        for i in range(15):
            for y in range(6):
                for x in range(6):
                    v=raw[i,:,y*64:(y+1)*64,x*64:(x+1)*64].tobytes();z=zlib.compress(v,1);index[f'{i},{y},{x}']=[f.tell(),len(z),digest(v)];f.write(z)
    path.with_suffix('.json').write_text(json.dumps(dict(shape=SHAPE,block=64,sha256=sha(path),index=index),separators=(',',':'))+'\n');return time.perf_counter()-t
class Reader:
    def __init__(self,path,capacity=16):
        j=json.loads(path.with_suffix('.json').read_text());assert j['shape']==list(SHAPE) and j['block']==64
        self.index=j['index'];self.capacity=capacity;self.cache=OrderedDict();self.hit=self.miss=self.inflated=0;end=0
        assert len(self.index)==540
        for i in range(15):
            for y in range(6):
                for x in range(6):
                    o,n,h=self.index[f'{i},{y},{x}'];assert o==end and n>0 and len(h)==64;end=o+n
        assert end==path.stat().st_size;self.fd=os.open(path,os.O_RDONLY)
    def close(self):os.close(self.fd)
    def block(self,i,y,x):
        k=f'{i},{y},{x}'
        if k in self.cache:self.hit+=1;self.cache.move_to_end(k);return self.cache[k]
        self.miss+=1;o,n,h=self.index[k];z=os.pread(self.fd,n,o);assert len(z)==n
        raw=zlib.decompress(z);assert len(raw)==21*64*64 and digest(raw)==h
        self.inflated+=len(raw);a=np.frombuffer(raw,np.uint8).reshape(21,64,64);self.cache[k]=a
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return a
    def read(self,i,y,x,n):
        if not (0<=i<15 and 0<n<=384 and 0<=x<=384-n and 0<=y<=384-n):raise ValueError('out of bounds')
        out=np.empty((21,n,n),np.uint8)
        for by in range(y//64,(y+n-1)//64+1):
            for bx in range(x//64,(x+n-1)//64+1):
                ya,yb=max(y,by*64),min(y+n,(by+1)*64);xa,xb=max(x,bx*64),min(x+n,(bx+1)*64)
                out[:,ya-y:yb-y,xa-x:xb-x]=self.block(i,by,bx)[:,ya-by*64:yb-by*64,xa-bx*64:xb-bx*64]
        return out
def run(d,out):
    out.mkdir(exist_ok=True);a,b,_,_=load(d);raw=np.concatenate([a['raw'],b['raw']]);assert raw.shape==SHAPE
    build_seconds={};t=time.perf_counter();np.savez_compressed(out/'all.npz',raw=raw);build_seconds['npz']=time.perf_counter()-t;t=time.perf_counter();np.save(out/'all.npy',raw);build_seconds['npy']=time.perf_counter()-t;build_seconds['blocks']=build(raw,out/'blocks.bin')
    rng=np.random.default_rng(3301);queries=[]
    for j in range(48):
        i=int(rng.integers(15));n=384 if j%4==0 else 128;y,x=map(int,rng.integers(385-n,size=2));queries.append((i,y,x,n))
    rows=[];counters=[]
    for rep in range(2):
        for method in np.roll(['npz','npy','blocks','resident'],rep):
            r=Reader(out/'blocks.bin') if method=='blocks' else None;m=np.load(out/'all.npy',mmap_mode='r') if method=='npy' else None
            for j,(i,y,x,n) in enumerate(queries):
                t=time.perf_counter()
                if method=='npz':
                    with np.load(out/'all.npz',allow_pickle=False) as z:v=z['raw'][i,:,y:y+n,x:x+n].copy()
                elif method=='npy':v=np.array(m[i,:,y:y+n,x:x+n],copy=True)
                elif method=='blocks':v=r.read(i,y,x,n)
                else:v=raw[i,:,y:y+n,x:x+n].copy()
                dt=time.perf_counter()-t;assert np.array_equal(v,raw[i,:,y:y+n,x:x+n]);rows.append([rep,str(method),j,n,dt])
            if r:counters.append(dict(rep=rep,hits=r.hit,misses=r.miss,inflated=r.inflated));r.close()
    p=out/'blocks.bin';r=Reader(p);assert np.array_equal(r.read(0,3,7,128),raw[0,:,3:131,7:135]);r.close();tests=1
    r=Reader(p)
    try:r.read(15,0,0,1);raise RuntimeError('bounds accepted')
    except ValueError:tests+=1
    r.close()
    with p.open('r+b') as f:old=f.read(1);f.seek(0);f.write(bytes([old[0]^1]))
    try:
        r=Reader(p)
        try:r.block(0,0,0);raise RuntimeError('corruption accepted')
        except (zlib.error,AssertionError):tests+=1
        finally:r.close()
    finally:
        with p.open('r+b') as f:f.write(old)
    assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];tests+=1
    sz=p.stat().st_size
    with p.open('r+b') as f:f.seek(sz-1);tail=f.read(1);f.truncate(sz-1)
    try:
        try:Reader(p);raise RuntimeError('truncation accepted')
        except AssertionError:tests+=1
    finally:
        with p.open('ab') as f:f.write(tail)
    summary={}
    for name in ('npz','npy','blocks','resident'):
        v=[r for r in rows if r[1]==name];summary[name]={'median_ms':float(np.median([r[4] for r in v])*1e3),'total_seconds':sum(r[4] for r in v),'small_median_ms':float(np.median([r[4] for r in v if r[3]==128])*1e3),'full_median_ms':float(np.median([r[4] for r in v if r[3]==384])*1e3)}
    return dict(schema='b33.storage.v1',bytes_raw=raw.nbytes,storage_bytes={'npz':(out/'all.npz').stat().st_size,'npy':(out/'all.npy').stat().st_size,'blocks':p.stat().st_size+p.with_suffix('.json').stat().st_size},build_seconds=build_seconds,summary=summary,queries=queries,rows=rows,checks=len(rows)+tests,integrity_tests=tests,chunk_counters=counters,npz_logical_inflate=2*len(queries)*raw.nbytes,scope='Warm filesystem, fresh application cache per repeat; rendered panels only, not native CT transport.')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data',type=Path);a.add_argument('out',type=Path);v=a.parse_args();r=run(v.data,v.out);(v.out/'result.json').write_text(json.dumps(r,separators=(',',':'))+'\n');print(json.dumps({k:x for k,x in r.items() if k not in ('rows','queries')},indent=2))
