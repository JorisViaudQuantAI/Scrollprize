"""B26: stage public native arrays, independently of network inference."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'b22_cross_parent'),str(HERE.parent/'b18_paired')]
from reader_1447 import acquire
from native_841 import prepare
from types import SimpleNamespace

def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['841','1447']);p.add_argument('cache',type=Path);p.add_argument('author',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    t=time.perf_counter()
    if a.mode=='841':
        panel=prepare(SimpleNamespace(cache=a.cache,author=a.author,out=a.out))
        raw=np.stack([v[1] for v in panel]);states=np.stack([v[2] for v in panel]);orig=np.array([[v[0]['y'],v[0]['x']] for v in panel]);names=np.array(['0841']*len(panel))
        cold=time.perf_counter()-t
        # Durable fixture before the separate inference process. Pin previous B22 bytes.
        expected=['5c66dce93edb94c25cef84c65ad7dab53f2e3ece3dc5b4c41fb954c7402c5a1f','65abdf34b924fbd17e7323e7336a6e691bf672a52a479de63c2b7529cdb885c9','55518a861c74c905c82a0b04ad49705837ed672030cea5200354f70ce4398191']
        assert [digest(x) for x in raw]==expected
        np.savez_compressed(a.out/'panel.npz',raw=raw,states=states,origins_yx=orig,names=names)
        print('B26_STAGED',json.dumps(dict(mode=a.mode,cold_seconds=cold,raw_sha256=expected)),flush=True)
        w=a.out/'warm';w.mkdir();t2=time.perf_counter()
        again=prepare(SimpleNamespace(cache=a.cache,author=a.author,out=w))
        assert all(np.array_equal(x[1],y[1]) and np.array_equal(x[2],y[2]) for x,y in zip(panel,again))
        elapsed=time.perf_counter()-t2;rc=json.loads((a.out/'prepare.json').read_text());rw=json.loads((w/'prepare.json').read_text())
        assert rw['native']['new_http_bytes']==0
        out=dict(cold_seconds=cold,warm_seconds=elapsed,cold_native_bytes=rc['native']['new_http_bytes'],warm_native_bytes=rw['native']['new_http_bytes'],exact_values=int(raw.size),raw_sha256=expected)
        (a.out/'cache_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print('B26_CACHE',json.dumps(out),flush=True)
    else:
        data,receipt=acquire(a.cache)
        raw=np.concatenate([v[2] for v in data]);states=np.concatenate([v[3] for v in data]);orig=np.array([[r['y'],r['x']] for w,rows,*_ in data for r in rows]);names=np.array([w for w,rows,*_ in data for r in rows])
        np.savez_compressed(a.out/'panel.npz',raw=raw,states=states,origins_yx=orig,names=names)
        out=dict(source=receipt,raw_sha256=[digest(x) for x in raw],states_sha256=[digest(x) for x in states],seconds=time.perf_counter()-t)
        (a.out/'prepare.json').write_text(json.dumps(out,indent=2)+'\n');print('B26_1447',json.dumps({k:v for k,v in out.items() if k!='source'}),flush=True)

if __name__=='__main__':main()
