"""P13 isolated CPU tests for exact stencil chunk access (no CT downloads)."""
from __future__ import annotations
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from stream_snap_render import RemoteZarrV2
from stencil_reader import StencilReader
from test_stream_snap_render import _write_local_zarr_v2

def run():
    rng=np.random.default_rng(1370);count=0;tests=[];fallback=0
    with tempfile.TemporaryDirectory() as temp:
        for j,dtype in enumerate((np.uint8,np.uint8,np.float32)):
            array=rng.integers(0,255,(55,53,59),dtype=np.uint8).astype(dtype)
            path=_write_local_zarr_v2(Path(temp)/str(j),array,chunks=(8,9,11),compressor={'id':'zlib','level':1})
            if j==1:(path/'0/1/2/2').unlink()
            shape=np.array(array.shape[::-1]);ints=rng.integers(0,shape,size=(2000,3)).astype(float)
            query=np.r_[rng.random((4000,3))*(shape-1),ints,np.nextafter(ints,np.inf),np.nextafter(ints,-np.inf),np.array([[np.nan,0,0],[np.inf,0,0],[-1,0,0],[shape[0],0,0]])]
            for bound in (0,16_000_000):
                for (cb,workers) in ((0.,1),(.015,3),(4.,4)):
                    with RemoteZarrV2(path,'0',cache_mb=cb,workers=workers) as ref, StencilReader(path,'0',cache_mb=cb,workers=workers,prefetch_mb=.01) as opt:
                        for coords in (query,query[:128].reshape(8,16,3),np.empty((0,3))):
                            aa=ref.sample_trilinear(coords,return_support=True,max_direct_box_voxels=bound)
                            bb=opt.sample_trilinear(coords,return_support=True,max_direct_box_voxels=bound)
                            assert np.array_equal(aa[0].view(np.uint32),bb[0].view(np.uint32)),(j,bound,cb,'values')
                            assert np.array_equal(aa[1],bb[1]),(j,bound,cb,'support')
                            count+=aa[0].size
                        fallback+=opt.sampler_stats['fallback']
                    tests.append(dict(kind=j,max_direct_box_voxels=bound,cache_mb=cb,workers=workers))
    assert count>100000 and fallback==6*3
    return dict(schema='p13.stencil_regression.v1',compared_values=count,cases=len(tests),fallback_invocations=fallback,scope='Synthetic Zarr and missing chunk, no private CT or GPU.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();j=run();a.out.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2))
