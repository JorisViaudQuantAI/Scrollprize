"""Opt-in uint8 zero-fill Zarr sampler: positive-weight chunk planning.

Interpolation remains scipy.ndimage.map_coordinates (order=1, float32 output).
No change to physical coordinates or the missing-source contract.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy import ndimage as ndi
try:
    from .stream_snap_render import RemoteZarrV2, _positive_weight_stencil_support
except ImportError:
    from stream_snap_render import RemoteZarrV2, _positive_weight_stencil_support

_BITS = np.array([4,2,1],np.int64)
_SUBSETS = tuple(tuple(i for i in range(8) if (i & j)==i) for j in range(8))
_OFFSETS = tuple((i//4,(i//2)%2,i%2) for i in range(8))

class StencilReader(RemoteZarrV2):
    """Batch chunk-cell requests; tight direct path for small dense bounding boxes.

    ``prefetch_mb`` bounds decoded chunk references per group batch (except a
    single cell's <=8 mandatory chunks); excludes resident LRU, point arrays and
    one dense interpolation block. Legacy fallback for non-uint8/zero-fill.
    """
    def __init__(self,*args,prefetch_mb=64.,**kwargs):
        if not np.isfinite(prefetch_mb) or prefetch_mb<=0:raise ValueError('invalid prefetch budget')
        super().__init__(*args,**kwargs)
        self.prefetch_bytes=int(prefetch_mb*1024**2)
        self.sampler_stats=dict(direct=0,sparse=0,fallback=0,groups=0,prefetch_batches=0,planned_chunk_references=0)

    def _interpolate(self,q,ids,lo,hi,arrays,out,available):
        size=hi-lo
        block=np.zeros(tuple(size),np.float32)
        missing=available is not None and any(a is None for a in arrays.values())
        support=np.zeros(block.shape,bool) if missing else None
        for key,a in arrays.items():
            if a is None:continue
            start=np.asarray(key)*self.chunks
            aa=np.maximum(lo,start);bb=np.minimum(hi,start+np.asarray(self.chunks))
            if np.any(bb<=aa):continue
            to=tuple(slice(int(x),int(y)) for x,y in zip(aa-lo,bb-lo))
            fr=tuple(slice(int(x),int(y)) for x,y in zip(aa-start,bb-start))
            block[to]=a[fr]
            if support is not None:support[to]=True
        local=(q-lo).T
        out[ids]=ndi.map_coordinates(block,local,order=1,mode='constant',cval=0.,output=np.float32)
        if available is not None:
            available[ids]=True if support is None else _positive_weight_stencil_support(support,local)

    def sample_trilinear(self,xyz,*,max_direct_box_voxels=16_000_000,return_support=False):
        stats=self.sampler_stats
        if self.dtype!=np.dtype(np.uint8) or self.fill_value!=0:
            stats['fallback']+=1
            return super().sample_trilinear(xyz,max_direct_box_voxels=max_direct_box_voxels,return_support=return_support)
        pts=np.asarray(xyz,np.float64)
        if pts.ndim==0 or pts.shape[-1]!=3:raise ValueError('expected XYZ coordinates')
        shape=pts.shape[:-1];q=pts.reshape(-1,3)[:,::-1];sh=np.asarray(self.shape,np.int64);c=np.asarray(self.chunks,np.int64)
        out=np.zeros(len(q),np.float32);avail=np.zeros(len(q),bool) if return_support else None
        def result():return (out.reshape(shape),avail.reshape(shape)) if return_support else out.reshape(shape)
        ids=np.flatnonzero(np.isfinite(q).all(1)&(q>=0).all(1)&(q<=sh-1).all(1))
        if not len(ids):return result()
        q=q[ids];lo=np.floor(q.min(0)).astype(np.int64);hi=np.ceil(q.max(0)).astype(np.int64)+1
        c0=lo//c;c1=(hi-1)//c;nc=int(np.prod(c1-c0+1))
        # Small boxes avoid sorting large, dense tile queries. They may include
        # extra chunks, but never more than eight, with explicit telemetry.
        if nc<=8 and int(np.prod(hi-lo))<=max_direct_box_voxels:
            keys=list(itertools.product(*(range(int(a),int(b)+1) for a,b in zip(c0,c1))))
            stats['direct']+=1;stats['planned_chunk_references']+=len(keys);stats['prefetch_batches']+=1
            self._interpolate(q,ids,lo,hi,self.get_chunks(keys),out,avail)
            return result()
        stats['sparse']+=1
        low=np.floor(q).astype(np.int64);cell=low//c
        flags=(((low+1)//c!=cell)&(q>low))@_BITS
        grid=(sh+c-1)//c
        key=np.ravel_multi_index(cell.T,tuple(grid));order=np.argsort(key,kind='stable');sorted_key=key[order]
        cuts=np.r_[0,np.flatnonzero(sorted_key[1:]!=sorted_key[:-1])+1,len(order)]
        batch=[];union=set();max_chunks=max(1,self.prefetch_bytes//self._expected_chunk_bytes)
        def flush():
            if not batch:return
            stats['prefetch_batches']+=1;stats['planned_chunk_references']+=len(union)
            fetched=self.get_chunks(sorted(union))
            for inds,keys,lo,hi in batch:self._interpolate(q[inds],ids[inds],lo,hi,{k:fetched[k] for k in keys},out,avail)
            batch.clear();union.clear()
        for a,b in zip(cuts[:-1],cuts[1:]):
            inds=order[a:b];base=cell[inds[0]]
            # A required neighboring chunk corresponds to a subset of at least
            # one point's positive-weight boundary flags, not their global OR.
            sub=set()
            for f in np.unique(flags[inds]):sub.update(_SUBSETS[int(f)])
            keys=tuple(tuple(int(x+y) for x,y in zip(base,_OFFSETS[i])) for i in sorted(sub))
            if batch and len(union.union(keys))>max_chunks:flush()
            qg=q[inds];lo=low[inds].min(0);hi=np.ceil(qg.max(0)).astype(np.int64)+1
            batch.append((inds,keys,lo,hi));union.update(keys);stats['groups']+=1
        flush();return result()
