"""CPU review ordering only. Never infer blankness from low score or missing support."""
import numpy as np


def regional_review(pf, pr, common, *, tile_size=512, threshold=.7, min_hits=1500,
                    min_ratio=5., max_bilateral=.05):
    """Evaluate both signs per tile; no coordinates or maps returned.

    This diagnostic is not a validated text classifier. In particular routine_review
    still requires review. Report common support, don't reinterpret unsupported pixels.
    """
    a,b,m=np.asarray(pf),np.asarray(pr),np.asarray(common)
    if a.shape!=b.shape or a.shape!=m.shape or m.ndim!=2 or m.dtype!=np.bool_:
        raise ValueError('inconsistent score/support geometry')
    if tile_size<1 or min_hits<1 or not 0<=threshold<=1 or min_ratio<1 or not 0<=max_bilateral<=1:
        raise ValueError('invalid diagnostic settings')
    n=int(m.sum())
    if not n:
        return dict(status='unresolved',common_pixels=0,qualified_tiles=0)
    for q in (a,b):
        if not np.isfinite(q[m]).all() or (q[m]<0).any() or (q[m]>1).any():
            raise ValueError('invalid supported scores')
    masks=[(q>=threshold)&m for q in (a,b)]
    counts={'forward':0,'reverse':0};eligible={'forward':0,'reverse':0}
    h,w=m.shape
    for y in range(0,h,tile_size):
        for x in range(0,w,tile_size):
            sl=np.s_[y:y+tile_size,x:x+tile_size];f,r=(v[sl] for v in masks)
            hits=[int(f.sum()),int(r.sum())];bil=int((f&r).sum())
            for j,sign in enumerate(counts):
                own,other=hits[j],hits[1-j]
                if own>=min_hits:
                    eligible[sign]+=1
                    if own/max(other,1)>=min_ratio and bil/own<max_bilateral:
                        counts[sign]+=1
    total=sum(counts.values())
    return dict(status='priority_review' if total else 'routine_review',common_pixels=n,
                common_fraction=n/m.size,qualified_tiles=total,by_direction=counts,
                eligible_tiles=eligible,p995=[float(np.quantile(q[m],.995)) for q in (a,b)],
                hits_ge_threshold=[int(v.sum()) for v in masks])
