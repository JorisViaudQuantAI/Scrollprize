"""Minimal 057 exact-cut extension: extra hard anchor arcs. Original solver preserved.
Derived from blob dc2ef1c8e38fd3cf06158e53a8d8a2cec0d1b4c7; objective and all existing constraints unchanged.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
import importlib.util,pathlib,sys
p=pathlib.Path(__file__).resolve().parents[1]/"dtw_order_3d"/"run.py"
sp=importlib.util.spec_from_file_location("base057_anchor_extension",p)
base=importlib.util.module_from_spec(sp);sys.modules[sp.name]=base;sp.loader.exec_module(base)
energy=base.energy
feasible=base.feasible
MAX_NODES=base.MAX_NODES
MAX_ARCS=base.MAX_ARCS

def cut3d(c,seed,anchor,*,lam=100,slope=4,gap=2,axes=(1,2),extra=()):
    c=np.asarray(c,np.int64);seed=np.asarray(seed,np.int64)
    S,Y,X,D=c.shape;K=D-1
    if D<2 or S<1 or min(Y,X)<1 or seed.shape!=(S,) or (seed<0).any() or (seed>=D).any() or not 0<=anchor[0]<Y or not 0<=anchor[1]<X:raise ValueError("Invalid cut inputs")
    if slope<0 or slope>=D or gap<0 or gap>=D or lam<0 or (gap and (np.diff(seed)<gap).any()) or any(a not in (1,2) for a in axes):raise ValueError("Infeasible cut contract")
    n=S*Y*X*K
    if n>MAX_NODES:raise ValueError("Threshold node cap")
    nodes=np.arange(n,dtype=np.int64).reshape(S,Y,X,K);src=n;snk=n+1
    diff=np.diff(c,axis=-1)
    finite=int(np.abs(diff).sum(dtype=np.int64))
    for a in axes:finite+=2*lam*K*(n//K//nodes.shape[a])*(nodes.shape[a]-1)
    H=finite+1
    if H>=2**30:raise ValueError("Integer capacity cap")
    us=[];vs=[];ws=[];count=0
    def add(a,b,z):
        nonlocal count
        a,b,z=np.broadcast_arrays(a,b,z);a=a.ravel();b=b.ravel();z=z.ravel();good=z>0
        if not good.any():return
        count+=int(good.sum())
        if count>MAX_ARCS:raise ValueError("Directed arc cap")
        us.append(a[good]);vs.append(b[good]);ws.append(z[good].astype(np.int64))
    add(nodes,snk,np.maximum(diff,0));add(src,nodes,np.maximum(-diff,0))
    add(nodes[...,1:],nodes[...,:-1],H)
    for ax in axes:
        lo=[slice(None)]*4;hi=lo.copy();lo[ax]=slice(None,-1);hi[ax]=slice(1,None)
        a,b=nodes[tuple(lo)],nodes[tuple(hi)]
        if lam:add(a,b,lam);add(b,a,lam)
        if slope<K:
            add(a[...,slope:],b[...,:K-slope],H)
            add(b[...,slope:],a[...,:K-slope],H)
    if gap and S>1:
        add(src,nodes[1:,:,:,gap-1],H)
        add(nodes[:-1,:,:,K-gap],snk,H)
        if gap<K:add(nodes[:-1,:,:,:K-gap],nodes[1:,:,:,gap:],H)
    constraints=[(s,anchor[0],anchor[1],int(z)) for s,z in enumerate(seed)]
    constraints += [tuple(map(int,a)) for a in extra]
    seen={}
    for s,y,x,z in constraints:
        if not (0<=s<S and 0<=y<Y and 0<=x<X and 0<=z<D):raise ValueError("Invalid additional anchor")
        key=(s,y,x)
        if key in seen and seen[key]!=z:raise ValueError("Conflicting hard anchors")
        seen[key]=z
    for (s,y,x),z in seen.items():
        a=nodes[s,y,x]
        if z>0:add(src,a[z-1],H)
        if z<K:add(a[z],snk,H)
    g=sparse.coo_matrix((np.concatenate(ws),(np.concatenate(us),np.concatenate(vs))),shape=(n+2,n+2)).tocsr()
    del us,vs,ws
    if g.data.max(initial=0)>=2**31:raise ValueError("Parallel arc overflow")
    f=maximum_flow(g,src,snk,method="dinic")
    if int(f.flow_value)>=H:raise ArithmeticError("Hard constraint cut")
    residual=(g-f.flow).tocsr()
    residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
    reach=breadth_first_order(residual,src,directed=True,return_predecessors=False)
    if snk in reach:raise ArithmeticError("Residual connects source and sink")
    m=np.zeros(n+2,bool);m[reach]=True;lev=m[nodes]
    if (lev[...,1:]&~lev[...,:-1]).any():raise ArithmeticError("Nonmonotone thresholds")
    x=lev.sum(-1,dtype=np.int64)
    if not feasible(x,seed,anchor,slope,gap,axes) or any(x[s,y,z]!=v for (s,y,z),v in seen.items()):raise ArithmeticError("Decoded infeasible")
    E=energy(c,x,lam,axes);constant=int(c[...,0].sum(dtype=np.int64)+np.minimum(diff,0).sum(dtype=np.int64))
    if E!=int(f.flow_value)+constant:raise ArithmeticError("Exact objective mismatch")
    return x,{"energy":E,"flow":int(f.flow_value),"constant":constant,"threshold_nodes":n,"arcs":int(g.nnz),"hard_capacity":H,"axes":list(axes)}
