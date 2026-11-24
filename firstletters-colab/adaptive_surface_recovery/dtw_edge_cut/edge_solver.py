"""Exact all-edge monotone correspondence extension of 061 anchor solver.
Source blob 190b0fb3cd1adb300ab7d6c20f01593d5ee3c6f0; no changes to 057 unary, gap, slope or exact energy checks.
The added XOR graph construction is equivalent to a submodular ordered-label
pairwise term, not a tree-path prior.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"multi_anchor_consensus"))
import anchored_cut as ac
base=ac.base
energy=base.energy
feasible=base.feasible
MAX_NODES=base.MAX_NODES
MAX_ARCS=base.MAX_ARCS

def edge_energy(x,maps,weight):
    """Independent direct arithmetic of the actual directed pairwise objective."""
    total=0
    for axis,reverse,m in maps:
        lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        p,q=x[tuple(lo)],x[tuple(hi)]
        if reverse:p,q=q,p
        take=np.take_along_axis(np.broadcast_to(m[None],(len(x),)+m.shape),p[...,None],axis=-1)[...,0]
        total+=int(weight)*int(np.abs(q-take).sum(dtype=np.int64))
    return total

def cut3d(c,seed,anchor,*,lam=100,slope=4,gap=2,axes=(1,2),extra=(),maps=(),weight=50):
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
    # Each directed map contributes at most 2*w per threshold XOR.
    if weight<0 or int(weight)!=weight:raise ValueError("Invalid edge weight")
    for axis,reverse,mp in maps:
        expected=(Y-1,X,D) if axis==1 else (Y,X-1,D)
        if axis not in axes or mp.shape!=expected or (mp<0).any() or (mp>=D).any() or (np.diff(mp,axis=-1)<0).any():raise ValueError("Invalid monotone DTW map")
        finite+=2*int(weight)*S*K*int(np.prod(mp.shape[:-1]))
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
    # |q-m(p)| = sum_t XOR(q>=t, p>=lower_bound(m,t)).
    # Map constants use terminals; internal threshold XOR uses two equal arcs.
    if weight:
        for axis,reverse,mp in maps:
            l=[slice(None)]*4;h=l.copy();l[axis]=slice(None,-1);h[axis]=slice(1,None)
            u,v=nodes[tuple(l)],nodes[tuple(h)]
            if reverse:u,v=v,u
            th=np.arange(1,D,dtype=np.int64)
            flat=mp.reshape(-1,D)
            k=np.stack([np.searchsorted(q,th,side="left") for q in flat]).reshape(mp.shape[:-1]+(K,))
            k=np.broadcast_to(k[None],u.shape)
            other=np.take_along_axis(u,np.clip(k-1,0,K-1),axis=-1)
            interior=(k>0)&(k<D)
            add(other[interior],v[interior],weight)
            add(v[interior],other[interior],weight)
            add(src,v[k==0],weight)
            add(v[k==D],snk,weight)
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
    E=energy(c,x,lam,axes)+edge_energy(x,maps,weight);constant=int(c[...,0].sum(dtype=np.int64)+np.minimum(diff,0).sum(dtype=np.int64))
    if E!=int(f.flow_value)+constant:raise ArithmeticError("Exact objective mismatch")
    return x,{"energy":E,"flow":int(f.flow_value),"constant":constant,"threshold_nodes":n,"arcs":int(g.nnz),"hard_capacity":H,"axes":list(axes)}
