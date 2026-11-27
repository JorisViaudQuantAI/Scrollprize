#!/usr/bin/env python3
"""B11: exact worst-case finite-population audit, conservative unresolved count."""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import comb, prod
import json

N=(23,23,38,25)
PLANS=((6,6,11,7),(9,10,16,10),(13,13,20,14))
ALPHA=Fraction(1,20)


def allocations(cap, total):
    if not cap:
        if total==0: yield ()
        return
    for a in range(max(0,total-sum(cap[1:])),min(total,cap[0])+1):
        for rest in allocations(cap[1:],total-a): yield (a,)+rest


def choose(n,k):
    return comb(n,k) if 0<=k<=n else 0


@lru_cache(maxsize=None)
def masses(N,n,m,k):
    return tuple(choose(m,x)*choose(N-m,n-x) for x in range(k+1))


def cdf_num(N,n,m,k):
    acc=[1]+[0]*k
    for a,b,c in zip(N,n,m):
        h=masses(a,b,c,k)
        acc=[sum(acc[j]*h[i-j] for j in range(i+1)) for i in range(k+1)]
    return sum(acc)


@lru_cache(maxsize=None)
def worst(N,n,M,k):
    if not N or len(N)!=len(n) or any(type(x)!=int for x in (*N,*n,M,k)) or any(a<=0 or not 0<=b<=a for a,b in zip(N,n)) or not 0<=M<=sum(N) or not 0<=k<=sum(n):
        raise ValueError('invalid population, allocation, target or count')
    den=prod(comb(a,b) for a,b in zip(N,n))
    best=-1; witness=None
    for m in allocations(N,M):
        num=cdf_num(N,n,m,k)
        if num>best: best,witness=num,m
    return Fraction(best,den),witness


def upper(N,n,k):
    lo,hi=0,sum(N)+1
    while hi-lo>1:
        m=(lo+hi)//2
        if worst(N,n,m,k)[0]<ALPHA: hi=m
        else: lo=m
    return lo


def balanced(N):
    n=[1]*len(N); out={sum(n):tuple(n)}
    while tuple(n)!=N:
        i=min((i for i in range(len(N)) if n[i]<N[i]),key=lambda i:(Fraction(n[i],N[i]),i))
        n[i]+=1; out[sum(n)]=tuple(n)
    return out


def min_budget(path,N,M,k):
    lo=max(min(path),k)-1;hi=max(path)
    if worst(N,path[hi],M,k)[0]>=ALPHA: return None
    while hi-lo>1:
        b=(lo+hi)//2
        if worst(N,path[b],M,k)[0]<ALPHA: hi=b
        else: lo=b
    p,m=worst(N,path[hi],M,k)
    return dict(budget=hi,draws=path[hi],p_exact=str(p),p=float(p),witness=m)


def literal(N,n,m,k):
    # Independent oracle: enumerate literal subsets, flags fixed as first m units.
    pools=[list(combinations(range(a),b)) for a,b in zip(N,n)]
    num=0; den=0
    for draws in product(*pools):
        num+=sum(sum(x<c for x in d) for d,c in zip(draws,m))<=k
        den+=1
    return Fraction(num,den)


def selftest():
    checks=0
    for a in range(1,5):
      for b in range(1,5):
       N0=(a,b)
       for n in product(range(a+1),range(b+1)):
        prev=[Fraction(1)]*(sum(n)+1)
        for M in range(a+b+1):
         for k in range(sum(n)+1):
          expected=max(literal(N0,n,m,k) for m in allocations(N0,M))
          p,_=worst(N0,n,M,k)
          assert p==expected and p<=prev[k]
          prev[k]=p;checks+=1
    assert worst(N,PLANS[1],6,0)[0]==Fraction(10090080,242272800)
    assert worst(N,N,6,5)[0]==0 and worst(N,N,6,6)[0]==1
    for args in [((2,),(3,),1,0),((2,),(1,),3,0),((2,),(1,),1,2),((2,),(1,),1,-1),((2,),(1.0,),1,0)]:
        try: worst(*args)
        except ValueError: pass
        else: raise AssertionError('invalid accepted')
    return dict(literal_subset_oracle_cases=checks,invalid_cases=5,b6_zero_reproduced=True,census_checked=True)


def main():
    tests=selftest(); rows=[]; minimum=[]; intervals=[]
    path=balanced(N); srs={b:(b,) for b in range(1,sum(N)+1)}
    for n in PLANS:
      for k in range(4):
        intervals.append(dict(budget=sum(n),draws=n,U=k,upper_95=upper(N,n,k)))
        for M in (6,12):
            p,m=worst(N,n,M,k)
            rows.append(dict(budget=sum(n),draws=n,U=k,target=M,p_exact=str(p),p=float(p),witness=m,reject=p<ALPHA))
    for M in (6,12):
      for k in range(4):
        minimum.append(dict(target=M,U=k,balanced=min_budget(path,N,M,k),srs=min_budget(srs,(sum(N),),M,k)))
    print(json.dumps(dict(schema='b11.audit_unresolved.v1',source=dict(b6_blob='1bf0bd72cd8507814d2b5e411622e25e1567a29a',strata=N),tests=tests,fixed_plans=rows,upper_bounds=intervals,minimum_budgets=minimum,scope='Design-only; no new reviews; resolved negatives assumed correct for visual flag endpoint; unresolved counted conservatively; no production skip.'),indent=2))


if __name__=='__main__': main()
