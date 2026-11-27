#!/usr/bin/env python3
"""B11b: globally minimal fixed positive-stratum design, exact rational proof."""
import json
from fractions import Fraction
from math import comb
from audit import N, ALPHA, allocations, worst, cdf_num, balanced, min_budget


def optimize():
    path=balanced(N); srs={b:(b,) for b in range(1,sum(N)+1)}
    rows=[]
    for M in (6,12):
        for k in range(4):
            lower=min_budget(srs,(sum(N),),M,k)['budget']
            upper=min_budget(path,N,M,k)['budget']
            tried=[]
            for budget in range(lower,upper+1):
                total=pruned=0; feasible=[]
                for n in allocations(N,budget):
                    if min(n)==0: continue
                    total+=1
                    # Necessary condition from flag concentrations in each stratum.
                    if any(Fraction(cdf_num((a,),(b,),(M,),k),comb(a,b))>=ALPHA for a,b in zip(N,n) if M<=a):
                        pruned+=1; continue
                    p,w=worst(N,n,M,k)
                    if p<ALPHA: feasible.append((n,p,w))
                tried.append(dict(budget=budget,designs=total,concentration_pruned=pruned,full_evaluations=total-pruned,feasible=len(feasible)))
                if feasible:
                    n,p,w=feasible[0]
                    rows.append(dict(target=M,U=k,minimum=budget,draws=n,p_exact=str(p),p=float(p),witness=w,srs_lower=lower,balanced_upper=upper,search=tried));break
            else: raise AssertionError('no feasible design at known upper bound')
    return rows


if __name__=='__main__':
    print(json.dumps(dict(schema='b11b.global_fixed_stratum_minimum.v1',rows=optimize()),separators=(',',':')))
