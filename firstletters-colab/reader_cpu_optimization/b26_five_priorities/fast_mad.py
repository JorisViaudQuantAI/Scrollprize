"""Exact uint8 specialization of Villa's unmasked, 1/99, float32 robust MAD.
Upstream: ScrollPrize/villa b29f310957c7b29e7d332f9876e0e487819cfaa3.
Do not use for masked normalization or different percentile/dtype contracts.
"""
import numpy as np


def reference(image):
    # Unmasked/default branch of upstream normalize_robust; operation order kept.
    arr=np.asarray(image).astype(np.float32,copy=True)
    valid=arr.reshape(-1)
    if valid.size==0:return np.ascontiguousarray(arr)
    std_preclip=float(np.std(valid));min_preclip=float(np.min(valid));max_preclip=float(np.max(valid))
    lower_val=float(np.percentile(valid,1.0));upper_val=float(np.percentile(valid,99.0))
    np.clip(arr,lower_val,upper_val,out=arr);valid=arr.reshape(-1)
    median=float(np.median(valid));mad=float(np.median(np.abs(valid-median)))
    scaled_mad=1.4826*mad
    if not np.isfinite(scaled_mad) or scaled_mad<1e-6:
        span=upper_val-lower_val
        fallback_values=[]
        if np.isfinite(std_preclip):fallback_values.append(abs(std_preclip))
        if np.isfinite(span):fallback_values.append(abs(span)/2.0)
        scaled_mad=next((v for v in fallback_values if v>=1e-6),None)
        if scaled_mad is None or not np.isfinite(scaled_mad):scaled_mad=1.0
    arr-=median;arr/=scaled_mad
    np.nan_to_num(arr,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    return np.ascontiguousarray(arr)


def _median(values,cumulative,n):
    a,b=np.searchsorted(cumulative,[(n-1)//2,n//2],side='right')
    return float(np.mean(values[[a,b]],dtype=np.float32))


def _quantile(values,cumulative,n,q):
    q=np.true_divide(q*100.0,np.float32(100),out=...)
    r=(n-1)*q;a=int(np.floor(r));b=min(a+1,n-1)
    k=np.searchsorted(cumulative,[a,b],side='right')
    return float(np.quantile(values[k],np.asarray(r-np.intp(a),dtype=np.asarray(r).dtype)))


def _lut(image):
    n=image.size
    if not n:return None
    hist=np.bincount(image.ravel(),minlength=256);cum=hist.cumsum()
    values=np.arange(256,dtype=np.float32)
    lo=_quantile(values,cum,n,.01);hi=_quantile(values,cum,n,.99)
    v=values.copy();np.clip(v,lo,hi,out=v)
    med=_median(v,cum,n);dev=np.abs(v-med)
    order=np.argsort(dev,kind='stable');mad=_median(dev[order],hist[order].cumsum(),n)
    scale=1.4826*mad
    if not np.isfinite(scale) or scale<1e-6:return None
    v-=med;v/=scale
    return v


def fast_uint8(image,fallback=reference):
    a=np.asarray(image)
    if a.dtype!=np.uint8:return fallback(a)
    lut=_lut(a)
    return fallback(a) if lut is None else np.ascontiguousarray(lut[a])


def fast_guarded(image,fallback=reference):
    a=np.asarray(image)
    if a.dtype==np.uint8:return fast_uint8(a,fallback)
    if a.dtype!=np.float32 or not a.size or not (a.min()>=0 and a.max()<=255):return fallback(a)
    if np.any((a==0)&np.signbit(a)):return fallback(a)
    b=a.astype(np.uint8)
    if not np.array_equal(a,b):return fallback(a)
    lut=_lut(b)
    return fallback(a) if lut is None else np.ascontiguousarray(lut[b])
