"""A geometry-only PHerc1203 mesh test of the frozen cross-scan registration."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import csv, hashlib, json
import numpy as np
import requests, tifffile
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

BASE = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
SAMPLE = 'PHerc1203'
MESH = 'auto_grown_20251005230830031'
ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT/'volumes.json').read_text())[SAMPLE]
AFFINE = np.array(SPEC['matrix'], float)
LINEAR = AFFINE[:, :3]
INVERSE = np.linalg.inv(LINEAR)
PITCH = 9.362
TRAIN = ROOT/'results'/'residuals.tsv'
assert hashlib.sha256(TRAIN.read_bytes()).hexdigest() == '624c459a6a6e1148802c8307b17e2514d6dcee77a0b76db801811e6e61a8f480'

def http(url, **kwargs):
    last = None
    for _ in range(3):
        try:
            r=requests.get(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last=e
    raise last

def meshfiles():
    # Use the current official XYZ files at the root of the segment, excluding editing snapshots.
    prefix=SAMPLE+'/segments/raw/'+MESH+'/'
    return {name:prefix+name for name in ('x.tif','y.tif','z.tif','meta.json')}

def get_mesh(keys):
    def get(axis):
        key=keys[axis+'.tif'];r=http(BASE+'/'+key)
        data=tifffile.imread(BytesIO(r.content))
        return data,hashlib.sha256(r.content).hexdigest()
    with ThreadPoolExecutor(3) as pool:
        results=list(pool.map(get,'xyz'))
    arrays=[r[0] for r in results]
    assert len({a.shape for a in arrays})==1 and arrays[0].ndim==2
    meta=http(BASE+'/'+keys['meta.json']).json() if 'meta.json' in keys else {}
    return arrays,dict(zip('xyz',[r[1] for r in results])),meta

def to_moving(x):
    return (np.asarray(x)-AFFINE[:,3])@INVERSE.T/4.

def select(arrays,shape0,shape1,levels,stride=8):
    ys,xs=np.mgrid[1:arrays[0].shape[0]-1:stride,1:arrays[0].shape[1]-1:stride]
    ys,xs=ys.ravel(),xs.ravel()
    def take(dv,du):
        return np.stack([a[ys+dv,xs+du] for a in arrays],axis=1).astype(float)
    p=take(0,0); neighbors=[take(0,1),take(0,-1),take(1,0),take(-1,0)]
    normal=np.cross(neighbors[0]-neighbors[1],neighbors[2]-neighbors[3])
    norm=np.linalg.norm(normal,axis=1)
    good=np.isfinite(p).all(1)&(p>0).all(1)&(norm>1e-4)
    for a in neighbors: good &= np.isfinite(a).all(1)&(a>0).all(1)&(np.linalg.norm(a-p,axis=1)<=28)
    fine=to_moving(p)
    good &= ((p>64)&(p<np.array(shape0)[::-1]-64)).all(1)
    good &= ((fine>64)&(fine<np.array(shape1)[::-1]-64)).all(1)
    eligible,sites={},[]
    for name,z in levels.items():
        ids=np.where(good&(np.abs(p[:,2]-z)<=160))[0]
        eligible[name]=len(ids)
        if not len(ids):continue
        # First point closest to the target height, then maximal physical spread, row-major ties.
        first=ids[np.argmin(np.abs(p[ids,2]-z))]
        selected=[first];dist=np.linalg.norm(p[ids]-p[first],axis=1)
        while len(selected)<4 and dist.max()>=100:
            k=ids[np.argmax(dist)];selected.append(k)
            dist=np.minimum(dist,np.linalg.norm(p[ids]-p[k],axis=1))
        for k in selected:
            n=normal[k]/norm[k]
            if n[np.argmax(np.abs(n))]<0:n=-n
            sites.append((name,str([int(xs[k]),int(ys[k])]),p[k],n))
    return int(len(p)),int(good.sum()),eligible,sites

def zmeta(name,level):
    url=BASE+'/'+SAMPLE+'/volumes/'+SPEC[name]+'/'+str(level)
    m=http(url+'/.zarray').json()
    assert m['dtype']=='|u1' and m['order']=='C' and m['compressor'] is None and m['chunks']==[128]*3
    return url,m

def box(grid,points):
    url,m=grid
    lo=np.floor(np.min(points,0)).astype(int)-2;hi=np.ceil(np.max(points,0)).astype(int)+3
    assert (lo>=0).all() and (hi<=np.array(m['shape'])[::-1]).all()
    a=np.zeros(tuple((hi-lo)[::-1]),dtype=np.uint8)
    tiles=[(x,y,z) for x in range(lo[0]//128,(hi[0]-1)//128+1) for y in range(lo[1]//128,(hi[1]-1)//128+1) for z in range(lo[2]//128,(hi[2]-1)//128+1)]
    def read(tile):
        t=np.array(tile);ll=np.maximum(lo,t*128);hh=np.minimum(hi,(t+1)*128)
        start=int((ll[2]-t[2]*128)*16384);length=int((hh[2]-ll[2])*16384)
        u=url+'/'+m.get('dimension_separator','.').join(map(str,tile[::-1]))
        r=http(u, headers={'Range':'bytes='+str(start)+'-'+str(start+length-1)})
        if r.status_code!=206 or len(r.content)!=length:raise IOError('Unexpected range '+u)
        slab=np.frombuffer(r.content,'u1').reshape(-1,128,128)
        a[ll[2]-lo[2]:hh[2]-lo[2],ll[1]-lo[1]:hh[1]-lo[1],ll[0]-lo[0]:hh[0]-lo[0]]=slab[:,ll[1]-128*t[1]:hh[1]-128*t[1],ll[0]-128*t[0]:hh[0]-128*t[0]]
    with ThreadPoolExecutor(6) as pool:list(pool.map(read,tiles))
    def sample(p):
        flat=(np.asarray(p).reshape(-1,3)-lo)[:,::-1].T
        return map_coordinates(a,flat,order=1,prefilter=False,output=float).reshape(np.asarray(p).shape[:-1])
    return sample

def prediction(rows,p):
    x=np.array([[float(r['fixed_'+ax]) for ax in 'xyz'] for r in rows]);d=np.array([[float(r['d'+ax+'_vox']) for ax in 'xyz'] for r in rows])
    origin=x.mean(0);D=(x-origin)/1000
    slope=np.linalg.solve(D.T@D+np.eye(3),D.T@(d-d.mean(0)))
    return d.mean(0)+(p-origin)@slope/1000

def measure(site,rows,coarse,fine):
    name,uv,p,n=site;training=[r for r in rows if r['site'][0]!=name]
    delta=prediction(training,p);dn=float(delta@n);dt=float(np.linalg.norm(delta-dn*n))
    sy,sx=np.mgrid[-32:32,-32:32]
    planes=np.array([p+np.stack((sx,sy,np.zeros_like(sx)+dz),axis=-1) for dz in (-13,13)])
    t=np.arange(-24,24.01,.5);profile=p+t[:,None]*n
    coords=np.concatenate([planes.reshape(-1,3),profile]);correct0=p-delta;correct=p-prediction(training,correct0)
    shift=p-correct
    f=box(coarse,coords);g=box(fine,np.concatenate([to_moving(coords),to_moving(coords-shift)]))
    stats=[]
    for plane in planes:
        ref=f(plane);a=g(to_moving(plane));b=g(to_moving(plane-shift))
        mask=binary_erosion((ref>3)&(a>3)&(b>3),iterations=2)
        mask[:8]=False;mask[-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
        out={'pixels':int(mask.sum())}
        if mask.sum()>=1152:
            def corr(x,y):
                xx,yy=x[mask],y[mask]
                return float(np.corrcoef(xx,yy)[0,1]) if min(np.std(xx),np.std(yy))>1e-6 else None
            filt=lambda v:gaussian_filter(v,1)-gaussian_filter(v,4)
            out.update(raw=[corr(ref,a),corr(ref,b)],dog=[corr(filt(ref),filt(a)),corr(filt(ref),filt(b))])
        stats.append(out)
    signal=gaussian_filter1d(f(profile),1.2);peaks=find_peaks(signal,prominence=12,distance=8)[0]
    near=peaks[np.abs(t[peaks])<=3];spacing=None
    if len(near)==1 and len(peaks)>=2:
        spacing=float(np.min(np.abs(t[peaks[peaks!=near[0]]]-t[near[0]])))
    # A second 1D diagnostic uses equal normal coordinates in fixed space.
    moving_profiles=[g(to_moving(profile)),g(to_moving(profile-shift))]
    profile_corr=[float(np.corrcoef(f(profile),q)[0,1]) for q in moving_profiles]
    peak_offsets=[]
    if len(near)==1:
        ref_t=t[near[0]]
        for q in moving_profiles:
            k=find_peaks(gaussian_filter1d(q,1.2),prominence=12,distance=8)[0]
            within=k[np.abs(t[k]-ref_t)<=6]
            peak_offsets.append(float(t[within[0]]-ref_t) if len(within)==1 else None)
    return dict(profile_corr=profile_corr,isolated_moving_peak_offsets=peak_offsets,height=name,uv=uv,p=p.tolist(),n=n.tolist(),train=len(training),d=delta.tolist(),normal_vox=dn,tangent_vox=dt,planes=stats,nearest_distinct_peak_vox=spacing,
                half_spacing_exceeded=None if spacing is None else bool(abs(dn)>=spacing/2))

def main():
    keys=meshfiles();arrays,hashes,meta=get_mesh(keys)
    coarse,fine=zmeta('coarse',0),zmeta('fine',2)
    rows=[r for r in csv.DictReader(TRAIN.open(),delimiter='\t') if r['sample']==SAMPLE and r['status']=='measured']
    frozen=json.loads((ROOT/'frozen-sites.json').read_text())['sites']
    levels={h:float(np.median([s['fixed'][2] for s in frozen if s['sample']==SAMPLE and s['id'][0]==h])) for h in 'ABC'}
    sampled,valid,eligible,sites=select(arrays,coarse[1]['shape'],fine[1]['shape'],levels)
    full_n,full_valid,full_eligible,full_sites=select(arrays,coarse[1]['shape'],fine[1]['shape'],levels,stride=1)
    # Exhaustive geometry supplies points only for strata entirely missed by stride eight.
    supplement=[s for s in full_sites if eligible[s[0]]==0]
    sites+=supplement
    print('SOURCE_SHA',json.dumps(hashes),flush=True)
    print('ALL_GEOMETRY',full_n,full_valid,full_eligible,flush=True)
    print('SELECTION',sampled,valid,eligible,'shape',arrays[0].shape,'metadata',meta,flush=True)
    results=[]
    for site in sites:
        r=measure(site,rows,coarse,fine);results.append(r);print('SITE',json.dumps(r,separators=(',',':')),flush=True)
    output=dict(mesh_keys=keys,source_sha256=hashes,mesh_meta=meta,shape=list(arrays[0].shape),sampled=sampled,eligible_by_bounds=valid,qualified_by_height=eligible,all_interior_vertices=full_n,all_valid=full_valid,all_qualified_by_height=full_eligible,target_heights=levels,full_grid_supplement=len(supplement),voxel_um=PITCH,results=results)
    Path('mesh-transfer-result.json').write_text(json.dumps(output,indent=2)+'\n')
    accepted=[p for r in results for p in r['planes'] if 'dog' in p and None not in p['dog']+p['raw']]
    print('SUMMARY',len(results),len(accepted),sum(p['pixels'] for p in accepted),'NORMAL_TANGENT_UM',
          [float(np.median([abs(r[k]) for r in results])*PITCH) for k in ('normal_vox','tangent_vox')] if results else [],
          {k:np.median([p[k] for p in accepted],axis=0).tolist() for k in ('dog','raw')} if accepted else {})
if __name__=='__main__':main()
