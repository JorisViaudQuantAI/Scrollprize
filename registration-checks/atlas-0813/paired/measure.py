"""Compare native CT profiles at the original 24 public atlas sites."""
import argparse
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
import check
import numpy as np
from scipy import ndimage, signal
import tifffile

OFFSETS = np.arange(-24, 24.5, .5)
PITCH = check.PITCH
TIE = 1.e-9
FRAMES = ('raw', 'same_xyz_new_frame', 'snapped_uv', 'projected_raw')


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if magnitude < 1.e-10 or not np.isfinite(magnitude):
        raise ValueError('Undefined orientation')
    return vector / magnitude


def frame(point, normal, tangent):
    normal = unit(normal)
    tangent = unit(tangent - normal * (tangent @ normal))
    return point, normal, tangent


def rays(f):
    p, n, t = f
    v = np.cross(n, t)
    lateral = np.array([[0., 0., 0.], 2*t, -2*t, 2*v, -2*v])
    return p + lateral[:, None, :] + OFFSETS[None, :, None] * n


def distance_set(values):
    if len(values) == 0:
        return {'minimum': None, 'indices': [], 'margin': None}
    ordered = np.sort(values)
    return {'minimum': float(ordered[0]),
            'indices': np.flatnonzero(np.abs(values - ordered[0]) <= TIE).tolist(),
            'margin': float(ordered[1] - ordered[0]) if len(values)>1 else None}


def peaks(traces):
    smooth = ndimage.gaussian_filter1d(traces.mean(0), 1.2)
    idx, _ = signal.find_peaks(smooth, prominence=12, distance=8)
    places = OFFSETS[idx]
    material = float(np.mean(traces[0] > 3))
    return places, material


class Mesh:
    def __init__(self, points, good, normals):
        y, x = np.where(good[:-1,:-1] & good[1:,:-1] & good[:-1,1:] & good[1:,1:])
        yx = np.stack([y,x], axis=-1)
        offsets = np.array([[[0,0],[0,1],[1,1]],[[0,0],[1,1],[1,0]]])
        self.uv = (yx[:, None, None, :] + offsets).reshape(-1,3,2)
        idx = tuple(self.uv.transpose(2,0,1))
        self.xyz = points[idx].astype(float)
        self.normals = normals[idx].astype(float)
        self.a, self.b, self.c = self.xyz.transpose(1,0,2)
        self.e, self.f = self.b-self.a, self.c-self.a
        self.ee, self.ef, self.ff = (np.einsum('ij,ij->i',u,v) for u,v in [(self.e,self.e),(self.e,self.f),(self.f,self.f)])
        self.det = self.ee*self.ff-self.ef**2

    def project(self, point):
        d=point-self.a
        de,df=(np.einsum('ij,ij->i',d,v) for v in (self.e,self.f))
        den=np.where(self.det>1.e-14,self.det,1.)
        u=(de*self.ff-df*self.ef)/den
        v=(df*self.ee-de*self.ef)/den
        bary=[np.stack([1-u-v,u,v],axis=-1)]
        acceptable=[(self.det>1.e-14)&(bary[0].min(axis=1)>=0)]
        for i,j in ((0,1),(1,2),(2,0)):
            start=self.xyz[:,i];edge=self.xyz[:,j]-start
            le=np.einsum('ij,ij->i',edge,edge)
            t=np.clip(np.einsum('ij,ij->i',point-start,edge)/np.maximum(le,1.e-20),0,1)
            w=np.zeros_like(bary[0]);w[:,i]=1-t;w[:,j]=t
            bary.append(w);acceptable.append(self.det>1.e-14)
        bary=np.stack(bary)
        pp=np.einsum('ctv,tvk->ctk',bary,self.xyz)
        ds=np.where(acceptable,np.linalg.norm(pp-point,axis=-1),np.inf)
        err=ds.min(axis=0);method=ds.argmin(axis=0)
        tied=np.flatnonzero(np.abs(err-err.min())<=1.e-7)
        i=tied[0];w=bary[method[i],i];picked=pp[method[i],i]
        reps=[]
        for k in tied:
            q=pp[method[k],k]
            if all(np.linalg.norm(q-r)>1.e-5 for r in reps):reps.append(q)
        uvs=self.uv[i]
        normal=w@self.normals[i]
        if not np.isfinite(normal).all() or np.linalg.norm(normal)<1.e-8:
            normal=np.cross(self.e[i],self.f[i])
        n=unit(normal)
        tangent=np.linalg.solve((uvs[1:]-uvs[0]).astype(float),self.xyz[i,1:]-self.xyz[i,0])[1]
        return frame(picked,n,tangent), {'triangle_uv_yx':uvs.tolist(),'barycentric':w.tolist(),
                'projected_uv_yx':(w@uvs).tolist(),'distance_vox':float(err[i]),
                'tied_triangles':len(tied),'distinct_tied_positions':len(reps)}


def unit_tests():
    rx,ry=np.meshgrid([0.,1.],[0.,1.])
    pts=np.stack([rx,ry,np.zeros_like(rx)],-1)
    normal=np.zeros_like(pts);normal[...,2]=1
    m=Mesh(pts,np.ones((2,2),bool),normal)
    for src,expected in [([.25,.25,2.],[.25,.25,0.]),([.8,.25,-3.],[.8,.25,0.]),([2.,.25,1.],[1.,.25,0.])]:
        (q,n,t),p=m.project(np.array(src))
        assert np.allclose(q,expected) and p['distinct_tied_positions']==1 and np.allclose(n,[0,0,1])
        assert abs(n@t)<1.e-8
    assert distance_set(np.array([1.,1.,4.]))['indices']==[0,1]
    assert distance_set(np.array([]))['minimum'] is None
    assert distance_set(np.array([2.]))['margin'] is None


def landmarks(f, places, material, a, p):
    center,n,_=f
    nearest=distance_set(np.abs(places)) if material>=.8 else distance_set([])
    obj={'center_minus_raw_xyz_vox':(center-p).tolist(),'normal_xyz':n.tolist(),
         'peak_offsets_vox':places.tolist(),'material_center_fraction':material,
         'nearest_peak_to_center':nearest}
    if a is not None:
        depth=float((a-center)@n)
        transverse=float(np.linalg.norm((a-center)-depth*n))
        crests=center+places[:,None]*n
        obj['fixed_raw_landmark']={'signed_depth_vox':depth,'ray_miss_vox':transverse,
             'center_distance_vox':float(np.linalg.norm(a-center)),
             'nearest_crest_to_landmark_3d':distance_set(np.linalg.norm(crests-a,axis=1)) if material>=.8 else distance_set([])}
    return obj


def summary(values):
    a=np.asarray([x for x in values if x is not None],float)
    return {'n':len(a),'median':float(np.median(a)) if len(a) else None,'max':float(a.max()) if len(a) else None}


def run():
    unit_tests()
    base=check.D
    report=json.loads((base/'result.json').read_text())
    def load(d):return np.stack([tifffile.imread(d/(q+'.tif')) for q in 'xyz'],-1)
    raw=load(base/'original'/check.W);snapped=load(base/'snapped'/check.W)
    for q in 'xyz':
        assert check.sha((base/'snapped'/check.W/(q+'.tif')).read_bytes())==report['snapped_sha256'][q]
    valid=np.isfinite(raw).all(2)&(raw>-.5).all(2)
    valid_s=np.isfinite(snapped).all(2)&(snapped>-.5).all(2)
    nraw,_=check.real_normals(raw,valid);nafter,ok=check.real_normals(snapped,valid_s)
    sites,min_dist=check.sample_sites(raw,valid,nraw)
    mesh=Mesh(snapped,valid_s,nafter)
    samples=[];all_frames=[];prepared=[]
    assert len(sites)==24
    for y,x in sites:
        p=raw[y,x].astype(float);s=snapped[y,x].astype(float)
        if not ok[y,x]:raise ValueError('Undefined snapped site normal')
        nr=nraw[y,x].astype(float);ns=nafter[y,x].astype(float)
        tr=(raw[y,x+1]-raw[y,x-1]).astype(float)
        ts=(snapped[y,x+1]-snapped[y,x-1]).astype(float)
        fq,proj=mesh.project(p)
        frames=[frame(p,nr,tr),frame(p,ns,ts),frame(s,ns,ts),fq]
        samples.extend(rays(f) for f in frames)
        all_frames.append(frames);prepared.append((p,s,proj))
    traces,receipt=check.ct_values(np.asarray(samples))
    traces=traces.reshape(24,len(FRAMES),5,len(OFFSETS))
    records=[];csvrows=[]
    for i,(yx,frames,prep,tset) in enumerate(zip(sites,all_frames,prepared,traces),1):
        p,s,pr=prep;delta=s-p;nr=frames[0][1];ns=frames[2][1]
        offsets,material=peaks(tset[0]);d=float(delta@nr)
        old=distance_set(abs(offsets));after=distance_set(abs(offsets-d))
        assert material>=.8 and old['indices']
        unique=len(old['indices'])==1
        anchor_t=float(offsets[old['indices'][0]]) if unique else None
        anchor_xyz=None if anchor_t is None else p+anchor_t*nr
        changed=bool(old['indices'][0]!=after['indices'][0]) if unique and len(after['indices'])==1 else None
        dt=float(np.linalg.norm(delta-d*nr))
        theta=float(np.rad2deg(np.arccos(np.clip(nr@ns,-1,1))))
        r={'id':i,'reserved':i%2==0,'public_uv_yx':yx.tolist(),'same_uv_delta_xyz_vox':delta.tolist(),
           'pitch_um':PITCH,'same_uv_displacement_vox':float(np.linalg.norm(delta)),
           'along_raw_normal_vox':d,'across_raw_normal_vox':dt,'normal_angle_deg':theta,
           'ray_tip_separation_vox':{str(k):float(k*np.linalg.norm(nr-ns)) for k in (6,12,24)},
           'projection_of_raw_onto_snap':pr,'fixed_landmark_raw_offset_vox':anchor_t,
           'legacy_raw_nearest':old,'legacy_snap_nearest':after,'legacy_nearest_peak_changed':changed,
           'legacy_same_peak_after_vox':None if not unique else abs(anchor_t-d),'rays':{}}
        if anchor_xyz is not None:
            fa,ap=mesh.project(anchor_xyz)
            ap.update({'projected_minus_raw_xyz_vox':(fa[0]-p).tolist(),'normal_xyz':fa[1].tolist()})
            r['projection_of_fixed_landmark_onto_snap']=ap
        for tag,f,tt in zip(FRAMES,frames,tset):
            pp,mm=peaks(tt);r['rays'][tag]=landmarks(f,pp,mm,anchor_xyz,p)
        nray=r['rays']['snapped_uv'];ar=nray.get('fixed_raw_landmark',{})
        pro=r['rays']['projected_raw'];aq=pro.get('fixed_raw_landmark',{})
        z={'id':i,'split':'reserved' if i%2==0 else 'setup','uv_row':int(yx[0]),'uv_col':int(yx[1]),
           'same_uv_shift_vox':r['same_uv_displacement_vox'],'same_uv_shift_um':r['same_uv_displacement_vox']*PITCH,
           'along_raw_normal_vox':d,'across_raw_normal_vox':dt,'normal_change_deg':theta,
           'normal_only_at_12_um':r['ray_tip_separation_vox']['12']*PITCH,
           'raw_project_distance_vox':pr['distance_vox'],'raw_project_distance_um':pr['distance_vox']*PITCH,
           'raw_projection_tied_positions':pr['distinct_tied_positions'],
           'raw_n_peaks':len(offsets),'raw_nearest_vox':old['minimum'],'raw_nearest_count':len(old['indices']),
           'raw_runner_up_gap_vox':old['margin'],'old_crest_signed_offset_vox':anchor_t,
           'snap_to_old_crest_vox':r['legacy_same_peak_after_vox'],'snap_to_any_old_peak_vox':after['minimum'],
           'snap_nearest_on_old_ray_count':len(after['indices']),'snap_runner_up_gap_on_old_ray_vox':after['margin'],
           'nearest_old_peak_changed':changed,'snap_n_peaks_new_ray':len(nray['peak_offsets_vox']),
           'snap_to_any_new_ray_peak_vox':nray['nearest_peak_to_center']['minimum'],
           'snap_new_ray_tie_count':len(nray['nearest_peak_to_center']['indices']),
           'anchor_depth_in_snap_vox':ar.get('signed_depth_vox'),'anchor_miss_from_snap_ray_um':None if not unique else ar['ray_miss_vox']*PITCH,
           'anchor_to_nearest_new_crest_3d_um':None if not unique or ar['nearest_crest_to_landmark_3d']['minimum'] is None else ar['nearest_crest_to_landmark_3d']['minimum']*PITCH,
           'anchor_miss_from_projected_ray_um':None if not unique else aq['ray_miss_vox']*PITCH}
        records.append(r);csvrows.append(z)
    # Reproduce both previously published paired endpoints in exactly the original frame.
    W=report['metrics']
    def verify(field,rows,ref):
        aa=np.array([v['raw_nearest_vox'] for v in rows]);bb=np.array([v[field] for v in rows]);diff=aa-bb
        ans={'n':len(rows),'median_before_vox':np.median(aa),'median_after_vox':np.median(bb),'median_paired_gain_vox':np.median(diff),
             'closer_over_quarter':(diff>.25).sum(),'further_over_quarter':(diff<-.25).sum(),'within_quarter':(abs(diff)<=.25).sum(),
             'within_two_before':(aa<=2).sum(),'within_two_after':(bb<=2).sum()}
        for k,v in ans.items():
            if not np.isclose(v,W[ref][k],rtol=0,atol=2.e-7):raise ValueError((ref,k,v,W[ref][k]))
    verify('snap_to_old_crest_vox',[v for v in csvrows if v['raw_nearest_count']==1],'primary_distance_to_original_peak')
    verify('snap_to_any_old_peak_vox',csvrows,'distance_to_any_peak')
    assert sum(v['nearest_old_peak_changed'] is True for v in csvrows)==W['changed_nearest_peak']
    def dump(name,obj):
        (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    with (OUT/'sites.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader()
        for row in csvrows:writer.writerow({k:round(v,8) if isinstance(v,float) else v for k,v in row.items()})
    tag={'source_protocol_sha256':report['protocol_sha256'],'extension_protocol_sha256':check.sha((OUT/'protocol.md').read_bytes()),
         'input_winding':check.W,'pitch_um':PITCH,'coordinates':'relative XYZ in native voxels, raw UV indexes public upstream TIFFs',
         'frame_order':FRAMES,'records':records}
    dump('geometry.json',tag)
    out={'n_sites':len(records),'minimum_site_separation_vox':min_dist,'reserved_ids':list(range(2,25,2)),'native_ct':receipt,
         'nearest_crest_switches_fixed_ray':W['changed_nearest_peak'],'raw_ties':sum(v['raw_nearest_count']!=1 for v in csvrows),
         'snap_ties_fixed_ray':sum(v['snap_nearest_on_old_ray_count']!=1 for v in csvrows),
         'all_three_coordinate_tiff_hashes_match':True,'legacy_endpoints_match':True,
         'summaries':{k:summary([v[k] for v in csvrows]) for k in ['same_uv_shift_um','normal_change_deg','normal_only_at_12_um',
                     'raw_project_distance_um','anchor_miss_from_snap_ray_um','anchor_to_nearest_new_crest_3d_um','anchor_miss_from_projected_ray_um']},
         'frames':{}}
    for tag in FRAMES:
        fs=[r['rays'][tag] for r in records]
        out['frames'][tag]={'material_and_peak':sum(f['material_center_fraction']>=.8 and len(f['peak_offsets_vox'])>0 for f in fs),
             'multi_peak_profiles':sum(len(f['peak_offsets_vox'])>1 for f in fs),
             'nearest_exact_ties':sum(len(f['nearest_peak_to_center']['indices'])>1 for f in fs),
             'nearest_within_half_voxel_of_runner_up':sum(f['nearest_peak_to_center']['margin'] is not None and f['nearest_peak_to_center']['margin']<=.5 for f in fs),
             'distance_to_nearest_vox':summary([f['nearest_peak_to_center']['minimum'] for f in fs])}
    dump('summary.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    flags=argparse.ArgumentParser()
    flags.add_argument('--self-test',action='store_true')
    if flags.parse_args().self_test:unit_tests()
    else:run()
