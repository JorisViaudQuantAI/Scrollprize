"""Check the published relative coordinates and derived distances without CT access."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ATOL=1.e-7


def near(first,second):
    np.testing.assert_allclose(first,second,rtol=0,atol=ATOL)


def nearest(values,record):
    if not len(values):
        assert record=={'minimum':None,'indices':[],'margin':None}
        return
    ordered=sorted(values)
    near(ordered[0],record['minimum'])
    assert record['indices']==[i for i,v in enumerate(values) if abs(v-ordered[0])<=1.e-9]
    if len(values)>1:near(ordered[1]-ordered[0],record['margin'])
    else:assert record['margin'] is None


def proj(doc,relative):
    uv=np.array(doc['triangle_uv_yx']);w=np.array(doc['barycentric'])
    assert min(w)>-ATOL
    near(sum(w),1)
    near(w@uv,doc['projected_uv_yx'])
    near(np.linalg.norm(relative),doc['distance_vox'])


def main():
    data=json.loads((HERE/'geometry.json').read_text())
    results=json.loads((HERE/'summary.json').read_text())
    baseline=json.loads((HERE.parent/'result.json').read_text())['metrics']
    with (HERE/'sites.csv').open() as f:csv_sites=list(csv.DictReader(f))
    assert len(data['records'])==len(csv_sites)==24
    changed=0;used=0;d1=[];d2=[]
    for i,(r,tab) in enumerate(zip(data['records'],csv_sites),1):
        assert r['id']==int(tab['id'])==i and r['reserved']==(i%2==0)
        assert r['public_uv_yx']==[int(tab['uv_row']),int(tab['uv_col'])]
        delta=np.array(r['same_uv_delta_xyz_vox']);rr=r['rays'];old=rr['raw'];snap=rr['snapped_uv'];qp=rr['projected_raw']
        n=np.asarray(old['normal_xyz']);n1=np.asarray(snap['normal_xyz'])
        near(n@delta,float(tab['along_raw_normal_vox']))
        near(rr['same_xyz_new_frame']['center_minus_raw_xyz_vox'],[0,0,0])
        near(rr['same_xyz_new_frame']['normal_xyz'],n1)
        near(snap['center_minus_raw_xyz_vox'],delta)
        near(np.linalg.norm(delta),r['same_uv_displacement_vox'])
        near(np.degrees(np.arccos(np.clip(n@n1,-1,1))),r['normal_angle_deg'])
        for k in [6,12,24]:near(np.linalg.norm(k*n-k*n1),r['ray_tip_separation_vox'][str(k)])
        proj(r['projection_of_raw_onto_snap'],qp['center_minus_raw_xyz_vox'])
        positions=np.array(old['peak_offsets_vox'])
        nearest(abs(positions),r['legacy_raw_nearest'])
        nearest(abs(positions-r['along_raw_normal_vox']),r['legacy_snap_nearest'])
        d2.append([r['legacy_raw_nearest']['minimum'],r['legacy_snap_nearest']['minimum']])
        chosen=r['fixed_landmark_raw_offset_vox']
        a=None if chosen is None else n*chosen
        if a is not None:
            p=r['projection_of_fixed_landmark_onto_snap']
            proj(p,np.asarray(p['projected_minus_raw_xyz_vox'])-a)
            near(abs(chosen-r['along_raw_normal_vox']),r['legacy_same_peak_after_vox'])
            d1.append([abs(chosen),r['legacy_same_peak_after_vox']])
            if r['legacy_nearest_peak_changed'] is not None:
                changed+=int(r['legacy_raw_nearest']['indices']!=r['legacy_snap_nearest']['indices']);used+=1
        for item in rr.values():
            o=np.asarray(item['center_minus_raw_xyz_vox']);normal=np.array(item['normal_xyz'])
            near(normal@normal,1)
            offsets=np.array(item['peak_offsets_vox']);supported=item['material_center_fraction']>=.8
            nearest(abs(offsets) if supported else [],item['nearest_peak_to_center'])
            if a is None:
                assert 'fixed_raw_landmark' not in item
                continue
            fixed=item['fixed_raw_landmark'];distance=(a-o)@normal
            near(distance,fixed['signed_depth_vox']);near(np.linalg.norm(np.cross(a-o,normal)),fixed['ray_miss_vox'])
            near(np.linalg.norm(a-o),fixed['center_distance_vox'])
            obs=o+normal*offsets[:,None]
            nearest(np.linalg.norm(obs-a,axis=1) if supported else [],fixed['nearest_crest_to_landmark_3d'])
    assert changed==baseline['changed_nearest_peak']==results['nearest_crest_switches_fixed_ray']
    assert used==baseline['unique_nearest_both']
    for array,key in [(d1,'primary_distance_to_original_peak'),(d2,'distance_to_any_peak')]:
        a,b=np.array(array).T;out=baseline[key];gain=a-b
        near([np.median(a),np.median(b),np.median(gain)],[out[k] for k in ('median_before_vox','median_after_vox','median_paired_gain_vox')])
        assert len(a)==out['n'] and sum(gain>.25)==out['closer_over_quarter'] and sum(gain<-.25)==out['further_over_quarter']
    print(f'24 public sites, 96 native CT frames, {len(d1)} unique original anchors, {changed}/{used} changed nearest crests. All coordinate-identity checks pass.')


if __name__=='__main__':main()
