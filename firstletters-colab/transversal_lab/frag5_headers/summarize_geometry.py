#!/usr/bin/env python3
"""Summarize already acquired header/PPM geometry records; no network or images."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'source_metadata'
plan=json.loads((ROOT/'preflight_plan.json').read_text())
h=json.loads((ROOT/'header_results.json').read_text())
g=json.loads((ROOT/'geometry_results.json').read_text())
r=json.loads((ROOT/'range_receipt.json').read_text())
fine=json.loads((OLD/'fine_volume_meta.json').read_text())
coarse=json.loads((OLD/'coarse_volume_meta.json').read_text())
affine=json.loads((OLD/'affine.json').read_text())

def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def matrix(p):
    a=affine['params']
    return [sum(a[i][j]*p[j] for j in range(3))+a[i][3] for i in range(3)]

def in_bounds(p, shape, trilinear=False):
    # Conservative complete eight-corner support uses floor(index)+1 on each axis.
    return all(math.isfinite(v) and 0 <= v and
        (v < dim-1 if trilinear else v <= dim-1) for v,dim in zip(p,shape))

fshape=[fine['width'],fine['height'],fine['slices']]
cshape=[coarse['width'],coarse['height'],coarse['slices']]
summary={
    'scope':'Metadata and sparse geometry only; no CT intensities or photographic/label pixels',
    'plan_sha256':hash_file(ROOT/'preflight_plan.json'),
    'preflight_runner_sha256':hash_file(ROOT/'header_geometry_preflight.py'),
    'headers_sha256':hash_file(ROOT/'header_results.json'),
    'geometry_records_sha256':hash_file(ROOT/'geometry_results.json'),
    'range_receipt_sha256':hash_file(ROOT/'range_receipt.json'),
    'response_body_bytes':r['actual_response_body_bytes_read'],
    'response_body_cap':r['response_body_byte_cap'],
    'requests_attempted':len(r['requests']),
    'successful_exact_206_requests':sum(x.get('success',False) and x.get('status')==206 for x in r['requests']),
    'no_unexpected_200_body_read':not any(x.get('status')==200 and x.get('body_bytes_read',0) for x in r['requests']),
    'objects':r['objects'],
    'surface_shape_yx':[int(h['ppm']['fields']['height']),int(h['ppm']['fields']['width'])],
    'ppm_data_offset_bytes':h['ppm']['data_offset_bytes'],
    'ppm_record_stride_bytes':48,
    'ppm_index_formula':'73 + 48 * (v * 4560 + u), fields X,Y,Z,NX,NY,NZ; no source-volume ID or endian field in header',
    'source_identity_status':'Directory-name/shape compatibility only; no embedded fine-volume UUID or render invocation was found in acquired headers',
    'fine_source_shape_xyz':fshape,'coarse_target_shape_xyz':cshape,
    'affine_metadata_sha256':hash_file(OLD/'affine.json'),
    'geometry_sample_count':len(g['records']), 'byte_order_compatibility':{},
    'conditional_bounds':[],
}

for endian in ('little','big'):
    rows=[x[endian] for x in g['records']]
    finite=[x for x in rows if x['finite']]
    unit=[x for x in finite if x['normal_magnitude'] is not None and abs(x['normal_magnitude']-1)<1e-6]
    zero=[x for x in finite if all(v==0 for v in x['values_xyz_nxyz'])]
    plausible=[x for x in unit if in_bounds(x['values_xyz_nxyz'][:3],fshape)]
    summary['byte_order_compatibility'][endian]={
        'finite_records':len(finite),'all_six_values_zero':len(zero),
        'unit_normal_within_1e_minus6':len(unit),'unit_normal_and_fine_center_in_bounds':len(plausible)}

nonzero=[]
for record in g['records']:
    values=record['little']['values_xyz_nxyz']
    norm=record['little']['normal_magnitude']
    if not record['little']['finite'] or norm is None or norm==0:
        continue
    p,n=values[:3],values[3:]
    # This is explicitly a conditional 65-plane p+n*d support diagnostic.
    # It does not establish that Frag5 was rendered with radius32/interval1.
    endpoints=[[p[j]+d*n[j] for j in range(3)] for d in (-32,32)]
    mapped=[matrix(q) for q in endpoints]
    row={'id':record['id'],'surface_yx':record['surface_yx'],
        'xyz_fine_candidate':p,'stored_normal_xyz':n,'normal_magnitude':norm,
        'fine_center_in_bounds':in_bounds(p,fshape),
        'mapped_coarse_center_xyz':matrix(p),'mapped_coarse_center_in_bounds':in_bounds(matrix(p),cshape),
        'conditional_fine_complete_trilinear_support_radius32':all(in_bounds(q,fshape,True) for q in endpoints),
        'conditional_mapped_ray_complete_coarse_trilinear_support_radius32':all(in_bounds(q,cshape,True) for q in mapped)}
    summary['conditional_bounds'].append(row)
    nonzero.append(values)

summary['little_endian_nonzero_geometry_summary']={
    'count':len(nonzero),
    'xyz_min':[min(row[j] for row in nonzero) for j in range(3)] if nonzero else None,
    'xyz_max':[max(row[j] for row in nonzero) for j in range(3)] if nonzero else None,
    'normal_magnitude_min':min((x['normal_magnitude'] for x in summary['conditional_bounds']),default=None),
    'normal_magnitude_max':max((x['normal_magnitude'] for x in summary['conditional_bounds']),default=None),
    'fine_centers_in_bounds':sum(x['fine_center_in_bounds'] for x in summary['conditional_bounds']),
    'mapped_coarse_centers_in_bounds':sum(x['mapped_coarse_center_in_bounds'] for x in summary['conditional_bounds']),
    'conditional_fine_complete_radius32':sum(x['conditional_fine_complete_trilinear_support_radius32'] for x in summary['conditional_bounds']),
    'conditional_mapped_ray_complete_coarse_radius32':sum(x['conditional_mapped_ray_complete_coarse_trilinear_support_radius32'] for x in summary['conditional_bounds']),
}
summary['limitations']=[
    'Sparse predeclared records are not a tissue/mask coverage estimate and do not certify any complete rectangular region.',
    'Little-endian plausibility does not make the host-native PPM format universally little-endian.',
    'Affine application is conditional on the unproven exact PPM source-volume identity; no independent registration validation is claimed.',
    'Radius32/interval1 bounds are a conditional geometry check, not recovered rendering-command provenance.',
    'Stored normals are used without normalization in the conditional original-ray support check.',
    'No photographic ink-label polarity, pixel values, manual annotation accuracy, physical face, scan radiometry or CT intensity comparison was inspected.',
]
(ROOT/'preflight_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:summary[key] for key in ['response_body_bytes','successful_exact_206_requests','byte_order_compatibility','little_endian_nonzero_geometry_summary']},indent=2))
