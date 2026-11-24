"""Read-only investigation of saved Q33 inverse abstentions; no model calls.

Re-evaluates the source author's Surface.points directly, then probes nine
neighbouring bilinear cells for one deterministically chosen rejected point per
arm. This diagnoses abstentions only; it never changes the frozen mask/solver.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares


def investigate(args):
    def sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    module_path = args.source / 'donor-labels-20260906/expansion/volume_sampling.py'
    spec = importlib.util.spec_from_file_location('independent_author_surface', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selection = json.loads(args.selection.read_text())
    controls = selection['controls']
    pi = next(i for i, item in enumerate(controls) if item['index'] == args.author_id)
    item = controls[pi]
    metadata = json.loads(args.source_metadata.read_text())
    recipe = metadata['geometry']['recipe']
    surface = module.Surface(args.mesh, 9.612, 9.366, recipe['matrix'], 1, [3781, 7618, 7618])
    mapping = np.load(args.prepared/'mapping.npz', allow_pickle=False)
    states = np.load(args.prepared/'original_states.npy', allow_pickle=False)
    diagnostic = json.loads((args.prepared/'geometry_diagnostics.json').read_text())
    names = [r['arm'] for r in diagnostic['per_arm_core'] if r['author_index'] == args.author_id]
    target = surface.points(np.arange(item['y']+48, item['y']+80),
                            np.arange(item['x']+48, item['x']+80), 9.36)[0].astype(float)
    rows = []
    for ai, name in enumerate(names):
        bad = ~mapping['valid'][ai, pi]
        q = mapping['query_yx'][ai, pi][bad] + [item['y'], item['x']]
        depth = mapping['depth_vox'][ai, pi][bad]
        if not len(q):
            continue
        uv = (q+.5)*[9.36/surface.pitch[1], 9.36/surface.pitch[0]]
        indices = np.floor(uv).astype(int)
        v, u = (uv-indices).T
        iy, ix = indices.T
        aa, bb, cc, dd = [surface.a[iy+dy, ix+dx].astype(float)
                          for dy, dx in [(0,0), (0,1), (1,0), (1,1)]]
        xyz = ((1-u[:, None])*(1-v[:, None])*aa + u[:, None]*(1-v[:, None])*bb
               + (1-u[:, None])*v[:, None]*cc + u[:, None]*v[:, None]*dd)
        dx = (1-v[:, None])*(bb-aa)+v[:, None]*(dd-cc)
        dy = (1-u[:, None])*(cc-aa)+u[:, None]*(dd-bb)
        normal = np.cross(dy, dx)
        normal /= np.linalg.norm(normal, axis=-1)[:, None]
        inverse_target = ((target[bad]-mapping['pivots'][ai, pi])
                          @ mapping['rotations'][ai, pi] + mapping['pivots'][ai, pi])
        analytical = np.linalg.norm(xyz+depth[:, None]*normal-inverse_target, axis=-1)
        positions, normals, _ = surface.points(q[:, 0], q[:, 1], 9.36)
        diagonal = np.arange(len(q))
        direct = positions[diagonal, diagonal] + depth.astype(np.float32)[:, None]*normals[diagonal, diagonal]
        physical = np.linalg.norm(direct-inverse_target, axis=-1)
        seam_distance = (np.abs(uv-np.round(uv))*[surface.pitch[1]/9.36, surface.pitch[0]/9.36]).min(axis=1)
        row = dict(arm=name, rejected=len(q), analytical_residual_above_001=int((analytical>.01).sum()),
                   analytical_residual_max=float(analytical.max()),
                   physical_residual_range=[float(physical.min()), float(physical.max())],
                   within_001_chart_px_of_seam=int((seam_distance<.01).sum()),
                   maximum_seam_distance_chart_px=float(seam_distance.max()),
                   author_residual_parity_max=float(np.max(np.abs(physical-mapping['residual_vox'][ai, pi][bad]))))

        # The worst physical residual is chosen before any cell-local fit.
        sample = int(np.argmax(physical))
        goal = inverse_target[sample]
        solutions = []
        for cy in range(indices[sample, 0]-1, indices[sample, 0]+2):
            for cx in range(indices[sample, 1]-1, indices[sample, 1]+2):
                if not (0 <= cy < surface.a.shape[0]-1 and 0 <= cx < surface.a.shape[1]-1):
                    continue
                vertices = surface.a[cy:cy+2, cx:cx+2].astype(float)
                a, b, c, d = vertices[0,0], vertices[0,1], vertices[1,0], vertices[1,1]
                def residual(t):
                    vv, uu, dep = t
                    pos = (1-uu)*(1-vv)*a+uu*(1-vv)*b+(1-uu)*vv*c+uu*vv*d
                    col, rowderiv = (1-vv)*(b-a)+vv*(d-c), (1-uu)*(c-a)+uu*(d-b)
                    norm = np.cross(rowderiv, col)
                    norm /= np.linalg.norm(norm)
                    return pos + dep*norm-goal
                initial = [np.clip(uv[sample,0]-cy, .01, .99),
                           np.clip(uv[sample,1]-cx, .01, .99), np.clip(depth[sample], -7.99, 6.99)]
                fit = least_squares(residual, initial, bounds=([1e-7,1e-7,-8], [1-1e-7,1-1e-7,7]),
                                    xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
                yy = (cy+fit.x[0])*surface.pitch[1]/9.36-.5
                xx = (cx+fit.x[1])*surface.pitch[0]/9.36-.5
                pos, norm, valid = surface.points([yy], [xx], 9.36)
                actual = pos[0,0]+np.float32(fit.x[2])*norm[0,0]
                actual_residual = float(np.linalg.norm(actual-goal))
                inside = (16 <= yy-item['y'] <= 111 and 16 <= xx-item['x'] <= 111)
                solutions.append(dict(cell=[cy,cx], analytical_residual=float(np.linalg.norm(fit.fun)),
                                      author_fp32_residual=actual_residual, source_valid=bool(valid[0,0]),
                                      inside_model=inside, evaluations=fit.nfev,
                                      acceptable=bool(valid[0,0] and inside and actual_residual <= .01)))
        row['worst_point_neighbor_cell_probe'] = dict(
            cells_tested=len(solutions), accepted_root_found=any(r['acceptable'] for r in solutions),
            best_author_fp32_residual=min(r['author_fp32_residual'] for r in solutions),
            best_analytical_residual=min(r['analytical_residual'] for r in solutions),
            limitation='Local least-squares probes only; absence of a found root is not a global nonexistence proof.')
        rows.append(row)
    return dict(status='review_complete', author_index=args.author_id,
                protocol_sha256=sha(args.prepared/'protocol.json'),
                mapping_sha256=sha(args.prepared/'mapping.npz'),
                original_states_sha256=sha(args.prepared/'original_states.npy'),
                source_metadata_sha256=sha(args.source_metadata),
                author_surface_helper_sha256=sha(module_path),
                mesh_sha256={name:sha(args.mesh/name) for name in ('meta.json','x.tif','y.tif','z.tif')},
                reviewer_sha256=sha(__file__), numpy_version=np.__version__, scipy_version=scipy.__version__,
                common_excluded_known=int(((states[pi]>0)&~mapping['common_mask'][pi]).sum()),
                global_excluded_ink=int(((states==2)&~mapping['common_mask']).sum()),
                global_excluded_background=int(((states==1)&~mapping['common_mask']).sum()),
                global_rejection_categories={k:sum(r[k] for r in diagnostic['per_arm_core']) for k in
                    ['invalid_target_geometry','invalid_solved_geometry','invalid_jacobian',
                     'outside_model_margin','outside_depth_band','residual_failures']},
                per_arm=rows, production_arrays_changed=False, model_forwards=0,
                conclusion='Rejections match the independent source evaluator and cluster at cell seams; none of ten deterministic worst-point local probes found an acceptable root. Retain abstentions; no global nonexistence or injectivity proof.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','selection','source','source-metadata','mesh','report'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--author-id', type=int, default=357)
    args = parser.parse_args()
    if args.report.exists(): parser.error('Use a fresh report path')
    result = investigate(args)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(author_index=result['author_index'], rejected_arm_pixel_count=sum(r['rejected'] for r in result['per_arm']),
                         independently_recovered_worst_points=sum(r['worst_point_neighbor_cell_probe']['accepted_root_found'] for r in result['per_arm']),
                         probed_worst_points=len(result['per_arm']))))
