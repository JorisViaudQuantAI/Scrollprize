"""Paired legacy/corrected replay on one frozen public winding graph.

No acquisition, trained model, text search, threshold tuning, or anatomical
winding-sense verdict is performed here. Storage transformations preserve every
source point and remap both sides of each crossing's local indices.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import signal
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Graph:
    def __init__(self, directory, protocol):
        self.directory = directory
        receipt = json.loads((directory/'acquisition.json').read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Source acquisition is incomplete')
        if receipt['protocol_sha256'] != sha(HERE/'protocol.json'):
            raise ValueError('Acquisition used a different protocol')
        for name, record in receipt['files'].items():
            path = directory/name
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('Source artifact changed: '+name)
        def load(name):
            return np.load(directory/(name+'.npy'), mmap_mode='r', allow_pickle=False)
        self.hids, self.coords, self.offsets = (load(x) for x in ('transverse_ids','coordinates','coordinate_offsets'))
        self.fam, self.lengths, self.nodes, self.coff = (load(x) for x in ('family','lengths','node_ids','cross_offsets'))
        self.parts, self.local, self.other = (load(x) for x in ('partners','self_local','partner_local'))
        self.hlookup = {int(t):i for i,t in enumerate(self.hids)}
        self.nlookup = {int(t):i for i,t in enumerate(self.nodes)}
        self.random = np.random.default_rng(protocol['random_half_seed']).integers(0,2,len(self.fam),dtype=np.uint8).astype(bool)

    def is_reversed(self, ids, condition):
        if condition == 'reverse_transverse':
            return self.fam[ids] == 0
        if condition == 'reverse_all':
            return np.ones(np.shape(ids), dtype=bool)
        if condition == 'random_half':
            return self.random[ids]
        return np.zeros(np.shape(ids), dtype=bool)

    def coordinates(self, node, condition):
        if node not in self.hlookup:
            raise ValueError('Longitudinal or unselected coordinates requested')
        i = self.hlookup[node]
        result = self.coords[self.offsets[i]:self.offsets[i+1]]
        return result[::-1] if self.is_reversed(node, condition) else result

    def crossings(self, node, condition):
        if node not in self.nlookup:
            raise ValueError('Unacquired adjacency requested')
        i = self.nlookup[node]
        a,b = self.coff[i:i+2]
        p,s,o = self.parts[a:b], self.local[a:b], self.other[a:b]
        s = self.lengths[node]-1-s if self.is_reversed(node,condition) else s
        o = np.where(self.is_reversed(p,condition), self.lengths[p]-1-o, o)
        if condition == 'reverse_crossing_rows':
            return p[::-1],s[::-1],o[::-1]
        return p,s,o

    def family(self, node):
        if not 0 <= node < len(self.fam):
            raise ValueError('Track ID out of bounds')
        return int(self.fam[node])

    def length(self, node):
        if not 0 <= node < len(self.lengths):
            raise ValueError('Track ID out of bounds')
        return int(self.lengths[node])


def diagnostics(graph, center):
    increasing = decreasing = zero = nonmonotone = eligible = 0
    zvalues = set()
    for node in graph.hids:
        a = graph.coordinates(int(node),'original')
        zvalues.add(int(a[0,0]))
        theta = np.unwrap(np.arctan2(a[:,1]-center[0],a[:,2]-center[1]))
        delta = theta[-1]-theta[0]
        increasing += int(delta > 0)
        decreasing += int(delta < 0)
        zero += int(delta == 0)
        sign = 1 if delta >= 0 else -1
        nonmonotone += int(np.any(sign*np.diff(theta) < -1e-12))
        eligible += int(len(a) >= 40)
    return {'transverse_tracks':len(graph.hids), 'start_eligible_tracks':eligible,
            'point_count':len(graph.coords), 'distinct_slices':len(zvalues),
            'increasing_angle_storage':increasing, 'decreasing_angle_storage':decreasing,
            'zero_endpoint_angle_span':zero, 'tracks_with_local_angular_backtracking':nonmonotone,
            'backtracking_interpretation':'Any signed angular decrement >1e-12, including integer-raster microsteps; not a physical fold count.',
            'origin_is_validated_umbilicus':False}


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args = ap.parse_args()
    out, source = args.output_dir.resolve(), args.source_dir.resolve()
    if out.is_relative_to(HERE.parents[1]) or out == source or out.is_relative_to(source):
        raise ValueError('Use a fresh output directory outside git and source acquisition')
    out.mkdir(parents=True,exist_ok=False)
    protocol = json.loads((HERE/'protocol.json').read_text())
    started = time.monotonic()
    def timeout(signum,frame):
        raise TimeoutError('Frozen analysis time budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(protocol['budgets']['maximum_analysis_seconds'])
    result = {'status':'started','protocol_sha256':sha(HERE/'protocol.json'),
              'runner_sha256':sha(__file__),'source_receipt_sha256':sha(source/'acquisition.json'),
              'analysis':{},'physical_winding_sense_verdict':None}
    all_summaries, all_rows = {}, {}
    try:
        graph = Graph(source,protocol)
        result['geometry_diagnostics'] = diagnostics(graph,protocol['selection']['center_yx'])
        fixed_path = HERE.parent/'methods_ledger/code/sense_chain.py'
        legacy_path = HERE/'legacy_adapter.py'
        functions = {'legacy':module(legacy_path,'q12_legacy').trace_legacy,
                     'corrected_v2':module(fixed_path,'q12_fixed').trace_winding_sense}
        result['implementation_sha256'] = {str(p.name):sha(p) for p in (fixed_path,legacy_path,HERE/'legacy_snapshot.py')}
        sel, traversal = protocol['selection'], protocol['traversal']
        cy,cx = sel['center_yx']
        points = [dict(z=sel['z0'],y=cy,x=cx)]
        kwargs = dict(z0=sel['z0'],dz=sel['dz'],nstart=sel['nstart'],seed=sel['start_seed'],
                      rmax=sel['rmax'],**traversal)
        for algorithm, func in functions.items():
            result['analysis'][algorithm] = {}
            for condition in protocol['conditions']:
                tick = time.monotonic()
                summary, rows = func(graph.hids,
                    lambda t:graph.coordinates(t,condition),lambda t:graph.crossings(t,condition),
                    graph.family,graph.length,points,mirror=-1 if condition == 'mirror_x' else 1,**kwargs)
                key = (algorithm,condition)
                all_summaries[key], all_rows[key] = summary,rows
                prefix = algorithm+'_'+condition
                np.save(out/(prefix+'_rows.npy'),rows,allow_pickle=False)
                ids = summary['completed_start_ids']
                np.save(out/(prefix+'_start_ids.npy'),np.asarray(ids,dtype=np.int64),allow_pickle=False)
                public = {k:v for k,v in summary.items() if k != 'completed_start_ids'}
                public.update(seconds=time.monotonic()-tick,rows_sha256=sha(out/(prefix+'_rows.npy')),
                              completed_start_ids_sha256=sha(out/(prefix+'_start_ids.npy')))
                result['analysis'][algorithm][condition] = public
                write(out/'result.json',result)
                print(json.dumps({'algorithm':algorithm,'condition':condition,
                                  'chains':summary['chains_full_turn'],'seconds':public['seconds']}),flush=True)
        transforms = protocol['conditions'][1:-1]
        gates = {}
        for algorithm in functions:
            base_summary = all_summaries[(algorithm,'original')]
            base_rows = all_rows[(algorithm,'original')]
            gates[algorithm] = {condition:{
                'complete_summary_equal':all_summaries[(algorithm,condition)] == base_summary,
                'ordered_rows_bitwise_equal':np.array_equal(all_rows[(algorithm,condition)],base_rows),
                'completed_start_ids_equal':all_summaries[(algorithm,condition)]['completed_start_ids'] == base_summary['completed_start_ids']}
                for condition in transforms}
        result['representation_invariance'] = gates
        result['fixed_storage_invariance_pass'] = all(all(v.values()) for v in gates['corrected_v2'].values())
        result['legacy_storage_defect_reproduced'] = any(not all(v.values()) for v in gates['legacy'].values())
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0)
        result.update(seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      runtime={'python':platform.python_version(),'numpy':np.__version__})
        write(out/'result.json',result)


if __name__ == '__main__':
    main()
