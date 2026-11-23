"""Acquire exactly the graph data used by the frozen Q12 diagnostic.

Public packed track stores and ZIP_STORED NumPy members are read with verified
HTTP byte ranges. Longitudinal *coordinates* are not used by the walker and are
not downloaded; their original IDs, lengths, and crossing indices are retained.
The complete adjacency rows of each retained node are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Ranges:
    def __init__(self, directory, limits):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.limits = limits
        self.rows = []
        self.etags = {}
        self.reserved = 0
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def get(self, url, lo, hi):
        """Read one half-open range; never silently accept a whole object."""
        if not (0 <= lo < hi):
            raise ValueError('Invalid byte range')
        with self.lock:
            if self.reserved + hi - lo > self.limits['maximum_source_body_bytes']:
                raise RuntimeError('Frozen download byte budget exceeded')
            if len(self.rows) >= self.limits['maximum_requests']:
                raise RuntimeError('Frozen request budget exceeded')
            if time.monotonic() - self.started > self.limits['maximum_acquisition_seconds']:
                raise RuntimeError('Frozen acquisition time budget exceeded')
            self.reserved += hi - lo
            record = {'url': url, 'range_halfopen': [lo, hi], 'status': 'started'}
            self.rows.append(record)
            previous = self.etags.get(url)
        headers = {'Range': f'bytes={lo}-{hi-1}', 'Accept-Encoding': 'identity'}
        if previous:
            headers['If-Match'] = previous
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                cr = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', cr)
                if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (lo, hi-1):
                    raise ValueError(f'Incorrect range response: {response.status} {cr}')
                if response.headers.get('Content-Encoding', 'identity') not in ('', 'identity'):
                    raise ValueError('Transformed range response')
                data = response.read(hi-lo+1)
                if len(data) != hi-lo:
                    raise ValueError('Incorrect response length')
                etag = response.headers.get('ETag')
                if not etag:
                    raise ValueError('Source lacks a stable ETag')
                with self.lock:
                    if url in self.etags and self.etags[url] != etag:
                        raise ValueError('Source object changed during acquisition')
                    self.etags[url] = etag
                record.update(status='complete', bytes=len(data), sha256=digest(data),
                              etag=etag, object_bytes=int(match.group(3)),
                              last_modified=response.headers.get('Last-Modified'))
                name = digest((url + f'#{lo}:{hi}').encode())
                (self.directory/name).write_bytes(data)
                record['cache_file'] = name
                return data
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise

    def write_receipt(self, path):
        obj = {'ranges': self.rows, 'source_body_bytes': sum(x.get('bytes', 0) for x in self.rows),
               'request_count': len(self.rows), 'seconds': time.monotonic()-self.started,
               'limits': self.limits}
        Path(path).write_text(json.dumps(obj, indent=2)+'\n')
        return obj


class RemoteZip(io.RawIOBase):
    """Minimal seekable adapter for Python's independently parsed ZIP directory."""
    def __init__(self, ranges, url, size):
        self.ranges, self.url, self.size, self.position = ranges, url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if pos < 0:
            raise ValueError('Negative seek')
        self.position = pos
        return pos

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position+size, self.size)
        if end <= self.position:
            return b''
        value = self.ranges.get(self.url, self.position, end)
        self.position = end
        return value


class Array:
    def __init__(self, ranges, url, offset, dtype, shape):
        self.ranges, self.url, self.offset = ranges, url, int(offset)
        self.dtype, self.shape = np.dtype(dtype), tuple(shape)
        self.row_bytes = self.dtype.itemsize * int(np.prod(self.shape[1:] or (1,)))
        if self.dtype.hasobject:
            raise ValueError('Object array forbidden')

    def block(self, start, stop):
        if not 0 <= start <= stop <= self.shape[0]:
            raise ValueError('Array read outside source')
        if stop == start:
            return np.empty((0, *self.shape[1:]), self.dtype)
        data = self.ranges.get(self.url, self.offset+start*self.row_bytes,
                               self.offset+stop*self.row_bytes)
        return np.frombuffer(data, self.dtype).reshape((stop-start, *self.shape[1:])).copy()

    def all(self):
        return self.block(0, self.shape[0])

    def rows(self, indices, gap=262144):
        indices = np.asarray(indices, np.int64)
        if not len(indices):
            return np.empty((0, *self.shape[1:]), self.dtype)
        ordered = np.unique(indices)
        intervals = coalesce([(int(x), int(x)+1) for x in ordered], gap//self.row_bytes)
        def one(interval):
            lo, hi = interval
            data = self.block(lo, hi)
            a, b = np.searchsorted(ordered, [lo, hi])
            return a, b, data[ordered[a:b]-lo]
        gathered = np.empty((len(ordered), *self.shape[1:]), self.dtype)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for a, b, value in pool.map(one, intervals):
                gathered[a:b] = value
        return gathered[np.searchsorted(ordered, indices)]


def coalesce(intervals, gap):
    out = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]+gap:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([lo, hi])
    return out


def zip_arrays(ranges, url, size):
    with zipfile.ZipFile(RemoteZip(ranges, url, size)) as archive:
        infos = archive.infolist()
    arrays = {}
    for item in infos:
        if item.compress_type != zipfile.ZIP_STORED or item.file_size != item.compress_size:
            raise ValueError('Only uncompressed ZIP_STORED members are supported')
        header = ranges.get(url, item.header_offset, item.header_offset+30)
        fields = struct.unpack('<4s5H3L2H', header)
        if fields[0] != b'PK\x03\x04':
            raise ValueError('Bad ZIP local header')
        payload = item.header_offset+30+fields[-2]+fields[-1]
        npy_header = ranges.get(url, payload, payload+min(item.file_size, 512))
        stream = io.BytesIO(npy_header)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported npy version {version}')
        if fortran or dtype.hasobject:
            raise ValueError('Unsupported NumPy layout')
        data_start = payload+stream.tell()
        if stream.tell()+int(np.prod(shape))*dtype.itemsize != item.file_size:
            raise ValueError('ZIP member size does not match array shape')
        # Metadata is a zero-dimensional unicode array.
        arrays[item.filename.removesuffix('.npy')] = Array(ranges, url, data_start, dtype, shape or (1,))
    return arrays


def read_adjacency(arrays, ids, offsets, gap):
    intervals = coalesce([(int(offsets[t]), int(offsets[t+1])) for t in ids], gap//8)
    starts, stops = offsets[ids], offsets[ids+1]
    result = {}
    def one(interval):
        lo, hi = interval
        return lo, hi, tuple(arrays[n].block(lo, hi) for n in ('partners','self_local','partner_local'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for lo, hi, values in pool.map(one, intervals):
            begin, end = np.searchsorted(starts, [lo, hi])
            for t, a, b in zip(ids[begin:end], starts[begin:end], stops[begin:end]):
                if b > hi:
                    raise ValueError('Node adjacency crosses its fetched interval')
                if b > a:
                    result[t] = tuple(v[a-lo:b-lo].copy() for v in values)
    for t, a, b in zip(ids, starts, stops):
        if a == b:
            result[t] = (np.empty(0, np.int64),)*3
    if len(result) != len(ids):
        raise ValueError('Incomplete adjacency extraction')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(HERE.parents[1]):
        raise ValueError('Raw sources must be outside the git worktree')
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE/'protocol.json').read_text())
    (out/'protocol.json').write_bytes((HERE/'protocol.json').read_bytes())
    ranges = Ranges(out/'range-cache', protocol['budgets'])
    root = protocol['source_root']
    store, archive = root+'.vctracks/', root+'.crossings.npz'
    result = {'status': 'started', 'protocol_sha256': digest((HERE/'protocol.json').read_bytes())}
    try:
        # Exact sizes are metadata inspected before the frozen coordinate experiment.
        rawmeta = ranges.get(store+'metadata.json', 0, 220)
        meta = json.loads(rawmeta)
        ntracks, npoints = meta['track_count'], meta['point_count']
        arrays = zip_arrays(ranges, archive, 2619127900)
        crossmeta = json.loads(str(arrays['metadata'].all()[0]))
        if meta['source_db_signature'] != crossmeta['db_signature']:
            raise ValueError('Track/crossing producer signatures differ')
        for name in ('source_ids','offsets','partners','self_local','partner_local'):
            if len(arrays[name].shape) != 1 or arrays[name].dtype.kind not in 'iu':
                raise ValueError('Crossing indices must be one-dimensional integer arrays')
        if arrays['source_ids'].shape != (ntracks,):
            raise ValueError('Crossing source ID count differs from track store')
        famarray = Array(ranges, store+'family_codes.i8', 0, 'i1', (ntracks,))
        zbarray = Array(ranges, store+'z_bounds.i32', 0, '<i4', (ntracks, 2))
        offarray = Array(ranges, store+'offsets.i64', 0, '<i8', (ntracks+1,))
        with ThreadPoolExecutor(max_workers=3) as pool:
            fam, zb, off = list(pool.map(lambda a: a.all(), (famarray, zbarray, offarray)))
        if off[0] != 0 or off[-1] != npoints or np.any(np.diff(off) < 0):
            raise ValueError('Invalid original point offsets')
        selection = protocol['selection']
        zlo, zhi = selection['z0']-selection['dz'], selection['z0']+selection['dz']
        ids = np.flatnonzero((fam == 0) & (zb[:,0] >= zlo) & (zb[:,1] <= zhi))
        print(json.dumps({'stage':'index', 'transverse_tracks':len(ids), 'points':int(np.diff(off)[ids].sum())}), flush=True)
        coordinate_array = Array(ranges, store+'coordinates.i32', 0, '<i4', (npoints,3))
        cintervals = coalesce([(int(off[i]),int(off[i+1])) for i in ids], 0)
        coords = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for (lo,hi), values in zip(cintervals, pool.map(lambda p: coordinate_array.block(*p), cintervals)):
                begin, end = np.searchsorted(off[ids], [lo, hi])
                for i in ids[begin:end]:
                    if off[i+1] > hi:
                        raise ValueError('Track crosses its fetched coordinate interval')
                    coords[int(i)] = values[off[i]-lo:off[i+1]-lo].copy()
        if len(coords) != len(ids) or any(not np.all(v[:,0] == v[0,0]) for v in coords.values()):
            raise ValueError('Incomplete or nonconstant-z transverse coordinates')
        coff = arrays['offsets'].all()
        if len(coff) != ntracks+1 or coff[0] != 0 or np.any(np.diff(coff)<0):
            raise ValueError('Invalid crossing offsets')
        if any(arrays[name].shape != (int(coff[-1]),) for name in ('partners','self_local','partner_local')):
            raise ValueError('Crossing index arrays have inconsistent lengths')
        hcross = read_adjacency(arrays, ids, coff, 262144)
        lengths = np.diff(off)
        def validate_crossings(crossings):
            for node, (partners, own, other) in crossings.items():
                if (np.any(partners < 0) or np.any(partners >= ntracks)
                    or np.any(own < 0) or np.any(own >= lengths[node])):
                    raise ValueError('Crossing node or local index outside original track')
                if np.any(other < 0) or np.any(other >= lengths[partners]):
                    raise ValueError('Partner-local index outside original track')
        validate_crossings(hcross)
        linked = np.unique(np.concatenate([v[0][fam[v[0]]==1] for v in hcross.values()]))
        print(json.dumps({'stage':'transverse', 'longitudinal_neighbors':len(linked)}), flush=True)
        vcross = read_adjacency(arrays, linked, coff, 262144)
        validate_crossings(vcross)
        allids = np.unique(np.r_[ids, linked])
        original_sourceids = Array(ranges, store+'source_ids.u64', 0, '<u8', (ntracks,)).rows(allids)
        crossing_sourceids = arrays['source_ids'].rows(allids)
        if not np.array_equal(original_sourceids, crossing_sourceids):
            raise ValueError('Source IDs are not aligned across track store and crossing archive')
        cross = hcross | vcross
        # Save a compact ragged representation. Global IDs remain unchanged.
        coordinate_offsets = np.r_[0, np.cumsum([len(coords[int(i)]) for i in ids])]
        np.save(out/'transverse_ids.npy', ids, allow_pickle=False)
        np.save(out/'coordinates.npy', np.concatenate([coords[int(i)] for i in ids]), allow_pickle=False)
        np.save(out/'coordinate_offsets.npy', coordinate_offsets, allow_pickle=False)
        np.save(out/'family.npy', fam, allow_pickle=False)
        np.save(out/'lengths.npy', lengths, allow_pickle=False)
        np.save(out/'node_ids.npy', allids, allow_pickle=False)
        np.save(out/'cross_offsets.npy', np.r_[0,np.cumsum([len(cross[int(i)][0]) for i in allids])], allow_pickle=False)
        for j, name in enumerate(('partners','self_local','partner_local')):
            np.save(out/(name+'.npy'), np.concatenate([cross[int(i)][j] for i in allids]), allow_pickle=False)
        files = {}
        for path in sorted(out.glob('*.npy')):
            with path.open('rb') as stream:
                files[path.name] = {'bytes':path.stat().st_size, 'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
        result.update(status='complete', store_metadata=meta, crossing_metadata=crossmeta,
                      transverse_tracks=len(ids), transverse_points=int(coordinate_offsets[-1]),
                      linked_longitudinal_tracks=len(linked), source_id_alignment_verified=len(allids),
                      retained_crossing_rows=sum(len(v[0]) for v in cross.values()), files=files,
                      constant_z_verified=True, original_graph_indices_preserved=True)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt = ranges.write_receipt(out/'range_receipt.json')
        result.update(source_body_bytes=receipt['source_body_bytes'], request_count=receipt['request_count'],
                      seconds=receipt['seconds'])
        (out/'acquisition.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
