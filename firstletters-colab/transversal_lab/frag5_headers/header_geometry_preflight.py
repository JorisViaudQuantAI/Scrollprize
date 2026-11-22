#!/usr/bin/env python3
"""Bounded public Frag5 metadata/header/sparse-PPM access. No image decoding."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import zlib
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'preflight_plan.json').read_text())
BASE = PLAN['surface_url']

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def sha(data):
    return hashlib.sha256(data).hexdigest()

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class Ranges:
    def __init__(self):
        self.path = ROOT / 'range_receipt.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'plan_sha256': sha((ROOT/'preflight_plan.json').read_bytes()),
            'response_body_byte_cap': PLAN['response_body_byte_cap'],
            'actual_response_body_bytes_read': 0, 'requests': [], 'objects': {}}
        if self.state['plan_sha256'] != sha((ROOT/'preflight_plan.json').read_bytes()):
            raise ValueError('The declared plan changed')
        self.opener = build_opener(NoRedirect())
        (ROOT/'ranges').mkdir(exist_ok=True)

    def get(self, url, start, count, purpose):
        if not url.startswith(BASE) or '?' in url or '#' in url:
            raise ValueError('Only the fixed public surface bundle is permitted')
        if start < 0 or not 1 <= count <= PLAN['maximum_single_range_bytes']:
            raise ValueError('Range lies outside the per-request bound')
        end = start + count - 1
        name = sha(f'{url}\n{start}\n{end}'.encode()) + '.bin'
        path = ROOT/'ranges'/name
        old = [r for r in self.state['requests'] if r.get('cache_file') == name and r.get('success')]
        if old:
            data = path.read_bytes()
            if len(data) != count or sha(data) != old[-1]['body_sha256']:
                raise ValueError('Cached acquired range changed')
            return data
        if self.state['actual_response_body_bytes_read'] + count > PLAN['response_body_byte_cap']:
            raise ValueError('Hard response-body budget would be exceeded')
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                   'User-Agent': 'Vesuvius-public-header-preflight/1'}
        obj = self.state['objects'].get(url)
        if obj and obj.get('etag'):
            headers['If-Match'] = obj['etag']
        row = {'canonical_url': url, 'method': 'GET', 'requested_range': [start,end],
               'purpose': purpose, 'cache_file': name, 'body_bytes_read': 0,
               'success': False, 'utc_unix_seconds': time.time()}
        self.state['requests'].append(row)
        save(self.path, self.state)
        response = None
        data = bytearray()
        try:
            try:
                response = self.opener.open(Request(url, headers=headers), timeout=30)
            except HTTPError as error:
                row['status'] = error.code
                row['content_length'] = error.headers.get('Content-Length')
                error.close()
                raise ValueError(f'HTTP status {error.code}; body not read') from None
            row.update({'status': response.status, 'content_length': response.headers.get('Content-Length'),
                'content_range': response.headers.get('Content-Range'), 'etag': response.headers.get('ETag'),
                'last_modified': response.headers.get('Last-Modified'),
                'content_encoding': response.headers.get('Content-Encoding'),
                'content_type': response.headers.get('Content-Type')})
            if response.status != 206:
                raise ValueError('Expected HTTP 206; unexpected body is not read')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', row['content_range'] or '')
            if not match or tuple(map(int, match.groups()[:2])) != (start,end):
                raise ValueError('Content-Range does not equal the requested byte interval')
            total = int(match.group(3))
            if end >= total or int(row['content_length'] or -1) != count:
                raise ValueError('Invalid object length or response Content-Length')
            if row['content_encoding'] not in (None, '', 'identity'):
                raise ValueError('Encoded transfer cannot establish raw byte offsets')
            if obj and (obj['content_length'] != total or obj.get('etag') != row['etag']):
                raise ValueError('Mutable source identity changed between ranges')
            self.state['objects'][url] = {'content_length': total, 'etag': row['etag'],
                                          'last_modified': row['last_modified']}
            while len(data) < count:
                chunk = response.read(min(8192, count-len(data)))
                if not chunk:
                    raise ValueError('Truncated range body')
                data.extend(chunk)
                row['body_bytes_read'] += len(chunk)
                self.state['actual_response_body_bytes_read'] += len(chunk)
                if self.state['actual_response_body_bytes_read'] > PLAN['response_body_byte_cap']:
                    raise ValueError('Hard body budget exceeded')
            path.write_bytes(data)
            row['body_sha256'] = sha(data)
            row['success'] = True
            return bytes(data)
        except BaseException as error:
            row['error_type'] = type(error).__name__
            row['error_summary'] = str(error) if isinstance(error, ValueError) else 'Transport error; no retry'
            if data:
                path.write_bytes(data)
                row['partial_body_sha256'] = sha(data)
            raise
        finally:
            if response is not None:
                response.close()
            save(self.path, self.state)

TIFF_TYPES = {1:(1,'B'),2:(1,None),3:(2,'H'),4:(4,'I'),5:(8,'II'),
              6:(1,'b'),7:(1,'B'),8:(2,'h'),9:(4,'i'),10:(8,'ii'),
              11:(4,'f'),12:(8,'d'),13:(4,'I'),16:(8,'Q'),17:(8,'q'),18:(8,'Q')}
TAG_NAMES = {256:'ImageWidth',257:'ImageLength',258:'BitsPerSample',259:'Compression',
    262:'PhotometricInterpretation',269:'DocumentName',270:'ImageDescription',271:'Make',272:'Model',
    273:'StripOffsets',274:'Orientation',277:'SamplesPerPixel',278:'RowsPerStrip',279:'StripByteCounts',
    282:'XResolution',283:'YResolution',284:'PlanarConfiguration',296:'ResolutionUnit',
    305:'Software',306:'DateTime',315:'Artist',317:'Predictor',322:'TileWidth',323:'TileLength',
    324:'TileOffsets',325:'TileByteCounts',339:'SampleFormat',340:'SMinSampleValue',341:'SMaxSampleValue'}

def tiff_header(ranges, plane):
    url=BASE+f'surface_volume/{plane}.tif'
    header=ranges.get(url,0,8,'TIFF header only')
    if header[:2] not in (b'II',b'MM'):
        raise ValueError('Unknown TIFF byte order')
    endian='<' if header[:2]==b'II' else '>'
    version=struct.unpack(endian+'H',header[2:4])[0]
    if version==42:
        offset=struct.unpack(endian+'I',header[4:8])[0]
        nsize, entrysize, pointerwidth, nfmt = 2,12,4,'H'
    elif version==43:
        if struct.unpack(endian+'HH',header[4:8])!=(8,0):
            raise ValueError('Unsupported BigTIFF offset contract')
        offset=struct.unpack(endian+'Q',ranges.get(url,8,8,'BigTIFF IFD offset'))[0]
        nsize,entrysize,pointerwidth,nfmt=8,20,8,'Q'
    else:
        raise ValueError('Unsupported TIFF version')
    n=struct.unpack(endian+nfmt,ranges.get(url,offset,nsize,'TIFF IFD entry count'))[0]
    if not 1 <= n <= 256:
        raise ValueError('Unexpected TIFF metadata entry count')
    body=ranges.get(url,offset+nsize,n*entrysize+pointerwidth,'TIFF IFD entries and next pointer only')
    tags=[]
    for i in range(n):
        entry=body[i*entrysize:(i+1)*entrysize]
        if version==42:
            tag,typ,count=struct.unpack(endian+'HHI',entry[:8]); slot=entry[8:12]
        else:
            tag,typ,count=struct.unpack(endian+'HHQ',entry[:12]); slot=entry[12:20]
        row={'tag':tag,'name':TAG_NAMES.get(tag,f'tag_{tag}'),'type':typ,'count':count}
        if typ not in TIFF_TYPES:
            row['not_decoded']='unknown metadata type';tags.append(row);continue
        size,fmt=TIFF_TYPES[typ]
        length=size*count
        if length > PLAN['maximum_single_range_bytes']:
            raise ValueError('TIFF tag metadata exceeds bounded single-range limit')
        if length <= pointerwidth:
            payload=slot[:length];row['storage']='inline'
        else:
            pointer=struct.unpack(endian+('I' if pointerwidth==4 else 'Q'),slot)[0]
            row['storage']='offset';row['value_offset']=pointer
            payload=ranges.get(url,pointer,length,f'TIFF tag {tag} metadata, never pixel strip/tile payload')
        if typ==2:
            row['value']=payload.rstrip(b'\0').decode('utf-8',errors='replace')
        else:
            row['value']=list(struct.unpack(endian+fmt*count,payload))
        tags.append(row)
    nxt=struct.unpack(endian+('I' if pointerwidth==4 else 'Q'),body[-pointerwidth:])[0]
    return {'canonical_url':url,'byte_order':'little' if endian=='<' else 'big',
            'version':version,'first_ifd_offset':offset,'entry_count':n,'next_ifd_offset':nxt,'tags':tags}

def png_header(ranges, name):
    url=BASE+name
    b=ranges.get(url,0,33,'PNG signature/IHDR/CRC only; no image pixels')
    if b[:8]!=b'\x89PNG\r\n\x1a\n' or b[8:16]!=b'\0\0\0\rIHDR':
        raise ValueError('Unexpected PNG signature or IHDR')
    fields=struct.unpack('>IIBBBBB',b[16:29])
    expected=struct.unpack('>I',b[29:33])[0]
    if zlib.crc32(b[12:29]) & 0xffffffff != expected:
        raise ValueError('PNG IHDR CRC mismatch')
    return {'canonical_url':url,**dict(zip(['width','height','bit_depth','color_type','compression_method','filter_method','interlace_method'],fields)),
            'ihdr_crc_valid':True,'pixel_payload_decoded':False}

def ppm_header(ranges):
    url=BASE+'result.ppm';data=b''
    for start in range(0,4096,64):
        data+=ranges.get(url,start,64,'PPM ASCII header search only; incidental binary suffix not decoded')
        marker=re.search(rb'(?:^|\n)<>\r?\n',data)
        if marker:
            offset=marker.end();header=data[:offset]
            text=header.decode('ascii')
            fields={}
            for line in text.splitlines():
                if ':' in line:
                    key,value=line.split(':',1);fields[key.strip()]=value.strip()
            return {'canonical_url':url,'ascii_header':text,'header_sha256':sha(header),
                    'data_offset_bytes':offset,'fields':fields,
                    'incidental_binary_suffix_bytes_not_decoded':len(data)-offset}
    raise ValueError('PPM ASCII header terminator absent within declared bound')

def geometry(ranges, headers):
    h=headers['ppm'];fields=h['fields']
    width,height=int(fields['width']),int(fields['height'])
    if int(fields['dim'])!=6 or fields['type']!='double' or fields['ordered'].lower()!='true':
        raise ValueError('PPM is not the expected ordered six-double geometry map')
    offset=h['data_offset_bytes'];length=ranges.state['objects'][h['canonical_url']]['content_length']
    if offset+48*width*height!=length:
        raise ValueError('PPM body does not have exactly six 8-byte values per surface pixel')
    coords=[(k*(height-1)//10,j*(width-1)//10) for k in range(1,10) for j in range(1,10)]
    coords += [(0,0),(0,width-1),(height-1,0),(height-1,width-1)]
    if len(set(coords))!=85:
        raise ValueError('Sparse predeclared geometry lattice is not distinct')
    save(ROOT/'geometry_sample_coordinates.json',{'selection_plan_sha256':sha((ROOT/'preflight_plan.json').read_bytes()),
        'surface_shape_yx':[height,width],'coords_yx':coords,'selection_without_CT_or_labels':True})
    records=[]
    for i,(y,x) in enumerate(coords):
        loc=offset+48*(y*width+x)
        data=ranges.get(h['canonical_url'],loc,48,f'Predeclared PPM geometry record {i:02d}, row-major YX')
        row={'id':i,'surface_yx':[y,x],'byte_offset':loc}
        for name,endian in [('little','<'),('big','>')]:
            values=struct.unpack(endian+'6d',data)
            finite=all(math.isfinite(v) for v in values)
            norm=math.hypot(*values[3:]) if finite else None
            row[name]={'finite':finite,'values_xyz_nxyz':[v if math.isfinite(v) else None for v in values],
                'normal_magnitude':norm if norm is not None and math.isfinite(norm) else None}
        records.append(row)
        if (i+1)%10==0:
            print(json.dumps({'geometry_records_acquired':i+1,'body_bytes':ranges.state['actual_response_body_bytes_read']}),flush=True)
    return {'sample_count':len(records),'declared_selection':PLAN['ppm_geometry_selection'],
        'record_stride_bytes':48,'byte_order_scope':'Both interpretations recorded; official writer contract reviewed separately',
        'records':records,'no_CT_or_label_pixels_accessed':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['headers','geometry']);args=ap.parse_args()
    ranges=Ranges()
    try:
        if args.stage=='headers':
            headers={'plan_sha256':sha((ROOT/'preflight_plan.json').read_bytes()),'tiffs':[],'pngs':[]}
            for plane in PLAN['tiff_planes']:
                item=tiff_header(ranges,plane);headers['tiffs'].append(item)
                save(ROOT/'header_results.json',headers)
                print(json.dumps({'stage':'TIFF metadata complete','plane':plane,'body_bytes':ranges.state['actual_response_body_bytes_read']}),flush=True)
            for name in PLAN['png_files']:
                headers['pngs'].append(png_header(ranges,name));save(ROOT/'header_results.json',headers)
            headers['ppm']=ppm_header(ranges);save(ROOT/'header_results.json',headers)
            print(json.dumps({'stage':'all headers complete','body_bytes':ranges.state['actual_response_body_bytes_read']}),flush=True)
        else:
            headers=json.loads((ROOT/'header_results.json').read_text())
            result=geometry(ranges,headers);save(ROOT/'geometry_results.json',result)
            print(json.dumps({'stage':'geometry complete','body_bytes':ranges.state['actual_response_body_bytes_read']}),flush=True)
    except BaseException as error:
        save(ROOT/f'failure_{args.stage}.json',{'stage':args.stage,'error_type':type(error).__name__,
            'summary':str(error) if isinstance(error,ValueError) else 'Transport/runtime failure; no automatic retry',
            'body_bytes_read':ranges.state['actual_response_body_bytes_read']})
        raise

if __name__=='__main__':
    main()
