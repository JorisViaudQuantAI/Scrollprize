"""Private staging + one-load Reader v3 burst, for pinned Villa b29f310.

No implicit target, no GPU dispatch. Stage and probability files must stay outside
this repository; no uint8 fallback or candidate publication.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch
import numpy as np

CHECKPOINT_SHA = 'ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a'
VILLA_INFER_BLOB = 'f6649e3f9c3e3a8c01d18aa254a70d5a1f0735db'
SCHEMA = 'rv3_private_stage.v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def outside_repo(path):
    p = Path(path).resolve()
    repo = Path(__file__).resolve().parent.parent
    if p == repo or repo in p.parents or any((q / '.git').exists() for q in (p, *p.parents)):
        raise ValueError('private output must be outside a Git checkout')
    return p


def validate_stack(stack, valid):
    if (stack.dtype != np.uint8 or stack.ndim != 3 or stack.shape[0] != 17
            or valid.dtype != np.bool_ or valid.shape != stack.shape[1:]
            or min(valid.shape) == 0):
        raise ValueError('expected uint8 17xHxW and matching bool HxW support')
    if not valid.any():
        raise ValueError('empty CT support; unresolved, not blank')


def stage_arrays(dest, stack, valid, provenance):
    """Write immutable stage via same-filesystem rename; fail on preexisting output."""
    dest = outside_repo(dest)
    if dest.exists():
        raise FileExistsError('stage exists; use a fresh directory')
    validate_stack(stack, valid)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('missing source provenance')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.rv3-stage-') as tmp:
        p = Path(tmp) / 'ready'; p.mkdir()
        np.save(p / 'stack.npy', stack, allow_pickle=False)
        np.save(p / 'valid.npy', valid, allow_pickle=False)
        rec = dict(schema=SCHEMA, shape=list(stack.shape), valid_pixels=int(valid.sum()),
                   sha256={n:sha(p/n) for n in ('stack.npy','valid.npy')}, provenance=provenance)
        (p/'manifest.json').write_text(json.dumps(rec, sort_keys=True, allow_nan=False)+'\n')
        p.rename(dest)
    return rec


def load_stage(src):
    src = outside_repo(src)
    rec = json.loads((src/'manifest.json').read_text())
    if rec.get('schema') != SCHEMA or set(rec.get('sha256', {})) != {'stack.npy','valid.npy'}:
        raise ValueError('bad stage schema')
    for n, expected in rec['sha256'].items():
        if sha(src/n) != expected:
            raise ValueError('stage hash mismatch')
    a = np.load(src/'stack.npy', mmap_mode='r', allow_pickle=False)
    b = np.load(src/'valid.npy', mmap_mode='r', allow_pickle=False)
    validate_stack(a,b)
    if list(a.shape)!=rec['shape'] or int(b.sum())!=rec['valid_pixels']:
        raise ValueError('stage metadata mismatch')
    return a,b,rec


def stage_meshes(mesh_dirs, ct_url, m7_url, dest, *, workers=16, ct_mb=2048, m7_mb=1024):
    """CPU-only shared reader staging. Inputs explicit local tifxyz directories."""
    import stream_snap_render as ssr
    dest = outside_repo(dest)
    dest.mkdir(parents=True, exist_ok=False)
    with ssr.RemoteZarrV2(m7_url,'0',cache_mb=m7_mb,workers=workers) as m7, ssr.RemoteZarrV2(ct_url,'0',cache_mb=ct_mb,workers=workers) as ct:
        for i,mesh in enumerate(mesh_dirs):
            t=time.perf_counter(); mesh=Path(mesh)
            x,y,z,_,meta=ssr.read_tifxyz(mesh)
            before=(m7.bytes_fetched,ct.bytes_fetched,m7.cache_hits,ct.cache_hits)
            a,b,c,geometry=ssr.snap_tifxyz_to_m7(x,y,z,m7,passes=2)
            stack,valid=ssr.render_surface_stack(a,b,c,ct,num_slices=17,slice_step=1.,flip_normals=False,scale=meta.get('scale',[.05,.05]),tile_size=256)
            after=(m7.bytes_fetched,ct.bytes_fetched,m7.cache_hits,ct.cache_hits)
            rec=dict(mesh_sha256={n:sha(mesh/n) for n in ('x.tif','y.tif','z.tif','meta.json') if (mesh/n).is_file()},ct_source=str(ct_url),m7_source=str(m7_url),ct_shape=list(ct.shape),m7_shape=list(m7.shape),geometry=geometry,scale=meta.get('scale',[.05,.05]),passes=2,render_seconds=time.perf_counter()-t,delta_counters=[int(b-a) for a,b in zip(before,after)])
            stage_arrays(dest/f'{i:04d}',stack,valid,rec)


def native_ratio(probability, weight, mask, tile_shape=(384,384)):
    """Capture native ratios with bounded temporary buffers; unknown stays invalid."""
    if tuple(probability.shape)!=mask.shape or tuple(weight.shape)!=mask.shape:
        raise ValueError('accumulator geometry mismatch')
    prob=np.zeros(mask.shape,np.float32);valid=np.zeros(mask.shape,bool)
    h,w=mask.shape;th,tw=map(int,tile_shape)
    for y in range(0,h,th):
        for x in range(0,w,tw):
            sl=np.s_[y:min(y+th,h),x:min(x+tw,w)]
            a=np.asarray(probability[sl],np.float32);b=np.asarray(weight[sl],np.float32)
            if not np.isfinite(b).all() or (b<0).any():raise ValueError('invalid weights')
            v=(b>1e-6)&mask[sl]
            if not np.isfinite(a[v]).all():raise ValueError('invalid supported score sum')
            q=np.zeros(a.shape,np.float32);np.divide(a,b,out=q,where=v)
            if not np.isfinite(q[v]).all():raise ValueError('invalid native ratio')
            prob[sl]=np.clip(q,0,1);valid[sl]=v
    if not valid.any():raise ValueError('no supported prediction; unresolved')
    return prob,valid


def check_villa(villa):
    p=Path(villa.__file__);b=p.read_bytes()
    if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=VILLA_INFER_BLOB:
        raise ValueError('unverified Villa infer.py revision')


class Burst:
    """Serial direct Villa API, configured once per process. No global rmtree hook."""
    def __init__(self, ckpt, *, allow_cpu=False):
        import torch
        from vesuvius.ink_detection.inference import infer as villa
        check_villa(villa)
        self.ckpt=Path(ckpt);self.torch=torch;self.villa=villa;self.config=None;self.device=None
        self.allow_cpu=allow_cpu;self.loads=0;self.load_seconds=0.
        if sha(self.ckpt)!=CHECKPOINT_SHA:raise ValueError('checkpoint SHA mismatch')
        if not torch.cuda.is_available() and not allow_cpu:raise RuntimeError('CUDA required; use explicit CPU test')
        torch.set_float32_matmul_precision('high')
    def args(self, volume, mask, output, bs):
        return self.villa.normalize_inference_paths(self.villa.parse_args([str(volume),str(self.ckpt),str(output),'--mask-path',str(mask),'--overlap','.5','--blend-mode','hann','--batch-size',str(bs),'--num-workers','0','--no-compile']))
    def sync(self):
        if self.device is not None and self.device.type=='cuda':self.torch.cuda.synchronize(self.device)
    def prepare(self,args):
        if self.config is not None:return
        t=time.perf_counter();v=self.villa
        c=v.configure_model(args);m,device=v.prepare_model_for_inference(c.model,gpu_ids=args.gpu_ids,compile_model=False,compile_mode=args.compile_mode)
        if device.type!='cuda' and not self.allow_cpu:raise RuntimeError('refuse non-CUDA fallback')
        self.device=device;self.config=replace(c,model=m);self.loads+=1;self.sync();self.load_seconds=time.perf_counter()-t
    def run(self, stack, valid, work, *, batch_sizes=(8,4,2,1), reload_each=False, keep_tiff=False):
        import tifffile,zarr
        validate_stack(stack,valid);work=outside_repo(work);work.mkdir(parents=True,exist_ok=False)
        volume=work/'surface.zarr';mask=work/'mask.tif'
        zarr.open_array(str(volume),mode='w',shape=stack.shape,chunks=(17,256,256),dtype=np.uint8,zarr_format=2)[:]=stack
        tifffile.imwrite(mask,valid.astype(np.uint8)*255)
        out={};v=self.villa
        try:
            for direction in ('forward','reverse'):
                attempts=[]
                if reload_each:self.config=None;gc.collect()
                for bs in batch_sizes:
                    path=work/f'{direction}-{bs}.tif';args=self.args(volume,mask,path,bs);self.prepare(args);capture={};original=v.write_output_tiff
                    def writer(pa,wa,output_path,tile_shape):
                        if capture:raise RuntimeError('multiple accumulator writes')
                        capture['prob'],capture['valid']=native_ratio(pa,wa,valid,tile_shape)
                        if keep_tiff:original(pa,wa,output_path,tile_shape)
                    self.sync();t=time.perf_counter();oom=False
                    try:
                        with patch.object(v,'write_output_tiff',writer):
                            v.infer_single_zarr(args=args,input_zarr=args.input_zarr,configured_model=self.config,device=self.device,output_tiff=path,layer_direction=direction)
                        self.sync()
                    except self.torch.cuda.OutOfMemoryError:
                        oom=True
                    dt=time.perf_counter()-t;attempts.append(dict(batch_size=bs,seconds=dt,oom=oom))
                    if oom:
                        capture.clear();path.unlink(missing_ok=True);gc.collect();self.torch.cuda.empty_cache();continue
                    if not capture:raise RuntimeError('native accumulator missing; no uint8 fallback')
                    out[direction]=dict(**capture,batch_size=bs,attempts=attempts,seconds=sum(a['seconds'] for a in attempts));break
                else:raise RuntimeError('all batch sizes exhausted; unresolved')
        finally:
            shutil.rmtree(volume,ignore_errors=True);mask.unlink(missing_ok=True)
        return out


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('stage');a.add_argument('--mesh',type=Path,action='append',required=True);a.add_argument('--ct',required=True);a.add_argument('--m7',required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--workers',type=int,default=16);a.add_argument('--ct-mb',type=float,default=2048);a.add_argument('--m7-mb',type=float,default=1024)
    b=sub.add_parser('burst');b.add_argument('--stage',type=Path,action='append',required=True);b.add_argument('--checkpoint',type=Path,required=True);b.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='stage':stage_meshes(a.mesh,a.ct,a.m7,a.out,workers=a.workers,ct_mb=a.ct_mb,m7_mb=a.m7_mb);return
    dest=outside_repo(a.out);dest.mkdir(parents=True,exist_ok=False);runner=Burst(a.checkpoint)
    for i,source in enumerate(a.stage):
        stack,valid,rec=load_stage(source);work=dest/f'{i:04d}';r=runner.run(stack,valid,work)
        for sign,v in r.items():
            np.save(work/f'{sign}_prob.npy',v['prob'],allow_pickle=False);np.save(work/f'{sign}_valid.npy',v['valid'],allow_pickle=False)
        (work/'receipt.json').write_text(json.dumps(dict(stage_manifest_sha256=sha(source/'manifest.json'),model_loads=runner.loads,load_seconds=runner.load_seconds,rows={s:{k:x for k,x in v.items() if k not in ('prob','valid')} for s,v in r.items()}),allow_nan=False)+'\n')
if __name__=='__main__':main()
