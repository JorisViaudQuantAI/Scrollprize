"""Replay the released PHerc. 306B CPU smoke on its published crop and three finalists.

Source and licence: SirProdigle/pherc0306b-first-letters, code MIT, model/data CC BY-NC 4.0.
Only the upstream public smoke field is an input. No work on an unread scroll is required.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

RELEASE='feabff6c54f95bd8e52a90e0a7a76881ee67d4cc'
DATA='f435998b55269762942a979052044309b29650ac'
WEIGHTS='e57c4d541c239e06450a98c1cf2015d3de8d887e'
REPO='https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/'+RELEASE+'/'
ORIGINS={'repo':REPO, 'hf-dataset':'https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/'+DATA+'/', 'hf-model':'https://huggingface.co/Prodigle/pherc0306b-ink-ahn/resolve/'+WEIGHTS+'/'}
MANIFEST_SHA='6e56c61594cfce44ab24d9566da2f6d88ad14e3a6edbafea1b1f465cfdd97770'
PUBLIC_FILES=['code/claim/ink_v8.py','code/claim/smoke_v7.py','v8in/config.json','v8in/ink8um/__init__.py','v8in/ink8um/inference.py','v8in/ink8um/modeling.py','v8in/model.safetensors','data/smoke/crop.npz']


def fetch(url,path,expected):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'public-pherc306b-verification'})
            with urllib.request.urlopen(req,timeout=100) as source,tmp.open('wb') as out:
                shutil.copyfileobj(source,out,1024*1024)
            with tmp.open('rb') as f:
                digest=hashlib.file_digest(f,'sha256').hexdigest()
            if expected!='-' and digest!=expected:
                raise ValueError(f'Bad published checksum for {path.name}')
            tmp.replace(path)
            return digest
        except OSError:
            if attempt==3:raise
            time.sleep(2**attempt)


def replay(outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='public-pherc306b-') as work:
        root=Path(work);index=root/'context.tsv'
        fetch(REPO+'docker/context.tsv',index,MANIFEST_SHA)
        with index.open() as f:
            records={path:(src,sha) for path,src,sha in csv.reader((l for l in f if l.strip() and not l.startswith('#')),delimiter='\t')}
        sources={}
        def materialize(rel):
            locator,expected=records[rel]
            kind,remote=locator.split(':',1)
            url=remote if kind=='url' else ORIGINS[kind]+remote
            sources[rel]=fetch(url,root/rel,expected)
            return root/rel
        for rel in PUBLIC_FILES:materialize(rel)
        env={**os.environ,'V8_DIR':str(root/'v8in'),'INK_DEVICE':'cpu','HF_HUB_OFFLINE':'1'}
        outputs=[]
        for seed in (0,1,2):
            weights_name=f'finalist/hn4_s{seed}.safetensors'
            weights=materialize(weights_name);reference=materialize(f'data/members/fwd_hn4_s{seed}.npy')
            output=outdir/f'seed-{seed}.json'
            subprocess.run([sys.executable,str(root/'code/claim/smoke_v7.py'),'run',str(root/'data/smoke/crop.npz'),'--weights',str(weights),'--sha',sources[weights_name],'--map',str(reference),'--tta','1','--out',str(output)],env=env,check=True)
            outputs.append(json.loads(output.read_text()))
            weights.unlink();reference.unlink()
        record={'method_repo_commit':RELEASE,'dataset_revision':DATA,'finalist_revision':WEIGHTS,'authors':'Liam Warner and Nicola Bodill; see the published credits for the underlying models and data.','input_hashes_sha256':sources,'results':outputs,'scope':'Numerical reproduction on the original published CT field, using the complete author-selected 36-window grid per seed, one pose. This cannot measure transfer to an unknown scroll.'}
        (outdir/'results.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);replay(ap.parse_args().output)
