"""Generate a pinned Villa integration patch; --apply modifies a clean checkout.
The measured uint8 reader variant is NOT installed: keep existing reader dtype.
Only the verified NumPy version enables the guarded exact-integer optimization.
"""
import argparse,difflib,hashlib
from pathlib import Path

INFER='f6649e3f9c3e3a8c01d18aa254a70d5a1f0735db'
NORM='3ad31df4fc05981e97e077a0476a9cd97b2c57b8'
ROOT=Path('vesuvius/src/vesuvius')
OLD='        return np.ascontiguousarray(normalize_robust(patch_ZYX))\n'
NEW='''        # B15: exact integer specialization, validated with NumPy 2.3.5.
        if np.__version__ == "2.3.5" and patch_ZYX.shape == (17, 384, 384):
            from .fast_mad import fast_guarded

            return np.ascontiguousarray(fast_guarded(patch_ZYX, fallback=normalize_robust))
        return np.ascontiguousarray(normalize_robust(patch_ZYX))
'''


def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('villa',type=Path);p.add_argument('--apply',action='store_true');a=p.parse_args()
    infer=a.villa/ROOT/'ink_detection/inference/infer.py';norm=a.villa/ROOT/'image_proc/intensity/normalization.py';helper=infer.with_name('fast_mad.py');raw=infer.read_bytes()
    if blob(raw)!=INFER or blob(norm.read_bytes())!=NORM:raise SystemExit('STOP: unexpected Villa source; do not patch')
    old=raw.decode();assert old.count(OLD)==1;new=old.replace(OLD,NEW,1);compile(new,str(infer),'exec')
    if helper.exists():raise SystemExit('STOP: helper exists')
    print(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(infer),tofile=str(infer))))
    if a.apply:
        helper.write_bytes(Path(__file__).with_name('fast_mad.py').read_bytes());infer.write_text(new);print('Installed candidate; run paired full Villa CPU inference before promotion')

if __name__=='__main__':main()
