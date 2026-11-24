"""slice_range.py URL_ZARR LEVEL Z0 OUT.png : one transverse plane of an uncompressed zarr v2 level via HTTP Range reads."""
import sys, json, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
url, lev, z0, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
za = json.load(urllib.request.urlopen(f'{url}/{lev}/.zarray'))
Z, Y, X = za['shape']; cz, cy, cx = za['chunks']; z = z0 >> lev
ci, k = z // cz, z % cz
img = np.zeros((Y, X), np.uint8)
def get(a):
    j, i = a
    r = urllib.request.Request(f'{url}/{lev}/{ci}/{j}/{i}', headers={'Range': f'bytes={k*cy*cx}-{(k+1)*cy*cx-1}'})
    try: b = urllib.request.urlopen(r, timeout=60).read()
    except Exception: return
    t = np.frombuffer(b, np.uint8).reshape(cy, cx)
    h, w = min(cy, Y - j*cy), min(cx, X - i*cx); img[j*cy:j*cy+h, i*cx:i*cx+w] = t[:h, :w]
with ThreadPoolExecutor(16) as ex: list(ex.map(get, [(j, i) for j in range(-(-Y//cy)) for i in range(-(-X//cx))]))
nz = img[img > 0]; lo, hi = (np.percentile(nz, [1, 99.5]) if nz.size else (0, 255))
Image.fromarray(np.clip((img.astype(float)-lo)/(hi-lo+1e-6)*255, 0, 255).astype(np.uint8)).save(out)
print(out, img.shape, 'nonzero', round((img > 0).mean(), 3))
