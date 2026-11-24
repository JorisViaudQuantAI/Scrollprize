#!/bin/bash
# gate.sh TAG : on-sheet gate for the meshes exported by chunk TAG: raw (+snapped) oncheck vs same-vertex +-8 vox control,
# cross-sections, 41-layer renders (native voxel, mesh at layer 20) of 4 windings + onsheet_score.
set -uo pipefail
S=${S:-/content/cache/wl}; TAG=$1; CODE=${CODE:?}; PY=$(cat $S/spiral_py.txt)
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
VOL=$B/${SC:?}/volumes/${VID:?}-9.362um-1.2m-113keV-masked.zarr
RD=$(ls -d $S/fits/*_$TAG | head -1)
$PY -c "import torch;c=torch.load('$RD/checkpoint_fitted.ckpt',map_location='cpu',weights_only=False);print('completed_iterations',c['completed_iterations'])" | tee $OUT/gate_step.txt
SNAP=${SNAP:-1} KEEP=${KEEP:-0} bash $CODE/check_fit.sh $TAG 200 2>&1 | tee $OUT/check.txt
W=$S/out/$TAG
$PY - <<PY | tee $OUT/gate_summary.txt
import csv, numpy as np
for v in ('fitted','snap'):
    try: R=list(csv.DictReader(open('$W/check/%s/oncheck.tsv'%v),delimiter='\t'))
    except Exception as e: print(v,'missing',e); continue
    print(v,'columns',list(R[0].keys()))
    I=[r for r in R if float(r.get('fit_ct_found',1))>=0.9]
    for k in R[0]:
        try: a=np.array([float(r[k]) for r in I])
        except: continue
        print(v,k,'inside n=%d mean %.3f median %.3f'%(len(a),np.nanmean(a),np.nanmedian(a)))
PY
python3 -c "import cv2,requests,tifffile" 2>/dev/null || pip install -q opencv-python-headless requests tifffile
SEL=$($PY - <<PY
import csv
R=[r for r in csv.DictReader(open('$W/check/fitted/oncheck.tsv'),delimiter='\t') if float(r.get('fit_ct_found',1))>=0.9]
R.sort(key=lambda r:int(r['winding'][1:4]))
n=len(R); idx=[int(n*f) for f in (0.2,0.4,0.6,0.8)] if n>=4 else range(n)
print(' '.join(R[i]['winding'][:4] for i in idx))
PY
)
echo "render windings: $SEL" | tee $OUT/render_windings.txt
mkdir -p $S/renders
for w in $SEL; do
  M=$W/fitted/$w; [ -d $M ] || continue
  read NR NC < <(python3 -c "import tifffile;a=tifffile.imread('$M/x.tif');print(*a.shape)")
  C0=$(( NC/2 - 40 )); [ $C0 -lt 0 ] && C0=0
  R=$S/renders/${TAG}_$w; rm -rf $R
  OMP_NUM_THREADS=4 python3 $CODE/render_tifxyz.py --volume $VOL --tifxyz $M --out $R --layers 41 --voxel-um 9.362 --workers 24 --cache-mb 4000 --grid-crop $C0 0 80 $NR 2>&1 | tail -1
  python3 $CODE/onsheet_score.py --layers $R/layers --out $OUT/onsheet_${w}.json --map $OUT/onsheet_${w}_map.npy 2>&1 | tail -2
  for f in layer_mid.png composite_max.png mask.png render_meta.json; do cp $R/$f $OUT/render_${w}_$f 2>/dev/null; done
done
python3 - <<PY | tee -a $OUT/gate_summary.txt
import json,glob
for f in sorted(glob.glob('$OUT/onsheet_w*.json')):
    d=json.load(open(f)); print(f.split('/')[-1], {k:d[k] for k in d if k in ('tiles','f_parallel','f_drift','f_still','n_tiles','scored')})
PY
