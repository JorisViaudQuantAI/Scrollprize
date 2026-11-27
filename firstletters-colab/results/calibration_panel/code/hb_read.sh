# hb_read.sh — sourced after hb_env.sh. One function per detector, all writing 8-bit prob*255 PNGs into a
# panel dir $C/panel/NAME as <det>__x_z<shift>.png (the at_atlas post naming), then post + share>0.5.
# Detectors (det names):
#   rv3old_fwd / rv3old_rev : OUR OLD PATH = infer_ink.py, 41-layer render, surface layer 20, z-shifts -3,+3 averaged,
#                              resample auto (9.362 -> 9.6 um), our tiling (crop_size from ckpt, overlap .5)
#   villa_fwd / villa_rev   : WINNER'S RECIPE on our render = villa infer @b29f310, central 17 layers (12..28),
#                              patch 384 stride 192 hann, robust norm, no resample (their venv)
#   s48_asc / s48_desc      : Studio packaged CLI bf66f753, fp16, stride 48 Hann, mesh index 20, shift 0
#   s96_asc / s96_desc      : same, stride 96 (unblended tiles)
PANEL=$C/panel; mkdir -p $PANEL
hb_villa() {  # NAME
  local n=$1 R=$C/renders/$1 P=$PANEL/$1; mkdir -p $P
  [ -s $P/villa_rev__x_z+0.png ] && return 0
  env -u PYTHONPATH $C/venv/bin/python - $R $C/v17_$n.zarr <<'PY'
import sys, glob, numpy as np, tifffile, zarr
R, Z = sys.argv[1], sys.argv[2]; fs = sorted(glob.glob(R + '/layers/*.tif'))
st = np.stack([tifffile.imread(f) for f in fs[12:29]])
zarr.open_array(Z, mode='w', shape=st.shape, chunks=(17, 256, 256), dtype=np.uint8, zarr_format=2)[:] = st
tifffile.imwrite(Z + '.mask.tif', ((tifffile.imread(fs[20]) > 0).astype(np.uint8) * 255))
PY
  local CK=$(ls ~/.cache/huggingface/hub/models--domenicor046--reader-v3/snapshots/39f626e595304c35c6ef574ffc0b6cc1e0ed44cb/reader-v3.pth)
  for d in forward reverse; do
    env -u PYTHONPATH $C/venv/bin/python -c "import sys; from vesuvius.ink_detection.inference import infer; infer.main(sys.argv[1:])" $C/v17_$n.zarr $CK $P/v_$d.tif \
      --mask-path $C/v17_$n.zarr.mask.tif --overlap 0.5 --blend-mode hann --direction $d --batch-size 4 --num-workers 0 --no-compile 2>&1 | grep -iE "error|patches for" | tail -2
  done
  python -c "
import tifffile, cv2
for d, k in (('forward','fwd'), ('reverse','rev')): cv2.imwrite('$P/villa_'+k+'__x_z+0.png', tifffile.imread('$P/v_'+d+'.tif'))"
  rm -rf $C/v17_$n.zarr
}
hb_rv3old() {  # NAME
  local n=$1 P=$PANEL/$1; mkdir -p $P; si_rv3 $n
  for f in $C/maps/$n/rv3/rv3_forward__*_z[+-]3.png; do ln -sf $f $P/rv3old_fwd__x_z${f##*_z}; done
  for f in $C/maps/$n/rv3/rv3_reverse__*_z[+-]3.png; do ln -sf $f $P/rv3old_rev__x_z${f##*_z}; done
}
hb_s48() {  # NAME ORDER...
  local n=$1; shift; local P=$PANEL/$n; mkdir -p $P
  for o in "$@"; do si_st $n $o 0; ln -sf $C/maps/$n/st${o}_s0/st${o}__studio_z+0.png $P/s48_${o}__x_z+0.png; done
}
hb_s96() {  # NAME ORDER...
  local n=$1; shift; local P=$PANEL/$n; mkdir -p $P
  for o in "$@"; do local M=$C/maps/$n/s96$o; mkdir -p $M
    [ -s $M/st96${o}__studio_z+0.png ] || { $ST --input $C/renders/$n --mesh-index 20 --order $o --shift 0 --stride 96 --precision fp16 --batch-size 32 --device cuda --out-prefix $M/x 2>&1 | tail -1
      python $CODE/si_tools.py stpng $M/x $M/st96${o}__studio_z+0.png > /dev/null; rm -f $M/x.prob.npy; }
    ln -sf $M/st96${o}__studio_z+0.png $P/s96_${o}__x_z+0.png; done
}
hb_post() {  # NAME : post every det present + share of p>0.5 on the dst mask -> $OUT/panel_NAME.json
  local n=$1 P=$PANEL/$1
  for d in $(ls $P | grep '__x_z' | sed 's/__x_z.*//' | sort -u); do si_post $n $d $P > /dev/null; done
  python - $n $P $C/dst/$n $OUT <<'PY'
import sys, glob, json, os, numpy as np, cv2
n, P, D, O = sys.argv[1:]
m = cv2.imread(D + '/mask.png', 0) > 0; res = {}
for f in sorted(glob.glob(P + '/*__x_z*.png')):
    det = os.path.basename(f).split('__')[0]; a = cv2.imread(f, 0)
    if a.shape != m.shape: a = cv2.resize(a, m.shape[::-1], interpolation=cv2.INTER_LINEAR)
    r = res.setdefault(det, {'share_gt05': []}); r['share_gt05'].append(round(float((a[m] > 127).mean()), 4))
for det in res:
    pj = f'{O}/post_{n}_{det}.json'
    if os.path.exists(pj):
        j = json.load(open(pj)); res[det].update(line_max=j.get('line_max'), row_max=j.get('row_max'), hot_z2=j.get('hot_frac_z2'), valid_cm2=j.get('valid_cm2'))
json.dump(res, open(f'{O}/panel_{n}.json', 'w'), indent=1); print(n, json.dumps(res))
PY
}
