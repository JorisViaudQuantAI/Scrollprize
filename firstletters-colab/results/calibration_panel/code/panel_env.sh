# wl_env.sh - lane wl ink side (from si_env.sh). Job sets WLSC=PHerc0211|PHerc0826. Cache: /content/cache/hbink.
set -eo pipefail
T0=$(date +%s)
C=/content/cache/hbink; mkdir -p $C/locks $C/ckpt $C/renders $C/dst $C/maps $C/red $C/dcache
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
case "${WLSC:-PHerc0846B}" in PHerc0211) VOL=$B/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr;; PHerc0846B) VOL=$B/PHerc0846B/volumes/20250804142305-9.362um-1.2m-113keV-masked.zarr;; esac
UM=9.362; case "$VOL" in *8.640um*) UM=8.64;; esac
stamp() { echo "$1 s=$(( $(date +%s)-${2:-$T0} ))" | tee -a $OUT/timings.txt; }
(
  flock 9
  if [ ! -f $C/pylib/.ok ]; then
    rm -rf $C/pylib.tmp
    python -m pip -q install --no-deps --target $C/pylib.tmp 'zarr==2.18.3' 'numcodecs==0.15.1' asciitree fasteners deprecated wrapt safetensors
    touch $C/pylib.tmp/.ok; rm -rf $C/pylib; mv $C/pylib.tmp $C/pylib
  fi
  python -c 'import requests, tifffile, skimage, timm, einops, cv2, scipy, imagecodecs' 2>/dev/null || python -m pip -q install requests tifffile scikit-image timm einops opencv-python-headless scipy imagecodecs
  [ -s $C/ckpt/argus-v3.pth ] || { curl -sL --fail -o $C/ckpt/ag.tmp https://huggingface.co/ruiconte/argus-v3/resolve/main/argus-v3.pth && mv $C/ckpt/ag.tmp $C/ckpt/argus-v3.pth; }
  [ -s $C/ckpt/reader-v3.pth ] || { curl -sL --fail -o $C/ckpt/rv3.tmp https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth && mv $C/ckpt/rv3.tmp $C/ckpt/reader-v3.pth; }
  if [ ! -d $C/studio-release/.git ]; then
    rm -rf $C/studio-release; git clone -q https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt $C/studio-release
    git -C $C/studio-release checkout -q bf66f753176792a966bca17822629fb8821f20c8
  fi
  if [ ! -f $C/code_v3/.ok ]; then
    rm -rf $C/code_v3.tmp; mkdir -p $C/code_v3.tmp; tar xzf $INPUTS/wl_ink_v3.tgz -C $C/code_v3.tmp --strip-components=1
    touch $C/code_v3.tmp/.ok; rm -rf $C/code_v3; mv $C/code_v3.tmp $C/code_v3
  fi
) 9>$C/locks/setup.lock
export PYTHONPATH=$C/pylib
CODE=$C/code_v3
RV3=$C/ckpt/reader-v3.pth; AG3=$C/ckpt/argus-v3.pth; SW=$C/ckpt/studio-softloss-20261023-ema.safetensors
ST="python $CODE/studio_t4.py --weights $SW --release-dir $C/studio-release"
INF="python $CODE/infer_ink.py"
export FL_W=$CODE/vw

# si_render NAME VOLUME MESHDIR : 41 layers, 8.64 um, flip-normals (renderer default), mesh at layer 20; + onsheet map + dst
si_render() {
  local n=$1 v=$2 m=$3 R=$C/renders/$1
  if [ ! -s $R/render_meta.json ]; then
    rm -rf $R.tmp
    OMP_NUM_THREADS=${RTHREADS:-3} python $CODE/render_tifxyz.py --volume $v --tifxyz $m --out $R.tmp --layers 41 --voxel-um $UM --surface-interp bicubic ${RXARGS:-} \
      --workers ${RWORKERS:-16} --cache-mb 3000 --disk-cache $C/dcache/$(basename $v .zarr) 2>&1 | tail -1 | sed "s/^/$n /" | tee -a $OUT/render.txt
    [ -s $R.tmp/render_meta.json ] || { echo "RENDER FAILED $n" | tee -a $OUT/render.txt; return 1; }
    rm -rf $R; mv $R.tmp $R
  fi
  if [ ! -s $C/dst/$n/area.json ]; then
    rm -f $C/renders/$n.onsheet.json
    python $CODE/onsheet_score.py --layers $R/layers --mid 20 --out $C/renders/$n.onsheet.json --map $C/renders/$n.onsheet_map.npy > /dev/null
    python $CODE/si_tools.py prep $R $C/renders/$n.onsheet_map.npy $C/dst/$n --tag $n --um $UM | tee -a $OUT/prep.txt
  fi
  mkdir -p $OUT/look; cp $C/dst/$n/area.json $OUT/look/${n}_area.json; cp $C/renders/$n.onsheet.json $OUT/look/${n}_onsheet.json
  cp $C/renders/$n.onsheet_map.npy $OUT/look/${n}_onsheet_map.npy; cp $C/dst/$n/mid20.jpg $OUT/look/${n}_mid20.jpg
  cp $C/dst/$n/mid20_par.jpg $OUT/look/${n}_mid20_par.jpg
}

# si_rv3 NAME : reader-v3 both directions, shifts -3,+3 (mean kept per direction), on the full render mask
si_rv3() {
  local n=$1 M=$C/maps/$1/rv3
  ls $M/rv3_reverse__*_z+3.png >/dev/null 2>&1 && return 0
  mkdir -p $M
  $INF $C/renders/$n --input-um $UM --surface-layer 20 -c $RV3 -o $M/rv3.png --direction both --z-shifts=-3,3 --save-each \
    --batch-size ${RBATCH:-16} --threads ${RTH:-6} --no-display 2>&1 | grep -E "tiles in|done in|Error|error" | sed "s/^/$n /" | tee -a $OUT/ink.txt || true
}

# si_post NAME DET MAPDIR : at_atlas post restricted to parallel tiles (dst mask)
si_post() {
  local n=$1 d=$2 M=$3
  python $CODE/at_atlas.py post $C/dst/$n $M $OUT --um $UM --det $d --red $C/red 2>&1 | grep -E "SUMMARY|Error|error" | cut -c1-600 | tee -a $OUT/summary.txt || true
  mv $OUT/post_$d.json $OUT/post_${n}_$d.json 2>/dev/null || true
  local t=$(python -c "import json;d=json.load(open('$C/dst/$n/prep.json'));print(d['scroll']+'_'+d['mesh'])")
  mkdir -p $OUT/red; cp $C/red/${t}_${d}_ov20.jpg $C/red/${t}_${d}_z20.png $C/red/${t}_${d}_z_half.npy $OUT/red/ 2>/dev/null || true
  cp -n $C/red/${t}_ct20.png $OUT/red/ 2>/dev/null || true
}

# si_st NAME ORDER SHIFT : Studio packaged CLI, stride 48 Hann, fp16, batch 16 -> 8-bit map png for post
si_st() {
  local n=$1 o=$2 s=$3 M=$C/maps/$1/st${2}_s$3
  [ -s $M/st${o}__studio_z+0.png ] && return 0
  mkdir -p $M; rm -f $M/x.*
  $ST --input $C/renders/$n --mesh-index 20 --order $o --shift $s --stride 48 --precision fp16 --batch-size 16 --device cuda --out-prefix $M/x 2>&1 | tail -1
  python $CODE/si_tools.py stpng $M/x $M/st${o}__studio_z+0.png | sed "s/^/$n /" | tee -a $OUT/st.txt
  cp $M/x.receipt.json $OUT/receipt_${n}_st${o}_s$s.json
  python -c "import numpy as np; np.save('$M/x.f16.npy', np.load('$M/x.prob.npy').astype(np.float16))"
  rm -f $M/x.prob.npy $M/x.preview.png
}

# wl_ag NAME : argus-v3 (drop-in reader-v3 weights) both directions, shifts -3,+3
wl_ag() {
  local n=$1 M=$C/maps/$1/ag3
  ls $M/ag3_reverse__*_z+3.png >/dev/null 2>&1 && return 0
  mkdir -p $M
  $INF $C/renders/$n --input-um $UM --surface-layer 20 -c $AG3 -o $M/ag3.png --direction both --z-shifts=-3,3 --save-each \
    --batch-size ${RBATCH:-16} --threads ${RTH:-6} --no-display 2>&1 | grep -E "tiles in|done in|Error|error" | sed "s/^/$n /" | tee -a $OUT/ink.txt || true
}
