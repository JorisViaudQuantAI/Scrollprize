# slot: cpu
# timeout: 7000
# regrow 0846B blind bl1/bl2 exactly as hb_011_grow (90 gen, 0306B params, threads differ: 3 vs 4), flatboi, gridq; compare 0.995/0.57
# BLIND recipe, grow stage: seeds by CT anatomy -> snap to m7 (+-48 vox) -> GrowPatch 90 gen (0306B winners' params) -> flatboi -> gridq
# vars: SCR (scroll dir name), VID (volume id), UMV (voxel um), SEEDS ("name:x:y:z[:crop]" ...), ZR ("z0 z1" normal-grid xy range)
SCR=PHerc0846B; VID=20250804142305; UMV=9.362; ZR="4000 8300"
SEEDS="bl1:3200:5000:5800 bl2:4800:4750:6400"
set -u; G0=/content/cache/blind/$SCR; mkdir -p $G0 /content/cache/hbvc; cd $G0
cp $INPUTS/fl_helpers.py $INPUTS/gridq.py $INPUTS/crop_pt.py .; export PYTHONPATH=$G0:/content/cache/hbink/pylib
python3 -c 'import requests, tifffile, imagecodecs, zarr, fsspec' 2>/dev/null || pip -q install requests tifffile imagecodecs fsspec 'zarr==2.18.3' 'numcodecs==0.15.1'
( flock 9; if [ ! -x /content/cache/hbvc/vc3d/squashfs-root/AppRun ]; then
  U=$(curl -s https://api.github.com/repos/ScrollPrize/villa/releases | python3 -c "import json,sys; d=json.load(sys.stdin); print([a['browser_download_url'] for r in d[:5] for a in r.get('assets',[]) if 'AppImage' in a['name']][0])")
  echo "appimage $U" | tee $OUT/appimage.txt; rm -rf /content/cache/hbvc/vc3d.tmp; mkdir -p /content/cache/hbvc/vc3d.tmp; cd /content/cache/hbvc/vc3d.tmp && curl -fsSL -o V.AppImage "$U" && chmod +x V.AppImage && ./V.AppImage --appimage-extract >/dev/null && rm V.AppImage
  cd /content/cache/hbvc; rm -rf vc3d; mv vc3d.tmp vc3d; fi ) 9>/tmp/hb_vc.lock
cd $G0; source $INPUTS/vcenv.sh
P=$SCR/representations/predictions/surfaces/$VID-surface-20260413222639-surface-m7-L0-th0.2; NG=$G0/ng; B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
set -- $ZR; t=$(date +%s); python3 -c "import fl_helpers as H; print(H.download_normal_grids('$P.normal-grids', '$NG', $1, $2, margin=0, threads=32))" 2>&1 | tail -1; echo "grids $(( $(date +%s)-t ))s" | tee -a $OUT/t.txt
fl() { PASTIX_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 OPENBLAS_CORETYPE=Haswell vc flatboi $1 --output=$2 50 symmetric_dirichlet > $2.log 2>&1; tail -1 $2.log; }
grow() { local n=$1 x=$2 y=$3 z=$4 cr=${5:-0} G=$G0/$1; mkdir -p $G
  local s=$(python3 -c "import fl_helpers as H; r=H.snap_seed('$P.zarr', ($x,$y,$z), 48); print(*r[:3]) if r else print('NONE')")
  echo "$n seed $x $y $z -> $s" | tee -a $OUT/seeds.txt; [ "$s" = NONE ] && return
  python3 -c "import fl_helpers as H; H.write_grow_params('$G/params.json', '$NG', $UMV, generations=90, threads=4)"
  local t=$(date +%s); ( cd $G && vc vc_grow_seg_from_seed -v $B/$P.zarr -t $G -p $G/params.json --seed $s ) > $G/grow.log 2>&1
  echo "$n grow $(( $(date +%s)-t ))s"; local M=$(ls -d $G/auto_grown_* | tail -1); rm -rf $G0/raw_$n; cp -r $M $G0/raw_$n
  if [ $cr -gt 0 ]; then python3 crop_pt.py $G0/raw_$n $G0/rawc_$n $s $cr | tee -a $OUT/crops.txt; fl $G0/rawc_$n $G0/flat_$n; else fl $G0/raw_$n $G0/flat_$n; fi; }
for sd in $SEEDS; do IFS=: read n x y z cr <<< "$sd"; grow $n $x $y $z ${cr:-0} > $OUT/g_$n.log 2>&1 & done; wait; cat $OUT/g_*.log
python3 gridq.py $(ls -d flat_* | grep -v -E "log|txt|png") 2>&1 | grep -v Trace | cut -c1-330 | tee $OUT/gridq.txt
tar czf $OUT/meshes_$SCR.tgz $(ls -d flat_* | grep -v -E "log|txt|png")
