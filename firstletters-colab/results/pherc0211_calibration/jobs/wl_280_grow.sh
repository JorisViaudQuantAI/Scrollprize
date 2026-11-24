# slot: cpu
# timeout: 10800
# Task 1 refinement arm (winners' loop): GrowPatch (0306B winners' params, 100 generations) seeded at the two BLIND look
# points (fw033 rev chevrons, fw035 rev V/Y), snapped to the nearest snap-mesh vertex; VC3D AppImage e5707e4
set -u; source $INPUTS/vcenv.sh
export PYTHONPATH=$PWD; cp $INPUTS/fl_helpers.py .
python3 -c 'import requests' 2>/dev/null || pip -q install requests
NGK=PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
NG=/content/cache/wl211/ngrid; M7=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr
t=$(date +%s); python3 -c "import fl_helpers as H; print(H.download_normal_grids('$NGK', '$NG', 7000, 9700, margin=0, threads=32))" 2>&1 | tail -3; echo "grids $(( $(date +%s)-t ))s" | tee -a $OUT/t.txt; du -sh $NG
grow() { local n=$1 x=$2 y=$3 z=$4 G=/content/cache/wl211/grow/$1; mkdir -p $G
  python3 -c "import fl_helpers as H; H.write_grow_params('$G/params.json', '$NG', 9.362, generations=100, threads=5)"
  local s=$(date +%s); ( cd $G && vc vc_grow_seg_from_seed -v $M7 -t $G -p $G/params.json --seed $x $y $z ) > $G/grow.log 2>&1
  echo "$n grow $(( $(date +%s)-s ))s"; tail -3 $G/grow.log; M=$(ls -d $G/auto_grown_* | tail -1); mkdir -p $OUT/$n; cp $M/*.tif $M/*.json $OUT/$n/ 2>/dev/null; ln -sfn $M $G/mesh; }
grow gp1 3280 3352 8802 > $OUT/gp1.log 2>&1 &
grow gp2 3900 3658 8009 > $OUT/gp2.log 2>&1 &
wait; cat $OUT/gp*.log
