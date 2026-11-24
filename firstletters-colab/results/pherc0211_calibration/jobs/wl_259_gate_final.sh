# slot: cpu
# after: wl_211_acw_c11
# timeout: 7200
# wl lane: on-sheet gate (raw vs +-8 vox control, snapped), xsects, renders, KEEP=1 mesh tarballs
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl211 SC=PHerc0211 VID=20250821151803 Z0=7650 Z1=9150 XZ="7900 8400 8900" XZC=8400 CROP="2750 2431 2000 2000"
T=$(cat $S/fitd_last); echo "tag $T" | tee $OUT/final_tag.txt
KEEP=1 bash $CODE/gate.sh $T 2>&1 | tail -60 > $OUT/gate.log
cp $S/ds/$SC/umbilicus.json $OUT/umbilicus_smoothed.json
ls -la $OUT > $OUT/ls.txt
