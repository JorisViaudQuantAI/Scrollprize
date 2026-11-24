# slot: cpu
# after: wl_312_acw_c12
# timeout: 7200
# wl lane: on-sheet gate (raw vs +-8 vox control, snapped), xsects, renders, KEEP=1 mesh tarballs
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl826 SC=PHerc0826 VID=20250821151701 Z0=11800 Z1=13300 XZ="12000 12550 13100" XZC=12550 CROP="4000 2900 1800 2400"
T=$(cat $S/fitd_last); echo "tag $T" | tee $OUT/final_tag.txt
KEEP=1 bash $CODE/gate.sh $T 2>&1 | tail -60 > $OUT/gate.log
cp $S/ds/$SC/umbilicus.json $OUT/umbilicus_smoothed.json
ls -la $OUT > $OUT/ls.txt
