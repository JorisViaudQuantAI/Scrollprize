# slot: cpu
# after: wl_010_data211
# timeout: 7200
# wl lane: PHerc0211 band z 7650-9150 data prep stages: sdt pools
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl211 SC=PHerc0211 VID=20250821151803 UMBF=20250821151803-umbilicus-20260808112626.json Z0=7650 Z1=9150
SDT_WORKERS=6 bash $CODE/prep_wl.sh sdt pools 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
du -sh $S/* > $OUT/du.txt 2>&1; free -g >> $OUT/du.txt
exit $rc
