# slot: cpu
# after: wl_520_data826
# timeout: 7200
# wl lane: PHerc0826 band z 11800-13300 data prep stages: sdt pools
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl826 SC=PHerc0826 VID=20250821151701 UMBF=20250821151701-umbilicus-20260808113303.json Z0=11800 Z1=13300
SDT_WORKERS=6 bash $CODE/prep_wl.sh sdt pools 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
du -sh $S/* > $OUT/du.txt 2>&1; free -g >> $OUT/du.txt
exit $rc
