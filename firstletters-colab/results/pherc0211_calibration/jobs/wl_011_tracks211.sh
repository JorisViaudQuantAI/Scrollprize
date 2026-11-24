# slot: cpu
# after: wl_001_setup211
# timeout: 5400
# wl lane: PHerc0211 published track store (dl.ash2txt.org) download
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl211 SC=PHerc0211 VID=20250821151803 UMBF=20250821151803-umbilicus-20260808112626.json Z0=7650 Z1=9150
bash $CODE/prep_wl.sh tracks 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
cp $S/ds/$SC/tracks/*.extract.json $OUT/ 2>/dev/null; cp $S/ds/$SC/tracks/*.vctracks/metadata.json $OUT/tracks_metadata.json 2>/dev/null
exit $rc
