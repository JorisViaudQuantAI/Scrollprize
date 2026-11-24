# slot: cpu
# after: wl_001_setup826
# timeout: 5400
# wl lane: PHerc0826 published track store (dl.ash2txt.org) download
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl826 SC=PHerc0826 VID=20250821151701 UMBF=20250821151701-umbilicus-20260808113303.json Z0=11800 Z1=13300
bash $CODE/prep_wl.sh tracks 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
cp $S/ds/$SC/tracks/*.extract.json $OUT/ 2>/dev/null; cp $S/ds/$SC/tracks/*.vctracks/metadata.json $OUT/tracks_metadata.json 2>/dev/null
exit $rc
