# slot: cpu
# after: wl_001_setup826
# timeout: 5400
# wl lane: PHerc0826 band z 11800-13300 data prep stages: m7 ct lasagna native patch spec (all official inputs)
rm -f /content/cache/wl826/prep/m7.ok /content/cache/wl826/prep/ct.ok; set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl826 SC=PHerc0826 VID=20250821151701 UMBF=20250821151701-umbilicus-20260808113303.json Z0=11800 Z1=13300
bash $CODE/prep_wl.sh m7 ct lasagna native patch spec 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
cp $S/native_build.log $S/native_cmake.log $OUT/ 2>/dev/null; git -C $S/villa diff > $OUT/fitter_patch.diff
cp $S/ds/$SC/umbilicus*.json $OUT/ 2>/dev/null; du -sh $S/* > $OUT/du.txt 2>&1; free -g >> $OUT/du.txt
exit $rc
