# slot: cpu
# after: wl_001_setup211
# timeout: 5400
# wl lane: PHerc0211 band z 7650-9150 data prep stages: m7 ct lasagna native patch spec (all official inputs)
rm -f /content/cache/wl211/prep/m7.ok /content/cache/wl211/prep/ct.ok; set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; export S=/content/cache/wl211 SC=PHerc0211 VID=20250821151803 UMBF=20250821151803-umbilicus-20260808112626.json Z0=7650 Z1=9150
bash $CODE/prep_wl.sh m7 ct lasagna native patch spec 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
cp $S/native_build.log $S/native_cmake.log $OUT/ 2>/dev/null; git -C $S/villa diff > $OUT/fitter_patch.diff
cp $S/ds/$SC/umbilicus*.json $OUT/ 2>/dev/null; du -sh $S/* > $OUT/du.txt 2>&1; free -g >> $OUT/du.txt
exit $rc
