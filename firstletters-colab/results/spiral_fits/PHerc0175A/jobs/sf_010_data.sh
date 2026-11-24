# slot: cpu
# after: sf_001_setup
# timeout: 7200
# sf lane: PHerc0175A z 6900-8400 dataset (tracks + placeholder DBM, m7 L0/L1, CT L4/L1, native vc modules, fitter patches, surf-SDT)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sf
[ -f $S/setup.ok ] || { echo "setup missing"; exit 2; }
SDT_WORKERS=5 bash $CODE/prep_sf.sh 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
cp $S/native_build.log $S/native_cmake.log $OUT/ 2>/dev/null; git -C $S/villa diff > $OUT/fitter_patch.diff
du -sh $S/* > $OUT/du.txt 2>&1; free -g >> $OUT/du.txt
exit $rc
