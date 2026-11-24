# slot: cpu
# after: sg_010_data
# timeout: 7200
# sg lane: PHerc1203 band z 9000-10400 data prep stages: sdt pools
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
SDT_WORKERS=4 bash $CODE/prep_sg.sh sdt pools 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
cp $S/native_build.log $S/native_cmake.log $OUT/ 2>/dev/null; git -C $S/villa diff > $OUT/fitter_patch.diff
cp $S/xt/work/worker*.log $OUT/ 2>/dev/null; cp $S/ds/PHerc1203/umbilicus*.json $OUT/ 2>/dev/null; cp $S/ds/PHerc1203/tracks/*.extract_params.json $OUT/ 2>/dev/null
cp $S/ds/PHerc1203/tracks/*.vctracks/metadata.json $OUT/tracks_metadata.json 2>/dev/null
du -sh $S/* > $OUT/du.txt 2>&1; free -g >> $OUT/du.txt
exit $rc
