# after: wl_711_acw_c11
# timeout: 2700
# wl lane: PHerc0826 z 11800-13300 (official inputs), ACW (catalogue flags = 0175A s), ONE 30,000-step schedule (track-DT from 10,000), chunk c12 (loop <= 2100s)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/wl_code_v2.tgz -C $PWD/code; export CODE=$PWD/code
export S=/content/cache/wl826 SC=PHerc0826 VID=20250821151701 Z0=11800 Z1=13300
PY=$(cat $S/spiral_py.txt)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > $OUT/nvsmi.csv 2>&1 & NV=$!
( while true; do free -m | awk 'NR==2{print strftime("%H:%M:%S"), $3}' >> $OUT/ram_used_mb.txt; sleep 30; done ) & RM=$!
LOOP_S=2100 bash $CODE/run_chunk.sh ACW c12 30000 2>&1 | tee $OUT/run.txt; rc=${PIPESTATUS[0]}
kill $NV $RM; awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2+0>m)m=$2+0} END{print "nvidia-smi peak used MiB:", m}' $OUT/nvsmi.csv | tee -a $OUT/run.txt
RD=$(ls -d $S/fits/*_c12 2>/dev/null | head -1)
if [ -n "$RD" ]; then
  tr '\r' '\n' < $S/logs/fit_c12.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS|it/s|Error|error|tracks|crossing|parameters|GiB" > $OUT/losses_c12.txt
  tail -c 300000 $S/logs/fit_c12.log > $OUT/fit_c12.log; ls -la $RD $RD/meshes/* 2>/dev/null | head -60 > $OUT/rundir.txt
  [ -f $RD/checkpoint_fitted.ckpt ] && cp $RD/checkpoint_fitted.ckpt $OUT/checkpoint_c12.ckpt
  for o in $(ls -dt $S/fits/*_c* | tail -n +3); do rm -f $o/checkpoint_fitted.ckpt; done
fi
df -h /content | tail -1 >> $OUT/run.txt
exit $rc
