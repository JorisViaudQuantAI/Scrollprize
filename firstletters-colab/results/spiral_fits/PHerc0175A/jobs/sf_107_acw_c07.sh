# after: sf_106_acw_c06
# timeout: 2700
# sf lane: PHerc0175A z 6900-8400, ACW, ONE 30,000-step schedule (track-DT from 10,000), chunk c07 (loop <= 2100s); saves checkpoint + logs to OUT
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sf
PY=$(cat $S/spiral_py.txt)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > $OUT/nvsmi.csv 2>&1 & NV=$!
free -g > $OUT/free_start.txt
( while true; do free -m | awk 'NR==2{print strftime("%H:%M:%S"), $3}' >> $OUT/ram_used_mb.txt; sleep 30; done ) & RM=$!
SC=PHerc0175A Z0=6900 Z1=8400 LOOP_S=2100 bash $CODE/run_chunk.sh ACW c07 30000 2>&1 | tee $OUT/run.txt; rc=${PIPESTATUS[0]}
kill $NV $RM; awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2+0>m)m=$2+0} END{print "nvidia-smi peak used MiB:", m}' $OUT/nvsmi.csv | tee -a $OUT/run.txt
RD=$(ls -d $S/fits/*_c07 2>/dev/null | head -1)
if [ -n "$RD" ]; then
  tr '\r' '\n' < $S/logs/fit_c07.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS|it/s|Error|error" > $OUT/losses_c07.txt
  tail -c 300000 $S/logs/fit_c07.log > $OUT/fit_c07.log; ls -la $RD $RD/meshes/* 2>/dev/null | head -60 > $OUT/rundir.txt
  [ -f $RD/checkpoint_fitted.ckpt ] && cp $RD/checkpoint_fitted.ckpt $OUT/checkpoint_c07.ckpt
  # keep only the newest two run dirs' checkpoints on the VM (meshes stay)
fi
df -h /content | tail -1 >> $OUT/run.txt
exit $rc
