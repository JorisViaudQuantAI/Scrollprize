# after: sg_108_cwb_c08
# timeout: 2700
# sg lane: PHerc1203 z 9000-10400 (OWN tracks), CW, ONE 30,000-step schedule (track-DT from 10,000), chunk c09 (loop <= 1980s); saves checkpoint + logs to OUT
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v5.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
PY=$(cat $S/spiral_py.txt)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > $OUT/nvsmi.csv 2>&1 & NV=$!
( while true; do free -m | awk 'NR==2{print strftime("%H:%M:%S"), $3}' >> $OUT/ram_used_mb.txt; sleep 30; done ) & RM=$!
SC=PHerc1203 Z0=9000 Z1=10400 LOOP_S=1980 bash $CODE/run_chunk.sh CW c09 30000 2>&1 | tee $OUT/run.txt; rc=${PIPESTATUS[0]}
kill $NV $RM; awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2+0>m)m=$2+0} END{print "nvidia-smi peak used MiB:", m}' $OUT/nvsmi.csv | tee -a $OUT/run.txt
RD=$(ls -d $S/fits/*_c09 2>/dev/null | head -1)
if [ -n "$RD" ]; then
  tr '\r' '\n' < $S/logs/fit_c09.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS|it/s|Error|error|tracks|crossing|parameters|GiB" > $OUT/losses_c09.txt
  tail -c 300000 $S/logs/fit_c09.log > $OUT/fit_c09.log; ls -la $RD $RD/meshes/* 2>/dev/null | head -60 > $OUT/rundir.txt
  [ -f $RD/checkpoint_fitted.ckpt ] && cp $RD/checkpoint_fitted.ckpt $OUT/checkpoint_c09.ckpt
  # free VM disk: drop checkpoints of run dirs older than the previous one (meshes stay)
  for o in $(ls -dt $S/fits/*_c* | tail -n +3); do rm -f $o/checkpoint_fitted.ckpt; done
fi
df -h /content | tail -1 >> $OUT/run.txt
exit $rc
