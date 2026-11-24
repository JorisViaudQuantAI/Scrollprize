#!/bin/bash
# mk_chunk.sh K AFTER SENSE [LOOP_S] [CODE_TGZ] : writes jobs/sg_1KK_<sense>_cKK.sh, one time-bounded chunk of the 30,000-step schedule
K=$(printf %02d $1); AFTER=$2; SENSE=$3; LOOP=${4:-2100}; TGZ=${5:-sg_code_v5.tgz}; s=${6:-$(echo $SENSE | tr A-Z a-z)}
cat > /home/user/vesuvius/spiral_1203/jobs/sg_1${K}_${s}_c${K}.sh <<JOB
# after: $AFTER
# timeout: 2700
# sg lane: PHerc1203 z 9000-10400 (OWN tracks), $SENSE, ONE 30,000-step schedule (track-DT from 10,000), chunk c$K (loop <= ${LOOP}s); saves checkpoint + logs to OUT
set -u; mkdir -p \$PWD/code && tar -xzf \$INPUTS/$TGZ -C \$PWD/code; export CODE=\$PWD/code; S=/content/cache/sg
PY=\$(cat \$S/spiral_py.txt)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > \$OUT/nvsmi.csv 2>&1 & NV=\$!
( while true; do free -m | awk 'NR==2{print strftime("%H:%M:%S"), \$3}' >> \$OUT/ram_used_mb.txt; sleep 30; done ) & RM=\$!
SC=PHerc1203 Z0=9000 Z1=10400 LOOP_S=$LOOP bash \$CODE/run_chunk.sh $SENSE c$K 30000 2>&1 | tee \$OUT/run.txt; rc=\${PIPESTATUS[0]}
kill \$NV \$RM; awk -F', ' 'NR>1{gsub(/ MiB/,"",\$2); if(\$2+0>m)m=\$2+0} END{print "nvidia-smi peak used MiB:", m}' \$OUT/nvsmi.csv | tee -a \$OUT/run.txt
RD=\$(ls -d \$S/fits/*_c$K 2>/dev/null | head -1)
if [ -n "\$RD" ]; then
  tr '\r' '\n' < \$S/logs/fit_c$K.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS|it/s|Error|error|tracks|crossing|parameters|GiB" > \$OUT/losses_c$K.txt
  tail -c 300000 \$S/logs/fit_c$K.log > \$OUT/fit_c$K.log; ls -la \$RD \$RD/meshes/* 2>/dev/null | head -60 > \$OUT/rundir.txt
  [ -f \$RD/checkpoint_fitted.ckpt ] && cp \$RD/checkpoint_fitted.ckpt \$OUT/checkpoint_c$K.ckpt
  # free VM disk: drop checkpoints of run dirs older than the previous one (meshes stay)
  for o in \$(ls -dt \$S/fits/*_c* | tail -n +3); do rm -f \$o/checkpoint_fitted.ckpt; done
fi
df -h /content | tail -1 >> \$OUT/run.txt
exit \$rc
JOB
echo jobs/sg_1${K}_${s}_c${K}.sh
