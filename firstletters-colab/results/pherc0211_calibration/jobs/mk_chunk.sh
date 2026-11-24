#!/bin/bash
# mk_chunk.sh K AFTER [LOOP_S] : writes jobs/wl_1KK_acw_cKK.sh (PHerc0211, ACW), one time-bounded chunk of the 30,000-step schedule
K=$(printf %02d $1); AFTER=$2; LOOP=${3:-2100}; TGZ=${4:-wl_code_v2.tgz}
cat > /home/user/vesuvius/wl/jobs/wl_2${K}_acw_c${K}.sh <<JOB
# after: $AFTER
# timeout: 2700
# wl lane: PHerc0211 z 7650-9150 (official tracks/Lasagna/umbilicus), ACW (catalogue flags = 0175A's), ONE 30,000-step schedule (track-DT from 10,000), chunk c$K (loop <= ${LOOP}s)
set -u; mkdir -p \$PWD/code && tar -xzf \$INPUTS/$TGZ -C \$PWD/code; export CODE=\$PWD/code
export S=/content/cache/wl211 SC=PHerc0211 VID=20250821151803 Z0=7650 Z1=9150
PY=\$(cat \$S/spiral_py.txt)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > \$OUT/nvsmi.csv 2>&1 & NV=\$!
( while true; do free -m | awk 'NR==2{print strftime("%H:%M:%S"), \$3}' >> \$OUT/ram_used_mb.txt; sleep 30; done ) & RM=\$!
LOOP_S=$LOOP bash \$CODE/run_chunk.sh ACW c$K 30000 2>&1 | tee \$OUT/run.txt; rc=\${PIPESTATUS[0]}
kill \$NV \$RM; awk -F', ' 'NR>1{gsub(/ MiB/,"",\$2); if(\$2+0>m)m=\$2+0} END{print "nvidia-smi peak used MiB:", m}' \$OUT/nvsmi.csv | tee -a \$OUT/run.txt
RD=\$(ls -d \$S/fits/*_c$K 2>/dev/null | head -1)
if [ -n "\$RD" ]; then
  tr '\r' '\n' < \$S/logs/fit_c$K.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS|it/s|Error|error|tracks|crossing|parameters|GiB" > \$OUT/losses_c$K.txt
  tail -c 300000 \$S/logs/fit_c$K.log > \$OUT/fit_c$K.log; ls -la \$RD \$RD/meshes/* 2>/dev/null | head -60 > \$OUT/rundir.txt
  [ -f \$RD/checkpoint_fitted.ckpt ] && cp \$RD/checkpoint_fitted.ckpt \$OUT/checkpoint_c$K.ckpt
  for o in \$(ls -dt \$S/fits/*_c* | tail -n +3); do rm -f \$o/checkpoint_fitted.ckpt; done
fi
df -h /content | tail -1 >> \$OUT/run.txt
exit \$rc
JOB
echo jobs/wl_2${K}_acw_c${K}.sh
