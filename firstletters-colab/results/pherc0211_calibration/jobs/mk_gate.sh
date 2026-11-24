#!/bin/bash
# mk_gate.sh NAME AFTER SCROLL KEEP : gate job on the tag in fitd_last (S per scroll)
N=$1; AFTER=$2; SCR=$3; KEEP=${4:-0}
if [ $SCR = 211 ]; then E='export S=/content/cache/wl211 SC=PHerc0211 VID=20250821151803 Z0=7650 Z1=9150 XZ="7900 8400 8900" XZC=8400 CROP="2750 2431 2000 2000"'
else E='export S=/content/cache/wl826 SC=PHerc0826 VID=20250821151701 Z0=11800 Z1=13300 XZ="12000 12550 13100" XZC=12550 CROP="4000 2900 1800 2400"'; fi
cat > /home/user/vesuvius/wl/jobs/$N.sh <<JOB
# slot: cpu
# after: $AFTER
# timeout: 7200
# wl lane: on-sheet gate (raw vs +-8 vox control, snapped), xsects, renders, KEEP=$KEEP mesh tarballs
set -u; mkdir -p \$PWD/code && tar -xzf \$INPUTS/wl_code_v2.tgz -C \$PWD/code; export CODE=\$PWD/code; $E
T=\$(cat \$S/fitd_last); echo "tag \$T" | tee \$OUT/final_tag.txt
KEEP=$KEEP bash \$CODE/gate.sh \$T 2>&1 | tail -60 > \$OUT/gate.log
cp \$S/ds/\$SC/umbilicus.json \$OUT/umbilicus_smoothed.json
ls -la \$OUT > \$OUT/ls.txt
JOB
echo $N.sh
