#!/bin/bash
# mk_read.sh BASE SCR V GROUP AFTER "wNNN wNNN ..." [AG=0] : jobs BASE_fl (cpu flatten), BASE_r (cpu render), BASE_i (gpu rv3 [+argus]), BASE_p (cpu post+montage)
B=$1; SCR=$2; V=$3; G=$4; AFTER=$5; WL=$6; AG=${7:-0}
case $SCR in 211) SC=PHerc0211;; 826) SC=PHerc0826;; esac
J=/home/user/vesuvius/wl/jobs
cat > $J/${B}_fl.sh <<JOB
# slot: cpu
# after: $AFTER
# timeout: 3600
# flatten (flatboi keep 100) $SC variant $V group $G: $WL
bash \$INPUTS/flat_set.sh $SCR $V $WL
JOB
cat > $J/${B}_r.sh <<JOB
# slot: cpu
# after: ${B}_fl
# timeout: 10800
# render $SC $V group $G (41 layers, bicubic, mesh at 20) + onsheet + dst
export WLSC=$SC; source \$INPUTS/wl_env.sh; export RWORKERS=16
for w in $WL; do [ -d \$C/meshes/$SCR/$V/\$w ] || continue; t=\$(date +%s); si_render $V\$w \$VOL \$C/meshes/$SCR/$V/\$w || true; stamp render_\$w \$t; done
df -h /content | tail -1 | tee -a \$OUT/timings.txt
JOB
cat > $J/${B}_i.sh <<JOB
# after: ${B}_r
# timeout: 7000
# reader-v3 both directions -3/+3 $( [ $AG = 1 ] && echo "+ argus-v3" ) on $SC $V group $G
export WLSC=$SC; source \$INPUTS/wl_env.sh
for w in $WL; do [ -s \$C/renders/$V\$w/render_meta.json ] || continue; t=\$(date +%s); si_rv3 $V\$w; [ $AG = 1 ] && wl_ag $V\$w; stamp ink_\$w \$t; done
JOB
cat > $J/${B}_p.sh <<JOB
# slot: cpu
# after: ${B}_i
# timeout: 7200
# post (parallel mask) + montages, $SC $V group $G
export WLSC=$SC; source \$INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
DETS="rv3_forward rv3_reverse"; [ $AG = 1 ] && DETS="\$DETS ag3_forward ag3_reverse"
for w in $WL; do n=$V\$w; [ -d \$C/maps/\$n/rv3 ] || continue
  for d in \$DETS; do case \$d in rv3*) si_post \$n \$d \$C/maps/\$n/rv3;; ag3*) si_post \$n \$d \$C/maps/\$n/ag3;; esac; done
  cp \$C/dst/\$n/area.json \$OUT/area_\$n.json 2>/dev/null; cp \$C/renders/\$n.onsheet.json \$OUT/onsheet_\$n.json 2>/dev/null
done
IDS=\$(for w in $WL; do echo -n "${SC}_$V\$w,"; done)
for d in \$DETS; do python \$CODE/at_atlas.py montage \$C/red \$OUT/montage_${G}_\$d.jpg --ids \${IDS%,} --det \$d --q 80; done
stamp post
JOB
echo "$J/${B}_{fl,r,i,p}.sh"
