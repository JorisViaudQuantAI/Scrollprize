#!/bin/bash
# fetch each finished sf chunk's checkpoint + logs (keep only newest ckpt locally); exits when sf_112 has ended
C=/home/user/vesuvius/colabcli/cjob; R=/home/user/vesuvius/spiral_full/results
while true; do
  st=$($C status 2>/dev/null)
  for k in 06 07 08 09 10 11 12; do
    j=sf_1${k}_acw_c$k; d=$R/sf_1$k
    [ -f $d/.fetched ] && continue
    echo "$st" | grep -qE "(ok|failed|timeout|skipped|cancelled) +[0-9]+s $j" || continue
    mkdir -p $d
    for f in run.txt losses_c$k.txt fit_c$k.log nvsmi.csv; do $C get $j $f $d/$f >/dev/null 2>&1; done
    if $C ls $j | grep -q checkpoint_c$k.ckpt; then $C get $j checkpoint_c$k.ckpt $d/checkpoint_c$k.ckpt >/dev/null 2>&1 && \
       for o in $R/sf_1*/checkpoint_c*.ckpt; do [ "$o" != "$d/checkpoint_c$k.ckpt" ] && rm -f $o; done; fi
    touch $d/.fetched; echo "$(date -u +%H:%M) fetched $j: $(grep -E 'completed_iterations now|schedule already' $d/run.txt | tail -1)"
  done
  echo "$st" | grep -qE "(ok|failed|timeout|skipped|cancelled) +[0-9]+s sf_112_acw_c12" && break
  echo "$st" | grep -qE "(failed|timeout) +[0-9]+s sf_1" && { echo "A CHUNK FAILED"; break; }
  sleep 300
done
