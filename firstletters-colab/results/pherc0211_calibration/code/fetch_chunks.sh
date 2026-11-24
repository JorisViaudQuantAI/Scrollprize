#!/bin/bash
# fetch_chunks.sh : one pass: download every finished wl chunk's logs + checkpoint (keeps only the newest checkpoint locally)
export COLAB_SESSION=g3; C=/home/user/vesuvius/colabcli/cjob; R=/home/user/vesuvius/wl/results
st=$($C status 2>/dev/null); echo "$st" | head -2
for j in $(echo "$st" | grep -oE "(ok|failed|timeout|cancelled) +[0-9]+s wl_[2-589][0-9][0-9]_[a-z]+_c[0-9]+" | awk '{print $3}' | sort); do
  k=${j##*_c}; d=$R/${j}; [ -f $d/.fetched ] && continue; mkdir -p $d
  for f in run.txt losses_c$k.txt fit_c$k.log nvsmi.csv ram_used_mb.txt rundir.txt DONE.txt; do $C get $j $f $d/$f >/dev/null 2>&1; done
  if $C ls $j | grep -q checkpoint_c$k.ckpt; then $C get $j checkpoint_c$k.ckpt $d/checkpoint_c$k.ckpt >/dev/null 2>&1 && true; \
     true; fi  # keep every checkpoint (coordinator 24 Nov; ~650 MB each)
  touch $d/.fetched; echo "$(date -u +%H:%M) fetched $j: $(grep -E 'completed_iterations now|schedule already|peak' $d/run.txt | tr '\n' ' ')" | tee -a $R/fetch_chunks.log
done
