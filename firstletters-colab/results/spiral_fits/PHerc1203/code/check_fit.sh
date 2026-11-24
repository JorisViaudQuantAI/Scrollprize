#!/bin/bash
# check_fit.sh TAG [N] [SNAP=1] : snap windings onto m7 (343 snap_meshes.py), on-sheet check (fitted and snapped), cross-sections
set -uo pipefail
S=/content/cache/sg; TAG=$1; N=${2:-200}; SNAP=${SNAP:-1}; CODE=${CODE:?}
PY=$(cat $S/spiral_py.txt); RD=$(ls -d $S/fits/*_$TAG | head -1); MD=$RD/meshes/fitted_$TAG
[ -d $MD ] || { echo "no meshes for $TAG"; exit 2; }
W=$S/out/$TAG; mkdir -p $W/fitted $W/snap $W/check
for d in $MD/w[0-9][0-9][0-9]_$TAG; do [ -f $d/x.tif ] && ln -sfn $d $W/fitted/$(basename $d _$TAG); done
ls $W/fitted | sort > $W/windings.txt; [ -n "${WSEL:-}" ] && { set -- $WSEL; awk -v a=$1 -v b=$2 '{n=substr($0,2,3)+0; if(n>=a&&n<=b) print}' $W/windings.txt > $W/w.tmp; mv $W/w.tmp $W/windings.txt; rm -rf $W/fitted_all; mkdir -p $W/fitted_all; mv $W/fitted/* $W/fitted_all/; for w in $(cat $W/windings.txt); do mv $W/fitted_all/$w $W/fitted/; done; }; echo "windings: $(wc -l < $W/windings.txt)"
$PY - <<PY > $OUT/areas_$TAG.tsv
import json,glob,os
t=0
for d in sorted(glob.glob('$W/fitted/w*')):
    m=json.load(open(d+'/meta.json')); t+=m['area_cm2']; print(os.path.basename(d), round(m['area_cm2'],3), m['bbox'], sep='\t')
print('TOTAL', round(t,2), sep='\t')
PY
tail -1 $OUT/areas_$TAG.tsv
if [ "$SNAP" = 1 ] && [ ! -f $W/snap/.done ]; then
  t0=$(date +%s)
  SURF_ZARR=$S/m7 $PY $CODE/snap_meshes.py $W/fitted PHerc1203 $W/snap --windings $W/windings.txt --z0 ${Z0:-9000} --z1 ${Z1:-10400} --level 0 --procs 6 > $W/snap.log 2>&1 && touch $W/snap/.done
  echo "snap rc=$? $(( $(date +%s) - t0 ))s"; tail -5 $W/snap.log; cp $W/snap.log $OUT/snap_$TAG.log; cp $W/snap/snap_report.tsv $OUT/snap_report_$TAG.tsv 2>/dev/null
fi
for v in fitted snap; do
  ls $W/$v/w* >/dev/null 2>&1 || continue
  t0=$(date +%s)
  $PY $CODE/oncheck.py $W/$v $W/check/$v --ct $S/ct --ct-level 1 --m7 $S/m7 --m7-level 0 --umb $S/ds/PHerc1203/umbilicus.json --n $N > $W/check/oncheck_$v.log 2>&1
  echo "oncheck $v rc=$? $(( $(date +%s) - t0 ))s"; cp $W/check/$v/oncheck.tsv $OUT/oncheck_${TAG}_$v.tsv; tail -3 $W/check/oncheck_$v.log
done
for v in fitted snap; do
  ls $W/$v/w* >/dev/null 2>&1 || continue
  $PY $CODE/xsect.py $W/$v $OUT --ct $S/ct --level 1 --z ${XZ:-9300 9700 10100} --tag _${TAG}_$v 2>&1 | tail -3 || true
done
# zoomed crops (full-res 1600 x 1600 around three points on a ring between umbilicus and edge), level 1
for v in fitted snap; do
  ls $W/$v/w* >/dev/null 2>&1 || continue
  $PY $CODE/xsect.py $W/$v $OUT --ct $S/ct --level 1 --z 9700 --crop ${CROP:-2950 2100 1600 1600} --tag _${TAG}_${v}_c 2>&1 | tail -1 || true
done
# keep the meshes: snapped windings (real dirs) and the fitted ones (dereferenced) as tarballs in the results
[ "${KEEP:-0}" = 1 ] && { tar -C $W -czhf $OUT/meshes_fitted_$TAG.tgz fitted; [ -d $W/snap ] && tar -C $W -czf $OUT/meshes_snap_$TAG.tgz snap; du -sh $OUT/meshes_*_$TAG.tgz; }
exit 0
