#!/bin/bash
# flat_set.sh SCR VARIANT(f|s) LIST... : flatboi (keep 100, symmetric Dirichlet, 50 it) each winding of the gate's last tag
# into /content/cache/wlink/meshes/<SCR>/<VARIANT>/wNNN ; stretch before/after to $OUT/stretch_<SCR>_<V>.txt
SCR=$1; V=$2; shift 2
case $SCR in 211) export WLSC=PHerc0211;; 826) export WLSC=PHerc0826;; esac; source $INPUTS/wl_env.sh; set +e +o pipefail
source $INPUTS/vcenv.sh
S=/content/cache/wl$SCR; T=$(cat $S/fitd_last); W=$S/out/$T; [ $V = f ] && SRC=$W/fitted || SRC=$W/snap
D=/content/cache/wlink/meshes/$SCR/$V; mkdir -p $D
for w in "$@"; do
  [ -f $D/$w/x.tif ] && continue; [ -d $SRC/$w ] || { echo "missing $w"; continue; }
  rm -rf $D/$w.tmp; PASTIX_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OPENBLAS_CORETYPE=Haswell vc flatboi $SRC/$w --output=$D/$w.tmp 50 symmetric_dirichlet > $D/$w.flat.log 2>&1
  [ -f $D/$w.tmp/x.tif ] && mv $D/$w.tmp $D/$w || echo "FLATBOI FAILED $w"
  python3 /content/cache/wlink/code_v3/stretch.py $SRC/$w | sed "s/^/raw /" >> $OUT/stretch_${SCR}_$V.txt
  [ -d $D/$w ] && python3 /content/cache/wlink/code_v3/stretch.py $D/$w | sed "s/^/flat /" >> $OUT/stretch_${SCR}_$V.txt
done
cd $D && tar czf $OUT/flat_${SCR}_${V}_$1.tgz $(for w in "$@"; do [ -d $w ] && echo $w; done)
