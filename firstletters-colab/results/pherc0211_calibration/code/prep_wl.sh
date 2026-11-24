#!/bin/bash
# prep_wl.sh STAGES... : band dataset for the pinned (0fb38c45) spiral fitter, 343/306B recipe as used on 0358 (report 15)
# with ALL-OFFICIAL inputs: published track store (dl.ash2txt.org spiral_datasets), published Lasagna nx/ny/grad_mag g2,
# published m7 L0/L1, official umbilicus (pre-smoothed sigma 75 by 306B smooth_umb.py).
# Env: S (cache dir), SC, VID, UMBF (official umbilicus file name), Z0, Z1. Idempotent: $S/prep/<stage>.ok
# Stages: tracks m7 ct lasagna native patch spec sdt pools
set -uo pipefail
S=${S:?}; CODE=${CODE:?}; Z0=${Z0:?}; Z1=${Z1:?}; SC=${SC:?}; VID=${VID:?}; UMBF=${UMBF:?}; UM=9.362
V=$S/villa/volume-cartographer/scripts/spiral; PY=$(cat $S/spiral_py.txt); export SPIRAL_DIR=$V
D=$S/ds/$SC; M=$S/m7; C=$S/ct; mkdir -p $D/tracks $D/lasagna_inputs $M $C $S/prep
BK=s3://vesuvius-challenge-open-data/$SC
MS=$BK/representations/predictions/surfaces/${VID}-surface-20260413222639-surface-m7-L0-th0.2.zarr
CS=$BK/volumes/${VID}-9.362um-1.2m-113keV-masked.zarr
LS=$BK/representations/predictions/lasagna/${VID}-lasagna-20260419180421
T=https://dl.ash2txt.org/datasets/spiral_datasets/$SC/$VID/tracks; P=${SC}_${VID}_surface_m7_L0_th0.2
T0=$(date +%s); log() { echo "$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s $*"; }
stage() { [ -f $S/prep/$1.ok ] && { log "skip $1"; return 0; }; log "start $1"; local t=$(date +%s); "$2" && { touch $S/prep/$1.ok; log "done $1 $(( $(date +%s) - t ))s"; } || { log "FAILED $1"; exit 1; }; }
for i in 1 2 3; do command -v aws >/dev/null && break; pip install -q awscli; sleep 5; done
command -v aws >/dev/null || { echo "aws cli missing"; exit 1; }
s3cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }
s3sync() { aws s3 sync --no-sign-request --only-show-errors "$@"; }

s_tracks() {
  cd $D/tracks; mkdir -p $P.dbm.vctracks
  curl -sf -o $P.extract.json $T/$P.extract.json
  for f in arclengths.f64 coordinates.i32 family_codes.i8 header.bin metadata.json offsets.i64 source_ids.u64 tortuosities.f64 z_bounds.i32; do
    ( want=$(curl -sI $T/$P.dbm.vctracks/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
      for t in 1 2 3 4 5 6 7 8; do [ -f $P.dbm.vctracks/$f ] && [ "$(stat -c %s $P.dbm.vctracks/$f)" = "$want" ] && exit 0
        curl -sf --retry 3 -C - -o $P.dbm.vctracks/$f $T/$P.dbm.vctracks/$f || sleep $((t*5)); done; exit 1 ) &
  done
  wait
  for f in $P.dbm.vctracks/*; do echo "$(stat -c %s $f) $f"; done
  $PY - <<PY
import json, os, urllib.request
m = json.load(open('$P.dbm.vctracks/metadata.json')); name, size, mt = m['source_db_signature'][0]
if not os.path.exists(name) or os.path.getsize(name) != size:
    head = urllib.request.urlopen(urllib.request.Request('$T/' + name, headers={'Range': 'bytes=0-65535'})).read()
    with open(name, 'wb') as f: f.write(head); f.truncate(size)
os.utime(name, ns=(mt, mt)); print('placeholder', name, size)
PY
  for f in arclengths.f64 coordinates.i32 family_codes.i8 header.bin metadata.json offsets.i64 source_ids.u64 tortuosities.f64 z_bounds.i32; do
    want=$(curl -sI $T/$P.dbm.vctracks/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
    [ "$(stat -c %s $P.dbm.vctracks/$f)" = "$want" ] || { echo "INCOMPLETE $f"; return 1; }
  done
  cd - >/dev/null
}
s_m7() {
  for f in .zattrs .zgroup 0/.zarray 1/.zarray; do s3cp $MS/$f $M/$f; done
  for zi in $(seq $((Z0/192-1)) $((Z1/192+1))); do s3sync $MS/0/$zi $M/0/$zi & done; wait
  read A B < <(python3 -c "print(max(0,((($Z0-64)//2)//256*256-64)//192), ((($Z1+64)//2+255)//256*256+64)//192)")
  for zi in $(seq $A $B); do s3sync $MS/1/$zi $M/1/$zi & done; wait; du -sh $M
  [ $(du -sm $M | cut -f1) -gt 200 ]
}
s_ct() {
  for f in .zattrs .zgroup 1/.zarray 4/.zarray; do s3cp $CS/$f $C/$f; done
  s3sync $CS/4 $C/4
  read A B < <(python3 -c "print(max(0,(($Z0-800)//2)//128), ((($Z1+800)//2)//128))")
  for zi in $(seq $A $B); do s3sync $CS/1/$zi $C/1/$zi & done; wait; du -sh $C
  [ $(du -sm $C | cut -f1) -gt 500 ]
}
s_lasagna() {
  for ch in nx ny grad_mag; do
    src=$LS/${SC}_${ch}.ome.zarr; dst=$D/lasagna_inputs/las_008_${ch}.ome.zarr; mkdir -p $dst/2
    for f in .zattrs .zgroup 2/.zarray; do s3cp $src/$f $dst/$f; done
    for zi in $(seq $((Z0/128-1)) $((Z1/128+1))); do s3sync $src/2/$zi $dst/2/$zi & done; wait
  done; du -sh $D/lasagna_inputs; cat $D/lasagna_inputs/las_008_nx.ome.zarr/2/.zarray
}
s_native() {
  uv pip install -q --python $PY nanobind cmake ninja || return 1
  local B=$(dirname $PY)
  PATH=$B:$PATH cmake -S $CODE/native -B $S/native_build -G Ninja -DPython_EXECUTABLE=$PY -DPKG_OUT=$S/native_pkg > $S/native_cmake.log 2>&1 || { tail -20 $S/native_cmake.log; return 1; }
  PATH=$B:$PATH cmake --build $S/native_build -j 8 > $S/native_build.log 2>&1 || { tail -30 $S/native_build.log; return 1; }
  echo '"""standalone build of the three nanobind modules used by the 0fb38c45 spiral fitter"""' > $S/native_pkg/vc/__init__.py
  PYTHONPATH=$S/native_pkg $PY -c "import vc.track_crossings, vc.track_store, vc.spiral_sampling; print('native ok')"
}
s_patch() {
  grep -q 'flpipe: shell optional' $V/fit_spiral.py || sed -i 's#filter_tracks_by_shell = bool(self.tracks_dbm_path) and bool(self.shell_path)$#filter_tracks_by_shell = bool(self.tracks_dbm_path) and bool(self.shell_path) and os.path.isdir(self.shell_path)  \# flpipe: shell optional#' $V/fit_spiral.py
  grep -q 'flpipe: shell optional' $V/fit_spiral.py || return 1
  $PY $CODE/patch_fitter.py $V && $PY $CODE/patch_deadline.py $V && git -C $S/villa diff --stat
}
s_spec() {
  s3cp $BK/representations/umbilicus/$UMBF $D/umbilicus_official.json || return 1
  $PY $CODE/smooth_umb.py $D/umbilicus_official.json $D/umbilicus.json --sigma 75 --step 25 || return 1
  for sense in CW ACW; do
    DD=$S/ds/${SC}_$sense; mkdir -p $DD
    for f in tracks lasagna_inputs umbilicus.json; do ln -sfn $D/$f $DD/$f; done
    cat > $DD/spiral-scroll.json <<JS
{"schema_version": 1, "name": "$SC", "voxel_size_um": $UM, "spiral_outward_sense": "$sense",
 "umbilicus": {"coordinate_scale": 1.0}, "normal_zarr_group": "2", "surf_sdt_zarr_group": "1", "lasagna_scale": 4,
 "paths": {"tracks_dbm": "tracks/$P.dbm"}}
JS
  done
}
s_sdt() {
  local OUTZ=$D/lasagna_inputs/las_008_surf_sdt.ome.zarr
  (cd $V && $PY make_surf_sdt.py --surf $M --group 1 --out $OUTZ --threshold 128 --working-voxel-um $UM \
     --z-range $((Z0-64)):$((Z1+64)) --ct $C --ct-group 4 --ct-dilate 1 --workers ${SDT_WORKERS:-5} --edt-threads 2 $([ -d $OUTZ ] && echo --resume)) 2>&1 | tail -15
  [ ${PIPESTATUS[0]} = 0 ] && du -sh $OUTZ
}
s_pools() { (cd $V && $PY pack_resident_pools.py $D/lasagna_inputs --what sdt,normals,grad_mag --sdt-group 1 --normal-group 2 --io-threads 8) 2>&1 | tail -20; ls -la $D/lasagna_inputs; du -sh $D/lasagna_inputs/*; }
for st in "$@"; do stage $st s_$st; done
log "stages done: $*"; df -h /content | tail -1
