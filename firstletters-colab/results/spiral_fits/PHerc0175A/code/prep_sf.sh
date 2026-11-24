#!/bin/bash
# prep_sf.sh : PHerc0175A band dataset for the pinned (0fb38c45) spiral fitter (306B/343 recipe; 306B fallbacks:
# synthetic normals from m7 = make_normals.py, run separately on the GPU; own umbilicus copied in at spec time).
# Idempotent: each stage leaves $S/prep/<stage>.ok
set -uo pipefail
S=/content/cache/sf; CODE=${CODE:?}; Z0=${Z0:-6900}; Z1=${Z1:-8400}
SC=PHerc0175A; VID=20250521115057; UM=8.64
V=$S/villa/volume-cartographer/scripts/spiral; PY=$(cat $S/spiral_py.txt)
D=$S/ds/$SC; M=$S/m7; C=$S/ct; mkdir -p $D/tracks $D/lasagna_inputs $M $C $S/prep
BK=s3://vesuvius-challenge-open-data/$SC
MS=$BK/representations/predictions/surfaces/${VID}-surface-20260413222639-surface-m7-L0-th0.2.zarr
CS=$BK/volumes/${VID}-8.640um-1.2m-116keV-masked.zarr
T=https://dl.ash2txt.org/datasets/spiral_datasets/$SC/$VID/tracks; P=${SC}_${VID}_surface_m7_L0_th0.2
T0=$(date +%s); log() { echo "$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s $*"; }
stage() { [ -f $S/prep/$1.ok ] && { log "skip $1"; return 0; }; log "start $1"; local t=$(date +%s); "$2" && { touch $S/prep/$1.ok; log "done $1 $(( $(date +%s) - t ))s"; } || { log "FAILED $1"; exit 1; }; }
command -v aws >/dev/null || pip install -q awscli
s3cp() { aws s3 cp --no-sign-request --only-show-errors "$@"; }
s3sync() { aws s3 sync --no-sign-request --only-show-errors "$@"; }
# level-1 chunk index range covering working z [Z0-800, Z1+800] (enough for SDT tiles and normals halo)
l1range() { python3 -c "print(max(0,(($Z0-800)//2)//$1), (($Z1+800)//2)//$1)"; }

s_tracks() {
  cd $D/tracks; mkdir -p $P.dbm.vctracks
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
  # completeness check: all 9 files at server size
  for f in arclengths.f64 coordinates.i32 family_codes.i8 header.bin metadata.json offsets.i64 source_ids.u64 tortuosities.f64 z_bounds.i32; do
    want=$(curl -sI $T/$P.dbm.vctracks/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
    [ "$(stat -c %s $P.dbm.vctracks/$f)" = "$want" ] || { echo "INCOMPLETE $f"; return 1; }
  done
}
s_m7() {
  for f in .zattrs .zgroup 0/.zarray 1/.zarray; do s3cp $MS/$f $M/$f; done
  for zi in $(seq $((Z0/192-1)) $((Z1/192+1))); do s3sync $MS/0/$zi $M/0/$zi & done; wait
  read A B < <(l1range 192); for zi in $(seq $A $B); do s3sync $MS/1/$zi $M/1/$zi & done; wait; du -sh $M
}
s_ct() {
  for f in .zattrs .zgroup 1/.zarray 4/.zarray; do s3cp $CS/$f $C/$f; done
  s3sync $CS/4 $C/4
  read A B < <(l1range 128); for zi in $(seq $A $B); do s3sync $CS/1/$zi $C/1/$zi & done; wait; du -sh $C
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
s_sdt() {
  local OUTZ=$D/lasagna_inputs/las_008_surf_sdt.ome.zarr
  (cd $V && $PY make_surf_sdt.py --surf $M --group 1 --out $OUTZ --threshold 128 --working-voxel-um $UM \
     --z-range $((Z0-64)):$((Z1+64)) --ct $C --ct-group 4 --ct-dilate 1 --workers ${SDT_WORKERS:-5} --edt-threads 2 $([ -d $OUTZ ] && echo --resume)) 2>&1 | tail -15
  [ ${PIPESTATUS[0]} = 0 ] && du -sh $OUTZ
}
stage tracks s_tracks
stage m7 s_m7
stage ct s_ct
stage native s_native
stage patch s_patch
stage sdt s_sdt
touch $S/prep/CPU.ok; log "cpu prep complete"; df -h /content | tail -1
