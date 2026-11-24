# slot: cpu
# after: sg_011_tracks
# timeout: 5400
# sg lane: winding sense of PHerc1203 from the data (report 15 s7.2 chain test) on OUR tracks: crossings sidecar (pinned builder), chains at 3 levels, x-mirror control
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v3.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
PY=$(cat $S/spiral_py.txt); V=$S/villa/volume-cartographer/scripts/spiral; D=$S/ds/PHerc1203; P=$D/tracks/PHerc1203_20250820131727_surface_m7_L0_th0.2_sg
t=$(date +%s); (cd $V && PYTHONPATH=$S/native_pkg $PY build_track_crossings.py $P.dbm --workers 10) 2>&1 | tail -5 | tee $OUT/crossings.log; echo "crossings $(( $(date +%s) - t ))s" | tee -a $OUT/crossings.log
ls -la $D/tracks | tee -a $OUT/crossings.log
export SC_TRACKS=$P.dbm.vctracks SC_CROSS=$P.dbm.crossings.npz SC_UMB=$D/umbilicus_official.json SC_OUT=$OUT
for z in 9300 9700 10100; do for m in 1 -1; do
  ( $PY $CODE/sense_chain.py $z 64 4000 20 0.3 200 16 1500 $m > $OUT/sense_z${z}_m${m}.json 2> $OUT/sense_z${z}_m${m}.err ) &
done; done; wait
for f in $OUT/sense_z*.json; do echo "$(basename $f) $(cat $f)"; done | tee $OUT/sense_summary.txt
# keep the crossings sidecar out of the fitter's way (the fitter computes exact crossings itself, as on 0175A)
mkdir -p $S/xt/crossings_aside; mv $P.dbm.crossings.npz $S/xt/crossings_aside/ 2>/dev/null; ls -la $D/tracks
