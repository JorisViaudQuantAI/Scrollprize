# slot: cpu
# after: sg_301_846_data
# timeout: 10800
# sg lane, PHerc0846A: the published store covers only z >= 4500 (extract.json z_min 4500), so OWN tracks for z 2386-3814 with the
# pinned extractor + published parameters (as validated on 0175A, report 23); spec re-pointed to them; crossings sidecar + sense chain test.
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/s846_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/s846; G=/content/cache/sg
PY=$(cat $G/spiral_py.txt); export SPIRAL_DIR=$G/villa/volume-cartographer/scripts/spiral; D=$S/ds/PHerc0846A; W=$S/xt
P=PHerc0846A_20250728152254_surface_m7_L0_th0.2_sg; mkdir -p $W $D/tracks_own
t=$(date +%s)
[ -f $W/bits.npy.done ] || { $PY $CODE/xtracks.py pack --m7 $S/m7 --z0 2386 --z1 3814 --bits $W/bits.npy --procs 4 | tail -2 && touch $W/bits.npy.done; }
$PY $CODE/xtracks.py run --bits $W/bits.npy --work $W/work --workers 11 --kpar 1 2>&1 | tail -3 | tee $OUT/run.txt
$PY $CODE/xtracks.py merge --work $W/work --workers 11 --out $D/tracks_own/$P.dbm 2>&1 | tail -2 | tee -a $OUT/run.txt
echo "tracks $(( $(date +%s) - t ))s" | tee -a $OUT/run.txt
cp $W/work/run_params.json $D/tracks_own/$P.extract_params.json; cp $W/work/run_params.json $OUT/; cp $D/tracks_own/$P.dbm.vctracks/metadata.json $OUT/tracks_metadata.json
cp $W/work/worker*.log $OUT/ 2>/dev/null
# re-point both sense specs to our own store
for sense in CW ACW; do DD=$S/ds/PHerc0846A_$sense; ln -sfn $D/tracks_own $DD/tracks; sed -i "s#tracks/[A-Za-z0-9_.]*\.dbm#tracks/$P.dbm#" $DD/spiral-scroll.json; cat $DD/spiral-scroll.json; done
# crossings sidecar (for the sense test only; then moved aside so the fitter computes its own, as on 0175A/1203)
(cd $SPIRAL_DIR && PYTHONPATH=$G/native_pkg $PY build_track_crossings.py $D/tracks_own/$P.dbm --workers 10) 2>&1 | tail -2 | tee $OUT/crossings.log
mkdir -p $W/crossings_aside; mv $D/tracks_own/$P.dbm.crossings.npz $W/crossings_aside/
export SC_TRACKS=$D/tracks_own/$P.dbm.vctracks SC_CROSS=$W/crossings_aside/$P.dbm.crossings.npz SC_UMB=$D/umbilicus.json SC_OUT=$OUT
for z in 2550 2750 2950 3150 3350 3550; do for m in 1 -1; do
  ( $PY $CODE/sense_chain.py $z 64 30000 20 0.3 200 16 1500 $m > $OUT/sense_z${z}_m${m}.json 2> $OUT/sense_z${z}_m${m}.err ) &
done; done; wait
for f in $OUT/sense_z*.json; do echo "$(basename $f) $(cat $f)"; done > $OUT/sense_summary.txt
ls -la $D/tracks_own; touch $S/prep/tracks_own.ok
