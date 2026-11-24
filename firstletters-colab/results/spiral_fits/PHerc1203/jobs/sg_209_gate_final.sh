# slot: cpu
# after: sg_111_cwb_c11
# timeout: 7200
# sg lane: final gate on the step-30,000 meshes (tag from fitd_last) + mesh tarballs + umbilicus + README
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v5.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
T=$(cat $S/fitd_last); echo "final tag $T" | tee $OUT/final_tag.txt
for t in $(ls $S/logs | sed -n 's/^fit_\(c[0-9]*\)\.log$/\1/p'); do tr '\r' '\n' < $S/logs/fit_$t.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS" > $OUT/losses_$t.txt; done
KEEP=1 bash $CODE/gate.sh $T 2>&1 | tee $OUT/gate.log
cp $S/ds/PHerc1203/umbilicus.json $OUT/umbilicus_PHerc1203_smoothed.json; cp $S/ds/PHerc1203/umbilicus_official.json $OUT/umbilicus_PHerc1203_official.json
cp $S/ds/PHerc1203/tracks/*.extract_params.json $OUT/
cat > $OUT/README_meshes.txt <<TXT
PHerc1203 spiral fit (sg lane), sense CW (from a track-chain test on our tracks, moderate support), band z 9000-10400 of volume
20250820131727 (9.362 um). villa fit_spiral.py @0fb38c45 (+343 shell-optional, +306B STOP_AT, +loop-time patch), winners' overrides,
ONE schedule: optimizer_num_training_steps 30000, loss_start_track_dt 10000, resumed in time-bounded chunks (tag $T is the last).
TRACKS ARE OUR OWN (pinned extract_surface_tracks.py with the published stores' parameters; validated against the published 0175A
store), not published ones. Normals/grad_mag: published Lasagna g2. Umbilicus: official, pre-smoothed sigma 75 (306B smooth_umb.py).
meshes_fitted_$T.tgz -> fitted/wNNN raw fit; meshes_snap_$T.tgz -> snap/wNNN (343 snap_meshes.py, m7 L0).
tifxyz x/y/z.tif float32 level-0 voxel coords, -1 invalid; step 20 vox; rows along z, columns around the umbilicus.
TXT
ls -la $OUT
