# slot: cpu
# after: sf_112_acw_c12
# timeout: 7200
# sf lane: final gate on the step-30,000 meshes (tag from fitd_last) + mesh tarballs + umbilicus + README
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v2.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sf
T=$(cat $S/fitd_last); echo "final tag $T" | tee $OUT/final_tag.txt
for t in $(ls $S/logs | sed -n 's/^fit_\(c[0-9]*\)\.log$/\1/p'); do tr '\r' '\n' < $S/logs/fit_$t.log | grep -E "^step |satisfied_|resuming|LOOP_SECONDS" > $OUT/losses_$t.txt; done
KEEP=1 bash $CODE/gate.sh $T 2>&1 | tee $OUT/gate.log
cp $S/ds/PHerc0175A/umbilicus.json $OUT/umbilicus_PHerc0175A.json
cat > $OUT/README_meshes.txt <<TXT
PHerc0175A spiral fit (sf lane), sense ACW (assumption, from 306B), band z 6900-8400 of volume 20250521115057 (8.64 um).
villa fit_spiral.py @0fb38c45 (+343 shell-optional, +306B STOP_AT, +loop-time patch), winners' overrides, ONE schedule:
optimizer_num_training_steps 30000, loss_start_track_dt 10000, resumed in time-bounded chunks (tag $T is the last).
Normals: synthetic from m7 (306B make_normals.py), not Lasagna. Umbilicus: own, hand-placed on 16 level-2 slices, unreviewed.
meshes_fitted_$T.tgz -> fitted/wNNN raw fit; meshes_snap_$T.tgz -> snap/wNNN (343 snap_meshes.py, m7 L0).
tifxyz x/y/z.tif float32 level-0 voxel coords, -1 invalid; step 20 vox; rows along z, columns around the umbilicus.
TXT
ls -la $OUT
