# slot: cpu
# timeout: 3600
# sg lane: 0175A validation, part 2: merge the already extracted parts (merge dir bug fixed) + comparison with the published store
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v4.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
PY=$(cat $S/spiral_py.txt); export SPIRAL_DIR=$S/villa/volume-cartographer/scripts/spiral; V=$S/val0175
t=$(date +%s); $PY $CODE/xtracks.py merge --work $V/work --workers 11 --out $V/out/PHerc0175A_val_surface_m7_L0_th0.2.dbm 2>&1 | tail -3 | tee $OUT/merge.txt; echo "merge $(( $(date +%s) - t ))s" | tee -a $OUT/merge.txt
cp $V/work/run_params.json $V/out/; cp $V/out/*.vctracks/metadata.json $OUT/mine_metadata.json
$PY $CODE/cmp_tracks.py $V/out/PHerc0175A_val_surface_m7_L0_th0.2.dbm.vctracks $V/pub --z0 7604 --z1 7804 --core0 7654 --core1 7754 --zov 7700 --bits $V/bits.npy --out $OUT --crop 4600 3600 1800 1800 2>&1 | tee $OUT/cmp.txt
$PY $CODE/cmp_tracks.py $V/out/PHerc0175A_val_surface_m7_L0_th0.2.dbm.vctracks $V/pub --z0 7604 --z1 7804 --core0 7654 --core1 7754 --zov 7700 --bits $V/bits.npy --out $OUT/full 2>&1 | tail -1
$PY -c "from PIL import Image; im=Image.open('$OUT/full/overlay_z7700.png'); im.resize((im.size[0]//4, im.size[1]//4)).save('$OUT/overlay_full_z7700_ds4.png')"; rm -rf $OUT/full
