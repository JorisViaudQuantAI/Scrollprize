# slot: cpu
# after: sg_001_setup
# timeout: 6000
# sg lane: validate our track extraction (pinned extract_surface_tracks.py, published parameters) on PHerc0175A z 7604-7804 vs the published store
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
PY=$(cat $S/spiral_py.txt); export SPIRAL_DIR=$S/villa/volume-cartographer/scripts/spiral
$PY -c "import cc3d, kimimaro, networkx, zarr; print('deps ok', kimimaro.__version__ if hasattr(kimimaro,'__version__') else '')" 2>&1 | tee $OUT/deps.txt
command -v aws >/dev/null || pip install -q awscli
V=$S/val0175; mkdir -p $V/m7 $V/pub; T0=$(date +%s)
MS=s3://vesuvius-challenge-open-data/PHerc0175A/representations/predictions/surfaces/20250521115057-surface-20260413222639-surface-m7-L0-th0.2.zarr
TP=https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0175A/20250521115057/tracks/PHerc0175A_20250521115057_surface_m7_L0_th0.2.dbm.vctracks
# published store in the background
( cd $V/pub; for f in offsets.i64 z_bounds.i32 family_codes.i8 arclengths.f64 source_ids.u64 metadata.json header.bin coordinates.i32; do curl -sf --retry 5 -C - -o $f $TP/$f; done; echo "pub dl done $(( $(date +%s) - T0 ))s" > $V/pub/.done ) &
for f in .zattrs .zgroup 0/.zarray; do aws s3 cp --no-sign-request --only-show-errors $MS/$f $V/m7/$f; done
for zi in 39 40; do aws s3 sync --no-sign-request --only-show-errors $MS/0/$zi $V/m7/0/$zi & done; wait %2 %3 2>/dev/null; wait $(jobs -p | tail -2) 2>/dev/null
echo "m7 dl $(( $(date +%s) - T0 ))s $(du -sh $V/m7 | cut -f1)" | tee $OUT/timing.txt
t=$(date +%s); $PY $CODE/xtracks.py pack --m7 $V/m7 --z0 7604 --z1 7804 --bits $V/bits.npy --procs 8 2>&1 | tail -2 | tee -a $OUT/timing.txt
echo "pack $(( $(date +%s) - t ))s" | tee -a $OUT/timing.txt
t=$(date +%s); rm -rf $V/work; $PY $CODE/xtracks.py run --bits $V/bits.npy --work $V/work --workers 11 --kpar 1 --lo-y 384 --lo-x 0 --hi-y 9362 --hi-x 9362 > $OUT/run.log 2>&1; echo "extract rc=$? $(( $(date +%s) - t ))s" | tee -a $OUT/timing.txt
cat $V/work/worker*.log | tail -30 > $OUT/workers_tail.txt
t=$(date +%s); $PY $CODE/xtracks.py merge --work $V/work --workers 11 --out $V/out/PHerc0175A_val_surface_m7_L0_th0.2.dbm 2>&1 | tail -3 | tee -a $OUT/timing.txt; echo "merge $(( $(date +%s) - t ))s" | tee -a $OUT/timing.txt
cp $V/work/run_params.json $V/out/ 2>/dev/null; cp $V/out/*.vctracks/metadata.json $OUT/mine_metadata.json
while [ ! -f $V/pub/.done ]; do sleep 20; done; cat $V/pub/.done | tee -a $OUT/timing.txt; ls -la $V/pub > $OUT/pub_ls.txt
$PY $CODE/cmp_tracks.py $V/out/PHerc0175A_val_surface_m7_L0_th0.2.dbm.vctracks $V/pub --z0 7604 --z1 7804 --core0 7654 --core1 7754 --zov 7700 --bits $V/bits.npy --out $OUT --crop 4600 3600 1800 1800 2>&1 | tee $OUT/cmp.txt
$PY $CODE/cmp_tracks.py $V/out/PHerc0175A_val_surface_m7_L0_th0.2.dbm.vctracks $V/pub --z0 7604 --z1 7804 --core0 7654 --core1 7754 --zov 7700 --bits $V/bits.npy --out $OUT/full 2>&1 | tail -1
$PY -c "from PIL import Image; im=Image.open('$OUT/full/overlay_z7700.png'); im.resize((im.size[0]//4, im.size[1]//4)).save('$OUT/overlay_full_z7700_ds4.png')"; rm -rf $OUT/full
df -h /content | tail -1
