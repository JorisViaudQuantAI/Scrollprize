# slot: cpu
# timeout: 7200
# sg lane (retry 2, +vesuvius/src on PYTHONPATH): villa flatten_spiral_checkpoint.py (pinned 0fb38c45) on the PHerc1203 step-30,000 checkpoint, CPU (the GPU runs the 0846A fit):
# Lasagna flatten_fast_nofilter with args device=cpu, compile/fused-triton off (speed-only switches); then cut per winding + spacing stats.
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v6.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
PY=$(cat $S/spiral_py.txt); V=$S/villa/volume-cartographer/scripts/spiral; L=$S/lasagna_cpu
CK=$(ls -d $S/fits/*_c10 | head -1)/checkpoint_fitted.ckpt; ls -la $CK | tee $OUT/ckpt.txt
rm -rf $L; cp -r $S/villa/lasagna $L
$PY - <<PY
import json; p='$L/configs/flatten_fast_nofilter.json'; c=json.load(open(p)); c['args']['device']='cpu'
for st in c['stages']: st['args'].update(compile_flatten=False, compile_flatten_combined=False, fused_flatten_adam_clamp=False)
json.dump(c,open(p,'w'),indent=1); print('config patched', c['args'])
PY
cp $L/configs/flatten_fast_nofilter.json $OUT/
(cd $L && PYTHONPATH=$L:$S/villa/vesuvius/src $PY -c 'import fit_service, fit, vc3d_fiber_format; print("imports ok")') 2>&1 | tail -3 | tee $OUT/import_check.txt
F=$S/flat/PHerc1203_c10_flat.tifxyz; mkdir -p $S/flat; rm -rf $F
t=$(date +%s)
( cd $V && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=$S/native_pkg:$L:$S/villa/vesuvius/src timeout 5400 $PY flatten_spiral_checkpoint.py $CK $F --umbilicus $S/ds/PHerc1203/umbilicus.json --lasagna-dir $L --device cpu --voxel-size-um 9.362 ) > $OUT/flatten.log 2>&1
echo "flatten rc=$? $(( $(date +%s) - t ))s" | tee -a $OUT/flatten.log; tail -c 3000 $OUT/flatten.log > $OUT/flatten_tail.log
[ -f $F/meta.json ] || exit 3
cat $F/meta.json > $OUT/flat_meta.json
$PY $CODE/flat_split.py $F $S/out/c10/fitted $S/flat/c10_windings --windings w018,w025,w026,w024,w015,w021,w029,w044,w016,w027,w046,w033,w051,w032,w063,w071,w037 2>&1 | tee $OUT/split.txt
cp $S/flat/c10_windings/spacing.tsv $OUT/; tar -C $S/flat -czf $OUT/flat_c10_windings.tgz c10_windings; tar -C $S/flat -czf $OUT/flat_c10_combined.tgz PHerc1203_c10_flat.tifxyz
