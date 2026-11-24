# slot: cpu
# timeout: 5400
# depth-sweep stitch arm: 81-layer renders (mesh at 40) of fw035, fw033 (nearest the letter sheet) and fw026 (control)
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for w in w035 w033 w026; do n=d$w; R=$C/renders/$n
  [ -s $R/render_meta.json ] || { OMP_NUM_THREADS=3 python $CODE/render_tifxyz.py --volume $VOL --tifxyz $C/meshes/211/f/$w --out $R --layers 81 --voxel-um 9.362 --surface-interp bicubic --workers 16 --cache-mb 3000 --disk-cache $C/dcache/$(basename $VOL .zarr) 2>&1 | tail -1; }
  python $CODE/onsheet_score.py --layers $R/layers --mid 40 --out $C/renders/$n.onsheet.json --map $C/renders/$n.onsheet_map.npy > /dev/null
  python $CODE/si_tools.py prep $R $C/renders/$n.onsheet_map.npy $C/dst/$n --tag $n --um 9.362 --mid 40 | tee -a $OUT/prep.txt
  cp $R/render_meta.json $OUT/render_meta_$n.json
done
