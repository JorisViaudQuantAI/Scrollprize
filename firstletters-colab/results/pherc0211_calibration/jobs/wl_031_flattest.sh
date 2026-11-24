# slot: cpu
# timeout: 1800
# wl lane: test flatboi (VC3D AppImage) on the winner mesh and one non-isometric HF 0826 winding
set -u; source $INPUTS/vcenv.sh; mkdir -p m && tar xzf $INPUTS/wl_ft_meshes.tgz -C m
vc flatboi --help > $OUT/flatboi_help.txt 2>&1; vc vc_render_tifxyz --help > $OUT/render_help.txt 2>&1
for n in winner h826w040; do
  for k in 100 25; do
    PASTIX_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OPENBLAS_CORETYPE=Haswell vc flatboi m/$n --output=$PWD/f_${n}_k$k 50 symmetric_dirichlet --keep=$k > $OUT/flat_${n}_k$k.log 2>&1
    ls -la f_${n}_k$k >> $OUT/flat_${n}_k$k.log 2>&1; mkdir -p $OUT/f_${n}_k$k; cp f_${n}_k$k/* $OUT/f_${n}_k$k/ 2>/dev/null
  done
done
