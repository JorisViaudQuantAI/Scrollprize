# after: wl_912_p2r, wl_901_s96
# timeout: 6000
# Task P2: Studio s96 then s48, BOTH orders (cheap: 12 x <=5 cm2); order chosen at review from normside
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for n in p00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 ; do [ -s $C/renders/$n/render_meta.json ] || continue
  for o in desc asc; do M=$C/maps/$n/s96$o; mkdir -p $M; t=$(date +%s)
    $ST --input $C/renders/$n --mesh-index 20 --order $o --shift 0 --stride 96 --precision fp16 --batch-size 32 --device cuda --out-prefix $M/x 2>&1 | tail -1
    python $CODE/si_tools.py stpng $M/x $M/st96${o}__studio_z+0.png > /dev/null; rm -f $M/x.prob.npy; stamp s96_${n}_$o $t
    t=$(date +%s); si_st $n $o 0; stamp s48_${n}_$o $t; done; done
