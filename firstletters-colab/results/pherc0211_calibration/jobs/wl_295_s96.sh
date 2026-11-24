# after: wl_290_r81
# timeout: 2400
# Studio stride 96 UNBLENDED (screening setting) both orders on fw033, fw035 (letter windings) and far controls fw012, fw062
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for n in fw033 fw035 fw012 fw062; do for o in desc asc; do M=$C/maps/$n/s96${o}; mkdir -p $M
  [ -s $M/st96${o}__studio_z+0.png ] && continue; t=$(date +%s)
  $ST --input $C/renders/$n --mesh-index 20 --order $o --shift 0 --stride 96 --precision fp16 --batch-size 32 --device cuda --out-prefix $M/x 2>&1 | tail -1
  python $CODE/si_tools.py stpng $M/x $M/st96${o}__studio_z+0.png; rm -f $M/x.prob.npy; stamp s96_${n}_$o $t; done; done
