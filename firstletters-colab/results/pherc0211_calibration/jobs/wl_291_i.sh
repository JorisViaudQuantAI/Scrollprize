# after: wl_290_r81
# timeout: 3600
# depth sweep: reader-v3 REVERSE (= reading order on our windings), shifts -15..+15 step 3, per-shift maps saved
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for n in dw035 dw033 dw026; do M=$C/maps/$n/sweep; mkdir -p $M
  $INF $C/renders/$n --input-um 9.362 --surface-layer 40 -c $RV3 -o $M/sw.png --direction reverse --z-shifts=-15,-12,-9,-6,-3,0,3,6,9,12,15 --save-each --batch-size 16 --threads 6 --no-display 2>&1 | tail -2
  ls $M | head -3; cd $M && tar cf $OUT/sweep_$n.tar *__*_z*.png && cd - >/dev/null
done
