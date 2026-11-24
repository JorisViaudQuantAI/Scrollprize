# after: wl_912_p2r, wl_901_s96
# timeout: 5400
# Task P2: Studio s48 in the reading order from normside (towards-umbilicus < 0.3 -> desc, > 0.7 -> asc, else both)
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
NS=/content/work/jobs/wl_912_p2r/normside.txt; [ -f $NS ] || NS=$(ls /content/bridge/results/wl_912_p2r/normside.txt 2>/dev/null)
python $CODE/normside.py /content/cache/wl211/ds/PHerc0211/umbilicus_official.json $(for n in p00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 ; do echo $C/meshes/211/p2/$n; done) > $OUT/normside.txt 2>&1
while read path _ _ frac _ _; do n=$(basename $path); [ -s $C/renders/$n/render_meta.json ] || continue
  O=$(python -c "f=float('$frac'); print('desc' if f<0.3 else ('asc' if f>0.7 else 'desc asc'))")
  for o in $O; do t=$(date +%s); si_st $n $o 0; stamp s48_${n}_$o $t; done; done < $OUT/normside.txt
