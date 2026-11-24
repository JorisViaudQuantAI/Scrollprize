# slot: cpu
# after: wl_915_p2i
# timeout: 3600
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
for n in p00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 ; do for o in desc asc; do [ -d $C/maps/$n/st${o}_s0 ] && si_post $n st$o $C/maps/$n/st${o}_s0; done; cp $C/dst/$n/area.json $OUT/area_$n.json 2>/dev/null; cp $C/renders/$n.onsheet.json $OUT/onsheet_$n.json 2>/dev/null; done
