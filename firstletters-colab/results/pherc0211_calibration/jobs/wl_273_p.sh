# slot: cpu
# after: wl_272_i
# timeout: 7200
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
for n in sw032 sw033 sw034 sw035 sw036 sw037 gw033 gw035; do for d in rv3_forward rv3_reverse; do si_post $n $d $C/maps/$n/rv3; done; cp $C/dst/$n/area.json $OUT/area_$n.json 2>/dev/null; done
for n in fw035 fw033 fw034 fw036 sw035 sw033 gw035 gw033; do for o in asc desc; do [ -d $C/maps/$n/st${o}_s0 ] && si_post $n st$o $C/maps/$n/st${o}_s0; done; done
stamp post
