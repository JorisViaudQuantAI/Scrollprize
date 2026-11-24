# slot: cpu
# after: wl_285_ig
# timeout: 3600
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
for n in gp1c gp2c; do for d in rv3_forward rv3_reverse; do si_post $n $d $C/maps/$n/rv3; done; for o in asc desc; do si_post $n st$o $C/maps/$n/st${o}_s0; done; cp $C/dst/$n/area.json $OUT/area_$n.json 2>/dev/null; done
