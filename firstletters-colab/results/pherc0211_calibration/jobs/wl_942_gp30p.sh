# slot: cpu
# after: wl_941_gp30i
# timeout: 1800
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=4
for o in desc asc; do si_post gp30 st$o $C/maps/gp30/st${o}_s0; done; cp $C/dst/gp30/area.json $C/renders/gp30.onsheet.json $OUT/ 2>/dev/null
