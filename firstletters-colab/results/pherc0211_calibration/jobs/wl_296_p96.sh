# slot: cpu
# after: wl_295_s96
# timeout: 3600
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
for n in fw033 fw035 fw012 fw062; do for o in desc asc; do si_post $n st96$o $C/maps/$n/s96$o; done; done
for n in fw012 fw062; do for o in desc asc; do [ -d $C/maps/$n/st${o}_s0 ] || true; done; done
