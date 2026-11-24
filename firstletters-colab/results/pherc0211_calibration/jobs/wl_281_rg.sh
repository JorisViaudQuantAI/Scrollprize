# slot: cpu
# after: wl_280_grow
# timeout: 3600
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
for n in gp1 gp2; do si_render $n $VOL /content/cache/wl211/grow/$n/mesh || true; done
