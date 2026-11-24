# slot: cpu
# after: wl_270_flS
# timeout: 7200
# Task 1 arms: render snapped-flattened w032-w037 (sw*) and the RAW FIT GRID as is (unflattened) w033, w035 (gw*)
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
for w in w032 w033 w034 w035 w036 w037; do si_render s$w $VOL $C/meshes/211/s/$w || true; done
G=/content/cache/wl211/out/c10/fitted
for w in w033 w035; do si_render g$w $VOL $G/$w || true; done
