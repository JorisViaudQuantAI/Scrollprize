# slot: cpu
# timeout: 3600
# GrowPatch patches cropped to z 7550-9300 within 1800 vox of the seed (gp1c, gp2c): render
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
mkdir -p $C/meshes/211/gp && tar xzf $INPUTS/wl_gpc.tgz -C $C/meshes/211/gp
for n in gp1c gp2c; do si_render $n $VOL $C/meshes/211/gp/$n || true; done
