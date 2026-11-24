# slot: cpu
# after: wl_040_inksetup
# timeout: 3600
# positive/reference: winner mesh PHerc0211 20261028201525 (published, isometric) through the wl render path
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
si_render pwin $VOL $C/meshes/winner; stamp render
