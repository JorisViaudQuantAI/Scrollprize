# after: wl_284_rg
# timeout: 4000
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for n in gp1c gp2c; do [ -s $C/renders/$n/render_meta.json ] || continue; si_rv3 $n; for o in asc desc; do si_st $n $o 0; done; done
