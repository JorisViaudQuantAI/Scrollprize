# after: wl_940_gp30
# timeout: 1500
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for o in desc asc; do t=$(date +%s); si_st gp30 $o 0; stamp s48_$o $t; done
