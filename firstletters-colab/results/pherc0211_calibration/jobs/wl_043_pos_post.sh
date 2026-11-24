# slot: cpu
# after: wl_042_pos_ink
# timeout: 3600
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=4
for d in rv3_forward rv3_reverse; do si_post pwin $d $C/maps/pwin/rv3; done
for d in ag3_forward ag3_reverse; do si_post pwin $d $C/maps/pwin/ag3; done
for o in asc desc; do si_post pwin st$o $C/maps/pwin/st${o}_s0; done
ls $C/maps/pwin/*/ > $OUT/maps_ls.txt; stamp post
