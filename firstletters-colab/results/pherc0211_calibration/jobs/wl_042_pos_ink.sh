# after: wl_041_pos_render
# timeout: 3000
# winner mesh: reader-v3 both directions -3/+3, argus-v3 both, Studio stride 48 fp16 asc/desc
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
t=$(date +%s); si_rv3 pwin; stamp rv3 $t; t=$(date +%s); wl_ag pwin; stamp ag3 $t
for o in asc desc; do t=$(date +%s); si_st pwin $o 0; stamp st_$o $t; done
