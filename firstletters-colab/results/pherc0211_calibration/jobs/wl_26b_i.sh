# after: wl_26b_r, wl_26a_i
# timeout: 7000
# reader-v3 both directions -3/+3  on PHerc0211 f group b2
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for w in w030 w031 w032 w033 w034 w035 w036 w037 w038 w039 w040 w041 w042 w043 w044 w045 w046 w047 w048 w049 ; do [ -s $C/renders/f$w/render_meta.json ] || continue; t=$(date +%s); si_rv3 f$w; [ 0 = 1 ] && wl_ag f$w; stamp ink_$w $t; done
