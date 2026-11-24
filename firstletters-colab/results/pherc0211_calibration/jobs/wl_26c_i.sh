# after: wl_26c_r, wl_26b_i
# timeout: 7000
# reader-v3 both directions -3/+3  on PHerc0211 f group b3
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for w in w050 w051 w052 w053 w054 w055 w056 w057 w058 w059 w060 w061 w062 w063 w064 w065 w066 w067 w068 w069 ; do [ -s $C/renders/f$w/render_meta.json ] || continue; t=$(date +%s); si_rv3 f$w; [ 0 = 1 ] && wl_ag f$w; stamp ink_$w $t; done
