# after: wl_26a_r
# timeout: 7000
# reader-v3 both directions -3/+3  on PHerc0211 f group b1
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for w in w010 w011 w012 w013 w014 w015 w016 w017 w018 w019 w020 w021 w022 w023 w024 w025 w026 w027 w028 w029 ; do [ -s $C/renders/f$w/render_meta.json ] || continue; t=$(date +%s); si_rv3 f$w; [ 0 = 1 ] && wl_ag f$w; stamp ink_$w $t; done
