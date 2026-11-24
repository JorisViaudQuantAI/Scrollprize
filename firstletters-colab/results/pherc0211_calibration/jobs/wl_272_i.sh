# after: wl_271_rS
# timeout: 6000
# Task 1 arms: reader-v3 both on sw032-037, gw033, gw035; Studio s48 asc+desc on fw033-036, sw033, sw035, gw033, gw035
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
for n in sw032 sw033 sw034 sw035 sw036 sw037 gw033 gw035; do [ -s $C/renders/$n/render_meta.json ] && si_rv3 $n; done
for n in fw035 fw033 fw034 fw036 sw035 sw033 gw035 gw033; do [ -s $C/renders/$n/render_meta.json ] || continue; for o in desc asc; do t=$(date +%s); si_st $n $o 0; stamp st_${n}_$o $t; done; done
