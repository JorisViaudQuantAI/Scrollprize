# slot: cpu
# after: wl_26a_i
# timeout: 7200
# post (parallel mask) + montages, PHerc0211 f group b1
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
DETS="rv3_forward rv3_reverse"; [ 0 = 1 ] && DETS="$DETS ag3_forward ag3_reverse"
for w in w010 w011 w012 w013 w014 w015 w016 w017 w018 w019 w020 w021 w022 w023 w024 w025 w026 w027 w028 w029 ; do n=f$w; [ -d $C/maps/$n/rv3 ] || continue
  for d in $DETS; do case $d in rv3*) si_post $n $d $C/maps/$n/rv3;; ag3*) si_post $n $d $C/maps/$n/ag3;; esac; done
  cp $C/dst/$n/area.json $OUT/area_$n.json 2>/dev/null; cp $C/renders/$n.onsheet.json $OUT/onsheet_$n.json 2>/dev/null
done
IDS=$(for w in w010 w011 w012 w013 w014 w015 w016 w017 w018 w019 w020 w021 w022 w023 w024 w025 w026 w027 w028 w029 ; do echo -n "PHerc0211_f$w,"; done)
for d in $DETS; do python $CODE/at_atlas.py montage $C/red $OUT/montage_b1_$d.jpg --ids ${IDS%,} --det $d --q 80; done
stamp post
