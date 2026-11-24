# slot: cpu
# after: wl_26b_i
# timeout: 7200
# post (parallel mask) + montages, PHerc0211 f group b2
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
DETS="rv3_forward rv3_reverse"; [ 0 = 1 ] && DETS="$DETS ag3_forward ag3_reverse"
for w in w030 w031 w032 w033 w034 w035 w036 w037 w038 w039 w040 w041 w042 w043 w044 w045 w046 w047 w048 w049 ; do n=f$w; [ -d $C/maps/$n/rv3 ] || continue
  for d in $DETS; do case $d in rv3*) si_post $n $d $C/maps/$n/rv3;; ag3*) si_post $n $d $C/maps/$n/ag3;; esac; done
  cp $C/dst/$n/area.json $OUT/area_$n.json 2>/dev/null; cp $C/renders/$n.onsheet.json $OUT/onsheet_$n.json 2>/dev/null
done
IDS=$(for w in w030 w031 w032 w033 w034 w035 w036 w037 w038 w039 w040 w041 w042 w043 w044 w045 w046 w047 w048 w049 ; do echo -n "PHerc0211_f$w,"; done)
for d in $DETS; do python $CODE/at_atlas.py montage $C/red $OUT/montage_b2_$d.jpg --ids ${IDS%,} --det $d --q 80; done
stamp post
