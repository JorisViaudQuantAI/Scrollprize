# slot: cpu
# after: wl_26c_fl
# timeout: 10800
# render PHerc0211 f group b3 (41 layers, bicubic, mesh at 20) + onsheet + dst
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
for w in w050 w051 w052 w053 w054 w055 w056 w057 w058 w059 w060 w061 w062 w063 w064 w065 w066 w067 w068 w069 ; do [ -d $C/meshes/211/f/$w ] || continue; t=$(date +%s); si_render f$w $VOL $C/meshes/211/f/$w || true; stamp render_$w $t; done
df -h /content | tail -1 | tee -a $OUT/timings.txt
