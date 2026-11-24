# slot: cpu
# after: wl_26a_fl
# timeout: 10800
# render PHerc0211 f group b1 (41 layers, bicubic, mesh at 20) + onsheet + dst
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
for w in w010 w011 w012 w013 w014 w015 w016 w017 w018 w019 w020 w021 w022 w023 w024 w025 w026 w027 w028 w029 ; do [ -d $C/meshes/211/f/$w ] || continue; t=$(date +%s); si_render f$w $VOL $C/meshes/211/f/$w || true; stamp render_$w $t; done
df -h /content | tail -1 | tee -a $OUT/timings.txt
