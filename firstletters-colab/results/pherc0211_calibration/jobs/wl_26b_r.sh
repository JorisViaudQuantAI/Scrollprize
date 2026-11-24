# slot: cpu
# after: wl_26b_fl
# timeout: 10800
# render PHerc0211 f group b2 (41 layers, bicubic, mesh at 20) + onsheet + dst
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
for w in w030 w031 w032 w033 w034 w035 w036 w037 w038 w039 w040 w041 w042 w043 w044 w045 w046 w047 w048 w049 ; do [ -d $C/meshes/211/f/$w ] || continue; t=$(date +%s); si_render f$w $VOL $C/meshes/211/f/$w || true; stamp render_$w $t; done
df -h /content | tail -1 | tee -a $OUT/timings.txt
