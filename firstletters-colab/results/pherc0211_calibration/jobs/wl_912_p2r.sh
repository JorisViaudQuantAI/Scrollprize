# slot: cpu
# after: wl_911_p2
# timeout: 5400
# Task P2: render the 12 cropped+flattened GrowPatch windows; normside; gridq
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
for n in p00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 ; do [ -d $C/meshes/211/p2/$n ] && si_render $n $VOL $C/meshes/211/p2/$n || true; done
python $CODE/normside.py /content/cache/wl211/ds/PHerc0211/umbilicus_official.json $(for n in p00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 ; do echo $C/meshes/211/p2/$n; done) > $OUT/normside.txt 2>&1
python $CODE/gridq.py $(for n in p00 p01 p02 p03 p04 p05 p06 p07 p08 p09 p10 p11 ; do echo $C/meshes/211/p2/$n; done) --json $OUT/gridq_p2.json > /dev/null 2>&1
cat $OUT/normside.txt
