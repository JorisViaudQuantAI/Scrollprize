# slot: cpu
# after: hb_001_setup
# timeout: 5400
# our old render path on the winner mesh (41 layers, bicubic, renderer default normals)
export WLSC=PHerc0846B; source $INPUTS/panel_env.sh
RWORKERS=24 RTHREADS=6 si_render p846 $VOL $C/win/input/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750; stamp render
cat $C/renders/p846/render_meta.json | head -c 1500 > $OUT/render_meta.json
