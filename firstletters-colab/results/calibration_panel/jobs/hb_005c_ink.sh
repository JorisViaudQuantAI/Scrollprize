# after: hb_004c_win
# timeout: 7200
# (b) our old rv3 path; B0 z0/no-resample; C our infer on their surface zarr; Studio s48/s96 both orders; compare
export WLSC=PHerc0846B; source $INPUTS/panel_env.sh
t=$(date +%s); si_rv3 p846; stamp rv3_old $t
M=$C/maps/p846/b0; mkdir -p $M; t=$(date +%s)
$INF $C/renders/p846 --input-um 9.362 --surface-layer 20 -c $RV3 -o $M/b0.png --direction both --resample never --save-each --save-float --batch-size 16 --threads 6 --no-display 2>&1 | grep -E "tiles in|Error" | tail -3; stamp b0 $t
M=$C/maps/p846/cw; mkdir -p $M; t=$(date +%s)
$INF $C/win/work/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750/surface_volume.zarr --input-um 9.362 --surface-layer 8 -c $RV3 -o $M/cw.png --mask $C/win/work/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750/mask.tif --direction both --resample never --save-each --save-float --batch-size 16 --threads 6 --no-display 2>&1 | grep -E "tiles in|Error" | tail -3; stamp cw $t
for o in asc desc; do t=$(date +%s); si_st p846 $o 0; stamp st48_$o $t; done
for o in desc asc; do M=$C/maps/p846/s96$o; mkdir -p $M; t=$(date +%s)
  $ST --input $C/renders/p846 --mesh-index 20 --order $o --shift 0 --stride 96 --precision fp16 --batch-size 32 --device cuda --out-prefix $M/x 2>&1 | tail -1
  python $CODE/si_tools.py stpng $M/x $M/st96${o}__studio_z+0.png; rm -f $M/x.prob.npy; stamp s96_$o $t; done
for d in rv3 b0 cw; do ls $C/maps/p846/$d; done > $OUT/maplist.txt
python $INPUTS/cmp.py $C/win/win_raw.tif $OUT/cmp $C/win/output/*_raw.tif $C/villa_our_forward.tif $C/villa_our_reverse.tif $(ls $C/maps/p846/rv3/*.png $C/maps/p846/b0/*.npy $C/maps/p846/b0/*.png $C/maps/p846/cw/*.npy $C/maps/p846/cw/*.png $C/maps/p846/st*/st*__studio_z+0.png $C/maps/p846/s96*/st96*png 2>/dev/null) 2>&1 | tee $OUT/cmp.txt
for d in asc desc; do si_post p846 st$d $C/maps/p846/st${d}_s0; done
stamp all
