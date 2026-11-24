# slot: cpu
# after: sg_301_846_data
# timeout: 3600
# sg lane, PHerc0846A: winding sense from the data (report 15 s7.2 chain test) on the PUBLISHED tracks + crossings, own umbilicus, x-mirror control
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/s846_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/s846
PY=$(cat $S/spiral_py.txt); D=$S/ds/PHerc0846A; P=PHerc0846A_20250728152254_surface_m7_L0_th0.2
export SC_TRACKS=$D/tracks/$P.dbm.vctracks SC_CROSS=$S/$P.dbm.crossings.npz SC_UMB=$D/umbilicus.json SC_OUT=$OUT
for z in 2550 2750 2950 3150 3350 3550; do for m in 1 -1; do
  ( $PY $CODE/sense_chain.py $z 64 30000 20 0.3 200 16 1500 $m > $OUT/sense_z${z}_m${m}.json 2> $OUT/sense_z${z}_m${m}.err ) &
done; done; wait
for f in $OUT/sense_z*.json; do echo "$(basename $f) $(cat $f)"; done | tee $OUT/sense_summary.txt
