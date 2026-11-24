# slot: cpu
# timeout: 3600
# sg lane: sense chain test, larger sample: 7 levels z 9100-10300, all transverse tracks as starts, both mirrors (crossings sidecar moved aside by sg_013)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v4.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sg
PY=$(cat $S/spiral_py.txt); D=$S/ds/PHerc1203; P=$D/tracks/PHerc1203_20250820131727_surface_m7_L0_th0.2_sg
export SC_TRACKS=$P.dbm.vctracks SC_CROSS=$S/xt/crossings_aside/$(basename $P).dbm.crossings.npz SC_UMB=$D/umbilicus_official.json SC_OUT=$OUT
for z in 9100 9300 9500 9700 9900 10100 10300; do for m in 1 -1; do
  ( $PY $CODE/sense_chain.py $z 64 30000 20 0.3 200 16 1500 $m > $OUT/sense_z${z}_m${m}.json 2> $OUT/sense_z${z}_m${m}.err ) &
done; done; wait
for f in $OUT/sense_z*.json; do echo "$(basename $f) $(cat $f)"; done | tee $OUT/sense_summary.txt
