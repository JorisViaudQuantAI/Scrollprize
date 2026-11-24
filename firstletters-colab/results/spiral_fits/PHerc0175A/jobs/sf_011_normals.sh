# after: sf_010_data
# timeout: 2400
# sf lane: synthetic normals from m7 (306B make_normals.py, GPU), resident pools (sdt,normals), dataset spec + own umbilicus
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/sf
Z0=6900; Z1=8400; SC=PHerc0175A; PY=$(cat $S/spiral_py.txt); V=$S/villa/volume-cartographer/scripts/spiral; D=$S/ds/$SC
[ -f $S/prep/CPU.ok ] || { echo "cpu prep missing"; exit 2; }
t0=$(date +%s)
if [ ! -f $D/lasagna_inputs/.syn_done ]; then
  $PY $CODE/make_normals.py $S/m7 $D/lasagna_inputs --z $((Z0-256)) $((Z1+256)) --tile 512 --device cuda > $OUT/make_normals.log 2>&1 && touch $D/lasagna_inputs/.syn_done
  echo "make_normals rc=$? $(( $(date +%s) - t0 ))s"; tail -3 $OUT/make_normals.log
fi
[ -f $D/lasagna_inputs/.syn_done ] || exit 3
du -sh $D/lasagna_inputs/* | tee $OUT/las_du.txt
# quick sanity of the normals: fraction of valid voxels and mean |nz| in the band
$PY - <<PY | tee $OUT/normals_stats.txt
import zarr, numpy as np
nx=zarr.open('$D/lasagna_inputs/las_008_nx.ome.zarr',mode='r')['2']; ny=zarr.open('$D/lasagna_inputs/las_008_ny.ome.zarr',mode='r')['2']
z=(($Z0+$Z1)//2)//4; a=np.asarray(nx[z]).astype(float); b=np.asarray(ny[z]).astype(float)
v=(a>0)|(b>0); n1=(a[v]-128)/127; n2=(b[v]-128)/127
print('shape',nx.shape,'slice z2',z,'valid frac',v.mean().round(3),'median |nx,ny|',np.median(np.hypot(n1,n2)).round(3))
PY
t1=$(date +%s)
(cd $V && $PY pack_resident_pools.py $D/lasagna_inputs --what sdt,normals --sdt-group 1 --normal-group 2 --io-threads 8) 2>&1 | tail -15 | tee $OUT/pools.log
rc=${PIPESTATUS[0]}; echo "pools rc=$rc $(( $(date +%s) - t1 ))s"; [ $rc = 0 ] || exit 4
cp $INPUTS/umbilicus_PHerc0175A.json $D/umbilicus.json
for sense in CW ACW; do
  DD=$S/ds/${SC}_$sense; mkdir -p $DD
  for f in tracks lasagna_inputs umbilicus.json; do ln -sfn $D/$f $DD/$f; done
  cat > $DD/spiral-scroll.json <<JS
{"schema_version": 1, "name": "$SC", "voxel_size_um": 8.64, "spiral_outward_sense": "$sense",
 "umbilicus": {"coordinate_scale": 1.0}, "normal_zarr_group": "2", "surf_sdt_zarr_group": "1", "lasagna_scale": 4,
 "paths": {"tracks_dbm": "tracks/${SC}_20250521115057_surface_m7_L0_th0.2.dbm"}}
JS
done
ls -la $D $D/lasagna_inputs | tee $OUT/ds_ls.txt
touch $S/prep/ALL.ok; echo "ALL ok"; df -h /content | tail -1
