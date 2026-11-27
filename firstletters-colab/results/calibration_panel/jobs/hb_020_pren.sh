# slot: cpu
# timeout: 5400
# panel renders (awarded scrolls only): 0211 winner mesh, 0211 fw033 (rr re-flattened, 30-mm crop as rr_200), 0306B claim crop, 0343 letter crop
export WLSC=PHerc0211; source $INPUTS/panel_env.sh; source $INPUTS/hb_read.sh
mkdir -p $C/meshes && tar xzf $INPUTS/pcmesh.tgz -C $C/meshes
RWORKERS=24 RTHREADS=6 si_render p211win $VOL $C/meshes/winner; stamp r_p211win
RXARGS="--grid-crop 68 0 160 88" RWORKERS=24 RTHREADS=6 si_render p211fw033 $VOL $C/meshes/w033; stamp r_fw033
UM=8.64; RXARGS="--grid-crop 115 70 235 75" RWORKERS=24 RTHREADS=6 si_render p306 $B/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr $CODE/pos/m306; stamp r_p306
UM=8.64; RXARGS="--grid-crop 40 20 240 76" RWORKERS=24 RTHREADS=6 si_render p343 $B/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr $CODE/pos/m343; stamp r_p343
for n in p211win p211fw033 p306 p343; do echo "$n $(python -c "import json;d=json.load(open('$C/renders/$n/render_meta.json'));print(d.get('voxel_um'), d.get('grid_crop_xywh'), d.get('flip_normals'))")"; done | tee $OUT/rmeta.txt
