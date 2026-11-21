# timeout: 4200
# Step 1 (CPU only): download the m7 normal grids around the seed, then headless GrowPatch
# (vc_grow_seg_from_seed from the VC3D AppImage). Parametrised copy of the job that ran on PHerc0175A.
# usage: INPUTS=... OUT=... [SCROLL=... SEED_X=... SEED_Y=... SEED_Z=... VOL_URL=... M7_URL=... NG_KEY=... GENS=80] bash 01_grow.sh
# Writes the mesh path to $OUT/mesh_path.txt and to $MESH_FILE (seed-specific, see env.sh); pass it on explicitly:
#   MESH=$(cat $OUT/mesh_path.txt) bash 02_render_ink.sh
source "$(dirname "$0")/env.sh"
ensure_tools
NG=$C/ngrid/$SCROLL
t=$(date +%s)
# normal grids for seed z +-3400 slices (9.4 GB and 21 min for PHerc0175A; reused by later seeds of the same scroll)
$PY - <<PY
import fl_helpers as H
H.download_normal_grids("$NG_KEY", "$NG", max(0, $SEED_Z - 3400), $SEED_Z + 3400, margin=0, threads=32)
PY
echo "normal grids seconds=$(( $(date +%s)-t ))" | tee -a $OUT/timings.txt; du -sh $NG | tee -a $OUT/timings.txt
G=$C/grow/${SCROLL}_${SEED_X}_${SEED_Y}_${SEED_Z}_g$GENS; mkdir -p $G
$PY - <<PY
import fl_helpers as H
print(H.write_grow_params("$G/params.json", "$NG", $UM, generations=$GENS, threads=2))
PY
if ! ls -d $G/auto_grown_* >/dev/null 2>&1; then
  vc vc_grow_seg_from_seed -v $M7_URL -t $G -p $G/params.json --seed $SEED_X $SEED_Y $SEED_Z 2>&1 | tail -60
fi
M=$(ls -d $G/auto_grown_* | tail -1); echo MESH $M
mkdir -p $OUT/mesh && cp $M/*.tif $M/*.json $OUT/mesh/ 2>/dev/null; ls -la $OUT/mesh
$PY - <<PY
import fl_helpers as H, json
print(json.dumps(H.tifxyz_stats("$M", None, $UM), indent=1))
PY
mkdir -p "$(dirname "$MESH_FILE")"; echo "$M" > "$MESH_FILE"; echo "$M" > $OUT/mesh_path.txt; echo "mesh path written to $MESH_FILE and $OUT/mesh_path.txt"
df -h /content | tail -1
