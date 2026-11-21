# timeout: 3000
# Step 2 (GPU): render 41 layers with vc_render_tifxyz, run reader-v3 and ink_9um (seeds 42+43) at
# depth shifts -3..+1 in both layer orders, combine with the PHerc. 343 rule (mean of per-model z-scores),
# write small previews. Parametrised copy of the job that ran on PHerc0175A (T4: render 4 min, reader-v3 7 min,
# ink_9um x2 16 min, previews 1 min).
# Hecate is not used here: inside infer_ink.py with bf16 autocast it ran at ~12 cm2/h on a T4; with
# --hecate-amp fp16 it runs at ~70 cm2/h (see README), still ~10x slower than the two models below.
# usage: INPUTS=... OUT=... [MESH=<tifxyz dir>, default: the mesh written by 01_grow.sh] bash 02_render_ink.sh
source "$(dirname "$0")/env.sh"
ensure_tools; ensure_models
RV3=$C/models/domenicor046/reader-v3/reader-v3.pth
I42=$C/models/scrollprize/ink_9um/hybrid_3d2d-seed42/step-075000.pth
I43=$C/models/scrollprize/ink_9um/hybrid_3d2d-seed43/step-075000.pth
M=${MESH:-$(cat $C/last_mesh_${SCROLL}.txt)}
[ -f $M/x.tif ] || { echo "no mesh at $M"; exit 1; }
R=$C/render/$(basename $M)_L41.zarr
if [ ! -f $R/0/.zarray ]; then
  vc vc_render_tifxyz --volume $VOL_URL --group-idx 0 --scale 1 --segmentation $M --num-slices 41 --slice-step 1 --cache-gb 6 --zarr-output $R 2>&1 | grep -v "^ tile" | tail -5
fi
I=$C/ink/$(basename $M); mkdir -p $I; gpumon
SH="-3,-2,-1,0,1"
for spec in "readerv3:$RV3" "ink9:$I42 $I43"; do
  tag=${spec%%:*}; cks=${spec#*:}; t=$(date +%s)
  [ -f $I/${tag}_reverse.png ] || python $C/code/infer_ink.py $R --model ink9um -c $cks -o $I/$tag.png --input-um $UM \
      --surface-layer 20 --z-shifts=$SH --direction both --save-each --batch-size 8 --device cuda 2>&1 | grep -v -i warn | tail -6
  echo "$tag seconds=$(( $(date +%s)-t ))" | tee -a $OUT/timings.txt
done
kill $(cat $C/gpumon.pid) 2>/dev/null || true
ls $I > $OUT/ink_files.txt
$PY - <<PY
import fl_helpers as H, numpy as np, cv2, glob, os, re
I, R, O, UM, NAME = "$I", "$R", "$OUT", float("$UM"), "$SCROLL"
r = H.open_render(R); valid = np.asarray(r[20]) > 0; hw = valid.shape
def small(a, dst, s=3): cv2.imwrite(dst, a[::s, ::s], [cv2.IMWRITE_JPEG_QUALITY, 88])
for direction in ("forward", "reverse"):
    maps = {}
    for p in glob.glob(f"{I}/*_{direction}__*_z*.png"):
        m = re.search(r"/([a-z0-9]+)_" + direction + r"__(.+)_z([+-]?[0-9.]+)\.png$", p)
        maps.setdefault(m.group(1) + ":" + m.group(2), {})[float(m.group(3))] = H.load_map(p, hw)
    print(direction, {k: sorted(v) for k, v in maps.items()})
    if not maps: continue
    common = sorted(set.intersection(*[set(v) for v in maps.values()]))
    ens = H.zscore_ensemble({k: {z: v[z] for z in common} for k, v in maps.items()}, valid)
    for z in common:
        ct = np.asarray(r[int(20 + round(z * 9.6 / UM))])
        H.claim_image(ens[z], ct, f"{O}/tmp.png", UM, title=f"{NAME} {direction} z{z:+g} zscore({len(maps)})")
        small(cv2.imread(f"{O}/tmp.png"), f"{O}/ens_{direction}_z{z:+g}_small.jpg")
    best = max(common, key=lambda z: np.percentile(ens[z][valid], 99.5))
    print(direction, "p99.5 per shift", {z: round(float(np.percentile(ens[z][valid], 99.5)), 2) for z in common}, "best", best)
    for tag in ("readerv3", "ink9"):
        p = f"{I}/{tag}_{direction}.png"
        if os.path.exists(p): small(H.stretch(cv2.imread(p, 0).astype(np.float32), 50, 99.8), f"{O}/{tag}_{direction}_small.jpg")
os.path.exists(f"{O}/tmp.png") and os.remove(f"{O}/tmp.png")
PY
sort -n $OUT/gpu_mem.txt | tail -1 | sed 's/^/peak gpu mem used: /' | tee -a $OUT/timings.txt
