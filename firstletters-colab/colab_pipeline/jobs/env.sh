# env.sh - sourced by 01_grow.sh and 02_render_ink.sh; rebuilds what a fresh Colab VM lacks (idempotent).
# Parametrised copy of the env script used for the measured PHerc0175A run (see README, "Colab pipeline").
# Every value below can be overridden by exporting it before sourcing.
#   INPUTS   folder holding fl_helpers.py and the ink/ folder of this package (on Colab: a Drive folder or /content/inputs)
#   OUT      folder for logs, timings, mesh copy and preview images
set -eo pipefail
: "${INPUTS:?set INPUTS to the folder with fl_helpers.py and ink/}"
: "${OUT:?set OUT to an output folder}"
mkdir -p "$OUT"
export C=${CACHE:-/content/cache}; mkdir -p $C
export SCROLL=${SCROLL:-PHerc0175A} UM=${UM:-8.64}
export SEED_X=${SEED_X:-5423} SEED_Y=${SEED_Y:-5806} SEED_Z=${SEED_Z:-6960}
export GENS=${GENS:-80}
# Hand-off from 01_grow.sh to 02_render_ink.sh. The default is specific to scroll + seed + generations, so two grows
# on the same scroll never overwrite each other's hand-off; 02_render_ink.sh uses MESH=<tifxyz dir> instead when set.
export MESH_FILE=${MESH_FILE:-$C/grow/last_mesh_${SCROLL}_${SEED_X}_${SEED_Y}_${SEED_Z}_g${GENS}.txt}
export BUCKET=${BUCKET:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com}
# volume, m7 surface prediction and its normal grids (defaults: PHerc0175A; see scrolls19.json in the notebook for the others)
export VOL_URL=${VOL_URL:-$BUCKET/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr}
export M7_URL=${M7_URL:-$BUCKET/PHerc0175A/representations/predictions/surfaces/20250521115057-surface-20260413222639-surface-m7-L0-th0.2.zarr}
export NG_KEY=${NG_KEY:-PHerc0175A/representations/predictions/surfaces/20250521115057-surface-20260413222639-surface-m7-L0-th0.2.normal-grids}
# VC3D Linux AppImage from the villa releases page. The measured run used build c8f3ba4
# (sha256 ed4846d5985ac3e67e2a5881e8c915a8fab111289a021f896cb49ab96dc290f9); the 'latest' tag moves.
export APPIMAGE_URL=${APPIMAGE_URL:-https://github.com/ScrollPrize/villa/releases/download/latest/VC3D-c8f3ba4-2026-11-20-linux-x86_64.AppImage}
export APPRUN=$C/vc3d/squashfs-root/AppRun
export VC3D_CONFIG_DIR=$C/vc3d_cfg QT_QPA_PLATFORM=offscreen
export PYTHONPATH=$C/pylib:$C/code
export PY="python"
ensure_tools() {
  if [ ! -x "$APPRUN" ]; then
    # pitfall 1: the AppImage needs GL/X/font libraries that a Colab VM lacks, even headless
    apt-get -qq update >/dev/null && apt-get -qq install -y --no-install-recommends libglvnd0 libglx0 libegl1 libopengl0 libgl1 \
      libgl1-mesa-dri libglx-mesa0 libx11-6 libx11-xcb1 libxcb1 libice6 libsm6 libfontconfig1 libfreetype6 libharfbuzz0b \
      libxkbcommon0 libdbus-1-3 >/dev/null
    mkdir -p $C/vc3d && cd $C/vc3d
    curl -fsSL -o VC3D.AppImage "$APPIMAGE_URL" || { echo "AppImage URL gone; set APPIMAGE_URL to a current release"; return 1; }
    sha256sum VC3D.AppImage
    # pitfall 2: no FUSE on Colab -> extract once and call squashfs-root/AppRun <tool>
    chmod +x VC3D.AppImage && ./VC3D.AppImage --appimage-extract >/dev/null && cd - >/dev/null
  fi
  # pitfall 3: VC3D tools stream the CT from S3 into a cache; put it on the big disk and keep 5 GiB free
  mkdir -p $VC3D_CONFIG_DIR; printf '[viewer]\nremote_cache_dir=%s/vc3d_cache\n[perf]\nremote_cache_min_free_gib=5\n' $C > $VC3D_CONFIG_DIR/VC3D.ini
  if [ ! -f $C/pylib/.ok2 ]; then
    # pitfall 4: 'python -m venv' fails on Colab (no ensurepip); install the few extra packages into a --target dir.
    # pitfall 5: zarr 3 cannot read these v2 stores the way the helpers expect -> pin zarr<3, numcodecs<0.16;
    #            the pinned numcodecs needs 'deprecated' (and 'wrapt').
    python -m pip -q install --no-deps --target $C/pylib 'zarr<3' 'numcodecs<0.16' asciitree fasteners deprecated wrapt && touch $C/pylib/.ok2
    python -c 'import fsspec, aiohttp, requests, tifffile, cv2' || python -m pip -q install fsspec aiohttp requests tifffile
  fi
  mkdir -p $C/code && cp "$INPUTS/fl_helpers.py" $C/code/ && cp -r "$INPUTS/ink/." $C/code/
}
ensure_models() {
  mkdir -p $C/models
  for f in "scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth" "scrollprize/ink_9um hybrid_3d2d-seed43/step-075000.pth" \
           "domenicor046/reader-v3 reader-v3.pth"; do
    set -- $f; d=$C/models/$1/$2; [ -s $d ] || { mkdir -p $(dirname $d); curl -fsSL -o $d https://huggingface.co/$1/resolve/main/$2; }
  done
}
vc() { local t=$1; shift; echo "\$ $t $*"; local s=$(date +%s); "$APPRUN" "$t" "$@"; local rc=$?; echo "$t rc=$rc seconds=$(( $(date +%s)-s ))" | tee -a $OUT/timings.txt; return $rc; }
gpumon() { ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> $OUT/gpu_mem.txt; sleep 15; done ) & echo $! > $C/gpumon.pid; }
