# slot: cpu
# timeout: 2400
# wl ink setup: python libs, reader-v3, argus-v3, Studio release + weights (SHA-256 checked), code
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
[ -s $SW ] || python -c "import sys; sys.path.insert(0,'$CODE'); import studio_t4 as T; from pathlib import Path; T.fetch_weights(Path('$SW'))"
sha256sum $RV3 $AG3 $SW | tee $OUT/sha.txt
python -c "import torch, cv2, numpy, zarr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cv2', cv2.__version__, 'numpy', numpy.__version__, 'zarr', zarr.__version__)" | tee $OUT/env.txt
mkdir -p $C/meshes && tar xzf $INPUTS/wl_pos_mesh.tgz -C $C/meshes
stamp setup
