#!/bin/bash
# Measure the official hecate.py (with the fp16 patch) at fp32 / bf16 / fp16 on one GPU, same input.
# Prints seconds per run (including process start and model load) and the max / mean absolute difference to fp32 (uint8 levels) and Pearson r.
# usage: bash measure_t4.sh [WORKDIR]     (needs: torch with CUDA, numpy, scipy, requests, opencv-python)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-./hecate_measure}; mkdir -p "$W"; cd "$W"
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -s hecate_9.6um.pth ] || curl -fsSL -o hecate_9.6um.pth https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth
[ -s crop.npy ] || python "$HERE/prepare_crop.py" crop.npy
for p in fp16 bf16 fp32; do
  rm -f out_$p.png out_$p.json
  s=$(date +%s.%N)
  python "$HERE/hecate.py" --checkpoint hecate_9.6um.pth --input crop.npy --output out_$p.png \
         --spacing-um 9.6 --device cuda --batch-size 32 --precision $p > /dev/null
  python -c "import sys; print('$p seconds', round($(date +%s.%N) - $s, 1))"
done
python - <<'PY'
import cv2, numpy as np
ref = cv2.imread("out_fp32.png", 0).astype(float)
for p in ("fp16", "bf16"):
    x = cv2.imread(f"out_{p}.png", 0).astype(float)
    print(p, "vs fp32: max", np.abs(x - ref).max(), "mean", round(np.abs(x - ref).mean(), 3),
          "r", round(np.corrcoef(x.ravel(), ref.ravel())[0, 1], 6))
PY
