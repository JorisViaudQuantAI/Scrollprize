# slot: cpu
# timeout: 3600
# sf lane: villa @0fb38c45 + uv env (Python 3.14) for the spiral fitter
set -eu
S=/content/cache/sf; mkdir -p $S
export UV_CACHE_DIR=$S/uvcache UV_PYTHON_INSTALL_DIR=$S/uvpython
T0=$(date +%s); t() { echo "TIMING $1 $(( $(date +%s) - T0 ))s" | tee -a $OUT/timings.txt; }
{ nproc; free -g; df -h /content | tail -1; nvidia-smi -L 2>&1 || true; python3 --version; } > $OUT/vm.txt 2>&1
if [ ! -f $S/villa.ok ]; then
  rm -rf $S/villa
  git clone -q --filter=blob:none https://github.com/ScrollPrize/villa.git $S/villa
  git -C $S/villa checkout -q 0fb38c45b488e192566ea83ae36369cb6db92652
  touch $S/villa.ok
fi
git -C $S/villa rev-parse HEAD > $OUT/villa_head.txt; t clone_villa
command -v uv >/dev/null || pip install -q uv
V=$S/villa/volume-cartographer/scripts/spiral
(cd $V && uv sync --python 3.14 2>&1 | tail -30) > $OUT/uv_sync.txt 2>&1
t uv_sync
PY=$S/villa/volume-cartographer/.venv/bin/python; [ -x $PY ] || PY=$V/.venv/bin/python; echo $PY > $S/spiral_py.txt
$PY -c 'import sys,torch,numpy,zarr; print(sys.version); print(torch.__version__, torch.version.cuda, numpy.__version__, zarr.__version__)' > $OUT/env.txt 2>&1
cat $OUT/env.txt
touch $S/setup.ok
