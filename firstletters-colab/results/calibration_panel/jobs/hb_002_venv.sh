# slot: cpu
# timeout: 3000
# winner's exact environment: python 3.14 venv + requirements.txt (villa @b29f310)
C=/content/cache/hbink; mkdir -p $C
python -m pip -q install uv
cd $C && mkdir -p win && tar xzf $INPUTS/win846.tgz -C win --strip-components=1
uv venv --python 3.14 $C/venv 2>&1 | tail -2
uv pip install --python $C/venv -r $C/win/requirements.txt 2>&1 | tail -5
$C/venv/bin/python -c "import vesuvius, torch, zarr; print('ok', torch.__version__, torch.cuda.is_available(), zarr.__version__)" | tee $OUT/venv.txt
