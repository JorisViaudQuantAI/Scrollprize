# after: hb_002_venv, hb_003b_ren
# timeout: 5400
# (a) the winner's script exactly; (D) villa infer on our render's central 17 layers, both directions
C=/content/cache/hbink; cd $C/win
$C/venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { python -m pip -q install uv; uv pip install --python $C/venv torch torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2; }
$C/venv/bin/python -c "import torch; print('venv torch', torch.__version__, torch.cuda.is_available())" | tee $OUT/venv_gpu.txt
python - <<'PY'
import json; p='input/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750.json'; d=json.load(open(p)); d['show']=False; json.dump(d,open(p,'w'))
PY
t=$(date +%s); $C/venv/bin/python infer_letters.py PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750 2>&1 | grep -v '^render:' | tail -15 | tee $OUT/win_log.txt; echo "win s=$(($(date +%s)-t))" | tee -a $OUT/timings.txt
cp output/*_raw.tif $OUT/win_repro_raw.tif
$C/venv/bin/python - <<'PY'
import numpy as np, tifffile, zarr, glob
C='/content/cache/hbink'; fs=sorted(glob.glob(C+'/renders/p846/layers/*.tif'))
st=np.stack([tifffile.imread(f) for f in fs[12:29]]); print(st.shape, st.dtype)
zarr.open_array(C+'/ourc17.zarr', mode='w', shape=st.shape, chunks=(17,256,256), dtype=np.uint8, zarr_format=2)[:]=st
m=(tifffile.imread(C+'/renders/p846/layers/20.tif')>0).astype(np.uint8)*255; tifffile.imwrite(C+'/ourc17_mask.tif', m)
PY
CK=$(ls ~/.cache/huggingface/hub/models--domenicor046--reader-v3/snapshots/*/reader-v3.pth | head -1)
for d in forward reverse; do t=$(date +%s)
$C/venv/bin/python -c "import sys; from vesuvius.ink_detection.inference import infer; infer.main(sys.argv[1:])" $C/ourc17.zarr $CK $C/villa_our_$d.tif --mask-path $C/ourc17_mask.tif --overlap 0.5 --blend-mode hann --direction $d --batch-size 4 --num-workers 0 --no-compile 2>&1 | tail -2
cp $C/villa_our_$d.tif $OUT/; echo "villa_$d s=$(($(date +%s)-t))" | tee -a $OUT/timings.txt; done
