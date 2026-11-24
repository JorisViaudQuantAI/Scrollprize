# slot: cpu
# timeout: 5400
# follow-up (private): GrowPatch seeded at the fw030_x1 stroke row (snap vertex 3789 3604 7937), 100 gen; crop cells within
# 1400 vox of the seed with z >= 7250 (extends above the band); flatboi; render; gridq; normside
set -u; source $INPUTS/vcenv.sh; export PYTHONPATH=$PWD; cp $INPUTS/fl_helpers.py .
NG=/content/cache/wl211/ngrid; M7=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr
G=/content/cache/wl211/gp30; mkdir -p $G
python3 -c "import fl_helpers as H; H.write_grow_params('$G/params.json', '$NG', 9.362, generations=100, threads=10)"
s=$(date +%s); ( cd $G && vc vc_grow_seg_from_seed -v $M7 -t $G -p $G/params.json --seed 3789 3604 7937 ) > $G/grow.log 2>&1; echo "grow $(( $(date +%s)-s ))s" | tee $OUT/t.txt
M=$(ls -d $G/auto_grown_* | tail -1)
python3 - <<PY
import numpy as np, tifffile, json, os
x,y,z=[tifffile.imread('$M/%s.tif'%c) for c in 'xyz']; v=z>0
d=np.sqrt((x-3789.)**2+(y-3604.)**2+(z-7937.)**2); m=v&(d<1400)&(z>=7250)
r,c=np.where(m); r0,r1,c0,c1=r.min(),r.max()+1,c.min(),c.max()+1; o='$G/crop'; os.makedirs(o,exist_ok=True)
for k,a in zip('xyz',(x,y,z)):
    b=a[r0:r1,c0:c1].astype(np.float32).copy(); b[~m[r0:r1,c0:c1]]=-1; tifffile.imwrite(o+'/'+k+'.tif',b)
mm=json.load(open('$M/meta.json')); mm.pop('bbox',None); json.dump(mm,open(o+'/meta.json','w'))
print('crop cells',m.sum(),'z',z[m].min(),z[m].max())
PY
D=/content/cache/wlink/meshes/211/gp30; rm -rf $D; mkdir -p $(dirname $D)
PASTIX_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OPENBLAS_CORETYPE=Haswell vc flatboi $G/crop --output=$D 50 symmetric_dirichlet > $G/flat.log 2>&1
mkdir -p $OUT/mesh; cp $D/* $OUT/mesh/
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export RWORKERS=16
si_render gp30 $VOL $D
python $CODE/gridq.py $D > $OUT/gridq.txt 2>&1; python $CODE/normside.py /content/cache/wl211/ds/PHerc0211/umbilicus_official.json $D > $OUT/normside.txt 2>&1; cat $OUT/gridq.txt $OUT/normside.txt
