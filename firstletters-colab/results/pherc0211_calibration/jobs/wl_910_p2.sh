# slot: cpu
# timeout: 10800
# Task P2: 12 GrowPatch seeds on successive sheets along ONE radius chosen by CT anatomy (z 8400, from the official
# umbilicus straight towards +y, start 150 vox out), sheet centres from m7 L0 runs; 100 generations; crop 120x120 cells
# around the seed; flatboi; into /content/cache/wlink/meshes/211/p2/pNN
set -u; source $INPUTS/vcenv.sh; export PYTHONPATH=$PWD; cp $INPUTS/fl_helpers.py .
NG=/content/cache/wl211/ngrid; M7=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr
P=/content/cache/wlink/meshes/211/p2; mkdir -p $P /content/cache/wl211/p2
python3 - <<'PY' > $OUT/seeds.txt
import json, numpy as np, zarr
u=json.load(open('/content/cache/wl211/ds/PHerc0211/umbilicus.json'))['control_points']; zz=np.array([p['z'] for p in u])
ux=float(np.interp(8400,zz,[p['x'] for p in u])); uy=float(np.interp(8400,zz,[p['y'] for p in u]))
m=zarr.open('/content/cache/wl211/m7/0', mode='r')
x=int(round(ux)); y0=int(round(uy))+150
col=np.asarray(m[8400, y0:y0+1500, x-1:x+2]).max(1) > 0
runs=[]; i=0
while i<len(col):
    if col[i]:
        j=i
        while j<len(col) and col[j]: j+=1
        if j-i>=2: runs.append((y0+i, y0+j-1))
        i=j
    else: i+=1
print('# umbilicus', round(ux,1), round(uy,1), 'ray +y from', y0, 'runs', len(runs))
for k,(a,b) in enumerate(runs[:12]): print('p%02d'%k, x, (a+b)//2, 8400, b-a+1)
PY
cat $OUT/seeds.txt
grow() { local n=$1 x=$2 y=$3 z=$4 G=/content/cache/wl211/p2/$1; mkdir -p $G
  python3 -c "import fl_helpers as H; H.write_grow_params('$G/params.json', '$NG', 9.362, generations=100, threads=3)"
  local s=$(date +%s); ( cd $G && vc vc_grow_seg_from_seed -v $M7 -t $G -p $G/params.json --seed $x $y $z ) > $G/grow.log 2>&1
  M=$(ls -d $G/auto_grown_* 2>/dev/null | tail -1); [ -n "$M" ] || { echo "$n NO MESH"; return; }
  python3 - <<PY
import numpy as np, tifffile, json, os
x,y,z=[tifffile.imread('$M/%s.tif'%c) for c in 'xyz']; v=z>0
d=(x-$x)**2+(y-$y)**2+(z-$z)**2; d[~v]=1e18; r,c=np.unravel_index(d.argmin(),d.shape)
r0,c0=max(0,r-60),max(0,c-60); o='$G/crop'; os.makedirs(o,exist_ok=True)
for k,a in zip('xyz',(x,y,z)): tifffile.imwrite(o+'/'+k+'.tif',a[r0:r0+120,c0:c0+120].astype(np.float32))
m=json.load(open('$M/meta.json')); m.pop('bbox',None); json.dump(m,open(o+'/meta.json','w'))
PY
  rm -rf $P/$n; PASTIX_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 OPENBLAS_CORETYPE=Haswell vc flatboi $G/crop --output=$P/$n 50 symmetric_dirichlet > $G/flat.log 2>&1
  echo "$n grow+crop+flat $(( $(date +%s)-s ))s $( [ -f $P/$n/x.tif ] && echo ok || echo FLATFAIL)"; }
i=0; while read n x y z w; do [ "${n:0:1}" = p ] || continue; grow $n $x $y $z & i=$((i+1)); [ $((i%3)) = 0 ] && wait; done < $OUT/seeds.txt; wait
cd $P && tar czf $OUT/p2_meshes.tgz p*; ls $P
