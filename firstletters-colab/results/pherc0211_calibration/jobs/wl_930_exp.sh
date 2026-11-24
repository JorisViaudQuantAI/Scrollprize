# slot: cpu
# timeout: 3600
# Task P3 export: raw Studio maps (uint8 = probability*255 as written by si_tools stpng) downscaled to 20 px/mm by area mean
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
mkdir -p $OUT/raw20
python - <<PY
import cv2, glob, os, json
C='$C'; fr=1000/9.362/20; out='$OUT/raw20'; rec=[]
for d in sorted(glob.glob(C+'/maps/fw0*/s96*'))+sorted(glob.glob(C+'/maps/fw0*/st*_s0'))+sorted(glob.glob(C+'/maps/pw_*/stdesc_s0')):
    f=glob.glob(d+'/*__studio_z+0.png')
    if not f: continue
    a=cv2.imread(f[0],0); n=d.split('/')[-2]; k=d.split('/')[-1]
    b=cv2.resize(a,(max(1,int(a.shape[1]/fr)),max(1,int(a.shape[0]/fr))),interpolation=cv2.INTER_AREA)
    o=f'{out}/{n}__{k}.png'; cv2.imwrite(o,b); rec.append(dict(file=os.path.basename(o),src_shape=list(a.shape),shape=list(b.shape)))
json.dump(rec,open(out+'/index.json','w'),indent=1); print(len(rec))
PY
for n in $(ls $C/renders | grep -E '^fw0[1-6][0-9]$'); do cp $C/renders/$n/render_meta.json $OUT/raw20/${n}_render_meta.json; done
