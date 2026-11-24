# slot: cpu
# after: wl_901_s96
# timeout: 9000
# Task P1: 20 mm windows (valid >= 50 %), post per window on st96 maps; per-winding post + z20 export (P3)
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=2
mkdir -p $OUT/win $OUT/export
w1() { python $INPUTS/pwin.py windows $C $1 st96$2 $C/maps/$1/s96$2 $OUT/win/$1_$2.json; si_post $1 st96$2 $C/maps/$1/s96$2 >/dev/null; }
i=0; for n in fw010 fw011 fw012 fw013 fw014 fw015 fw016 fw017 fw018 fw019 fw020 fw021 fw022 fw023 fw024 fw025 fw026 fw027 fw028 fw029 fw030 fw031 fw032 fw033 fw034 fw035 fw036 fw037 fw038 fw039 fw040 fw041 fw042 fw043 fw044 fw045 fw046 fw047 fw048 fw049 fw050 fw051 fw052 fw053 fw054 fw055 fw056 fw057 fw058 fw059 fw060 fw061 fw062 fw063 fw064 fw065 fw066 fw067 fw068 fw069 ; do w1 $n desc & i=$((i+1)); [ $((i%4)) = 0 ] && wait; done; wait
for n in fw014 fw018 fw021 fw030 fw033 fw041 fw047 fw049 fw052; do w1 $n asc & done; wait
cp $C/red/PHerc0211_fw0*_st96*_z20.png $OUT/export/ 2>/dev/null; for n in fw010 fw011 fw012 fw013 fw014 fw015 fw016 fw017 fw018 fw019 fw020 fw021 fw022 fw023 fw024 fw025 fw026 fw027 fw028 fw029 fw030 fw031 fw032 fw033 fw034 fw035 fw036 fw037 fw038 fw039 fw040 fw041 fw042 fw043 fw044 fw045 fw046 fw047 fw048 fw049 fw050 fw051 fw052 fw053 fw054 fw055 fw056 fw057 fw058 fw059 fw060 fw061 fw062 fw063 fw064 fw065 fw066 fw067 fw068 fw069 ; do cp $C/dst/$n/mid20_par.jpg $OUT/export/${n}_mid20_par.jpg 2>/dev/null; done
python - <<PY
import cv2,glob,os
for n in "fw010 fw011 fw012 fw013 fw014 fw015 fw016 fw017 fw018 fw019 fw020 fw021 fw022 fw023 fw024 fw025 fw026 fw027 fw028 fw029 fw030 fw031 fw032 fw033 fw034 fw035 fw036 fw037 fw038 fw039 fw040 fw041 fw042 fw043 fw044 fw045 fw046 fw047 fw048 fw049 fw050 fw051 fw052 fw053 fw054 fw055 fw056 fw057 fw058 fw059 fw060 fw061 fw062 fw063 fw064 fw065 fw066 fw067 fw068 fw069 ".split():
    for nm,f in (('valid','$C/renders/%s/mask.png'%n),('parallel','$C/dst/%s/mask.png'%n)):
        m=cv2.imread(f,0)
        if m is None: continue
        fr=1000/9.362/20; cv2.imwrite('$OUT/export/%s_%s_mask20.png'%(n,nm), cv2.resize(m,(int(m.shape[1]/fr),int(m.shape[0]/fr)),interpolation=cv2.INTER_AREA))
PY
