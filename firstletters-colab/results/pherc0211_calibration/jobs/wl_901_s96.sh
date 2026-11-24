# timeout: 9000
# Task P1: Studio stride 96 unblended, reading order (desc) on all 60 flattened windings fw010-069; asc on 9 random (rng 211)
export WLSC=PHerc0211; source $INPUTS/wl_env.sh
run96() { local n=$1 o=$2 M=$C/maps/$1/s96$2; mkdir -p $M; [ -s $M/st96${o}__studio_z+0.png ] && return 0; local t=$(date +%s)
  $ST --input $C/renders/$n --mesh-index 20 --order $o --shift 0 --stride 96 --precision fp16 --batch-size 32 --device cuda --out-prefix $M/x 2>&1 | tail -1
  python $CODE/si_tools.py stpng $M/x $M/st96${o}__studio_z+0.png >/dev/null; rm -f $M/x.prob.npy; echo "$n $o $(( $(date +%s)-t ))s" >> $OUT/t96.txt; }
for n in fw010 fw011 fw012 fw013 fw014 fw015 fw016 fw017 fw018 fw019 fw020 fw021 fw022 fw023 fw024 fw025 fw026 fw027 fw028 fw029 fw030 fw031 fw032 fw033 fw034 fw035 fw036 fw037 fw038 fw039 fw040 fw041 fw042 fw043 fw044 fw045 fw046 fw047 fw048 fw049 fw050 fw051 fw052 fw053 fw054 fw055 fw056 fw057 fw058 fw059 fw060 fw061 fw062 fw063 fw064 fw065 fw066 fw067 fw068 fw069 ; do run96 $n desc; done
for n in fw014 fw018 fw021 fw030 fw033 fw041 fw047 fw049 fw052; do run96 $n asc; done
