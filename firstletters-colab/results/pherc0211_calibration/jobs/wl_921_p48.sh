# slot: cpu
# after: wl_920_s48
# timeout: 5400
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=2
i=0; for w in fw044_x1 fw033_x1 fw031_x1 fw032_x1 fw035_x0 fw030_x1 fw035_x1 fw067_x2 fw029_x0 fw057_x3 fw061_x1 fw055_x2 fw041_x1 fw049_x1 fw041_x0 fw045_x1 fw019_x0 fw013_x0 fw069_x3 fw034_x0 fw056_x2 fw059_x3 fw055_x1 fw067_x3 fw033_x0 fw050_x0 fw059_x0 fw036_x0 fw047_x0 fw066_x3 fw060_x3 fw057_x1 fw060_x2 fw026_x0 fw048_x2 fw045_x2 fw020_x0 fw063_x0 fw053_x2 fw069_x4; do o=pw_$w; si_post $o stdesc $C/maps/$o/stdesc_s0 & i=$((i+1)); [ $((i%4)) = 0 ] && wait; done; wait
