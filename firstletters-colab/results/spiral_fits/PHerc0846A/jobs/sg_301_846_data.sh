# slot: cpu
# timeout: 7200
# sg lane, PHerc0846A z 2450-3750: data prep (published tracks + crossings, m7, CT, own umbilicus spec, surf-SDT)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/s846_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/s846
SDT_WORKERS=5 bash $CODE/prep_846.sh link m7 ct tracks spec sdt 2>&1 | tee $OUT/prep.log; rc=${PIPESTATUS[0]}
du -sh $S/* > $OUT/du.txt 2>&1; exit $rc
