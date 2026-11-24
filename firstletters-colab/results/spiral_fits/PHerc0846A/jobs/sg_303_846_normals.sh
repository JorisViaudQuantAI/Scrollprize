# after: sg_301_846_data
# timeout: 2400
# sg lane, PHerc0846A: synthetic normals from m7 (306B make_normals.py, GPU) + resident pools (sdt, normals)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/s846_code_v1.tgz -C $PWD/code; export CODE=$PWD/code; S=/content/cache/s846
bash $CODE/prep_846.sh normals pools 2>&1 | tee $OUT/prep.log; exit ${PIPESTATUS[0]}
