# slot: cpu
# after: sg_404_cw_c04
# timeout: 5400
# sg lane, PHerc0846A: on-sheet gate for chunk c04 (raw + snapped vs +-8 vox control, cross-sections, 4 x 41-layer renders + onsheet_score)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/s846_code_v2.tgz -C $PWD/code; export CODE=$PWD/code
KEEP=1 bash $CODE/gate.sh c04 2>&1 | tee $OUT/gate.log
