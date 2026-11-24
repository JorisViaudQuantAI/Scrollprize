# slot: cpu
# after: sg_104_cwb_c04
# timeout: 5400
# sg lane: on-sheet gate for chunk c04 (raw + snapped vs +-8 vox control, cross-sections, 4 x 41-layer renders + onsheet_score)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sg_code_v5.tgz -C $PWD/code; export CODE=$PWD/code
KEEP=1 bash $CODE/gate.sh c04 2>&1 | tee $OUT/gate.log
