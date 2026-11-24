# slot: cpu
# after: ws_103_cw_c03
# timeout: 5400
# ws lane: on-sheet gate for chunk c03 (raw + snapped vs +-8 vox control, cross-sections, 4 x 41-layer renders + onsheet_score)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v2.tgz -C $PWD/code; export CODE=$PWD/code
KEEP=0 bash $CODE/gate.sh c03 2>&1 | tee $OUT/gate.log
