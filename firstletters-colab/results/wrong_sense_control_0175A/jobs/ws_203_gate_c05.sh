# slot: cpu
# after: ws_105_cw_c05
# timeout: 5400
# ws lane: on-sheet gate for chunk c05 (raw + snapped vs +-8 vox control, cross-sections, 4 x 41-layer renders + onsheet_score)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v2.tgz -C $PWD/code; export CODE=$PWD/code
KEEP=0 bash $CODE/gate.sh c05 2>&1 | tee $OUT/gate.log
