# slot: cpu

# timeout: 5400
# sf lane: on-sheet gate for chunk c01 (raw + snapped vs +-8 vox control, cross-sections, 4 x 41-layer renders + onsheet_score)
set -u; mkdir -p $PWD/code && tar -xzf $INPUTS/sf_code_v2.tgz -C $PWD/code; export CODE=$PWD/code
KEEP=0 bash $CODE/gate.sh c01 2>&1 | tee $OUT/gate.log
