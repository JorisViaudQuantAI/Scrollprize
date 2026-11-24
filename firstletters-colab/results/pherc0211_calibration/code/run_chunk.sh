#!/bin/bash
# run_chunk.sh SENSE TAG N : one time-bounded chunk of a complete N-step schedule (band Z0..Z1), resuming from the last chunk.
set -uo pipefail
S=${S:-/content/cache/wl}; SENSE=$1; TAG=$2; N=$3; CODE=${CODE:?}
V=$S/villa/volume-cartographer/scripts/spiral; PY=$(cat $S/spiral_py.txt)
$PY $CODE/patch_deadline.py $V || exit 4
LAST=$(cat $S/fitd_last 2>/dev/null || true); RES=""
if [ -n "$LAST" ]; then
  CK=$(ls -d $S/fits/*_$LAST | head -1)/checkpoint_fitted.ckpt
  DONE=$($PY -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['completed_iterations'])")
  echo "last chunk $LAST completed_iterations=$DONE (target $N)"
  [ "$DONE" -ge "$N" ] && { echo "schedule already complete"; exit 0; }
  RES=$LAST
fi
export FIT_SPIRAL_LOOP_SECONDS=${LOOP_S:-1680}
TDT=$(( 10000 * N / 30000 ))
EXTRA=", \"loss_start_track_dt\": $TDT" H=$N bash $CODE/run_fit.sh $SENSE $TAG $N $RES; rc=$?
[ $rc = 0 ] && echo $TAG > $S/fitd_last
RD=$(ls -d $S/fits/*_$TAG 2>/dev/null | head -1)
[ -n "$RD" ] && $PY -c "import torch;print('completed_iterations now', torch.load('$RD/checkpoint_fitted.ckpt',map_location='cpu',weights_only=False)['completed_iterations'])"
exit $rc
