#!/bin/bash
# run_fit.sh SENSE TAG STOP_AT [RESUME_TAG] : one (resumable) chunk of the band fit with the winners' overrides.
# Env: H (horizon, optimizer_num_training_steps; default 10000), Z0/Z1, EXTRA (extra JSON overrides, comma-prefixed)
set -uo pipefail
S=/content/cache/sf; SENSE=$1; TAG=$2; STOP=$3; RES=${4:-}
Z0=${Z0:-6900}; Z1=${Z1:-8400}; H=${H:-10000}; EXTRA=${EXTRA:-}
V=$S/villa/volume-cartographer/scripts/spiral; PY=$(cat $S/spiral_py.txt)
[ -f $S/prep/ALL.ok ] || { echo "prep missing (VM restarted?)"; exit 2; }
mkdir -p $S/fits $S/fitcache $S/logs
OV="{\"z_begin\": $Z0, \"z_end\": $Z1, \"input_disable_patches\": true, \"loss_weight_shell_outer\": 0, \"loss_weight_shell_patch_radius\": 0, \"dense_spacing_mode\": \"phase\", \"optimizer_num_training_steps\": $H$EXTRA}"
echo "overrides: $OV"
RESARG=""
if [ -n "$RES" ]; then RD=$(ls -d $S/fits/*_$RES | head -1); RESARG=$RD/checkpoint_fitted.ckpt; [ -f $RESARG ] || { echo "no checkpoint for $RES"; exit 3; }; echo "resume from $RESARG"; fi
go() {
  ( cd $V && env WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 FIT_SPIRAL_STOP_AT=$STOP FIT_SPIRAL_OUT_DIR=$S/fits FIT_SPIRAL_RUN_TAG=$TAG \
      ${RESARG:+FIT_SPIRAL_RESUME_PATH=$RESARG} PYTHONPATH=$S/native_pkg FIT_SPIRAL_CONFIG_OVERRIDES="$OV" "$@" \
      $PY fit_spiral.py --dataset $S/ds/${SC:-PHerc0175A}_$SENSE --cache $S/fitcache ) > $S/logs/fit_$TAG.log 2>&1
}
( while true; do [ -n "${OUT:-}" ] && tail -c 60000 $S/logs/fit_$TAG.log > $OUT/fit_live_$TAG.log 2>/dev/null; sleep 60; done ) & LIVE=$!
t0=$(date +%s); go; rc=$?
if [ $rc != 0 ] && grep -qi -E "triton|ptxas|sm_75" $S/logs/fit_$TAG.log; then
  echo "fit failed with a Triton-looking error; retry eager (FIT_SPIRAL_TRITON=0)"; cp $S/logs/fit_$TAG.log $S/logs/fit_${TAG}_triton_fail.log
  go FIT_SPIRAL_TRITON=0; rc=$?
fi
kill $LIVE 2>/dev/null; echo "fit rc=$rc wall=$(( $(date +%s) - t0 ))s"
exit $rc
