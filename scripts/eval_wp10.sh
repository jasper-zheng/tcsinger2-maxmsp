#!/usr/bin/env bash
# WP10 evaluation of the baselines (sung, stage 2) and the F0 generator milestones on `valid` and `test_unseen` (scripts/eval_f0gen.py, 200 clips,
# 3 seeds, 32 steps, cfg 1.0), resumable: a milestone whose <label>.json already exists for a split is skipped.
# Log: logs/eval_wp10.log; results: samples/wp10/<split>/.
#
# Usage: nohup bash scripts/eval_wp10.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
LOG=logs/eval_wp10.log
FLOW=logs/2026-10-14T16-46-00_f0gen/checkpoints/trainstep_checkpoints
REG=logs/2026-10-15T02-29-32_f0gen_reg/checkpoints/trainstep_checkpoints
CKPTS=(
  $FLOW/epoch=000040-step=000090000.ckpt
  $FLOW/epoch=000044-step=000100000.ckpt
  $REG/epoch=000013-step=000030000.ckpt
  $REG/epoch=000017-step=000040000.ckpt
  $REG/epoch=000044-step=000100000.ckpt
)
for split in valid test_unseen; do
  if [ -f "samples/wp10/$split/sung.json" ] && [ -f "samples/wp10/$split/stage2.json" ]; then
    echo "==== $(date -u '+%F %T') $split baselines exist; skipped" >> $LOG
  else
    echo "==== $(date -u '+%F %T') $split baselines" >> $LOG
    python scripts/eval_f0gen.py --methods sung stage2 --split $split >> $LOG 2>&1 \
      || echo "==== $(date -u '+%F %T') FAILED $split baselines" >> $LOG
  fi
  for ck in "${CKPTS[@]}"; do
    run=$(echo "$ck" | cut -d/ -f2); run=${run#*_}
    label="${run}_$(basename "$ck" .ckpt)_s32_cfg1"
    if [ -f "samples/wp10/$split/$label.json" ]; then
      echo "==== $(date -u '+%F %T') $split $label exists; skipped" >> $LOG; continue
    fi
    echo "==== $(date -u '+%F %T') $split $label" >> $LOG
    python scripts/eval_f0gen.py --methods "f0gen:$ck" --steps 32 --cfg 1.0 --split $split >> $LOG 2>&1 \
      || echo "==== $(date -u '+%F %T') FAILED $split $label" >> $LOG
  done
done
# Listening set: the first 16 clips of each split synthesised by stage 2 in `given` mode with the sung curve, the
# stage-2 predictor and the two candidates (samples/wp10/<split>/synth/, skipped when its metrics.json exists).
for split in valid test_unseen; do
  if [ -f "samples/wp10/$split/synth/metrics.json" ]; then
    echo "==== $(date -u '+%F %T') $split synth exists; skipped" >> $LOG; continue
  fi
  echo "==== $(date -u '+%F %T') $split synth" >> $LOG
  python scripts/eval_f0gen.py --methods sung stage2 "f0gen:${CKPTS[0]}" "f0gen:${CKPTS[2]}" --synth 16 --table_only \
    --split $split >> $LOG 2>&1 || echo "==== $(date -u '+%F %T') FAILED $split synth" >> $LOG
done
echo "==== $(date -u '+%F %T') eval_wp10.sh done" >> $LOG
