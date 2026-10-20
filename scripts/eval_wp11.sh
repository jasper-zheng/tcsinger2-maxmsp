#!/usr/bin/env bash
# WP11: how few sampling steps stage 2 and the F0 generator need. Resumable: every configuration whose result exists is
# skipped, so after a shutdown the same command goes on where it stopped.
#   1. scripts/eval_f0gen.py on the exported F0 generator at 64, 32, 16, 8, 4, 2, 1 Euler steps (200 clips, 3 seeds,
#      curves saved) for valid and test_unseen -> samples/wp11_steps/f0gen/<split>/
#   2. scripts/eval_steps.py --part f0ref, stage2, pipeline (100 clips), listen, speed -> samples/wp11_steps/
# Log: logs/eval_wp11.log.
#
# Usage: nohup bash scripts/eval_wp11.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
LOG=logs/eval_wp11.log
OUT=samples/wp11_steps
F0=useful_ckpts/f0gen_gtsinger.ckpt
log() { echo "==== $(date -u '+%F %T') $*" >> $LOG; }
run() { log "$1"; shift; "$@" >> $LOG 2>&1 || log "FAILED $*"; }

for split in valid test_unseen; do
  for K in 64 32 16 8 4 2 1; do
    if [ -f "$OUT/f0gen/$split/f0gen_gtsinger_s${K}_cfg1_curves.npz" ]; then log "$split f0gen s$K exists; skipped"; continue; fi
    run "$split f0gen s$K" python scripts/eval_f0gen.py --methods "f0gen:$F0" --steps $K --cfg 1.0 --split $split --save_curves --out_dir $OUT/f0gen
  done
  run "$split f0ref" python scripts/eval_steps.py --part f0ref --split $split
done
for part in stage2 pipeline listen; do
  for split in valid test_unseen; do
    run "$split $part" python scripts/eval_steps.py --part $part --split $split
  done
done
if [ -f "$OUT/speed.json" ]; then log "speed exists; skipped"; else run "speed" python scripts/eval_steps.py --part speed; fi
log "eval_wp11.sh done"
