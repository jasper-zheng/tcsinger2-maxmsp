#!/usr/bin/env bash
# Evaluates the milestone checkpoints of a Custom Audio Encoder run (WP5) with scripts/eval_con.py, one after
# the other (about 2 min each). A milestone that already has samples/wp5_retrieval/step<N>/metrics.json is
# skipped, so this can be rerun after a shutdown or while training goes on. Output is appended to logs/con_eval.log.
#
# Usage: bash scripts/eval_con_milestones.sh [run directory, default: the newest logs/*_ae_con]
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
run=${1:-$(ls -dt logs/*_ae_con | head -1)}
log=logs/con_eval.log

for ckpt in "$run"/checkpoints/trainstep_checkpoints/*.ckpt; do
    step=$(basename "$ckpt" .ckpt); step=$((10#${step##*step=}))
    if [ -e "samples/wp5_retrieval/step$(printf '%08d' "$step")/metrics.json" ]; then
        echo "==== $(date '+%F %T') skipping $ckpt (evaluated)" | tee -a "$log"
        continue
    fi
    echo "==== $(date '+%F %T') evaluating $ckpt" | tee -a "$log"
    python scripts/eval_con.py --ckpt "$ckpt" >> "$log" 2>&1
done
echo "==== $(date '+%F %T') done" | tee -a "$log"
