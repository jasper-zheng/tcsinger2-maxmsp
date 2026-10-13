#!/usr/bin/env bash
# Trains the Custom Audio Encoder (WP5): starts a run, or resumes the newest logs/*_ae_con run that has a resume
# checkpoint (milestones alone do not count). Runs in the foreground; output is appended to logs/con_train.log. Extra arguments go to
# main.py, e.g. `--max_steps 30000` (one optimizer, so a step is a batch).
#
# Usage: bash scripts/train_con.sh [main.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
mkdir -p logs
log=logs/con_train.log

run=""
while IFS= read -r d; do
    if [ -n "$(find "$d/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit 2>/dev/null)" ]; then run="$d"; break; fi
done < <(ls -dt logs/*_ae_con 2>/dev/null || true)

if [ -n "$run" ]; then
    echo "==== $(date '+%F %T') resuming $run" | tee -a "$log"
    python main.py -r "$run" -t --gpus 0, --no-test true "$@" >> "$log" 2>&1
else
    echo "==== $(date '+%F %T') starting a new run" | tee -a "$log"
    python main.py --base configs/ae_con.yaml -t --gpus 0, --no-test true "$@" >> "$log" 2>&1
fi
echo "==== $(date '+%F %T') main.py exited" | tee -a "$log"
