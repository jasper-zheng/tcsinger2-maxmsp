#!/usr/bin/env bash
# Trains stage 2 (WP6): starts the run, or resumes the newest logs/*_tcsinger2 run that has a checkpoint.
# Runs in the foreground; output is appended to logs/tcsinger2_train.log. Extra arguments go to main.py,
# e.g. `--max_steps 300000` to train further or `data.params.max_tokens=32000`.
#
# Usage: nohup bash scripts/train_tcsinger2.sh [main.py arguments] > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
mkdir -p logs
log=logs/tcsinger2_train.log

run=""
while IFS= read -r d; do
    if [ -n "$(find "$d/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit 2>/dev/null)" ]; then run="$d"; break; fi
done < <(ls -dt logs/*_tcsinger2 2>/dev/null || true)

if [ -n "$run" ]; then
    echo "==== $(date '+%F %T') resuming $run" | tee -a "$log"
    python main.py -r "$run" -t --gpus 0, --no-test true --scale_lr false "$@" >> "$log" 2>&1
else
    echo "==== $(date '+%F %T') starting a new run" | tee -a "$log"
    python main.py --base configs/tcsinger2.yaml -t --gpus 0, --no-test true --scale_lr false "$@" >> "$log" 2>&1
fi
echo "==== $(date '+%F %T') main.py exited" | tee -a "$log"
