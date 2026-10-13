#!/usr/bin/env bash
# Trains the VAE (WP4): starts the run, or resumes the newest logs/*_ae_singing run that has a checkpoint.
# Runs in the foreground; output is appended to logs/vae_train.log. Extra arguments go to main.py,
# e.g. `model.params.lossconfig.params.disc_start=100001` or `--max_steps 160000` (both in global
# steps = 2 x batches).
#
# Usage: bash scripts/train_vae.sh [main.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
mkdir -p logs
log=logs/vae_train.log

run=""
while IFS= read -r d; do
    if [ -n "$(find "$d/checkpoints" -name '*.ckpt' -print -quit 2>/dev/null)" ]; then run="$d"; break; fi
done < <(ls -dt logs/*_ae_singing 2>/dev/null || true)

if [ -n "$run" ]; then
    echo "==== $(date '+%F %T') resuming $run" | tee -a "$log"
    python main.py -r "$run" -t --gpus 0, --no-test true "$@" >> "$log" 2>&1
else
    echo "==== $(date '+%F %T') starting a new run" | tee -a "$log"
    python main.py --base configs/ae_singing.yaml -t --gpus 0, --no-test true "$@" >> "$log" 2>&1
fi
echo "==== $(date '+%F %T') main.py exited" | tee -a "$log"
