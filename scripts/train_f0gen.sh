#!/usr/bin/env bash
# Trains the F0 generator (WP10): starts the run of a config, or resumes the newest logs/*_<config name> run
# that has a checkpoint. Runs in the foreground; output is appended to logs/<config name>_train.log. Extra
# arguments go to main.py, e.g. `--max_steps 150000` or `data.params.max_items=64`. A run whose rolling
# checkpoint step=<max_steps>.ckpt exists has finished and is skipped.
#
# Usage: nohup bash scripts/train_f0gen.sh configs/f0gen.yaml [main.py arguments] > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
config=$1; shift
name=$(basename "$config" .yaml)
mkdir -p logs
log=logs/${name}_train.log

run=""
while IFS= read -r d; do
    if [ -n "$(find "$d/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit 2>/dev/null)" ]; then run="$d"; break; fi
done < <(ls -dt logs/*_"$name" 2>/dev/null || true)

# a run that has reached the config's max_steps is finished (unless --max_steps is passed to train further)
max_steps=$(grep -E '^ *max_steps:' "$config" | head -1 | awk '{print $2}')
if [ -n "$run" ] && [[ " $* " != *" --max_steps "* ]] && [ -e "$run/checkpoints/step=$(printf '%08d' "$max_steps").ckpt" ]; then
    echo "==== $(date '+%F %T') $run has finished ($max_steps steps); nothing to do" | tee -a "$log"
    exit 0
fi

if [ -n "$run" ]; then
    echo "==== $(date '+%F %T') resuming $run" | tee -a "$log"
    python main.py -r "$run" -t --gpus 0, --no-test true --scale_lr false "$@" >> "$log" 2>&1
else
    echo "==== $(date '+%F %T') starting a new run of $config" | tee -a "$log"
    python main.py --base "$config" -t --gpus 0, --no-test true --scale_lr false "$@" >> "$log" 2>&1
fi
echo "==== $(date '+%F %T') main.py exited" | tee -a "$log"
