#!/usr/bin/env bash
# Evaluates the milestone checkpoints of a VAE run (WP4) with scripts/eval_vae.py, one after the other
# (about 4 min each). A milestone whose samples/wp4_recon/step<N>/metrics.json already has the current
# metrics is skipped, so this can be rerun after a shutdown. Output is appended to logs/vae_eval.log.
#
# Usage: bash scripts/eval_vae_milestones.sh [run directory] [global steps, default: all milestones]
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
run=${1:-logs/2026-10-09T00-11-41_ae_singing}
shift || true
log=logs/vae_eval.log

if [ $# -gt 0 ]; then
    ckpts=(); for s in "$@"; do ckpts+=("$run"/checkpoints/trainstep_checkpoints/*-step="$(printf '%09d' "$s")".ckpt); done
else
    ckpts=("$run"/checkpoints/trainstep_checkpoints/*.ckpt)
fi
for ckpt in "${ckpts[@]}"; do
    step=$(basename "$ckpt" .ckpt); step=$((10#${step##*step=}))
    if grep -q detail_ratio_freq "samples/wp4_recon/step$(printf '%08d' "$step")/metrics.json" 2>/dev/null; then
        echo "==== $(date '+%F %T') skipping $ckpt (evaluated)" | tee -a "$log"
        continue
    fi
    echo "==== $(date '+%F %T') evaluating $ckpt" | tee -a "$log"
    python scripts/eval_vae.py --ckpt "$ckpt" >> "$log" 2>&1
done
echo "==== $(date '+%F %T') done" | tee -a "$log"
