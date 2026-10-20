#!/usr/bin/env bash
# Steps 1 to 5 in order. Every step skips what has finished, so after an interruption (or a shutdown) the same
# command carries on where the work stopped. Each step logs to $TCS_WORK/logs/<step>.log.
#
# Usage: nohup bash scripts-new/run_all.sh > /dev/null 2>&1 &
set -euo pipefail
dir="$(dirname "$0")"
bash "$dir/1_data_gtsinger.sh"
bash "$dir/2_data_extra.sh"
bash "$dir/3_train_vae.sh"
bash "$dir/4_train_stage2.sh"
bash "$dir/5_train_f0gen.sh"
