#!/usr/bin/env bash
# WP10 training, resumable as a whole: the flow run (configs/f0gen.yaml), then the regression control
# (configs/f0gen_reg.yaml). Each is started, resumed from its newest checkpoint, or skipped when finished, so
# after a shutdown the same command carries on where the work stopped. Logs: logs/f0gen_train.log,
# logs/f0gen_reg_train.log.
#
# Usage: nohup bash scripts/train_wp10.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/train_f0gen.sh configs/f0gen.yaml
bash scripts/train_f0gen.sh configs/f0gen_reg.yaml
