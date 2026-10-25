#!/usr/bin/env bash
# Step 3: the mel VAE (stage 1), on GTSinger -> $TCS_WORK/ckpts/vae.ckpt.
#
# configs/ae_singing.yaml: 80 mel bins to 20 channels at 4x temporal compression, with a duration predictor;
# 16 clips of 2,000 frames per batch, reconstruction + KL until batch 40,000, adversarial loss from there,
# 70,000 batches in all (about 17 h on an A6000). Lightning counts a step per optimizer, so the step numbers of
# the config, the checkpoints and TensorBoard are twice the batch numbers (max_steps 140000).
#
# Starts the run in $TCS_WORK/logs/<time>_ae_singing, or resumes it from its newest checkpoint (every 1,000
# batches; milestones every 10,000 in checkpoints/trainstep_checkpoints), or skips it when it has finished;
# then exports the final weights. Extra arguments go to main.py, e.g. --max_steps 160000 to train further.
# Output is appended to $TCS_WORK/logs/train_vae.log.
#
# Usage: nohup bash scripts-new/3_train_vae.sh [main.py arguments] > /dev/null 2>&1 &
source "$(dirname "$0")/lib.sh"
log_to train_vae

need "$DATA/sing/new.tsv" "run scripts-new/1_data_gtsinger.sh first"
# the learning rate is the config's 4.5e-6 x batch size (--scale_lr true)
train configs/ae_singing.yaml true "data.params.spec_dir_path=$DATA/sing" -- "$@"
export_run vae "$CKPTS/vae.ckpt"
