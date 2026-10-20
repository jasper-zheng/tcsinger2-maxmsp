#!/usr/bin/env bash
# Step 4: the flow-matching transformer (stage 2) -> $TCS_WORK/ckpts/tcsinger2.ckpt and tcsinger2.yaml.
# Needs the VAE of step 3; trains on GTSinger plus the extra corpora if step 2 built them.
#
# configs/tcsinger2.yaml: notes, phonemes, language, caption (FLAN-T5) and an audio prompt in; F0 optional
# (the model predicts it from the notes when none is given). Batches of up to 48,000 frames; each epoch takes
# every GTSinger training clip and a new random subset of the extra corpora worth 40% of the frames; AdamW 5e-5
# after 10,000 warm-up steps, bf16, EMA; 200,000 steps (about 29 h on an A6000, 17 h on an A100). Every 5,000
# steps it writes 8 fixed validation clips, from notes alone and with their F0, to <run>/samples/.
#
# Starts the run in $TCS_WORK/logs/<time>_tcsinger2, or resumes it from its newest checkpoint (every 1,000 steps;
# milestones every 10,000), or skips it when it has finished; then exports the final weights with the run's
# config. Extra arguments go to main.py, e.g. --max_steps 300000 to train further, or data.params.max_tokens=32000
# for a GPU with less memory. Output is appended to $TCS_WORK/logs/train_stage2.log.
#
# Usage: nohup bash scripts-new/4_train_stage2.sh [main.py arguments] > /dev/null 2>&1 &
source "$(dirname "$0")/lib.sh"
log_to train_stage2

need "$CKPTS/vae.ckpt" "run scripts-new/3_train_vae.sh first"
train configs/tcsinger2.yaml false "data.params.main_spec_dir_path=$(manifest_dirs)" \
    "model.params.first_stage_config.params.ckpt_path=$CKPTS/vae.ckpt" -- "$@"
export_run stage2 "$CKPTS/tcsinger2.ckpt"
