#!/usr/bin/env bash
# Step 5: the F0 generator for notes-only synthesis -> $TCS_WORK/ckpts/f0gen.ckpt.
# Needs only the data of steps 1 and 2 (not the VAE or stage 2), so it can also run before them.
#
# configs/f0gen.yaml: a flow-matching transformer (8 layers, hidden 384) that makes an F0 curve from the notes,
# phonemes and language and the F0 of a prompt clip, with the vibrato and spread of sung curves that stage 2's own
# predictor lacks; 100,000 steps (about 6.5 h on an A6000). At inference, sing.py --f0_model uses it; stage 2 then
# sings its curve as a given F0. The released f0gen_gtsinger.ckpt is the milestone of step 90,000, which was
# chosen by listening; this step exports the final one (the two measured the same).
#
# Every F0 of the manifests is read into one cache file ($TCS_WORK/features/f0gen_cache) on the first start.
# Starts, resumes or skips the run in $TCS_WORK/logs/<time>_f0gen as step 3 does; then exports the EMA weights.
# Extra arguments go to main.py. Output is appended to $TCS_WORK/logs/train_f0gen.log.
#
# Usage: nohup bash scripts-new/5_train_f0gen.sh [main.py arguments] > /dev/null 2>&1 &
source "$(dirname "$0")/lib.sh"
log_to train_f0gen

train configs/f0gen.yaml false "data.params.main_spec_dir_path=$(manifest_dirs)" -- "$@"
export_run f0gen "$CKPTS/f0gen.ckpt"
