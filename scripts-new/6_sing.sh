#!/usr/bin/env bash
# Step 6: sing a score with the models the pipeline trained (py/sing.py with the exports in $TCS_WORK/ckpts and
# the pipeline's phone set and lexicon). All arguments go to py/sing.py; paths are relative to the repository root.
#
# Usage (from the repository root):
#   bash scripts-new/6_sing.sh --score examples/twinkle_en.json --out samples/twinkle.wav --singer "female alto"
#   bash scripts-new/6_sing.sh --score examples/twinkle_en.json --out samples/twinkle_f0gen.wav --f0_model work/ckpts/f0gen.ckpt
#   bash scripts-new/6_sing.sh --score examples/f0_vibrato_it.json --out samples/given_f0.wav --prompt voice.wav
#   bash scripts-new/6_sing.sh --help
source "$(dirname "$0")/lib.sh"

need "$CKPTS/tcsinger2.ckpt" "run scripts-new/4_train_stage2.sh first"
python scripts-new/py/sing.py --ckpt "$CKPTS/tcsinger2.ckpt" --config "$CKPTS/tcsinger2.yaml" "$@"
