#!/usr/bin/env bash
# Step 0: Python environment, pretrained models and pronunciation dictionaries.
#
# - Installs scripts-new/requirements.txt (Python 3.11; torch 2.1.1 with CUDA 12.1 from the PyTorch index; pyworld
#   is built from source and needs a C compiler) into TCS_VENV, which is created if it does not exist, or into the
#   active environment if TCS_VENV is empty. TCS_SYSTEM_SITE=1 creates the venv on top of the system's packages,
#   e.g. to reuse an image's torch.
# - Downloads into useful_ckpts/ what is not there yet: FLAN-T5-large (text encoder) and the RMVPE pitch tracker.
#   The 48 kHz HiFi-GAN vocoder is on Google Drive and has to be downloaded by hand (the script says where).
# - Downloads GTSinger's MFA pronunciation dictionaries for the inference front-end into $TCS_WORK/data/mfa_dict.
#
# Usage: bash scripts-new/0_setup.sh
source "$(dirname "$0")/lib.sh"
log_to setup

if [ -n "$TCS_VENV" ] && [ ! -x "$TCS_VENV/bin/python" ]; then
    python3 -m venv ${TCS_SYSTEM_SITE:+--system-site-packages} "$TCS_VENV"
    set +u; source "$TCS_VENV/bin/activate"; set -u
fi
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r scripts-new/requirements.txt
# PyPI's taming-transformers 0.0.1 lacks taming.modules.*; this fork has them. Its dependencies are installed above.
pip install --no-deps taming-transformers-rom1504
python -c "
import torch, pytorch_lightning as pl
print(f'torch {torch.__version__} | cuda {torch.cuda.is_available()} | pytorch-lightning {pl.__version__}')
assert torch.__version__.startswith('2.1.1'), 'expected torch 2.1.1'
"

python - <<'EOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download

if not os.path.exists('useful_ckpts/flan-t5-large/model.safetensors'):
    snapshot_download('google/flan-t5-large', local_dir='useful_ckpts/flan-t5-large', local_dir_use_symlinks=False,
                      allow_patterns=['*.json', 'spiece.model', 'model.safetensors'])
if not os.path.exists('useful_ckpts/rmvpe/model.pt'):
    hf_hub_download('verstar/STARS', 'rmvpe/model.pt', local_dir='useful_ckpts', local_dir_use_symlinks=False)
EOF
if [ ! -e useful_ckpts/hifigan/config.yaml ] || [ -z "$(ls useful_ckpts/hifigan/*.ckpt 2>/dev/null)" ]; then
    echo "missing: the HiFi-GAN vocoder. Download config.yaml and model_ckpt_steps_1000000.ckpt from"
    echo "  https://drive.google.com/drive/folders/1ve9cm_Yn3CQWSqkzMuRL33Uj1BNh51lR"
    echo "into useful_ckpts/hifigan/."
fi

# GTSinger's pronunciation dictionaries (AaronZ345/GTSinger, main on 2026-10-14)
GTSINGER_COMMIT=0aebbed687ede804c8349753fd652bf3afc5eb39
mkdir -p "$DATA/mfa_dict"
for lang in english french german italian japanese korean russian spanish; do
    f="$DATA/mfa_dict/${lang}_mfa.dict"
    [ -s "$f" ] || curl -fsSL -o "$f" \
        "https://raw.githubusercontent.com/AaronZ345/GTSinger/$GTSINGER_COMMIT/Data-Process/utils/mfa_dict/${lang}_mfa.dict"
done
log "setup done${TCS_VENV:+; set TCS_VENV=$TCS_VENV in scripts-new/config.sh if it is not set there}"
