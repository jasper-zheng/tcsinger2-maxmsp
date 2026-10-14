#!/usr/bin/env bash
# Create (or refresh) a venv for inference on macOS (Apple silicon: the models run on MPS) and download
# GTSinger's MFA pronunciation dictionaries to data/mfa_dict (ldm/infer/g2p.py finds them there).
# The checkpoints are not downloaded: copy useful_ckpts/ from the training machine.
# Usage: bash scripts/setup_env_mac.sh        then: source .venv/bin/activate
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${TCSINGER2_VENV:-$REPO_DIR/.venv}"
GTSINGER_COMMIT=0aebbed687ede804c8349753fd652bf3afc5eb39  # AaronZ345/GTSinger, main on 2026-10-14

if command -v uv >/dev/null; then
    [ -x "$VENV_DIR/bin/python" ] || uv venv --python 3.11 "$VENV_DIR"
    PIP="uv pip install --python $VENV_DIR/bin/python"
else
    [ -x "$VENV_DIR/bin/python" ] || python3.11 -m venv "$VENV_DIR"
    PIP="$VENV_DIR/bin/python -m pip install"
fi
$PIP -r "$REPO_DIR/requirements-mac.txt"
# PyPI's taming-transformers 0.0.1 lacks taming.modules.*; this fork has them.
$PIP --no-deps taming-transformers-rom1504

DICT_DIR="$REPO_DIR/data/mfa_dict"
mkdir -p "$DICT_DIR"
for lang in english french german italian japanese korean russian spanish; do
    f="$DICT_DIR/${lang}_mfa.dict"
    [ -s "$f" ] || curl -fsSL -o "$f" \
        "https://raw.githubusercontent.com/AaronZ345/GTSinger/$GTSINGER_COMMIT/Data-Process/utils/mfa_dict/${lang}_mfa.dict"
done

"$VENV_DIR/bin/python" - <<'EOF'
import torch, pytorch_lightning as pl
print(f"torch {torch.__version__} | mps {torch.backends.mps.is_available()} | pytorch-lightning {pl.__version__}")
EOF
for f in tcsinger2_gtsinger.ckpt vae_gtsinger.ckpt flan-t5-large hifigan rmvpe/model.pt; do
    [ -e "$REPO_DIR/useful_ckpts/$f" ] || echo "missing: useful_ckpts/$f (copy it from the training machine)"
done
echo "OK: activate with  source $VENV_DIR/bin/activate"
