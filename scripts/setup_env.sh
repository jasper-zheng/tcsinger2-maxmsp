#!/usr/bin/env bash
# Create (or refresh) the persistent TCSinger2 venv on /notebooks.
# The venv inherits the image's torch/CUDA stack via --system-site-packages and only adds what is missing.
# Usage: bash scripts/setup_env.sh        then: source /notebooks/envs/tcsinger2/bin/activate
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${TCSINGER2_VENV:-/notebooks/envs/tcsinger2}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/python -m pip"
$PIP install --no-cache-dir -r "$REPO_DIR/requirements-env.txt" -c "$REPO_DIR/constraints-env.txt"
# PyPI's taming-transformers 0.0.1 lacks taming.modules.*; this fork has them. Its deps are already satisfied.
$PIP install --no-cache-dir --no-deps taming-transformers-rom1504

"$VENV_DIR/bin/python" - <<'EOF'
import torch, pytorch_lightning as pl
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | pytorch-lightning {pl.__version__}")
assert torch.__version__.startswith("2.1.1"), "torch was replaced inside the venv"
EOF
echo "OK: activate with  source $VENV_DIR/bin/activate"
