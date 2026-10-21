#!/usr/bin/env bash
# Linux environment for RTX 5090 with CUDA 12.8.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Use Linux (or a configured WSL2 GPU environment) for LIBERO." >&2
  exit 1
fi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[robot,dev]'
python -m pip check
echo 'Installation finished. Activate .venv and follow docs/RUN_5090.md.'
