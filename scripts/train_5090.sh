#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mealvla train --config "${1:-configs/train_5090.json}"
