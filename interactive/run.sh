#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
relic_python="${RELIC_ML_PYTHON:-$HOME/.local/share/venvs/robotics-ml/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export RELIC_INTERACTIVE_EXE="${RELIC_INTERACTIVE_EXE:-${RELIC_INTERACTIVE_BUILD:-build-product}/relic_interactive}"
exec "$relic_python" -m interactive.app "$@"
