#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
relic_python="${RELIC_ML_PYTHON:-$HOME/.local/share/venvs/robotics-ml/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export RELIC_INTERACTIVE_EXE="${RELIC_INTERACTIVE_EXE:-${RELIC_INTERACTIVE_BUILD:-build-product}/relic_interactive}"
if [[ "${1:-}" == "--legacy" ]]; then
    shift
    exec "$relic_python" -m interactive.app "$@"
fi
relic_viewer_python="${RELIC_VIEWER_PYTHON:-$HOME/.local/share/relic-viewer-venv/bin/python}"
if [[ ! -x "$relic_viewer_python" ]]; then
    relic_viewer_python="$relic_python"
fi
exec "$relic_viewer_python" -m interactive.research "$@"
