#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
relic_python="${RELIC_ML_PYTHON:-$HOME/.local/share/venvs/robotics-ml/bin/python}"
relic_mujoco="${MUJOCO_ROOT:-$HOME/.local/opt/mujoco-3.16.0}"
"$relic_python" -c 'import mujoco,numpy,PIL; assert mujoco.__version__ == "3.16.0"'
relic_build="${RELIC_INTERACTIVE_BUILD:-build-product}"
cmake -S . -B "$relic_build" -DRELIC_INTERACTIVE=ON -DRELIC_ADAPTIVE=ON -DMUJOCO_ROOT="$relic_mujoco" -DPython3_EXECUTABLE="$relic_python" -DCMAKE_BUILD_TYPE=Release
cmake --build "$relic_build" -j4
