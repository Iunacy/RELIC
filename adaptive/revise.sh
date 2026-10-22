#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
relic_ml_python="${RELIC_ML_PYTHON:-/home/iris/.local/share/venvs/robotics-ml/bin/python}"
relic_sdk="${MUJOCO_ROOT:-${MUJOCO_SDK:-/home/iris/.local/opt/mujoco-3.16.0}}"
cmake -S . -B build-adaptive-validation -DRELIC_ADAPTIVE=ON -DCMAKE_BUILD_TYPE=Release -DMUJOCO_ROOT="$relic_sdk"
cmake --build build-adaptive-validation -j 4
ctest --test-dir build-adaptive-validation --output-on-failure -j 4
python adaptive/revision.py qualify
python adaptive/revision.py teacher-qualify
python adaptive/revision.py collect
python adaptive/revision.py tune
python adaptive/revision.py convergence
python adaptive/revision/convergence_replication.py
"$relic_ml_python" adaptive/revision/convergence_report.py
if [[ ! -f adaptive/runs/revision-20261022/prepared/manifest.json ]]; then
  "$relic_ml_python" adaptive/revision.py prepare
fi
"$relic_ml_python" adaptive/revision.py train
"$relic_ml_python" adaptive/revision.py evaluate
"$relic_ml_python" adaptive/revision.py report
"$relic_ml_python" adaptive/revision_plot.py adaptive/runs/revision-20261022
python adaptive/revision_demo.py
"$relic_ml_python" adaptive/revision/analytical_robustness.py
