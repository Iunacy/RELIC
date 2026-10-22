#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${RELIC_ADAPTIVE_BUILD:-build-adaptive}"
RUN_DIR="${RELIC_ADAPTIVE_RUN:-adaptive/runs/reproduce}"
PYTHON_BIN="${RELIC_ML_PYTHON:-/home/iris/.local/share/venvs/robotics-ml/bin/python}"
EPISODES="${RELIC_EPISODES:-200}"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DRELIC_ADAPTIVE=ON -DMUJOCO_ROOT="${MUJOCO_ROOT:-/home/iris/.local/opt/mujoco-3.16.0}"
cmake --build "$BUILD_DIR" -j 4
ctest --test-dir "$BUILD_DIR" --output-on-failure -j 4
python3 adaptive/pipeline.py generate --exe "$BUILD_DIR/adaptive/relic_episode" --output "$RUN_DIR/data" --episodes "$EPISODES" --workers 4
python3 adaptive/calibrate.py "$RUN_DIR/data" "$RUN_DIR/bounds"
python3 adaptive/learning.py teachers --dataset "$RUN_DIR/data" --output "$RUN_DIR/teachers" --exe "$BUILD_DIR/adaptive/relic_episode" --bounds "$RUN_DIR/bounds/bounds.txt" --episodes 120 --attempts 3 --workers 3
"$PYTHON_BIN" adaptive/learning.py train --teachers "$RUN_DIR/teachers" --output "$RUN_DIR/nominal" --bounds "$RUN_DIR/bounds/bounds.txt" --categories nominal,recovery
"$PYTHON_BIN" adaptive/learning.py train --teachers "$RUN_DIR/teachers" --output "$RUN_DIR/models" --bounds "$RUN_DIR/bounds/bounds.txt"
python3 adaptive/pipeline.py generate --exe "$BUILD_DIR/adaptive/relic_episode" --output "$RUN_DIR/data" --episodes "${RELIC_EVAL_POOL:-512}" --workers 4
python3 adaptive/evaluate.py --dataset "$RUN_DIR/data" --output "$RUN_DIR/evaluation" --exe "$BUILD_DIR/adaptive/relic_episode" --bounds "$RUN_DIR/bounds/bounds.txt" --models "$RUN_DIR/models" --nominal "$RUN_DIR/nominal" --min-index 320 --episodes 64 --workers 3
"$PYTHON_BIN" adaptive/demo.py --exe "$BUILD_DIR/adaptive/relic_episode" --bounds "$RUN_DIR/bounds/bounds.txt" --model "$RUN_DIR/models/shared.txt" --output "$RUN_DIR/demo" --plot
