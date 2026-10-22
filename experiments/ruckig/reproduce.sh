#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
sdk=${MUJOCO_ROOT:-/home/iris/.local/opt/mujoco-3.16.0}
out=${1:-$root/build-ruckig/results}
sh "$here/verify_upstream.sh"
cmake -S "$root" -B "$root/build-baseline" -DMUJOCO_ROOT="$sdk" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build-baseline" -j 4
ctest --test-dir "$root/build-baseline" --output-on-failure
cmake -S "$root" -B "$root/build-ruckig" -DMUJOCO_ROOT="$sdk" -DCMAKE_BUILD_TYPE=Release -DRELIC_RUCKIG_EXPERIMENT=ON
cmake --build "$root/build-ruckig" -j 4
ctest --test-dir "$root/build-ruckig" --output-on-failure
# This executable re-runs qualification before creating any actuator execution.
"$root/build-ruckig/experiments/ruckig/relic_ruckig_benchmark" "$root/examples" "$out"

"$root/build-ruckig/experiments/ruckig/relic_ruckig_diagnostics" "$root/examples" "$out/diagnostics"
python3 "$here/analyze_diagnostics.py" "$out/diagnostics" --baseline "$here/results/comparison.csv"
