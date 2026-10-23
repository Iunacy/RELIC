#!/usr/bin/env bash
# Explicit isolated install; never invoked by preflight or qualification.
# Prerequisite: existing Ubuntu 22.04 x86_64 GPU host, supported NVIDIA driver,
# >=32 GB RAM, >=16 GB RTX VRAM, and uv or Python 3.11. No provisioning/driver changes.
set -euo pipefail
cd "$(dirname "$0")/.."
root="${RELIC_ISAAC_ROOT:-$PWD/embodiment/cache/runtime}"
mkdir -p "$root"
if [[ ! -d "$root/venv" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.11.17 --seed "$root/venv"
    else
        python3.11 -m venv "$root/venv"
    fi
fi
py="$root/venv/bin/python"
"$py" -m pip install 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
"$py" -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com -c embodiment/runtime_constraints.txt
if [[ ! -d "$root/IsaacLab-2.3.0" ]]; then
    git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$root/IsaacLab-2.3.0"
fi
[[ "$(git -C "$root/IsaacLab-2.3.0" rev-parse HEAD)" == 3c6e67bb5c7ada942a6d1884ab69338f57596f77 ]]

"$py" -m pip install --no-build-isolation 'flatdict==4.0.1'
"$py" -m pip install -e "$root/IsaacLab-2.3.0/source/isaaclab" -e "$root/IsaacLab-2.3.0/source/isaaclab_assets" -c embodiment/runtime_constraints.txt
"$py" -m pip check
"$py" -m pip freeze > "$root/resolved-requirements.txt"
printf 'Activate with: source %q\n' "$root/venv/bin/activate"
