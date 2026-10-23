#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
relic_viewer_environment="${RELIC_VIEWER_ENV:-$HOME/.local/share/relic-viewer-venv}"
if [[ ! -x "$relic_viewer_environment/bin/python" ]]; then
    uv venv --python 3.12 "$relic_viewer_environment"
fi
uv pip install --python "$relic_viewer_environment/bin/python" -r interactive/research_requirements.txt
"$relic_viewer_environment/bin/python" -c 'from embodiment.replay_render import panda_asset, panda_visuals; panda_visuals(panda_asset())'
