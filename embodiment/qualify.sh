#!/usr/bin/env bash
# Run inside the isolated pinned Python 3.11 environment on an existing GPU host.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m embodiment.preflight --output embodiment/evidence/preflight.json
python -m unittest discover -s embodiment/tests -v
python -m embodiment.assets
python -m embodiment.qualify --output "embodiment/runs/$(date -u +%Y%m%dT%H%M%SZ)" "$@"
