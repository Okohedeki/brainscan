#!/usr/bin/env bash
# Run a brainscan module inside the WSL venv from the project root.
#   scripts/run.sh -m brainscan.server
#   scripts/run.sh -m brainscan.harness.mini_agent --task fix_fizzbuzz
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.venvs/brainscan/bin/activate"
exec python "$@"
