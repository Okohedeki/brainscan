#!/usr/bin/env bash
# Run a brainscan module inside the WSL venv from the project root.
#   scripts/run.sh -m brainscan.server
#   scripts/run.sh -m brainscan.harness.mini_agent --task fix_fizzbuzz
# Machine-specific settings (e.g. BRAINSCAN_MODEL) go in brainscan.env, which is not committed.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f brainscan.env ]; then
  set -a
  # shellcheck disable=SC1091
  source brainscan.env
  set +a
fi
source "$HOME/.venvs/brainscan/bin/activate"
exec python "$@"
