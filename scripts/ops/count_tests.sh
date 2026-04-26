#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

if [ -d ".venv-linux" ]; then
    # shellcheck disable=SC1091
    source .venv-linux/bin/activate
elif [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

if python3 -m pytest --collect-only -q tests/ -m "not slow" --ignore=tests/root_legacy --timeout=30 >"$LOG_FILE" 2>&1; then
    status=0
else
    status=$?
fi
tail -5 "$LOG_FILE"
exit "$status"
