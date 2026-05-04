#!/usr/bin/env bash
set -Eeuo pipefail

# BIZRA Node0 Rust Bus Bridge Bootstrap v1
# Purpose:
#   Build/install the PyO3 Rust Bus bridge into the exact Python venv used by
#   Node0.
#
# Safety:
#   - Does not start daemon
#   - Does not run mission
#   - Does not print secrets
#   - Does not touch Node1 / public demo / publish flow

REPO_ROOT="${REPO_ROOT:-/data/bizra/repos/bizra-data-lake-release-final}"
VENV_PY="${VENV_PY:-/data/bizra/repos/bizra-data-lake/.venv/bin/python}"
VENV_BIN="$(dirname "$VENV_PY")"
VENV_ROOT="$(dirname "$VENV_BIN")"
VENV_PIP="${VENV_PIP:-$VENV_BIN/pip}"
PYO3_DIR="${PYO3_DIR:-$REPO_ROOT/bizra-omega/bizra-python}"
LM_STUDIO_URL="${LM_STUDIO_URL:-http://127.0.0.1:1234}"
BIZRA_BROWSER_MODE="${BIZRA_BROWSER_MODE:-mock}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "[node0-rust-bus] $*"
}

require_path() {
  local path="$1"
  [[ -e "$path" ]] || fail "missing required path: $path"
}

info "repo root: $REPO_ROOT"
info "python:    $VENV_PY"
info "pyo3 dir:  $PYO3_DIR"

require_path "$REPO_ROOT"
require_path "$VENV_PY"
require_path "$VENV_PIP"
require_path "$PYO3_DIR/Cargo.toml"
require_path "$PYO3_DIR/pyproject.toml"

cd "$PYO3_DIR"

info "checking Python version"
"$VENV_PY" - <<'PY'
import sys

print("python_executable=", sys.executable)
print("python_version=", sys.version.split()[0])
PY

info "checking maturin"
if ! "$VENV_PY" -m maturin --version >/dev/null 2>&1; then
  info "maturin missing; installing into Node0 venv"
  "$VENV_PY" -m pip install maturin
else
  "$VENV_PY" -m maturin --version
fi

info "checking patchelf"
if ! command -v patchelf >/dev/null 2>&1 && [[ ! -x "$VENV_BIN/patchelf" ]]; then
  info "patchelf missing; installing into Node0 venv"
  "$VENV_PY" -m pip install patchelf
else
  info "patchelf available"
fi

info "building/installing PyO3 bridge into Node0 venv"
PATH="$VENV_BIN:$PATH" \
  VIRTUAL_ENV="$VENV_ROOT" \
  "$VENV_PY" -m maturin develop --release --pip-path "$VENV_PIP"

info "verifying Python import: from bizra import PyEventBridge"
"$VENV_PY" - <<'PY'
from bizra import PyEventBridge

print("PyEventBridge=FOUND")
print("module_import=PASS")
PY

info "running non-mutating Node0 status"
cd "$REPO_ROOT"

LM_STUDIO_URL="$LM_STUDIO_URL" \
BIZRA_BROWSER_MODE="$BIZRA_BROWSER_MODE" \
"$VENV_PY" scripts/node0_activate.py status

info "completed Rust Bus bridge bootstrap"
