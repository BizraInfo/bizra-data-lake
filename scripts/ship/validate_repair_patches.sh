#!/usr/bin/env bash
# BIZRA focused repair patch validation.
#
# This script validates Lane B repair files without starting Node0 runtime,
# dispatching missions, activating Node1, or routing to external providers.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ ! -f .venv/bin/activate ]]; then
  printf 'ERROR: .venv/bin/activate not found under %s\n' "$REPO_ROOT" >&2
  exit 1
fi
source .venv/bin/activate

REPAIR_FILES=(
  bizra-omega/bizra-python/src/lib.rs
  core/vault/vault.py
  tests/core/test_rust_bridge.py
  tests/e2e_http/test_pyo3_bridge.py
)

PASS=0
FAIL=0
TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_OUT"' EXIT

check() {
  local name="$1"
  shift
  printf '  %-52s' "$name"
  if "$@" >"$TMP_OUT" 2>&1; then
    printf 'PASS\n'
    PASS=$((PASS + 1))
  else
    printf 'FAIL\n'
    tail -20 "$TMP_OUT" | sed 's/^/    /'
    FAIL=$((FAIL + 1))
  fi
}

check_optional_pytest() {
  local name="$1"
  local path="$2"
  shift 2
  if [[ ! -e "$path" ]]; then
    printf '  %-52sSKIP (missing: %s)\n' "$name" "$path"
    return 0
  fi
  check "$name" python -m pytest "$path" "$@"
}

printf '%s\n\n' '== Repair Patch Focused Validation =='

check 'ruff: Lane B Python files' \
  ruff check core/vault/vault.py tests/core/test_rust_bridge.py \
    tests/e2e_http/test_pyo3_bridge.py

check 'black: Lane B Python files' \
  black --check core/vault/vault.py tests/core/test_rust_bridge.py \
    tests/e2e_http/test_pyo3_bridge.py

check 'cargo check: bizra-omega workspace' \
  cargo check --workspace --manifest-path bizra-omega/Cargo.toml

check 'pytest: Rust bridge focused tests' \
  python -m pytest tests/core/test_rust_bridge.py -q --timeout=60

check 'pytest: PyO3 bridge focused tests' \
  python -m pytest tests/e2e_http/test_pyo3_bridge.py -q --timeout=60

check_optional_pytest 'pytest: legacy vault tests' \
  tests/core/test_vault.py -q --timeout=60

check_optional_pytest 'pytest: vault property tests' \
  tests/property_based/test_vault_snr_properties.py -q --timeout=60

check 'git diff --check: Lane B files' \
  git diff --check -- "${REPAIR_FILES[@]}"

printf '\nPassed: %d  Failed: %d\n' "$PASS" "$FAIL"

if [[ "$FAIL" -gt 0 ]]; then
  printf 'DO-NOT-COMMIT: repair validation failed.\n' >&2
  exit 1
fi

printf 'PASS: repair validation is clean.\n'
