#!/usr/bin/env bash
# BIZRA local commit + ship gate coordinator.
#
# Default mode is validation-only. Use --commit to create Lane A/Lane B commits.
# This script never pushes and never creates a PR.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel)"
cd "$REPO_ROOT"

COMMIT=false
RUN_FULL_PYTEST=false
FULL_PYTEST_LOG="${FULL_PYTEST_LOG:-/tmp/bizra-full-pytest.log}"
BASE_BRANCH="${BIZRA_PR_BASE_BRANCH:-feat/economic-constitution-v1}"

usage() {
  cat <<'USAGE'
Usage: scripts/ship/commit_and_ship.sh [--commit] [--full-pytest]

Options:
  --commit        Commit Lane A and/or Lane B if their files are dirty.
  --full-pytest   Run full pytest fail-closed with tee + pipefail.
  -h, --help      Show this help.

Environment:
  BIZRA_PR_BASE_BRANCH  Branch that PRs should target. Current branch must differ.
  FULL_PYTEST_LOG       Log path for --full-pytest (default: /tmp/bizra-full-pytest.log).

Safety:
  - No git push.
  - No gh pr create.
  - No daemon, mission, Node1, public demo, external routing, or economic claim.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      COMMIT=true
      ;;
    --full-pytest)
      RUN_FULL_PYTEST=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f .venv/bin/activate ]]; then
  printf 'ERROR: .venv/bin/activate not found under %s\n' "$REPO_ROOT" >&2
  exit 1
fi
source .venv/bin/activate

LANE_A_FILES=(
  docs/provenance/BIZRA_GENESIS_PROVENANCE_LEDGER_V0_1.md
  docs/product/DEMA_PRODUCT_CONSTITUTION_V0_1.md
  docs/skills/DEMA_SAFE_MONETIZATION_SKILL_V0_1.md
  .proof-forge/EVIDENCE_INDEX.json
  .proof-forge/receipts/2026-05-04_032654.json
  .proof-forge/summaries/2026-05-04_032654.md
  PROOF_SUMMARY.md
)

LANE_B_FILES=(
  bizra-omega/bizra-python/src/lib.rs
  core/vault/vault.py
  tests/core/test_rust_bridge.py
  tests/e2e_http/test_pyo3_bridge.py
)

has_changes() {
  git status --porcelain -- "$@" | grep -q .
}

require_branch_safety() {
  local branch
  branch="$(git branch --show-current)"
  if [[ -z "$branch" ]]; then
    printf 'ERROR: detached HEAD is not a safe ship branch.\n' >&2
    exit 1
  fi
  if [[ "$branch" == main || "$branch" == master ]]; then
    printf 'ERROR: refuse to ship from protected branch %s.\n' "$branch" >&2
    exit 1
  fi
  if [[ "$branch" == "$BASE_BRANCH" ]]; then
    printf 'ERROR: current branch equals PR base (%s). Create a real head branch.\n' \
      "$BASE_BRANCH" >&2
    printf 'Example: git switch -c ship/node0-proof-product\n' >&2
    exit 1
  fi
}

require_operator_key_absent() {
  if [[ -e .proof-forge/operator_key.json ]]; then
    printf 'ERROR: .proof-forge/operator_key.json is present; remove it before ship.\n' >&2
    exit 1
  fi
}

run_full_pytest() {
  printf '\n== Full pytest (fail-closed) ==\n'
  rm -f "$FULL_PYTEST_LOG"
  set -o pipefail
  python -m pytest -q --timeout=60 2>&1 | tee "$FULL_PYTEST_LOG"
}

commit_lane_a() {
  if ! has_changes "${LANE_A_FILES[@]}"; then
    printf 'Lane A: no dirty provenance/proof files.\n'
    return 0
  fi
  for file in "${LANE_A_FILES[@]}"; do
    [[ -e "$file" ]] || {
      printf 'ERROR: Lane A expected file missing: %s\n' "$file" >&2
      exit 1
    }
  done
  git add "${LANE_A_FILES[@]}"
  git diff --cached --check
  git commit -m 'docs(provenance): seal BIZRA genesis governance package' \
    -m 'Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'
}

commit_lane_b() {
  if ! has_changes "${LANE_B_FILES[@]}"; then
    printf 'Lane B: no dirty repair files.\n'
    return 0
  fi
  scripts/ship/validate_repair_patches.sh
  git add "${LANE_B_FILES[@]}"
  git diff --cached --check
  git commit -m 'fix(node0): harden Rust bridge and vault validation' \
    -m 'Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'
}

run_ship_gates() {
  printf '\n== Ship gates ==\n'
  git merge-base --is-ancestor b08f2208 HEAD
  python scripts/ci_secret_scan.py
  python tools/proof_forge/forge_evidence.py --verify --allow-legacy-unsigned --project-dir .
  make gate
  make spearpoint
  python .claude/skills/cross-lang-sync/audit_constants.py
  git diff --check
  if "$RUN_FULL_PYTEST"; then
    run_full_pytest
  else
    printf 'Full pytest: NOT RUN. Use --full-pytest before claiming full-suite green.\n'
  fi
}

require_branch_safety
require_operator_key_absent

if "$COMMIT"; then
  commit_lane_a
  commit_lane_b
else
  printf 'Commit mode disabled; validating only. Pass --commit to create local commits.\n'
fi

run_ship_gates

printf '\nPASS: local ship coordinator completed without push or PR creation.\n'
printf 'PR branch note: current branch is a head branch; target base should be %s.\n' \
  "$BASE_BRANCH"
