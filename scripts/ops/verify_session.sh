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

run_pytest_with_tail() {
    local lines="$1"
    shift
    local log_file
    log_file="$(mktemp)"
    if "$@" >"$log_file" 2>&1; then
        tail -"${lines}" "$log_file"
        rm -f "$log_file"
        return 0
    fi

    local status=$?
    tail -"${lines}" "$log_file"
    rm -f "$log_file"
    return "$status"
}

echo "=== ENTROPY ROUTER TESTS ==="
run_pytest_with_tail 5 python3 -m pytest tests/core/reasoning/test_entropy_router.py -q --timeout=30
echo ""
echo "=== SAT MODULE IMPORT ==="
python3 -c "
from core.sat.mint_court import MintCourt, MintPhase, MintVerdict, EvidenceSnapshot, SATScorecard, MintDistribution, MintReceipt
print('MintCourt imports: OK')
print(f'Phases: {[p.value for p in MintPhase]}')
print(f'Verdicts: {[v.value for v in MintVerdict]}')
" 2>&1
echo ""
echo "=== SAT GATE TESTS ==="
run_pytest_with_tail 5 python3 -m pytest tests/core/sat/ -q --timeout=30
echo ""
echo "=== GIT STATUS ==="
git log --oneline -5
git diff --stat HEAD~1
