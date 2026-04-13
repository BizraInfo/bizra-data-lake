#!/usr/bin/env bash
# BIZRA Promotion Gate — local CI pipeline for MVDA→Core promotions
# Usage: bash scripts/promotion_gate.sh [component_name]
set -euo pipefail

COMPONENT="${1:-all}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/data/bizra/logs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/promotion-gate-${COMPONENT}-${TIMESTAMP}.log"

cd "$REPO_ROOT"
source .venv/bin/activate

echo "=== BIZRA Promotion Gate ===" | tee "$LOG_FILE"
echo "Component: ${COMPONENT}" | tee -a "$LOG_FILE"
echo "Timestamp: $(date -Iseconds)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

PASS=0
FAIL=0

run_check() {
    local name="$1"
    local cmd="$2"
    echo -n "  [${name}] " | tee -a "$LOG_FILE"
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        echo "PASS" | tee -a "$LOG_FILE"
        PASS=$((PASS + 1))
    else
        echo "FAIL" | tee -a "$LOG_FILE"
        FAIL=$((FAIL + 1))
    fi
}

# ── Gate 1: Import chain ──
echo "Gate 1: Import Chain" | tee -a "$LOG_FILE"
run_check "core.integration.constants" "python -c 'from core.integration.constants import UNIFIED_IHSAN_THRESHOLD'"
run_check "core.proof_engine.receipt" "python -c 'from core.proof_engine.receipt import Receipt'"
run_check "core.proof_engine.canonical" "python -c 'from core.proof_engine.canonical import hex_digest'"
run_check "core.proof_engine.evidence_audit" "python -c 'from core.proof_engine.evidence_audit import audit_evidence'"
run_check "bizra_config.GOLD_PATH" "python -c 'from bizra_config import GOLD_PATH; assert GOLD_PATH.exists(), f\"{GOLD_PATH} missing\"'"

# ── Gate 2: Component tests ──
echo "" | tee -a "$LOG_FILE"
echo "Gate 2: Component Tests" | tee -a "$LOG_FILE"
run_check "evidence_audit_tests" "python -m pytest tests/core/proof_engine/test_evidence_audit.py -q --timeout=60"

if [ -f "tests/core/proof_engine/test_sat_validator.py" ]; then
    run_check "sat_validator_tests" "python -m pytest tests/core/proof_engine/test_sat_validator.py -q --timeout=300"
fi

# ── Gate 3: Regression ──
echo "" | tee -a "$LOG_FILE"
echo "Gate 3: Regression (proof_engine)" | tee -a "$LOG_FILE"
run_check "proof_engine_full" "python -m pytest tests/core/proof_engine/ -q --timeout=60"

# ── Gate 4: Spearpoint ancestry ──
echo "" | tee -a "$LOG_FILE"
echo "Gate 4: Spearpoint Ancestry" | tee -a "$LOG_FILE"
run_check "spearpoint_reachable" "git merge-base --is-ancestor b08f2208 HEAD"

# ── Gate 5: Config truth ──
echo "" | tee -a "$LOG_FILE"
echo "Gate 5: Config Truth" | tee -a "$LOG_FILE"
run_check "gold_path_exists" "python -c 'from bizra_config import GOLD_PATH; assert GOLD_PATH.exists()'"
run_check "ihsan_threshold" "python -c 'from core.integration.constants import UNIFIED_IHSAN_THRESHOLD; assert UNIFIED_IHSAN_THRESHOLD >= 0.95'"

# ── Summary ──
echo "" | tee -a "$LOG_FILE"
TOTAL=$((PASS + FAIL))
echo "=== RESULT: ${PASS}/${TOTAL} passed, ${FAIL} failed ===" | tee -a "$LOG_FILE"
echo "Log: ${LOG_FILE}" | tee -a "$LOG_FILE"

if [ "$FAIL" -gt 0 ]; then
    echo "PROMOTION BLOCKED" | tee -a "$LOG_FILE"
    exit 1
else
    echo "PROMOTION CLEAR" | tee -a "$LOG_FILE"
    exit 0
fi
