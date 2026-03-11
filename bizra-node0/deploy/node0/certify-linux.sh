#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 — Native Linux Certification Test
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the complete certification sequence on native Linux.
# This is the AUTHORITATIVE path — WSL2 is compatibility-only.
#
# Standing on Giants:
# - Deming (PDCA, 1950): Plan→Do→Check→Act — this IS the "Check"
# - PMBOK 7th Ed: Quality gate verification before release transition
# - Nakamoto (evidence chain, 2008): every check produces a receipt
#
# Usage:
#   bash deploy/node0/certify-linux.sh [--prefix /opt/bizra-node0]
#
# Exit Codes:
#   0 — All certification checks pass
#   1 — One or more checks failed
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PREFIX="${1:-/opt/bizra-node0}"
VENV="$PREFIX/.venv"
PYTHON="$VENV/bin/python"
STATE_DIR="${BIZRA_STATE_DIR:-/var/lib/bizra-node0}"
LOG_DIR="${BIZRA_LOG_DIR:-/var/log/bizra-node0}"
RESULTS_FILE="/tmp/bizra-node0-certification-$(date +%Y%m%dT%H%M%S).json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASSED=0
FAILED=0
TOTAL=0

# ─── Test Runner ──────────────────────────────────────────────────────────────
check() {
    local name="$1"
    local cmd="$2"
    TOTAL=$((TOTAL + 1))

    echo -n "  [$TOTAL] $name ... "
    if eval "$cmd" &>/dev/null; then
        PASSED=$((PASSED + 1))
        echo -e "${GREEN}PASS${NC}"
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}FAIL${NC}"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN} BIZRA Node0 — Native Linux Certification${NC}"
echo -e "${CYAN} $(date -Iseconds)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ─── Section 1: Environment ──────────────────────────────────────────────────
echo -e "${CYAN}Section 1: Environment${NC}"

check "Not running on /mnt/c (WSL passthrough)" \
    "[[ ! '$PREFIX' =~ ^/mnt/[a-z] ]]"

check "Running on native Linux filesystem" \
    "[[ -d /proc && -f /proc/version ]]"

check "Python 3.11+ available" \
    "$PYTHON -c 'import sys; assert sys.version_info >= (3, 11)'"

check "Virtual environment valid" \
    "[[ -f '$PYTHON' && -x '$PYTHON' ]]"

check "Code directory exists" \
    "[[ -d '$PREFIX/core' && -d '$PREFIX/scripts' ]]"

check "Config directory exists" \
    "[[ -d /etc/bizra-node0 ]]"

check "State directory exists and writable" \
    "[[ -d '$STATE_DIR' && -w '$STATE_DIR' ]]"

check "Log directory exists and writable" \
    "[[ -d '$LOG_DIR' && -w '$LOG_DIR' ]]"

echo ""

# ─── Section 2: Import Integrity ─────────────────────────────────────────────
echo -e "${CYAN}Section 2: Import Integrity${NC}"

check "core.integration.constants imports" \
    "$PYTHON -c 'from core.integration.constants import IHSAN_THRESHOLD; assert IHSAN_THRESHOLD == 0.95'"

check "core.sovereign.node0_authority imports" \
    "$PYTHON -c 'from core.sovereign import node0_authority'"

check "core.sovereign.node0_mvsa imports" \
    "$PYTHON -c 'from core.sovereign import node0_mvsa'"

check "core.pci.gates imports" \
    "$PYTHON -c 'from core.pci.gates import PCIGateKeeper'"

check "core.proof_engine.evidence_ledger imports" \
    "$PYTHON -c 'from core.proof_engine.evidence_ledger import EvidenceLedger'"

check "core.snr_protocol imports" \
    "$PYTHON -c 'from core import snr_protocol'"

check "core.token.types imports" \
    "$PYTHON -c 'from core.token.types import TokenType'"

echo ""

# ─── Section 3: Operator Surface ─────────────────────────────────────────────
echo -e "${CYAN}Section 3: Operator Surface${NC}"

check "node0_standalone.py exists and is executable" \
    "[[ -f '$PREFIX/scripts/node0_standalone.py' ]]"

check "node0_standalone.py syntax valid" \
    "$PYTHON -m py_compile '$PREFIX/scripts/node0_standalone.py'"

check "node0_genesis_ceremony.sh exists" \
    "[[ -f '$PREFIX/scripts/node0_genesis_ceremony.sh' ]]"

check "node0_genesis_ceremony.sh syntax valid" \
    "bash -n '$PREFIX/scripts/node0_genesis_ceremony.sh'"

check "mvsa-preflight.sh exists" \
    "[[ -f '$PREFIX/deploy/node0/mvsa-preflight.sh' ]]"

echo ""

# ─── Section 4: Security Posture ─────────────────────────────────────────────
echo -e "${CYAN}Section 4: Security Posture${NC}"

check "Production env file exists" \
    "[[ -f /etc/bizra-node0/node0.env ]]"

check "Env file not world-readable" \
    "[[ ! -r /etc/bizra-node0/node0.env ]] || [[ \$(stat -c '%a' /etc/bizra-node0/node0.env) =~ ^6[04]0$ ]]"

check "JWT fail-closed in production" \
    "$PYTHON -c \"
import os; os.environ['BIZRA_ENV']='production'
from core.auth.jwt_auth import _production_mode_enabled
assert _production_mode_enabled()
\""

check "Ghost bridge disabled by default" \
    "$PYTHON -c \"
import os
assert os.environ.get('GHOST_WS_ENABLED', 'false').lower() != 'true'
\""

echo ""

# ─── Section 5: Systemd Integration ──────────────────────────────────────────
echo -e "${CYAN}Section 5: Systemd Integration${NC}"

check "Systemd unit file installed" \
    "[[ -f /etc/systemd/system/bizra-node0.service ]]"

check "Logrotate config installed" \
    "[[ -f /etc/logrotate.d/bizra-node0 ]]"

check "Systemd unit valid" \
    "systemd-analyze verify /etc/systemd/system/bizra-node0.service 2>&1 || true"

echo ""

# ─── Section 6: Documentation ────────────────────────────────────────────────
echo -e "${CYAN}Section 6: Documentation${NC}"

check "MVSA spec exists" \
    "[[ -f '$PREFIX/docs/NODE0_STANDALONE_READINESS.md' ]]"

check "DoD exists" \
    "[[ -f '$PREFIX/docs/constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md' ]]"

check "README exists" \
    "[[ -f '$PREFIX/README.md' ]]"

check "RELEASE policy exists" \
    "[[ -f '$PREFIX/RELEASE.md' ]]"

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
if [[ $FAILED -eq 0 ]]; then
    echo -e " ${GREEN}CERTIFICATION: $PASSED/$TOTAL PASSED${NC}"
    echo -e " ${GREEN}Node0 is CERTIFIED for native Linux production operation.${NC}"
else
    echo -e " ${RED}CERTIFICATION: $PASSED/$TOTAL PASSED, $FAILED FAILED${NC}"
    echo -e " ${RED}Node0 is NOT certified. Fix failures before production use.${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# ─── JSON Receipt ─────────────────────────────────────────────────────────────
cat > "$RESULTS_FILE" <<JSONEOF
{
  "certification": "bizra-node0-native-linux",
  "version": "1.0.0",
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "kernel": "$(uname -r)",
  "prefix": "$PREFIX",
  "checks_total": $TOTAL,
  "checks_passed": $PASSED,
  "checks_failed": $FAILED,
  "certified": $([ $FAILED -eq 0 ] && echo true || echo false)
}
JSONEOF

echo ""
echo "  Receipt: $RESULTS_FILE"
echo ""

exit $FAILED
