#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 MVSA Deployment Preflight
# ═══════════════════════════════════════════════════════════════════════════════
#
# Validates all MVSA artifacts exist and are schema-compliant before allowing
# deployment. MUST pass before docker-compose up or Kubernetes apply.
#
# Standing on Giants:
# - Deming (1950): PDCA — preflight IS the "Plan" validation
# - Burns (2011): 12-factor app — config validation at deploy boundary
# - PMBOK 7th Ed: Quality gate before release transition
#
# Usage:
#   bash deploy/node0/mvsa-preflight.sh [--state-dir /path/to/sovereign_state]
#
# Exit Codes:
#   0 - All preflight checks pass (deploy allowed)
#   1 - One or more checks failed (deploy BLOCKED)
#   2 - Missing dependencies (jq not installed)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ──
STATE_DIR="${1:-sovereign_state}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Override with --state-dir flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-dir) STATE_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Resolve to absolute
if [[ "$STATE_DIR" != /* ]]; then
    STATE_DIR="$PROJECT_ROOT/$STATE_DIR"
fi

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASS++))
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((FAIL++))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    ((WARN++))
}

# ── Dependencies ──
if ! command -v jq &>/dev/null; then
    echo -e "${RED}ERROR: jq is required for preflight. Install with: apt install jq${NC}"
    exit 2
fi

echo "════════════════════════════════════════════════════"
echo "  BIZRA Node0 MVSA Deployment Preflight"
echo "  State dir: $STATE_DIR"
echo "════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Check 1: Required artifacts exist
# ═══════════════════════════════════════════════════════════════════════════════
echo "▸ Artifact presence"

REQUIRED_FILES=(
    "node0_genesis.json"
    "genesis_hash.txt"
    "node0_lifecycle.json"
    "node0_assets.json"
    "urp_pledge.json"
    "pat_awareness.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$STATE_DIR/$f" ]]; then
        check_pass "$f exists"
    else
        check_fail "$f MISSING"
    fi
done

# Optional but desired
OPTIONAL_FILES=(
    "node0_mvsa_proof.json"
    "node0_authority_migration.json"
    "pat_roster.txt"
    "sat_roster.txt"
)

for f in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "$STATE_DIR/$f" ]]; then
        check_pass "$f exists (optional)"
    else
        check_warn "$f missing (optional)"
    fi
done

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Check 2: Lifecycle v2 schema compliance
# ═══════════════════════════════════════════════════════════════════════════════
echo "▸ Lifecycle v2 schema"

LIFECYCLE="$STATE_DIR/node0_lifecycle.json"
if [[ -f "$LIFECYCLE" ]]; then
    SCHEMA_VER=$(jq -r '.schema_version // "unknown"' "$LIFECYCLE" 2>/dev/null)
    if [[ "$SCHEMA_VER" == "2.0.0" ]]; then
        check_pass "schema_version = 2.0.0"
    else
        check_fail "schema_version = $SCHEMA_VER (expected 2.0.0)"
    fi

    STATUS=$(jq -r '.status // "unknown"' "$LIFECYCLE" 2>/dev/null)
    case "$STATUS" in
        ready)   check_pass "status = ready" ;;
        degraded) check_warn "status = degraded (deploy allowed with caution)" ;;
        blocked) check_fail "status = blocked (deploy NOT allowed)" ;;
        *)       check_fail "status = $STATUS (unexpected)" ;;
    esac

    # Validate all 11 gates are present
    GATE_COUNT=$(jq '.gates | length' "$LIFECYCLE" 2>/dev/null || echo "0")
    if [[ "$GATE_COUNT" -ge 11 ]]; then
        check_pass "gates: $GATE_COUNT present (≥ 11)"
    else
        check_fail "gates: $GATE_COUNT present (expected ≥ 11)"
    fi

    # Required sections
    for section in "origin" "identity" "artifacts" "gates" "mvsa" "mission" "restart_recovery" "compat"; do
        HAS=$(jq "has(\"$section\")" "$LIFECYCLE" 2>/dev/null)
        if [[ "$HAS" == "true" ]]; then
            check_pass "section '$section' present"
        else
            check_fail "section '$section' MISSING"
        fi
    done
else
    check_fail "node0_lifecycle.json NOT FOUND"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Check 3: Genesis authority
# ═══════════════════════════════════════════════════════════════════════════════
echo "▸ Genesis authority"

GENESIS="$STATE_DIR/node0_genesis.json"
if [[ -f "$GENESIS" ]]; then
    HAS_IDENTITY=$(jq 'has("identity")' "$GENESIS" 2>/dev/null)
    HAS_PAT=$(jq 'has("pat_team")' "$GENESIS" 2>/dev/null)
    HAS_SAT=$(jq 'has("sat_team")' "$GENESIS" 2>/dev/null)
    HAS_HASH=$(jq 'has("genesis_hash")' "$GENESIS" 2>/dev/null)

    [[ "$HAS_IDENTITY" == "true" ]] && check_pass "identity present" || check_fail "identity MISSING"
    [[ "$HAS_PAT" == "true" ]]      && check_pass "pat_team present" || check_fail "pat_team MISSING"
    [[ "$HAS_SAT" == "true" ]]      && check_pass "sat_team present" || check_fail "sat_team MISSING"
    [[ "$HAS_HASH" == "true" ]]     && check_pass "genesis_hash present" || check_fail "genesis_hash MISSING"

    PAT_COUNT=$(jq '.pat_team.agents | length' "$GENESIS" 2>/dev/null || echo "0")
    SAT_COUNT=$(jq '.sat_team.agents | length' "$GENESIS" 2>/dev/null || echo "0")
    if [[ "$PAT_COUNT" -ge 7 ]]; then
        check_pass "PAT agents: $PAT_COUNT (≥ 7)"
    else
        check_warn "PAT agents: $PAT_COUNT (expected ≥ 7)"
    fi
    if [[ "$SAT_COUNT" -ge 5 ]]; then
        check_pass "SAT agents: $SAT_COUNT (≥ 5)"
    else
        check_warn "SAT agents: $SAT_COUNT (expected ≥ 5)"
    fi
else
    check_fail "node0_genesis.json NOT FOUND"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Check 4: MVSA proof (if present)
# ═══════════════════════════════════════════════════════════════════════════════
echo "▸ MVSA proof artifact"

PROOF="$STATE_DIR/node0_mvsa_proof.json"
if [[ -f "$PROOF" ]]; then
    PROOF_STATUS=$(jq -r '.status // "unknown"' "$PROOF" 2>/dev/null)
    BOOTSTRAP=$(jq -r '.network.bootstrap_ok // false' "$PROOF" 2>/dev/null)
    SELF_VAL=$(jq -r '.consensus.self_validation_ok // false' "$PROOF" 2>/dev/null)

    [[ "$PROOF_STATUS" == "ready" ]] && check_pass "proof status = ready" || check_fail "proof status = $PROOF_STATUS"
    [[ "$BOOTSTRAP" == "true" ]]     && check_pass "bootstrap_ok = true" || check_fail "bootstrap_ok = $BOOTSTRAP"
    [[ "$SELF_VAL" == "true" ]]      && check_pass "self_validation_ok = true" || check_fail "self_validation_ok = $SELF_VAL"
else
    check_warn "node0_mvsa_proof.json not found (Rust binary may not have run)"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════"
TOTAL=$((PASS + FAIL))
echo -e "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$WARN warnings${NC} / $TOTAL checks"

if [[ "$FAIL" -eq 0 ]]; then
    echo -e "  ${GREEN}✓ MVSA PREFLIGHT PASSED — deploy allowed${NC}"
    echo "════════════════════════════════════════════════════"
    exit 0
else
    echo -e "  ${RED}✗ MVSA PREFLIGHT FAILED — deploy BLOCKED${NC}"
    echo "════════════════════════════════════════════════════"
    exit 1
fi
