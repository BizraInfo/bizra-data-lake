#!/usr/bin/env bash
# BIZRA Node0 Health Check Script
# Verifies all services are operational
#
# Usage: ./scripts/healthcheck.sh [--verbose]

set -euo pipefail

VERBOSE=false
[ "${1:-}" = "--verbose" ] && VERBOSE=true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check() {
    local name=$1
    local url=$2
    local expected=${3:-200}

    if response=$(curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
        if [ "$response" = "$expected" ]; then
            echo -e "${GREEN}✓${NC} $name"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} $name (HTTP $response)"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} $name (unreachable)"
        return 1
    fi
}

check_file() {
    local name=$1
    local path=$2

    if [ -f "$path" ]; then
        echo -e "${GREEN}✓${NC} $name"
        [ "$VERBOSE" = true ] && echo "   $(head -1 "$path")"
        return 0
    else
        echo -e "${RED}✗${NC} $name (not found)"
        return 1
    fi
}

echo "═══════════════════════════════════════════════════════════════"
echo "  BIZRA Node0 Health Check"
echo "═══════════════════════════════════════════════════════════════"
echo ""

FAILED=0

# Service checks
echo "Services:"
check "API Server" "http://localhost:3001/api/v1/health" || ((FAILED++))
check "Resource Pool" "http://localhost:8946/health" || ((FAILED++))
check "Ollama LLM" "http://localhost:11434/api/tags" || ((FAILED++))

echo ""
echo "Sovereign State:"
SOVEREIGN_DIR="${BIZRA_SOVEREIGN_STATE_DIR:-./sovereign_state}"
check_file "Genesis Hash" "$SOVEREIGN_DIR/genesis_hash.txt" || ((FAILED++))
check_file "PAT Roster" "$SOVEREIGN_DIR/pat_roster.txt" || ((FAILED++))
check_file "SAT Roster" "$SOVEREIGN_DIR/sat_roster.txt" || ((FAILED++))
check_file "Node0 Genesis" "$SOVEREIGN_DIR/node0_genesis.json" || ((FAILED++))

echo ""
echo "Docker Containers:"
if docker compose ps --format "{{.Name}}: {{.Status}}" 2>/dev/null | head -5; then
    :
else
    echo -e "${YELLOW}⚠${NC} Docker Compose not available"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED check(s) failed${NC}"
    exit 1
fi
