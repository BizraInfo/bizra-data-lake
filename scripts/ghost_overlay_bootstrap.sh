#!/usr/bin/env bash
# ============================================================================
# BIZRA Ghost Overlay — Development Bootstrap
# ============================================================================
#
# Sets up the Ghost Overlay development environment on a BIZRA Node0 machine.
# Run from the BIZRA-DATA-LAKE root directory.
#
# Standing on Giants: Boyd (OODA) · Shannon (SNR) · Lamport (single truth)
#
# Usage:
#   ./scripts/ghost_overlay_bootstrap.sh [--check-only]
#
# Prerequisites:
#   - BIZRA-DATA-LAKE repo cloned
#   - Python 3.11+ with venv
#   - Rust toolchain (for bizra-action)
#   - AutoHotkey v2 on Windows host (for overlay rendering)
#
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
GOLD='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK_ONLY="${1:-}"

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; ERRORS=$((ERRORS + 1)); }
info() { echo -e "  ${CYAN}→${NC} $1"; }
header() { echo -e "\n${GOLD}[$1]${NC}"; }

ERRORS=0

# ============================================================================
# 1. PREFLIGHT CHECKS
# ============================================================================
header "1/6 PREFLIGHT"

# Repo root
if [ -f "$REPO_ROOT/pyproject.toml" ]; then
    pass "BIZRA-DATA-LAKE root: $REPO_ROOT"
else
    fail "Not in BIZRA-DATA-LAKE root (pyproject.toml not found)"
fi

# Python
if command -v python &>/dev/null; then
    PY_VER=$(python --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
        pass "Python $PY_VER"
    else
        fail "Python $PY_VER (need 3.11+)"
    fi
else
    fail "Python not found"
fi

# Venv
if [ -d "$REPO_ROOT/.venv-linux" ]; then
    pass "Venv: .venv-linux exists"
else
    fail "Venv: .venv-linux not found"
fi

# Rust
if command -v cargo &>/dev/null; then
    RUST_VER=$(rustc --version 2>&1 | awk '{print $2}')
    pass "Rust $RUST_VER"
else
    fail "Rust toolchain not found"
fi

# ============================================================================
# 2. PYTHON DEPENDENCIES
# ============================================================================
header "2/6 PYTHON DEPENDENCIES"

# Check critical packages
for pkg in fastapi uvicorn httpx websockets; do
    if python -c "import $pkg" 2>/dev/null; then
        pass "$pkg installed"
    else
        fail "$pkg NOT installed"
        if [ "$CHECK_ONLY" != "--check-only" ]; then
            info "Installing $pkg..."
            pip install "$pkg" -q
        fi
    fi
done

# Desktop bridge dependencies
if python -c "from core.bridges.desktop_bridge import DesktopBridge" 2>/dev/null; then
    pass "Desktop Bridge importable"
else
    fail "Desktop Bridge import failed"
fi

# ============================================================================
# 3. RUST ACTION BUS
# ============================================================================
header "3/6 RUST ACTION BUS (bizra-action)"

ACTION_CRATE="$REPO_ROOT/bizra-omega/bizra-action"
if [ -f "$ACTION_CRATE/Cargo.toml" ]; then
    pass "bizra-action crate found"
else
    fail "bizra-action crate not found at $ACTION_CRATE"
fi

if [ "$CHECK_ONLY" != "--check-only" ]; then
    info "Building bizra-action (release)..."
    if (cd "$REPO_ROOT/bizra-omega" && cargo build -p bizra-action --release 2>/dev/null); then
        pass "bizra-action built successfully"
    else
        fail "bizra-action build failed"
    fi
else
    info "Skipping build (--check-only mode)"
fi

# Run tests
if [ "$CHECK_ONLY" != "--check-only" ]; then
    info "Running bizra-action tests..."
    if (cd "$REPO_ROOT/bizra-omega" && cargo test -p bizra-action --release 2>/dev/null); then
        pass "bizra-action tests pass"
    else
        fail "bizra-action tests failed"
    fi
fi

# ============================================================================
# 4. PORTS & NETWORKING
# ============================================================================
header "4/6 PORTS & NETWORKING"

check_port() {
    local port=$1 name=$2
    if ! ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        pass "Port $port ($name) available"
    else
        OWNER=$(ss -tlnp 2>/dev/null | grep ":${port} " | awk '{print $NF}' | head -1)
        fail "Port $port ($name) in use by $OWNER"
    fi
}

check_port 9742 "Desktop Bridge JSON-RPC"
check_port 9743 "Ghost WS Bridge"

# ============================================================================
# 5. CONSTITUTIONAL THRESHOLDS
# ============================================================================
header "5/6 CONSTITUTIONAL THRESHOLDS"

# Verify thresholds are importable from single source of truth
THRESHOLD_CHECK=$(python -c "
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD,
    ADL_GINI_THRESHOLD, ADL_HARBERGER_TAX_RATE
)
print(f'ihsan={UNIFIED_IHSAN_THRESHOLD}')
print(f'snr={UNIFIED_SNR_THRESHOLD}')
print(f'gini={ADL_GINI_THRESHOLD}')
print(f'harberger={ADL_HARBERGER_TAX_RATE}')
" 2>/dev/null || echo "IMPORT_FAILED")

if echo "$THRESHOLD_CHECK" | grep -q "ihsan=0.95"; then
    pass "Ihsan threshold: 0.95 (from constants.py)"
else
    fail "Cannot import Ihsan threshold from constants.py"
fi

if echo "$THRESHOLD_CHECK" | grep -q "snr=0.85"; then
    pass "SNR threshold: 0.85 (from constants.py)"
else
    fail "Cannot import SNR threshold from constants.py"
fi

# ============================================================================
# 6. WINDOWS-SIDE CHECK (AHK)
# ============================================================================
header "6/6 WINDOWS-SIDE (AHK)"

# Check if running in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    pass "WSL detected — Windows host available"

    # Check for AHK on Windows side
    if /mnt/c/Windows/System32/cmd.exe /c "where AutoHotkey64.exe" 2>/dev/null | grep -qi autohotkey; then
        pass "AutoHotkey v2 found on Windows"
    else
        info "AutoHotkey v2 not detected on Windows PATH"
        info "Install from: https://www.autohotkey.com/v2/"
        info "(Ghost Overlay AHK script requires AHK v2)"
    fi
else
    info "Not running in WSL — skip Windows-side checks"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo -e "${GOLD}════════════════════════════════════════${NC}"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}  Ghost Overlay Bootstrap: ALL CHECKS PASS${NC}"
    echo -e "${GOLD}════════════════════════════════════════${NC}"
    echo ""
    echo "  Next steps:"
    echo "    1. Start Desktop Bridge:  python -c 'from core.bridges.desktop_bridge import DesktopBridge; import asyncio; asyncio.run(DesktopBridge().start())'"
    echo "    2. Start Ghost WS Bridge: uvicorn core.bridges.ghost_ws:app --host 0.0.0.0 --port 9743"
    echo "    3. Launch AHK Overlay:    Start-Process scripts/ghost_overlay.ahk (on Windows)"
    echo ""
else
    echo -e "${RED}  Ghost Overlay Bootstrap: $ERRORS ERRORS${NC}"
    echo -e "${GOLD}════════════════════════════════════════${NC}"
    echo ""
    echo "  Fix the errors above, then re-run this script."
    echo ""
fi

exit "$ERRORS"
