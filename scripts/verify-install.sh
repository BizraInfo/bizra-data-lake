#!/bin/bash
# ═══════════════════════════════════════════════════════════
# BIZRA Post-Install Verification — Node1 Smoke Test
# ═══════════════════════════════════════════════════════════
#
# Run after install.sh to confirm your node is operational.
#
# Usage:
#   ./scripts/verify-install.sh
#
# Exit codes:
#   0 — All checks passed. Node is operational.
#   1 — One or more checks failed. See output for details.
# ═══════════════════════════════════════════════════════════

set -euo pipefail

G="\033[38;2;125;211;192m"
D="\033[38;2;201;169;98m"
R="\033[0m"
B="\033[1m"
RD="\033[38;2;239;68;68m"

PASS=0
FAIL=0
WARN=0

check_pass() { echo -e "  ${G}PASS${R}  $1"; PASS=$((PASS+1)); }
check_fail() { echo -e "  ${RD}FAIL${R}  $1"; FAIL=$((FAIL+1)); }
check_warn() { echo -e "  ${D}WARN${R}  $1"; WARN=$((WARN+1)); }

echo ""
echo -e "  ${B}BIZRA Node Verification${R}"
echo -e "  ──────────────────────"
echo ""

# ── 1. Repo structure ────────────────────────────────────
echo -e "  ${D}[1/7]${R} Repository structure"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[ -f "pyproject.toml" ]           && check_pass "pyproject.toml exists" || check_fail "pyproject.toml missing — are you in the repo root?"
[ -d "bizra-omega/bizra-core" ]   && check_pass "Rust workspace present" || check_fail "bizra-omega/bizra-core missing"
[ -d "core" ]                     && check_pass "Python core/ present" || check_fail "core/ directory missing"
[ -f ".env.example" ]             && check_pass ".env.example present" || check_warn ".env.example missing"

# ── 2. System dependencies ───────────────────────────────
echo ""
echo -e "  ${D}[2/7]${R} System dependencies"

command -v python3 &>/dev/null    && check_pass "Python3: $(python3 --version 2>&1)" || check_fail "python3 not found"
command -v cargo &>/dev/null      && check_pass "Cargo: $(cargo --version 2>&1)" || check_fail "cargo not found — install Rust"
command -v git &>/dev/null        && check_pass "Git: $(git --version 2>&1)" || check_fail "git not found"

if dpkg -s libz3-dev &>/dev/null 2>&1; then
    check_pass "libz3-dev installed"
elif [ "$(uname -s)" = "Linux" ]; then
    check_fail "libz3-dev missing — run: sudo apt install libz3-dev"
fi

# ── 3. Rust binaries ─────────────────────────────────────
echo ""
echo -e "  ${D}[3/7]${R} Rust binaries"

BINARY="bizra-omega/target/release/bizra-node"
if [ -f "$BINARY" ]; then
    SIZE=$(ls -lh "$BINARY" | awk '{print $5}')
    check_pass "bizra-node binary ($SIZE)"
else
    check_fail "bizra-node not built — run: cd bizra-omega && cargo build --release -p bizra-node"
fi

API_BINARY="bizra-omega/target/release/bizra-api"
if [ -f "$API_BINARY" ]; then
    SIZE=$(ls -lh "$API_BINARY" | awk '{print $5}')
    check_pass "bizra-api binary ($SIZE)"
else
    check_warn "bizra-api not built (optional)"
fi

# ── 4. Python environment ────────────────────────────────
echo ""
echo -e "  ${D}[4/7]${R} Python environment"

# Try common venv locations
VENV=""
for candidate in .venv .venv-linux; do
    if [ -f "$candidate/bin/activate" ]; then
        VENV="$candidate"
        break
    fi
done

if [ -n "$VENV" ]; then
    check_pass "Virtual env found: $VENV"
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"

    python -c "import core" 2>/dev/null        && check_pass "import core — OK" || check_fail "import core failed — run: pip install -e ."
    python -c "import pytest" 2>/dev/null       && check_pass "pytest available" || check_warn "pytest missing — run: pip install -e '.[dev]'"
else
    check_fail "No Python venv found (.venv or .venv-linux)"
fi

# ── 5. Ollama / LLM backend ─────────────────────────────
echo ""
echo -e "  ${D}[5/7]${R} LLM backend"

if curl -s --max-time 5 http://localhost:11434/api/tags &>/dev/null; then
    MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "0")
    check_pass "Ollama responding ($MODEL_COUNT models)"
    if [ "$MODEL_COUNT" -eq 0 ]; then
        check_warn "No models loaded — run: ollama pull qwen2.5:3b"
    fi
else
    check_warn "Ollama not responding on localhost:11434 (needed for missions)"
fi

# ── 6. Rust tests (fast subset) ─────────────────────────
echo ""
echo -e "  ${D}[6/7]${R} Rust tests (quick check)"

if command -v cargo &>/dev/null && [ -f "bizra-omega/Cargo.toml" ]; then
    cd bizra-omega
    TEST_OUT=$(cargo test --release -p bizra-core -p bizra-hooks -p bizra-agent 2>&1 | grep "test result:" || true)
    cd ..
    RUST_PASS=$(echo "$TEST_OUT" | grep -oP '\d+ passed' | awk '{s+=$1} END {print s+0}')
    RUST_FAIL=$(echo "$TEST_OUT" | grep -oP '\d+ failed' | awk '{s+=$1} END {print s+0}')
    if [ "$RUST_FAIL" -eq 0 ] && [ "$RUST_PASS" -gt 0 ]; then
        check_pass "Rust core tests: $RUST_PASS passed, 0 failed"
    else
        check_fail "Rust core tests: $RUST_PASS passed, $RUST_FAIL failed"
    fi
else
    check_warn "Skipping Rust tests (cargo or workspace not found)"
fi

# ── 7. Cross-language sync ───────────────────────────────
echo ""
echo -e "  ${D}[7/7]${R} Constitutional sync"

if [ -n "$VENV" ]; then
    SYNC_OK=$(python3 -c "
from core.integration.constants import CANONICAL_THRESHOLDS
ok = CANONICAL_THRESHOLDS.get('IHSAN_THRESHOLD') == 0.95
print('yes' if ok else 'no')
" 2>/dev/null || echo "error")

    if [ "$SYNC_OK" = "yes" ]; then
        check_pass "IHSAN_THRESHOLD = 0.95 (constitutional floor)"
    elif [ "$SYNC_OK" = "no" ]; then
        check_fail "IHSAN_THRESHOLD != 0.95 — constitutional constants out of sync"
    else
        check_warn "Could not verify constants (import error)"
    fi
else
    check_warn "Skipping sync check (no venv)"
fi

# ── Summary ──────────────────────────────────────────────
echo ""
echo -e "  ──────────────────────"
TOTAL=$((PASS+FAIL+WARN))
echo -e "  ${G}$PASS passed${R}  ${RD}$FAIL failed${R}  ${D}$WARN warnings${R}  ($TOTAL checks)"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${G}${B}NODE OPERATIONAL${R} — ready to run: ./scripts/bizra"
    echo ""
    exit 0
else
    echo -e "  ${RD}${B}NODE NOT READY${R} — fix the failures above, then re-run this script."
    echo ""
    exit 1
fi
