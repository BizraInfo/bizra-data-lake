#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 — One-Node Live Launcher
# ═══════════════════════════════════════════════════════════════════════════════
#
# Launches the full sovereign stack: LLM inference, evidence chain, mission
# pipeline — all on one machine, zero cloud dependency.
#
# Standing on Giants:
#   Shannon (SNR) · Boyd (OODA) · Nakamoto (evidence chain) · Lamport (consensus)
#
# Usage:
#   ./scripts/node0_live.sh                  # Start on default port 8888
#   ./scripts/node0_live.sh --port 9000      # Custom port
#   ./scripts/node0_live.sh --preflight-only # Check deps, don't start
#
# Requirements:
#   - Ollama running (localhost:11434) with at least one model
#   - Python 3.11+ with .venv-linux activated
#   - BIZRA-DATA-LAKE as working directory
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${1:-8888}"
PREFLIGHT_ONLY=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --port)     shift; PORT="$1"; shift ;;
        --port=*)   PORT="${arg#*=}" ;;
        --preflight-only) PREFLIGHT_ONLY=true ;;
        [0-9]*)     PORT="$arg" ;;
    esac
done

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BIZRA Node0 — One-Node Live Launcher${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ─── Preflight ────────────────────────────────────────────────────────────────
echo -e "${CYAN}Preflight checks:${NC}"
ERRORS=0

# 1. Project root
if [ -f "$PROJECT_ROOT/pyproject.toml" ] && [ -d "$PROJECT_ROOT/core" ]; then
    ok "Project root: $PROJECT_ROOT"
else
    fail "Not in BIZRA-DATA-LAKE root"; ERRORS=$((ERRORS + 1))
fi

# 2. Python + venv
if [ -d "$PROJECT_ROOT/.venv-linux" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv-linux/bin/activate" 2>/dev/null || true
    PYTHON_VER=$(python3 --version 2>&1)
    ok "Python: $PYTHON_VER (.venv-linux active)"
else
    fail "Missing .venv-linux — run: python3 -m venv .venv-linux"; ERRORS=$((ERRORS + 1))
fi

# 3. Ollama
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    MODEL_COUNT=$(curl -sf http://localhost:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "0")
    ok "Ollama: ${MODEL_COUNT} models available"
else
    fail "Ollama not responding on localhost:11434"; ERRORS=$((ERRORS + 1))
fi

# 4. LM Studio (optional — Tier 1 primary, Ollama is sufficient fallback)
WSL_GATEWAY=$(ip route show default 2>/dev/null | awk '{print $3}' || echo "172.22.48.1")
if curl -sf "http://${WSL_GATEWAY}:1234/api/v1/models" > /dev/null 2>&1; then
    LM_MODELS=$(curl -sf "http://${WSL_GATEWAY}:1234/api/v1/models" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
    ok "LM Studio: ${LM_MODELS} models on ${WSL_GATEWAY}:1234 (Tier 1 primary)"
else
    warn "LM Studio not detected — Ollama will be primary backend"
fi

# 5. Key Python deps
python3 -c "import httpx, fastapi, uvicorn" 2>/dev/null && ok "Python deps: httpx, fastapi, uvicorn" \
    || { fail "Missing Python deps — run: pip install httpx fastapi uvicorn"; ERRORS=$((ERRORS + 1)); }

# 6. Core imports
python3 -c "from core.sovereign.mission import MissionOrchestrator; from core.inference.gateway import InferenceGateway" 2>/dev/null \
    && ok "Core imports: MissionOrchestrator, InferenceGateway" \
    || { fail "Core import failure — check core/ package"; ERRORS=$((ERRORS + 1)); }

echo ""
if [ "$ERRORS" -gt 0 ]; then
    fail "Preflight failed with $ERRORS error(s). Fix above issues and retry."
    exit 1
fi

if [ "$PREFLIGHT_ONLY" = true ]; then
    ok "All preflight checks passed. Ready to launch."
    exit 0
fi

# ─── Environment ──────────────────────────────────────────────────────────────
echo -e "${CYAN}Setting environment:${NC}"

# Enable LLM inference (the ignition key)
export BIZRA_ENABLE_LLM=1
ok "BIZRA_ENABLE_LLM=1 (LLM inference enabled)"

# Ollama URL
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
ok "OLLAMA_HOST=$OLLAMA_HOST"

# LM Studio auto-detection
export LMSTUDIO_HOST="${LMSTUDIO_HOST:-$WSL_GATEWAY}"
ok "LMSTUDIO_HOST=$LMSTUDIO_HOST"

# API key (generate if not set — localhost-only is safe without)
if [ -z "${BIZRA_NODE0_API_KEY:-}" ]; then
    warn "BIZRA_NODE0_API_KEY not set (localhost-only access, no auth required)"
else
    ok "BIZRA_NODE0_API_KEY set (auth enabled)"
fi

# Note: BIZRA_USERSTORE_MASTER_SECRET is NOT needed for localhost.
# UserStore auto-generates and persists a local key file if absent
# (core/auth/user_store.py L300). Only set this for multi-node secret sync.

echo ""

# ─── Launch ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}Launching Node0 on port ${PORT}...${NC}"
echo -e "  API:     http://127.0.0.1:${PORT}"
echo -e "  Health:  http://127.0.0.1:${PORT}/health"
echo -e "  Task:    POST http://127.0.0.1:${PORT}/task"
echo ""
echo -e "${GREEN}  بسم الله الرحمن الرحيم${NC}"
echo -e "${GREEN}  كل بذرة تحمل في داخلها مخطط غابة بأكملها${NC}"
echo ""

cd "$PROJECT_ROOT"
exec python3 scripts/node0_standalone.py serve --host 127.0.0.1 --port "$PORT"
