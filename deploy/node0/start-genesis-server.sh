#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA NODE0 Genesis Server — Start Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./deploy/node0/start-genesis-server.sh              # Local (bare-metal)
#   ./deploy/node0/start-genesis-server.sh --docker      # Docker Compose
#   ./deploy/node0/start-genesis-server.sh --status      # Health check only
#
# Standing on Giants: Boyd (OODA preflight) · Deming (verify before deploy)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
PORT="${NODE0_GENESIS_PORT:-7770}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[node0-genesis]${NC} $*"; }
ok()  { echo -e "  ${GREEN}OK${NC}: $*"; }
fail(){ echo -e "  ${RED}FAIL${NC}: $*"; }
warn(){ echo -e "  ${YELLOW}WARN${NC}: $*"; }

# ── Preflight Checks ──

preflight() {
    log "Preflight checks..."
    local errors=0

    # 1. Repo root
    if [ ! -f "$ROOT/bizra-constitution/node0_server.py" ]; then
        fail "node0_server.py not found at $ROOT/bizra-constitution/"
        ((errors++))
    else
        ok "node0_server.py found"
    fi

    # 2. Python + venv
    if [ -d "$ROOT/.venv-linux" ]; then
        ok "venv: .venv-linux"
    else
        fail "venv not found at $ROOT/.venv-linux"
        ((errors++))
    fi

    # 3. Ollama reachability
    if curl -s -m 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        local model_count
        model_count=$(curl -s "$OLLAMA_URL/api/tags" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "0")
        ok "Ollama: $model_count models at $OLLAMA_URL"
    else
        warn "Ollama not reachable at $OLLAMA_URL (inference will use template fallback)"
    fi

    # 4. Port available
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        fail "Port $PORT already in use"
        ((errors++))
    else
        ok "Port $PORT available"
    fi

    # 5. FastAPI importable
    if "$ROOT/.venv-linux/bin/python3" -c "import fastapi, uvicorn, pydantic" 2>/dev/null; then
        ok "Dependencies: fastapi, uvicorn, pydantic"
    else
        fail "Missing Python dependencies — run: pip install fastapi uvicorn pydantic"
        ((errors++))
    fi

    # 6. Bridge check
    local components
    components=$("$ROOT/.venv-linux/bin/python3" -c "
import sys; sys.path.insert(0, '$ROOT')
from core.bridges.constitutional_engine import availability_report
r = availability_report()
total = sum(1 for v in r['components'].values() if v)
print(f'{total}/{len(r[\"components\"])}')
" 2>/dev/null || echo "error")
    if [ "$components" = "11/11" ]; then
        ok "Bridge: $components components"
    else
        warn "Bridge: $components components (expected 11/11)"
    fi

    if [ $errors -gt 0 ]; then
        echo ""
        fail "$errors preflight check(s) failed. Aborting."
        exit 1
    fi

    log "Preflight: ALL PASS"
    echo ""
}

# ── Health Check ──

health_check() {
    local url="http://localhost:${PORT}/health"
    log "Health check: $url"
    if curl -sf "$url" 2>/dev/null | python3 -m json.tool; then
        ok "Server healthy"
    else
        fail "Server not responding on port $PORT"
        exit 1
    fi
}

# ── Start Modes ──

start_local() {
    preflight

    log "Starting NODE0 Genesis Server (local mode)"
    log "Port: $PORT | Ollama: $OLLAMA_URL"
    echo ""

    mkdir -p "$ROOT/logs/node0-genesis"

    cd "$ROOT/bizra-constitution"
    exec "$ROOT/.venv-linux/bin/python3" -m uvicorn \
        node0_server:create_app \
        --factory \
        --host 127.0.0.1 \
        --port "$PORT" \
        --workers 1 \
        --log-level info \
        --timeout-keep-alive 30
}

start_docker() {
    preflight

    log "Starting NODE0 Genesis Server (Docker mode)"

    # Ensure network exists
    docker network create bizra-network 2>/dev/null || true

    # Ensure shared evidence dir exists
    mkdir -p "$ROOT/04_GOLD/node0_evidence"

    cd "$ROOT"
    docker compose -f deploy/node0-genesis-compose.yaml up -d --build

    log "Waiting for health check..."
    sleep 5

    for i in $(seq 1 12); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            ok "Container healthy after ${i}x5s"
            health_check
            return 0
        fi
        sleep 5
    done

    fail "Container failed to become healthy within 60s"
    docker compose -f deploy/node0-genesis-compose.yaml logs --tail 30
    exit 1
}

# ── Main ──

case "${1:-}" in
    --docker)
        start_docker
        ;;
    --status)
        health_check
        ;;
    --preflight)
        preflight
        ;;
    ""|--local)
        start_local
        ;;
    *)
        echo "Usage: $0 [--local|--docker|--status|--preflight]"
        exit 1
        ;;
esac
