#!/usr/bin/env bash
# ==============================================================================
# BIZRA Mission Bridge Startup Script
# ==============================================================================
#
# Starts the Desktop Bridge with all required env vars for mission execution.
# Handles: venv activation, secret loading, model warmup, PID management.
#
# Usage:
#   ./scripts/start_mission_bridge.sh           # Start bridge (foreground)
#   ./scripts/start_mission_bridge.sh --daemon   # Start bridge (background)
#   ./scripts/start_mission_bridge.sh --stop     # Graceful stop
#   ./scripts/start_mission_bridge.sh --status   # Check bridge status
#
# Standing on Giants: Boyd (OODA lifecycle) | Lamport (ordered startup)
# ==============================================================================

set -euo pipefail

BIZRA_ROOT="${BIZRA_DATA_LAKE_ROOT:-/mnt/c/BIZRA-DATA-LAKE}"
VENV="${BIZRA_ROOT}/.venv-linux"
PID_FILE="${BIZRA_ROOT}/sovereign_state/bridge.pid"
LOG_FILE="/tmp/bizra_bridge.log"
SECRETS_FILE="/etc/bizra/secrets.env"
LOCAL_SECRETS="${BIZRA_ROOT}/deploy/node0/.env.local"
BRIDGE_PORT=9742

# ── Color output ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[BRIDGE]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[BRIDGE]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[BRIDGE]${NC} $*"; }
log_err()   { echo -e "${RED}[BRIDGE]${NC} $*"; }

# ── Preflight checks ────────────────────────────────────────────────
preflight() {
    local errors=0

    # 1. Repo root
    if [[ ! -d "${BIZRA_ROOT}/core/bridges" ]]; then
        log_err "BIZRA root not found at ${BIZRA_ROOT}"
        ((errors++))
    fi

    # 2. Python venv
    if [[ ! -f "${VENV}/bin/python3" ]]; then
        log_err "Python venv not found at ${VENV}"
        ((errors++))
    fi

    # 3. Bridge module importable
    if ! "${VENV}/bin/python3" -c "import core.bridges.desktop_bridge" 2>/dev/null; then
        log_err "Bridge module import failed — check for SyntaxError"
        ((errors++))
    fi

    # 4. Port not already in use by another process
    if ss -tlnp 2>/dev/null | grep -q ":${BRIDGE_PORT} "; then
        local existing_pid
        existing_pid=$(ss -tlnp 2>/dev/null | grep ":${BRIDGE_PORT} " | grep -oP 'pid=\K\d+' | head -1)
        if [[ -n "${existing_pid}" ]]; then
            log_warn "Port ${BRIDGE_PORT} already in use by PID ${existing_pid}"
            log_warn "Run: $0 --stop  or  kill ${existing_pid}"
            ((errors++))
        fi
    fi

    if [[ ${errors} -gt 0 ]]; then
        log_err "Preflight failed with ${errors} error(s)"
        return 1
    fi

    log_ok "Preflight passed"
    return 0
}

# ── Load secrets ─────────────────────────────────────────────────────
load_secrets() {
    # Priority: env vars > local .env > system secrets
    if [[ -f "${LOCAL_SECRETS}" ]]; then
        log_info "Loading secrets from ${LOCAL_SECRETS}"
        set -a
        # shellcheck disable=SC1090
        source "${LOCAL_SECRETS}"
        set +a
    elif [[ -f "${SECRETS_FILE}" ]]; then
        log_info "Loading secrets from ${SECRETS_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRETS_FILE}"
        set +a
    fi

    # Generate bridge token if not set
    if [[ -z "${BIZRA_BRIDGE_TOKEN:-}" ]]; then
        BIZRA_BRIDGE_TOKEN=$("${VENV}/bin/python3" -c "
import hashlib, os
print(hashlib.blake2b(b'bizra-node0-' + os.urandom(16), digest_size=16).hexdigest())
")
        export BIZRA_BRIDGE_TOKEN
        log_warn "Generated ephemeral BIZRA_BRIDGE_TOKEN (not persistent)"
    fi

    # Generate signing keys if not set
    if [[ -z "${BIZRA_RECEIPT_PRIVATE_KEY_HEX:-}" ]]; then
        local keypair
        keypair=$("${VENV}/bin/python3" -c "
from core.pci.crypto import generate_keypair
priv, pub = generate_keypair()
print(f'{priv} {pub}')
" 2>/dev/null || echo "")
        if [[ -n "${keypair}" ]]; then
            BIZRA_RECEIPT_PRIVATE_KEY_HEX="${keypair%% *}"
            BIZRA_RECEIPT_PUBLIC_KEY_HEX="${keypair##* }"
            export BIZRA_RECEIPT_PRIVATE_KEY_HEX BIZRA_RECEIPT_PUBLIC_KEY_HEX
            log_warn "Generated ephemeral signing keys (not persistent)"
        fi
    fi

    # Required env vars
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${BIZRA_ROOT}"
    export BIZRA_DATA_LAKE_ROOT="${BIZRA_ROOT}"
    export BIZRA_ENABLE_LLM="${BIZRA_ENABLE_LLM:-1}"
    export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
}

# ── Model warmup ─────────────────────────────────────────────────────
warmup_model() {
    if [[ "${BIZRA_ENABLE_LLM:-}" != "1" ]]; then
        log_info "LLM disabled — skipping warmup"
        return 0
    fi

    local ollama_url="${OLLAMA_HOST:-http://127.0.0.1:11434}"
    log_info "Warming up Ollama model (phi3:mini)..."

    # Check Ollama reachability first
    if ! curl -sS -m 3 "${ollama_url}/api/tags" >/dev/null 2>&1; then
        log_warn "Ollama not reachable at ${ollama_url} — skipping warmup"
        return 0
    fi

    # Send a minimal generate request to load model weights into memory
    local warmup_start
    warmup_start=$(date +%s%N)

    local resp
    resp=$(curl -sS -m 120 "${ollama_url}/api/generate" \
        -d '{"model":"phi3:mini","prompt":"hello","stream":false,"options":{"num_predict":1}}' \
        2>/dev/null || echo '{"error":"timeout"}')

    local warmup_end
    warmup_end=$(date +%s%N)
    local warmup_ms=$(( (warmup_end - warmup_start) / 1000000 ))

    if echo "${resp}" | grep -q '"response"'; then
        log_ok "Model warm (${warmup_ms}ms)"
    else
        log_warn "Warmup response: ${resp:0:100}"
    fi
}

# ── Start bridge ─────────────────────────────────────────────────────
start_bridge() {
    local daemon="${1:-false}"

    preflight || exit 1
    load_secrets

    # Activate venv
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"

    # Warm model in background (don't block bridge start)
    warmup_model &
    local warmup_pid=$!

    # Ensure sovereign_state dir exists
    mkdir -p "${BIZRA_ROOT}/sovereign_state"
    mkdir -p "$(dirname "${LOG_FILE}")"

    if [[ "${daemon}" == "true" ]]; then
        log_info "Starting bridge in daemon mode..."
        nohup python -m core.bridges.desktop_bridge > "${LOG_FILE}" 2>&1 &
        local bridge_pid=$!
        echo "${bridge_pid}" > "${PID_FILE}"

        # Wait for port to come up
        local retries=0
        while [[ ${retries} -lt 10 ]]; do
            if ss -tlnp 2>/dev/null | grep -q ":${BRIDGE_PORT} "; then
                log_ok "Bridge started (PID ${bridge_pid}, port ${BRIDGE_PORT})"
                log_info "Log: ${LOG_FILE}"
                log_info "PID file: ${PID_FILE}"

                # Wait for warmup to complete
                wait "${warmup_pid}" 2>/dev/null || true
                return 0
            fi
            sleep 1
            ((retries++))
        done

        log_err "Bridge failed to start — check ${LOG_FILE}"
        tail -20 "${LOG_FILE}" 2>/dev/null
        rm -f "${PID_FILE}"
        return 1
    else
        log_info "Starting bridge in foreground (Ctrl+C to stop)..."
        # Wait for warmup before starting
        wait "${warmup_pid}" 2>/dev/null || true
        exec python -m core.bridges.desktop_bridge
    fi
}

# ── Stop bridge ──────────────────────────────────────────────────────
stop_bridge() {
    # Try PID file first
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid=$(cat "${PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            log_info "Sending SIGTERM to PID ${pid}..."
            kill -TERM "${pid}"

            local retries=0
            while kill -0 "${pid}" 2>/dev/null && [[ ${retries} -lt 15 ]]; do
                sleep 1
                ((retries++))
            done

            if kill -0 "${pid}" 2>/dev/null; then
                log_warn "Bridge didn't stop gracefully — sending SIGKILL"
                kill -9 "${pid}" 2>/dev/null || true
            fi

            rm -f "${PID_FILE}"
            log_ok "Bridge stopped"
            return 0
        else
            log_warn "PID ${pid} not running — cleaning up stale PID file"
            rm -f "${PID_FILE}"
        fi
    fi

    # Fallback: find by process name
    local pids
    pids=$(pgrep -f "python.*core\.bridges\.desktop_bridge" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
        log_info "Stopping bridge processes: ${pids}"
        echo "${pids}" | xargs kill -TERM 2>/dev/null || true
        sleep 2
        echo "${pids}" | xargs kill -9 2>/dev/null || true
        log_ok "Bridge stopped"
    else
        log_info "No bridge process found"
    fi
}

# ── Status ───────────────────────────────────────────────────────────
status_bridge() {
    echo ""
    echo "=== BIZRA Mission Bridge Status ==="
    echo ""

    # Bridge process
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        log_ok "Bridge: RUNNING (PID $(cat "${PID_FILE}"))"
    elif pgrep -f "python.*core\.bridges\.desktop_bridge" >/dev/null 2>&1; then
        local pid
        pid=$(pgrep -f "python.*core\.bridges\.desktop_bridge" | head -1)
        log_ok "Bridge: RUNNING (PID ${pid}, no PID file)"
    else
        log_warn "Bridge: STOPPED"
    fi

    # Port
    if ss -tlnp 2>/dev/null | grep -q ":${BRIDGE_PORT} "; then
        log_ok "Port ${BRIDGE_PORT}: LISTENING"
    else
        log_warn "Port ${BRIDGE_PORT}: NOT LISTENING"
    fi

    # Ollama
    if curl -sS -m 3 "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
        local model_count
        model_count=$(curl -sS -m 3 "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" 2>/dev/null | \
            python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "?")
        log_ok "Ollama: UP (${model_count} models available)"

        # Check if any model is warm
        local warm
        warm=$(curl -sS -m 3 "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/ps" 2>/dev/null | \
            python3 -c "import sys,json; m=json.load(sys.stdin).get('models',[]); print(m[0]['name'] if m else 'none')" 2>/dev/null || echo "?")
        if [[ "${warm}" != "none" && "${warm}" != "?" ]]; then
            log_ok "  Warm model: ${warm}"
        else
            log_warn "  No warm model (first mission will be slow)"
        fi
    else
        log_warn "Ollama: NOT REACHABLE"
    fi

    # LM Studio
    local lm_models
    lm_models=$(curl -sS -m 3 "http://172.22.48.1:1234/v1/models" 2>/dev/null | \
        python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
    if [[ "${lm_models}" -gt 0 ]]; then
        log_ok "LM Studio: UP (${lm_models} model(s) loaded — GPU PRIMARY)"
    elif [[ "${lm_models}" == "0" ]]; then
        log_warn "LM Studio: UP but 0 models loaded — falling back to Ollama CPU"
    else
        log_warn "LM Studio: NOT REACHABLE"
    fi

    # GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "?")
        if [[ "${gpu_info}" != "?" ]]; then
            log_info "GPU: ${gpu_info}"
        fi
    fi

    # Env vars
    echo ""
    echo "Environment:"
    [[ -n "${BIZRA_BRIDGE_TOKEN:-}" ]] && log_ok "  BIZRA_BRIDGE_TOKEN: set" || log_warn "  BIZRA_BRIDGE_TOKEN: NOT SET"
    [[ -n "${BIZRA_RECEIPT_PRIVATE_KEY_HEX:-}" ]] && log_ok "  SIGNING_KEYS: set" || log_warn "  SIGNING_KEYS: NOT SET"
    [[ -n "${BRAVE_API_KEY:-}" ]] && log_ok "  BRAVE_API_KEY: set" || log_warn "  BRAVE_API_KEY: NOT SET"
    [[ "${BIZRA_ENABLE_LLM:-}" == "1" ]] && log_ok "  BIZRA_ENABLE_LLM: enabled" || log_warn "  BIZRA_ENABLE_LLM: disabled"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────
case "${1:-}" in
    --daemon|-d)
        start_bridge true
        ;;
    --stop|-s)
        stop_bridge
        ;;
    --status|-S)
        status_bridge
        ;;
    --warmup|-w)
        load_secrets
        warmup_model
        ;;
    --help|-h)
        echo "Usage: $0 [--daemon|--stop|--status|--warmup|--help]"
        echo ""
        echo "  (no args)   Start bridge in foreground"
        echo "  --daemon    Start bridge in background"
        echo "  --stop      Stop bridge gracefully"
        echo "  --status    Show bridge and infrastructure status"
        echo "  --warmup    Pre-load LLM model weights only"
        exit 0
        ;;
    "")
        start_bridge false
        ;;
    *)
        log_err "Unknown option: $1"
        echo "Usage: $0 [--daemon|--stop|--status|--warmup|--help]"
        exit 1
        ;;
esac
