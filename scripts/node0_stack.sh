#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# BIZRA Node0 — Unified Stack Orchestrator
# ═══════════════════════════════════════════════════════════════════════
# Single entry point for the entire BIZRA service mesh.
# Replaces ad-hoc multi-compose startup with standardized PDCA cycle.
#
# Standing on Giants:
#   Deming (PDCA quality cycle) · Docker Compose v2 · 12-Factor App
#
# Usage:
#   ./scripts/node0_stack.sh start    # Start all services in order
#   ./scripts/node0_stack.sh stop     # Graceful shutdown
#   ./scripts/node0_stack.sh status   # Health dashboard
#   ./scripts/node0_stack.sh restart  # Stop + Start
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DUAL_AGENTIC_DIR="/mnt/c/BIZRA-Dual-Agentic-system--main"
SHARED_NETWORK="bizra-mesh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[BIZRA]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; }

# ── Shared Network ─────────────────────────────────────────────────────
ensure_network() {
    if ! docker network inspect "$SHARED_NETWORK" &>/dev/null; then
        docker network create "$SHARED_NETWORK" &>/dev/null
        ok "Created shared network: $SHARED_NETWORK"
    else
        ok "Shared network exists: $SHARED_NETWORK"
    fi
}

# ── Connect container to shared network ────────────────────────────────
bridge_container() {
    local container="$1"
    if docker inspect "$container" &>/dev/null; then
        if ! docker inspect "$container" --format '{{json .NetworkSettings.Networks}}' | grep -q "$SHARED_NETWORK"; then
            docker network connect "$SHARED_NETWORK" "$container" 2>/dev/null && \
                ok "Bridged $container → $SHARED_NETWORK" || true
        fi
    fi
}

# ── Health Check ───────────────────────────────────────────────────────
check_service() {
    local name="$1" port="$2" path="${3:-/}" expected="${4:-200}"
    local code
    code=$(curl -sf -o /dev/null -w "%{http_code}" "localhost:$port$path" 2>/dev/null || echo "000")
    if [ "$code" = "$expected" ] || [ "$code" = "302" ]; then
        ok "$name :$port ($code)"
    else
        fail "$name :$port ($code)"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════════
cmd_start() {
    log "Starting BIZRA Node0 Stack (PDCA: Plan→Do→Check→Act)"
    echo ""

    # Phase 1: Plan — ensure prerequisites
    log "Phase 1: PLAN — Prerequisites"
    ensure_network

    # Phase 2: Do — start services in dependency order
    log "Phase 2: DO — Starting services"

    # Layer 0: Infrastructure (Dual-Agentic provides databases + monitoring)
    if [ -d "$DUAL_AGENTIC_DIR" ]; then
        log "  Layer 0: Infrastructure (Dual-Agentic)"
        (cd "$DUAL_AGENTIC_DIR" && docker compose up -d postgres synapse wisdom vectors 2>&1 | tail -3)
        ok "Databases started"
        sleep 3

        # Layer 1: Core services
        log "  Layer 1: Core services"
        (cd "$DUAL_AGENTIC_DIR" && docker compose up -d elite refinery 2>&1 | tail -3)
        ok "Elite + Refinery started"

        # Layer 2: Monitoring
        log "  Layer 2: Monitoring"
        (cd "$DUAL_AGENTIC_DIR" && docker compose up -d grafana prometheus 2>&1 | tail -3)
        ok "Grafana + Prometheus started"

        # Layer 3: Auxiliary
        log "  Layer 3: Auxiliary"
        (cd "$DUAL_AGENTIC_DIR" && docker compose up -d finance agentic-flow 2>&1 | tail -3)
        ok "Finance + Agentic-Flow started"
    else
        warn "Dual-Agentic dir not found at $DUAL_AGENTIC_DIR — skipping infrastructure"
    fi

    # Layer 4: Data Lake services
    log "  Layer 4: Data Lake kernel"
    (cd "$REPO_ROOT" && docker compose -f docker-compose.unified.yml up -d kernel 2>&1 | tail -3)
    ok "Kernel started"

    # Phase 3: Check — bridge networks and verify health
    log "Phase 3: CHECK — Bridging networks"
    sleep 5

    # Bridge kernel to Dual-Agentic network for service discovery
    bridge_container "bizra-data-lake-kernel-1"

    # Bridge all key containers to shared mesh
    for ctr in \
        bizra-dual-agentic-system--main-synapse-1 \
        bizra-dual-agentic-system--main-elite-1 \
        bizra-dual-agentic-system--main-wisdom-1 \
        bizra-dual-agentic-system--main-vectors-1 \
        bizra-dual-agentic-system--main-postgres-1 \
        bizra-data-lake-kernel-1 \
        bizra-python-api; do
        bridge_container "$ctr" 2>/dev/null
    done

    # Phase 4: Act — verify health
    echo ""
    log "Phase 4: ACT — Health verification"
    sleep 5
    cmd_status
}

# ═══════════════════════════════════════════════════════════════════════
# STOP
# ═══════════════════════════════════════════════════════════════════════
cmd_stop() {
    log "Stopping BIZRA Node0 Stack (graceful)"
    echo ""

    # Stop in reverse order
    log "Stopping Data Lake kernel..."
    (cd "$REPO_ROOT" && docker compose -f docker-compose.unified.yml stop kernel 2>&1 | tail -3)

    if [ -d "$DUAL_AGENTIC_DIR" ]; then
        log "Stopping Dual-Agentic services..."
        (cd "$DUAL_AGENTIC_DIR" && docker compose stop 2>&1 | tail -3)
    fi

    ok "Stack stopped"
}

# ═══════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════
cmd_status() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║            BIZRA NODE0 — SERVICE MESH STATUS                ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    printf "║ %-20s %-8s %-10s\n" "SERVICE" "PORT" "STATUS"
    echo "╠══════════════════════════════════════════════════════════════╣"

    check_service "Kernel"       8010 "/healthz"
    check_service "Elite"        8080 "/health"
    check_service "Refinery"     8081 "/health"
    check_service "Grafana"      3000 "/"
    check_service "Prometheus"   9090 "/-/healthy"

    # Redis
    local redis_ok
    if [ -n "${BIZRA_REDIS_CACHE_PASSWORD:-}" ]; then
        redis_ok=$(redis-cli -p 6379 -a "$BIZRA_REDIS_CACHE_PASSWORD" PING 2>/dev/null | grep -c PONG || true)
    else
        redis_ok=$(redis-cli -p 6379 PING 2>/dev/null | grep -c PONG || true)
    fi
    if [ "$redis_ok" = "1" ]; then ok "Redis Cache :6379 (PONG)"; else fail "Redis Cache :6379"; fi

    local synapse_ok
    if [ -n "${BIZRA_REDIS_PASSWORD:-}" ]; then
        synapse_ok=$(redis-cli -p 6380 -a "$BIZRA_REDIS_PASSWORD" PING 2>/dev/null | grep -c PONG || true)
    else
        synapse_ok=$(redis-cli -p 6380 PING 2>/dev/null | grep -c PONG || true)
    fi
    if [ "$synapse_ok" = "1" ]; then ok "Redis Synapse :6380 (PONG)"; else warn "Redis Synapse :6380 (AUTH?)"; fi

    # Ollama
    local ollama_ok
    ollama_ok=$(curl -sf localhost:11434/ 2>/dev/null | grep -c "running" || true)
    if [ "$ollama_ok" = "1" ]; then ok "Ollama :11434 (running)"; else fail "Ollama :11434"; fi

    check_service "Neo4j"        7474 "/"
    check_service "ChromaDB"     8001 "/api/v1/heartbeat"

    # K8s
    local k8s_nodes
    k8s_nodes=$(kubectl --context k3d-bizra-prod get nodes --no-headers 2>/dev/null | grep -c Ready || true)
    ok "K3d bizra-prod ($k8s_nodes nodes Ready)"

    # Containers
    local total_ctr
    total_ctr=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
    ok "Total containers: $total_ctr"

    # Node0 Pilot
    if [ -f "$REPO_ROOT/sovereign_state/proactive.pid" ]; then
        local pilot_pid
        pilot_pid=$(cat "$REPO_ROOT/sovereign_state/proactive.pid")
        if ps -p "$pilot_pid" &>/dev/null; then
            local uptime
            uptime=$(ps -p "$pilot_pid" -o etime --no-headers | tr -d ' ')
            ok "Node0 Pilot PID $pilot_pid (uptime: $uptime)"
        else
            warn "Node0 Pilot PID $pilot_pid (STALE — process dead)"
        fi
    else
        warn "Node0 Pilot not running"
    fi

    # GPU
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    ok "GPU: $gpu_info"

    echo "╚══════════════════════════════════════════════════════════════╝"
}

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
case "${1:-status}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_stop; sleep 3; cmd_start ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
