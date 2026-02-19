#!/bin/bash
# BIZRA Elite Framework — Deployment Script
# DevOps v1.0.0 | Constitutional Deployment
#
# Usage: ./deploy.sh [command] [options]
#
# Commands:
#   up        Start all services
#   down      Stop all services
#   status    Show service status
#   logs      Tail service logs
#   validate  Run pre-deployment validation
#   rollback  Rollback to previous version

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/elite-compose.yaml"
ENV_FILE="$SCRIPT_DIR/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check env file
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warn ".env file not found. Copying template..."
        cp "$SCRIPT_DIR/env.template" "$ENV_FILE"
        log_warn "Please configure $ENV_FILE with your secrets"
    fi

    log_success "Prerequisites check passed"
}

validate_deployment() {
    log_info "Running pre-deployment validation..."

    cd "$PROJECT_ROOT"

    # Activate venv — support both .venv (standard) and .venv-linux (WSL)
    local VENV_ACTIVATE=""
    if [[ -f ".venv-linux/bin/activate" ]]; then
        VENV_ACTIVATE=".venv-linux/bin/activate"
    elif [[ -f ".venv/bin/activate" ]]; then
        VENV_ACTIVATE=".venv/bin/activate"
    fi

    if [[ -n "$VENV_ACTIVATE" ]]; then
        source "$VENV_ACTIVATE"

        # Run elite tests
        log_info "Running Elite Framework tests..."
        python -m pytest tests/core/elite/ -v --tb=short || {
            log_error "Elite tests failed. Aborting deployment."
            exit 1
        }

        # Validate constitutional thresholds from authoritative source
        log_info "Validating constitutional thresholds..."
        python -c "
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    ADL_GINI_THRESHOLD,
)
assert UNIFIED_IHSAN_THRESHOLD >= 0.95, f'Ihsan threshold too low: {UNIFIED_IHSAN_THRESHOLD}'
assert UNIFIED_SNR_THRESHOLD >= 0.85, f'SNR threshold too low: {UNIFIED_SNR_THRESHOLD}'
assert SNR_THRESHOLD_T0_ELITE >= 0.98, f'Elite SNR too low: {SNR_THRESHOLD_T0_ELITE}'
assert ADL_GINI_THRESHOLD <= 0.40, f'Gini gate too loose: {ADL_GINI_THRESHOLD}'
print(f'Constitutional thresholds validated: Ihsan={UNIFIED_IHSAN_THRESHOLD}, SNR={UNIFIED_SNR_THRESHOLD}, Elite={SNR_THRESHOLD_T0_ELITE}, Gini<={ADL_GINI_THRESHOLD}')
" || {
            log_error "Constitutional threshold validation failed"
            exit 1
        }

        # Validate Phase 44 hash table infrastructure loads
        log_info "Validating Phase 44 infrastructure..."
        python -c "
from core.hashtable import BloomFilter, MerkleTree, SkillCache
bf = BloomFilter(100)
bf.add(b'test')
assert b'test' in bf, 'BloomFilter membership check failed'
mt = MerkleTree()
mt.append(b'leaf')
assert mt.leaf_count == 1, 'MerkleTree append failed'
print('Phase 44 Hash Table Infrastructure: OK')
" || {
            log_error "Phase 44 infrastructure validation failed"
            exit 1
        }

        log_success "Pre-deployment validation passed"
    else
        log_warn "Python venv not found. Skipping tests."
    fi
}

deploy_up() {
    log_info "Starting BIZRA Elite services..."

    check_prerequisites

    if [[ "${VALIDATE:-true}" == "true" ]]; then
        validate_deployment
    fi

    cd "$SCRIPT_DIR"

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    log_success "Services started. Checking health..."

    sleep 5
    docker compose -f "$COMPOSE_FILE" ps

    record_version

    log_success "Deployment complete!"
    echo ""
    log_info "Access points:"
    echo "  - Grafana:    http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Quality Gate: http://localhost:8095"
}

deploy_down() {
    log_info "Stopping BIZRA Elite services..."

    cd "$SCRIPT_DIR"
    docker compose -f "$COMPOSE_FILE" down

    log_success "Services stopped"
}

show_status() {
    log_info "Service status:"
    cd "$SCRIPT_DIR"
    docker compose -f "$COMPOSE_FILE" ps
}

show_logs() {
    local service="${1:-}"
    cd "$SCRIPT_DIR"

    if [[ -n "$service" ]]; then
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker compose -f "$COMPOSE_FILE" logs -f
    fi
}

rollback() {
    log_warn "Rolling back to previous version..."

    local VERSION_FILE="$SCRIPT_DIR/.deploy-version"
    local PREVIOUS_FILE="$SCRIPT_DIR/.deploy-version.prev"

    if [[ ! -f "$PREVIOUS_FILE" ]]; then
        log_error "No previous version recorded. Cannot rollback."
        log_info "Rollback requires at least two deployments with version tracking."
        exit 1
    fi

    local PREV_VERSION
    PREV_VERSION="$(cat "$PREVIOUS_FILE")"
    log_info "Rolling back to: $PREV_VERSION"

    # Stop current services
    deploy_down

    # Restore previous version tag
    cp "$PREVIOUS_FILE" "$VERSION_FILE"

    # Set image tag for compose
    export BIZRA_IMAGE_TAG="$PREV_VERSION"

    # Restart with previous version (skip validation — it was validated when first deployed)
    VALIDATE=false deploy_up

    log_success "Rollback to $PREV_VERSION complete"
}

# Record deployment version for rollback support
record_version() {
    local VERSION_FILE="$SCRIPT_DIR/.deploy-version"
    local PREVIOUS_FILE="$SCRIPT_DIR/.deploy-version.prev"
    local GIT_SHA

    GIT_SHA="$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    local VERSION="${BIZRA_IMAGE_TAG:-${GIT_SHA}}"

    # Shift current to previous
    if [[ -f "$VERSION_FILE" ]]; then
        cp "$VERSION_FILE" "$PREVIOUS_FILE"
    fi

    echo "$VERSION" > "$VERSION_FILE"
    log_info "Recorded deployment version: $VERSION"
}

# =============================================================================
# MCP SERVICE MESH OPERATIONS
# =============================================================================

MCP_COMPOSE_FILE="$SCRIPT_DIR/mcp-compose.yaml"

mcp_up() {
    log_info "Starting BIZRA MCP Service Mesh..."

    check_prerequisites

    cd "$SCRIPT_DIR"

    # Build MCP image
    log_info "Building MCP container image..."
    docker compose -f "$MCP_COMPOSE_FILE" --env-file "$ENV_FILE" build

    # Start services
    docker compose -f "$MCP_COMPOSE_FILE" --env-file "$ENV_FILE" up -d

    log_success "MCP services starting. Waiting for health checks..."

    sleep 10
    docker compose -f "$MCP_COMPOSE_FILE" ps

    log_success "MCP Service Mesh deployed!"
    echo ""
    log_info "MCP access points:"
    echo "  - Gateway:    http://localhost:8080"
    echo "  - Sovereign:  http://localhost:8081"
    echo "  - Ecosystem:  http://localhost:8082"
    echo "  - DDAGI:      http://localhost:8083"
    echo "  - Peak:       http://localhost:8084"
    echo "  - Lake:       http://localhost:8085"
}

mcp_down() {
    log_info "Stopping BIZRA MCP Service Mesh..."

    cd "$SCRIPT_DIR"
    docker compose -f "$MCP_COMPOSE_FILE" down

    log_success "MCP services stopped"
}

mcp_status() {
    log_info "MCP Service Mesh status:"
    cd "$SCRIPT_DIR"
    docker compose -f "$MCP_COMPOSE_FILE" ps

    echo ""
    log_info "Health checks:"
    for port in 8080 8081 8082 8083 8084 8085; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            log_success "  :${port} — healthy"
        else
            log_warn "  :${port} — unavailable"
        fi
    done
}

mcp_logs() {
    local service="${1:-}"
    cd "$SCRIPT_DIR"

    if [[ -n "$service" ]]; then
        docker compose -f "$MCP_COMPOSE_FILE" logs -f "$service"
    else
        docker compose -f "$MCP_COMPOSE_FILE" logs -f
    fi
}

# =============================================================================
# FULL STACK (Elite + MCP)
# =============================================================================

all_up() {
    deploy_up
    echo ""
    mcp_up
    log_success "Full BIZRA stack deployed (Elite + MCP)"
}

all_down() {
    mcp_down
    deploy_down
    log_success "Full BIZRA stack stopped"
}

# =============================================================================
# MAIN
# =============================================================================

case "${1:-help}" in
    up)
        deploy_up
        ;;
    down)
        deploy_down
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    validate)
        validate_deployment
        ;;
    rollback)
        rollback
        ;;
    mcp-up)
        mcp_up
        ;;
    mcp-down)
        mcp_down
        ;;
    mcp-status)
        mcp_status
        ;;
    mcp-logs)
        mcp_logs "${2:-}"
        ;;
    all-up)
        all_up
        ;;
    all-down)
        all_down
        ;;
    help|*)
        echo "BIZRA Elite Framework — Deployment"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Core Commands:"
        echo "  up           Start Elite services"
        echo "  down         Stop Elite services"
        echo "  status       Show Elite service status"
        echo "  logs         Tail logs (optionally specify service)"
        echo "  validate     Run pre-deployment validation"
        echo "  rollback     Rollback to previous version"
        echo ""
        echo "MCP Commands:"
        echo "  mcp-up       Start MCP Service Mesh (6 servers)"
        echo "  mcp-down     Stop MCP Service Mesh"
        echo "  mcp-status   Show MCP status + health checks"
        echo "  mcp-logs     Tail MCP logs (optionally specify service)"
        echo ""
        echo "Full Stack:"
        echo "  all-up       Start Elite + MCP together"
        echo "  all-down     Stop all services"
        echo ""
        echo "Environment variables:"
        echo "  VALIDATE=false  Skip pre-deployment tests"
        ;;
esac
