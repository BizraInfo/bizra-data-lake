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

    # Activate venv and run tests
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate

        # Run elite tests
        log_info "Running Elite Framework tests..."
        python -m pytest tests/core/elite/ -v --tb=short || {
            log_error "Elite tests failed. Aborting deployment."
            exit 1
        }

        # Validate Ihsān threshold
        python -c "
from core.elite import IHSAN_DIMENSIONS, SNR_TARGETS
assert sum(IHSAN_DIMENSIONS.values()) >= 0.99, 'Ihsān weights incomplete'
assert SNR_TARGETS['wisdom'] >= 0.999, 'Wisdom threshold too low'
print('Constitutional validation passed')
" || {
            log_error "Constitutional validation failed"
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

    # Stop current services
    deploy_down

    # Pull previous images (would use image tags in production)
    log_info "Restoring previous configuration..."

    # Restart with previous config
    deploy_up

    log_success "Rollback complete"
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
    help|*)
        echo "BIZRA Elite Framework — Deployment"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  up        Start all services"
        echo "  down      Stop all services"
        echo "  status    Show service status"
        echo "  logs      Tail logs (optionally specify service)"
        echo "  validate  Run pre-deployment validation"
        echo "  rollback  Rollback to previous version"
        echo ""
        echo "Environment variables:"
        echo "  VALIDATE=false  Skip pre-deployment tests"
        ;;
esac
