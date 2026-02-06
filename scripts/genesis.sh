#!/bin/bash
#
# BIZRA GENESIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# The One Script That Starts Everything
#
# Usage:
#   ./genesis.sh              # Full boot
#   ./genesis.sh --quick      # Skip Docker, local only
#   ./genesis.sh --status     # Just show status
#   ./genesis.sh --stop       # Graceful shutdown
#
# Giants Protocol Applied:
#   Al-Khwarizmi — Algorithmic boot sequence
#   Ibn Sina — Diagnostic pre-flight checks
#   Al-Jazari — Engineering orchestration
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIZRA_DATA_LAKE="${SCRIPT_DIR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "   ____  ___ ________  ___       ____  _____ _   _ _____ ____ ___ ____  "
    echo "  | __ )|_ _|__  /  _ \/ \      / ___|| ____| \ | | ____/ ___|_ _/ ___| "
    echo "  |  _ \ | |  / /| |_) / _ \    | |  _ |  _| |  \| |  _| \___ \| |\___ \ "
    echo "  | |_) || | / /_|  _ / ___ \   | |_| || |___| |\  | |___ ___) | | ___) |"
    echo "  |____/___|____|_| \_\/_/   \_\   \____|_____|_| \_|_____|____/___|____/ "
    echo ""
    echo "   The Seed That Grows Into a Forest — Block0 Ignition Sequence"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight Checks (Ibn Sina's Diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

preflight_checks() {
    log_step "Pre-flight checks..."
    
    local passed=true
    
    # Python
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        log_info "Python: $PYTHON_VERSION"
    else
        passed=false
    fi
    
    # Docker (optional)
    if check_command docker; then
        if docker info &>/dev/null; then
            log_info "Docker: Available"
        else
            log_warn "Docker: Not running"
        fi
    else
        log_warn "Docker: Not installed (will use host services)"
    fi
    
    # Ollama (check if running)
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        log_info "Ollama: Running locally"
    else
        log_warn "Ollama: Not running on localhost"
    fi
    
    # LM Studio (check if running)
    if curl -s http://192.168.56.1:1234/v1/models &>/dev/null 2>&1; then
        log_info "LM Studio: Running on host"
    else
        log_warn "LM Studio: Not available"
    fi
    
    # Check critical files
    if [[ -f "${BIZRA_DATA_LAKE}/nucleus.py" ]]; then
        log_info "Nucleus: Found"
    else
        log_error "Nucleus: Not found at ${BIZRA_DATA_LAKE}/nucleus.py"
        passed=false
    fi
    
    if [[ -f "${BIZRA_DATA_LAKE}/accumulator.py" ]]; then
        log_info "Accumulator: Found"
    else
        log_warn "Accumulator: Not found"
    fi
    
    if [[ -f "${BIZRA_DATA_LAKE}/flywheel.py" ]]; then
        log_info "Flywheel: Found"
    else
        log_warn "Flywheel: Not found"
    fi
    
    # Genesis manifest
    if [[ -f "${BIZRA_DATA_LAKE}/genesis-manifest.yaml" ]]; then
        log_info "Genesis Manifest: Found"
    else
        log_warn "Genesis Manifest: Not found (run verify_genesis_readiness.py first)"
    fi
    
    echo ""
    
    if [[ "$passed" == "false" ]]; then
        log_error "Pre-flight checks failed"
        return 1
    fi
    
    log_info "Pre-flight checks passed"
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Boot Sequence (Al-Khwarizmi's Algorithm)
# ─────────────────────────────────────────────────────────────────────────────

boot_sequence() {
    local quick_mode="$1"
    
    log_step "Boot sequence initiated..."
    
    cd "${BIZRA_DATA_LAKE}"
    
    # Set Python path
    export PYTHONPATH="${BIZRA_DATA_LAKE}:${PYTHONPATH:-}"
    
    if [[ "$quick_mode" == "true" ]]; then
        log_info "Quick mode: Skipping Docker components"
        python3 nucleus.py start --skip-docker
    else
        log_info "Full mode: Including Docker components"
        
        # Start Docker containers if docker-compose exists
        if [[ -f "docker-compose.flywheel.yml" ]]; then
            log_step "Starting Docker stack..."
            docker compose -f docker-compose.flywheel.yml up -d || log_warn "Docker stack failed (continuing)"
            sleep 5
        fi
        
        python3 nucleus.py start
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Status Check
# ─────────────────────────────────────────────────────────────────────────────

show_status() {
    log_step "Checking status..."
    
    cd "${BIZRA_DATA_LAKE}"
    export PYTHONPATH="${BIZRA_DATA_LAKE}:${PYTHONPATH:-}"
    
    python3 nucleus.py health
}

# ─────────────────────────────────────────────────────────────────────────────
# Shutdown
# ─────────────────────────────────────────────────────────────────────────────

shutdown_stack() {
    log_step "Graceful shutdown..."
    
    cd "${BIZRA_DATA_LAKE}"
    export PYTHONPATH="${BIZRA_DATA_LAKE}:${PYTHONPATH:-}"
    
    # Stop nucleus
    python3 nucleus.py stop || true
    
    # Stop Docker if running
    if [[ -f "docker-compose.flywheel.yml" ]]; then
        docker compose -f docker-compose.flywheel.yml down || true
    fi
    
    log_info "Shutdown complete"
}

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Shell
# ─────────────────────────────────────────────────────────────────────────────

interactive_shell() {
    cd "${BIZRA_DATA_LAKE}"
    export PYTHONPATH="${BIZRA_DATA_LAKE}:${PYTHONPATH:-}"
    
    python3 nucleus.py shell
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    banner
    
    local command="${1:-start}"
    local quick_mode="false"
    
    # Parse flags
    for arg in "$@"; do
        case $arg in
            --quick|-q)
                quick_mode="true"
                ;;
            --status|-s)
                command="status"
                ;;
            --stop)
                command="stop"
                ;;
            --shell|-i)
                command="shell"
                ;;
            --help|-h)
                command="help"
                ;;
        esac
    done
    
    case $command in
        start)
            if ! preflight_checks; then
                exit 1
            fi
            boot_sequence "$quick_mode"
            ;;
        status)
            show_status
            ;;
        stop)
            shutdown_stack
            ;;
        shell)
            if ! preflight_checks; then
                exit 1
            fi
            boot_sequence "true"
            interactive_shell
            ;;
        help)
            echo "Usage: $0 [command] [flags]"
            echo ""
            echo "Commands:"
            echo "  start      Full boot sequence (default)"
            echo "  status     Show component status"
            echo "  stop       Graceful shutdown"
            echo "  shell      Interactive mode"
            echo ""
            echo "Flags:"
            echo "  --quick    Skip Docker, local services only"
            echo "  --status   Alias for status command"
            echo "  --stop     Alias for stop command"
            echo "  --shell    Alias for shell command"
            echo "  --help     Show this help"
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
}

main "$@"
