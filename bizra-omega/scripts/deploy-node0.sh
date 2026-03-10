#!/usr/bin/env bash
# BIZRA Node0 Deployment Script
# Genesis Hash: a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b
#
# Usage:
#   ./scripts/deploy-node0.sh [--genesis] [--cuda]
#
# Options:
#   --genesis   Run Genesis ceremony (first time only)
#   --cuda      Enable CUDA/GPU support

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SOVEREIGN_STATE_DIR="${BIZRA_SOVEREIGN_STATE_DIR:-$PROJECT_DIR/sovereign_state}"

# Parse arguments
RUN_GENESIS=false
USE_CUDA=false

for arg in "$@"; do
    case $arg in
        --genesis) RUN_GENESIS=true ;;
        --cuda) USE_CUDA=true ;;
        *) log_error "Unknown option: $arg" ;;
    esac
done

# Banner
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║     ██████╗ ██╗███████╗██████╗  █████╗                            ║"
echo "║     ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                           ║"
echo "║     ██████╔╝██║  ███╔╝ ██████╔╝███████║                           ║"
echo "║     ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                           ║"
echo "║     ██████╔╝██║███████╗██║  ██║██║  ██║                           ║"
echo "║     ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                           ║"
echo "║                                                                    ║"
echo "║           Node0 Deployment — MoMo (محمد) Dubai, UAE               ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
fi

# Check for .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    log_warn ".env file not found, copying from .env.example..."
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_success "Created .env from template"
        log_warn "Please review and update .env with your configuration"
    else
        log_error ".env.example not found"
    fi
fi

# Create sovereign state directory
mkdir -p "$SOVEREIGN_STATE_DIR"
log_success "Sovereign state directory: $SOVEREIGN_STATE_DIR"

# Run Genesis ceremony if requested
if [ "$RUN_GENESIS" = true ]; then
    log_info "Running Genesis ceremony..."

    if [ -f "$SOVEREIGN_STATE_DIR/genesis_hash.txt" ]; then
        EXISTING_HASH=$(cat "$SOVEREIGN_STATE_DIR/genesis_hash.txt" | tr -d '[:space:]')
        log_warn "Genesis already exists with hash: $EXISTING_HASH"
        read -p "Are you sure you want to regenerate? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing Genesis"
            RUN_GENESIS=false
        fi
    fi

    if [ "$RUN_GENESIS" = true ]; then
        # Build and run genesis
        docker build -t bizra:genesis "$PROJECT_DIR"
        docker run --rm -v "$SOVEREIGN_STATE_DIR:/home/bizra/.bizra/sovereign_state" \
            bizra:genesis node0-genesis
        log_success "Genesis ceremony complete!"

        if [ -f "$SOVEREIGN_STATE_DIR/genesis_hash.txt" ]; then
            GENESIS_HASH=$(cat "$SOVEREIGN_STATE_DIR/genesis_hash.txt" | tr -d '[:space:]')
            echo ""
            echo "═══════════════════════════════════════════════════════════════"
            echo "  Genesis Hash: $GENESIS_HASH"
            echo "═══════════════════════════════════════════════════════════════"
            echo ""
        fi
    fi
fi

# Verify Genesis exists
if [ ! -f "$SOVEREIGN_STATE_DIR/genesis_hash.txt" ]; then
    log_error "No Genesis found. Run with --genesis flag first."
fi

GENESIS_HASH=$(cat "$SOVEREIGN_STATE_DIR/genesis_hash.txt" | tr -d '[:space:]')
log_success "Genesis Hash: $GENESIS_HASH"

# Build Docker image
log_info "Building Docker image..."
cd "$PROJECT_DIR"

if [ "$USE_CUDA" = true ]; then
    docker build -t bizra:node0 --build-arg CUDA=1 .
    log_success "Built CUDA-enabled image"
else
    docker build -t bizra:node0 .
    log_success "Built CPU image"
fi

# Deploy with docker-compose
log_info "Deploying Node0..."

if [ "$USE_CUDA" = true ]; then
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
else
    docker compose up -d
fi

log_success "Node0 deployed!"

# Wait for health check
log_info "Waiting for services to become healthy..."
sleep 5

# Check service status
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Service Status"
echo "═══════════════════════════════════════════════════════════════"
docker compose ps
echo ""

# Health check
if curl -sf http://localhost:3001/api/v1/health > /dev/null 2>&1; then
    log_success "API is healthy at http://localhost:3001"
else
    log_warn "API not responding yet (may still be starting)"
fi

# PAT/SAT roster summary
if [ -f "$SOVEREIGN_STATE_DIR/pat_roster.txt" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PAT (Personal Agentic Team) — 7 Agents"
    echo "═══════════════════════════════════════════════════════════════"
    cat "$SOVEREIGN_STATE_DIR/pat_roster.txt"
fi

if [ -f "$SOVEREIGN_STATE_DIR/sat_roster.txt" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SAT (Shared Agentic Team) — Protocol Army"
    echo "═══════════════════════════════════════════════════════════════"
    cat "$SOVEREIGN_STATE_DIR/sat_roster.txt"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║   Node0 is LIVE!                                                   ║"
echo "║                                                                    ║"
echo "║   API:           http://localhost:3001                             ║"
echo "║   Resource Pool: http://localhost:8946                             ║"
echo "║   Ollama:        http://localhost:11434                            ║"
echo "║                                                                    ║"
echo "║   Genesis: $GENESIS_HASH         ║"
echo "║                                                                    ║"
echo "║   بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ                                            ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
