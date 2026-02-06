#!/bin/bash
# ============================================================================
# Proactive Sovereign Entity - Start Script
# ============================================================================
# Launches the 24/7 Proactive Sovereign Entity with all subsystems.
#
# Usage:
#   ./scripts/start_proactive.sh [--mode MODE] [--config CONFIG]
#
# Modes:
#   reactive          - Traditional request-response only
#   proactive_suggest - Detect & suggest, require approval
#   proactive_auto    - Auto-execute within constraints
#   proactive_partner - Full proactive partner mode (default)
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs/proactive"
PID_FILE="$PROJECT_ROOT/sovereign_state/proactive.pid"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config/proactive_config.yaml}"
MODE="${MODE:-proactive_partner}"
VENV_DIR="$PROJECT_ROOT/.venv-linux"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--mode MODE] [--config CONFIG]"
            echo ""
            echo "Modes: reactive, proactive_suggest, proactive_auto, proactive_partner"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/sovereign_state/checkpoints"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Proactive Sovereign Entity already running (PID: $OLD_PID)"
        echo "Use scripts/stop_proactive.sh to stop first"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found at $VENV_DIR"
    echo "Please create it with: python -m venv $VENV_DIR"
    exit 1
fi

# Export environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export BIZRA_PROACTIVE_MODE="$MODE"
export BIZRA_CONFIG_FILE="$CONFIG_FILE"

echo "============================================================================"
echo "    PROACTIVE SOVEREIGN ENTITY - STARTING"
echo "============================================================================"
echo ""
echo "  Mode:    $MODE"
echo "  Config:  $CONFIG_FILE"
echo "  Logs:    $LOG_DIR"
echo "  PID:     $PID_FILE"
echo ""

# Create startup log
STARTUP_LOG="$LOG_DIR/startup.log"
echo "[$(date -Iseconds)] Starting Proactive Sovereign Entity" > "$STARTUP_LOG"
echo "[$(date -Iseconds)] Mode: $MODE" >> "$STARTUP_LOG"
echo "[$(date -Iseconds)] Config: $CONFIG_FILE" >> "$STARTUP_LOG"

# Launch the entity
python -c "
import asyncio
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('$LOG_DIR/sovereign.log'),
    ]
)

from core.sovereign.proactive_integration import (
    ProactiveSovereignEntity,
    EntityConfig,
    EntityMode,
)
from core.sovereign.autonomy_matrix import AutonomyLevel

# Get mode from environment
mode_str = os.environ.get('BIZRA_PROACTIVE_MODE', 'proactive_partner')
mode_map = {
    'reactive': EntityMode.REACTIVE,
    'proactive_suggest': EntityMode.PROACTIVE_SUGGEST,
    'proactive_auto': EntityMode.PROACTIVE_AUTO,
    'proactive_partner': EntityMode.PROACTIVE_PARTNER,
}
mode = mode_map.get(mode_str, EntityMode.PROACTIVE_PARTNER)

# Create configuration
config = EntityConfig(
    mode=mode,
    ihsan_threshold=0.95,
    default_autonomy=AutonomyLevel.AUTOLOW,
)

# Create and start entity
entity = ProactiveSovereignEntity(config)

async def main():
    try:
        await entity.start()
    except KeyboardInterrupt:
        print('\nShutting down...')
        entity.stop()

asyncio.run(main())
" &

# Save PID
echo $! > "$PID_FILE"

echo ""
echo "  Proactive Sovereign Entity started (PID: $(cat "$PID_FILE"))"
echo ""
echo "  To monitor: tail -f $LOG_DIR/sovereign.log"
echo "  To stop:    ./scripts/stop_proactive.sh"
echo ""
echo "============================================================================"

# Log startup completion
echo "[$(date -Iseconds)] Entity started with PID $(cat "$PID_FILE")" >> "$STARTUP_LOG"
