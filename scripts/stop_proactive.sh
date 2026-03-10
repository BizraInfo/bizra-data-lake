#!/bin/bash
# ============================================================================
# Proactive Sovereign Entity - Stop Script
# ============================================================================
# Gracefully stops the running Proactive Sovereign Entity.
#
# Usage:
#   ./scripts/stop_proactive.sh [--force]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/sovereign_state/proactive.pid"
LOG_DIR="$PROJECT_ROOT/logs/proactive"

FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
fi

echo "============================================================================"
echo "    PROACTIVE SOVEREIGN ENTITY - STOPPING"
echo "============================================================================"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "  No running entity found (PID file missing)"
    exit 0
fi

PID=$(cat "$PID_FILE")

# Check if process is running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "  Entity not running (stale PID file)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "  Stopping entity (PID: $PID)..."

if [ "$FORCE" = true ]; then
    # Force kill
    kill -9 "$PID" 2>/dev/null || true
    echo "  Force killed"
else
    # Graceful shutdown
    kill -SIGTERM "$PID" 2>/dev/null

    # Wait for process to exit (up to 30 seconds)
    WAIT=0
    while kill -0 "$PID" 2>/dev/null && [ $WAIT -lt 30 ]; do
        sleep 1
        WAIT=$((WAIT + 1))
        echo -n "."
    done
    echo ""

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "  Graceful shutdown timed out, forcing..."
        kill -9 "$PID" 2>/dev/null || true
    fi
fi

# Remove PID file
rm -f "$PID_FILE"

# Log shutdown
if [ -d "$LOG_DIR" ]; then
    echo "[$(date -Iseconds)] Entity stopped" >> "$LOG_DIR/startup.log"
fi

echo ""
echo "  Proactive Sovereign Entity stopped"
echo ""
echo "============================================================================"
