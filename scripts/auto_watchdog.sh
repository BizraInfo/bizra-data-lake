#!/bin/bash
# BIZRA Auto-Watchdog — restarts kernel if dead, runs daily manifest at midnight
# Add to crontab: */5 * * * * bash /mnt/c/BIZRA-DATA-LAKE/scripts/auto_watchdog.sh >> /mnt/c/BIZRA-DATA-LAKE/sovereign_state/watchdog.log 2>&1
set -e
REPO="/mnt/c/BIZRA-DATA-LAKE"
LOG="$REPO/sovereign_state/watchdog.log"
MANIFEST_DIR="$REPO/sovereign_state/manifests"
MANIFEST_EVIDENCE="$REPO/evidence/manifests"

# Check kernel health
if curl -s http://127.0.0.1:9740/api/health > /dev/null 2>&1; then
    # Kernel is alive — check if it's midnight for daily manifest
    HOUR=$(date -u +%H)
    MINUTE=$(date -u +%M)
    if [ "$HOUR" = "20" ] && [ "$MINUTE" -lt "05" ]; then
        # ~midnight Dubai (UTC+4) — generate daily manifest
        TODAY=$(date -u +%Y-%m-%d)
        if [ ! -f "$MANIFEST_DIR/manifest_${TODAY}.json" ]; then
            echo "[$(date -Iseconds)] Generating daily manifest for $TODAY"
            cd "$REPO" && python3 scripts/first_manifest.sh 2>/dev/null || true
        fi
    fi
else
    echo "[$(date -Iseconds)] Kernel OFFLINE — restarting"
    cd "$REPO"
    rm -f sovereign_state/kernel.pid
    pkill -9 -f kernel_daemon 2>/dev/null || true
    pkill -9 -f ghost_ws 2>/dev/null || true
    pkill -9 -f desktop_bridge 2>/dev/null || true
    sleep 2
    nohup python3 -u core/sovereign/kernel_daemon.py >> sovereign_state/kernel_boot.log 2>&1 &
    disown
    echo "[$(date -Iseconds)] Kernel restarted PID=$!"
    # Wait and verify
    sleep 45
    if curl -s http://127.0.0.1:9740/api/health > /dev/null 2>&1; then
        echo "[$(date -Iseconds)] Kernel HEALTHY after restart"
    else
        echo "[$(date -Iseconds)] Kernel FAILED to restart — check kernel_boot.log"
    fi
fi
