#!/bin/bash
# BIZRA 7-Day Autonomous Watchdog — self-contained loop for WSL2
# Usage: nohup bash scripts/7day_watchdog.sh &
# No set -e — watchdog must survive transient failures
REPO="/mnt/c/BIZRA-DATA-LAKE"
LOG="$REPO/sovereign_state/watchdog.log"
CHECK_INTERVAL=300  # 5 minutes
DAY_COUNT=0
MAX_DAYS=7

echo "[$(date -Iseconds)] 7-day watchdog started (check every ${CHECK_INTERVAL}s)" >> "$LOG"

while [ $DAY_COUNT -lt $MAX_DAYS ]; do
    # Check kernel health
    if curl -s http://127.0.0.1:9740/api/health > /dev/null 2>&1; then
        BEAT=$(curl -s http://127.0.0.1:9740/api/heartbeat 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    l=d.get('latest',{})
    print('beat=%s health=%s rss=%s' % (l.get('beat',0), l.get('health','?'), l.get('memory_rss_mb',0)))
except: print('parse_fail')
" 2>/dev/null || echo "parse_fail")
        echo "[$(date -Iseconds)] ALIVE $BEAT" >> "$LOG"
    else
        echo "[$(date -Iseconds)] OFFLINE — restarting kernel" >> "$LOG"
        cd "$REPO"
        rm -f sovereign_state/kernel.pid
        pkill -9 -f kernel_daemon 2>/dev/null || true
        pkill -9 -f ghost_ws 2>/dev/null || true
        pkill -9 -f desktop_bridge 2>/dev/null || true
        sleep 3
        nohup python3 -u core/sovereign/kernel_daemon.py >> sovereign_state/kernel_boot.log 2>&1 &
        disown
        echo "[$(date -Iseconds)] Kernel restarted PID=$!" >> "$LOG"
        sleep 50
        if curl -s http://127.0.0.1:9740/api/health > /dev/null 2>&1; then
            echo "[$(date -Iseconds)] Restart SUCCESSFUL" >> "$LOG"
        else
            echo "[$(date -Iseconds)] Restart FAILED" >> "$LOG"
        fi
    fi

    # Check if a new day started (UTC midnight = Dubai 4am)
    CURRENT_DAY=$(date -u +%Y-%m-%d)
    MANIFEST="$REPO/sovereign_state/manifests/manifest_${CURRENT_DAY}.json"
    HOUR=$(date -u +%H)
    if [ "$HOUR" = "20" ] && [ ! -f "$MANIFEST" ]; then
        # Midnight Dubai — generate daily manifest
        echo "[$(date -Iseconds)] Generating manifest for $CURRENT_DAY" >> "$LOG"
        cd "$REPO" && bash scripts/first_manifest.sh >> "$LOG" 2>&1 || true
        # Copy to evidence
        mkdir -p "$REPO/evidence/manifests"
        cp "$MANIFEST" "$REPO/evidence/manifests/" 2>/dev/null || true
        DAY_COUNT=$((DAY_COUNT + 1))
        echo "[$(date -Iseconds)] Manifest #$DAY_COUNT generated" >> "$LOG"
    fi

    sleep $CHECK_INTERVAL
done

echo "[$(date -Iseconds)] 7-day watchdog complete — $DAY_COUNT manifests generated" >> "$LOG"
