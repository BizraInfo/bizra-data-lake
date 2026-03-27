#!/bin/bash
# Test run — first iteration only, with error output visible
REPO="/mnt/c/BIZRA-DATA-LAKE"
LOG="$REPO/sovereign_state/watchdog.log"

echo "Testing watchdog first iteration..."

if curl -s http://127.0.0.1:9740/api/health > /dev/null 2>&1; then
    echo "Kernel ALIVE"
    BEAT=$(curl -s http://127.0.0.1:9740/api/heartbeat 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    l=d.get('latest',{})
    print('beat=%s health=%s rss=%s' % (l.get('beat',0), l.get('health','?'), l.get('memory_rss_mb',0)))
except: print('parse_fail')
" 2>/dev/null || echo "parse_fail")
    echo "Status: $BEAT"
    echo "[$(date -Iseconds)] ALIVE $BEAT" >> "$LOG"
    echo "Logged to: $LOG"
else
    echo "Kernel OFFLINE"
fi

echo "---"
tail -3 "$LOG" 2>/dev/null
echo "TEST_DONE"
