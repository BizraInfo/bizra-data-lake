#!/bin/bash
# Launch watchdog as a new session leader (survives terminal close)
cd /mnt/c/BIZRA-DATA-LAKE
setsid bash scripts/7day_watchdog.sh < /dev/null &
echo "WATCHDOG_LAUNCHED_SETSID PID=$!"
sleep 3
ps aux | grep 7day_watchdog | grep -v grep && echo "CONFIRMED_RUNNING" || echo "LAUNCH_FAILED"
tail -3 sovereign_state/watchdog.log 2>/dev/null
