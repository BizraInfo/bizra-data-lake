#!/bin/bash
ps aux | grep 7day_watchdog | grep -v grep && echo "WATCHDOG_RUNNING" || echo "WATCHDOG_NOT_RUNNING"
echo "---"
cat /mnt/c/BIZRA-DATA-LAKE/sovereign_state/watchdog.log 2>/dev/null | tail -5
echo "---DONE---"
