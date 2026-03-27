#!/bin/bash
# Install BIZRA auto-watchdog into crontab (runs every 5 minutes)
SCRIPT="/mnt/c/BIZRA-DATA-LAKE/scripts/auto_watchdog.sh"
LOG="/mnt/c/BIZRA-DATA-LAKE/sovereign_state/watchdog.log"
CRON_LINE="*/5 * * * * bash $SCRIPT >> $LOG 2>&1"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "auto_watchdog"; then
    echo "Watchdog already installed in crontab"
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "Watchdog installed: runs every 5 minutes"
    echo "Log: $LOG"
fi

# Verify
echo ""
echo "Current crontab:"
crontab -l 2>/dev/null | grep bizra || echo "(none)"
echo ""
echo "7-day autonomous run is now configured."
echo "The watchdog will:"
echo "  - Check kernel health every 5 minutes"
echo "  - Auto-restart if kernel is offline"
echo "  - Generate daily manifest at midnight Dubai time"
