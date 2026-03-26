#!/bin/bash
# BIZRA Kernel Starter — clean stale PIDs and launch
set -e
PID_FILE="/mnt/c/BIZRA-DATA-LAKE/sovereign_state/kernel.pid"
rm -f "$PID_FILE"
pkill -9 -f kernel_daemon 2>/dev/null || true
pkill -9 -f ghost_ws 2>/dev/null || true
pkill -9 -f desktop_bridge 2>/dev/null || true
sleep 2
cd /mnt/c/BIZRA-DATA-LAKE
nohup python3 -u core/sovereign/kernel_daemon.py >> /mnt/c/BIZRA-DATA-LAKE/sovereign_state/kernel_boot.log 2>&1 &
echo "KERNEL_PID=$!"
echo "LAUNCHED"
