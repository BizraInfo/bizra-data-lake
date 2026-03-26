#!/bin/bash
# BIZRA Kernel — background start with no pipe
set -e
cd /mnt/c/BIZRA-DATA-LAKE
rm -f sovereign_state/kernel.pid
pkill -9 -f kernel_daemon 2>/dev/null || true
pkill -9 -f ghost_ws 2>/dev/null || true
pkill -9 -f desktop_bridge 2>/dev/null || true
sleep 2
export PYTHONUNBUFFERED=1
nohup python3 core/sovereign/kernel_daemon.py > sovereign_state/kernel_boot.log 2>&1 &
PID=$!
disown $PID
echo "KERNEL_STARTED PID=$PID"
# Wait for health
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:9740/api/health > /dev/null 2>&1; then
    echo "KERNEL_ALIVE after ${i}s"
    curl -s http://127.0.0.1:9740/api/heartbeat
    exit 0
  fi
  sleep 1
done
echo "KERNEL_TIMEOUT — check sovereign_state/kernel_boot.log"
tail -5 sovereign_state/kernel_boot.log 2>/dev/null
exit 1
