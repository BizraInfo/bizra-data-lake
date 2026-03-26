#!/bin/bash
echo "=== PROCESS CHECK ==="
ps aux | grep kernel_daemon | grep -v grep || echo "KERNEL_PROCESS_NOT_RUNNING"
echo ""
echo "=== BOOT LOG ==="
cat /mnt/c/BIZRA-DATA-LAKE/sovereign_state/kernel_boot.log 2>/dev/null | tail -20
echo ""
echo "=== PORT CHECK ==="
ss -tlnp 2>/dev/null | grep 9740 || echo "PORT_9740_NOT_LISTENING"
echo "=== DONE ==="
