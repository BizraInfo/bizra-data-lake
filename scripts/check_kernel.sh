#!/bin/bash
curl -s http://127.0.0.1:9740/api/health 2>/dev/null && echo "" || echo "KERNEL_NOT_READY"
echo "---"
tail -3 /mnt/c/BIZRA-DATA-LAKE/sovereign_state/kernel.log 2>/dev/null
echo "---"
cat /mnt/c/BIZRA-DATA-LAKE/sovereign_state/kernel_boot.log 2>/dev/null | tail -5
echo "---DONE---"
