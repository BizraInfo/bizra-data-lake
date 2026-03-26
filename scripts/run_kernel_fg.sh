#!/bin/bash
# Run kernel in foreground to see actual crash output
cd /mnt/c/BIZRA-DATA-LAKE
rm -f sovereign_state/kernel.pid
python3 core/sovereign/kernel_daemon.py 2>&1 | head -80
