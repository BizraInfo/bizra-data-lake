#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
timeout 120 python3 core/pat/sovereign_scan.py 04_GOLD 2>&1
