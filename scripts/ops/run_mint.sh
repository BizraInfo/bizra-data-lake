#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 core/sat/mint_court.py NODE0_FOUNDER 04_GOLD 2>&1
