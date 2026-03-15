#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 -m pytest tests/core/reasoning/ -x -q --timeout=30 2>&1 | tail -15
