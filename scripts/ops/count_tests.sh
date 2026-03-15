#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 -m pytest --collect-only -q tests/ -m "not slow" --ignore=tests/root_legacy --timeout=30 2>&1 | tail -5
