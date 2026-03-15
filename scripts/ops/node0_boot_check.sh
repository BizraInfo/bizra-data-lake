#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate 2>&1
echo "=== PYTHON VERSION ==="
python --version 2>&1
echo "=== IMPORT CHECK ==="
python -c "from core.sovereign.runtime import SovereignRuntime; print('OK: SovereignRuntime')" 2>&1
python -c "from core.sovereign.api import create_app; print('OK: create_app')" 2>&1
python -c "from core.integration.constants import IHSAN_THRESHOLD, LMSTUDIO_URL; print(f'OK: Ihsan={IHSAN_THRESHOLD} LM={LMSTUDIO_URL}')" 2>&1
echo "=== GENESIS CHECK ==="
python -c "import json; d=json.load(open('sovereign_state/node0_genesis.json')); print(f'Node: {d[\"identity\"][\"name\"]} ID: {d[\"identity\"][\"node_id\"]}')" 2>&1
echo "=== LM STUDIO CHECK ==="
curl -s --max-time 3 http://172.22.48.1:1234/v1/models 2>&1 | python -c "import sys,json; d=json.load(sys.stdin); print(f'LM Studio: {len(d.get(\"data\",[]))} models')" 2>&1 || echo "LM Studio: not reachable"
echo "=== FRONTEND CHECK ==="
ls -la frontend/package.json 2>&1
ls -la frontend/dist/index.html 2>&1 || echo "No dist build yet"
echo "=== DONE ==="
