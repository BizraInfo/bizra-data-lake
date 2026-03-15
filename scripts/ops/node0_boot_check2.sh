#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate 2>&1

echo "=== API ENTRY POINT ==="
grep -n "def serve\|def create_app\|app = \|def run_server\|FastAPI\|uvicorn" core/sovereign/api.py 2>&1 | head -20

echo ""
echo "=== SERVE FUNCTION ==="
grep -n "def serve\|async def serve" core/sovereign/api.py 2>&1

echo ""
echo "=== __MAIN__ SERVE ROUTE ==="
grep -A5 "serve" core/sovereign/__main__.py 2>&1 | grep -A3 "run_server\|api\|serve"

echo ""
echo "=== LM STUDIO GATEWAY ==="
python -c "from core.integration.constants import LMSTUDIO_HOST, LMSTUDIO_PORT, LMSTUDIO_URL; print(f'Host: {LMSTUDIO_HOST}\nPort: {LMSTUDIO_PORT}\nURL: {LMSTUDIO_URL}')" 2>&1

echo ""
echo "=== OLLAMA CHECK ==="
curl -s --max-time 2 http://localhost:11434/api/tags 2>&1 | head -3 || echo "Ollama: not running"

echo ""
echo "=== VITE PROXY TARGET ==="
grep -A3 "proxy" frontend/vite.config.ts 2>&1

echo ""
echo "=== SOVEREIGN SERVE PORT ==="
grep "8080\|8000\|default.*port" core/sovereign/__main__.py 2>&1 | head -5
