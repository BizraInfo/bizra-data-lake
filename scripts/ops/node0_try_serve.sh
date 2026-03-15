#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate 2>&1
echo "=== STARTING SOVEREIGN SERVER ON PORT 8000 ==="
echo "Ollama at localhost:11434 with 7 models"
echo "Press Ctrl+C to stop"
python -m core.sovereign serve --port 8000 2>&1
