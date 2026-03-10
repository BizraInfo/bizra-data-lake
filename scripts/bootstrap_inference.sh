#!/bin/bash
# BIZRA INFERENCE BOOTSTRAP
# Run this on the Windows side with RTX 4090
# Created: 2026-01-29

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "    BIZRA INFERENCE BOOTSTRAP"
echo "═══════════════════════════════════════════════════════════════"

MODEL_DIR="${BIZRA_MODEL_DIR:-/mnt/c/BIZRA-DATA-LAKE/models}"
MODEL_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_REPO="Qwen/Qwen2.5-1.5B-Instruct-GGUF"

mkdir -p "$MODEL_DIR"

echo ""
echo "[1/4] Checking Python environment..."
python3 --version || { echo "Python3 not found"; exit 1; }

echo ""
echo "[2/4] Installing llama-cpp-python..."
if python3 -c "import llama_cpp" 2>/dev/null; then
    echo "  Already installed"
else
    pip install llama-cpp-python 2>/dev/null || \
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 || \
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
fi

echo ""
echo "[3/4] Downloading model..."
if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "  Model already exists: $MODEL_DIR/$MODEL_NAME"
else
    echo "  Downloading $MODEL_NAME from $MODEL_REPO..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$MODEL_REPO" "$MODEL_NAME" --local-dir "$MODEL_DIR"
    else
        echo "  huggingface-cli not found. Installing..."
        pip install huggingface_hub
        huggingface-cli download "$MODEL_REPO" "$MODEL_NAME" --local-dir "$MODEL_DIR"
    fi
fi

echo ""
echo "[4/4] Validating..."
cd /mnt/c/BIZRA-DATA-LAKE
python3 -c "
import sys
from pathlib import Path
from core.inference.gateway import InferenceGateway, InferenceConfig
import asyncio

MODEL_DIR = Path('/mnt/c/BIZRA-DATA-LAKE/models')
gguf_files = list(MODEL_DIR.glob('*.gguf'))

if not gguf_files:
    print('❌ No model found in', MODEL_DIR)
    sys.exit(1)

model_path = gguf_files[0]
print(f'Found model: {model_path.name}')

config = InferenceConfig(
    model_path=str(model_path),
    n_gpu_layers=-1,
    context_length=4096,
)

async def test():
    gateway = InferenceGateway(config)
    success = await gateway.initialize()
    
    if not success:
        print('❌ Gateway failed to initialize')
        return False
    
    print(f'✅ Gateway status: {gateway.status.value}')
    
    result = await gateway.infer(
        'What is BIZRA? Answer in one sentence.',
        max_tokens=50,
    )
    
    print(f'✅ Inference successful')
    print(f'   Model: {result.model}')
    print(f'   Speed: {result.tokens_per_second} tok/s')
    print(f'   Response: {result.content[:100]}...')
    
    return True

success = asyncio.run(test())
sys.exit(0 if success else 1)
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "    BOOTSTRAP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
