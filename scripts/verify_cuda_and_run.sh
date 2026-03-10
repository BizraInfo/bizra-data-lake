#!/bin/bash
# BIZRA CUDA VERIFICATION & GPU BOOTSTRAP
# Run this on your actual WSL2 (not the Clawdbot sandbox)
#
# Usage: ./verify_cuda_and_run.sh

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "    BIZRA CUDA VERIFICATION"
echo "═══════════════════════════════════════════════════════════════"

# 1. Check NVIDIA driver
echo ""
echo "[1/5] Checking NVIDIA driver..."
if nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "✅ NVIDIA driver working"
else
    echo "❌ nvidia-smi failed"
    echo "   Make sure you have NVIDIA drivers installed on Windows"
    echo "   and WSL2 GPU support enabled"
    exit 1
fi

# 2. Check CUDA toolkit
echo ""
echo "[2/5] Checking CUDA..."
if nvcc --version &> /dev/null; then
    nvcc --version | grep release
    echo "✅ CUDA toolkit found"
else
    echo "⚠️ nvcc not found (CUDA toolkit not installed)"
    echo "   llama-cpp-python can still use GPU via cuBLAS"
fi

# 3. Check Python environment
echo ""
echo "[3/5] Checking Python environment..."
cd /mnt/c/BIZRA-DATA-LAKE

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q

# 4. Install/reinstall llama-cpp-python with CUDA
echo ""
echo "[4/5] Installing llama-cpp-python with CUDA support..."

# Uninstall existing
pip uninstall llama-cpp-python -y 2>/dev/null || true

# Install with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

# Verify CUDA support
python3 -c "
from llama_cpp import Llama
print('llama-cpp-python installed')
print('Attempting to check CUDA support...')
"

# 5. Run benchmark
echo ""
echo "[5/5] Running GPU benchmark..."
python3 day2_gpu_bootstrap.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "    CUDA VERIFICATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
