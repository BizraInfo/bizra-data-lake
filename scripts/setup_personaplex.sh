#!/bin/bash
# BIZRA PERSONAPLEX SETUP
# Full-duplex voice interface for BIZRA
#
# PersonaPlex = Speech-to-Speech + Voice Cloning + Persona Control
# Released: 2026-01-15 by NVIDIA (Commercial OK)

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "    BIZRA VOICE: PERSONAPLEX SETUP"
echo "═══════════════════════════════════════════════════════════════"

cd /mnt/c/BIZRA-DATA-LAKE

# 1. Install Opus codec
echo ""
echo "[1/5] Installing Opus audio codec..."
sudo apt-get update -qq
sudo apt-get install -y libopus-dev

# 2. Clone PersonaPlex
echo ""
echo "[2/5] Cloning PersonaPlex..."
if [ -d "personaplex" ]; then
    echo "PersonaPlex already cloned, pulling latest..."
    cd personaplex && git pull && cd ..
else
    git clone https://github.com/NVIDIA/personaplex.git
fi

# 3. Install PersonaPlex
echo ""
echo "[3/5] Installing PersonaPlex (Moshi)..."
source .venv/bin/activate
pip install personaplex/moshi/.

# 4. Set up HuggingFace token
echo ""
echo "[4/5] HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ HF_TOKEN not set"
    echo ""
    echo "To use PersonaPlex, you need to:"
    echo "1. Accept the license at: https://huggingface.co/nvidia/personaplex-7b-v1"
    echo "2. Create a token at: https://huggingface.co/settings/tokens"
    echo "3. Run: export HF_TOKEN=your_token_here"
    echo ""
else
    echo "✅ HF_TOKEN is set"
fi

# 5. Test installation
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
import sys
try:
    import moshi
    print('✅ Moshi/PersonaPlex installed successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "    PERSONAPLEX INSTALLATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To launch the voice server:"
echo "  export HF_TOKEN=your_token"
echo '  SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"'
echo ""
echo "Then open: https://localhost:8998"
echo ""
echo "Available voices:"
echo "  Natural (female): NATF0, NATF1, NATF2, NATF3"
echo "  Natural (male):   NATM0, NATM1, NATM2, NATM3"
echo "  Variety:          VARF0-4, VARM0-4"
echo ""
