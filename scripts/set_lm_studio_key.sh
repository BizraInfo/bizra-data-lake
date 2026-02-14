#!/bin/bash
# Set LM Studio API Key
# Usage: source set_lm_studio_key.sh YOUR_API_KEY

if [ -z "$1" ]; then
    echo "Usage: source $0 YOUR_API_KEY"
    echo ""
    echo "Get your API key from LM Studio:"
    echo "  1. Open LM Studio"
    echo "  2. Go to Developer → Local Server"
    echo "  3. Copy your API key"
    echo ""
    echo "Or disable authentication in LM Studio settings."
    exit 1
fi

export LM_STUDIO_API_KEY="$1"
echo "✓ LM_STUDIO_API_KEY set"
echo ""
echo "Test with: bizra status"
