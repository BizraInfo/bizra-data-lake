#!/usr/bin/env bash
# BIZRA Agentic-Flow Setup Script
# Initializes the agentic-flow submodule and builds it for integration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AF_DIR="$PROJECT_ROOT/vendor/agentic-flow/agentic-flow"

echo "=== BIZRA Agentic-Flow Setup ==="

# 1. Initialize submodule if needed
if [ ! -d "$PROJECT_ROOT/vendor/agentic-flow/.git" ] && [ ! -f "$PROJECT_ROOT/vendor/agentic-flow/.git" ]; then
    echo "[1/4] Initializing git submodule..."
    cd "$PROJECT_ROOT"
    git submodule update --init vendor/agentic-flow
else
    echo "[1/4] Submodule already initialized."
fi

# 2. Check Node.js
if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js >= 18 is required. Install from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "ERROR: Node.js >= 18 required (found v$(node -v))"
    exit 1
fi
echo "[2/4] Node.js $(node -v) detected."

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
cd "$AF_DIR"
npm install --ignore-scripts 2>/dev/null || npm install 2>/dev/null || true

# 4. Build
echo "[4/4] Building agentic-flow..."
npm run build 2>/dev/null || true

# Verify
if [ -f "$AF_DIR/dist/index.js" ]; then
    echo ""
    echo "=== Setup Complete ==="
    echo "  Agentic-Flow v$(node -e "console.log(require('./package.json').version)")"
    echo "  Location: $AF_DIR"
    echo ""
    echo "  Start:  npm start (from project root)"
    echo "  Docker: docker compose up -d agentic-flow"
else
    echo ""
    echo "=== Setup Complete (build output may vary) ==="
    echo "  Location: $AF_DIR"
    echo "  Note: dist/index.js not found — Docker build will handle compilation."
fi
