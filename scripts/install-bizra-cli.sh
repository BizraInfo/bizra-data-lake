#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BIZRA CLI Installer
# Makes 'bizra' available from any terminal
#
# Usage: bash install-bizra-cli.sh
#
# What it does:
#   1. Creates ~/.bizra/ directory structure
#   2. Copies bizra_cli.py to ~/.bizra/
#   3. Creates a 'bizra' symlink/script in PATH
#   4. Sets BIZRA_ROOT and BIZRA_FRONTEND env vars
#
# After install, just type: bizra
# ═══════════════════════════════════════════════════════════════

set -e

BIZRA_HOME="$HOME/.bizra"
BIZRA_BIN="$BIZRA_HOME/bin"
CLI_SOURCE="$(cd "$(dirname "$0")" && pwd)/bizra-cli.py"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   BIZRA CLI Installer                            ║"
echo "║   Sovereign Mission Operating System             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Step 1: Create directories
echo "→ Creating BIZRA directories..."
mkdir -p "$BIZRA_HOME"/{bin,sovereign_state,logs,models}

# Step 2: Copy CLI
if [ -f "$CLI_SOURCE" ]; then
    cp "$CLI_SOURCE" "$BIZRA_HOME/bin/bizra_cli.py"
    echo "  ✓ CLI installed to $BIZRA_HOME/bin/bizra_cli.py"
else
    echo "  ✗ Cannot find bizra-cli.py. Run this script from the same directory."
    exit 1
fi

# Step 3: Create launcher script
cat > "$BIZRA_BIN/bizra" << 'LAUNCHER'
#!/bin/bash
# BIZRA CLI launcher — finds the best Python and runs the CLI
BIZRA_CLI_DIR="$(dirname "$(readlink -f "$0")")"

# Find Python
if [ -f "$BIZRA_ROOT/.venv-linux/bin/python" ]; then
    PY="$BIZRA_ROOT/.venv-linux/bin/python"
elif [ -f "$BIZRA_ROOT/.venv/bin/python" ]; then
    PY="$BIZRA_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi

exec "$PY" "$BIZRA_CLI_DIR/bizra_cli.py" "$@"
LAUNCHER

chmod +x "$BIZRA_BIN/bizra"
echo "  ✓ Launcher created at $BIZRA_BIN/bizra"

# Step 4: Auto-detect BIZRA_ROOT
DETECTED_ROOT=""
for candidate in \
    "/mnt/c/BIZRA-DATA-LAKE" \
    "$HOME/BIZRA-DATA-LAKE" \
    "$HOME/bizra" \
    "$HOME/BIZRA" \
    "$(pwd)"; do
    if [ -f "$candidate/core/sovereign/api.py" ]; then
        DETECTED_ROOT="$candidate"
        break
    fi
done

DETECTED_FRONTEND=""
for candidate in \
    "/mnt/c/award-winner-design" \
    "$HOME/award-winner-design" \
    "$HOME/bizra-frontend"; do
    if [ -f "$candidate/next.config.mjs" ] || [ -f "$candidate/next.config.js" ]; then
        DETECTED_FRONTEND="$candidate"
        break
    fi
done

# Step 5: Add to PATH and set env vars
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    # Remove old BIZRA entries
    sed -i '/# BIZRA CLI/d' "$SHELL_RC" 2>/dev/null || true
    sed -i '/BIZRA_HOME/d' "$SHELL_RC" 2>/dev/null || true
    sed -i '/BIZRA_ROOT/d' "$SHELL_RC" 2>/dev/null || true
    sed -i '/BIZRA_FRONTEND/d' "$SHELL_RC" 2>/dev/null || true
    sed -i '/\.bizra\/bin/d' "$SHELL_RC" 2>/dev/null || true

    # Add new entries
    echo "" >> "$SHELL_RC"
    echo "# BIZRA CLI" >> "$SHELL_RC"
    echo "export BIZRA_HOME=\"$BIZRA_HOME\"" >> "$SHELL_RC"
    if [ -n "$DETECTED_ROOT" ]; then
        echo "export BIZRA_ROOT=\"$DETECTED_ROOT\"" >> "$SHELL_RC"
    fi
    if [ -n "$DETECTED_FRONTEND" ]; then
        echo "export BIZRA_FRONTEND=\"$DETECTED_FRONTEND\"" >> "$SHELL_RC"
    fi
    echo 'export PATH="$BIZRA_HOME/bin:$PATH"' >> "$SHELL_RC"

    echo "  ✓ Added to $SHELL_RC"
fi

# Also export for current session
export PATH="$BIZRA_BIN:$PATH"
export BIZRA_HOME="$BIZRA_HOME"
[ -n "$DETECTED_ROOT" ] && export BIZRA_ROOT="$DETECTED_ROOT"
[ -n "$DETECTED_FRONTEND" ] && export BIZRA_FRONTEND="$DETECTED_FRONTEND"

echo ""
echo "─────────────────────────────────────────────────"
echo ""
echo "  ✓ BIZRA CLI installed!"
echo ""
if [ -n "$DETECTED_ROOT" ]; then
    echo "  BIZRA_ROOT:     $DETECTED_ROOT"
fi
if [ -n "$DETECTED_FRONTEND" ]; then
    echo "  BIZRA_FRONTEND: $DETECTED_FRONTEND"
fi
echo "  BIZRA_HOME:     $BIZRA_HOME"
echo ""
echo "  Commands:"
echo "    bizra              Launch everything"
echo "    bizra status       Check node health"
echo "    bizra mission \"..\" Submit a mission"
echo "    bizra briefing     Morning briefing from DEMA"
echo "    bizra wallet       Check SEED/BLOOM balance"
echo "    bizra doctor       Diagnose issues"
echo ""
echo "  Open a new terminal, then type: bizra"
echo ""
echo "  \"One mission, one proof, remembered forever.\""
echo ""
