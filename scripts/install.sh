#!/bin/bash
# ═══════════════════════════════════════════════════════════
# BIZRA Node Installer — From zero to sovereign in one command
# ═══════════════════════════════════════════════════════════
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/BizraInfo/bizra-data-lake/main/scripts/install.sh | bash
#
# Or with options:
#   curl -fsSL ... | bash -s -- --seed 192.168.1.100:9750 --name "MyNode"
#
# What it does:
#   1. Installs Ollama (if not present)
#   2. Pulls qwen2.5:3b model
#   3. Clones BIZRA repo
#   4. Builds bizra-node binary
#   5. Configures seed node (NODE0 bootstrap)
#   6. Runs first boot (mints 7 PAT + 5 SAT)
#
# بذرة واحدة تصنع غابة — One seed makes a forest.
# ═══════════════════════════════════════════════════════════

set -euo pipefail

G="\033[38;2;125;211;192m"
D="\033[38;2;201;169;98m"
R="\033[0m"
B="\033[1m"
DIM="\033[2m"
RD="\033[38;2;239;68;68m"

# Parse arguments
SEED_NODE=""
NODE_NAME=""
BIZRA_HOME="${BIZRA_HOME:-$HOME/bizra}"
SKIP_OLLAMA=false
MODEL="${BIZRA_MODEL:-gemma4:e4b}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)      SEED_NODE="$2"; shift 2 ;;
        --name)      NODE_NAME="$2"; shift 2 ;;
        --home)      BIZRA_HOME="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --skip-ollama) SKIP_OLLAMA=true; shift ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${G}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║           ${B}بِذْرَة${R}${G}  —  BIZRA NODE INSTALLER             ║"
echo "  ║                                                       ║"
echo "  ║   ${DIM}Every human is a node. Every node is a seed.${R}${G}       ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${R}"

# ── Step 1: System check ─────────────────────────────────
echo -e "  ${D}[1/6]${R} System check..."

# Check OS
OS="$(uname -s)"
case "$OS" in
    Linux)  echo -e "  ${G}●${R} Linux detected" ;;
    Darwin) echo -e "  ${G}●${R} macOS detected" ;;
    *)      echo -e "  ${RD}✗${R} Unsupported OS: $OS"; exit 1 ;;
esac

# Install system dependencies (Linux only)
if [ "$OS" = "Linux" ]; then
    NEED_PKGS=""
    for pkg in build-essential pkg-config libssl-dev libz3-dev python3-venv; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            NEED_PKGS="$NEED_PKGS $pkg"
        fi
    done
    if [ -n "$NEED_PKGS" ]; then
        echo -e "  ${DIM}○${R} Installing system deps:$NEED_PKGS"
        sudo apt-get update -qq && sudo apt-get install -y -qq $NEED_PKGS
        echo -e "  ${G}●${R} System deps installed"
    else
        echo -e "  ${G}●${R} System deps present"
    fi
fi

# Check Rust
if command -v cargo &>/dev/null; then
    echo -e "  ${G}●${R} Rust: $(rustc --version 2>/dev/null | head -1)"
else
    echo -e "  ${DIM}○${R} Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "  ${G}●${R} Rust installed"
fi

# Check Python 3.11+
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "  ${G}●${R} Python: $PY_VER"
else
    echo -e "  ${RD}✗${R} Python 3 not found. Install python3 first."
    exit 1
fi

# Check git
if ! command -v git &>/dev/null; then
    echo -e "  ${RD}✗${R} git not found. Install git first."
    exit 1
fi

# ── Step 2: Ollama ─────────────────────────────────────
echo -e "  ${D}[2/6]${R} Ollama setup..."

if [ "$SKIP_OLLAMA" = true ]; then
    echo -e "  ${DIM}○${R} Skipped (--skip-ollama)"
else
    # Always install/upgrade to latest (Gemma 4 requires Ollama >= 0.20)
    echo -e "  ${DIM}○${R} Installing/upgrading Ollama (latest)..."
    if [ "$OS" = "Linux" ] && ! command -v zstd &>/dev/null; then
        sudo apt-get install -y -qq zstd 2>/dev/null
    fi
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "  ${G}●${R} Ollama: $(ollama --version 2>&1 | head -1)"
fi

# Pull Gemma 4 fleet
if ! [ "$SKIP_OLLAMA" = true ]; then
    for m in "gemma4:e4b" "gemma4:e2b" "gemma3:1b" "nomic-embed-text:latest"; do
        if ollama list 2>/dev/null | grep -q "${m%%:*}"; then
            echo -e "  ${G}●${R} $m ready"
        else
            echo -e "  ${DIM}○${R} Pulling $m..."
            ollama pull "$m"
            echo -e "  ${G}●${R} $m pulled"
        fi
    done
fi

# ── Step 3: Clone repo ─────────────────────────────────
echo -e "  ${D}[3/6]${R} Clone BIZRA..."

mkdir -p "$BIZRA_HOME"
if [ -d "$BIZRA_HOME/bizra-data-lake" ]; then
    echo -e "  ${G}●${R} Repo exists, pulling latest..."
    cd "$BIZRA_HOME/bizra-data-lake"
    git pull --ff-only 2>/dev/null || echo -e "  ${DIM}  Pull skipped${R}"
else
    cd "$BIZRA_HOME"
    git clone --depth 1 https://github.com/BizraInfo/bizra-data-lake.git
    echo -e "  ${G}●${R} Cloned"
fi

cd "$BIZRA_HOME/bizra-data-lake"

# ── Step 4: Build ──────────────────────────────────────
echo -e "  ${D}[4/6]${R} Building bizra-node (first time takes 2-5 min)..."

cd bizra-omega
cargo build --release -p bizra-node 2>&1 | tail -3
cd ..

BINARY="bizra-omega/target/release/bizra-node"
if [ ! -f "$BINARY" ]; then
    echo -e "  ${RD}✗${R} Build failed"
    exit 1
fi
echo -e "  ${G}●${R} Binary: $(ls -lh "$BINARY" | awk '{print $5}')"

# Python environment
echo -e "  ${DIM}○${R} Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -e ".[dev]" 2>&1 | tail -1
echo -e "  ${G}●${R} Python venv ready"

# ── Step 5: Configure ─────────────────────────────────
echo -e "  ${D}[5/6]${R} Configure node..."

STATE_DIR="$HOME/.bizra/node-$(date +%s | tail -c 5)"
mkdir -p "$STATE_DIR"

# Set node name
if [ -z "$NODE_NAME" ]; then
    NODE_NAME="node-$(whoami)"
fi

# Write config
cat > "$STATE_DIR/node.json" << NODEEOF
{
    "name": "$NODE_NAME",
    "seed_node": "$SEED_NODE",
    "model": "$MODEL",
    "created_at": $(date +%s)000,
    "gossip_port": 9750
}
NODEEOF

echo -e "  ${G}●${R} Name: $NODE_NAME"
echo -e "  ${G}●${R} State: $STATE_DIR"
[ -n "$SEED_NODE" ] && echo -e "  ${G}●${R} Seed: $SEED_NODE" || echo -e "  ${DIM}○${R} Standalone (no seed node)"

# ── Step 6: First boot ────────────────────────────────
echo -e "  ${D}[6/6]${R} First boot — minting your agents..."

# Teach identity
printf 'TEACH\tfact\t%s\t9500\t%d\nSHUTDOWN\n' "My name is $(whoami), I am a BIZRA node" "$(date +%s)" | \
    "$BINARY" --user 1 --ihsan 9500 --state-dir "$STATE_DIR" --no-banner 2>/dev/null

echo -e "  ${G}●${R} 7 PAT agents minted (your sovereign team)"
echo -e "  ${G}●${R} 5 SAT agents minted (system validators)"

# ── Done ──────────────────────────────────────────────
echo ""
echo -e "  ${G}╔═══════════════════════════════════════════════════════╗${R}"
echo -e "  ${G}║${R}  ${B}INSTALLATION COMPLETE${R}                                ${G}║${R}"
echo -e "  ${G}╠═══════════════════════════════════════════════════════╣${R}"
echo -e "  ${G}║${R}  ${DIM}Verify:${R} ./scripts/verify-install.sh                 ${G}║${R}"
echo -e "  ${G}║${R}  ${DIM}Run:${R}    ./scripts/bizra                              ${G}║${R}"
echo -e "  ${G}║${R}                                                        ${G}║${R}"
echo -e "  ${G}║${R}  ${DIM}Your 7 agents are waiting.${R}                          ${G}║${R}"
echo -e "  ${G}║${R}  ${DIM}بذرة واحدة تصنع غابة${R}                               ${G}║${R}"
echo -e "  ${G}╚═══════════════════════════════════════════════════════╝${R}"
echo ""
