#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# BIZRA NODE0 — Unified Installer v0.1
# بسم الله الرحمن الرحيم
#
# Usage:  chmod +x install.sh && ./install.sh
# Target: Ubuntu 24.04 (fresh or existing NODE0)
# Output: Working bizra-omega workspace + gateway + dema CLI
#
# What this does:
#   1. Verifies/installs system dependencies
#   2. Clones or updates both repos
#   3. Builds the Rust workspace (release mode)
#   4. Builds the frontend (Next.js)
#   5. Installs systemd services (gateway + next dev)
#   6. Runs the test suite to verify installation
#   7. Prints the operator's first commands
#
# What this does NOT do:
#   - Touch your data (~/Downloads, /data/bizra, etc.)
#   - Install cloud services or phone-home telemetry
#   - Require root for anything except systemd unit installation
#   - Modify your shell profile without asking
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
BIZRA_HOME="${BIZRA_HOME:-/data/bizra}"
REPO_DL="${BIZRA_HOME}/repos/bizra-data-lake"
REPO_AWD="${BIZRA_HOME}/repos/award-winner-design"
GH_ORG="BizraInfo"
RUST_MIN="1.80.0"
NODE_MIN="20.0.0"
GATEWAY_PORT=7421
NEXT_PORT=3002

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
GOLD='\033[0;33m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[BIZRA]${RESET} $1"; }
warn() { echo -e "${GOLD}[WARN]${RESET} $1"; }
fail() { echo -e "${RED}[FAIL]${RESET} $1"; exit 1; }

# ── Phase 1: System Dependencies ─────────────────────────────────────
log "Phase 1/7: Checking system dependencies..."

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        warn "$1 not found. Installing..."
        return 1
    fi
    return 0
}

# Git
check_cmd git || sudo apt-get update && sudo apt-get install -y git

# Build essentials (needed for Rust compilation)
if ! dpkg -l build-essential &>/dev/null 2>&1; then
    log "Installing build-essential, pkg-config, libssl-dev..."
    sudo apt-get update
    sudo apt-get install -y build-essential pkg-config libssl-dev
fi

# Rust
if ! check_cmd rustc; then
    log "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
RUST_VER=$(rustc --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
log "Rust: $RUST_VER"

# Node.js
if ! check_cmd node; then
    log "Installing Node.js via nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
    nvm install --lts
fi
NODE_VER=$(node --version | tr -d 'v')
log "Node: $NODE_VER"

# pnpm
if ! check_cmd pnpm; then
    log "Installing pnpm..."
    npm install -g pnpm
fi
log "pnpm: $(pnpm --version)"

# ── Phase 2: Repository Setup ────────────────────────────────────────
log "Phase 2/7: Setting up repositories..."

mkdir -p "$BIZRA_HOME/repos"

clone_or_pull() {
    local repo="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        log "Updating $repo..."
        cd "$dest" && git pull --ff-only origin main && cd -
    else
        log "Cloning $repo..."
        git clone "git@github.com:${GH_ORG}/${repo}.git" "$dest"
    fi
}

clone_or_pull "bizra-data-lake" "$REPO_DL"
clone_or_pull "award-winner-design" "$REPO_AWD"

# ── Phase 3: Rust Build ──────────────────────────────────────────────
log "Phase 3/7: Building Rust workspace (release)..."

cd "$REPO_DL/bizra-omega"

# Build the full workspace
cargo build --release 2>&1 | tail -3

# Verify key binaries exist
[ -f target/release/bizra-cognition-gateway ] || fail "Gateway binary not found"
[ -f target/release/dema ] || fail "Dema CLI binary not found"

log "Gateway binary: target/release/bizra-cognition-gateway ✓"
log "Dema CLI binary: target/release/dema ✓"

# ── Phase 4: Frontend Build ──────────────────────────────────────────
log "Phase 4/7: Building frontend..."

cd "$REPO_AWD"
pnpm install --frozen-lockfile 2>&1 | tail -3
pnpm build 2>&1 | tail -5

log "Frontend build complete ✓"

# ── Phase 5: Symlinks + PATH ─────────────────────────────────────────
log "Phase 5/7: Setting up operator paths..."

BINDIR="$HOME/.local/bin"
mkdir -p "$BINDIR"

# Symlink dema CLI to user PATH
ln -sf "$REPO_DL/bizra-omega/target/release/dema" "$BINDIR/dema"
ln -sf "$REPO_DL/bizra-omega/target/release/bizra-cognition-gateway" "$BINDIR/bizra-gateway"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$BINDIR:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$BINDIR:$PATH"
    log "Added $BINDIR to PATH (reload shell or source ~/.bashrc)"
fi

log "dema → $BINDIR/dema ✓"
log "bizra-gateway → $BINDIR/bizra-gateway ✓"

# ── Phase 6: Test Suite ──────────────────────────────────────────────
log "Phase 6/7: Running verification tests..."

cd "$REPO_DL/bizra-omega"
echo ""
log "Rust tests:"
RUST_RESULT=$(cargo test -p bizra-cognition -p bizra-cognition-gateway --release 2>&1 | tail -5)
echo "$RUST_RESULT"

cd "$REPO_AWD"
echo ""
log "Frontend tests:"
FRONT_RESULT=$(pnpm test 2>&1 | tail -5)
echo "$FRONT_RESULT"

# ── Phase 7: Service Setup (Optional) ────────────────────────────────
log "Phase 7/7: Systemd services (optional)..."

read -p "Install systemd services for gateway? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo tee /etc/systemd/system/bizra-gateway.service > /dev/null <<EOF
[Unit]
Description=BIZRA Cognition Gateway
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$REPO_DL/bizra-omega/target/release/bizra-cognition-gateway
Restart=on-failure
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable bizra-gateway
    log "bizra-gateway.service installed and enabled ✓"
    log "Start with: sudo systemctl start bizra-gateway"
else
    log "Skipped systemd setup. Start manually:"
    log "  bizra-gateway  (terminal 1)"
    log "  cd $REPO_AWD && pnpm dev  (terminal 2)"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo -e "${GREEN}BIZRA NODE0 — Installation Complete${RESET}"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Your first commands:"
echo ""
echo "  # Start the gateway (terminal 1)"
echo "  bizra-gateway"
echo ""
echo "  # Use Dema (terminal 2)"
echo "  dema                    # status"
echo "  dema health             # gateway liveness"
echo "  dema activate           # activate as principal"
echo "  dema chain              # view receipt chain"
echo "  dema submit \"hello\"     # submit a mission"
echo ""
echo "  # Open Dema web UI (terminal 3, optional)"
echo "  cd $REPO_AWD && pnpm dev"
echo "  # Then open http://localhost:3002/dema"
echo ""
echo "Repos:"
echo "  $REPO_DL"
echo "  $REPO_AWD"
echo ""
echo "Binaries:"
echo "  $(which dema 2>/dev/null || echo "$BINDIR/dema")"
echo "  $(which bizra-gateway 2>/dev/null || echo "$BINDIR/bizra-gateway")"
echo ""
echo -e "${GOLD}Close it. Prove it. Reveal it.${RESET}"
echo "══════════════════════════════════════════════════════════════"
