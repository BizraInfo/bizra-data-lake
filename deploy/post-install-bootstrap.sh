#!/usr/bin/env bash
# BIZRA Node0 — Post-Install Bootstrap for Native Ubuntu 24.04
# Run this AFTER fresh Ubuntu install + first login
# Usage: bash post-install-bootstrap.sh
set -euo pipefail

echo "=== BIZRA Node0 Bootstrap ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo ""

# ── 1. System packages ────────────────────────────────────────────
echo "[1/8] Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq \
  build-essential git curl wget unzip cmake ninja-build pkg-config \
  libssl-dev libffi-dev libz3-dev libopenblas-dev libomp-dev \
  python3.12 python3.12-venv python3.12-dev python3-pip \
  htop tmux ripgrep fd-find jq tree \
  openssh-server net-tools ca-certificates gnupg lsb-release

# ── 2. Rust ───────────────────────────────────────────────────────
echo "[2/8] Installing Rust..."
if ! command -v rustup &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
rustup update stable
echo "Rust: $(rustc --version)"

# ── 3. Docker ─────────────────────────────────────────────────────
echo "[3/8] Installing Docker..."
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "Docker installed. You may need to log out and back in for group to take effect."
fi

# ── 4. Node.js ────────────────────────────────────────────────────
echo "[4/8] Installing Node.js 22..."
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt install -y -qq nodejs
  sudo npm install -g pnpm
fi
echo "Node: $(node --version)"

# ── 5. Clone repos ────────────────────────────────────────────────
echo "[5/8] Cloning BIZRA repos..."
mkdir -p "$HOME/bizra"
cd "$HOME/bizra"

[ -d bizra-data-lake ] || git clone https://github.com/BizraInfo/bizra-data-lake.git
[ -d BIZRA-OS ] || git clone https://github.com/BizraInfo/BIZRA-OS.git
[ -d bizra-node0-genesis ] || git clone https://github.com/BizraInfo/bizra-node0-genesis.git

echo "HEAD: $(cd bizra-data-lake && git log --oneline -1)"

# ── 6. Python venv ────────────────────────────────────────────────
echo "[6/8] Setting up Python environment..."
cd "$HOME/bizra/bizra-data-lake"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q 2>/dev/null || pip install -e . -q
echo "Python: $(python --version)"

# ── 7. Rust workspace ────────────────────────────────────────────
echo "[7/8] Building Rust workspace..."
cd "$HOME/bizra/bizra-data-lake/bizra-omega"
cargo build --workspace --release 2>&1 | tail -3
echo "Rust build complete."

# ── 8. Restore secrets prompt ────────────────────────────────────
echo ""
echo "[8/8] MANUAL: Restore secrets from USB backup"
echo ""
echo "Mount your D: USB drive, then run:"
echo "  sudo mount /dev/sdX1 /mnt/usb"
echo "  cp /mnt/usb/BIZRA-SECRETS-BACKUP/.env ~/bizra/bizra-data-lake/"
echo "  cp /mnt/usb/BIZRA-SECRETS-BACKUP/.bizra_api_token ~/"
echo "  cp /mnt/usb/BIZRA-SECRETS-BACKUP/.gitconfig ~/"
echo "  cp /mnt/usb/BIZRA-SECRETS-BACKUP/.bashrc ~/"
echo ""
echo "If 04_GOLD was copied to USB:"
echo "  cp -r /mnt/usb/04_GOLD ~/bizra/bizra-data-lake/04_GOLD"
echo ""
echo "Otherwise regenerate: python corpus_manager.py && python vector_engine.py"
echo ""
echo "=== BIZRA Node0 Bootstrap Complete ==="
echo "Reboot, then run: cd ~/bizra/bizra-data-lake && source .venv/bin/activate && pytest tests/ -x -q"
