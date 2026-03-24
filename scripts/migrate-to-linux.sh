#!/bin/bash
# ============================================================================
# BIZRA Node0 — Native Linux Bootstrap
# ============================================================================
# Run after installing Ubuntu 24.04 LTS dual boot.
# This script installs ALL dependencies and sets up the BIZRA development
# environment on native Linux with CUDA GPU acceleration.
#
# Usage:
#   sudo bash scripts/migrate-to-linux.sh
#
# Prerequisites:
#   - Ubuntu 24.04 LTS installed
#   - Internet connection
#   - RTX 4090 detected by system
#
# Standing on Giants: Deming (PDCA — verify each step before next)
# ============================================================================

set -euo pipefail

# Colors
G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
B='\033[1;34m'
NC='\033[0m'

log()  { echo -e "${G}[BIZRA]${NC} $1"; }
warn() { echo -e "${Y}[WARN]${NC} $1"; }
err()  { echo -e "${R}[ERROR]${NC} $1"; }
step() { echo -e "\n${B}═══════════════════════════════════════${NC}"; echo -e "${B}  Step $1${NC}"; echo -e "${B}═══════════════════════════════════════${NC}"; }

# Must run as root for system packages
if [ "$EUID" -ne 0 ]; then
    err "Run with sudo: sudo bash $0"
    exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(eval echo "~${REAL_USER}")

log "BIZRA Node0 Native Linux Bootstrap"
log "User: ${REAL_USER} | Home: ${REAL_HOME}"
log "Starting at $(date)"

# ══════════════════════════════════════════════════════════════════
step "1/10: System Packages"
# ══════════════════════════════════════════════════════════════════

apt update
apt install -y \
    build-essential git curl wget unzip \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    libz3-dev pkg-config libssl-dev libffi-dev \
    cmake ninja-build \
    htop tmux ripgrep fd-find jq \
    libopenblas-dev libomp-dev

log "System packages installed"

# ══════════════════════════════════════════════════════════════════
step "2/10: NVIDIA Drivers + CUDA"
# ══════════════════════════════════════════════════════════════════

if ! command -v nvidia-smi &>/dev/null; then
    log "Installing NVIDIA drivers..."
    apt install -y nvidia-driver-555 nvidia-cuda-toolkit nvidia-cuda-dev
    warn "REBOOT REQUIRED after driver install. Re-run this script after reboot."
    warn "Run: sudo reboot"
    exit 0
else
    log "NVIDIA driver already installed"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# ══════════════════════════════════════════════════════════════════
step "3/10: Rust Toolchain"
# ══════════════════════════════════════════════════════════════════

if ! sudo -u "${REAL_USER}" bash -c 'command -v rustc &>/dev/null'; then
    log "Installing Rust..."
    sudo -u "${REAL_USER}" bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
    log "Rust installed"
else
    RUST_VER=$(sudo -u "${REAL_USER}" bash -c 'source ~/.cargo/env && rustc --version')
    log "Rust already installed: ${RUST_VER}"
fi

# ══════════════════════════════════════════════════════════════════
step "4/10: Ollama (Native CUDA)"
# ══════════════════════════════════════════════════════════════════

if ! command -v ollama &>/dev/null; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama service
systemctl enable ollama
systemctl start ollama
sleep 3

# Pull models
log "Pulling Ollama models (this may take a while)..."
MODELS=(
    "qwen2.5:3b"
    "llama3.1:8b"
    "deepseek-r1:14b"
    "phi3:mini"
    "mistral:latest"
    "nomic-embed-text:latest"
    "moondream:1.8b"
)

for model in "${MODELS[@]}"; do
    log "Pulling ${model}..."
    sudo -u "${REAL_USER}" ollama pull "${model}" || warn "Failed to pull ${model}"
done

log "Ollama ready: $(ollama list | wc -l) models"

# ══════════════════════════════════════════════════════════════════
step "5/10: Docker"
# ══════════════════════════════════════════════════════════════════

if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker "${REAL_USER}"
    log "Docker installed (re-login for group to take effect)"
else
    log "Docker already installed: $(docker --version)"
fi

# Install NVIDIA Container Toolkit for GPU containers
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    log "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt update
    apt install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log "NVIDIA Container Toolkit installed"
fi

# ══════════════════════════════════════════════════════════════════
step "6/10: Node.js (via nvm)"
# ══════════════════════════════════════════════════════════════════

if ! sudo -u "${REAL_USER}" bash -c 'command -v node &>/dev/null'; then
    log "Installing nvm + Node.js 22..."
    sudo -u "${REAL_USER}" bash -c '
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        nvm install 22
        npm install -g pnpm
    '
    log "Node.js installed"
else
    NODE_VER=$(sudo -u "${REAL_USER}" bash -c 'node --version')
    log "Node.js already installed: ${NODE_VER}"
fi

# ══════════════════════════════════════════════════════════════════
step "7/10: uv (Python Package Manager)"
# ══════════════════════════════════════════════════════════════════

if ! sudo -u "${REAL_USER}" bash -c 'command -v uv &>/dev/null'; then
    log "Installing uv..."
    sudo -u "${REAL_USER}" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    log "uv installed"
else
    log "uv already installed"
fi

# ══════════════════════════════════════════════════════════════════
step "8/10: Clone BIZRA + Python Environment"
# ══════════════════════════════════════════════════════════════════

BIZRA_DIR="${REAL_HOME}/bizra-data-lake"

if [ ! -d "${BIZRA_DIR}" ]; then
    log "Cloning BIZRA..."
    sudo -u "${REAL_USER}" git clone https://github.com/BizraInfo/bizra-data-lake.git "${BIZRA_DIR}"
else
    log "BIZRA already cloned at ${BIZRA_DIR}"
    sudo -u "${REAL_USER}" bash -c "cd ${BIZRA_DIR} && git pull origin main"
fi

log "Setting up Python virtual environment..."
sudo -u "${REAL_USER}" bash -c "
    cd ${BIZRA_DIR}
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e '.[full]'
"
log "Python environment ready"

# ══════════════════════════════════════════════════════════════════
step "9/10: Build Rust Workspace"
# ══════════════════════════════════════════════════════════════════

log "Building bizra-omega (release mode)..."
sudo -u "${REAL_USER}" bash -c "
    source ~/.cargo/env
    cd ${BIZRA_DIR}/bizra-omega
    RUSTFLAGS='-C target-cpu=native' cargo build --workspace --release
"
log "Rust build complete"

# ══════════════════════════════════════════════════════════════════
step "10/10: Environment Configuration"
# ══════════════════════════════════════════════════════════════════

# Set BIZRA env vars in .bashrc
BASHRC="${REAL_HOME}/.bashrc"
if ! grep -q "BIZRA_DATA_LAKE_ROOT" "${BASHRC}"; then
    log "Adding BIZRA environment to .bashrc..."
    cat >> "${BASHRC}" << 'BIZRA_ENV'

# ── BIZRA Node0 Environment ──────────────────────────────────────
export BIZRA_DATA_LAKE_ROOT="$HOME/bizra-data-lake"
export OLLAMA_HOST="http://localhost:11434"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Activate BIZRA venv on cd
bizra() {
    cd "$BIZRA_DATA_LAKE_ROOT"
    source .venv/bin/activate
    ./scripts/bizra "$@"
}
# ── End BIZRA ────────────────────────────────────────────────────
BIZRA_ENV
    log "Environment configured in .bashrc"
fi

# ══════════════════════════════════════════════════════════════════
echo ""
echo -e "${G}═══════════════════════════════════════════════════════${NC}"
echo -e "${G}  BIZRA Node0 — Native Linux Bootstrap Complete!${NC}"
echo -e "${G}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo "  1. Restore state:  tar -xzf bizra-backup-*.tar.gz"
echo "     cp -r */dot-bizra ~/.bizra"
echo "     cp */dot-env ~/bizra-data-lake/.env"
echo "     cp -r */04_GOLD ~/bizra-data-lake/"
echo ""
echo "  2. Verify:  cd ~/bizra-data-lake && source .venv/bin/activate"
echo "     nvidia-smi"
echo "     cargo test --workspace --release -C bizra-omega"
echo "     pytest tests/core/pci/ -q"
echo "     ./scripts/bizra mission 'Hello from native Linux'"
echo ""
echo "  3. Start daemon:  ./scripts/bizra"
echo ""
echo -e "  ${B}بذرة واحدة تصنع غابة — One seed makes a forest.${NC}"
echo ""
