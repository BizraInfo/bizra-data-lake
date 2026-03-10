#!/usr/bin/env bash
# ============================================================
# BIZRA Alpha-100 Installer Bootstrap
# ============================================================
# Usage:
#   curl -sSL https://install.bizra.ai/alpha100 | bash
#   OR
#   ./bizra-install.sh [--provider local|anthropic|openai]
#                      [--local-backend ollama|lmstudio]
#                      [--model <id>]
#                      [--reflex-mode disabled|shadow|active]
#                      [--policy-file <path>]
#                      [--state-dir <path>]
# ============================================================

set -euo pipefail

BIZRA_VERSION="${BIZRA_VERSION:-0.1.0}"
BIZRA_REPO="https://github.com/bizra-ai/bizra-omega"
INSTALL_DIR="${HOME}/.bizra"
BIN_DIR="${INSTALL_DIR}/bin"

# ── Colors ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; GOLD='\033[0;33m'; NC='\033[0m'
info()  { echo -e "  ${GREEN}[ok]${NC}  $1"; }
warn()  { echo -e "  ${GOLD}[!!]${NC}  $1"; }
fail()  { echo -e "  ${RED}[ERR]${NC} $1"; exit 1; }

# ── Detect OS / Arch ─────────────────────────────────────────
detect_platform() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "$os" in
    linux*)  OS="linux" ;;
    darwin*) OS="macos" ;;
    *)       fail "Unsupported OS: $os. Use WSL on Windows." ;;
  esac

  case "$arch" in
    x86_64|amd64)  ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *)             fail "Unsupported architecture: $arch" ;;
  esac

  TARGET="${OS}-${ARCH}"
  info "Platform: ${TARGET}"
}

# ── Check prerequisites ─────────────────────────────────────
check_prereqs() {
  command -v curl >/dev/null 2>&1 || fail "curl is required but not installed."

  if command -v node >/dev/null 2>&1; then
    local node_ver
    node_ver="$(node --version 2>/dev/null | sed 's/^v//')"
    info "Node.js: v${node_ver}"
  else
    warn "Node.js not found. Required for LLM bridge."
  fi
}

# ── Download binary ──────────────────────────────────────────
download_binary() {
  local artifact="bizra-install-${BIZRA_VERSION}-${TARGET}"
  local url="${BIZRA_REPO}/releases/download/v${BIZRA_VERSION}/${artifact}.tar.gz"
  local checksum_url="${BIZRA_REPO}/releases/download/v${BIZRA_VERSION}/SHA256SUMS"

  mkdir -p "${BIN_DIR}"

  echo ""
  echo "  Downloading BIZRA installer..."

  if curl -sSL --fail -o "/tmp/${artifact}.tar.gz" "${url}" 2>/dev/null; then
    # Verify checksum if available
    if curl -sSL --fail -o "/tmp/SHA256SUMS" "${checksum_url}" 2>/dev/null; then
      local expected
      expected="$(grep "${artifact}.tar.gz" /tmp/SHA256SUMS | awk '{print $1}')"
      if [ -n "$expected" ]; then
        local actual
        actual="$(sha256sum "/tmp/${artifact}.tar.gz" | awk '{print $1}')"
        if [ "$expected" != "$actual" ]; then
          fail "Checksum mismatch! Expected: ${expected}, Got: ${actual}"
        fi
        info "Checksum verified"
      fi
    fi

    tar -xzf "/tmp/${artifact}.tar.gz" -C "${BIN_DIR}"
    chmod +x "${BIN_DIR}/bizra-install"
    info "Installed: ${BIN_DIR}/bizra-install"
    INSTALLER="${BIN_DIR}/bizra-install"
    return 0
  fi

  warn "Release artifact not available. Trying source build..."
  return 1
}

# ── Source build fallback ────────────────────────────────────
source_build() {
  if ! command -v cargo >/dev/null 2>&1; then
    fail "No prebuilt binary available and Rust toolchain not found.\n  Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  fi

  local rust_ver
  rust_ver="$(rustc --version | awk '{print $2}')"
  info "Rust toolchain: ${rust_ver}"

  local workspace_dir
  # Try to find the workspace
  if [ -d "./bizra-omega" ]; then
    workspace_dir="./bizra-omega"
  elif [ -d "../bizra-omega" ]; then
    workspace_dir="../bizra-omega"
  else
    fail "Cannot find bizra-omega workspace for source build.\n  Clone it: git clone ${BIZRA_REPO}"
  fi

  echo ""
  echo "  Building from source (this may take a few minutes)..."
  (cd "${workspace_dir}" && cargo build --release -p bizra-installer -p bizra-node) || \
    fail "Source build failed. Check Rust errors above."

  mkdir -p "${BIN_DIR}"
  cp "${workspace_dir}/target/release/bizra-install" "${BIN_DIR}/" 2>/dev/null || true
  cp "${workspace_dir}/target/release/bizra-node" "${BIN_DIR}/" 2>/dev/null || true
  chmod +x "${BIN_DIR}/bizra-install" "${BIN_DIR}/bizra-node" 2>/dev/null || true

  INSTALLER="${BIN_DIR}/bizra-install"
  info "Built from source: ${INSTALLER}"
}

# ── Run installer ────────────────────────────────────────────
run_installer() {
  if [ ! -x "${INSTALLER}" ]; then
    fail "Installer binary not found at ${INSTALLER}"
  fi

  echo ""
  echo "  Running: bizra-install alpha100 install $*"
  echo ""
  "${INSTALLER}" alpha100 install "$@"
}

# ── Add to PATH ──────────────────────────────────────────────
update_path() {
  if ! echo "$PATH" | grep -q "${BIN_DIR}"; then
    # Add to shell profile
    local profile=""
    if [ -f "${HOME}/.bashrc" ]; then profile="${HOME}/.bashrc";
    elif [ -f "${HOME}/.zshrc" ]; then profile="${HOME}/.zshrc";
    elif [ -f "${HOME}/.profile" ]; then profile="${HOME}/.profile";
    fi

    if [ -n "$profile" ]; then
      if ! grep -q "bizra/bin" "$profile" 2>/dev/null; then
        echo "" >> "$profile"
        echo "# BIZRA" >> "$profile"
        echo 'export PATH="${HOME}/.bizra/bin:${PATH}"' >> "$profile"
        info "Added ${BIN_DIR} to PATH in $(basename "$profile")"
      fi
    fi
    export PATH="${BIN_DIR}:${PATH}"
  fi
}

# ── Main ─────────────────────────────────────────────────────
main() {
  echo ""
  echo "  BIZRA Alpha-100 Installer"
  echo "  ========================="
  echo ""

  detect_platform
  check_prereqs

  INSTALLER=""
  download_binary || source_build

  update_path
  run_installer "$@"

  echo ""
  echo "  ========================="
  echo "  Installation complete."
  echo ""
  echo "  Next steps:"
  echo "    1. bizra-install alpha100 doctor    # Verify installation"
  echo "    2. bizra-install alpha100 launch    # Start Node0 in shadow mode"
  echo "    3. node ~/.bizra/alpha100/llm_bridge.js  # Connect LLM bridge"
  echo ""
  echo "  Your knowledge stays on YOUR device."
  echo ""
}

main "$@"
