#!/usr/bin/env bash
set -euo pipefail

# Defensive WSL/AppSec bootstrap for BIZRA Omega.
# This script intentionally installs audit, hardening, testing, and supply-chain
# tools for authorized security engineering work. It does NOT install exploit
# frameworks, credential attack suites, phishing kits, or offensive tooling.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

APT_PACKAGES=(
  ca-certificates
  curl
  git
  build-essential
  pkg-config
  clang
  lld
  llvm
  cmake
  libssl-dev
  zlib1g-dev
  libz3-dev
  jq
  shellcheck
  ripgrep
  fd-find
  hyperfine
  strace
  lsof
  python3-pip
  python3-venv
  pipx
)

CARGO_PACKAGES=(
  cargo-audit
  cargo-deny
  cargo-nextest
  cargo-llvm-cov
  cargo-fuzz
  cargo-outdated
  cargo-machete
)

usage() {
  cat <<'EOF'
Usage:
  scripts/ops/bootstrap_defensive_security_wsl.sh [--base-only] [--skip-semgrep]

What it does:
  - Installs defensive workstation packages on Ubuntu/WSL
  - Installs Rust security/quality tooling with cargo
  - Optionally installs Semgrep with pipx

What it does not do:
  - Install exploit kits or offensive attack frameworks
  - Modify shell dotfiles
  - Configure live targets or credentials
EOF
}

BASE_ONLY=0
INSTALL_SEMGREP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-only)
      BASE_ONLY=1
      shift
      ;;
    --skip-semgrep)
      INSTALL_SEMGREP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is intended for Linux/WSL environments." >&2
  exit 1
fi

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *debian* ]]; then
    echo "This bootstrap currently targets Ubuntu/Debian-like systems." >&2
    exit 1
  fi
fi

echo "== BIZRA Defensive WSL Bootstrap =="
echo "Repo root: $ROOT_DIR"
echo "WSL distro: ${WSL_DISTRO_NAME:-not-detected}"

echo
echo "== apt packages =="
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PACKAGES[@]}"

if [[ $BASE_ONLY -eq 0 ]]; then
  echo
  echo "== cargo tools =="
  for pkg in "${CARGO_PACKAGES[@]}"; do
    echo "-- installing $pkg"
    cargo install "$pkg" --locked || cargo install "$pkg"
  done

  if [[ $INSTALL_SEMGREP -eq 1 ]]; then
    echo
    echo "== semgrep =="
    export PATH="$HOME/.local/bin:$PATH"
    pipx install semgrep || pipx upgrade semgrep
  fi
fi

echo
echo "== verification =="
for tool in rustc cargo cargo-audit cargo-deny cargo-nextest cargo-llvm-cov cargo-fuzz cargo-outdated cargo-machete shellcheck rg fd-find hyperfine semgrep; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%-18s %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '%-18s %s\n' "$tool" "not-installed"
  fi
done

cat <<'EOF'

Recommended next checks:
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace
  cargo audit
  cargo llvm-cov --workspace --lcov --output-path lcov.info
  semgrep --config auto .

Repo guidance:
  docs/WSL_DEFENSIVE_SECURITY_WORKSTATION.md

EOF
