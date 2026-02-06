# Installation Guide

Complete installation and setup guide for BIZRA CLI.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Post-Installation Setup](#post-installation-setup)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 32+ GB |
| Storage | 10 GB | 100+ GB SSD |
| GPU | Optional | NVIDIA RTX (for local LLM) |

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| Rust | 1.75+ | Build the CLI |
| Git | 2.0+ | Version control |
| Python | 3.10+ | Voice interface (optional) |
| Node.js | 18+ | MCP servers (optional) |

### Operating System

- **Windows 10/11**: Native or WSL2
- **Linux**: Ubuntu 22.04+, Fedora 38+
- **macOS**: 13+ (Ventura)

---

## Installation Methods

### Method 1: Build from Source (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/BIZRA-Sovereign/bizra-omega.git
cd bizra-omega

# 2. Build release binary
cargo build -p bizra-cli --release

# 3. Verify installation
./target/release/bizra --version
```

### Method 2: Pre-built Binary

```bash
# Download latest release
curl -LO https://github.com/BIZRA-Sovereign/bizra-omega/releases/latest/download/bizra-linux-x64.tar.gz

# Extract
tar -xzf bizra-linux-x64.tar.gz

# Move to PATH
sudo mv bizra /usr/local/bin/
```

### Method 3: Cargo Install

```bash
cargo install --git https://github.com/BIZRA-Sovereign/bizra-omega bizra-cli
```

---

## Post-Installation Setup

### Step 1: Initialize Configuration

```bash
# Create config directory
mkdir -p ~/.bizra/config

# Copy default configurations
cp -r bizra-omega/bizra-cli/config/* ~/.bizra/config/
```

### Step 2: Edit Your Profile

Open `~/.bizra/config/sovereign_profile.yaml` and customize:

```yaml
identity:
  name: "Your Name"
  title: "Your Title"
  location: "Your City, Country"
  timezone: "Your/Timezone"
```

### Step 3: Set Environment Variables

**Linux/macOS (~/.bashrc or ~/.zshrc):**
```bash
export BIZRA_HOME="$HOME/.bizra"
export BIZRA_OMEGA_PATH="/path/to/bizra-omega"
export PATH="$PATH:$BIZRA_OMEGA_PATH/target/release"

# LM Studio (if using)
export LMSTUDIO_HOST="192.168.56.1"
export LMSTUDIO_PORT="1234"

# HuggingFace (for voice)
export HF_TOKEN="your_token_here"
```

**Windows (PowerShell profile):**
```powershell
$env:BIZRA_HOME = "$env:USERPROFILE\.bizra"
$env:BIZRA_OMEGA_PATH = "C:\path\to\bizra-omega"
$env:PATH += ";$env:BIZRA_OMEGA_PATH\target\release"
```

### Step 4: Configure LLM Backend

Edit `~/.bizra/config/mcp_servers.yaml`:

```yaml
servers:
  bizra-inference:
    config:
      lmstudio_host: "192.168.56.1:1234"  # Your LM Studio address
      ollama_fallback: "localhost:11434"   # Fallback
```

### Step 5: Create Launch Script

**Windows (bizra.bat):**
```batch
@echo off
"%BIZRA_OMEGA_PATH%\target\release\bizra.exe" %*
```

**Linux/macOS (add to PATH or create alias):**
```bash
alias bizra='$BIZRA_OMEGA_PATH/target/release/bizra'
```

---

## Verification

### Basic Verification

```bash
# Check version
bizra --version
# Expected: bizra-cli 1.0.0

# Check status
bizra status
# Should display node status with FATE gates

# List agents
bizra agent list
# Should show 7 PAT agents
```

### Full Verification

```bash
# Test TUI
bizra
# Press 'q' to quit

# Test command
bizra info
# Should show system info with Arabic text

# Test help
bizra --help
# Should show all commands
```

### Verification Checklist

- [ ] `bizra --version` shows version
- [ ] `bizra status` shows node status
- [ ] `bizra agent list` shows 7 agents
- [ ] `bizra info` displays correctly (including Arabic)
- [ ] TUI launches and renders properly
- [ ] Configuration files are loaded

---

## Troubleshooting

### Build Errors

**Error: "rustc version too old"**
```bash
rustup update stable
rustup default stable
```

**Error: "missing linker"**
```bash
# Ubuntu/Debian
sudo apt install build-essential

# Fedora
sudo dnf install gcc

# macOS
xcode-select --install
```

### Runtime Errors

**Error: "configuration not found"**
```bash
# Ensure config directory exists
mkdir -p ~/.bizra/config
cp -r bizra-cli/config/* ~/.bizra/config/
```

**Error: "LM Studio connection failed"**
- Verify LM Studio is running
- Check host/port in config
- Test: `curl http://192.168.56.1:1234/v1/models`

### Display Issues

**Arabic text not rendering:**
- Install Arabic fonts
- Use a Unicode-capable terminal
- Windows Terminal or iTerm2 recommended

**TUI looks broken:**
- Ensure terminal is 80x24 minimum
- Try different terminal emulator
- Check TERM environment variable

### Permission Issues

**Linux "permission denied":**
```bash
chmod +x ./target/release/bizra
```

**Windows "not recognized":**
- Add to PATH
- Use full path to executable

---

## Next Steps

Once installed:

1. [First Run Guide](02-FIRST-RUN.md) - Initial experience
2. [Personalization](03-PERSONALIZATION.md) - Customize your profile
3. [PAT Agents](04-PAT-AGENTS.md) - Meet your team

---

## Getting Help

- **Documentation**: This guide
- **Issues**: GitHub Issues
- **Community**: Discord (coming soon)

---

**Installation complete!** Run `bizra` to start your journey.
