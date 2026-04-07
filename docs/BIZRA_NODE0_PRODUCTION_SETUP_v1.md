# BIZRA Node0 — Production Setup Guide

> **Standing on the Shoulders of Giants**
> _"We have built nothing from nothing. Every proof, every gate, every line of inference stands on the shoulders of Euclid, Ibn Khaldun, Shannon, Lamport, and every engineer who wrote the open-source foundations beneath us."_

---

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

---

**Document:** BIZRA_NODE0_PRODUCTION_SETUP_v1.md
**Operator:** Mohamed Beshr (MoMo) — Sole Sovereign Node
**Hardware:** MSI Titan 18 HX (i9-14900HX · RTX 4090 16GB · 128GB DDR5)
**Location:** Dubai, UAE — GMT+4
**Repository:** https://github.com/BizraInfo/bizra-data-lake
**Release Target:** v1.0.0-genesis
**CI Passing:** 1,539 Rust tests · 117 Python PCI tests · 0 failures

---

## Mission Statement

This guide takes BIZRA Node0 from zero to a fully sovereign, constitutionally-governed, locally-running AI node in one day. Every command is exact. Every status is truth-labelled. When you finish Part 8, you will have a running node with cryptographic identity, constitutional enforcement (Ihsān ≥ 0.95), federated proof-carrying inference, and a 24-hour heartbeat log sealed on your drive.

---

## Architecture at a Glance

```
Windows 11 Host
├── LM Studio (port 1234) ── serves deepseek-r1, qwen2.5-32b, llava, qwen2.5-coder
├── Docker Desktop ──────── Redis, Node Gateway, URP services
└── WSL2 (Ubuntu)
    ├── /mnt/c/BIZRA-DATA-LAKE  ← runtime codebase (git repo)
    ├── /mnt/b/BIZRA-SOVEREIGN  ← evidence vault (constitutional proofs)
    ├── Ollama ──────────────── fallback models (qwen2.5:3b, phi3:mini, etc.)
    ├── Kernel Daemon ────────── port 9740
    ├── MCP Transport ────────── port 9741 (JSON-RPC / TCP)
    ├── Desktop Bridge ───────── port 9742
    ├── Ghost WebSocket ──────── port 9743
    └── bizra (TUI) ─────────── Rust CLI, 12 widgets
```

---

## Part 1 — Prerequisites

**Status: TRUTH — Install everything before touching the codebase.**

### 1.1 Windows 11 + WSL2

Open PowerShell as Administrator:

```powershell
# Enable WSL2
wsl --install
wsl --set-default-version 2

# Install Ubuntu 24.04
wsl --install -d Ubuntu-24.04

# Verify WSL2 is active
wsl --list --verbose
# Expected:   NAME            STATE    VERSION
#             Ubuntu-24.04    Running  2
```

After Ubuntu boots, create your UNIX user. Then from inside WSL2:

```bash
# Update the package list
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
  build-essential curl git wget \
  pkg-config libssl-dev \
  python3.12 python3.12-venv python3.12-dev python3-pip \
  redis-tools jq unzip ca-certificates

# Verify Python
python3.12 --version
# Expected: Python 3.12.x
```

### 1.2 Rust Toolchain (1.91+)

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Set to stable and update
rustup default stable
rustup update stable

# Verify — must be 1.91 or higher
rustc --version
cargo --version

# Install cargo-nextest for fast parallel test runs (optional but recommended)
cargo install cargo-nextest --locked
```

**AVX-512 profile (optional — unlocks maximum throughput on your i9-14900HX):**

```bash
# Add to ~/.cargo/config.toml
mkdir -p ~/.cargo
cat >> ~/.cargo/config.toml <<'EOF'
[profile.omega]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native", "-C", "target-feature=+avx512f,+avx512bw,+avx512vl"]
EOF
```

### 1.3 Python 3.12 Virtual Environment

This will be created inside the repo in Part 2. Confirm the interpreter now:

```bash
python3.12 -m venv --version
# Expected: no error (venv module present)
```

### 1.4 Node.js (for fate-binding JS runtime)

```bash
# Install via nvm for version control
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc

nvm install 20
nvm use 20
node --version   # Expected: v20.x.x
npm --version
```

### 1.5 Docker Desktop (Windows side)

1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Install on Windows with **WSL2 backend** enabled.
3. In Docker Desktop → Settings → Resources → WSL Integration → enable **Ubuntu-24.04**.
4. Back in WSL2, verify:

```bash
docker --version
# Expected: Docker version 27.x.x, build ...

docker compose version
# Expected: Docker Compose version v2.x.x
```

### 1.6 LM Studio (Windows host)

Download LM Studio from https://lmstudio.ai and install on Windows.

Pull the required models from the LM Studio UI (Discover tab):

| Role | Model |
|------|-------|
| Reasoning | `deepseek-r1-distill-qwen-32b` |
| Agentic | `qwen2.5-32b-instruct` |
| Vision | `llava-v1.6-mistral-7b` |
| Code | `qwen2.5-coder-32b` |
| Embedding | `nomic-embed-text` |

Enable the **Local Server** in LM Studio (port 1234). Confirm from WSL2:

```bash
# Get WSL2 gateway IP (Windows host address)
WSL_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo "WSL gateway: $WSL_HOST_IP"
# Typically: 172.22.48.1

# Test LM Studio reachability
curl -s http://${WSL_HOST_IP}:1234/v1/models | jq '.data[].id'
# Expected: list of model IDs you loaded
```

### 1.7 Ollama (WSL fallback)

```bash
# Install Ollama in WSL2
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull MOE bridge fallback models
ollama pull qwen2.5:3b          # pat_k — knowledge
ollama pull qwen2.5-coder:7b    # pat_s — code
ollama pull deepseek-r1:14b     # pat_r — reasoning
ollama pull phi3:mini            # sat_g + sat_v — governance / verification
ollama pull nomic-embed-text    # embeddings fallback

# Confirm models are available
ollama list
```

---

## Part 2 — Clone and Build

**Status: TRUTH — The build is reproducible. 1,539 Rust tests must pass.**

### 2.1 Clone the Repository

```bash
# Navigate to the Windows-mounted drive (persistent across reboots)
cd /mnt/c

# Clone to BIZRA-DATA-LAKE (if not already present)
# If the directory already exists, skip clone and do git pull instead
if [ ! -d "BIZRA-DATA-LAKE/.git" ]; then
  git clone https://github.com/BizraInfo/bizra-data-lake.git BIZRA-DATA-LAKE
else
  cd BIZRA-DATA-LAKE && git pull origin main && cd /mnt/c
fi

cd BIZRA-DATA-LAKE
export BIZRA_ROOT=$(pwd)
echo "BIZRA root: $BIZRA_ROOT"

# Check commit count (should be ~762+)
git log --oneline | wc -l
```

### 2.2 Python Virtual Environment

```bash
cd $BIZRA_ROOT

# Create venv with Python 3.12
python3.12 -m venv .venv

# Activate
source .venv/bin/activate

# Confirm interpreter
python --version   # Python 3.12.x
which python       # .../BIZRA-DATA-LAKE/.venv/bin/python

# Install project in editable mode (reads pyproject.toml)
pip install --upgrade pip
pip install -e .

# Install additional production dependencies
pip install maturin cryptography fastapi uvicorn redis aiohttp
```

### 2.3 Build the Rust Workspace

```bash
cd $BIZRA_ROOT/bizra-omega

# Standard release build (all 26 crates)
cargo build --workspace --release

# --- OR --- AVX-512 omega profile (uses your i9-14900HX instruction set)
# cargo build --workspace --profile omega

# Expected: "Finished release [optimized] target(s)"
# This will take 5–15 minutes on first build; subsequent builds are incremental.

# Confirm key binaries are present
ls -lh target/release/bizra-node
ls -lh target/release/bizra
```

### 2.4 Build the Python↔Rust Bridge

```bash
cd $BIZRA_ROOT/bizra-omega/bizra-python
source ../.venv/bin/activate 2>/dev/null || source $BIZRA_ROOT/.venv/bin/activate

# Build and install the maturin wheel into venv
maturin develop --release

# Verify bridge
python -c "import bizra; print(f'bizra v{bizra.__version__}')"
# Expected: bizra v2.0.0
```

### 2.5 Docker Compose — Start Backing Services

```bash
cd $BIZRA_ROOT/.tmp_prod_artifacts_v2/deploy

# Bring up Redis + Node Gateway + URP services
docker compose up -d

# Verify containers are running
docker compose ps
# Expected (6 services):
#   redis          — redis:7-alpine     — Up (healthy)
#   node-gateway   — FastAPI            — Up    :8000->8000
#   urp-registry   — URP               — Up    :8011->8011
#   urp-kg         — URP Knowledge Graph— Up
#   urp-consensus  — URP Consensus      — Up
#   urp-verify     — URP Verification   — Up

# Quick Redis health check
docker exec -it $(docker compose ps -q redis) redis-cli ping
# Expected: PONG
```

---

## Part 3 — Environment Configuration

**Status: TRUTH — Secrets must never appear in version control.**

### 3.1 Copy the Environment Template

```bash
cd $BIZRA_ROOT

cp .env.example .env
```

### 3.2 Generate Cryptographic Secrets

```bash
# Generate BIZRA_API_KEY (32 bytes → 64 hex chars)
BIZRA_API_KEY=$(openssl rand -hex 32)
echo "BIZRA_API_KEY=$BIZRA_API_KEY"

# Generate URP_ADMIN_TOKEN
URP_ADMIN_TOKEN=$(openssl rand -hex 32)
echo "URP_ADMIN_TOKEN=$URP_ADMIN_TOKEN"

# Generate JWT secret
BIZRA_JWT_SECRET=$(openssl rand -hex 32)
echo "BIZRA_JWT_SECRET=$BIZRA_JWT_SECRET"

# Write them into .env (replace placeholder values)
sed -i "s|^BIZRA_API_KEY=.*|BIZRA_API_KEY=${BIZRA_API_KEY}|" .env
sed -i "s|^URP_ADMIN_TOKEN=.*|URP_ADMIN_TOKEN=${URP_ADMIN_TOKEN}|" .env
```

### 3.3 Edit .env — Full Production Values

Open `.env` and confirm the following are set correctly:

```bash
# Core identity
BIZRA_ENV=production
BIZRA_LOG_LEVEL=INFO

# Already generated above
BIZRA_API_KEY=<64-hex-chars>
URP_ADMIN_TOKEN=<64-hex-chars>
BIZRA_JWT_SECRET=<64-hex-chars>

# Redis (Docker internal network)
REDIS_URL=redis://redis:6379/0

# CORS — tighten for production (add your frontend origin)
BIZRA_CORS_ORIGINS=http://localhost:3000

# Rate limiting
BIZRA_RATE_LIMIT_RPM=120

# LM Studio (Windows host — use your WSL gateway IP)
LM_STUDIO_HOST=172.22.48.1
LM_STUDIO_PORT=1234
LM_STUDIO_BASE_URL=http://172.22.48.1:1234/v1

# Fallback inference
OLLAMA_BASE_URL=http://localhost:11434

# Skip auth for local dev, but NEVER in production
BIZRA_SKIP_AUTH=false
```

> **Note on WSL IP:** The gateway IP `172.22.48.1` is typical but may differ on your machine. Always derive it fresh:
> ```bash
> cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
> ```

### 3.4 Redis Security Hardening

```bash
# Edit docker-compose.yml to add password auth and bind restriction
# File: $BIZRA_ROOT/.tmp_prod_artifacts_v2/deploy/docker-compose.yml

REDIS_PASSWORD=$(openssl rand -hex 16)
echo "Redis password: $REDIS_PASSWORD"  # Save this!

# Add to .env
echo "REDIS_PASSWORD=${REDIS_PASSWORD}" >> $BIZRA_ROOT/.env

# Override Redis command in compose to enforce auth + local binding
# Add to the redis service in docker-compose.yml:
#   command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD} --bind 127.0.0.1

# Update REDIS_URL to include password
sed -i "s|^REDIS_URL=.*|REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0|" $BIZRA_ROOT/.env
```

### 3.5 Configure sovereign_profile.yaml

```bash
cd $BIZRA_ROOT/bizra-omega/bizra-cli/config

# Edit sovereign_profile.yaml with your identity
cat > sovereign_profile.yaml <<'EOF'
# BIZRA Node0 — Sovereign Profile
# Operator: Mohamed Beshr (MoMo)
# Location: Dubai, UAE (GMT+4)

node:
  id: "node0"
  operator: "Mohamed Beshr"
  alias: "MoMo"
  timezone: "Asia/Dubai"
  locale: "en-AE"

hardware:
  cpu: "i9-14900HX"
  gpu: "RTX 4090 16GB"
  ram_gb: 128
  storage:
    evidence: "B:/BIZRA-SOVEREIGN"
    runtime: "C:/BIZRA-DATA-LAKE"

models:
  primary_backend: "lm_studio"
  lm_studio_url: "http://172.22.48.1:1234/v1"
  fallback_backend: "ollama"
  ollama_url: "http://localhost:11434"

  fleet:
    reasoning:
      model: "deepseek-r1-distill-qwen-32b"
      backend: "lm_studio"
    agentic:
      model: "qwen2.5-32b-instruct"
      backend: "lm_studio"
    vision:
      model: "llava-v1.6-mistral-7b"
      backend: "lm_studio"
    code:
      model: "qwen2.5-coder-32b"
      backend: "lm_studio"
    embedding:
      model: "nomic-embed-text"
      backend: "lm_studio"

domains:
  primary: "bizra.info"
  ai: "bizra.ai"

constitutional:
  ihsan_threshold: 0.95
  strict_ihsan_threshold: 0.99
  snr_threshold_minimum: 0.85
  snr_threshold_t0_elite: 0.98
  adl_gini_threshold: 0.35
  runtime_ihsan_threshold: 1.0
EOF
```

---

## Part 4 — Genesis Ceremony

**Status: TRUTH — The genesis ceremony creates your cryptographic identity. It runs exactly once.**

### 4.1 Understand What Happens

The genesis ceremony:
1. Derives a BLAKE3 identity hash from your profile
2. Generates an Ed25519 keypair (your node's signing identity)
3. Writes a sealed genesis manifest to disk
4. Records the genesis block in the local proof ledger

### 4.2 Add the Node Binary to PATH

```bash
# Add the Rust binary to your shell PATH
echo 'export PATH="$PATH:/mnt/c/BIZRA-DATA-LAKE/bizra-omega/target/release"' >> ~/.bashrc
source ~/.bashrc

# Also link for convenience
sudo ln -sf /mnt/c/BIZRA-DATA-LAKE/bizra-omega/target/release/bizra-node /usr/local/bin/bizra-node
sudo ln -sf /mnt/c/BIZRA-DATA-LAKE/bizra-omega/target/release/bizra /usr/local/bin/bizra

# Confirm
bizra-node --version
bizra --version
```

### 4.3 Run the Genesis Ceremony

```bash
# Activate venv
source /mnt/c/BIZRA-DATA-LAKE/.venv/bin/activate

# Source environment
export $(grep -v '^#' /mnt/c/BIZRA-DATA-LAKE/.env | xargs)

# Run genesis — replace <your-blake3-hash> with your chosen operator hash
# To generate a deterministic hash from your name:
OPERATOR_HASH=$(echo -n "Mohamed Beshr:node0:bizra.info" | sha256sum | awk '{print $1}')
echo "Operator hash: $OPERATOR_HASH"

# Run Node0 genesis
bizra-node --user ${OPERATOR_HASH} --genesis --config /mnt/c/BIZRA-DATA-LAKE/bizra-omega/bizra-cli/config/sovereign_profile.yaml

# Expected output:
# ╔═══════════════════════════════════════╗
# ║     BIZRA Node0 Genesis Ceremony      ║
# ╚═══════════════════════════════════════╝
# [✓] BLAKE3 identity computed
# [✓] Ed25519 keypair generated
# [✓] Genesis seal written
# [✓] Constitutional thresholds loaded
# [✓] Node0 is ALIVE
```

### 4.4 Run the first_breath.py Synapse Proof

```bash
cd /mnt/c/BIZRA-DATA-LAKE
python first_breath.py

# Expected: PAT/SAT dual pipeline proof completes with Ihsān gate passing
```

### 4.5 Verify Genesis Seal

```bash
# Verify trust chain with the CLI
bizra trust

# Expected:
# [✓] Ed25519 signature: VALID
# [✓] BLAKE3 identity: MATCHES
# [✓] Genesis manifest: SEALED
# [✓] Trust chain: INTACT

# Also inspect the manifest directly
cat /mnt/c/BIZRA-DATA-LAKE/GENESIS_MANIFEST_VERIFIED.yaml | head -30
```

---

## Part 5 — Start the Full Stack

**Status: TRUTH — Services must start in dependency order. Use tmux or separate terminals.**

### 5.0 Install tmux (recommended for managing multiple processes)

```bash
sudo apt install -y tmux

# Start a new tmux session with named windows
tmux new-session -d -s bizra -n redis
tmux new-window -t bizra -n ollama
tmux new-window -t bizra -n kernel
tmux new-window -t bizra -n gateway
tmux new-window -t bizra -n mcp
tmux new-window -t bizra -n ghost
tmux new-window -t bizra -n cli

echo "Tmux session 'bizra' created. Attach with: tmux attach -t bizra"
```

### 5.1 Start Redis

Redis is already running via Docker Compose (Step 2.5). Confirm it's healthy:

```bash
docker compose -f /mnt/c/BIZRA-DATA-LAKE/.tmp_prod_artifacts_v2/deploy/docker-compose.yml ps redis
# Status: Up (healthy)
```

If it needs a restart:

```bash
docker compose -f /mnt/c/BIZRA-DATA-LAKE/.tmp_prod_artifacts_v2/deploy/docker-compose.yml restart redis
```

### 5.2 Start Ollama (WSL fallback models)

```bash
# In tmux window 'ollama'
tmux send-keys -t bizra:ollama "ollama serve" Enter

# Wait 3 seconds, then verify
sleep 3
curl -s http://localhost:11434/api/tags | jq '.models[].name'
# Expected: your pulled models listed
```

### 5.3 Start LM Studio (Windows side)

On the Windows host, open LM Studio and:
1. Load `deepseek-r1-distill-qwen-32b` or `qwen2.5-32b-instruct` as the active model.
2. Ensure the Local Server is running on port 1234.
3. Verify from WSL:

```bash
WSL_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
curl -s http://${WSL_HOST}:1234/v1/models | jq '.data | length'
# Expected: >= 1
```

### 5.4 Start Kernel Daemon (port 9740)

```bash
# In tmux window 'kernel'
tmux send-keys -t bizra:kernel \
  "cd /mnt/c/BIZRA-DATA-LAKE && source .venv/bin/activate && export \$(grep -v '^#' .env | xargs) && python core/sovereign/kernel_daemon.py" \
  Enter

# Wait for startup
sleep 5

# Health check
curl -s http://localhost:9740/health | jq
# Expected: {"status": "healthy", "ihsan": true, "node": "node0"}
```

### 5.5 Start Node Gateway (port 8000)

The Node Gateway is already containerised via Docker Compose. Verify:

```bash
curl -s http://localhost:8000/health | jq
# Expected: {"status": "ok", "service": "node-gateway"}

# If it failed to start, check logs:
docker compose -f /mnt/c/BIZRA-DATA-LAKE/.tmp_prod_artifacts_v2/deploy/docker-compose.yml logs node-gateway --tail 50
```

To run it directly (outside Docker, e.g. for debugging):

```bash
# In tmux window 'gateway' — only if NOT using Docker
tmux send-keys -t bizra:gateway \
  "cd /mnt/c/BIZRA-DATA-LAKE && source .venv/bin/activate && export \$(grep -v '^#' .env | xargs) && uvicorn services.node_gateway.main:app --host 0.0.0.0 --port 8000 --reload" \
  Enter
```

### 5.6 Start MCP Transport (port 9741)

```bash
# In tmux window 'mcp'
tmux send-keys -t bizra:mcp \
  "cd /mnt/c/BIZRA-DATA-LAKE && bizra-node --mcp --port 9741 --user \${OPERATOR_HASH}" \
  Enter

sleep 3

# Verify MCP port is listening
ss -tlnp | grep 9741
# Expected: LISTEN 0  ... 0.0.0.0:9741
```

### 5.7 Start Ghost WebSocket Bridge (port 9743)

```bash
# In tmux window 'ghost'
tmux send-keys -t bizra:ghost \
  "cd /mnt/c/BIZRA-DATA-LAKE && source .venv/bin/activate && export \$(grep -v '^#' .env | xargs) && python core/bridges/ghost_ws_bridge.py --port 9743" \
  Enter

sleep 3

# Verify WebSocket port
ss -tlnp | grep 9743
# Expected: LISTEN
```

### 5.8 Launch the CLI TUI

```bash
# In tmux window 'cli' (or a fresh terminal)
tmux send-keys -t bizra:cli \
  "cd /mnt/c/BIZRA-DATA-LAKE && source .venv/bin/activate && bizra" \
  Enter

# The Rust TUI should open with its 12 widget dashboard:
#   [Kernel] [Mission] [Ghost] [Wallet] [Federation] [Proof]
#   [Agents] [Briefing] [Status] [Organize] [Skills] [Heartbeat]
```

---

## Part 6 — Verify Production Readiness

**Status: TRUTH — Do not tag v1.0.0 until every check below passes.**

### 6.1 Service Health Matrix

Run all checks in sequence:

```bash
#!/usr/bin/env bash
# Save as: /mnt/c/BIZRA-DATA-LAKE/scripts/health_check_all.sh
set -e

PASS=0; FAIL=0

check() {
  local name="$1"; local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "[✓] $name"
    ((PASS++))
  else
    echo "[✗] $name — FAILED"
    ((FAIL++))
  fi
}

echo "=== BIZRA Node0 Health Matrix ==="
check "Redis"           "docker exec \$(docker ps -q -f name=redis) redis-cli ping | grep -q PONG"
check "Node Gateway"    "curl -sf http://localhost:8000/health"
check "Kernel Daemon"   "curl -sf http://localhost:9740/health"
check "MCP Transport"   "ss -tlnp | grep -q 9741"
check "Ghost WS"        "ss -tlnp | grep -q 9743"
check "Ollama"          "curl -sf http://localhost:11434/api/tags"
check "LM Studio"       "curl -sf http://\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):1234/v1/models"
check "URP Registry"    "curl -sf http://localhost:8011/health"

echo ""
echo "Result: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo "ALL SERVICES HEALTHY — Node0 is ALIVE" || exit 1
```

```bash
chmod +x /mnt/c/BIZRA-DATA-LAKE/scripts/health_check_all.sh
bash /mnt/c/BIZRA-DATA-LAKE/scripts/health_check_all.sh
```

### 6.2 Constitutional Health Endpoint

```bash
curl -s http://localhost:9740/v1/health/constitutional | jq
# Expected:
# {
#   "node": "node0",
#   "ihsan_threshold": 0.95,
#   "strict_ihsan_threshold": 0.99,
#   "snr_threshold_minimum": 0.85,
#   "adl_gini_threshold": 0.35,
#   "runtime_ihsan_threshold": 1.0,
#   "constitutional_status": "COMPLIANT",
#   "timestamp": "..."
# }
```

### 6.3 Rust Test Suite

```bash
cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega

# Run full workspace test (exclude bindings that require special env)
cargo test --workspace \
  --exclude fate-binding \
  --exclude iceoryx-bridge \
  --exclude bizra-python \
  -- --test-threads=24   # use all cores on i9-14900HX

# Expected: 1,539 passed, 0 failed

# Key constitutional crates specifically
cargo test -p bizra-protocol    # BLAKE3 + Ed25519 chain
cargo test -p bizra-sippar      # Zero-drift Babylonian arithmetic
cargo test -p bizra-hooks       # Nervous system — 76 tests
```

### 6.4 Python Test Suite

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate

# Core PCI tests (117 tests)
pytest tests/ -v \
  --ignore=tests/test_integration_production.py \
  -m "not slow and not requires_ollama and not requires_network" \
  --tb=short

# Rust bridge tests
python tests/test_rust_bridge.py
python tests/test_rust_bridge_v2.py

# With Ollama running — include MOE bridge tests
pytest tests/ -m "requires_ollama" -v

# Expected: 117+ passed, 0 failed
```

### 6.5 Walking Skeleton Test

```bash
cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega

# The Walking Skeleton test runs end-to-end through all constitutional layers
cargo test -p bizra-core walking_skeleton -- --nocapture

# Expected:
# [✓] Layer 1: Constitutional gate activated
# [✓] Layer 2: Identity seal verified
# [✓] Layer 3: FATE gate passed (Ihsān ≥ 0.95)
# [✓] Layer 4: Proof receipt generated
# [✓] Layer 5: Federation gossip initiated
# Walking Skeleton: COMPLETE
```

### 6.6 Cross-Language Threshold Sync

This verifies that Rust and Python agree on every constitutional constant:

```bash
cd /mnt/c/BIZRA-DATA-LAKE
python -c "
from core.sovereign.cmn_runtime import IHSAN_THRESHOLD, STRICT_IHSAN_THRESHOLD
from core.sovereign.cmn_runtime import SNR_THRESHOLD_MINIMUM, ADL_GINI_THRESHOLD
from core.sovereign.cmn_runtime import RUNTIME_IHSAN_THRESHOLD

assert IHSAN_THRESHOLD == 0.95,         f'FAIL: {IHSAN_THRESHOLD}'
assert STRICT_IHSAN_THRESHOLD == 0.99,  f'FAIL: {STRICT_IHSAN_THRESHOLD}'
assert SNR_THRESHOLD_MINIMUM == 0.85,   f'FAIL: {SNR_THRESHOLD_MINIMUM}'
assert ADL_GINI_THRESHOLD == 0.35,      f'FAIL: {ADL_GINI_THRESHOLD}'
assert RUNTIME_IHSAN_THRESHOLD == 1.0,  f'FAIL: {RUNTIME_IHSAN_THRESHOLD}'

print('[✓] All constitutional thresholds: Python verified')
"

# Rust side: tested by bizra-protocol crate
cd bizra-omega
cargo test -p bizra-protocol test_constitutional_constants -- --nocapture
# Expected: [✓] All constitutional thresholds: Rust verified
```

### 6.7 Run a Live Mission

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate

# Run the genesis mission (requires LM Studio or Ollama running)
python genesis_mission.py

# Expected:
# [✓] Ihsān gate: PASSED (score ≥ 0.95)
# [✓] SNR gate: PASSED (score ≥ 0.85)
# [✓] Mission receipt: SIGNED (Ed25519)
# [✓] Mission complete
```

---

## Part 7 — Run the 24-Hour Heartbeat

**Status: TRUTH — The heartbeat is proof of sustained operation. 288 ticks = 24 hours at 5-minute intervals.**

### 7.1 Start the Heartbeat Daemon

```bash
# Create the heartbeat log directory on the evidence drive
mkdir -p /mnt/b/BIZRA-SOVEREIGN/heartbeat/$(date +%Y%m%d)

# Start the heartbeat daemon (background, nohup)
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate

nohup python core/sovereign/heartbeat_daemon.py \
  --interval 300 \
  --ticks 288 \
  --evidence-dir /mnt/b/BIZRA-SOVEREIGN/heartbeat/$(date +%Y%m%d) \
  --node-id node0 \
  > /mnt/b/BIZRA-SOVEREIGN/heartbeat/heartbeat_$(date +%Y%m%d).log 2>&1 &

HEARTBEAT_PID=$!
echo $HEARTBEAT_PID > /mnt/b/BIZRA-SOVEREIGN/heartbeat/heartbeat.pid
echo "Heartbeat daemon started. PID: $HEARTBEAT_PID"
echo "Log: /mnt/b/BIZRA-SOVEREIGN/heartbeat/heartbeat_$(date +%Y%m%d).log"
```

### 7.2 Monitor the Heartbeat

```bash
# Watch live heartbeat log
tail -f /mnt/b/BIZRA-SOVEREIGN/heartbeat/heartbeat_$(date +%Y%m%d).log

# Check tick count at any time
grep -c "TICK" /mnt/b/BIZRA-SOVEREIGN/heartbeat/heartbeat_$(date +%Y%m%d).log
# Expected: increases from 1 to 288 over 24 hours

# Check via CLI
bizra status | grep heartbeat
```

Each tick logs:
- Timestamp (Dubai GMT+4)
- Kernel health (port 9740 response)
- Constitutional compliance score
- Active model (LM Studio or Ollama fallback)
- Redis ping latency
- BLAKE3 tick hash (tamper-evident chain)

### 7.3 Package the Evidence Bundle

After 288 ticks (or at any checkpoint):

```bash
cd /mnt/b/BIZRA-SOVEREIGN

# Create timestamped evidence bundle
BUNDLE_DATE=$(date +%Y%m%d_%H%M%S)
BUNDLE_DIR="evidence-bundle-${BUNDLE_DATE}"
mkdir -p $BUNDLE_DIR

# Copy heartbeat logs
cp -r heartbeat/ $BUNDLE_DIR/

# Copy genesis provenance
cp /mnt/c/BIZRA-DATA-LAKE/GENESIS_PROVENANCE.json $BUNDLE_DIR/
cp /mnt/c/BIZRA-DATA-LAKE/GENESIS_MANIFEST_VERIFIED.yaml $BUNDLE_DIR/

# Snapshot sovereign profile (no secrets)
cp /mnt/c/BIZRA-DATA-LAKE/bizra-omega/bizra-cli/config/sovereign_profile.yaml $BUNDLE_DIR/

# Generate BLAKE3 manifest of the entire bundle
find $BUNDLE_DIR -type f | sort | xargs -I{} sh -c 'echo "$(b3sum {} | awk "{print \$1}") {}"' \
  > $BUNDLE_DIR/BUNDLE_MANIFEST.b3

echo "Evidence bundle: /mnt/b/BIZRA-SOVEREIGN/$BUNDLE_DIR"
```

### 7.4 Tag v1.0.0-genesis

```bash
cd /mnt/c/BIZRA-DATA-LAKE

# Confirm all tests pass one final time
cd bizra-omega && cargo test --workspace --exclude fate-binding --exclude iceoryx-bridge --exclude bizra-python
cd ..
pytest tests/ -m "not slow and not requires_network" -q

# Tag
git tag -a v1.0.0-genesis \
  -m "BIZRA Node0 Genesis — Mohamed Beshr — Dubai, UAE — $(date '+%Y-%m-%d %H:%M %Z')

Constitutional compliance: VERIFIED
Heartbeat: 288 ticks
Rust tests: 1,539 passed
Python PCI tests: 117 passed
Ihsān threshold: 0.95
BLAKE3 identity: SEALED"

# Push tag to remote
git push origin v1.0.0-genesis

echo "[✓] v1.0.0-genesis tagged and pushed"
```

---

## Part 8 — Post-Production Operations

**Status: TRUTH — A sovereign node never stops being maintained.**

### 8.1 Daily Manifest Validation

Run every morning (add to cron or WSL task):

```bash
# Add to crontab: crontab -e
# 07:00 Dubai time = 03:00 UTC
# 0 3 * * * /mnt/c/BIZRA-DATA-LAKE/scripts/daily_manifest_check.sh

cat > /mnt/c/BIZRA-DATA-LAKE/scripts/daily_manifest_check.sh <<'EOF'
#!/usr/bin/env bash
set -e
LOG="/mnt/b/BIZRA-SOVEREIGN/manifest_check_$(date +%Y%m%d).log"
echo "=== Daily Manifest Check $(date) ===" | tee $LOG

cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate

# 1. Verify genesis seal
bizra trust >> $LOG 2>&1

# 2. Quick Rust constitutional test
cd bizra-omega
cargo test -p bizra-protocol -p bizra-sippar -p bizra-hooks -q >> $LOG 2>&1

# 3. Health check all services
bash scripts/health_check_all.sh >> $LOG 2>&1

# 4. Constitutional health endpoint
curl -sf http://localhost:9740/v1/health/constitutional | jq >> $LOG 2>&1

echo "Manifest check complete: $(date)" | tee -a $LOG
EOF

chmod +x /mnt/c/BIZRA-DATA-LAKE/scripts/daily_manifest_check.sh
```

### 8.2 Log Rotation

The kernel daemon (`core/sovereign/kernel_daemon.py`) manages its own log rotation with:
- Maximum log file size: **5 MB**
- Backup count: **3 files**

This is configured via Python's `RotatingFileHandler`. For additional system-level rotation:

```bash
# Install logrotate config for BIZRA logs
sudo tee /etc/logrotate.d/bizra <<'EOF'
/mnt/b/BIZRA-SOVEREIGN/heartbeat/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    dateext
    dateformat -%Y%m%d
}

/mnt/b/BIZRA-SOVEREIGN/manifest_check_*.log {
    weekly
    rotate 12
    compress
    missingok
    notifempty
}
EOF

# Test logrotate config
sudo logrotate --debug /etc/logrotate.d/bizra
```

### 8.3 Backup Strategy

| Drive | Contents | Backup Method |
|-------|----------|---------------|
| `B:\BIZRA-SOVEREIGN` | Constitutional evidence, genesis proofs, heartbeat logs, BLAKE3 manifests | Immutable — never delete. Compress monthly to `B:\BIZRA-SOVEREIGN-ARCHIVE-YYYYMM.tar.zst` |
| `C:\BIZRA-DATA-LAKE` | Runtime codebase | Git remote (GitHub). No additional backup needed — Git is the backup. |

```bash
# Monthly evidence archive (run on the 1st of each month)
ARCHIVE_NAME="BIZRA-SOVEREIGN-$(date +%Y%m).tar.zst"
tar --zstd -cf /mnt/b/${ARCHIVE_NAME} /mnt/b/BIZRA-SOVEREIGN/
echo "Archive: /mnt/b/${ARCHIVE_NAME}"

# Verify archive integrity
tar --zstd -tf /mnt/b/${ARCHIVE_NAME} | wc -l
```

### 8.4 Updating the Model Fleet

When new model versions are available:

```bash
# LM Studio: Download new model in UI, update sovereign_profile.yaml model names

# Ollama: Pull new version
ollama pull qwen2.5:7b          # upgrade from :3b if RAM allows
ollama pull deepseek-r1:32b     # upgrade reasoning model

# After any model change, run the constitutional mission to verify Ihsān gate
cd /mnt/c/BIZRA-DATA-LAKE
python genesis_mission.py

# Update sovereign_profile.yaml model entries to match new versions
nano bizra-omega/bizra-cli/config/sovereign_profile.yaml
```

### 8.5 Monitoring and Alerting

```bash
# Install a lightweight status dashboard (optional)
# BIZRA exposes Prometheus-compatible metrics at:
curl -s http://localhost:9740/metrics | head -30

# Quick Node0 status report (run anytime)
bizra status

# View the 12-widget TUI dashboard
bizra

# Specific subsystems
bizra ghost      # See what agents are processing
bizra wallet     # SEED token balance
bizra briefing   # Morning briefing from your AI
bizra agents     # Status of all 12 sovereign agents
```

### 8.6 CI Workflow Reference

Your repository has 21 active CI workflows. Key ones for local validation:

| Workflow | What it Checks | Local Equivalent |
|----------|---------------|------------------|
| `walking-skeleton.yml` | End-to-end proof chain | `cargo test -p bizra-core walking_skeleton` |
| `canonical-validation-gate.yml` | Constitutional constant sync | `python -c "from core.sovereign.cmn_runtime import *; ..."` |
| `proof-pyramid-gate.yml` | Proof receipt generation | `cargo test -p bizra-protocol` |
| `performance.yml` | Query latency < 200ms | `python core/performance/profile_query_latency.py` |
| `security.yml` | Auth + injection tests | `pytest tests/test_integration_production.py` |
| `wire-completeness-audit.yml` | All ports wired | `bash scripts/health_check_all.sh` |
| `quality-management.yml` | Ihsān gate on all outputs | `pytest tests/ -m quality` |

---

## Port Reference

| Port | Service | Protocol | Host |
|------|---------|----------|------|
| 1234 | LM Studio | HTTP/OpenAI-compat | Windows (172.22.48.1) |
| 6379 | Redis | TCP | Docker internal |
| 8000 | Node Gateway | HTTP (FastAPI) | WSL / Docker |
| 8011 | URP Registry | HTTP | Docker |
| 9740 | Kernel Daemon | HTTP | WSL |
| 9741 | MCP Transport | JSON-RPC / TCP | WSL |
| 9742 | Desktop Bridge | TCP | WSL |
| 9743 | Ghost WebSocket | WebSocket | WSL |
| 11434 | Ollama | HTTP | WSL |

---

## Constitutional Threshold Reference

| Constant | Value | Meaning |
|----------|-------|---------|
| `IHSAN_THRESHOLD` | 0.95 | Minimum quality for any inference output |
| `STRICT_IHSAN_THRESHOLD` | 0.99 | High-stakes decisions threshold |
| `SNR_THRESHOLD_MINIMUM` | 0.85 | Minimum signal-to-noise ratio |
| `SNR_THRESHOLD_T0_ELITE` | 0.98 | Elite T0 knowledge quality |
| `ADL_GINI_THRESHOLD` | 0.35 | Maximum resource distribution inequality |
| `RUNTIME_IHSAN_THRESHOLD` | 1.0 | Runtime loop — perfection required |

---

## MOE Bridge Expert Routing

| Pattern Tag | Model | Role |
|-------------|-------|------|
| `pat_r` | `deepseek-r1:14b` | Reasoning |
| `pat_k` | `qwen2.5:3b` | Knowledge |
| `pat_s` | `qwen2.5-coder:7b` | Code |
| `sat_g` | `phi3:mini` | Governance |
| `sat_v` | `phi3:mini` | Verification |

---

## Troubleshooting

### LM Studio unreachable from WSL

```bash
# Check Windows firewall — allow port 1234 inbound for WSL2
# In PowerShell (as Admin):
New-NetFirewallRule -DisplayName "LM Studio WSL" -Direction Inbound -Protocol TCP -LocalPort 1234 -Action Allow

# Refresh WSL gateway IP
cat /etc/resolv.conf | grep nameserver
```

### Rust build fails on AVX-512

```bash
# Fall back to standard release without AVX-512
cargo build --workspace --release
# Remove the target-feature flags from ~/.cargo/config.toml if needed
```

### Python bridge import error

```bash
# Rebuild the maturin wheel
cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega/bizra-python
maturin develop --release
python -c "import bizra; print(bizra.__version__)"
```

### Redis auth failure after password change

```bash
# Connect with password
redis-cli -h 127.0.0.1 -p 6379 -a "${REDIS_PASSWORD}" ping
# Update REDIS_URL in .env if needed
```

### Docker Compose services not starting

```bash
cd /mnt/c/BIZRA-DATA-LAKE/.tmp_prod_artifacts_v2/deploy
docker compose logs --tail 100
docker compose down && docker compose up -d
```

---

## Completion Checklist

```
[ ] WSL2 running Ubuntu 24.04
[ ] Rust 1.91+ installed
[ ] Python 3.12 venv active
[ ] Node.js 20+ installed
[ ] Docker Desktop with WSL2 backend
[ ] LM Studio serving on port 1234 (Windows)
[ ] Ollama serving on port 11434 (WSL)
[ ] Repository cloned to /mnt/c/BIZRA-DATA-LAKE
[ ] Rust workspace built (cargo build --workspace --release)
[ ] Python bridge built (maturin develop --release)
[ ] Docker services up (Redis, Gateway, URP x4)
[ ] .env configured with generated secrets
[ ] sovereign_profile.yaml set to MoMo / Dubai / GMT+4
[ ] Genesis ceremony completed (Ed25519 keypair sealed)
[ ] bizra trust: VALID
[ ] first_breath.py: PASSED
[ ] Kernel Daemon: port 9740 healthy
[ ] Node Gateway: port 8000 healthy
[ ] MCP Transport: port 9741 listening
[ ] Ghost WS: port 9743 listening
[ ] Rust tests: 1,539 passed
[ ] Python PCI tests: 117 passed
[ ] Walking Skeleton: COMPLETE
[ ] Constitutional thresholds: cross-language sync VERIFIED
[ ] Live mission: Ihsān gate PASSED
[ ] Heartbeat daemon: started (288 ticks / 24 hours)
[ ] Evidence bundle: packaged to B:\BIZRA-SOVEREIGN
[ ] v1.0.0-genesis: tagged and pushed
```

---

_"The node that cannot prove itself cannot govern itself."_

**BIZRA Node0 — Standing sovereign. Standing certain. Standing tall.**

---

*Document: BIZRA_NODE0_PRODUCTION_SETUP_v1.md*
*Generated: 2026 — MIT License — Copyright BIZRA Sovereign*
*Repository: https://github.com/BizraInfo/bizra-data-lake*
