#!/bin/bash
# ═══════════════════════════════════════════════════════════
# BIZRA NODE0 — ACTIVATION
# Run this once. Your sovereign AI boots.
# ═══════════════════════════════════════════════════════════
#
# From WSL terminal:
#   cd /mnt/c/BIZRA-DATA-LAKE
#   chmod +x activate.sh && ./activate.sh
#
# What this does:
#   1. Activates the Python venv
#   2. Checks your LM Studio / Ollama connection
#   3. Starts the API server on port 8080
#   4. You open node0_activate.html in Brave
#   5. Your sovereign node is live. Type missions. It works.
# ═══════════════════════════════════════════════════════════

set -e
GOLD='\033[38;2;201;169;98m'
GREEN='\033[38;2;45;212;160m'
RED='\033[38;2;240;96;80m'
DIM='\033[2m'
BOLD='\033[1m'
R='\033[0m'

echo ""
echo -e "${GOLD}╔════════════════════════════════════════════════╗${R}"
echo -e "${GOLD}║           BIZRA NODE0 ACTIVATION               ║${R}"
echo -e "${GOLD}║                البذرة                           ║${R}"
echo -e "${GOLD}╚════════════════════════════════════════════════╝${R}"
echo ""

# Step 1: Virtual environment
echo -e "${DIM}[1/5]${R} Activating Python environment..."
if [ -f ".venv-linux/bin/activate" ]; then
    source .venv-linux/bin/activate
    echo -e "  ${GREEN}✓${R} venv active: $(python --version)"
else
    echo -e "  ${RED}✗${R} .venv-linux not found"
    echo -e "  ${DIM}Run: python3 -m venv .venv-linux && source .venv-linux/bin/activate && pip install -e '.[dev]'${R}"
    exit 1
fi

# Step 2: Check core imports
echo -e "${DIM}[2/5]${R} Checking sovereign kernel..."
python -c "from core.sovereign.runtime import SovereignRuntime; print('  \033[38;2;45;212;160m✓\033[0m SovereignRuntime loaded')" 2>/dev/null || {
    echo -e "  ${RED}✗${R} Cannot import SovereignRuntime. Run: pip install -e '.[dev]'"
    exit 1
}

# Step 3: Check identity
echo -e "${DIM}[3/5]${R} Checking node identity..."
if [ -f "sovereign_state/node0_genesis.json" ]; then
    NODE_NAME=$(python -c "import json; d=json.load(open('sovereign_state/node0_genesis.json')); print(d['identity']['name'])" 2>/dev/null)
    NODE_ID=$(python -c "import json; d=json.load(open('sovereign_state/node0_genesis.json')); print(d['identity']['node_id'])" 2>/dev/null)
    PUB_KEY=$(python -c "import json; d=json.load(open('sovereign_state/node0_genesis.json')); print(d['identity']['public_key'][:16])" 2>/dev/null)
    PAT_COUNT=$(python -c "import json; d=json.load(open('sovereign_state/node0_genesis.json')); print(len(d.get('pat_team',{}).get('agents',[])))" 2>/dev/null)
    SAT_COUNT=$(python -c "import json; d=json.load(open('sovereign_state/node0_genesis.json')); print(len(d.get('sat_team',{}).get('agents',[])))" 2>/dev/null)
    echo -e "  ${GREEN}✓${R} ${NODE_NAME} · ${NODE_ID}"
    echo -e "  ${GREEN}✓${R} Signer: ${PUB_KEY}..."
    echo -e "  ${GREEN}✓${R} PAT: ${PAT_COUNT} agents · SAT: ${SAT_COUNT} agents"
else
    echo -e "  ${DIM}No genesis found. Running onboard...${R}"
    python -m core.sovereign onboard
fi

# Step 4: Check LLM backends
echo -e "${DIM}[4/5]${R} Checking inference backends..."
LMSTUDIO_URL=$(python -c "from core.integration.constants import LMSTUDIO_URL; print(LMSTUDIO_URL)" 2>/dev/null)
if curl -s --max-time 2 "${LMSTUDIO_URL}/v1/models" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${R} LM Studio at ${LMSTUDIO_URL}"
elif curl -s --max-time 2 "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${R} Ollama at localhost:11434"
else
    echo -e "  ${RED}⚠${R}  No LLM backend detected. Start LM Studio or Ollama."
fi

# Step 5: Launch
echo -e "${DIM}[5/5]${R} Starting sovereign API server..."
echo ""
echo -e "${GOLD}═══════════════════════════════════════════════════${R}"
echo -e "${BOLD} Node0 is booting on port 8080${R}"
echo ""
echo -e " ${GREEN}Option A:${R} Open node0_activate.html in Brave"
echo -e "          ${DIM}(visual interface with receipts)${R}"
echo ""
echo -e " ${GREEN}Option B:${R} Open another WSL terminal and run:"
echo -e "          ${DIM}cd /mnt/c/BIZRA-DATA-LAKE${R}"
echo -e "          ${DIM}source .venv-linux/bin/activate${R}"
echo -e "          ${DIM}python -m core.sovereign${R}"
echo -e "          ${DIM}(terminal REPL with full commands)${R}"
echo -e "${GOLD}═══════════════════════════════════════════════════${R}"
echo ""
echo -e "${DIM}Press Ctrl+C to shut down${R}"
echo ""

python -m core.sovereign serve --port 8080
