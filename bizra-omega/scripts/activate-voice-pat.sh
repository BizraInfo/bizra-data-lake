#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Node0 Voice Activation — PersonaPlex-7B-V1
# ═══════════════════════════════════════════════════════════════════════════════
#
# Your first interaction with your PAT team will be through voice.
# "The voice is the soul. The role is the purpose. The Ihsān is the constraint."
#
# Genesis Hash: a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b
#
# Usage:
#   ./scripts/activate-voice-pat.sh [--guardian NAME] [--cpu-offload]
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PERSONAPLEX_DIR="/mnt/c/BIZRA-DATA-LAKE/personaplex"
SOVEREIGN_STATE="${BIZRA_SOVEREIGN_STATE_DIR:-/mnt/c/BIZRA-DATA-LAKE/sovereign_state}"

# Defaults
GUARDIAN="strategist"
CPU_OFFLOAD=""
PORT=8998

# Parse arguments
for arg in "$@"; do
    case $arg in
        --guardian=*) GUARDIAN="${arg#*=}" ;;
        --guardian) shift; GUARDIAN="$1" ;;
        --cpu-offload) CPU_OFFLOAD="--cpu-offload" ;;
        --port=*) PORT="${arg#*=}" ;;
        *) ;;
    esac
done

# Banner
clear
echo ""
echo -e "${CYAN}"
cat << 'EOF'
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║     ██████╗  █████╗ ████████╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗    ║
║     ██╔══██╗██╔══██╗╚══██╔══╝    ██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ║
║     ██████╔╝███████║   ██║       ██║   ██║██║   ██║██║██║     █████╗      ║
║     ██╔═══╝ ██╔══██║   ██║       ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ║
║     ██║     ██║  ██║   ██║        ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ║
║     ╚═╝     ╚═╝  ╚═╝   ╚═╝         ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ║
║                                                                            ║
║                    Personal Agentic Team — Voice Interface                 ║
║                          Powered by PersonaPlex-7B-V1                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# PAT Voice Mapping
declare -A PAT_VOICES
PAT_VOICES=(
    ["strategist"]="NATM1.pt"    # Professional male - Sun Tzu, Clausewitz, Porter
    ["researcher"]="NATF1.pt"    # Clear analytical female - Shannon, Besta, Hinton
    ["developer"]="NATM0.pt"     # Scholarly male - Knuth, Dijkstra, Thompson
    ["analyst"]="NATF2.pt"       # Authoritative female - Tukey, Fisher, Bayes
    ["reviewer"]="NATM2.pt"      # Calm wise male - Hoare, Dijkstra, Meyer
    ["executor"]="NATM3.pt"      # Commanding male - Deming, Taylor, Ohno
    ["guardian"]="NATF3.pt"      # Friendly female - Al-Ghazali, Rawls, Anthropic
)

declare -A PAT_PROMPTS
PAT_PROMPTS=(
    ["strategist"]="You are the Strategist of MoMo's Personal Agentic Team. Your role is to plan, analyze, and make strategic decisions. You stand on the shoulders of Sun Tzu, Clausewitz, and Porter. You speak with clarity and vision. All decisions must meet the Ihsān threshold of 0.95. You are speaking with your sovereign, MoMo, in Dubai."

    ["researcher"]="You are the Researcher of MoMo's Personal Agentic Team. Your role is to search, synthesize, and cite information. You stand on the shoulders of Shannon, Besta, and Hinton. You speak with precision and curiosity. Always cite sources and quantify confidence. You are speaking with your sovereign, MoMo, in Dubai."

    ["developer"]="You are the Developer of MoMo's Personal Agentic Team. Your role is to code, test, and deploy software. You stand on the shoulders of Knuth, Dijkstra, and Thompson. You speak technically but accessibly. You follow TDD principles. You are speaking with your sovereign, MoMo, in Dubai."

    ["analyst"]="You are the Analyst of MoMo's Personal Agentic Team. Your role is pattern recognition, measurement, and prediction. You stand on the shoulders of Tukey, Fisher, and Bayes. You speak with statistical rigor and clarity. Always show confidence intervals. You are speaking with your sovereign, MoMo, in Dubai."

    ["reviewer"]="You are the Reviewer of MoMo's Personal Agentic Team. Your role is to validate, critique, and improve work. You stand on the shoulders of Hoare, Dijkstra, and Meyer. You speak constructively and precisely. Focus on correctness, security, and maintainability. You are speaking with your sovereign, MoMo, in Dubai."

    ["executor"]="You are the Executor of MoMo's Personal Agentic Team. Your role is to execute tasks, monitor progress, and report status. You stand on the shoulders of Deming, Taylor, and Ohno. You speak with action-oriented clarity. Escalate blockers immediately. You are speaking with your sovereign, MoMo, in Dubai."

    ["guardian"]="You are the Guardian of MoMo's Personal Agentic Team. Your role is to protect, audit, and enforce ethical constraints. You stand on the shoulders of Al-Ghazali, Rawls, and Anthropic. You speak with wisdom and care. FATE Gates: Ihsān ≥ 0.95, Adl Gini ≤ 0.35. You are speaking with your sovereign, MoMo, in Dubai."
)

# Validate guardian selection
GUARDIAN_LOWER=$(echo "$GUARDIAN" | tr '[:upper:]' '[:lower:]')
if [[ ! -v "PAT_VOICES[$GUARDIAN_LOWER]" ]]; then
    echo -e "${RED}Unknown PAT agent: $GUARDIAN${NC}"
    echo ""
    echo "Available agents:"
    for agent in "${!PAT_VOICES[@]}"; do
        echo "  - $agent"
    done
    exit 1
fi

VOICE="${PAT_VOICES[$GUARDIAN_LOWER]}"
PROMPT="${PAT_PROMPTS[$GUARDIAN_LOWER]}"

echo -e "${GREEN}Node0 Genesis:${NC} $(cat "$SOVEREIGN_STATE/genesis_hash.txt" 2>/dev/null | head -c 16)..."
echo ""
echo -e "${BLUE}Activating PAT Agent:${NC} ${GUARDIAN_LOWER^}"
echo -e "${BLUE}Voice Profile:${NC} $VOICE"
echo ""

# Check HuggingFace token
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo "Set it with: export HF_TOKEN=your_token"
    echo "Get token at: https://huggingface.co/settings/tokens"
    echo ""
fi

# Check PersonaPlex installation
if [ ! -d "$PERSONAPLEX_DIR/moshi" ]; then
    echo -e "${RED}PersonaPlex not found at $PERSONAPLEX_DIR${NC}"
    exit 1
fi

# Create temp SSL directory
SSL_DIR=$(mktemp -d)
echo -e "${BLUE}SSL Directory:${NC} $SSL_DIR"

# Activate venv if exists
if [ -d "$PERSONAPLEX_DIR/.venv" ]; then
    echo -e "${BLUE}Activating Python environment...${NC}"
    source "$PERSONAPLEX_DIR/.venv/bin/activate"
fi

# Display connection info
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${CYAN}  Voice Interface Starting...${NC}"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo -e "  ${GREEN}Web UI:${NC}      https://localhost:$PORT"
echo -e "  ${GREEN}Agent:${NC}       ${GUARDIAN_LOWER^}"
echo -e "  ${GREEN}Voice:${NC}       $VOICE"
echo ""
echo -e "  ${YELLOW}Note: Accept the self-signed certificate in your browser${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo -e "${CYAN}بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ${NC}"
echo ""
echo -e "${GREEN}Your ${GUARDIAN_LOWER^} is ready to speak with you, MoMo.${NC}"
echo ""

# Create prompt file
PROMPT_FILE=$(mktemp)
echo "$PROMPT" > "$PROMPT_FILE"

# Launch PersonaPlex server
cd "$PERSONAPLEX_DIR"
python -m moshi.server \
    --ssl "$SSL_DIR" \
    --port "$PORT" \
    --device cuda \
    $CPU_OFFLOAD \
    2>&1 | while read -r line; do
        # Colorize output
        if [[ "$line" == *"ERROR"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" == *"INFO"* ]]; then
            echo -e "${BLUE}$line${NC}"
        elif [[ "$line" == *"connection"* ]]; then
            echo -e "${GREEN}$line${NC}"
        else
            echo "$line"
        fi
    done

# Cleanup
rm -rf "$SSL_DIR" "$PROMPT_FILE" 2>/dev/null || true
