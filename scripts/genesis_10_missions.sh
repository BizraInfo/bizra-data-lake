#!/bin/bash
# ═══════════════════════════════════════════════════════════
# BIZRA Genesis — 10 Mission Chain Test
# Proves the proof pyramid works end-to-end on NODE0
# ═══════════════════════════════════════════════════════════
#
# Prerequisites:
#   - Ollama running with qwen2.5:3b loaded
#   - bizra-node release binary built
#
# Usage:
#   chmod +x scripts/genesis_10_missions.sh
#   ./scripts/genesis_10_missions.sh
#
# Standing on: Deming (PDCA), Satoshi (hash chains), Al-Ghazali (Ihsan)
# ═══════════════════════════════════════════════════════════

set -euo pipefail

BINARY="./bizra-omega/target/release/bizra-node"
STATE_DIR="/tmp/bizra-genesis-test-$$"
MISSION_LOG="$STATE_DIR/missions.log"

# Verify prerequisites
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run: cd bizra-omega && cargo build --release -p bizra-node"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "ERROR: Ollama not reachable at localhost:11434"
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo "  BIZRA Genesis — 10 Mission Chain Test"
echo "  Binary: $BINARY"
echo "  State:  $STATE_DIR"
echo "═══════════════════════════════════════════════════════"

mkdir -p "$STATE_DIR"

# ── Phase 1: Teach identity ─────────────────────────────
echo ""
echo "[Phase 1/4] Teaching identity..."

TEACH_COMMANDS=(
    "TEACH\tfact\tI am Mumo, founder of BIZRA\t9900\t1000"
    "TEACH\tpreference\tI prefer Rust for core systems\t9500\t1001"
    "TEACH\tgoal\tBuilding sovereign AI for 8 billion\t9000\t1002"
    "TEACH\tpattern\tI work after Fajr prayer\t8500\t1003"
    "TEACH\tfact\tNode0 runs on i9-14900HX with 128GB DDR5\t9000\t1004"
)

for cmd in "${TEACH_COMMANDS[@]}"; do
    RESPONSE=$(echo -e "$cmd" | timeout 10 "$BINARY" \
        --user 1 --ihsan 9500 --state-dir "$STATE_DIR" \
        --no-banner --no-auto-session 2>/dev/null || true)
    echo "  TEACH → ${RESPONSE:0:60}"
done

echo "  Identity: 5 facts taught"

# ── Phase 2: Run 10 Missions ────────────────────────────
echo ""
echo "[Phase 2/4] Running 10 missions with Ollama inference..."

MISSIONS=(
    "What are the BIZRA constitutional thresholds?"
    "Explain the difference between PAT and SAT agents"
    "How does the SEED token economy work?"
    "What is Ihsan and why is the threshold 0.95?"
    "Describe the EventBus architecture"
    "What makes BIZRA different from ChatGPT?"
    "How does the receipt chain ensure integrity?"
    "What is a TeleScript and how does it travel?"
    "Explain the HHMM memory hierarchy"
    "What should I work on next for Genesis?"
)

PASS_COUNT=0
FAIL_COUNT=0
RECEIPT_IDS=()

for i in "${!MISSIONS[@]}"; do
    MISSION="${MISSIONS[$i]}"
    TS=$((2000 + i * 100))
    NUM=$((i + 1))

    # Send through the node with Ollama inference enabled
    RESPONSE=$(echo -e "START_SESSION\t$TS\nRECEIVE\t$MISSION\t$((TS+1))\nEND_SESSION\t$((TS+50))" | \
        BIZRA_ENABLE_OLLAMA_EXECUTE=1 \
        BIZRA_OLLAMA_MODEL=qwen2.5:3b \
        timeout 60 "$BINARY" \
            --user 1 --ihsan 9500 --state-dir "$STATE_DIR" \
            --no-banner --no-auto-session 2>/dev/null || true)

    # Extract receipt_id from response
    RECEIPT_ID=$(echo "$RESPONSE" | grep -oP 'receipt_id=\K[a-f0-9]+' | head -1 || true)
    GUARDIAN=$(echo "$RESPONSE" | grep -oP 'guardian_approved=\K[a-z]+' | head -1 || true)
    INFERENCE=$(echo "$RESPONSE" | grep -oP 'inference_executed=\K[a-z]+' | head -1 || true)

    if [ -n "$RECEIPT_ID" ] && [ "$GUARDIAN" = "true" ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        RECEIPT_IDS+=("$RECEIPT_ID")
        echo "  [$NUM/10] PASS | receipt=${RECEIPT_ID:0:16}... | inference=$INFERENCE | guardian=$GUARDIAN"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  [$NUM/10] FAIL | guardian=$GUARDIAN | inference=$INFERENCE"
    fi

    # Log full response
    echo "--- Mission $NUM ---" >> "$MISSION_LOG"
    echo "Q: $MISSION" >> "$MISSION_LOG"
    echo "R: $RESPONSE" >> "$MISSION_LOG"
    echo "" >> "$MISSION_LOG"
done

# ── Phase 3: Verify Receipt Chain ───────────────────────
echo ""
echo "[Phase 3/4] Verifying receipt chain..."

UNIQUE_RECEIPTS=$(printf '%s\n' "${RECEIPT_IDS[@]}" | sort -u | wc -l)
echo "  Receipts collected: ${#RECEIPT_IDS[@]}"
echo "  Unique receipts:    $UNIQUE_RECEIPTS"
echo "  Chain continuity:   $([ ${#RECEIPT_IDS[@]} -eq $UNIQUE_RECEIPTS ] && echo 'VALID (all unique)' || echo 'WARNING: duplicates detected')"

# ── Phase 4: Check Knowledge Persistence ────────────────
echo ""
echo "[Phase 4/4] Checking knowledge persistence..."

KNOWS_ME=$(echo "KNOWS_ME" | timeout 10 "$BINARY" \
    --user 1 --ihsan 9500 --state-dir "$STATE_DIR" \
    --no-banner --no-auto-session 2>/dev/null || true)
SCORE=$(echo "$KNOWS_ME" | grep -oP 'score=\K[0-9.]+' | head -1 || echo "0")
echo "  knows_me score: $SCORE"

# Check state files
echo "  State files:"
ls -lh "$STATE_DIR"/ 2>/dev/null | grep -v "^total" | while read line; do
    echo "    $line"
done

# ── Summary ────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  GENESIS 10-MISSION CHAIN — RESULTS"
echo "═══════════════════════════════════════════════════════"
echo "  Passed:    $PASS_COUNT / 10"
echo "  Failed:    $FAIL_COUNT / 10"
echo "  Receipts:  ${#RECEIPT_IDS[@]} collected, $UNIQUE_RECEIPTS unique"
echo "  Knows Me:  $SCORE"
echo "  State Dir: $STATE_DIR"
echo "  Full Log:  $MISSION_LOG"
echo "═══════════════════════════════════════════════════════"

if [ $PASS_COUNT -ge 8 ]; then
    echo ""
    echo "  GENESIS GATE: PASS"
    echo "  Block 0 is ready to mint."
    echo ""
    echo "  بذرة واحدة تصنع غابة"
    echo "  One seed makes a forest."
    exit 0
else
    echo ""
    echo "  GENESIS GATE: NOT YET ($PASS_COUNT/10, need 8+)"
    exit 1
fi
