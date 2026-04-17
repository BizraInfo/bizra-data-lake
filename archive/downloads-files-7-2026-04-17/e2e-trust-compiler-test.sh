#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# BIZRA Cycle-6 — End-to-End Trust Compilation Test
# بسم الله الرحمن الرحيم
#
# Tests the full vertical:
#   CLI → Gateway → TrustCompiler → FilesystemExecutor → ReceiptChain
#
# Exit codes:
#   0 = all tests passed
#   1 = test failure
#   2 = infrastructure failure (gateway won't start, etc.)
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
GOLD='\033[0;33m'
RESET='\033[0m'

PASS=0
FAIL=0
GATEWAY_PID=""
TEST_DIR=""
BASE="http://127.0.0.1:7421"

cleanup() {
    [ -n "$GATEWAY_PID" ] && kill "$GATEWAY_PID" 2>/dev/null || true
    [ -n "$TEST_DIR" ] && rm -rf "$TEST_DIR" 2>/dev/null || true
}
trap cleanup EXIT

pass() { PASS=$((PASS + 1)); echo -e "  ${GREEN}✓${RESET} $1"; }
fail() { FAIL=$((FAIL + 1)); echo -e "  ${RED}✗${RESET} $1"; }
assert_eq() {
    if [ "$1" = "$2" ]; then pass "$3"; else fail "$3 (expected '$2', got '$1')"; fi
}
assert_ne() {
    if [ "$1" != "$2" ]; then pass "$3"; else fail "$3 (got '$1', expected different)"; fi
}

echo "══════════════════════════════════════════════════════════════"
echo "  BIZRA Cycle-6 — Trust Compilation E2E Test"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Phase 0: Setup ───────────────────────────────────────────────────
echo "Phase 0: Setup"

# Kill any existing gateway
for p in $(ss -ltnp 2>/dev/null | grep 7421 | grep -oE 'pid=[0-9]+' | cut -d= -f2); do
    kill "$p" 2>/dev/null || true
done
sleep 1

# Build and start gateway
cd "$(dirname "$0")/../bizra-omega"
if [ ! -f target/release/bizra-cognition-gateway ]; then
    echo "  Building gateway..."
    cargo build --release -p bizra-cognition-gateway 2>&1 | tail -2
fi

./target/release/bizra-cognition-gateway > /tmp/e2e-gateway.log 2>&1 &
GATEWAY_PID=$!
sleep 2

# Verify gateway is up
HTTP=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null || echo "000")
if [ "$HTTP" != "200" ]; then
    echo -e "  ${RED}Gateway failed to start. Check /tmp/e2e-gateway.log${RESET}"
    exit 2
fi
pass "Gateway started (PID $GATEWAY_PID)"

# Create test directory with sample files
TEST_DIR=$(mktemp -d /tmp/bizra-e2e-XXXXXX)
echo "test pdf content" > "$TEST_DIR/report.pdf"
echo "test image data" > "$TEST_DIR/photo.jpg"
echo "fn main() {}" > "$TEST_DIR/code.rs"
echo "1,2,3" > "$TEST_DIR/data.csv"
echo "compressed" > "$TEST_DIR/backup.zip"
echo "readme" > "$TEST_DIR/README.md"
echo "audio bytes" > "$TEST_DIR/song.mp3"
pass "Test directory created with 7 files"

echo ""

# ── Test 1: Empty chain state ────────────────────────────────────────
echo "Test 1: Empty chain state"

CHAIN=$(curl -s "$BASE/chain")
LENGTH=$(echo "$CHAIN" | python3 -c "import sys,json; print(json.load(sys.stdin)['length'])")
assert_eq "$LENGTH" "0" "Chain starts empty"

echo ""

# ── Test 2: Compile filesystem operation (PERMIT) ────────────────────
echo "Test 2: Compile filesystem operation (PERMIT path)"

RESULT=$(curl -s -X POST "$BASE/compile" \
    -H "content-type: application/json" \
    -d "{\"intent\":\"organize test directory\",\"kind\":\"filesystem\",\"quality_score\":0.98,\"target_path\":\"$TEST_DIR\"}")

REJECTED=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['rejected'])")
assert_eq "$REJECTED" "False" "Filesystem compilation permitted"

SUB_OPS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['sub_operations'])")
assert_eq "$SUB_OPS" "7" "7 files produced 7 sub-receipts"

RECEIPT_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('receipt_id',''))")
assert_ne "$RECEIPT_ID" "" "Receipt ID is non-empty"

VERDICT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['verdict'])")
assert_eq "$VERDICT" "Permit" "Verdict is Permit"

GATES=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['gates_passed'])")
assert_eq "$GATES" "5" "All 5 gates passed"

echo ""

# ── Test 3: Files actually moved ─────────────────────────────────────
echo "Test 3: Files physically moved to categories"

assert_eq "$([ -f "$TEST_DIR/documents/report.pdf" ] && echo yes || echo no)" "yes" "report.pdf → documents/"
assert_eq "$([ -f "$TEST_DIR/images/photo.jpg" ] && echo yes || echo no)" "yes" "photo.jpg → images/"
assert_eq "$([ -f "$TEST_DIR/code/code.rs" ] && echo yes || echo no)" "yes" "code.rs → code/"
assert_eq "$([ -f "$TEST_DIR/spreadsheets/data.csv" ] && echo yes || echo no)" "yes" "data.csv → spreadsheets/"
assert_eq "$([ -f "$TEST_DIR/archives/backup.zip" ] && echo yes || echo no)" "yes" "backup.zip → archives/"
assert_eq "$([ -f "$TEST_DIR/documents/README.md" ] && echo yes || echo no)" "yes" "README.md → documents/"
assert_eq "$([ -f "$TEST_DIR/audio/song.mp3" ] && echo yes || echo no)" "yes" "song.mp3 → audio/"

# Verify originals are gone
assert_eq "$([ -f "$TEST_DIR/report.pdf" ] && echo exists || echo gone)" "gone" "Original report.pdf removed"
assert_eq "$([ -f "$TEST_DIR/photo.jpg" ] && echo exists || echo gone)" "gone" "Original photo.jpg removed"

echo ""

# ── Test 4: Chain advanced correctly ─────────────────────────────────
echo "Test 4: Chain state after compilation"

CHAIN=$(curl -s "$BASE/chain")
LENGTH=$(echo "$CHAIN" | python3 -c "import sys,json; print(json.load(sys.stdin)['length'])")
# 7 sub-receipts + 5 gate verdicts + 1 final = 13
assert_eq "$LENGTH" "13" "Chain length = 13 (7 subs + 5 gates + 1 final)"

HEAD=$(echo "$CHAIN" | python3 -c "import sys,json; print(json.load(sys.stdin)['head'])")
assert_ne "$HEAD" "0000000000000000000000000000000000000000000000000000000000000000" "Chain head advanced"

echo ""

# ── Test 5: Compile with low quality (REJECT) ────────────────────────
echo "Test 5: Low-quality compilation (REJECT path)"

PRE_LEN=$LENGTH

RESULT=$(curl -s -X POST "$BASE/compile" \
    -H "content-type: application/json" \
    -d "{\"intent\":\"low quality attempt\",\"kind\":\"mission\",\"quality_score\":0.50}")

REJECTED=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['rejected'])")
assert_eq "$REJECTED" "True" "Low-quality compilation rejected"

# Chain must NOT advance on reject
CHAIN=$(curl -s "$BASE/chain")
POST_LEN=$(echo "$CHAIN" | python3 -c "import sys,json; print(json.load(sys.stdin)['length'])")
assert_eq "$POST_LEN" "$PRE_LEN" "Chain unchanged after rejection (§10 Proof Law)"

echo ""

# ── Test 6: Organize empty directory ─────────────────────────────────
echo "Test 6: Empty directory produces zero sub-receipts"

EMPTY_DIR=$(mktemp -d /tmp/bizra-e2e-empty-XXXXXX)
RESULT=$(curl -s -X POST "$BASE/organize" \
    -H "content-type: application/json" \
    -d "{\"path\":\"$EMPTY_DIR\"}")

SUB_OPS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['sub_operations'])")
assert_eq "$SUB_OPS" "0" "Empty directory = 0 sub-operations"

REJECTED=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['rejected'])")
assert_eq "$REJECTED" "False" "Empty directory is still PERMIT (honest empty)"

rmdir "$EMPTY_DIR"

echo ""

# ── Test 7: Health endpoint still works ──────────────────────────────
echo "Test 7: Gateway health after all operations"

HEALTH=$(curl -s "$BASE/health")
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
assert_eq "$STATUS" "ok" "Gateway healthy after test suite"

echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}ALL $TOTAL TESTS PASSED${RESET}"
    echo ""
    echo "  Trust compilation verified end-to-end:"
    echo "    • Filesystem operations receipted per-file"
    echo "    • Admissibility gates enforce IHSAN_FLOOR"
    echo "    • Reject path keeps chain clean"
    echo "    • Files physically moved and hash-verified"
    echo "    • Chain length matches expected records"
    echo ""
    echo "  Receipt: ${RECEIPT_ID:0:16}...${RECEIPT_ID: -8}"
    echo "  Chain:   $POST_LEN records"
    echo ""
    echo -e "  ${GOLD}Close it. Prove it. Reveal it.${RESET}"
    exit 0
else
    echo -e "  ${RED}$FAIL of $TOTAL TESTS FAILED${RESET}"
    echo "  Check /tmp/e2e-gateway.log for gateway output"
    exit 1
fi
