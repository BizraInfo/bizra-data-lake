#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# BIZRA Cycle-6 G4 — End-to-End Polyglot Smoke Test
#
# بسم الله الرحمن الرحيم
#
# Niyyah §G4 contract:
#   "scripts/e2e-polyglot/ contains the full-stack smoke test; one CI
#    workflow runs it on every push; the test proves a real receipt
#    sealed through the polyglot chain."
#
# This test exercises the polyglot vertical:
#   Bash harness → HTTP (Rust gateway v0.2) → admissibility chain →
#   receipt artifact → ReceiptChain → HTTP read-back.
#
# Test 8 (Cycle-6 Arc 3): when BIZRA_RECEIPT_STORE_PATH is set, the chain
# rehydrates from sled + chain_snapshot.json across gateway restart.
#
# Test 9 (Arc 3.1+): scripts/operator-smoke-arc3.sh with
# BIZRA_RECEIPT_STORE_PATH=default resolves under DATA_LAKE_ROOT (isolated in CI).
#
# G3 precedent (cycle-6/g3-authority-adr.md) declares external
# award-winner-design as operator-facing authority. This test is
# intentionally gateway-direct so it runs in CI without depending on
# the external repo. External-proxy verification is a DR-drill concern
# covered by the rollback runbook, not every-push CI.
#
# Exit codes:
#   0 = all tests passed (G4 green)
#   1 = test assertion failure
#   2 = infrastructure failure (gateway won't boot, port conflict, etc.)
#
# Environment:
#   BIZRA_E2E_PORT (default: 7431)  — custom port to avoid local-dev clashes
#   BIZRA_E2E_SKIP_BUILD=1          — skip cargo build (must already exist)
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────
PORT="${BIZRA_E2E_PORT:-7431}"
BASE="http://127.0.0.1:${PORT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATEWAY_BIN="${REPO_ROOT}/bizra-omega/target/release/bizra-cognition-gateway"
LOG="/tmp/e2e-polyglot-gateway.log"

RED='\033[0;31m'; GREEN='\033[0;32m'; GOLD='\033[0;33m'; RESET='\033[0m'

PASS=0; FAIL=0; GATEWAY_PID=""
PERSIST_STORE_DIR=""

stop_gateway() {
    if [ -n "${GATEWAY_PID}" ] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
        kill "${GATEWAY_PID}" 2>/dev/null || true
        wait "${GATEWAY_PID}" 2>/dev/null || true
    fi
    GATEWAY_PID=""
}

wait_gateway_health() {
    local base="$1"
    local log="$2"
    for _ in $(seq 1 40); do
        sleep 0.25
        if curl -sf "${base}/health" > /dev/null 2>&1; then
            return 0
        fi
    done
    echo "--- gateway log (${log}) ---"
    tail -40 "${log}" 2>/dev/null || true
    return 1
}

start_gateway() {
    local port="$1"
    local log="$2"
    local store_path="${3:-}"
    stop_gateway
    if [ -n "${store_path}" ]; then
        BIZRA_COGNITION_PORT="${port}" BIZRA_RECEIPT_STORE_PATH="${store_path}" \
            "${GATEWAY_BIN}" > "${log}" 2>&1 &
    else
        BIZRA_COGNITION_PORT="${port}" "${GATEWAY_BIN}" > "${log}" 2>&1 &
    fi
    GATEWAY_PID=$!
    wait_gateway_health "http://127.0.0.1:${port}" "${log}"
}

cleanup() {
    stop_gateway
    if [ -n "${PERSIST_STORE_DIR}" ] && [ -d "${PERSIST_STORE_DIR}" ]; then
        rm -rf "${PERSIST_STORE_DIR}"
    fi
}
trap cleanup EXIT

pass() { PASS=$((PASS + 1)); echo -e "  ${GREEN}✓${RESET} $1"; }
fail() { FAIL=$((FAIL + 1)); echo -e "  ${RED}✗${RESET} $1"; }
info() { echo -e "  ${GOLD}▶${RESET} $1"; }

# ─── Pre-flight ──────────────────────────────────────────────────────
echo "═══ BIZRA G4 e2e-polyglot — $(date -u +%Y-%m-%dT%H:%M:%SZ) ═══"

# JSON tool — prefer jq, fall back to python3
if command -v jq > /dev/null; then
    JSON() { jq -r "$1"; }
elif command -v python3 > /dev/null; then
    JSON() { python3 -c "import sys,json; d=json.load(sys.stdin); k=sys.argv[1].lstrip('.'); [d:=d.get(p,{}) if isinstance(d,dict) else d for p in k.split('.')]; print(d if d else '')" "$1"; }
else
    echo "FATAL: neither jq nor python3 available for JSON parsing"
    exit 2
fi

# Port availability
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "FATAL: port ${PORT} already in use. Set BIZRA_E2E_PORT or free the port."
    exit 2
fi

# Build gateway if requested + missing
if [ "${BIZRA_E2E_SKIP_BUILD:-0}" != "1" ] && [ ! -x "${GATEWAY_BIN}" ]; then
    info "building gateway release binary..."
    (cd "${REPO_ROOT}/bizra-omega" && cargo build --release -p bizra-cognition-gateway > /dev/null 2>&1)
fi

if [ ! -x "${GATEWAY_BIN}" ]; then
    echo "FATAL: gateway binary missing at ${GATEWAY_BIN}. Build first with: cargo build --release -p bizra-cognition-gateway"
    exit 2
fi

# ─── Start gateway (in-memory; tests 1–7) ───────────────────────────────
info "starting gateway on ${BASE} (log: ${LOG})"
if ! start_gateway "${PORT}" "${LOG}"; then
    echo "FATAL: gateway did not respond to /health within 10s"
    exit 2
fi

pass "gateway boot + /health ok on ${BASE}"

# ─── Test 1: /chain starts empty ──────────────────────────────────────
CHAIN_STATE=$(curl -sf "${BASE}/chain")
LEN=$(echo "${CHAIN_STATE}" | JSON '.length')
if [ "${LEN}" = "0" ]; then
    pass "test 1: /chain starts empty (length=0)"
else
    fail "test 1: expected length=0, got ${LEN}"
fi

# ─── Test 2: POST valid mission — all 5 gates PERMIT ─────────────────
MISSION_REQ=$(cat <<'EOF'
{
  "intent": "G4 polyglot e2e smoke — verify full lawful loop",
  "operatorSessionId": "1111111111111111111111111111111111111111111111111111111111111111",
  "currentState": {
    "hash": "2222222222222222222222222222222222222222222222222222222222222222",
    "summary": "Gateway unreceipted",
    "metric": 0.0
  },
  "idealState": {
    "hash": "3333333333333333333333333333333333333333333333333333333333333333",
    "summary": "Gateway sealed a real receipt",
    "metric": 1.0
  },
  "evidenceHash": "4444444444444444444444444444444444444444444444444444444444444444",
  "qualityScore": 0.98,
  "derivesFromCanonical": true,
  "faceOnly": false
}
EOF
)

MISSION_RES=$(curl -sf -X POST -H "Content-Type: application/json" \
    -d "${MISSION_REQ}" "${BASE}/v1/mission" 2>/dev/null || \
    curl -sf -X POST -H "Content-Type: application/json" \
    -d "${MISSION_REQ}" "${BASE}/mission")

VERDICT=$(echo "${MISSION_RES}" | JSON '.admissibility.verdict' 2>/dev/null || echo "")
RECEIPT_ID=$(echo "${MISSION_RES}" | JSON '.receiptId' 2>/dev/null || echo "")

if [ "${VERDICT}" = "Permit" ]; then
    pass "test 2: POST /mission → Permit verdict (5/5 gates passed)"
else
    fail "test 2: expected verdict=Permit, got '${VERDICT}'"
    echo "    raw response: ${MISSION_RES}" | head -c 300
fi

if [ -n "${RECEIPT_ID}" ] && [ "${#RECEIPT_ID}" -eq 64 ]; then
    pass "test 3: receiptId is a 64-char hex string (${RECEIPT_ID:0:16}…)"
else
    fail "test 3: expected 64-char receiptId, got '${RECEIPT_ID}' (len=${#RECEIPT_ID})"
fi

# ─── Test 4: chain length advanced ────────────────────────────────────
CHAIN_AFTER=$(curl -sf "${BASE}/chain")
LEN_AFTER=$(echo "${CHAIN_AFTER}" | JSON '.length')
if [ "${LEN_AFTER}" != "0" ]; then
    pass "test 4: /chain length advanced after mission (length=${LEN_AFTER})"
else
    fail "test 4: chain did not advance (still length=0)"
fi

# ─── Test 5: receipt retrievable via chain lookup ─────────────────────
if [ -n "${RECEIPT_ID}" ]; then
    RECEIPT=$(curl -sf "${BASE}/chain/${RECEIPT_ID}" 2>/dev/null || echo "")
    if echo "${RECEIPT}" | grep -q '"id"'; then
        pass "test 5: GET /chain/{receipt_id} returns receipt metadata"
    else
        fail "test 5: receipt lookup failed"
    fi
fi

# ─── Test 6: low-quality mission is rejected (IHSAN_FLOOR < 0.95) ────
LOW_REQ=$(echo "${MISSION_REQ}" | sed 's/"qualityScore": 0.98/"qualityScore": 0.50/')
LOW_HTTP=$(curl -s -o /tmp/e2e-low-body.json -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" -d "${LOW_REQ}" "${BASE}/mission" 2>/dev/null || echo "000")
if [ "${LOW_HTTP}" = "422" ]; then
    pass "test 6: low-quality mission fail-closed with HTTP 422 (IHSAN_FLOOR)"
else
    fail "test 6: expected HTTP 422 on low-quality mission, got ${LOW_HTTP}"
fi

# ─── Test 7: unknown hash returns 404 ────────────────────────────────
UNK=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/chain/$(printf 'f%.0s' {1..64})")
if [ "${UNK}" = "404" ]; then
    pass "test 7: unknown hash /chain/{fff...} returns HTTP 404"
else
    fail "test 7: expected HTTP 404 on unknown hash, got ${UNK}"
fi

# ─── Test 8: authoritative store survives restart (Cycle-6 Arc 3) ───
info "test 8: restart persistence via BIZRA_RECEIPT_STORE_PATH"
PERSIST_STORE_DIR="$(mktemp -d /tmp/bizra-e2e-receipt-store.XXXXXX)"
PERSIST_LOG="/tmp/e2e-polyglot-persist.log"

PERSIST_MISSION_REQ=$(cat <<'EOF'
{
  "intent": "G4 Arc3 persist — receipt chain must survive gateway restart",
  "operatorSessionId": "5555555555555555555555555555555555555555555555555555555555555555",
  "currentState": {
    "hash": "6666666666666666666666666666666666666666666666666666666666666666",
    "summary": "Ephemeral boot",
    "metric": 0.0
  },
  "idealState": {
    "hash": "7777777777777777777777777777777777777777777777777777777777777777",
    "summary": "Durable chain sealed",
    "metric": 1.0
  },
  "evidenceHash": "8888888888888888888888888888888888888888888888888888888888888888",
  "qualityScore": 0.98,
  "derivesFromCanonical": true,
  "faceOnly": false
}
EOF
)

if start_gateway "${PORT}" "${PERSIST_LOG}" "${PERSIST_STORE_DIR}"; then
    PERSIST_RES=$(curl -sf -X POST -H "Content-Type: application/json" \
        -d "${PERSIST_MISSION_REQ}" "${BASE}/mission" 2>/dev/null || echo "")
    PERSIST_RECEIPT_ID=$(echo "${PERSIST_RES}" | JSON '.receiptId' 2>/dev/null || echo "")
    PERSIST_LEN=$(curl -sf "${BASE}/chain" | JSON '.length')

    stop_gateway

    if [ -n "${PERSIST_RECEIPT_ID}" ] && [ "${PERSIST_LEN}" != "0" ]; then
        if start_gateway "${PORT}" "${PERSIST_LOG}" "${PERSIST_STORE_DIR}"; then
            LEN_AFTER_RESTART=$(curl -sf "${BASE}/chain" | JSON '.length')
            if [ "${LEN_AFTER_RESTART}" != "0" ]; then
                RECEIPT_AFTER_RESTART=$(curl -sf "${BASE}/chain/${PERSIST_RECEIPT_ID}" 2>/dev/null || echo "")
                if echo "${RECEIPT_AFTER_RESTART}" | grep -q '"id"'; then
                    pass "test 8: receipt chain rehydrates after restart (store=${PERSIST_STORE_DIR})"
                else
                    fail "test 8: receipt lookup failed after restart"
                fi
            else
                fail "test 8: chain length=0 after restart (expected persisted chain)"
            fi
        else
            fail "test 8: gateway failed to boot for restart rehydration"
        fi
    else
        fail "test 8: failed to seal receipt before restart (receiptId='${PERSIST_RECEIPT_ID}', length=${PERSIST_LEN})"
    fi
else
    fail "test 8: persist gateway failed first boot"
fi

# ─── Test 9: operator default token (Arc 3.1+) ───────────────────────
info "test 9: operator smoke — BIZRA_RECEIPT_STORE_PATH=default (isolated)"
stop_gateway
OPERATOR_SMOKE_LOG="/tmp/e2e-operator-smoke-arc3.log"
if BIZRA_OPERATOR_SMOKE_ISOLATED=1 \
    BIZRA_OPERATOR_SMOKE_SKIP_BUILD=1 \
    BIZRA_OPERATOR_SMOKE_PORT=7432 \
    BIZRA_OPERATOR_SMOKE_LOG="${OPERATOR_SMOKE_LOG}" \
    bash "${REPO_ROOT}/scripts/operator-smoke-arc3.sh" > "${OPERATOR_SMOKE_LOG}" 2>&1; then
    pass "test 9: operator default-store smoke (see ${OPERATOR_SMOKE_LOG})"
else
    fail "test 9: operator-smoke-arc3.sh failed (see ${OPERATOR_SMOKE_LOG})"
    tail -20 "${OPERATOR_SMOKE_LOG}" 2>/dev/null | sed 's/^/  | /' || true
fi

# ─── Summary ──────────────────────────────────────────────────────────
echo
echo "═══ RESULTS: ${PASS} passed / ${FAIL} failed ═══"

if [ "${FAIL}" -eq 0 ]; then
    echo -e "${GREEN}G4 GREEN — polyglot lawful loop proven end-to-end.${RESET}"
    echo "الحمد لله."
    exit 0
else
    echo -e "${RED}G4 RED — ${FAIL} assertion(s) failed. See gateway log:${RESET} ${LOG}"
    tail -30 "${LOG}" 2>/dev/null | sed 's/^/  | /'
    exit 1
fi
