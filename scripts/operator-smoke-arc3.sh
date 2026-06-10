#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# BIZRA Cycle-6 Arc 3 — Operator Golden Path Smoke Test
#
# Proves authoritative receipt persistence for operators:
#   BIZRA_RECEIPT_STORE_PATH=default (or explicit path) survives gateway restart.
#
# Hardened against sled single-writer lock failures:
#   - exclusive stop → wait → sleep → start lifecycle (mirrors e2e-polyglot)
#   - readiness via GET /health (not /chain)
#   - trap cleanup on EXIT/INT/TERM
#
# Exit codes:
#   0 = smoke passed
#   1 = assertion failure (chain did not survive restart)
#   2 = infrastructure failure (build, boot, port conflict)
#
# Environment:
#   BIZRA_OPERATOR_SMOKE_PORT        (default: 7439)
#   BIZRA_RECEIPT_STORE_PATH         (default: default — operator token)
#   BIZRA_DATA_LAKE_ROOT             (default: /data/bizra)
#   BIZRA_OPERATOR_SMOKE_ISOLATED=1  use temp DATA_LAKE_ROOT (no shared store)
#   BIZRA_OPERATOR_SMOKE_SKIP_BUILD=1 skip cargo build
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

PORT="${BIZRA_OPERATOR_SMOKE_PORT:-7439}"
BASE="http://127.0.0.1:${PORT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_BIN="${REPO_ROOT}/bizra-omega/target/release/bizra-cognition-gateway"
LOG="${BIZRA_OPERATOR_SMOKE_LOG:-/tmp/operator-smoke-arc3-gateway.log}"
STORE_PATH="${BIZRA_RECEIPT_STORE_PATH:-default}"
DATA_LAKE_ROOT="${BIZRA_DATA_LAKE_ROOT:-/data/bizra}"
ISOLATED_ROOT=""

RED='\033[0;31m'
GREEN='\033[0;32m'
GOLD='\033[0;33m'
RESET='\033[0m'

GATEWAY_PID=""

json_field() {
    python3 -c "import sys,json; d=json.load(sys.stdin); k=sys.argv[1].lstrip('.'); [d:=d.get(p,{}) if isinstance(d,dict) else d for p in k.split('.')]; print(d if d else '')" "$1"
}

stop_gateway() {
    if [ -n "${GATEWAY_PID}" ] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
        kill "${GATEWAY_PID}" 2>/dev/null || true
        wait "${GATEWAY_PID}" 2>/dev/null || true
    fi
    GATEWAY_PID=""
    sleep 2
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
    stop_gateway
    : > "${log}"
    BIZRA_COGNITION_PORT="${port}" \
    BIZRA_RECEIPT_STORE_PATH="${STORE_PATH}" \
    BIZRA_DATA_LAKE_ROOT="${DATA_LAKE_ROOT}" \
        "${GATEWAY_BIN}" >> "${log}" 2>&1 &
    GATEWAY_PID=$!
    wait_gateway_health "http://127.0.0.1:${port}" "${log}"
}

cleanup() {
    stop_gateway
    if [ -n "${ISOLATED_ROOT}" ] && [ -d "${ISOLATED_ROOT}" ]; then
        rm -rf "${ISOLATED_ROOT}"
    fi
}
trap cleanup EXIT INT TERM

info() { echo -e "  ${GOLD}▶${RESET} $1"; }
pass() { echo -e "  ${GREEN}✓${RESET} $1"; }
fail() { echo -e "  ${RED}✗${RESET} $1"; exit 1; }

echo "═══ BIZRA Arc 3 operator smoke — $(date -u +%Y-%m-%dT%H:%M:%SZ) ═══"
echo "  main: $(cd "${REPO_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [ "${BIZRA_OPERATOR_SMOKE_ISOLATED:-0}" = "1" ]; then
    ISOLATED_ROOT="$(mktemp -d /tmp/bizra-operator-smoke-root.XXXXXX)"
    DATA_LAKE_ROOT="${ISOLATED_ROOT}"
    info "isolated DATA_LAKE_ROOT=${DATA_LAKE_ROOT}"
fi

if [ "${STORE_PATH}" = "default" ]; then
    RESOLVED_STORE="${DATA_LAKE_ROOT}/sovereign_state/authoritative_receipt_store"
else
    RESOLVED_STORE="${STORE_PATH}"
fi
info "store path mode: BIZRA_RECEIPT_STORE_PATH=${STORE_PATH}"
info "resolved store: ${RESOLVED_STORE}"

if [ "${BIZRA_OPERATOR_SMOKE_SKIP_BUILD:-0}" != "1" ]; then
    info "building bizra-cognition-gateway (release)"
    if ! (cd "${REPO_ROOT}/bizra-omega" && cargo build --release -p bizra-cognition-gateway); then
        echo "FATAL: cargo build failed" >&2
        exit 2
    fi
fi

if [ ! -x "${GATEWAY_BIN}" ]; then
    echo "FATAL: gateway binary missing: ${GATEWAY_BIN}" >&2
    exit 2
fi

MISSION='{
  "intent": "Arc 3 operator smoke — restart persistence",
  "operatorSessionId": "1111111111111111111111111111111111111111111111111111111111111111",
  "currentState": {
    "hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "summary": "Smoke boot",
    "metric": 0.0
  },
  "idealState": {
    "hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "summary": "Smoke durable",
    "metric": 1.0
  },
  "evidenceHash": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "qualityScore": 0.98,
  "derivesFromCanonical": true,
  "faceOnly": false
}'

info "boot gateway (first start)"
start_gateway "${PORT}" "${LOG}" || fail "gateway failed first boot (check sled lock / port ${PORT})"

MISSION_RES=$(curl -sf -X POST "${BASE}/mission" \
    -H "Content-Type: application/json" \
    -d "${MISSION}") || fail "POST /mission failed"

RECEIPT_ID=$(echo "${MISSION_RES}" | json_field '.receiptId')
if [ -z "${RECEIPT_ID}" ]; then
    fail "mission response missing receiptId: ${MISSION_RES}"
fi

CHAIN_BEFORE=$(curl -sf "${BASE}/chain")
BEFORE_LEN=$(echo "${CHAIN_BEFORE}" | json_field '.length')
BEFORE_HEAD=$(echo "${CHAIN_BEFORE}" | json_field '.head')

info "chain before restart: length=${BEFORE_LEN} head=${BEFORE_HEAD:0:16}..."

stop_gateway

info "boot gateway (restart)"
start_gateway "${PORT}" "${LOG}" || fail "gateway failed restart boot (sled lock? wait for prior process exit)"

CHAIN_AFTER=$(curl -sf "${BASE}/chain")
AFTER_LEN=$(echo "${CHAIN_AFTER}" | json_field '.length')
AFTER_HEAD=$(echo "${CHAIN_AFTER}" | json_field '.head')

RECEIPT_AFTER=$(curl -sf "${BASE}/chain/${RECEIPT_ID}" 2>/dev/null || echo "")

info "chain after restart: length=${AFTER_LEN} head=${AFTER_HEAD:0:16}..."

if [ "${BEFORE_LEN}" != "${AFTER_LEN}" ] || [ "${BEFORE_LEN}" -eq 0 ]; then
    fail "chain length mismatch or zero (before=${BEFORE_LEN} after=${AFTER_LEN})"
fi

if [ "${BEFORE_HEAD}" != "${AFTER_HEAD}" ]; then
    fail "chain head changed across restart"
fi

if ! echo "${RECEIPT_AFTER}" | grep -q '"id"'; then
    fail "receipt lookup failed after restart for ${RECEIPT_ID}"
fi

pass "chain survived restart (length=${AFTER_LEN}, receipt=${RECEIPT_ID:0:16}...)"
echo
echo -e "${GREEN}Arc 3 operator smoke GREEN.${RESET}"
echo "  store: ${RESOLVED_STORE}"
echo "  log:   ${LOG}"
exit 0
