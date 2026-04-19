#!/usr/bin/env bash
# fire-dry-run-harness.sh — Cycle-8 Day 11-12
#
# بسم الله الرحمن الرحيم
#
# Operator-side harness that simulates a tester's install + first-run
# flow against locally-built binaries. Produces a JSON run-report that
# can be aggregated across testers.
#
# NOT a replacement for the manual checklist in fire-dry-run-checklist.md
# — this harness runs the AUTOMATED checkpoints. Manual steps (Daughter
# Test, Phase 0 pre-flight, Phase 5 reporting) still need a human.
#
# Per Cycle-8 doctrinal constraint:
#   - Witness-grade only
#   - No tokenomics, no staking, no slashing
#   - Does NOT require a remote witness; witness-peer check is Phase 4
#     and is SKIPPED if $BIZRA_WITNESS_PEER_URL is empty.
#
# Usage:
#   scripts/fire-dry-run-harness.sh [--output <path>] [--fixture-dir <path>]
#
# Env vars respected:
#   BIZRA_DRY_RUN_DEMA_BIN          — path to dema binary (default: built)
#   BIZRA_DRY_RUN_GATEWAY_BIN       — path to gateway binary (default: built)
#   BIZRA_WITNESS_PEER_URL          — witness peer; if empty, Phase 4 SKIP

set -eu

OUTPUT=""
FIXTURE_DIR=""
REPO_ROOT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --fixture-dir) FIXTURE_DIR="$2"; shift 2 ;;
        --help|-h)
            grep "^#" "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -z "$REPO_ROOT" ]; then
    echo "error: run from inside the bizra-data-lake repo" >&2
    exit 3
fi

DEMA_BIN="${BIZRA_DRY_RUN_DEMA_BIN:-$REPO_ROOT/bizra-omega/target/release/dema}"
GATEWAY_BIN="${BIZRA_DRY_RUN_GATEWAY_BIN:-$REPO_ROOT/bizra-omega/target/release/bizra-cognition-gateway}"

if [ -z "$FIXTURE_DIR" ]; then
    FIXTURE_DIR=$(mktemp -d -t bizra-dry-run-fixture-XXXXXX)
    echo "test-file-a" > "$FIXTURE_DIR/a.txt"
    echo "test-file-b" > "$FIXTURE_DIR/b.md"
    mkdir -p "$FIXTURE_DIR/sub"
    echo "cleanup trap set; fixture at $FIXTURE_DIR" >&2
    trap 'rm -rf "$FIXTURE_DIR"' EXIT
fi

# ─── Phase 1 — binary existence & hashing ───────────────────────────
phase_1_binaries() {
    local dema_sha gw_sha
    if [ ! -x "$DEMA_BIN" ]; then
        echo "PHASE_1_FAIL: dema binary not found or not executable at $DEMA_BIN" >&2
        return 1
    fi
    if [ ! -x "$GATEWAY_BIN" ]; then
        echo "PHASE_1_FAIL: gateway binary not found or not executable at $GATEWAY_BIN" >&2
        return 1
    fi
    dema_sha=$(sha256sum "$DEMA_BIN" | awk '{print $1}')
    gw_sha=$(sha256sum "$GATEWAY_BIN" | awk '{print $1}')
    echo "$dema_sha $gw_sha"
}

# ─── Phase 2 — start gateway (tmp cache), run dema organize ─────────
phase_2_organize() {
    local cache_root
    cache_root=$(mktemp -d -t bizra-dry-run-cache-XXXXXX)

    # Start gateway in background with a fresh cache.
    BIZRA_DEMA_CACHE_ROOT="$cache_root" \
        BIZRA_IDENTITY_ANCHOR="$cache_root/identity/credentials.json" \
        BIZRA_COGNITION_PORT="7442" \
        nohup "$GATEWAY_BIN" > "$cache_root/gateway.log" 2>&1 &
    local gw_pid=$!

    # Wait for listener.
    local attempts=0
    while [ "$attempts" -lt 20 ]; do
        if curl -sS -m 1 http://127.0.0.1:7442/health > /dev/null 2>&1; then
            break
        fi
        sleep 0.25
        attempts=$((attempts + 1))
    done
    if [ "$attempts" -ge 20 ]; then
        kill -9 "$gw_pid" 2>/dev/null || true
        rm -rf "$cache_root"
        echo "PHASE_2_FAIL: gateway did not bind within 5s" >&2
        return 1
    fi

    # Register the fixture as allowlisted.
    BIZRA_DEMA_CACHE_ROOT="$cache_root" \
        BIZRA_IDENTITY_ANCHOR="$cache_root/identity/credentials.json" \
        BIZRA_GATEWAY_URL="http://127.0.0.1:7442" \
        "$DEMA_BIN" register-resource --kind filesystem --id "$FIXTURE_DIR" --allowlisted > /dev/null 2>&1

    # Run organize.
    local organize_out
    organize_out=$(BIZRA_DEMA_CACHE_ROOT="$cache_root" \
        BIZRA_IDENTITY_ANCHOR="$cache_root/identity/credentials.json" \
        BIZRA_GATEWAY_URL="http://127.0.0.1:7442" \
        "$DEMA_BIN" organize "$FIXTURE_DIR" 2>&1 || true)

    # Cleanup
    kill -9 "$gw_pid" 2>/dev/null || true
    wait "$gw_pid" 2>/dev/null || true

    # Extract chain_head from output.
    local chain_head
    chain_head=$(echo "$organize_out" | grep -oE 'chain_head:\s*[a-f0-9]{64}' | awk '{print $2}' | head -1)

    if [ -z "$chain_head" ]; then
        echo "PHASE_2_FAIL: organize did not emit chain_head" >&2
        echo "---full output---" >&2
        echo "$organize_out" >&2
        rm -rf "$cache_root"
        return 1
    fi

    rm -rf "$cache_root"
    echo "$chain_head"
}

# ─── Phase 3 — witness probe (SKIP if no peer configured) ───────────
phase_3_witness() {
    local peer="${BIZRA_WITNESS_PEER_URL:-}"
    if [ -z "$peer" ]; then
        echo "SKIP: BIZRA_WITNESS_PEER_URL not set"
        return 0
    fi
    local resp
    resp=$(curl -sS -m 5 "$peer/witness/head/node0" 2>&1 || true)
    if echo "$resp" | grep -q "chain_head_hex"; then
        echo "OK: witness reachable, observation retrieved"
    else
        echo "FAIL: witness at $peer did not return valid observation" >&2
        return 1
    fi
}

# ─── Run all phases ─────────────────────────────────────────────────
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_NS=$(date -u +%s%N)

PHASE_1_RESULT="SKIP"
PHASE_2_RESULT="SKIP"
PHASE_3_RESULT="SKIP"
DEMA_SHA=""
GW_SHA=""
CHAIN_HEAD=""
OVERALL="PENDING"

if HASHES=$(phase_1_binaries); then
    DEMA_SHA=$(echo "$HASHES" | awk '{print $1}')
    GW_SHA=$(echo "$HASHES" | awk '{print $2}')
    PHASE_1_RESULT="PASS"
else
    PHASE_1_RESULT="FAIL"
    OVERALL="FAIL"
fi

if [ "$PHASE_1_RESULT" = "PASS" ]; then
    if HEAD=$(phase_2_organize); then
        CHAIN_HEAD="$HEAD"
        PHASE_2_RESULT="PASS"
    else
        PHASE_2_RESULT="FAIL"
        OVERALL="FAIL"
    fi
fi

if [ "$OVERALL" != "FAIL" ]; then
    if W=$(phase_3_witness); then
        PHASE_3_RESULT=$(echo "$W" | head -1 | awk '{print $1}' | tr -d ':')
    else
        PHASE_3_RESULT="FAIL"
    fi
    OVERALL="PASS"
fi

END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
END_NS=$(date -u +%s%N)
DURATION_MS=$(( (END_NS - START_NS) / 1000000 ))

REPORT=$(cat <<EOF
{
  "schema": "bizra-fire-dry-run-v1",
  "overall": "$OVERALL",
  "phase_1_binaries": "$PHASE_1_RESULT",
  "phase_2_organize": "$PHASE_2_RESULT",
  "phase_3_witness": "$PHASE_3_RESULT",
  "dema_sha256": "$DEMA_SHA",
  "gateway_sha256": "$GW_SHA",
  "chain_head_hex": "$CHAIN_HEAD",
  "started_at_iso": "$START_ISO",
  "ended_at_iso": "$END_ISO",
  "duration_ms": $DURATION_MS,
  "fixture_dir": "$FIXTURE_DIR",
  "dema_bin": "$DEMA_BIN",
  "gateway_bin": "$GATEWAY_BIN"
}
EOF
)

if [ -n "$OUTPUT" ]; then
    printf '%s\n' "$REPORT" > "$OUTPUT"
else
    printf '%s\n' "$REPORT"
fi

[ "$OVERALL" = "PASS" ] && exit 0 || exit 1
