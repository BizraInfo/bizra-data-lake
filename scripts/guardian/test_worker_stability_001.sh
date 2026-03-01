#!/usr/bin/env bash
# WORKER-STABILITY-001 — Acceptance test for scoped audit worker
#
# Acceptance criteria (from root cause analysis 2026-02-27):
#   AC1: Scoped audit run completes < 300s
#   AC2: Output is non-empty (contains JSON vulnerability report)
#   AC3: durationMs in result log ≈ actual wall time (no 10,000s+ anomaly)
#   AC4: No runaway process after completion (no orphaned audit PID)
#
# Usage:
#   bash scripts/guardian/test_worker_stability_001.sh
#
# Requires: npx in PATH, ANTHROPIC_API_KEY or LM_API_TOKEN set

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/.claude-flow/logs/headless"
TIMEOUT_SECS=300
PASS=0
FAIL=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { log "✅ PASS: $*"; PASS=$((PASS + 1)); }
fail() { log "❌ FAIL: $*"; FAIL=$((FAIL + 1)); }

# ── Pre-flight ──────────────────────────────────────────────────────────────
log "=== WORKER-STABILITY-001 Acceptance Test ==="
log "Repo: $REPO_ROOT"
log "Timeout budget: ${TIMEOUT_SECS}s"
echo ""

# Verify audit is not currently marked as running (ghost state check)
GHOST=$(python3 -c "
import json, sys
with open('$REPO_ROOT/.claude-flow/daemon-state.json') as f:
    d = json.load(f)
print(d['workers']['audit']['isRunning'])
" 2>/dev/null || echo "unknown")
if [[ "$GHOST" == "False" || "$GHOST" == "false" ]]; then
    pass "AC4-pre: audit.isRunning = false (no ghost state)"
else
    fail "AC4-pre: audit.isRunning = $GHOST (ghost state detected before run)"
fi

# ── Trigger one scoped audit run ────────────────────────────────────────────
log ""
log "Dispatching scoped audit worker..."
START_TS=$(date +%s)

# Count log files before
BEFORE_COUNT=$(ls "$LOG_DIR"/audit_*_result.log 2>/dev/null | wc -l)

# Dispatch the audit worker (foreground, captured)
timeout "$TIMEOUT_SECS" npx claude-flow@alpha swarm --trigger audit \
    --scope "core/governance,core/pci,core/proof_engine" \
    --maxFiles 60 --maxChars 120000 2>&1 | tee /tmp/audit_test_output.txt || true

END_TS=$(date +%s)
WALL_SECS=$((END_TS - START_TS))

log "Wall time: ${WALL_SECS}s"

# ── AC1: completed within 300s ───────────────────────────────────────────────
if [[ "$WALL_SECS" -lt "$TIMEOUT_SECS" ]]; then
    pass "AC1: completed in ${WALL_SECS}s < ${TIMEOUT_SECS}s"
else
    fail "AC1: ran for ${WALL_SECS}s — still exceeds ${TIMEOUT_SECS}s budget"
fi

# ── AC2: output is non-empty ─────────────────────────────────────────────────
AFTER_COUNT=$(ls "$LOG_DIR"/audit_*_result.log 2>/dev/null | wc -l)
LATEST_RESULT=$(ls -t "$LOG_DIR"/audit_*_result.log 2>/dev/null | head -1)

if [[ -n "$LATEST_RESULT" && "$AFTER_COUNT" -gt "$BEFORE_COUNT" ]]; then
    SUCCESS_FLAG=$(python3 -c "
import json
with open('$LATEST_RESULT') as f:
    content = f.read()
# Find JSON block after the header
import re
m = re.search(r'\{.*\}', content, re.DOTALL)
if m:
    d = json.loads(m.group())
    print(d.get('success', False))
else:
    print(False)
" 2>/dev/null || echo "false")

    if [[ "$SUCCESS_FLAG" == "True" || "$SUCCESS_FLAG" == "true" ]]; then
        pass "AC2: result log written, success=true, output non-empty"
    else
        fail "AC2: result log written but success=false — see $LATEST_RESULT"
        cat "$LATEST_RESULT"
    fi
else
    fail "AC2: no new result log written — worker may not have run"
fi

# ── AC3: durationMs ≈ wall time (no runaway) ──────────────────────────────────
if [[ -n "$LATEST_RESULT" ]]; then
    DURATION_MS=$(python3 -c "
import json, re
with open('$LATEST_RESULT') as f:
    content = f.read()
m = re.search(r'\{.*\}', content, re.DOTALL)
if m:
    d = json.loads(m.group())
    print(d.get('durationMs', -1))
else:
    print(-1)
" 2>/dev/null || echo "-1")

    DURATION_SECS=$((${DURATION_MS%.*} / 1000))
    DELTA=$((WALL_SECS - DURATION_SECS))
    if [[ "$DELTA" -lt 0 ]]; then DELTA=$((-DELTA)); fi

    if [[ "$DURATION_SECS" -lt "$TIMEOUT_SECS" && "$DELTA" -lt 30 ]]; then
        pass "AC3: durationMs=${DURATION_MS}ms ≈ wall=${WALL_SECS}s (delta=${DELTA}s < 30s)"
    else
        fail "AC3: durationMs=${DURATION_MS}ms vs wall=${WALL_SECS}s — anomaly detected (delta=${DELTA}s)"
    fi
fi

# ── AC4: no orphaned audit process ───────────────────────────────────────────
ORPHAN=$(pgrep -f "claude.*audit\|audit.*worker" 2>/dev/null | head -3 || true)
if [[ -z "$ORPHAN" ]]; then
    pass "AC4: no orphaned audit process found"
else
    fail "AC4: orphaned process(es) detected: $ORPHAN"
    kill -9 $ORPHAN 2>/dev/null || true
    log "  Force-killed orphaned PIDs: $ORPHAN"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
log "=== Results: ${PASS} PASS / ${FAIL} FAIL ==="
if [[ "$FAIL" -eq 0 ]]; then
    log "✅ WORKER-STABILITY-001: ALL CRITERIA MET — promote to WORKER-STABILITY-002"
    exit 0
else
    log "❌ WORKER-STABILITY-001: ${FAIL} criteria failed — do NOT promote to WORKER-STABILITY-002"
    exit 1
fi
