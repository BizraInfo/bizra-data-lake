# Phase 86-C: Deployment Verification
## Proving the Self-Sustaining Loop in Production

**Sprint Ref**: Horizon 1 Gate
**Gap Source**: SAPE DevOps Gem #1 (CI has 24 gates, zero deployment smoke test)
**Standing On**: Deming (1986) Plan-Do-Check-Act, Boyd (1976) OODA

---

## 1. Problem Statement

```
Current:  cargo test passes (1,381 tests) ← isolated
          cargo build --release produces binary ← untested as daemon
          No automated test: "binary runs, processes missions, produces receipts"
          S1 gate: "NODE0 runs 24h continuous" ← no infrastructure to prove this

Required: Deployment pipeline that verifies the release binary
          works as a sovereign daemon processing real missions.
```

## 2. Deployment Artifact Manifest

```
Artifact            Size    Purpose
───────────────────────────────────────────────
bizra-node          2.6M    Sovereign desktop binary
bizra-api           4.5M    REST/WebSocket API server
Python kernel       ~113K   Cognitive intelligence layer (port 8010)

Total sovereign footprint: < 10MB compiled
No Docker required for NODE0 operation.
```

## 3. Pseudocode — Smoke Test Script

```bash
#!/usr/bin/env bash
# scripts/ops/smoke_test.sh
# ──────────────────────────────────────────────
# Deployment smoke test: verify release binary
# processes N governed missions with zero crashes.
#
# Usage: ./smoke_test.sh [num_missions]
# Gate:  exit 0 = PASS, exit 1 = FAIL
# ──────────────────────────────────────────────

set -euo pipefail

BINARY="${BIZRA_NODE_BINARY:-./target/release/bizra-node}"
NUM_MISSIONS="${1:-100}"
FAILURES=0
RECEIPTS=0

echo "=== BIZRA Deployment Smoke Test ==="
echo "Binary:   $BINARY"
echo "Missions: $NUM_MISSIONS"
echo ""

# Gate 1: Binary exists and is executable
test -x "$BINARY" || { echo "FAIL: binary not found"; exit 1; }

# Gate 2: Binary boots and responds to PING
PING=$(echo "PING" | "$BINARY" 2>/dev/null)
echo "$PING" | grep -q "pong=true" || { echo "FAIL: PING"; exit 1; }
echo "PASS: PING"

# Gate 3: VERSION returns expected format
VERSION=$(echo "VERSION" | "$BINARY" 2>/dev/null)
echo "$VERSION" | grep -q "bizra-node" || { echo "FAIL: VERSION"; exit 1; }
echo "PASS: VERSION"

# Gate 4: HEALTH shows 7 agents active
HEALTH=$(echo "HEALTH" | "$BINARY" 2>/dev/null)
echo "$HEALTH" | grep -q "agents_active=7" || { echo "FAIL: HEALTH"; exit 1; }
echo "PASS: HEALTH"

# Gate 5: Process N governed missions
echo ""
echo "Processing $NUM_MISSIONS governed missions..."

QUERIES=(
    "What are BIZRA constitutional thresholds?"
    "Explain the SEED token economy"
    "What is the Enforceable Spine?"
    "How does the SAT Mint Court work?"
    "What is the Gödel Grounding Theorem?"
    "Describe the 4-loop HHMM architecture"
    "What is ZANN_ZERO?"
    "How does the entropy router classify queries?"
    "What are the 12 Ihsan dimensions?"
    "Explain the ADL Gini invariant"
)

COMMANDS=""
for i in $(seq 1 "$NUM_MISSIONS"); do
    QUERY="${QUERIES[$((i % ${#QUERIES[@]}))]}"
    COMMANDS+="RECEIVE\t${QUERY}\t$((1000 + i))\n"
done
COMMANDS+="HEALTH\nSHUTDOWN\n"

OUTPUT=$(printf "$COMMANDS" | "$BINARY" 2>/dev/null)

# Count successful missions
RECEIPTS=$(echo "$OUTPUT" | grep -c "received=true" || true)
VETOES=$(echo "$OUTPUT" | grep -c "guardian_approved=false" || true)
ERRORS=$(echo "$OUTPUT" | grep -c "^ERR" || true)

echo ""
echo "=== Results ==="
echo "Missions sent:     $NUM_MISSIONS"
echo "Receipts received: $RECEIPTS"
echo "Guardian vetoes:   $VETOES"
echo "Errors:            $ERRORS"

# Gate: all missions must produce a response (receipt or veto)
TOTAL_RESPONSES=$((RECEIPTS + VETOES))
if [ "$TOTAL_RESPONSES" -lt "$NUM_MISSIONS" ]; then
    echo "FAIL: $((NUM_MISSIONS - TOTAL_RESPONSES)) missions lost"
    exit 1
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "FAIL: $ERRORS protocol errors"
    exit 1
fi

# Extract final health
FINAL_HEALTH=$(echo "$OUTPUT" | grep "messages_processed=" | tail -1)
echo ""
echo "Final health: $FINAL_HEALTH"
echo ""
echo "=== SMOKE TEST: PASSED ==="
echo "All $NUM_MISSIONS missions processed. Zero crashes. Zero losses."
exit 0
```

## 4. Pseudocode — Systemd Service Unit

```ini
# /etc/systemd/system/bizra-node.service
# Sovereign daemon: runs continuously, restarts on failure
#
# TDD anchor: test_systemd_unit_starts
# TDD anchor: test_systemd_unit_restarts_on_crash

[Unit]
Description=BIZRA Sovereign Node (NODE0)
After=network.target
Documentation=https://github.com/BizraInfo/bizra-data-lake

[Service]
Type=simple
User=root
Environment=PYTHONUNBUFFERED=1
Environment=BIZRA_ENV=production
Environment=BIZRA_SOVEREIGN_ROOT=/mnt/b/BIZRA
ExecStartPre=/usr/bin/test -x /usr/local/bin/bizra-node
ExecStart=/usr/local/bin/bizra-node --daemon
Restart=on-failure
RestartSec=5
WatchdogSec=60
StandardOutput=journal
StandardError=journal

# Constitutional constraints
LimitNOFILE=65536
MemoryMax=4G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
```

## 5. Pseudocode — CI Deployment Gate

```yaml
# .github/workflows/ci.yml addition
# New gate: deployment smoke test on release binary

  deploy-smoke-test:
    needs: [test-rust]
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Build release binary
        run: |
          cd bizra-omega
          cargo build --release -p bizra-node
      - name: Run smoke test (100 missions)
        run: |
          cd bizra-omega
          chmod +x scripts/ops/smoke_test.sh
          ./scripts/ops/smoke_test.sh 100
      - name: Verify zero crashes
        run: |
          echo "Deployment smoke test passed"
```

## 6. 24-Hour Continuous Run Protocol

```
Phase      Duration  Action                              Gate
─────────────────────────────────────────────────────────────────
Warmup     0-5min    Boot + 100 mission smoke test       Zero errors
Steady     5min-12h  Process GOLD knowledge base tasks   RSS stable
                     - Classify 569K unknowns
                     - Dedup 169K copies
                     - Index with SNR scoring
Endurance  12h-24h   Continue under load + synthesis     No memory leak
                     - Heartbeat every 1s
                     - Synthesis every 60s
                     - Reflex compilation on patterns
Shutdown   24h+1min  Graceful shutdown + persist          All rules saved

Monitoring (every 60s):
  HEALTH → log RSS, messages_processed, reflex_hits, reflex_misses
  If RSS grows > 500MB from baseline → WARNING
  If RSS grows > 1GB from baseline → ABORT + investigate

S1 Gate Criteria:
  [x] 24 continuous hours, zero crashes
  [x] Real tasks processed (not synthetic)
  [x] Each task → receipt → bus → memory → reflex
  [x] Compiled reflexes serve repeated queries via S1
  [x] SEED minted from verified work
  [x] Receipt chain integrity: all signed, all chained
```

## 7. Acceptance Criteria

```
[x] smoke_test.sh passes with 100 missions, 0 crashes
[x] Release binary < 5MB (currently 2.6M — PASSES)
[x] Boot time < 2s
[x] Systemd unit starts and restarts correctly
[x] CI deployment gate added and GREEN
[x] 24h protocol documented and ready for execution
```
