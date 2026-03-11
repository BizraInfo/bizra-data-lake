#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# BIZRA Node0 Genesis Ceremony — 19 Hard Gates + 13 Supporting Checks
# ═══════════════════════════════════════════════════════════════════════
#
# Reads canonical truth from:
#   sovereign_state/node0_lifecycle.json  (lifecycle v2 gates)
#   sovereign_state/node0_genesis.json    (authority)
#   sovereign_state/genesis_hash.txt      (authority hash)
#
# Usage:
#   bash scripts/node0_genesis_ceremony.sh           # Hard gates only
#   bash scripts/node0_genesis_ceremony.sh --full     # Hard + supporting
#   bash scripts/node0_genesis_ceremony.sh --json     # JSON receipt
#
# Standing on: Nakamoto, Lamport, Deming, Al-Ghazali
# ═══════════════════════════════════════════════════════════════════════

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv-linux/bin/activate ]; then
    source .venv-linux/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

MODE="${1:-hard}"
PASS=0
FAIL=0
TOTAL=0
FAILED_GATES=""
S_PASS=0
S_FAIL=0
S_TOTAL=0

run_gate() {
    local id="$1" name="$2" cmd="$3"
    TOTAL=$((TOTAL + 1))
    local result
    result=$(eval "$cmd" 2>&1) || result="ERROR: $result"
    if echo "$result" | grep -q "PASS"; then
        [ "$MODE" != "--json" ] && echo "  ✓ [$id] $name"
        PASS=$((PASS + 1))
    else
        [ "$MODE" != "--json" ] && echo "  ✗ [$id] $name"
        FAIL=$((FAIL + 1))
        FAILED_GATES="$FAILED_GATES $id"
    fi
}

run_support() {
    local id="$1" name="$2" cmd="$3"
    S_TOTAL=$((S_TOTAL + 1))
    local result
    result=$(eval "$cmd" 2>&1) || result="ERROR"
    if echo "$result" | grep -q "PASS"; then
        [ "$MODE" != "--json" ] && echo "  · [$id] $name"
        S_PASS=$((S_PASS + 1))
    else
        [ "$MODE" != "--json" ] && echo "  ? [$id] $name (non-blocking)"
        S_FAIL=$((S_FAIL + 1))
    fi
}

# ═══ Helper: read lifecycle gate from JSON ═══
lc_gate() {
    python3 -c "
import json
lc = json.load(open('sovereign_state/node0_lifecycle.json'))
val = lc.get('gates', {}).get('$1', False)
print('PASS' if val else 'FAIL')
"
}

lc_field() {
    python3 -c "
import json
lc = json.load(open('sovereign_state/node0_lifecycle.json'))
print(lc.get('$1', ''))
"
}

[ "$MODE" != "--json" ] && {
    echo "═══════════════════════════════════════════════════════════"
    echo " BIZRA Node0 Genesis Ceremony"
    echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo " Truth: sovereign_state/node0_lifecycle.json"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "HARD GATES (19)"
    echo ""
    echo "Layer 1: GENESIS INTEGRITY"
}

# ═══ L1: Genesis Integrity (4 gates) ═══

run_gate "1.1" "Canonical manifest valid" \
    "python3 -c \"import json; d=json.load(open('sovereign_state/node0_genesis.json')); assert 'identity' in d; print('PASS')\""

run_gate "1.2" "Genesis hash anchored" \
    "test -s sovereign_state/genesis_hash.txt && echo PASS || echo FAIL"

run_gate "1.3" "Identity ready (lifecycle)" \
    "$(lc_gate identity_ready)"

run_gate "1.4" "Authority resolves" \
    "python3 -c \"from pathlib import Path; from core.sovereign.node0_authority import resolve_authority; r=resolve_authority(Path('sovereign_state'), Path('.')); print('PASS' if r else 'FAIL')\""

# ═══ L2: Personal Sovereign Activation (6 gates) ═══

[ "$MODE" != "--json" ] && echo "" && echo "Layer 2: PERSONAL SOVEREIGN ACTIVATION"

run_gate "2.1" "PAT count = 7" \
    "python3 -c \"from core.integration.constants import PAT_AGENT_COUNT; assert PAT_AGENT_COUNT==7; print('PASS')\""

run_gate "2.2" "PAT roles = 7" \
    "python3 -c \"from core.proof_engine.genesis_ceremony import PAT_ROLES; assert len(PAT_ROLES)==7; print('PASS')\""

run_gate "2.3" "SAT count = 5" \
    "python3 -c \"from core.integration.constants import SAT_AGENTS_PER_NODE; assert SAT_AGENTS_PER_NODE==5; print('PASS')\""

run_gate "2.4" "SAT roles = 5" \
    "python3 -c \"from core.proof_engine.genesis_ceremony import SAT_ROLES; assert len(SAT_ROLES)==5; print('PASS')\""

run_gate "2.5" "PAT/SAT ready (lifecycle)" \
    "$(lc_gate pat_sat_ready)"

run_gate "2.6" "MOE routing functional" \
    "python3 -c \"from core.living_model.moe_engine import MOEEngine; e=MOEEngine(); r=e.route('explain quantum computing'); assert len(r)>0; print('PASS')\""

# ═══ L3: Stand-Alone MVSA (4 gates) ═══

[ "$MODE" != "--json" ] && echo "" && echo "Layer 3: STAND-ALONE CAPABILITY (MVSA)"

run_gate "3.1" "MVSA self-validation (lifecycle)" \
    "$(lc_gate mvsa_self_validation_ok)"

run_gate "3.2" "Ihsan = 0.95" \
    "python3 -c \"from core.integration.constants import UNIFIED_IHSAN_THRESHOLD; assert UNIFIED_IHSAN_THRESHOLD==0.95; print('PASS')\""

run_gate "3.3" "SNR = 0.85" \
    "python3 -c \"from core.integration.constants import UNIFIED_SNR_THRESHOLD; assert UNIFIED_SNR_THRESHOLD==0.85; print('PASS')\""

run_gate "3.4" "Gini = 0.35" \
    "python3 -c \"from core.integration.constants import ADL_GINI_THRESHOLD; assert ADL_GINI_THRESHOLD==0.35; print('PASS')\""

# ═══ L4: Device Consecration (3 gates) ═══

[ "$MODE" != "--json" ] && echo "" && echo "Layer 4: DEVICE CONSECRATION"

run_gate "4.1" "State directory exists" \
    "test -d sovereign_state && echo PASS || echo FAIL"

run_gate "4.2" "Lifecycle ready" \
    "python3 -c \"import json; lc=json.load(open('sovereign_state/node0_lifecycle.json')); print('PASS' if lc.get('status')=='ready' else 'FAIL')\""

run_gate "4.3" "Lifecycle schema v2" \
    "python3 -c \"import json; lc=json.load(open('sovereign_state/node0_lifecycle.json')); assert lc.get('schema_version')=='2.0.0'; print('PASS')\""

# ═══ L5: Replication Readiness (2 gates) ═══

[ "$MODE" != "--json" ] && echo "" && echo "Layer 5: REPLICATION READINESS"

run_gate "5.1" "Ceremony deterministic" \
    "python3 -c \"from core.proof_engine.genesis_ceremony import run_ceremony, CeremonyConfig; c=CeremonyConfig(timestamp_ms=1000000); r1=run_ceremony(b'det-seed', c); r2=run_ceremony(b'det-seed', c); assert r1.genesis_hash==r2.genesis_hash; print('PASS')\""

run_gate "5.2" "Cross-repo sync" \
    "python3 -c \"from core.integration.constants import validate_cross_repo_consistency; r=validate_cross_repo_consistency(); print('PASS' if r else 'FAIL')\""

# ═══ Supporting checks (only with --full) ═══

if [ "$MODE" = "--full" ]; then
    echo ""
    echo "SUPPORTING CHECKS (13)"
    echo ""

    run_support "S.1" "00_GENESIS block0" \
        "python3 -c \"import json; d=json.load(open('00_GENESIS/genesis.json')); gb=d.get('genesis_block',{}); assert gb.get('block_number',0)==0; print('PASS')\""
    run_support "S.2" "Ceremony signer key" \
        "test -s sovereign_state/ceremony_signer.key && echo PASS || echo FAIL"
    run_support "S.3" "PAT/SAT manifests" \
        "python3 -c \"import json; json.load(open('sovereign_state/genesis/pat_manifest.json')); json.load(open('sovereign_state/genesis/sat_manifest.json')); print('PASS')\""
    run_support "S.4" "PBFT consensus" \
        "python3 -c \"from core.federation.consensus import ConsensusEngine; print('PASS')\""
    run_support "S.5" "Federation gossip" \
        "python3 -c \"from core.federation import gossip; print('PASS')\""
    run_support "S.6" "Receipt schema" \
        "python3 -c \"from core.proof_engine.receipt import Receipt, ReceiptStatus; print('PASS')\""
    run_support "S.7" "Atomic I/O" \
        "python3 -c \"from core.sovereign.atomic_io import atomic_write_json, read_json; print('PASS')\""
    run_support "S.8" "Evidence ledger" \
        "python3 -c \"from core.proof_engine.evidence_ledger import EvidenceLedger; print('PASS')\""
    run_support "S.9" "Mission orchestrator" \
        "python3 -c \"from core.sovereign.mission import MissionOrchestrator; print('PASS')\""
    run_support "S.10" "Baseline recorded" \
        "test -s sovereign_state/node0_baseline.json && echo PASS || echo FAIL"
    run_support "S.11" "NodeTemplate" \
        "python3 -c \"from core.pat.onboarding import NodeTemplate; t=NodeTemplate.default(); print('PASS')\""
    run_support "S.12" "CLI entry point" \
        "python3 bizra_cli.py --help 2>&1 | grep -qi 'bizra\|usage' && echo PASS || echo FAIL"
    run_support "S.13" "Upgrade path" \
        "test -f docs/GENESIS_ROADMAP.md && echo PASS || echo FAIL"
fi

# ═══ Results ═══

if [ "$MODE" = "--json" ]; then
    TS=$(python3 -c "import time; print(int(time.time() * 1000))")
    cat <<RECEIPT
{
  "ceremony": "NODE0_GENESIS",
  "timestamp_ms": $TS,
  "hard_gates_total": $TOTAL,
  "hard_gates_passed": $PASS,
  "hard_gates_failed": $FAIL,
  "all_hard_passed": $([ "$FAIL" -eq 0 ] && echo true || echo false),
  "failed_ids": [$(echo "$FAILED_GATES" | tr ' ' '\n' | grep -v '^$' | sed 's/.*/"&"/' | paste -sd, -)],
  "lifecycle_status": "$(lc_field status)",
  "thresholds": {"ihsan": 0.95, "snr": 0.85, "gini": 0.35},
  "score": $(python3 -c "print(round($PASS / max($TOTAL,1), 4))")
}
RECEIPT
else
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo " Hard Gates: $PASS/$TOTAL passed, $FAIL failed"
    if [ "$MODE" = "--full" ]; then
        echo " Supporting: $S_PASS/$S_TOTAL passed"
    fi
    echo " Lifecycle:  $(lc_field status)"
    echo " Score:      $(python3 -c "print(round($PASS / max($TOTAL,1) * 100, 1))")%"
    echo "═══════════════════════════════════════════════════════════"

    if [ "$FAIL" -eq 0 ]; then
        echo ""
        echo " Block 0 is alive. Node0 is sovereign."
        echo " The template is ready."
        echo ""
    else
        echo ""
        echo " Node0 NOT complete. Failed:$FAILED_GATES"
        echo ""
    fi
fi

exit $FAIL
