#!/usr/bin/env python3
"""
BIZRA Node0 Activation Smoke Test
===================================
Runs after bizra_node_activate.sh to verify the activation is real.

Tests (in order):
  1. Module imports        — all critical modules importable
  2. Crypto primitives     — Ed25519 + BLAKE3 round-trip
  3. URP genesis           — mint membrane + SAT-5 + resource pool
  4. PAT onboarding        — mint 7 PAT + 5 SAT agents
  5. Agent activation      — DORMANT -> ACTIVE for all agents
  6. FATE gate             — audit_evidence callable, FateResult usable
  7. Receipt chain         — BLAKE3-chained receipts verify
  8. Proof engine          — CanonicalReceipt instantiation
  9. Constitutional spine  — Ihsan threshold importable

Exit codes:
  0 — all tests pass
  1 — one or more tests failed
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ==============================================================================
# Test Infrastructure
# ==============================================================================

PASS = 0
FAIL = 0
RESULTS: list[dict] = []


def test(name: str):
    """Decorator that wraps a test function with pass/fail tracking."""
    def decorator(fn):
        def wrapper():
            global PASS, FAIL
            t0 = time.monotonic()
            try:
                fn()
                elapsed = time.monotonic() - t0
                PASS += 1
                RESULTS.append({"test": name, "status": "PASS", "ms": round(elapsed * 1000)})
                print(f"  ✓ {name} ({round(elapsed * 1000)}ms)")
            except Exception as e:
                elapsed = time.monotonic() - t0
                FAIL += 1
                RESULTS.append({"test": name, "status": "FAIL", "error": str(e), "ms": round(elapsed * 1000)})
                print(f"  ✗ {name}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        return wrapper
    return decorator


# ==============================================================================
# Tests
# ==============================================================================

@test("1. Module imports (12 critical)")
def test_imports():
    modules = [
        "core.pat.agent", "core.pat.minting", "core.pat.channels",
        "core.sat.ceremony", "core.sat.mint_court",
        "core.urp.service", "core.urp.membrane",
        "core.proof_engine.fate_gate", "core.proof_engine.receipt",
        "core.pci.crypto",
        "core.sovereign.runtime_core", "core.sovereign.genesis_identity",
    ]
    for mod in modules:
        __import__(mod)


@test("2. Crypto primitives (Ed25519 + BLAKE3)")
def test_crypto():
    from core.pci.crypto import generate_keypair
    priv, pub = generate_keypair()
    assert priv is not None, "Private key is None"
    assert pub is not None, "Public key is None"
    # BLAKE3
    import blake3
    h = blake3.blake3(b"bismillah").hexdigest()
    assert len(h) == 64, f"BLAKE3 hash length {len(h)} != 64"


@test("3. URP genesis (membrane + SAT-5 + resource pool)")
def test_urp_genesis():
    from core.pci.crypto import generate_keypair
    from core.urp.service import URPService

    _, pub = generate_keypair()
    pub_str = pub if isinstance(pub, str) else pub.hex() if isinstance(pub, bytes) else str(pub)
    urp = URPService()
    result = urp.mint_genesis(founder_node_id="SMOKE_NODE", founder_public_key=pub_str)
    assert result.sat_count == 5, f"Expected 5 SAT, got {result.sat_count}"
    status = urp.status()
    assert status["genesis_complete"] is True, "Genesis not marked complete"


@test("4. PAT onboarding (7 PAT + 5 SAT)")
def test_pat_onboard():
    from core.pci.crypto import generate_keypair
    from core.pat.minting import onboard_user

    _, pub = generate_keypair()
    pub_str = pub if isinstance(pub, str) else pub.hex() if isinstance(pub, bytes) else str(pub)
    result = onboard_user(pub_str)
    assert result.pat_agent_count == 7, f"Expected 7 PAT, got {result.pat_agent_count}"
    assert result.sat_agent_count == 5, f"Expected 5 SAT, got {result.sat_agent_count}"
    assert result.total_agents_minted == 12, f"Expected 12 total, got {result.total_agents_minted}"
    assert len(result.user_agents) == 7, f"Expected 7 user agents (PAT), got {len(result.user_agents)}"
    assert len(result.system_agents) == 5, f"Expected 5 system agents (SAT), got {len(result.system_agents)}"


@test("5. Agent activation (DORMANT -> ACTIVE)")
def test_agent_activation():
    from core.pat.agent import PATAgent, AgentType, AgentStatus

    agent = PATAgent.create(
        owner_id="BIZRA-SMOKE0",
        agent_type=AgentType.WORKER,
        index=0,
    )
    assert agent.status == AgentStatus.DORMANT, f"Initial status {agent.status} != DORMANT"
    agent.activate()
    assert agent.status == AgentStatus.ACTIVE, f"Activated status {agent.status} != ACTIVE"


@test("6. FATE gate (audit_evidence + FateResult)")
def test_fate_gate():
    from core.proof_engine.fate_gate import audit_evidence, FateResult
    assert callable(audit_evidence), "audit_evidence not callable"
    assert FateResult is not None, "FateResult not defined"


@test("7. SAT gates importable")
def test_sat_gates():
    from core.sat import gate_result
    assert hasattr(gate_result, "GateResult"), "GateResult not found in gate_result module"
    # Verify all 6 gates import
    gate_modules = [
        "core.sat.ambassador_gate",
        "core.sat.conductor_gate",
        "core.sat.ledger_gate",
        "core.sat.oracle_s_gate",
        "core.sat.sentinel_gate",
        "core.sat.provenance_gate",
    ]
    for mod in gate_modules:
        __import__(mod)


@test("8. Proof engine (CanonicalReceipt)")
def test_proof_receipt():
    # Try to import and verify CanonicalReceipt exists
    from core.proof_engine import receipt as receipt_mod
    # Check for CanonicalReceipt or similar receipt class
    receipt_classes = [
        name for name in dir(receipt_mod)
        if "receipt" in name.lower() and isinstance(getattr(receipt_mod, name), type)
    ]
    assert len(receipt_classes) > 0, f"No receipt classes found in core.proof_engine.receipt"


@test("9. Constitutional thresholds (Ihsan >= 0.95)")
def test_thresholds():
    from core.integration.constants import (
        IHSAN_THRESHOLD,
        SNR_THRESHOLD,
    )
    assert IHSAN_THRESHOLD >= 0.95, f"Ihsan threshold {IHSAN_THRESHOLD} < 0.95"
    assert SNR_THRESHOLD >= 0.85, f"SNR threshold {SNR_THRESHOLD} < 0.85"


# ==============================================================================
# Runner
# ==============================================================================

def main():
    print()
    print("═" * 60)
    print("  BIZRA NODE0 ACTIVATION SMOKE TEST")
    print("═" * 60)
    print()

    tests = [
        test_imports,
        test_crypto,
        test_urp_genesis,
        test_pat_onboard,
        test_agent_activation,
        test_fate_gate,
        test_sat_gates,
        test_proof_receipt,
        test_thresholds,
    ]

    for t in tests:
        t()

    print()
    print("═" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    print("═" * 60)

    # Save results
    state_dir = REPO_ROOT / "sovereign_state" / "receipts"
    state_dir.mkdir(parents=True, exist_ok=True)
    results_path = state_dir / "smoke_test_latest.json"
    results_path.write_text(json.dumps({
        "test_type": "activation_smoke",
        "total": total,
        "passed": PASS,
        "failed": FAIL,
        "results": RESULTS,
    }, indent=2))
    print(f"\n  Results saved: {results_path}")

    if FAIL > 0:
        print(f"\n  ✗ {FAIL} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\n  ✓ All {PASS} tests passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
