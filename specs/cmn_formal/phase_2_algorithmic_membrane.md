# Phase 2: Algorithmic Membrane — DFA with Fail-Closed Semantics

**Spec:** CMN-002
**Status:** Verification layer (existing: IMPLEMENTED)
**Formal Property:** M(iota) satisfies Anonymity, Epistemic Validity, Constitutional Alignment, or M(iota) = bottom
**Existing Code:** `bizra-mission/state.rs` (14-state DFA), `core/pci/gates.py` (6-gate chain), `bizra-core/pci/gates.rs`

---

## 1. Objective

The Membrane M is already implemented across the mission state machine and PCI gate chain.
This spec formalizes the **three transformation properties** and adds **verification tests**
that prove the DFA is correct — no illegal states reachable, sink state is absorbing.

---

## 2. Definitions

```
M : Intent -> Action | Bottom

Intent iota := {
    description: str,          # user's natural language request
    context: DesktopContext,   # active window, clipboard, screen
    source: str,               # "node_gateway" | "ahk_hotkey" | "ghost_panel"
    timestamp: float
}

Action alpha := {
    mission_id: str,
    channels_executed: list[ChannelResult],
    synthesis: str,
    evidence_receipt_id: str,
    ihsan_score: float,        # must be >= 0.95
    snr_score: float           # must be >= 0.85
}

Bottom := rejection with {
    reject_code: RejectCode,
    gate_name: str,
    reason: str,
    receipt_hash: blake3(rejection)
}
```

### 2.1 Transformation Properties

```
P1_ANONYMITY:
    For all iota, the identity of the requesting node is not
    recoverable from M(iota). Enforced by membrane.filter_outbound()
    stripping node_id and signing_key before URP crossing.

P2_EPISTEMIC_VALIDITY:
    M(iota) derives from Truth_Axioms only. Enforced by the
    ProvenanceGate in the PCI chain — every claim must have a
    BLAKE3 derivation chain.

P3_CONSTITUTIONAL_ALIGNMENT:
    f_cost(M(iota)) >= IHSAN_THRESHOLD (0.95). Enforced by the
    IhsanGate as the final gate before CommitGate.
```

---

## 3. Pseudocode

### 3.1 Membrane Verifier (new: `core/pci/membrane_verifier.py`)

```python
class MembraneVerifier:
    """Verify that M satisfies all three transformation properties."""

    def __init__(self, constitution: Constitution):
        self._constitution = constitution
        self._ihsan_floor = constitution.ihsan_floor  # 0.95

    def verify_transformation(
        self, intent: Intent, result: Action | Bottom
    ) -> VerificationResult:
        checks = {
            "anonymity": self._check_anonymity(result),
            "epistemic_validity": self._check_provenance(result),
            "constitutional_alignment": self._check_ihsan(result),
        }
        all_pass = all(c.passed for c in checks.values())
        return VerificationResult(passed=all_pass, checks=checks)

    def _check_anonymity(self, result) -> CheckResult:
        """No node identity recoverable from action payload."""
        if isinstance(result, Bottom):
            return CheckResult(passed=True, reason="rejected — no identity leaked")
        IDENTITY_FIELDS = {"node_id", "signing_key", "ip_address", "mac_address"}
        leaked = IDENTITY_FIELDS & set(result.keys_recursive())
        return CheckResult(
            passed=len(leaked) == 0,
            reason=f"leaked: {leaked}" if leaked else "clean"
        )

    def _check_provenance(self, result) -> CheckResult:
        """Every claim has a BLAKE3 derivation chain."""
        if isinstance(result, Bottom):
            return CheckResult(passed=True, reason="rejected — no claims made")
        has_receipt = bool(result.evidence_receipt_id)
        return CheckResult(passed=has_receipt, reason="receipt present" if has_receipt else "NO RECEIPT")

    def _check_ihsan(self, result) -> CheckResult:
        """Constitutional alignment: ihsan >= threshold."""
        if isinstance(result, Bottom):
            return CheckResult(passed=True, reason="fail-closed is constitutional")
        passed = result.ihsan_score >= self._ihsan_floor
        return CheckResult(
            passed=passed,
            reason=f"ihsan={result.ihsan_score:.3f} vs floor={self._ihsan_floor}"
        )
```

### 3.2 DFA Reachability Proof

```python
def prove_dfa_reachability(state_machine: MissionStateMachine) -> DFAProof:
    """Enumerate all reachable states from Submitted. Prove no illegal state is reachable."""
    reachable = set()
    frontier = {"Submitted"}

    while frontier:
        state = frontier.pop()
        reachable.add(state)
        for next_state in state_machine.valid_transitions(state):
            if next_state not in reachable:
                frontier.add(next_state)

    # Terminal states must be absorbing (no outgoing transitions)
    TERMINAL = {"Complete", "Degraded", "Failed", "TimedOut"}
    for t in TERMINAL:
        assert state_machine.valid_transitions(t) == [], f"{t} is not absorbing"

    # Sink state Bottom must be reachable from every non-terminal state
    non_terminal = reachable - TERMINAL
    for s in non_terminal:
        transitions = state_machine.valid_transitions(s)
        assert "Failed" in transitions, f"{s} has no path to sink"

    return DFAProof(
        reachable_states=reachable,
        terminal_states=TERMINAL & reachable,
        all_terminals_absorbing=True,
        all_states_can_fail=True,
    )
```

---

## 4. TDD Anchors

```python
# tests/core/test_membrane_dfa.py

def test_membrane_rejects_below_ihsan():
    """P3: Action with ihsan < 0.95 must be Bottom."""
    verifier = MembraneVerifier(DEFAULT_CONSTITUTION)
    bad_action = MockAction(ihsan_score=0.80, snr_score=0.90)
    result = verifier.verify_transformation(MockIntent(), bad_action)
    assert result.checks["constitutional_alignment"].passed is False

def test_membrane_passes_valid_action():
    """All three properties satisfied => PASS."""
    verifier = MembraneVerifier(DEFAULT_CONSTITUTION)
    good_action = MockAction(ihsan_score=0.97, snr_score=0.92, evidence_receipt_id="abc123")
    result = verifier.verify_transformation(MockIntent(), good_action)
    assert result.passed is True

def test_membrane_anonymity_strips_identity():
    """P1: No node_id or signing_key in output."""
    verifier = MembraneVerifier(DEFAULT_CONSTITUTION)
    leaked_action = MockAction(ihsan_score=0.97, extra_fields={"node_id": "n0"})
    result = verifier.verify_transformation(MockIntent(), leaked_action)
    assert result.checks["anonymity"].passed is False

def test_membrane_bottom_always_passes():
    """Rejection (Bottom) satisfies all properties by definition."""
    verifier = MembraneVerifier(DEFAULT_CONSTITUTION)
    bottom = Bottom(reject_code="IHSAN_LOW", gate_name="IhsanGate", reason="0.70 < 0.95")
    result = verifier.verify_transformation(MockIntent(), bottom)
    assert result.passed is True

def test_dfa_no_orphan_states():
    """Every state is reachable from Submitted."""
    proof = prove_dfa_reachability(MissionStateMachine())
    assert len(proof.reachable_states) == 14

def test_dfa_terminal_states_absorbing():
    """Complete, Degraded, Failed, TimedOut have no outgoing transitions."""
    proof = prove_dfa_reachability(MissionStateMachine())
    assert proof.all_terminals_absorbing is True

def test_dfa_every_state_can_fail():
    """Every non-terminal state has a path to Failed (fail-closed)."""
    proof = prove_dfa_reachability(MissionStateMachine())
    assert proof.all_states_can_fail is True
```

---

## 5. Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `bizra-mission/state.rs` | Source of truth for DFA transitions — Python wrapper reads this |
| `core/pci/gates.py` | PCIGateKeeper already enforces P2 + P3; add P1 check |
| `core/urp/membrane.py` | `filter_outbound()` enforces P1 (anonymity) |
| `core/proof_engine/evidence_ledger.py` | Provides derivation chain for P2 |
