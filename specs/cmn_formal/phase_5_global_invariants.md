# Phase 5: Global Invariants — Continuous Constitutional Validation

**Spec:** CMN-005
**Status:** New (existing: PARTIAL — thresholds defined, no continuous checker)
**Formal Property:** The system continuously validates S (Sovereign), M (Membrane), Z (Zann=0), R (Riba=0)
**Existing Code:** `core/integration/constants.py`, `core/urp/constitution.py`

---

## 1. Objective

Create a `GlobalInvariantChecker` that periodically validates all four CMN properties
and emits constitutional health receipts. This is the final integration layer that
composes Phases 1-4 into a single provable system.

---

## 2. Definitions

```
SystemState := {
    sovereignty:    SovereigntyStatus,    # Omega_n disjoint URP
    membrane:       MembraneStatus,       # DFA + 3 properties
    zann:           ZannStatus,           # derivation chains valid
    riba:           RibaStatus,           # exact arithmetic, no interest
}

GlobalInvariant := sovereignty.ok AND membrane.ok AND zann.ok AND riba.ok

ConstitutionalHealthReceipt := {
    timestamp:      float,
    invariants:     dict[str, bool],      # {S: T, M: T, Z: T, R: T}
    ihsan_score:    float,                # composite
    receipt_hash:   BLAKE3Hash,
    prev_receipt:   BLAKE3Hash,           # chain to previous health check
    violations:     list[Violation],
}
```

---

## 3. Pseudocode

### 3.1 GlobalInvariantChecker (new: `core/governance/invariant_checker.py`)

```python
class GlobalInvariantChecker:
    """Periodically validates all four CMN properties."""

    def __init__(
        self,
        workspace_boundary: WorkspaceBoundary,
        membrane_verifier: MembraneVerifier,
        proof_of_truth: ProofOfTruth,
        riba_auditor: RibaZeroAuditor,
        health_ledger_path: Path,
    ):
        self._sovereignty = workspace_boundary
        self._membrane = membrane_verifier
        self._zann = proof_of_truth
        self._riba = riba_auditor
        self._health_ledger = health_ledger_path
        self._prev_hash = GENESIS_HASH

    async def check_all(self) -> ConstitutionalHealthReceipt:
        """Run all four invariant checks and emit a chained receipt."""
        results = {}

        # S: Sovereignty
        results["sovereignty"] = self._sovereignty.check_disjoint()

        # M: Membrane (sample recent missions)
        results["membrane"] = await self._check_membrane_sample()

        # Z: Zann Zero (verify recent knowledge entries)
        results["zann_zero"] = self._check_zann_sample()

        # R: Riba Zero (audit ledger)
        riba_result = self._riba.audit()
        results["riba_zero"] = riba_result.riba_zero

        # Composite ihsan
        pass_count = sum(1 for v in results.values() if v)
        ihsan = pass_count / len(results)

        # Emit receipt
        receipt = ConstitutionalHealthReceipt(
            timestamp=time.time(),
            invariants=results,
            ihsan_score=ihsan,
            violations=riba_result.violations if not riba_result.riba_zero else [],
            receipt_hash="",  # computed below
            prev_receipt=self._prev_hash,
        )
        receipt.receipt_hash = blake3(
            json.dumps(receipt.to_dict(), sort_keys=True).encode()
        )
        self._prev_hash = receipt.receipt_hash

        # Persist
        self._append_to_ledger(receipt)

        return receipt

    async def _check_membrane_sample(self) -> bool:
        """Sample last N missions, verify all reached terminal state legally."""
        # Read recent mission receipts from evidence ledger
        recent = self._read_recent_missions(limit=10)
        for mission in recent:
            if mission.state not in LEGAL_TERMINAL_STATES:
                return False
        return True

    def _check_zann_sample(self) -> bool:
        """Sample last N knowledge entries, verify chain integrity."""
        recent = self._read_recent_knowledge(limit=10)
        for entry in recent:
            result = self._zann.validate_entry(entry)
            if not result.zann_zero:
                return False
        return True

    def _append_to_ledger(self, receipt: ConstitutionalHealthReceipt):
        """Append receipt to health ledger JSONL."""
        with open(self._health_ledger, "a") as f:
            f.write(json.dumps(receipt.to_dict()) + "\n")
```

### 3.2 Health Daemon Loop

```python
async def run_invariant_daemon(
    checker: GlobalInvariantChecker,
    interval_seconds: float = ENV("BIZRA_INVARIANT_CHECK_INTERVAL", 300),
):
    """Run health checks on a schedule. Emit alerts on violation."""
    while True:
        receipt = await checker.check_all()

        if not all(receipt.invariants.values()):
            failed = [k for k, v in receipt.invariants.items() if not v]
            await emit_alert(
                severity="CRITICAL",
                message=f"Constitutional violation: {failed}",
                receipt=receipt,
            )

        await asyncio.sleep(interval_seconds)
```

---

## 4. TDD Anchors

```python
# tests/core/test_global_invariants.py

@pytest.mark.asyncio
async def test_all_invariants_pass():
    """Healthy system => all four invariants True, ihsan = 1.0."""
    checker = build_healthy_checker()
    receipt = await checker.check_all()
    assert all(receipt.invariants.values())
    assert receipt.ihsan_score == 1.0

@pytest.mark.asyncio
async def test_sovereignty_violation_detected():
    """Omega_n overlap with URP => sovereignty = False."""
    checker = build_checker_with_sovereignty_violation()
    receipt = await checker.check_all()
    assert receipt.invariants["sovereignty"] is False
    assert receipt.ihsan_score < 1.0

@pytest.mark.asyncio
async def test_riba_violation_detected():
    """Float amount in ledger => riba_zero = False."""
    checker = build_checker_with_riba_violation()
    receipt = await checker.check_all()
    assert receipt.invariants["riba_zero"] is False
    assert len(receipt.violations) > 0

@pytest.mark.asyncio
async def test_health_receipts_are_chained():
    """Consecutive checks produce chained BLAKE3 receipts."""
    checker = build_healthy_checker()
    r1 = await checker.check_all()
    r2 = await checker.check_all()
    assert r2.prev_receipt == r1.receipt_hash
    assert r1.prev_receipt == GENESIS_HASH

@pytest.mark.asyncio
async def test_partial_failure_reported():
    """3 of 4 invariants pass => ihsan = 0.75."""
    checker = build_checker_with_zann_violation()
    receipt = await checker.check_all()
    assert receipt.ihsan_score == 0.75
    assert receipt.invariants["zann_zero"] is False
    assert receipt.invariants["sovereignty"] is True

def test_health_ledger_is_append_only():
    """Health ledger grows monotonically, never truncated."""
    checker = build_healthy_checker()
    asyncio.run(checker.check_all())
    size_1 = checker._health_ledger.stat().st_size
    asyncio.run(checker.check_all())
    size_2 = checker._health_ledger.stat().st_size
    assert size_2 > size_1
```

---

## 5. System Properties Table

| Property | Symbol | Gate | Threshold | Enforcement |
|----------|--------|------|-----------|-------------|
| Sovereignty | S | WorkspaceBoundary | disjoint | Fail-closed rejection |
| Membrane | M | PCI + DFA | ihsan >= 0.95 | Sink state Bottom |
| Zann Zero | Z = 0 | PoT validator | chain valid | Reject unprovenanced claims |
| Riba Zero | R = 0 | Sippar auditor | integer-only | Reject float amounts |
| **Composite** | **SMZR** | **GlobalInvariantChecker** | **all True** | **Alert + receipt chain** |

---

## 6. Comparative Summary (from formal paper)

| Property | Monolithic AI | P2P Blockchain | BIZRA (CMN) |
|----------|--------------|----------------|-------------|
| Integrity | Probabilistic (Z > 0) | Consensus (51% risk) | Axiomatic (Z = 0) |
| Privacy | Trusted third party | Public ledger | Topological membrane |
| Economy | Interest-based (R > 0) | Speculative | Impact-based (R = 0) |
| Scaling | Logarithmic | Sub-linear | Linear O(N) |

---

## 7. Implementation Sequence

```
Week 1: WorkspaceBoundary (Phase 1) + tests
Week 2: MembraneVerifier (Phase 2) + DFA reachability proof
Week 3: ProofOfTruth (Phase 3) + chain fork detection
Week 4: RibaZeroAuditor (Phase 4) + Sippar Python bridge
Week 5: GlobalInvariantChecker (Phase 5) + health daemon + integration tests
```

Each phase ships with its own test file. No phase depends on later phases —
they compose at Phase 5 but are independently testable.
