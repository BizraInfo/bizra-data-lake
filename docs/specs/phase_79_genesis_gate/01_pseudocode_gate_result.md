# Phase 79: GateResult Model — Pseudocode

---

## Module: `core/sat/gate_result.py`

```pseudocode
IMPORT dataclass, field FROM dataclasses
IMPORT datetime
IMPORT typing (List, Tuple, Dict, Any, Optional)

ENUM GateVerdict:
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"

ENUM CheckStatus:
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    SKIPPED = "SKIPPED"

DATACLASS CheckResult:
    name: str
    status: CheckStatus
    evidence: str = ""          # Brief explanation
    is_manual: bool = False     # True for human attestation checks

    PROPERTY passed -> bool:
        RETURN status == PASS

DATACLASS GateResult:
    agent: str                  # "Sentinel", "Oracle-S", "Ledger", "Conductor", "Ambassador"
    layer: str                  # "STRUCTURAL_INTEGRITY", etc.
    checks: List[CheckResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    PROPERTY passed -> bool:
        RETURN all(c.passed FOR c IN checks IF c.status != SKIPPED)

    PROPERTY failed -> List[CheckResult]:
        RETURN [c FOR c IN checks IF NOT c.passed AND c.status != SKIPPED]

    PROPERTY verdict -> GateVerdict:
        RETURN APPROVED IF passed ELSE BLOCKED

    PROPERTY stats -> Dict[str, int]:
        RETURN {
            "total": len(checks),
            "pass": count(PASS),
            "fail": count(FAIL),
            "partial": count(PARTIAL),
            "not_impl": count(NOT_IMPLEMENTED),
            "skipped": count(SKIPPED),
        }

    METHOD to_dict() -> Dict[str, Any]:
        RETURN {
            "agent": agent,
            "layer": layer,
            "passed": passed,
            "verdict": verdict.value,
            "timestamp": timestamp,
            "stats": stats,
            "checks": [
                {"name": c.name, "status": c.status.value, "evidence": c.evidence}
                FOR c IN checks
            ],
            "failed": [c.name FOR c IN failed],
        }

    METHOD to_receipt() -> Dict[str, Any]:
        # For evidence chain integration
        RETURN {
            "type": "sat_gate",
            "agent": agent,
            "layer": layer,
            "verdict": verdict.value,
            "timestamp": timestamp,
            "check_count": len(checks),
            "pass_count": stats["pass"],
            "fail_count": stats["fail"],
            "failed_names": [c.name FOR c IN failed],
        }
```

---

## TDD Anchors

```pseudocode
TEST test_gate_result_all_pass:
    checks = [CheckResult("a", PASS), CheckResult("b", PASS)]
    gate = GateResult("Sentinel", "STRUCTURAL_INTEGRITY", checks)
    ASSERT gate.passed == True
    ASSERT gate.verdict == APPROVED
    ASSERT gate.stats["pass"] == 2

TEST test_gate_result_one_fail:
    checks = [CheckResult("a", PASS), CheckResult("b", FAIL)]
    gate = GateResult("Sentinel", "STRUCTURAL_INTEGRITY", checks)
    ASSERT gate.passed == False
    ASSERT gate.verdict == BLOCKED
    ASSERT len(gate.failed) == 1

TEST test_gate_result_skipped_ignored:
    checks = [CheckResult("a", PASS), CheckResult("b", SKIPPED)]
    gate = GateResult("Sentinel", "STRUCTURAL_INTEGRITY", checks)
    ASSERT gate.passed == True  # SKIPPED doesn't block

TEST test_gate_result_partial_is_fail:
    checks = [CheckResult("a", PARTIAL)]
    gate = GateResult("Sentinel", "STRUCTURAL_INTEGRITY", checks)
    ASSERT gate.passed == False  # PARTIAL is not PASS

TEST test_to_dict_roundtrip:
    gate = GateResult("Ledger", "ECONOMIC_SOUNDNESS", [CheckResult("x", PASS)])
    d = gate.to_dict()
    ASSERT d["agent"] == "Ledger"
    ASSERT d["verdict"] == "APPROVED"
    ASSERT len(d["checks"]) == 1

TEST test_to_receipt_schema:
    gate = GateResult("Oracle-S", "CONSTITUTIONAL_COMPLIANCE", [CheckResult("y", FAIL)])
    r = gate.to_receipt()
    ASSERT r["type"] == "sat_gate"
    ASSERT r["fail_count"] == 1
    ASSERT "y" IN r["failed_names"]
```
