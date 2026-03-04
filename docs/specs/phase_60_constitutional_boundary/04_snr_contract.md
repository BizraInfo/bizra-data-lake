# Step 4: Cross-Plane SNR Contract Enforcement

## Standing on Giants: Shannon (1948) | Dijkstra (contracts between modules)

## Problem Statement

SAPE audit finding F9 identified SNR normalization divergence. Codebase
exploration reveals the situation is partially resolved:

**Canonical function exists:** `core/snr_protocol.py:normalize_snr_linear()`
implements the logistic formula `snr / (1 + snr)` mapping unbounded ratios
to [0, 1]. This is correct.

**Problem:** Not all code paths use it. The exploration found:

| Location | Formula | Status |
|----------|---------|--------|
| `core/snr_protocol.py` | `snr / (1 + snr)` | Canonical |
| `core/sovereign/mission.py:763` | `normalize_snr_linear(...)` | Correct |
| `core/sovereign/snr_maximizer.py:157` | `signal / (noise + 1e-10)` | Raw ratio — no normalization |
| `core/apex/snr_apex_engine.py:673` | `signal_power / noise_power` | Raw ratio — used for dB diagnostic |
| `.tmp_prod_artifacts_v2/*/snr.py` | `normalize_snr_linear(...)` | Correct |

The `snr_maximizer.py` and `snr_apex_engine.py` return unbounded `snr_linear`
values for internal diagnostics (correct), but callers may pass these to
receipt emission without normalization (incorrect).

**Solution:** Add a contract test that verifies every receipt-emitting code
path produces SNR values in [0, 1]. Add a linting rule that flags direct
`snr_linear` usage near receipt construction.

## Target Files

| File | Action |
|------|--------|
| `core/snr_protocol.py` | Update: add `assert_snr_normalized()` guard |
| `tests/core/test_snr_contract.py` | New: contract tests for all SNR-emitting paths |
| `core/proof_engine/evidence_ledger.py` | Update: validate SNR range on append |

## Pseudocode

### core/snr_protocol.py — Guard Function

```pseudocode
# Add to existing module:

FUNCTION assert_snr_normalized(value: float, source: str = "unknown") -> float:
    """Guard: ensures an SNR value is in [0, 1] range.

    Call this at every boundary where SNR enters a receipt or evidence
    document. If the value is out of range, it's a bug in the caller —
    they forgot to call normalize_snr_linear().

    Args:
        value: the SNR score to validate
        source: caller identifier for debugging

    Returns:
        The validated value (unchanged)

    Raises:
        ValueError: if value is outside [0, 1]
    """
    IF NOT (0.0 <= value <= 1.0):
        RAISE ValueError(
            f"SNR value {value} from '{source}' is outside [0, 1]. "
            f"Did you forget to call normalize_snr_linear()?"
        )
    RETURN value
```

### evidence_ledger.py — SNR Validation on Append

```pseudocode
# In EvidenceLedger.append(), add validation:

FUNCTION append(self, receipt: dict) -> str:
    """Append a receipt to the evidence ledger.

    Added validation: if receipt contains 'snr' or 'snr_score',
    verify the value is normalized to [0, 1].
    """
    # Existing validation...

    # SNR boundary check (Phase 60)
    snr_value = receipt.get("snr") or receipt.get("snr_score")
    IF snr_value IS NOT None AND isinstance(snr_value, (int, float)):
        IF NOT (0.0 <= float(snr_value) <= 1.0):
            IF self._validate_on_append:
                RAISE ValueError(
                    f"Receipt SNR value {snr_value} outside [0,1]. "
                    f"Use normalize_snr_linear() before emission."
                )

    # Continue with existing append logic...
```

## TDD Anchors

```pseudocode
TEST normalize_snr_linear_basic:
    """Canonical normalization maps ratios to [0, 1]."""
    FROM core.snr_protocol IMPORT normalize_snr_linear
    ASSERT normalize_snr_linear(0.0) == 0.0
    ASSERT normalize_snr_linear(1.0) == 0.5
    ASSERT abs(normalize_snr_linear(9.0) - 0.9) < 0.01
    ASSERT normalize_snr_linear(float("inf")) == 1.0  # saturates

TEST normalize_snr_linear_negative_clamped:
    ASSERT normalize_snr_linear(-5.0) == 0.0

TEST normalize_snr_linear_output_always_in_range:
    """Property: output is always in [0, 1] for any input."""
    FROM core.snr_protocol IMPORT normalize_snr_linear
    FOR x IN [-100, -1, 0, 0.001, 0.5, 1, 2, 10, 100, 1e6]:
        result = normalize_snr_linear(x)
        ASSERT 0.0 <= result <= 1.0, f"normalize_snr_linear({x}) = {result}"

TEST assert_snr_normalized_accepts_valid:
    FROM core.snr_protocol IMPORT assert_snr_normalized
    ASSERT assert_snr_normalized(0.0) == 0.0
    ASSERT assert_snr_normalized(0.5) == 0.5
    ASSERT assert_snr_normalized(1.0) == 1.0

TEST assert_snr_normalized_rejects_invalid:
    FROM core.snr_protocol IMPORT assert_snr_normalized
    WITH pytest.raises(ValueError, match="outside.*0.*1"):
        assert_snr_normalized(1.5)
    WITH pytest.raises(ValueError, match="outside.*0.*1"):
        assert_snr_normalized(-0.1)
    WITH pytest.raises(ValueError, match="outside.*0.*1"):
        assert_snr_normalized(9.0, source="test_caller")

TEST evidence_ledger_rejects_unnormalized_snr:
    """Ledger refuses to append receipts with out-of-range SNR."""
    FROM core.proof_engine.evidence_ledger IMPORT EvidenceLedger
    ledger = EvidenceLedger(path=tmp_path / "test.jsonl", validate_on_append=True)
    bad_receipt = {
        "receipt_id": "a" * 16,
        "action": "test",
        "snr": 1.5,  # NOT normalized — should be rejected
        "reason_codes": ["TEST"],
    }
    WITH pytest.raises(ValueError, match="SNR value"):
        ledger.append(bad_receipt)

TEST evidence_ledger_accepts_normalized_snr:
    ledger = EvidenceLedger(path=tmp_path / "test.jsonl", validate_on_append=True)
    good_receipt = {
        "receipt_id": "b" * 16,
        "action": "test",
        "snr": 0.85,
        "reason_codes": ["TEST"],
    }
    # Should not raise
    ledger.append(good_receipt)

TEST mission_orchestrator_emits_normalized_snr:
    """End-to-end: MissionOrchestrator produces SNR in [0, 1]."""
    FROM core.sovereign.mission IMPORT MissionOrchestrator
    # Mock dependencies, run a mission, check receipt
    result = await orchestrator.execute_mission(mock_request)
    IF "snr" IN result.evidence:
        ASSERT 0.0 <= result.evidence["snr"] <= 1.0

TEST all_snr_producing_modules_use_canonical:
    """Grep contract: no raw snr_linear values leak to receipts."""
    # Scan all Python files for receipt-related SNR emission
    # Ensure they call normalize_snr_linear() or assert_snr_normalized()
    suspect_patterns = [
        "snr_linear",  # raw value used near receipt
    ]
    receipt_patterns = [
        "receipt", "evidence", "emit",
    ]
    # Any file containing both a suspect pattern and a receipt pattern
    # without also containing normalize_snr_linear must be flagged
    violations = scan_codebase(suspect_patterns, receipt_patterns, exclude="normalize_snr")
    ASSERT len(violations) == 0, f"Unnormalized SNR near receipts: {violations}"
```

## Acceptance Criteria

1. `assert_snr_normalized()` exists in `core/snr_protocol.py`
2. `EvidenceLedger.append()` validates SNR range when `validate_on_append=True`
3. Contract test scans all receipt-emitting code for normalized SNR usage
4. No raw `snr_linear` values leak into evidence receipts
5. Full test suite GREEN
