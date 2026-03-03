# Step 1: SNR Engine Normalization Promotion

## Standing on Giants: Shannon (1948) — SNR as bounded quality metric

## Problem Statement

The SNR engine (`core/proof_engine/snr.py`) returns unbounded ratio values.
The evidence schema requires scores in [0, 1]. Currently, the normalization
`min(snr_linear / (1 + snr_linear), 1.0)` lives in `core/sovereign/mission.py`
(the consumer), not in the engine itself. Every new consumer must independently
discover and implement this normalization.

**Violated Principle:** Information hiding (Parnas, 1972). Consumers should
not need to know about the engine's internal representation.

## Target Files

| File | Action |
|------|--------|
| `core/proof_engine/snr.py` | Add `normalize()` method and `normalized_score` property |
| `core/sovereign/mission.py` | Remove inline normalization, use engine's normalized output |
| `tests/core/proof_engine/test_snr_normalization.py` | New test file |

## Pseudocode

### snr.py — Add normalization to SNRResult

```pseudocode
CLASS SNRResult:
    # Existing fields
    score_linear: float       # Unbounded ratio (signal / noise)
    policy_digest: str        # BLAKE3 hex digest of policy
    components: dict          # Per-signal breakdown

    PROPERTY normalized_score -> float:
        """Return score mapped to [0, 1] via logistic saturation.

        Mathematical property:
            f(x) = x / (1 + x) for x >= 0
            - Monotonically increasing
            - Bounded: f(0) = 0, lim f(x) = 1
            - Preserves ordering: if a > b then f(a) > f(b)

        Standing on Giants: Shannon (1948), logistic normalization.
        """
        IF self.score_linear <= 0:
            RETURN 0.0
        raw = self.score_linear / (1.0 + self.score_linear)
        RETURN min(raw, 1.0)  # Belt-and-suspenders for float edge cases

    METHOD to_evidence_dict() -> dict:
        """Serialize for evidence ledger (always uses normalized score)."""
        RETURN {
            "snr_score": self.normalized_score,
            "snr_raw": self.score_linear,
            "policy_digest": self.policy_digest,
        }
```

### mission.py — Remove inline normalization

```pseudocode
# BEFORE (current):
snr_score = min(analysis.score_linear / (1 + analysis.score_linear), 1.0)

# AFTER (promoted):
snr_score = analysis.normalized_score
```

## TDD Anchors

### Test File: `tests/core/proof_engine/test_snr_normalization.py`

```pseudocode
TEST normalized_score_zero_input:
    result = SNRResult(score_linear=0.0, ...)
    ASSERT result.normalized_score == 0.0

TEST normalized_score_unit_input:
    result = SNRResult(score_linear=1.0, ...)
    ASSERT result.normalized_score == 0.5  # 1/(1+1) = 0.5

TEST normalized_score_large_input:
    result = SNRResult(score_linear=100.0, ...)
    ASSERT 0.99 < result.normalized_score <= 1.0

TEST normalized_score_negative_input:
    result = SNRResult(score_linear=-1.0, ...)
    ASSERT result.normalized_score == 0.0  # Clamp negatives to 0

TEST normalized_score_preserves_ordering:
    """Property: if a > b >= 0 then f(a) > f(b)."""
    FOR (a, b) IN [(0.5, 0.1), (10, 5), (100, 99), (0.01, 0.001)]:
        ra = SNRResult(score_linear=a, ...)
        rb = SNRResult(score_linear=b, ...)
        ASSERT ra.normalized_score > rb.normalized_score

TEST normalized_score_bounded:
    """Property: output is always in [0, 1]."""
    FOR x IN [0, 0.001, 0.5, 1.0, 10, 100, 1e6, float('inf')]:
        result = SNRResult(score_linear=x, ...)
        ASSERT 0.0 <= result.normalized_score <= 1.0

TEST to_evidence_dict_uses_normalized:
    result = SNRResult(score_linear=3.0, ...)
    d = result.to_evidence_dict()
    ASSERT d["snr_score"] == result.normalized_score  # 0.75
    ASSERT d["snr_raw"] == 3.0  # Raw preserved for debugging

TEST mission_uses_normalized_score:
    """Integration: mission.py no longer does its own normalization."""
    # Grep mission.py for the old pattern
    source = read_file("core/sovereign/mission.py")
    ASSERT "/ (1 +" NOT IN source  # Old normalization removed
    ASSERT "normalized_score" IN source  # New property used
```

## Acceptance Criteria

1. `SNRResult.normalized_score` always returns value in [0, 1]
2. `mission.py` has zero inline normalization code
3. All existing SNR tests pass unchanged
4. New normalization tests pass (8 tests)
5. Full suite remains GREEN
6. Evidence receipt emitted: `{step: "snr_normalization", status: "complete"}`

## Rollback Plan

If normalization changes break downstream consumers:
1. Revert `snr.py` changes
2. Restore inline normalization in `mission.py`
3. File issue for a phased migration with deprecation warnings
