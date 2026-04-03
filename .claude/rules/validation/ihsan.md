---
paths:
  - "constitution/**/*.yaml"
  - "src/ihsan.rs"
  - "src/ihsan_gate.py"
  - "core/**/ihsan*.py"
  - "bizra_kernel/ihsan*.py"
---

# Ihsān (إحسان) Validation Rules

Rules for working with BIZRA's ethical excellence framework.

## Constitution Integrity

### Single Source of Truth
The Ihsān constitution is defined in `constitution/ihsan_v1.yaml`.
- All implementations MUST derive from this file
- Never hardcode dimension weights in code
- Cross-reference Rust (`src/ihsan.rs`) and Python implementations

### 8 Ethical Dimensions

| Dimension | Weight | Purpose |
|-----------|--------|---------|
| correctness | 0.22 | Logical accuracy and validity |
| safety | 0.22 | Safety constraint compliance |
| user_benefit | 0.14 | Value delivered to user |
| efficiency | 0.12 | Resource optimization |
| auditability | 0.12 | Transparency and traceability |
| anti_centralization | 0.08 | Decentralization promotion |
| robustness | 0.06 | System resilience |
| adl_fairness | 0.04 | Equitable treatment |

### Invariants
- Weights MUST sum to exactly 1.0
- All 8 dimensions MUST be present
- Production threshold MUST be >= 0.99
- NEVER lower the production threshold

## Implementation Requirements

### When Modifying Constitution
1. Update `constitution/ihsan_v1.yaml` (single source)
2. Verify weights sum to 1.0
3. Update Rust implementation (`src/ihsan.rs`)
4. Update Python implementation (`core/` and `bizra_kernel/`)
5. Update tests in both languages
6. Update documentation in `CLAUDE.md`
7. Generate evidence receipt for change

### Score Calculation
```python
def calculate_ihsan_score(dimensions: dict[str, float]) -> float:
    """Calculate weighted Ihsān score."""
    weights = load_constitution()["dimensions"]

    total = 0.0
    for dim, score in dimensions.items():
        weight = weights[dim]["weight"]
        total += score * weight

    return total
```

### Gate Enforcement
```rust
// Fail-closed pattern
if ihsan_score < IHSAN_THRESHOLD {
    let receipt = Receipt::rejection(
        &task,
        vec!["IHSAN_GATE_FAILURE"],
        EscalationLevel::High,
    );
    receipts.emit(receipt).await?;

    return Err(BizraError::IhsanGateFailure {
        score: ihsan_score,
        threshold: IHSAN_THRESHOLD,
    });
}
```

## Validation Checklist

When editing Ihsān-related code:

- [ ] Constitution file is not hardcoded
- [ ] All 8 dimensions are handled
- [ ] Weights sum to 1.0
- [ ] Threshold comparison uses >= (not >)
- [ ] Failures emit receipts
- [ ] Failures escalate appropriately
- [ ] Never silently proceed on failure

## Common Mistakes to Avoid

### DO NOT
```python
# Bad - hardcoded weights
WEIGHTS = {"correctness": 0.22, "safety": 0.22, ...}

# Bad - wrong threshold check
if score > 0.99:  # Should be >=

# Bad - silent failure
if score < THRESHOLD:
    return None  # Should fail loudly
```

### DO
```python
# Good - load from constitution
weights = yaml.safe_load(open("constitution/ihsan_v1.yaml"))["dimensions"]

# Good - correct threshold
if score >= IHSAN_THRESHOLD:

# Good - fail-closed
if score < THRESHOLD:
    await fate.escalate(EscalationLevel.HIGH, "Ihsan gate failure")
    raise IhsanGateError(score, THRESHOLD)
```

## Testing Requirements

- Test each dimension individually
- Test score calculation with known inputs
- Test gate enforcement at boundary (0.99, 0.989, 0.991)
- Test constitution loading from file
- Test failure escalation and receipt generation
