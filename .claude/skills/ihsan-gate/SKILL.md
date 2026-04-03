---
name: ihsan-gate
description: Ihsan (Excellence) ethical gate validation
---

# Ihsan Gate Skill

Ihsan (Arabic: Excellence) is BIZRA's ethical scoring system.

## 8 Dimensions (constitution/ihsan_v1.yaml)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| correctness | 0.22 | Is it factually right? |
| safety | 0.22 | Is it safe for users? |
| user_benefit | 0.14 | Does it help users? |
| efficiency | 0.12 | Is it optimal? |
| auditability | 0.12 | Can it be reviewed? |
| anti_centralization | 0.08 | Does it decentralize? |
| robustness | 0.06 | Is it resilient? |
| adl_fairness | 0.04 | Is it fair? |

**Total weights MUST sum to 1.0**

## Threshold

- **Production**: 0.95
- **CI**: 0.95
- **Development**: 0.95

All outputs are gated - execution FAILS if score < threshold.

## Fail-Closed Enforcement

```rust
if !validation.consensus_reached {
    let escalation = fate.escalate_rejection(...);
    receipts.emit_rejection(...);
    return Err(...);  // NEVER proceed
}
```

## Key Files

- `constitution/ihsan_v1.yaml` - Single source of truth
- `src/ihsan.rs` - Rust implementation
- `bizra_kernel/ihsan_gate.py` - Python implementation

## Validation Command

Run `/ihsan` to validate Ihsan constitution.

## Constitution Guard

Never modify `constitution/ihsan_v1.yaml` without:
1. Verifying weights sum to 1.0
2. Testing all 8 dimensions present
3. Ensuring threshold = 0.95
4. Updating both Rust and Python code
