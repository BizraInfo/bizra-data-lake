---
paths:
  - "src/receipts.rs"
  - "core/fate.py"
  - "docs/evidence/**/*"
  - "bizra_kernel/*receipt*.py"
---

# Receipt Evidence Rules

Rules for BIZRA's receipt-native architecture.

## Receipt Schema

### Required Fields
Every receipt MUST contain these fields:

```json
{
  "receipt_id": "unique-identifier",
  "timestamp": "RFC3339-format",
  "task_summary": "what-was-done",
  "rejection_codes": [],
  "escalation_level": "None|Low|Medium|High|Critical",
  "integrity_hash": "SHA-256-hash"
}
```

### Optional Fields
```json
{
  "metadata": {
    "type": "build|test|validation|commit|deploy|evidence",
    "status": "success|failure",
    "generator": "component-name",
    "generator_version": "1.0.0"
  },
  "evidence_chain": {
    "parent_receipts": ["parent-receipt-id"],
    "chain_depth": 1,
    "root_receipt": "original-receipt-id"
  }
}
```

## Receipt Schema Guard

When modifying `src/receipts.rs` or `core/fate.py`:

1. **Both files MUST stay synchronized**
2. Update all receipt parsers in `tests/`
3. Update evidence docs in `docs/execution/`
4. Update CLAUDE.md documentation
5. Maintain backward compatibility
6. Generate schema change receipt

## Integrity Hash

### Calculation
```python
import hashlib

def compute_integrity_hash(receipt_id: str, timestamp: str, task_summary: str) -> str:
    hash_input = f"{receipt_id}{timestamp}{task_summary}"
    return hashlib.sha256(hash_input.encode()).hexdigest()
```

### Verification
```rust
fn verify_integrity(receipt: &Receipt) -> bool {
    let expected = compute_hash(
        &receipt.receipt_id,
        &receipt.timestamp,
        &receipt.task_summary
    );
    constant_time_compare(&expected, &receipt.integrity_hash)
}
```

## Storage Rules

### Append-Only
- NEVER delete receipts
- NEVER modify existing receipts
- Create new receipts for corrections

### Location
- Primary: `docs/evidence/receipts/`
- Format: `{operation}-{timestamp}.json`
- One receipt per file

### Naming Convention
```
build-20260120-103000-abc123.json
test-20260120-102500-def456.json
validation-20260120-102000-ghi789.json
```

## Escalation Levels

| Level | Meaning | Action |
|-------|---------|--------|
| None | Normal operation | Log only |
| Low | Minor issue | Log with warning |
| Medium | Requires attention | Alert team |
| High | Security/quality concern | Human review |
| Critical | Immediate intervention | Block + alert |

## Receipt Types

### Build Receipts
```json
{
  "receipt_id": "build-...",
  "task_summary": "Rust release build completed",
  "rejection_codes": [],
  "escalation_level": "None"
}
```

### Validation Receipts
```json
{
  "receipt_id": "validation-...",
  "task_summary": "Ihsān gate passed (score: 0.995)",
  "rejection_codes": [],
  "escalation_level": "None"
}
```

### Rejection Receipts
```json
{
  "receipt_id": "rejection-...",
  "task_summary": "Request rejected: SAT consensus failed",
  "rejection_codes": ["SAT_CONSENSUS_FAILURE", "BIAS_DETECTED"],
  "escalation_level": "High"
}
```

## Implementation Patterns

### Emit on Success
```rust
let receipt = Receipt::success(&task, "Build completed");
receipts.emit(receipt).await?;
```

### Emit on Failure
```rust
let receipt = Receipt::rejection(
    &task,
    vec!["IHSAN_GATE_FAILURE"],
    EscalationLevel::High,
);
receipts.emit(receipt).await?;

// Then fail
return Err(BizraError::IhsanGateFailure(...));
```

### Emit with Chain
```python
receipt = create_receipt(
    task_summary="Test suite passed",
    parent_receipt=build_receipt_id
)
await emit_receipt(receipt)
```

## Validation Requirements

When processing receipts:

1. Verify JSON is valid
2. Check all required fields present
3. Verify integrity hash
4. Check timestamp is valid RFC3339
5. Verify escalation_level is valid enum
6. Log validation errors, don't crash

## Testing

- Test receipt creation with all field combinations
- Test integrity hash calculation and verification
- Test chain linking
- Test backward compatibility with old receipts
- Test failure handling for invalid receipts
