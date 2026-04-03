---
name: receipt-generation
description: Generate BIZRA evidence receipts for all operations
---

# Receipt Generation Skill

This skill provides patterns for generating BIZRA evidence receipts.

## Receipt Schema (src/receipts.rs)

Every receipt MUST include:

```json
{
  "receipt_id": "type-session-timestamp-hash",
  "receipt_type": "orchestration|validation|gate|citation|novelty|telemetry",
  "timestamp": "ISO-8601",
  "session_id": "session identifier",
  "operation": "operation description",
  "status": "valid|pending|failed|corrupted",
  "data": {},
  "evidence_chain": ["parent_receipt_ids"],
  "integrity_hash": "SHA-256"
}
```

## Receipt Types

| Type | Purpose | Key Data |
|------|---------|----------|
| `pat_orchestration` | Full PAT pipeline | snr, novelty, gates |
| `pat_validation` | Validation result | gate_results, scores |
| `pat_gate` | Single gate | gate_id, checks, corrections |
| `pat_citation` | Citation check | valid_count, domains |
| `pat_novelty` | Novelty boost | original, boosted |
| `pat_telemetry` | Telemetry snapshot | metrics window |

## Storage

- Path: `docs/evidence/receipts/pat/`
- Format: JSON files, append-only
- Chain: `docs/evidence/receipts/pat/evidence_chain.jsonl`

## Python Usage

```python
from bizra_kernel.pat_receipt_pipeline import PATReceiptPipeline

pipeline = PATReceiptPipeline(session_id)
receipt = await pipeline.emit_orchestration_receipt(
    query="...",
    mode="STANDARD",
    snr_score=0.985,
    novelty_score=0.82,
    domain_count=3,
    gates_passed=5,
    gates_total=5,
    overall_pass=True,
    processing_time_ms=1500.0,
)
```

## Integrity

- All receipts are SHA-256 hashed
- Append-only storage (never delete/modify)
- Evidence chains link parent receipts
- Verify with `receipt.verify()`

## Key Files

- `src/receipts.rs` - Rust receipt schema
- `core/fate.py` - Python receipt handling
- `bizra_kernel/pat_receipt_pipeline.py` - PAT receipts
