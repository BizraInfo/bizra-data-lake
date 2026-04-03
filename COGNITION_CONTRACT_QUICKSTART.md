# Unified Cognition Contract - Quick Start

**Status:** ✓ Python (8010) | ⏳ Rust (8080)
**Created:** 2026-02-14

---

## Files Created

1. **Contract Schema**
   - `config/cognition_contract.json` - JSON Schema defining request/response formats

2. **Python Implementation**
   - `core/main.py` - Added `/v1/cognition` endpoint (~300 lines)
   - Includes: FATE gating, Ihsān floor, SNR tier, receipt emission

3. **Validation**
   - `scripts/validate_cognition_contract.py` - Validates schema and implementation

4. **Documentation**
   - `docs/architecture/UNIFIED_COGNITION_CONTRACT_v1.md` - Full specification

---

## Quick Test

```bash
# 1. Start Python kernel
python -m core.main

# 2. Test endpoint (requires BIZRA_API_TOKEN)
curl -X POST http://localhost:8010/v1/cognition \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain the Ihsān principle",
    "ihsan_floor": 0.95
  }'

# 3. Validate implementation
python scripts/validate_cognition_contract.py
```

---

## API Summary

### Request

```json
{
  "task": "Your task here",           // Required
  "user_id": "alice",                 // Optional, default: "anonymous"
  "ihsan_floor": 0.95,                // Optional, default: 0.95
  "snr_floor": 7.0,                   // Optional, default: 7.0
  "context": {"key": "value"},        // Optional
  "taint_secrecy": "Internal",        // Optional: Public/Internal/Confidential/Secret
  "taint_integrity": "Validated"      // Optional: Untrusted/Validated/Attested/Sovereign
}
```

### Response (Success)

```json
{
  "result": "Execution result",
  "ihsan_score": 0.97,
  "snr_tier": "T5",                   // T0-T6
  "receipt_id": "abc123...",
  "total_latency_ms": 1690
}
```

### Response (Error)

```json
{
  "error": "Detailed error message",
  "code": "SAT_BLOCKED",              // SAT_BLOCKED | IHSAN_GATE_FAILED | EXECUTION_FAILED | INTERNAL_ERROR
  "receipt_id": "abc123..."
}
```

---

## SNR Tiers

| Tier | Ihsān Range | Description |
|------|-------------|-------------|
| T6 | 0.99-1.00 | Transcendent |
| T5 | 0.95-0.99 | Outstanding |
| T4 | 0.90-0.95 | Excellent |
| T3 | 0.85-0.90 | Good |
| T2 | 0.80-0.85 | Acceptable |
| T1 | 0.70-0.80 | Basic |
| T0 | 0.00-0.70 | Below threshold |

---

## Error Handling

All errors are **fail-closed** with receipt emission:

1. **SAT_BLOCKED** (403): FATE gate rejected the request (malicious/harmful)
2. **IHSAN_GATE_FAILED** (403): Achieved Ihsān score below `ihsan_floor`
3. **EXECUTION_FAILED** (500): LLM execution error
4. **INTERNAL_ERROR** (500): Unexpected error

---

## Security Features

- ✓ **FATE Gating**: Pre-validates all requests (cannot be disabled)
- ✓ **Ihsān Floor**: Enforces minimum ethical score
- ✓ **Receipt Emission**: Append-only evidence for all requests
- ✓ **Fail-Closed**: Errors block execution, never proceed silently
- ✓ **Taint Tracking**: Secrecy + integrity metadata
- ⏳ **Ed25519 Signatures**: TODO (v1.1.0)

---

## Integration Points

### Python Kernel (8010)

```python
# Already implemented in core/main.py
@app.post("/v1/cognition")
async def unified_cognition(req: UnifiedCognitionRequest):
    # 1. Validate schema
    # 2. FATE gate
    # 3. SAPE execution
    # 4. Ihsān enforcement
    # 5. Receipt emission
    return UnifiedCognitionResponse(...)
```

### Rust Core (8080) - TODO

```rust
// src/http.rs
#[post("/v1/cognition")]
async fn unified_cognition(req: Json<UnifiedCognitionRequest>) -> Result<Json<UnifiedCognitionResponse>> {
    // 1. FATE gate
    // 2. PAT/SAT execution
    // 3. Ihsān scoring
    // 4. Receipt emission
    // 5. Ed25519 signing
    Ok(Json(response))
}
```

---

## Receipts

All requests emit receipts to `docs/evidence/receipts/`:

```
docs/evidence/receipts/
├── kernel_request_20260214_123456Z_abc123/
│   ├── receipt.json          # Main receipt
│   └── evidence.json         # Optional evidence artifacts
└── kernel_request_20260214_123457Z_def456/
    └── receipt.json
```

Receipt schema:

```json
{
  "schema": "bizra_unified_cognition_receipt_v1",
  "generated_at": "2026-02-14T12:34:56Z",
  "request_id": "abc123",
  "endpoint": "/v1/cognition",
  "status": "SUCCESS",
  "ihsan_score": 0.97,
  "snr_tier": "T5",
  "total_latency_ms": 1690,
  "fate_seal": {...},
  "integrity_hash": "sha256:..."
}
```

---

## Next Steps

### Immediate (Current Sprint)

- [x] Create contract schema
- [x] Implement Python endpoint
- [x] Add validation script
- [ ] Add integration tests
- [ ] Implement Ed25519 signing

### Short-term (Next Sprint)

- [ ] Implement Rust endpoint
- [ ] Add cross-layer tests
- [ ] Benchmark latency comparison
- [ ] Add Prometheus metrics

### Long-term (Future)

- [ ] WebSocket streaming support
- [ ] Multi-turn dialogue
- [ ] Circuit breaker patterns
- [ ] Rate limiting per user_id

---

## Testing

### Unit Tests

```bash
# Python tests (TODO)
pytest tests/test_cognition_contract.py -v

# Rust tests (TODO)
cargo test cognition::tests --lib
```

### Integration Tests

```bash
# Start services
docker compose up -d

# Run validation
python scripts/validate_cognition_contract.py

# Test Python endpoint
curl -X POST http://localhost:8010/v1/cognition \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -d '{"task": "Test request"}'

# Test Rust endpoint (TODO)
curl -X POST http://localhost:8080/v1/cognition \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -d '{"task": "Test request"}'
```

### Load Testing

```bash
# Using hey (TODO)
hey -n 1000 -c 10 -m POST \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -d '{"task": "Load test"}' \
  http://localhost:8010/v1/cognition
```

---

## Troubleshooting

### Python Endpoint Not Found

```bash
# Check server is running
curl http://localhost:8010/healthz

# Check logs
python -m core.main
```

### Schema Validation Fails

```bash
# Validate schema
python scripts/validate_cognition_contract.py

# Check schema exists
ls -la config/cognition_contract.json
```

### FATE Gate Rejects All Requests

```bash
# Check FATE configuration
export BIZRA_FATE_STRICT=0  # Lenient mode (dev only)

# Check constitution
cat constitution/ihsan_v1.yaml
```

---

## Related Files

| File | Purpose |
|------|---------|
| `config/cognition_contract.json` | Contract schema |
| `core/main.py` | Python endpoint implementation |
| `scripts/validate_cognition_contract.py` | Validation script |
| `docs/architecture/UNIFIED_COGNITION_CONTRACT_v1.md` | Full specification |
| `core/fate.py` | FATE engine |
| `core/sape.py` | SAPE planning |
| `constitution/ihsan_v1.yaml` | Ihsān constitution |

---

**Questions?** See `docs/architecture/UNIFIED_COGNITION_CONTRACT_v1.md` for details.
