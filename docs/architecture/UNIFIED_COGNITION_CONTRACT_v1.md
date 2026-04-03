# BIZRA Unified Cognition Contract v1

**Status:** IMPLEMENTED (Python) | TODO (Rust)
**Created:** 2026-02-14
**Schema:** `config/cognition_contract.json`
**Endpoints:** `POST /v1/cognition` (Python: 8010, Rust: 8080)

---

## Overview

The Unified Cognition Contract defines a standardized API interface between BIZRA's dual implementation layers:

- **Rust Core** (port 8080): Production PAT/SAT engine, MCP, A2A, SAPE, FATE
- **Python Kernel** (port 8010): FastAPI, SAPE planning, FATE engine, LLM routing

This contract ensures both layers can interoperate seamlessly while maintaining:
- Receipt-native evidence tracking
- Fail-closed error handling
- Ihsān-gated execution
- SNR tier classification
- Taint tracking (secrecy + integrity)

---

## API Specification

### Request Format

```json
{
  "task": "Analyze the trade-offs between...",
  "user_id": "alice",
  "ihsan_floor": 0.95,
  "snr_floor": 7.0,
  "context": {
    "domain": "ethics",
    "urgency": "high"
  },
  "taint_secrecy": "Internal",
  "taint_integrity": "Validated"
}
```

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `task` | string | ✓ | - | Task description (1-10,000 chars) |
| `user_id` | string | | "anonymous" | User identifier |
| `ihsan_floor` | number | | 0.95 | Minimum Ihsān score (0.0-1.0) |
| `snr_floor` | number | | 7.0 | Minimum SNR score |
| `context` | object | | {} | Additional context (key-value pairs) |
| `taint_secrecy` | enum | | "Internal" | Public, Internal, Confidential, Secret |
| `taint_integrity` | enum | | "Untrusted" | Untrusted, Validated, Attested, Sovereign |

### Response Format

```json
{
  "result": "Analysis: ...",
  "ihsan_score": 0.97,
  "snr_tier": "T5",
  "receipt_id": "abc123...",
  "signature": "ed25519:deadbeef...",
  "signer_public_key": "ed25519:cafebabe...",
  "sat_validation_ms": 234,
  "pat_execution_ms": 1456,
  "total_latency_ms": 1690
}
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `result` | string | ✓ | Execution result |
| `ihsan_score` | number | ✓ | Achieved Ihsān score (0.0-1.0) |
| `snr_tier` | enum | ✓ | SNR tier: T0-T6 |
| `receipt_id` | string | ✓ | Evidence receipt identifier |
| `signature` | string | | Ed25519 signature (hex) |
| `signer_public_key` | string | | Signer public key (hex) |
| `sat_validation_ms` | integer | | SAT validation latency |
| `pat_execution_ms` | integer | | PAT execution latency |
| `total_latency_ms` | integer | ✓ | Total request latency |

### Error Format

```json
{
  "error": "FATE rejection: malicious intent detected",
  "code": "SAT_BLOCKED",
  "escalation_id": "esc_xyz",
  "receipt_id": "abc123..."
}
```

#### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `SAT_BLOCKED` | 403 | FATE gate rejected (pre-validation) |
| `IHSAN_GATE_FAILED` | 403 | Ihsān score below floor |
| `EXECUTION_FAILED` | 500 | LLM execution error |
| `INTERNAL_ERROR` | 500 | Unexpected error |

---

## SNR Tier Classification

SNR tiers are computed from Ihsān scores:

| Tier | Ihsān Range | Description |
|------|-------------|-------------|
| T0 | 0.00-0.70 | Below threshold |
| T1 | 0.70-0.80 | Basic |
| T2 | 0.80-0.85 | Acceptable |
| T3 | 0.85-0.90 | Good |
| T4 | 0.90-0.95 | Excellent |
| T5 | 0.95-0.99 | Outstanding |
| T6 | 0.99-1.00 | Transcendent |

---

## Request Flow

```
Client Request
    ↓
[Schema Validation] (best-effort, log warnings)
    ↓
[FATE Gate Pre-Validation]
    ├─ REJECTED → 403 SAT_BLOCKED
    └─ APPROVED → Continue
        ↓
[SAPE Execution]
    ├─ LLMCallError → 500 EXECUTION_FAILED
    └─ Success → Continue
        ↓
[Ihsān Score Computation]
    ↓
[Ihsān Floor Enforcement]
    ├─ Below Floor → 403 IHSAN_GATE_FAILED
    └─ Passed → Continue
        ↓
[SNR Tier Classification]
    ↓
[Receipt Emission]
    ↓
Response
```

---

## Security Features

### Fail-Closed Error Handling

All error paths **block execution** and emit rejection receipts:

```python
if seal.verdict == "REJECTED":
    payload = {
        "schema": "bizra_unified_cognition_receipt_v1",
        "status": "BLOCKED_BY_FATE",
        "fate_seal": seal.model_dump(),
    }
    _write_receipt(payload)
    raise HTTPException(status_code=403, detail={...})
```

### FATE Gating

FATE (Fail-Safe Agentic Trust Escalation) validates **every request** before execution:

- **Pre-validation:** Blocks malicious/harmful prompts
- **Post-validation:** (Future) Validates generated outputs
- **No bypass:** Cannot be disabled per-request

### Ihsān Floor Enforcement

Requests with `ihsan_floor` parameter enforce minimum ethical score:

```python
if ihsan_score < req.ihsan_floor:
    # Block and emit receipt
    raise HTTPException(status_code=403, ...)
```

### Taint Tracking

Requests include taint metadata for auditability:

- **Secrecy:** Public, Internal, Confidential, Secret
- **Integrity:** Untrusted, Validated, Attested, Sovereign

---

## Receipt Schema

Every request emits an append-only receipt:

```json
{
  "schema": "bizra_unified_cognition_receipt_v1",
  "generated_at": "2026-02-14T12:34:56.789Z",
  "truth_label": "MEASURED",
  "request_id": "abc123...",
  "endpoint": "/v1/cognition",
  "status": "SUCCESS",
  "ihsan_score": 0.97,
  "snr_tier": "T5",
  "sat_validation_ms": 234,
  "total_latency_ms": 1690,
  "model_used": "deepseek-r1:8b",
  "provider_used": "ollama",
  "attempts": [...],
  "fate_seal": {...}
}
```

Receipts are stored in `docs/evidence/receipts/` with SHA-256 integrity hashes.

---

## Implementation Status

### Python Kernel (Port 8010) ✓

**File:** `core/main.py`
**Endpoint:** `POST /v1/cognition`
**Status:** IMPLEMENTED

Features:
- ✓ Schema validation (best-effort)
- ✓ FATE gate pre-validation
- ✓ SAPE execution delegation
- ✓ Ihsān floor enforcement
- ✓ SNR tier computation
- ✓ Receipt emission
- ✗ Ed25519 signing (TODO)

### Rust Core (Port 8080) TODO

**File:** `src/http.rs`
**Endpoint:** `POST /v1/cognition`
**Status:** NOT IMPLEMENTED

Required:
- [ ] Implement `UnifiedCognitionRequest` struct
- [ ] Implement `UnifiedCognitionResponse` struct
- [ ] Add `/v1/cognition` endpoint handler
- [ ] Integrate with existing PAT/SAT engine
- [ ] Add Ed25519 signature generation
- [ ] Emit receipts to `docs/evidence/receipts/`
- [ ] Add integration tests

---

## Usage Examples

### cURL

```bash
curl -X POST http://localhost:8010/v1/cognition \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain the Ihsān principle in BIZRA",
    "user_id": "developer",
    "ihsan_floor": 0.95,
    "context": {
      "domain": "ethics",
      "audience": "technical"
    }
  }'
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8010/v1/cognition",
    headers={"Authorization": f"Bearer {api_token}"},
    json={
        "task": "Analyze the trade-offs between...",
        "ihsan_floor": 0.95,
        "snr_floor": 7.0,
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Result: {data['result']}")
    print(f"Ihsān: {data['ihsan_score']:.4f} ({data['snr_tier']})")
    print(f"Receipt: {data['receipt_id']}")
else:
    error = response.json()
    print(f"Error: {error['code']} - {error['error']}")
```

### Rust Client (Future)

```rust
use bizra::cognition::UnifiedCognitionRequest;

let req = UnifiedCognitionRequest {
    task: "Explain SAPE probing".to_string(),
    user_id: Some("admin".to_string()),
    ihsan_floor: Some(0.95),
    ..Default::default()
};

let resp = client.post("/v1/cognition")
    .json(&req)
    .send()
    .await?;

println!("Ihsān: {:.4f}", resp.ihsan_score);
println!("Receipt: {}", resp.receipt_id);
```

---

## Validation Script

Run the validation script to verify implementation:

```bash
python scripts/validate_cognition_contract.py
```

Output:

```
[1/3] Loading contract schema: config/cognition_contract.json
✓ Schema loaded successfully

[2/3] Validating schema structure
  ✓ UnifiedCognitionRequest defined
  ✓ UnifiedCognitionResponse defined
  ✓ CognitionError defined

[3/3] Validating Python implementation
  ✓ /v1/cognition endpoint defined
  ✓ FATE gating present
  ✓ Ihsan floor enforcement present

============================================================
✓ ALL VALIDATIONS PASSED
============================================================
```

---

## Next Steps

### Phase 1: Python Stabilization (Current)

- [x] Create contract schema (`config/cognition_contract.json`)
- [x] Implement Python endpoint (`core/main.py`)
- [x] Add validation script
- [ ] Add integration tests (`tests/test_cognition_contract.py`)
- [ ] Implement Ed25519 signing

### Phase 2: Rust Implementation (Next)

- [ ] Create Rust types in `src/cognition.rs`
- [ ] Add endpoint in `src/http.rs`
- [ ] Integrate with PAT/SAT engine
- [ ] Add receipt emission
- [ ] Add integration tests

### Phase 3: Cross-Layer Testing

- [ ] Test Rust→Python delegation
- [ ] Test Python→Rust delegation
- [ ] Benchmark latency comparison
- [ ] Load testing (1000 req/s)
- [ ] Chaos testing (service failures)

### Phase 4: Production Hardening

- [ ] Rate limiting per user_id
- [ ] Request timeout enforcement
- [ ] Circuit breaker for LLM failures
- [ ] Metrics dashboard (Prometheus/Grafana)
- [ ] Alert thresholds (Ihsān < 0.95, latency > 5s)

---

## Related Documentation

- **Contract Schema:** `config/cognition_contract.json`
- **Python Implementation:** `core/main.py` (line 1970+)
- **FATE Engine:** `core/fate.py`
- **SAPE Planning:** `core/sape.py`
- **Receipt Schema:** `src/receipts.rs` (Rust), `core/fate.py` (Python)
- **Ihsān Constitution:** `constitution/ihsan_v1.yaml`

---

## Changelog

### v1.0.0 (2026-02-14)

- Initial implementation
- Python endpoint at `/v1/cognition` (port 8010)
- Schema validation (best-effort)
- FATE gate pre-validation
- Ihsān floor enforcement
- SNR tier classification
- Receipt emission

### Future Versions

- **v1.1.0:** Ed25519 signature generation
- **v1.2.0:** Rust endpoint implementation
- **v2.0.0:** Cross-layer streaming (WebSocket)
- **v3.0.0:** Multi-turn dialogue support

---

**Maintainer:** BIZRA Core Team
**Contact:** See `README.md` for contribution guidelines
