# SNR Enforcement System

## Overview

The SNR (Signal-to-Noise Ratio) Enforcement System provides constitutional compliance for quality thresholds across all BIZRA operations. It implements fail-closed semantics, ensuring that operations with insufficient signal quality are rejected with full evidence trails.

**Status**: PRODUCTION
**Constitution**: `constitution/pat_enforcement_v1.yaml`
**Implementation**: `bizra_kernel/snr_enforcer.py`
**Integration**: `core/pci/gates.py` (SNR gate)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SNR Enforcement Flow                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Constitution Loading                                            │
│     ├─ Load PAT enforcement thresholds                             │
│     ├─ target_snr: 0.98 (aspirational)                            │
│     ├─ minimum_snr: 0.95 (hard threshold)                         │
│     └─ escalate_below: 0.90 (escalation trigger)                  │
│                                                                      │
│  2. Operation Execution                                             │
│     ├─ Calculate SNR score                                         │
│     └─ Create EnforcementContext                                   │
│                                                                      │
│  3. Threshold Check (FAIL-CLOSED)                                   │
│     ├─ SNR >= threshold → PASS                                     │
│     └─ SNR < threshold → REJECT                                    │
│                                                                      │
│  4. Receipt Emission (on rejection)                                 │
│     ├─ Generate receipt_id (SHA-256)                              │
│     ├─ Compute integrity_hash                                      │
│     ├─ Emit JSONL to docs/evidence/receipts/snr/                 │
│     └─ Log rejection with evidence                                 │
│                                                                      │
│  5. Integration Points                                              │
│     ├─ PCI Gate Chain (MEDIUM tier, <150ms)                       │
│     ├─ SNR Tracker (metrics recording)                            │
│     └─ FATE Escalation (if SNR < escalate_below)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Constitutional Thresholds

From `constitution/pat_enforcement_v1.yaml`:

```yaml
snr_integration:
  tier_name: "PAT"
  target_snr: 0.98      # Aspirational target (99.8% signal)
  minimum_snr: 0.95     # Hard threshold (operations fail below)
  escalate_below: 0.90  # FATE escalation trigger
```

### Threshold Semantics

- **target_snr (0.98)**: Aspirational target for peak performance
  - Used for monitoring and optimization
  - Not enforced as hard requirement
  - Exceeding this indicates excellent signal quality

- **minimum_snr (0.95)**: Constitutional hard requirement
  - Operations with SNR < 0.95 are **REJECTED**
  - Rejection receipts are emitted
  - Fail-closed: no exceptions

- **escalate_below (0.90)**: FATE escalation threshold
  - SNR < 0.90 triggers escalation to human review
  - Indicates systematic quality degradation
  - Requires investigation and correction

## SNR Calculation

SNR score is calculated by `SNRTracker`:

```python
SNR = (useful_tokens / total_tokens) × confidence × ethical_compliance × safety × directness
```

Where:
- **useful_tokens**: Tokens contributing to the answer (excludes filler, repetition)
- **total_tokens**: Total tokens in response
- **confidence**: Model confidence in response (0-1)
- **ethical_compliance**: Ihsān ethical score (0-1)
- **safety**: Safety compliance score (0-1)
- **directness**: How directly the response addresses the query (0-1)

## Usage

### Basic Enforcement

```python
from bizra_kernel.snr_enforcer import enforce_snr, OperationType

# Enforce SNR threshold
result = enforce_snr(
    operation_type=OperationType.REASONING,
    agent_id="pat-master-reasoner",
    snr_score=0.96,
    task_id="task-123",
    details={"query": "analyze code"}
)

if not result.passed:
    raise OperationRejected(
        code=result.rejection_code,
        message=result.message,
        receipt_id=result.receipt_id
    )
```

### Async Enforcement

```python
from bizra_kernel.snr_enforcer import enforce_snr_async

result = await enforce_snr_async(
    operation_type="synthesis",
    agent_id="creative-synthesizer",
    snr_score=0.97,
)
```

### Custom Enforcer Instance

```python
from bizra_kernel.snr_enforcer import SNREnforcer, EnforcementContext

enforcer = SNREnforcer(
    constitution_path="constitution/pat_enforcement_v1.yaml",
    emit_receipts=True,
    receipt_dir="docs/evidence/receipts/snr/"
)

context = EnforcementContext(
    operation_type=OperationType.VALIDATION,
    agent_id="sat-validator",
    snr_score=0.94,  # Below 0.95 threshold
    task_id="validation-456",
    details={"source": "PAT proposal"}
)

result = enforcer.enforce(context)
```

### Integration with SNR Tracker

```python
from bizra_kernel.snr_enforcer import SNREnforcer
from bizra_kernel.snr_tracker import SNRMetrics, SNRTracker

tracker = SNRTracker()
enforcer = SNREnforcer(snr_tracker=tracker)

# Record metrics
metrics = SNRMetrics(
    total_tokens=1000,
    useful_tokens=920,
    confidence_score=0.95,
    ethical_compliance=0.97,
    tool_directness=1.0,
    latency_ms=250,
    agent_role="master-reasoner"
)

enforcer.record_metrics(metrics)

# Enforce threshold
result = enforcer.enforce(context)
```

## Operation Types

SNR enforcement supports per-operation-type thresholds:

| Operation Type | Default Threshold | Description |
|---------------|-------------------|-------------|
| `REASONING` | 0.95 | Strategic reasoning and analysis |
| `SYNTHESIS` | 0.95 | Creative synthesis and generation |
| `VALIDATION` | 0.95 | SAT validation operations |
| `RETRIEVAL` | 0.95 | Knowledge retrieval queries |
| `GENERATION` | 0.95 | Content generation |
| `PAT_EXECUTION` | 0.95 | PAT agent execution |
| `SAT_VALIDATION` | 0.95 | SAT validation |
| `SAPE_PROBE` | 0.95 | SAPE probe execution |
| `DEFAULT` | 0.95 | Fallback for unclassified ops |

### Custom Operation Thresholds

Override thresholds in constitution:

```yaml
snr_integration:
  target_snr: 0.98
  minimum_snr: 0.95
  operation_thresholds:
    reasoning: 0.97      # Higher threshold for reasoning
    synthesis: 0.96      # Higher for synthesis
    retrieval: 0.92      # Lower for retrieval
```

## Receipt Emission

### Receipt Schema

Rejection receipts follow this schema:

```json
{
  "receipt_id": "snr-reject-a1b2c3d4e5f6g7h8",
  "timestamp": "2026-01-27T12:34:56.789Z",
  "rejection_code": 7,
  "rejection_name": "REJECT_SNR_BELOW_MIN",
  "operation_type": "reasoning",
  "agent_id": "pat-master-reasoner",
  "task_id": "task-123",
  "session_id": null,
  "snr_score": 0.93,
  "threshold": 0.95,
  "target_snr": 0.98,
  "delta": -0.02,
  "message": "SNR enforcement REJECTED: ...",
  "evidence": {
    "operation_type": "reasoning",
    "agent_id": "pat-master-reasoner",
    "snr_score": 0.93,
    "threshold": 0.95,
    "target_snr": 0.98,
    "delta": -0.02,
    "context_details": {...}
  },
  "context": {...},
  "integrity_hash": "sha256:abc123..."
}
```

### Receipt Storage

Receipts are stored in JSONL format:
- **Path**: `docs/evidence/receipts/snr/YYYY-MM-DD.jsonl`
- **Format**: One JSON object per line
- **Append-only**: Never modified after creation
- **Integrity**: Each receipt has SHA-256 integrity hash

### Receipt Querying

```bash
# View today's rejections
cat docs/evidence/receipts/snr/$(date +%Y-%m-%d).jsonl | jq '.'

# Count rejections by agent
cat docs/evidence/receipts/snr/*.jsonl | jq -r '.agent_id' | sort | uniq -c

# Find rejections below 0.90 (escalation candidates)
cat docs/evidence/receipts/snr/*.jsonl | jq 'select(.snr_score < 0.90)'

# Verify integrity
cat docs/evidence/receipts/snr/*.jsonl | python scripts/verify_receipt_integrity.py
```

## PCI Gate Integration

SNR enforcement is integrated into the PCI gate chain at the MEDIUM tier:

```python
from core.pci import GateChain, PCIEnvelope

chain = GateChain(
    current_policy_hash=policy_hash,
    current_state_hash=state_hash,
    snr_threshold=0.95,
    use_snr_enforcer=True  # Enable enforcer with receipt emission
)

passed, rejection, results = chain.verify(envelope)

if not passed:
    # SNR gate failed, receipt already emitted
    print(f"Rejection: {rejection.message}")
    print(f"Receipt ID: {results[gate_index].details['receipt_id']}")
```

### Gate Sequence

```
CHEAP tier (<10ms):
  1. SCHEMA
  2. SIGNATURE
  3. TIMESTAMP
  4. REPLAY
  5. ROLE

MEDIUM tier (<150ms):
  6. SNR ← Enforcer integrated here
  7. IHSAN
  8. POLICY

EXPENSIVE tier (<2000ms):
  9. FATE
  10. FORMAL
```

## Fail-Closed Semantics

The enforcer implements strict fail-closed semantics:

### Rejection Scenarios

1. **SNR below threshold**
   - Result: REJECT
   - Code: REJECT_SNR_BELOW_MIN (7)
   - Receipt: YES

2. **Missing SNR score**
   - Result: REJECT
   - Code: REJECT_INTERNAL_ERROR (99)
   - Receipt: YES

3. **Constitution load error**
   - Result: Fall back to default thresholds
   - Code: N/A (continues with defaults)
   - Receipt: NO

4. **Receipt emission error**
   - Result: Log error, continue rejection
   - Code: Same as rejection
   - Receipt: ATTEMPTED (logged as failed)

### Never Silently Fail

```python
# ✅ CORRECT: Fail visibly
result = enforce_snr(...)
if not result.passed:
    raise OperationRejected(result.rejection_code, result.message)

# ❌ WRONG: Silent failure
result = enforce_snr(...)
if not result.passed:
    logger.warning("SNR below threshold")  # DON'T DO THIS
    # Never proceed without raising error
```

## Statistics & Monitoring

### Enforcement Statistics

```python
enforcer = get_snr_enforcer()
stats = enforcer.get_statistics()

print(f"Total enforcements: {stats['enforcements']}")
print(f"Rejections: {stats['rejections']}")
print(f"Rejection rate: {stats['rejection_rate']:.2%}")
print(f"Receipts emitted: {stats['receipts_emitted']}")
```

### Metrics Dashboard

Key metrics to monitor:

- **Enforcement count**: Total enforcement checks
- **Rejection rate**: Percentage of operations rejected
- **Average SNR**: Average SNR across all operations
- **SNR by agent**: Per-agent SNR performance
- **SNR by operation type**: Per-operation-type SNR
- **Threshold compliance**: % operations meeting target_snr (0.98)

### Alerts

Set up alerts for:

- Rejection rate > 5% (indicates systematic quality issues)
- Average SNR < 0.92 (approaching escalation threshold)
- Any agent with consistent SNR < 0.95
- Receipt emission failures

## Testing

### Unit Tests

Run the test suite:

```bash
pytest tests/test_snr_enforcer.py -v
```

Test coverage includes:
- Threshold loading from constitution
- Enforcement pass/fail logic
- Receipt emission
- Integration with SNR tracker
- Async enforcement
- Edge cases (zero, exact threshold, perfect SNR)

### Integration Tests

```bash
pytest tests/test_pci_gates.py::test_snr_gate_with_enforcer -v
```

### Manual Testing

```python
from bizra_kernel.snr_enforcer import enforce_snr

# Test pass
result = enforce_snr("reasoning", "test-agent", 0.97)
assert result.passed

# Test reject
result = enforce_snr("reasoning", "test-agent", 0.92)
assert not result.passed
assert result.receipt_id is not None
```

## Troubleshooting

### Common Issues

**Issue**: Enforcer not loading constitution

```
WARNING: Constitution not found at constitution/pat_enforcement_v1.yaml, using defaults
```

**Solution**: Verify constitution file exists and is readable:

```bash
ls -l constitution/pat_enforcement_v1.yaml
cat constitution/pat_enforcement_v1.yaml | grep snr_integration
```

---

**Issue**: Receipts not being emitted

**Solution**: Check receipt directory permissions:

```bash
ls -ld docs/evidence/receipts/snr/
# Should be writable
chmod 755 docs/evidence/receipts/snr/
```

---

**Issue**: High rejection rate (>5%)

**Solution**: Investigate SNR calculation components:

```python
from bizra_kernel.snr_tracker import SNRTracker

tracker = SNRTracker()
stats = tracker.get_statistics()

# Check agent rankings
for ranking in stats['agent_rankings']:
    if ranking['avg_snr'] < 0.95:
        print(f"Low SNR agent: {ranking['agent']} ({ranking['avg_snr']:.4f})")

# Check for optimization candidates
candidates = tracker.detect_patterns()
for pattern in candidates:
    print(f"Optimization needed: {pattern}")
```

---

**Issue**: PCI gate not using enforcer

**Solution**: Ensure `use_snr_enforcer=True` when creating GateChain:

```python
chain = GateChain(
    ...,
    use_snr_enforcer=True  # Must be True
)
```

## Future Enhancements

### Planned Features

1. **Dynamic Threshold Adjustment**
   - Auto-adjust thresholds based on historical performance
   - Per-agent adaptive thresholds

2. **FATE Integration**
   - Automatic escalation when SNR < escalate_below
   - Human review routing for systematic failures

3. **Pattern Detection**
   - Detect recurring rejection patterns
   - Suggest SAPE elevation candidates

4. **Real-Time Monitoring**
   - WebSocket streaming of enforcement decisions
   - Live dashboard with SNR metrics

5. **Multi-Tier Thresholds**
   - Different thresholds for dev/staging/production
   - Environment-aware configuration

## References

- **Constitution**: `constitution/pat_enforcement_v1.yaml`
- **Implementation**: `bizra_kernel/snr_enforcer.py`
- **Tests**: `tests/test_snr_enforcer.py`
- **SNR Tracker**: `bizra_kernel/snr_tracker.py`
- **PCI Gates**: `core/pci/gates.py`
- **Reject Codes**: `core/pci/reject_codes.py`

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-27 | 1.0.0 | Initial implementation |
|  |  | - Constitution-based threshold loading |
|  |  | - Fail-closed enforcement semantics |
|  |  | - Receipt emission on rejection |
|  |  | - PCI gate integration |
|  |  | - Comprehensive test suite |

---

**Status**: PRODUCTION
**Maintainer**: BIZRA Core Team
**Last Updated**: 2026-01-27
