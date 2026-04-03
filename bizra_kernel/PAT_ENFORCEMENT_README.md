# PAT Enforcement System

**Peak Autonomous Think Tank — Maximum Enforcement Pipeline**

## Overview

The PAT Enforcement System is BIZRA's highest-tier validation framework, implementing 5 sequential gates that enforce strict quality, novelty, and ethical standards beyond the baseline Ihsan thresholds.

**Status**: CANONICAL — Constitution-driven enforcement
**Constitution**: `constitution/pat_enforcement_v1.yaml`

## Thresholds

| Metric | Threshold | Comparison to Ihsan |
|--------|-----------|---------------------|
| **SNR** | ≥ 0.98 | Stricter than Ihsan (0.95) |
| **Novelty** | ≥ 0.75 | Semantic distance from known patterns |
| **Ihsan** | ≥ 0.95 | Inherited from Ihsan constitution |
| **Domains** | ≥ 3 | Unrelatedness ≥ 0.70 |
| **Practitioners** | ≥ 3 per domain | Top 1% tier only |

## Architecture

### 5-Gate Sequential Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PAT REQUEST                                      │
│         (query, domains, synthesis, practitioners, response)             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  GATE 1: PRE-REASONING (Domain Analysis)                                │
│  ─────────────────────────────────────────────────────────────          │
│  • Domain count ≥ 3                                                     │
│  • Unrelatedness score ≥ 0.70                                           │
│  • Correction: expand_domains                                            │
│  • Fail Action: BLOCK                                                   │
│  • Latency Budget: 500ms                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ PASS
┌─────────────────────────────────────────────────────────────────────────┐
│  GATE 2: MID-SYNTHESIS (Running SNR Check)                              │
│  ─────────────────────────────────────────────────────────────          │
│  • Running SNR ≥ 0.95                                                   │
│  • No contradictions                                                     │
│  • Claim tags present                                                    │
│  • Correction: prune_low_quality_nodes                                  │
│  • Fail Action: RETRY_SYNTHESIS                                         │
│  • Latency Budget: 1000ms                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ PASS
┌─────────────────────────────────────────────────────────────────────────┐
│  GATE 3: POST-SYNTHESIS (Final Validation)                              │
│  ─────────────────────────────────────────────────────────────          │
│  • Final SNR ≥ 0.98                                                     │
│  • Novelty score ≥ 0.75                                                 │
│  • Domain coverage complete                                              │
│  • Ihsan score ≥ 0.95                                                   │
│  • Correction: additional_synthesis_pass                                │
│  • Fail Action: BLOCK                                                   │
│  • Latency Budget: 1500ms                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ PASS
┌─────────────────────────────────────────────────────────────────────────┐
│  GATE 4: PRACTITIONER (Elite Verification)                              │
│  ─────────────────────────────────────────────────────────────          │
│  • Practitioners ≥ 3 per domain                                         │
│  • All practitioners top 1% tier                                        │
│  • Relevance scores ≥ 0.60                                              │
│  • Correction: fetch_additional_practitioners                           │
│  • Fail Action: WARN (does not block)                                  │
│  • Latency Budget: 800ms                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ PASS or WARN
┌─────────────────────────────────────────────────────────────────────────┐
│  GATE 5: RESPONSE (6-Section Structure)                                 │
│  ─────────────────────────────────────────────────────────────          │
│  • Section count == 6                                                   │
│  • All claims tagged                                                     │
│  • Evidence trail complete                                               │
│  • Correction: reformat_response                                        │
│  • Fail Action: BLOCK                                                   │
│  • Latency Budget: 300ms                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ ALL GATES PASSED
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAT ENFORCEMENT RESULT                                │
│  • Receipt ID generated (SHA-256)                                       │
│  • Receipt emitted to docs/evidence/receipts/pat/                       │
│  • Telemetry recorded                                                    │
│  • Response returned to caller                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### Core Pipeline

**File**: `bizra_kernel/pat_enforcement_pipeline.py`

Main enforcement engine that orchestrates all 5 gates sequentially.

**Key Classes**:
- `PATEnforcementPipeline`: Main pipeline executor
- `PATRequest`: Input request structure
- `PATEnforcementResult`: Output result with receipt
- `GateResult`: Per-gate validation result
- `PATTelemetry`: Real-time monitoring

**Usage**:
```python
from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATRequest,
)

# Initialize pipeline
pipeline = PATEnforcementPipeline(
    snr_minimum=0.98,
    novelty_minimum=0.75,
    ihsan_minimum=0.95,
)

# Create request
request = PATRequest(
    session_id="session_001",
    task_id="task_001",
    query="Your query here",
    domains=[...],
    synthesis_nodes=[...],
    practitioners=[...],
    response_sections=[...],
)

# Execute enforcement
result = await pipeline.enforce(request)

# Check result
if result.passed:
    print(f"Passed! Receipt: {result.receipt_id}")
else:
    print(f"Failed at gate: {result.gate_results[-1].gate_id}")
```

### Supporting Components

#### 1. Domain Validator

**File**: `bizra_kernel/pat_domain_validator.py`

Validates domain diversity and cross-pollination for Gate 1.

**Key Functions**:
- Compute semantic unrelatedness between domains
- Detect cross-domain synthesis connections
- Expand domain list when insufficient

**Usage**:
```python
from bizra_kernel.pat_domain_validator import PATDomainValidator

validator = PATDomainValidator()
result = await validator.validate(domains, context)

if not result.passed:
    expanded = await validator.expand_domains(domains, query)
```

#### 2. Novelty Probe

**File**: `bizra_kernel/pat_novelty_probe.py`

Measures semantic novelty of insights for Gate 3.

**Key Functions**:
- Compute semantic distance from known patterns
- Register new patterns when accepted
- Hypergraph boosting for novel connections

**Usage**:
```python
from bizra_kernel.pat_novelty_probe import PATNoveltyProbe

probe = PATNoveltyProbe(novelty_threshold=0.75)
result = await probe.probe(insight, embedding=None, domain="ML")

if result.passed:
    pattern_id = await probe.register_pattern(insight, domain)
```

#### 3. Citation Validator

**File**: `bizra_kernel/pat_citation_validator.py`

Validates elite practitioner credentials for Gate 4.

**Key Functions**:
- Verify practitioner tier (top 1%)
- Count practitioners per domain
- Compute relevance scores
- Fetch additional practitioners

**Usage**:
```python
from bizra_kernel.pat_citation_validator import PATCitationValidator

validator = PATCitationValidator()
result = await validator.validate(practitioners, domains, query)

if not result.passed:
    additional = await validator.fetch_additional_practitioners(
        domain="Database Systems",
        query=query,
        current_count=1,
    )
```

## 6-Section Response Structure

Gate 5 enforces this canonical structure:

### 1. Executive Synthesis
- **Format**: Bullet list (max 5 bullets)
- **Requirements**: Claim tagged, evidence linked
- **Example**:
  ```
  • [MEASURED] Parallel processing improves throughput by 3x
  • [DERIVED] Caching reduces database load by 40%
  ```

### 2. Domain Cross-Pollination Map
- **Format**: Structured map
- **Requirements**: Domains listed, connections mapped, synthesis points identified
- **Example**:
  ```
  Distributed Systems ←→ Machine Learning: RL for consensus
  Database Systems ←→ ML: Learned indexes
  ```

### 3. Elite Practitioner Anchoring
- **Format**: Practitioner table
- **Requirements**: Min 3 per domain, tier verified, relevance scored
- **Example**:
  ```
  | Name | Tier | Domain | Relevance |
  |------|------|--------|-----------|
  | Dr. Smith | top_1% | Distributed | 0.85 |
  ```

### 4. Novel Insight Synthesis
- **Format**: Narrative
- **Requirements**: Novelty ≥ 0.75, semantic distance verified, source patterns cited
- **Example**:
  ```
  [NOVEL] Using quantum-inspired algorithms for distributed consensus
  shows 5x faster convergence (semantic distance: 0.82 from classical approaches)
  ```

### 5. Validation Evidence Trail
- **Format**: Evidence table
- **Requirements**: Gate statuses, SNR scores, Ihsan scores, receipt IDs
- **Example**:
  ```
  | Gate | Status | Latency | Score |
  |------|--------|---------|-------|
  | Gate 1 | PASSED | 450ms | domain_count=3 |
  | Gate 2 | PASSED | 980ms | snr=0.97 |
  ```

### 6. Actionable Recommendations
- **Format**: Categorized list
- **Requirements**: what_we_know, what_we_assume, what_we_test_next
- **Example**:
  ```
  **What We Know**:
  - Parallel execution scales linearly to 16 cores

  **What We Assume**:
  - Network latency remains under 10ms

  **What We Test Next**:
  - Benchmark at 1000 concurrent connections
  ```

## Claim Tags

PAT enforces claim tagging from the constitution:

| Tag | Weight | Description | Requires |
|-----|--------|-------------|----------|
| **MEASURED** | 1.00 | Empirically verified data | citation, methodology |
| **IMPLEMENTED** | 0.95 | Code exists, tests passed | code_path, test_evidence |
| **DERIVED** | 0.90 | Logically derived from facts | source_claims |
| **DESIGNED** | 0.75 | Specification only | design_doc |
| **TARGET** | 0.50 | Aspiration/goal | - |
| **HYPOTHESIS** | 0.40 | Requires testing | test_plan |
| **METAPHOR** | 0.00 | Figurative language | - |
| **NOVEL** | 1.00 | Novel insight (novelty ≥ 0.75) | novelty_score |
| **CROSS_DOMAIN** | 0.95 | Multi-domain synthesis | source_domains, connection_type |

## Correction Protocols

Each gate can attempt corrections before failing:

### 1. expand_domains
- **Trigger**: domain_count < 3
- **Action**: Query adjacent clusters for related domains
- **Max Retries**: 3

### 2. prune_low_quality_nodes
- **Trigger**: running_snr < 0.95
- **Action**: Remove nodes below SNR 0.85
- **Preserve**: Highest SNR per lens

### 3. additional_synthesis_pass
- **Trigger**: final_snr < 0.98 or novelty < 0.75
- **Action**: Synthesize new cross-domain insight
- **Max Retries**: 2

### 4. fetch_additional_practitioners
- **Trigger**: practitioners_per_domain < 3
- **Action**: Query practitioner registry
- **Fallback**: Use top available

### 5. reformat_response
- **Trigger**: section_count != 6 or missing tags
- **Action**: Apply template strict
- **Validate After**: true

## Receipt Schema

PAT receipts are emitted to `docs/evidence/receipts/pat/`:

```json
{
  "receipt_type": "PAT_ENFORCEMENT",
  "version": "1.0",
  "session_id": "session_001",
  "task_id": "task_001",
  "passed": true,
  "gate_results": [
    {
      "gate_id": "gate_1_pre_reasoning",
      "status": "PASSED",
      "passed": true,
      "latency_ms": 450,
      "checks": {
        "domain_count_ok": true,
        "unrelatedness_ok": true
      },
      "scores": {
        "domain_count": 3.0,
        "unrelatedness_score": 0.75
      },
      "correction_attempts": 0,
      "evidence": [
        "Domain count: 3 (required: 3)",
        "Unrelatedness score: 0.7500 (required: 0.70)"
      ]
    }
    // ... other gates
  ],
  "final_snr": 0.98,
  "final_novelty": 0.80,
  "final_ihsan": 0.97,
  "domain_count": 3,
  "practitioner_count": 4,
  "total_latency_ms": 3500,
  "correction_attempts": 0,
  "receipt_id": "a3f7b2c8d1e4f5a6",
  "receipt_path": "docs/evidence/receipts/pat/a3f7b2c8d1e4f5a6.json",
  "timestamp": "2026-01-27T12:00:00.000Z"
}
```

## Integration Points

### Existing Components

The PAT enforcement pipeline integrates with:

1. **ihsan_gate.py**: Ethical compliance (Gate 3)
2. **snr_tracker.py**: SNR monitoring (Gates 2, 3)
3. **core/sape.py**: SAPE elevation system
4. **core/fate.py**: FATE escalation (on gate failures)
5. **core/pci/receipt.py**: Receipt generation protocol

### External Systems

- **BIZRA-DATA-LAKE**: Pattern database for novelty detection
- **Vector embeddings**: Semantic distance computation
- **Practitioner registry**: Elite expert database
- **Knowledge graph**: Domain relationships

## Testing

**Test File**: `bizra_kernel/test_pat_enforcement.py`

Run all tests:
```bash
pytest bizra_kernel/test_pat_enforcement.py -v --asyncio-mode=auto
```

Run specific test:
```bash
pytest bizra_kernel/test_pat_enforcement.py::test_full_pipeline_pass -v
```

Test coverage:
- Individual gate validation
- Full pipeline pass/fail
- Correction mechanisms
- Receipt generation
- Telemetry tracking
- Integration with components

## Telemetry

The `PATTelemetry` class tracks real-time metrics:

```python
from bizra_kernel.pat_enforcement_pipeline import PATTelemetry

telemetry = PATTelemetry()

# Record enforcement
telemetry.record_enforcement(result)

# Get stats
stats = telemetry.get_stats()
print(f"Pass rate: {stats['pass_rate']:.2%}")
print(f"Average latency: {stats['average_latency_ms']}ms")
print(f"Gate failures: {stats['gate_failure_counts']}")
```

**Metrics**:
- Total enforcements
- Pass/fail counts
- Per-gate failure counts
- Average latency
- Pass rate

## Performance

**Latency Budgets** (from constitution):

| Gate | Budget | Typical |
|------|--------|---------|
| Gate 1 | 500ms | ~450ms |
| Gate 2 | 1000ms | ~980ms |
| Gate 3 | 1500ms | ~1400ms |
| Gate 4 | 800ms | ~750ms |
| Gate 5 | 300ms | ~280ms |
| **Total** | **4100ms** | **~3500ms** |

## Fail-Closed Behavior

PAT enforcement is **fail-closed**:

- Any gate failure BLOCKS execution (except Gate 4 which warns)
- No silent failures — all failures emit rejection receipts
- Correction attempts are logged with evidence
- Failed requests do not proceed to next gate

## Future Enhancements

### Planned Features

1. **Async Parallel Gate Execution** (where safe)
2. **LLM-based Correction Strategies**
3. **Dynamic Threshold Adjustment** (based on stakes)
4. **Real-time Dashboard** (live gate monitoring)
5. **A/B Testing Framework** (gate variations)

### Integration Targets

- `/peak` command in Claude Code
- REST API endpoint: `POST /v1/pat/enforce`
- Integration with PAT Unified Orchestrator
- SAPE elevation feedback loop

## See Also

- **Constitution**: `constitution/pat_enforcement_v1.yaml`
- **Ihsan Gate**: `bizra_kernel/ihsan_gate.py`
- **SNR Tracker**: `bizra_kernel/snr_tracker.py`
- **SAPE Engine**: `core/sape.py`
- **FATE Engine**: `core/fate.py`

---

**Last Updated**: 2026-01-27
**Version**: 1.0
**Status**: CANONICAL
