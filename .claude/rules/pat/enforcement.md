---
paths:
  - "bizra_kernel/pat_*.py"
  - "constitution/pat_*.yaml"
  - "config/pat_enforcement/**"
---

# PAT Enforcement Rules

Peak Autonomous Agentic Think Tank - Maximum enforcement rules for LLM configuration layer.

## Core Thresholds (Non-Negotiable)

These thresholds are defined in `constitution/pat_enforcement_v1.yaml` and MUST be enforced:

| Metric | Threshold | Enforcement |
|--------|-----------|-------------|
| **SNR** | >= 0.98 | Block if below |
| **Novelty** | >= 0.75 | Block if below |
| **Domains** | >= 3 | Block if insufficient |
| **Practitioners** | >= 3 per domain | Warn if insufficient |
| **Ihsān** | >= 0.95 | Block if below (inherited) |

## 5 Validation Gates

All PAT outputs MUST pass through these gates in order:

### Gate 1: Pre-Reasoning (Domain Analysis)
- Verify 3+ domains from different clusters
- Calculate unrelatedness score >= 0.70
- **Correction**: Query adjacent clusters if insufficient

### Gate 2: Mid-Synthesis (Quality Checkpoint)
- Check running SNR >= 0.95
- Detect contradictions in thought graph
- Verify claim tags present
- **Correction**: Prune low-quality nodes

### Gate 3: Post-Synthesis (Final Validation)
- Verify final SNR >= 0.98
- Confirm novelty score >= 0.75
- Validate domain coverage
- **Correction**: Additional synthesis pass

### Gate 4: Practitioner Verification
- Ensure 3+ practitioners per domain
- Verify all are top 1% tier
- Check relevance scores valid
- **Correction**: Fetch additional practitioners

### Gate 5: Response Structure
- Confirm all 6 sections present
- Validate all claims tagged
- Verify evidence trail complete
- **Correction**: Reformat response

## Claim Tagging Requirements

Every factual claim MUST be tagged with one of:

| Tag | Weight | When to Use |
|-----|--------|-------------|
| `[MEASURED]` | 1.00 | Empirically verified data |
| `[IMPLEMENTED]` | 0.95 | Code exists and tested |
| `[DERIVED]` | 0.90 | Logically derived from facts |
| `[DESIGNED]` | 0.75 | Specification only |
| `[TARGET]` | 0.50 | Aspiration/goal |
| `[HYPOTHESIS]` | 0.40 | Requires testing |
| `[METAPHOR]` | 0.00 | Figurative only |
| `[NOVEL]` | 1.00 | Novel insight (distance >= 0.75) |
| `[CROSS_DOMAIN]` | 0.95 | Multi-domain synthesis |

## Code Patterns

### Instantiate PAT Engine
```python
from bizra_kernel.pat_enforcement_engine import PATEnforcementEngine
from bizra_kernel.got_orchestrator import GoTOrchestrator

engine = PATEnforcementEngine(session_id)
got = GoTOrchestrator(session_id)
```

### Run Full Validation
```python
result = await engine.run_full_validation(
    query=user_query,
    response=generated_response,
    got=thought_graph,
    context={
        "novelty_score": novelty_probe.probe(response).novelty_score,
        "practitioners": practitioner_registry.find_practitioners_for_domains(domains, query),
        "sections": response_formatter.extract_sections(response),
    }
)

if not result.overall_pass:
    # Handle validation failure
    for gate in result.gate_results:
        if gate.status == GateStatus.FAILED:
            print(f"Gate {gate.gate_name} failed: {gate.checks_failed}")
```

### Validate Cross-Pollination
```python
from bizra_kernel.pat_domain_validator import DomainCrossPollinationValidator

validator = DomainCrossPollinationValidator()
result = validator.validate(content)

if not result.gate_passed:
    # Apply corrections
    suggested = validator.suggest_expansion(result.clusters)
```

## Integration Points

When modifying PAT files, ensure integration with:

1. **SAPE Engine** (`bizra_kernel/sape_engine.py`)
   - NOVELTY probe type added
   - Weight: 0.12
   - PAT elevation priority boosts

2. **SNR Tracker** (`bizra_kernel/snr_tracker.py`)
   - TARGET_SNR_PAT = 0.98
   - check_pat_compliance() method
   - get_pat_statistics() method

3. **GoT Orchestrator** (`bizra_kernel/got_orchestrator.py`)
   - Multi-lens thought generation
   - Cross-domain synthesis
   - Cluster SNR calculation

## Testing

Run PAT validation tests:
```bash
python3 -c "
from bizra_kernel.pat_enforcement_engine import PATEnforcementEngine
from bizra_kernel.got_orchestrator import GoTOrchestrator
import asyncio

engine = PATEnforcementEngine('test')
got = GoTOrchestrator('test')
got.add_thought('Test thought', lens='test', snr=0.95)

result = asyncio.run(engine.run_full_validation(
    'test query', 'test response', got, {}
))
print(f'Pass: {result.overall_pass}, SNR: {result.snr_score:.3f}')
"
```

## Receipt Generation

All PAT validations MUST emit receipts to `docs/evidence/receipts/pat/`:
- `pat_session_id`
- `timestamp`
- `gate_results`
- `snr_score`
- `novelty_score`
- `integrity_hash` (SHA-256)
