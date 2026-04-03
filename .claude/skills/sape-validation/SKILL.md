---
name: sape-validation
description: SAPE (Symbolic-Abstraction Probe Elevation) validation system
---

# SAPE Validation Skill

SAPE is BIZRA's 9-probe verification system with pattern elevation.

## 9 Probes

| Probe | Weight | Threshold | Purpose |
|-------|--------|-----------|---------|
| threat_scan | 0.15 | 0.95 | Security threats |
| compliance | 0.12 | 0.95 | Policy compliance |
| bias | 0.12 | 0.90 | Bias detection |
| user_benefit | 0.12 | 0.85 | User value |
| correctness | 0.12 | 0.95 | Factual accuracy |
| safety | 0.15 | 0.95 | Safety checks |
| groundedness | 0.08 | 0.85 | Evidence backing |
| relevance | 0.07 | 0.80 | Task relevance |
| fluency | 0.07 | 0.80 | Output quality |
| **novelty** | 0.12 | 0.75 | PAT: Novel insights |

## Pattern Elevation

Patterns with >3 repetitions get elevated to kernel shortcuts:

```python
if pattern.occurrences > 3:
    sape.elevate_pattern(pattern.hash, optimized_shortcut)
```

## PAT Extension

SAPE is extended for PAT with:
- `NOVELTY` probe type (semantic distance)
- PAT elevation priority boost (1.3x for novel insights)

## Key Files

- `src/sape.rs` - Rust SAPE engine
- `core/sape.py` - Python SAPE logic
- `bizra_kernel/sape_engine.py` - Extended engine
- `bizra_kernel/pat_novelty_probe.py` - Novelty probe

## Usage

```python
from bizra_kernel.sape_engine import SAPEEngine, SapeProbeType

engine = SAPEEngine()
result = await engine.run_probes(content, context)

if result.passed:
    print(f"SAPE Score: {result.overall_score}")
```

## Validation Command

Run `/sape` to validate SAPE probe configuration.
