---
name: pat-enforcement
description: PAT (Peak Autonomous Think Tank) enforcement validation and compliance
---

# PAT Enforcement Skill

This skill provides knowledge for PAT system validation and enforcement.

## PAT Thresholds (Constitution: constitution/pat_enforcement_v1.yaml)

| Metric | Threshold | Description |
|--------|-----------|-------------|
| SNR | >= 0.98 | Signal-to-noise ratio (99.8% signal) |
| Novelty | >= 0.75 | Semantic distance from known patterns |
| Ihsan | >= 0.95 | Ethical excellence score |
| Unrelatedness | >= 0.70 | Cross-domain distance |
| Domains | >= 3 | Minimum unrelated domains |
| Practitioners | >= 3/domain | Elite practitioners (top 1%) |

## 5 Validation Gates

1. **Gate 1 - Pre-Reasoning**: Domain analysis, unrelatedness check
2. **Gate 2 - Mid-Synthesis**: Running SNR check, contradiction detection
3. **Gate 3 - Post-Synthesis**: Final SNR/novelty/coverage validation
4. **Gate 4 - Practitioner**: Elite practitioner verification
5. **Gate 5 - Response**: 6-section structure enforcement

## 6-Section Response Structure

Every PAT response MUST include:

1. **Executive Synthesis** - Max 5 claim-tagged bullets
2. **Domain Cross-Pollination Map** - Domains, connections, synthesis
3. **Elite Practitioner Anchoring** - 3+ per domain, top_1% tier
4. **Novel Insight Synthesis** - Novelty score >= 0.75
5. **Validation Evidence Trail** - Gate status, scores, receipt ID
6. **Actionable Recommendations** - Know/Assume/Test Next

## Claim Tags

| Tag | Weight | Use |
|-----|--------|-----|
| `[MEASURED]` | 1.00 | Empirically verified |
| `[IMPLEMENTED]` | 0.95 | Working code |
| `[DERIVED]` | 0.90 | Logically derived |
| `[NOVEL]` | 1.00 | Cross-domain insight |
| `[CROSS_DOMAIN]` | 0.95 | Multi-domain synthesis |
| `[HYPOTHESIS]` | 0.40 | Requires testing |

## Key Files

- `constitution/pat_enforcement_v1.yaml` - PAT constitution
- `config/pat_enforcement/pat_domains.yaml` - Domain registry
- `bizra_kernel/pat_enforcement_engine.py` - 5-gate engine
- `bizra_kernel/pat_unified_orchestrator.py` - Master controller

## Validation Command

Run `/pat` to validate PAT constitution compliance.
