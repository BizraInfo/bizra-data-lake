---
paths:
  - "bizra_kernel/pat_response_formatter.py"
---

# PAT Response Structure Rules

Mandatory 6-section structure for all PAT-enforced outputs.

## Required Sections

Every PAT response MUST include these 6 sections in order:

### 1. Executive Synthesis
- Maximum 5 bullet points
- Each bullet MUST be claim-tagged
- Focus on key insights only

```markdown
## Executive Synthesis

- [MEASURED] SNR achieved 0.98 in production testing
- [IMPLEMENTED] 5-gate validation pipeline operational
- [NOVEL] Cross-domain synthesis reveals optimization pathway
```

### 2. Domain Cross-Pollination Map
- List engaged domains (minimum 3)
- Map cross-domain connections
- Identify synthesis points

```markdown
## Domain Cross-Pollination Map

**Domains Engaged:** Mathematics, Computer Science, Philosophy

**Cross-Domain Connections:**
- **Mathematics** → **Computer Science**: Formal proofs enable verification
- **Philosophy** → **Computer Science**: Ethics guides AI alignment

**Synthesis Points:**
- Mathematical rigor + ethical frameworks = trustworthy AI
```

### 3. Elite Practitioner Anchoring
- Minimum 3 practitioners per domain
- All must be top 1% tier
- Include relevance scores

```markdown
## Elite Practitioner Anchoring

| Domain | Practitioner | Tier | Relevance | Key Contributions |
| ------ | ------------ | ---- | --------- | ----------------- |
| Mathematics | Terence Tao | top_1% | 0.95 | harmonic analysis, primes |
| Computer Science | Leslie Lamport | top_1% | 0.92 | distributed systems, TLA+ |
| Philosophy | Derek Parfit | top_1% | 0.88 | ethics, rationality |
```

### 4. Novel Insight Synthesis
- State novelty score (must be >= 0.75)
- Explain the novel insight
- Show semantic distance from known patterns

```markdown
## Novel Insight Synthesis

**Novelty Score:** 0.82

The synthesis reveals that formal verification methods from mathematics,
combined with ethical frameworks from philosophy, can address the AI
alignment problem in ways not captured by either domain alone.
```

### 5. Validation Evidence Trail
- Show all gate results
- Include SNR, novelty, Ihsān scores
- Provide receipt ID

```markdown
## Validation Evidence Trail

| Gate | Status | Score | Checks |
| ---- | ------ | ----- | ------ |
| Domain Analysis | passed | 0.85 | domains, unrelatedness |
| Quality Check | passed | 0.96 | snr, contradictions |
| Final Validation | passed | 0.98 | snr, novelty |
| Practitioner | passed | 1.00 | count, tier |
| Response Format | passed | 1.00 | sections, tags |

**Overall Scores:**
- SNR: 0.980
- Novelty: 0.820
- Ihsān: 0.960

**Receipt ID:** `pat-abc123def456`
```

### 6. Actionable Recommendations
- What we know (facts)
- What we assume (hypotheses)
- What we should test next (actions)

```markdown
## Actionable Recommendations

### What We Know
- [MEASURED] 5-gate validation achieves 0.98 SNR
- [IMPLEMENTED] Cross-domain synthesis is operational

### What We Assume
- [HYPOTHESIS] Long-term stability will improve with more data
- [HYPOTHESIS] Additional domains may improve novelty

### What We Should Test Next
- Validate with production workloads
- Measure practitioner relevance over time
- Test edge cases with low-novelty inputs
```

## Formatting Rules

1. Use standard markdown headers (`##`)
2. Use tables for structured data
3. Include claim tags in brackets: `[TAG]`
4. Keep bullet points concise
5. Never skip sections (all 6 required)

## Validation

Use the response formatter to validate:

```python
from bizra_kernel.pat_response_formatter import PATResponseFormatter

formatter = PATResponseFormatter()
result = formatter.validate_response(response_content)

if not result["is_valid"]:
    print(f"Missing sections: {result['section_status']}")
    print(f"Errors: {result['validation_errors']}")
```

## Common Errors

1. **Missing sections**: All 6 sections required
2. **Untagged claims**: Every factual claim needs a tag
3. **No evidence trail**: Must include receipt ID
4. **Insufficient practitioners**: Need 3+ per domain
5. **Low novelty**: Must achieve >= 0.75

## Example Complete Response

See `bizra_kernel/pat_response_formatter.py:__main__` for a complete example.
