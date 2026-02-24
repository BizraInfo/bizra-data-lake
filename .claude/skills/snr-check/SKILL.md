---
name: snr-check
description: Check SNR (Signal-to-Noise Ratio) of content
---

Analyze the SNR of the following content: $ARGUMENTS

## SNR Analysis
Evaluate against these dimensions:

1. **Signal Components**
   - Relevance (0-1): How relevant to the query?
   - Novelty (0-1): New information vs redundant?
   - Groundedness (0-1): Supported by evidence?
   - Coherence (0-1): Logically consistent?
   - Actionability (0-1): Leads to next steps?

2. **Noise Components**
   - Redundancy: Repeated information
   - Ambiguity: Unclear statements
   - Verbosity: Unnecessary words
   - Inconsistency: Contradictions

## Output
```
Signal Score: [0-1]
Noise Score: [0-1]
SNR (Linear): [signal/noise]
SNR (dB): [10*log10(SNR)]

Ihsān Threshold: 0.95
Status: [PASS/FAIL]

Recommendations for improvement:
- [specific suggestions]
```
