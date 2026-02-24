---
name: sovereign-query
description: Query the Sovereign Engine for answers with full Graph-of-Thoughts reasoning
context: fork
agent: sovereign-researcher
---

Process the following query through the BIZRA Sovereign Engine pipeline:

Query: $ARGUMENTS

## Processing Steps
1. **Parse Intent**: Understand what is being asked
2. **Gather Context**: Search codebase, docs, memory
3. **Graph Reasoning**: Explore multiple hypothesis branches
4. **SNR Maximization**: Filter noise, amplify signal
5. **Synthesize**: Combine best insights
6. **Validate**: Check against Ihsān constraints (SNR ≥ 0.95)

## Output Format
```
## Understanding
[What was asked]

## Analysis
[Multi-branch reasoning with confidence scores]

## Answer
[Synthesized response]

## Confidence
[SNR score and validation status]

## Sources
[Citations and provenance]
```
