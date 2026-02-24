# Evaluation Harness (CLEAR + ABC, Fail-Closed)

Use this reference when defining execution metrics and validation gates.

## Objectives
- Measure performance across CLEAR dimensions: Cost, Latency, Efficacy, Assurance, Reliability.
- Prevent benchmark inflation through ABC checks.
- Enforce strict gates before submission.

## CLEAR Metrics
- `cost`: total tokens, USD cost, API/tool calls.
- `latency`: p50 and p95 latency, completion time.
- `efficacy`: task completion, accuracy/pass rate.
- `assurance`: safety violations, hallucination proxy, reproducibility.
- `reliability`: cross-seed consistency, variance, failure recovery.

## ABC Integrity Checklist
Require all items in strict mode:
- sufficient test cases
- diverse task distribution
- no reward hacking
- temporal holdout
- adversarial probes
- null model baseline
- human or trusted baseline
- multi-run consistency
- cost tracking
- failure analysis

## Seed Sweep Policy
- Minimum seed count: `3`.
- Compute per-seed score and variance.
- Gate thresholds:
  - fail if seeds < minimum
  - fail if variance > `integrity_profile.max_seed_variance`

## HAL-Inspired Reliability Checks
- Run each target with the same input set across seeds.
- Record run-to-run spread and unstable cases.
- Flag "high reasoning effort, low consistency" patterns for ablation.

## Gate Order (Strict)
1. Reproducibility gate.
2. Integrity gate.
3. Budget gate.
4. Submission anti-gaming gate.

Stop immediately when a gate fails and emit rollback receipt.
