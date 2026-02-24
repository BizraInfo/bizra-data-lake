# Ablation Protocol (AbGen-Style Attribution)

Use this reference for the Ablate stage.

## Goal
Measure which components improve, do not affect, or degrade outcomes.

## Required Procedure
1. Define baseline score using current architecture.
2. Register components and categories.
3. Run controlled ablations per component.
4. Measure contribution = baseline - ablated score.
5. Rank components by contribution and significance.

## Experiment Types
- remove
- disable
- replace
- degrade
- permute
- isolate

Use at least one deterministic ablation type for every candidate component.

## Significance Policy
- Minimum runs per component: `3`.
- Treat weak effects below configured threshold as non-actionable.
- Promote only changes with stable positive contribution.

## Decision Rules
- Essential: high positive contribution, preserve and harden.
- Beneficial: positive contribution, candidate for optimization.
- Neutral/marginal: monitor or simplify.
- Harmful: remove or redesign before submission.

## Output Contract
`ablation_report.json` should include:
- baseline score
- per-component contribution
- significance estimate
- ranked weak components
- recommended architecture actions
