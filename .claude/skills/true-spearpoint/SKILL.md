---
name: true-spearpoint
description: Run fail-closed benchmark dominance campaigns for agentic systems. Use when setting up CLEAR-style evaluation harnesses, executing ablation-driven architecture optimization, and producing reproducible leaderboard submissions with anti-gaming and budget gates for SWE-bench Verified, HLE, or AgentBeats.
metadata:
  short-description: Execute strict benchmark dominance loop
---

# True Spearpoint — Benchmark Dominance Loop

Run benchmark campaigns as an engineering loop, not a one-off test.

## Intake Contract
Collect these inputs before execution:
- `benchmark_targets`: `swe_bench_verified`, `hle`, `agentbeats`, or `all`
- `model_or_solver_spec`: planner/solver/verifier routing and versions
- `budget_policy`: max cost, max latency, max token budget

Optional but recommended:
- `integrity_profile`: seed sweep and anti-gaming settings
- `submission_policy`: integrity and cost gate behavior
- `execution_mode`: `strict` (default), `balanced`, or `explore`

Use `configs/spearpoint.yaml` as the source of truth.

## Decision Tree
1. Select battlefield profile from `references/benchmark-targets.md`.
2. Select execution mode:
- `strict`: fail-closed, block on any critical gate.
- `balanced`: allow non-critical continuation with warnings and receipts.
- `explore`: collect diagnostics without submission blocking.
3. Confirm budget policy and submission policy.
4. Execute the loop through `scripts/spearpoint_run.py`.

## Execute the Loop
Run phases in this order:
1. Evaluate: compute CLEAR metrics and ABC-integrity checks.
2. Ablate: identify weak/harmful components and effect sizes.
3. Architect: propose deterministic upgrades from ablation results.
4. Submit: validate anti-gaming and record leaderboard result.
5. Analyze: summarize score delta, cost efficiency, and next backlog.

Do not skip gates in `strict` mode.

## Fail-Closed Gates
Apply these gates before marking success:
1. Reproducibility gate: seed sweep meets minimum and variance bound.
2. Integrity gate: leak scan, null-model probe, and injection probes pass.
3. Budget gate: cost, latency, and token caps stay within policy.
4. Submission gate: anti-gaming validation passes.

If any strict gate fails:
- emit `rollback_receipt.json`
- return non-zero exit code
- stop submission

## Artifact Contract
Emit deterministic JSON artifacts for every run:
- `evaluation_report.json`
- `ablation_report.json`
- `submission_bundle.json`
- `campaign_summary.json`
- `rollback_receipt.json` on gate failure

Each artifact must include:
- `run_id`
- `target`
- `mode`
- `timestamp_utc`
- `gate_status`

## Reference Loading Rules
- Load `references/evaluation-harness.md` when defining metrics, gates, and seed policy.
- Load `references/ablation-protocol.md` when designing component experiments.
- Load `references/architecture-playbook.md` when selecting upgrades.
- Load `references/submission-campaign.md` for anti-gaming, receipts, and campaign flow.
- Load `references/benchmark-targets.md` for SWE/HLE/AgentBeats target specifics.

## Commands
```bash
python .claude/skills/true-spearpoint/scripts/spearpoint_run.py \
  --config .claude/skills/true-spearpoint/configs/spearpoint.yaml \
  --out /tmp/spearpoint-run
```

```bash
python .claude/skills/true-spearpoint/scripts/spearpoint_run.py \
  --config .claude/skills/true-spearpoint/configs/spearpoint.yaml \
  --mode strict --target all --out /tmp/spearpoint-campaign
```
