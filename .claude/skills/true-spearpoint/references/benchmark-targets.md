# Benchmark Targets (Canonical Battlefields)

Use this reference to map target choice to constraints and expectations.

## Canonical Targets

## `swe_bench_verified`
- Domain: autonomous software engineering.
- Priority metrics: patch correctness, completion rate, reproducibility.
- Operational emphasis: tool-use reliability and regression control.

## `hle`
- Domain: abstract reasoning.
- Priority metrics: reasoning accuracy, consistency across seeds, safety.
- Operational emphasis: high-assurance reasoning and anti-gaming checks.

## `agentbeats`
- Domain: dynamic agentic generalization.
- Priority metrics: adaptability, benchmark creation/solve behavior, stability.
- Operational emphasis: long-horizon reliability and campaign discipline.

## Target Selection Rule
- Use `swe_bench_verified` for engineering loops and code agents.
- Use `hle` for reasoning and formal analysis loops.
- Use `agentbeats` for adaptive, multi-stage autonomous behavior.
- Use `all` for campaign sweeps in deterministic order:
  1. `swe_bench_verified`
  2. `hle`
  3. `agentbeats`

## Output Isolation
- Store artifacts per target in dedicated directories.
- Never mix target receipts in one folder during multi-target runs.
