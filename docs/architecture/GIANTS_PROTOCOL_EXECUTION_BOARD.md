# Giants Protocol Execution Board

## Objective
Operationalize open-source giant patterns into BIZRA with evidence-driven rollout:
- scout gate: SNR >= 0.84 and Ihsan >= 0.75
- production gate: SNR >= 0.98 and Ihsan >= 0.95

## Priority Stack
1. Graph-of-Thought Deliberation Upgrade
2. Benchmark and Constitutional Gate Harness
3. Meta Agent Cross-Pollination Task Force

## Phase 1: Pilot Sprint (3-5 days)
1. Wire deliberation trace capture in `core/apex/unified_orchestrator.py`.
2. Add branch/merge reasoning metrics in `core/apex/validation_pipeline.py`.
3. Extend `scripts/sape_deep_probe.py` to evaluate trace coherence and novelty.

Success criteria:
- At least one pilot path reaches scout gate.
- No regression in fail-closed behavior.

## Phase 2: Benchmark Sprint (3-5 days)
1. Add deterministic benchmark suite in `scripts/performance_benchmark.py`.
2. Add constitutional scorecard export in `scripts/quality_radar_elite.py`.
3. Gate benchmark outputs through `core/genesis/verifier.py`.

Success criteria:
- Repeatable benchmark results across at least 3 runs.
- SNR and Ihsan deltas are emitted as machine-readable evidence.

## Phase 3: Cross-Pollination Sprint (5-7 days)
1. Introduce adapter interface for external patterns in `scripts/elite_orchestrator.py`.
2. Add task-force governance rules in `config/substrate_v1.yaml`.
3. Add rollout checklist and rollback policy before production gate.

Success criteria:
- Patterns from at least 2 giants integrated behind feature flags.
- All merged paths keep production gate requirements explicit and testable.

## Command Surface
Generate ranked backlog:

```bash
python scripts/giants_protocol_pipeline.py --top 5 --output markdown
```

Export machine-readable backlog:

```bash
python scripts/giants_protocol_pipeline.py --top 10 --output json --out-file output/giants_backlog.json
```

