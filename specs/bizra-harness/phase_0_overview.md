# Phase 0 — BIZRA Harness Overview

> Standing on Giants: Deming (PDCA quality cycle, 1950) · Lamport (distributed
> reliability, 1978) · Shannon (SNR information theory, 1948) · Al-Ghazali
> (Ihsan ethics, 1095)

## Problem Statement

BIZRA has rich subsystem-level validation (8-pillar smoke suite, CLEAR
framework, BDL loop, 9 guardrails, proof engine benchmarks) but **no unified
harness** that:

1. Orchestrates all subsystems in a single run
2. Produces a unified verdict (PASS / FAIL / INCONCLUSIVE)
3. Compares against sealed baselines for regression detection
4. Persists run history for trend analysis
5. Generates operator-readable reports

The 8-pillar smoke suite (`tests/integration/test_autonomous_pilot.py`)
validates that each subsystem boots — it does **not** validate that they work
together under realistic claims against constitutional thresholds.

## Scope

### In Scope

- `core/harness/` — New module (5 files, ~800 LOC estimated)
- `tests/core/harness/` — Unit + integration tests (~400 LOC estimated)
- Pytest marker `@pytest.mark.harness` for selective invocation
- JSON + optional HTML report output
- Baseline persistence in `.spearpoint/` directory
- Integration with existing modules (no rewrites):
  - `core/spearpoint/` — Orchestrator, AutoEvaluator, BDL
  - `core/benchmark/` — CLEAR, guardrails, leaderboard
  - `core/proof_engine/` — BenchResult, EvidenceLedger
  - `core/integration/constants.py` — All thresholds (single source)

### Out of Scope

- Modifying any existing subsystem internals
- Cloud deployment or CI runner integration (future phase)
- Real LLM inference (harness uses mock/stub call_fn by default)
- UI dashboard (harness is CLI + JSON + optional HTML)

## Architecture

```
                        HarnessRunner
                            |
            +---------------+---------------+
            |               |               |
       PillarSuite    ScenarioLibrary   BaselineStore
            |               |               |
   +--------+--------+     |          .spearpoint/
   |   |    |    |   |     |          baselines.jsonl
  Boot Token Evid SNR ...  |          runs.jsonl
   |        |        |     |
   v        v        v     v
  SovereignRuntime  AutoEvaluator  BenchResult
  TokenLedger       CLEARFramework EvidenceLedger
  SNRFacade         GuardrailSuite
```

## Module Map

| File | Purpose | LOC est. |
|---|---|---|
| `core/harness/__init__.py` | Public re-exports | ~20 |
| `core/harness/types.py` | HarnessConfig, HarnessResult, Verdict, PillarResult | ~120 |
| `core/harness/runner.py` | HarnessRunner orchestration engine | ~250 |
| `core/harness/scenarios.py` | ScenarioLibrary, HarnessScenario, baseline comparison | ~150 |
| `core/harness/persistence.py` | BaselineStore, RunHistory, regression detection | ~140 |
| `core/harness/report.py` | JSON + HTML report generation | ~120 |

## Constraints

1. **All thresholds from `constants.py`** — no hardcoded values
2. **No secrets** — claims, scenarios, and configs contain no tokens or keys
3. **Async-first** — HarnessRunner.run() is async; sync wrapper provided
4. **Deterministic by default** — mock call_fn for offline runs
5. **File limit** — each module < 500 lines
6. **Existing patterns** — follow frozen dataclass + Protocol conventions

## Dependencies (Existing Only)

```
core.integration.constants    → thresholds
core.spearpoint.config        → TierPolicy, resolve_tier()
core.spearpoint.auto_evaluator → AutoEvaluator
core.benchmark.clear_framework → CLEARFramework, CLEARMetrics
core.benchmark.guardrails      → GuardrailSuite, GuardrailResult
core.proof_engine.bench        → BenchResult, BenchSample
core.proof_engine.evidence_ledger → EvidenceLedger, Receipt
core.sovereign.runtime_core    → SovereignRuntime
core.iaas.snr_calculator       → SNRFacade
```

## Success Criteria

- `pytest -m harness` runs the full harness in < 30 seconds (mocked)
- `HarnessResult.verdict` is deterministic given same inputs
- Regression detection catches any SNR drop > 0.02 from baseline
- Report JSON is valid, < 100 KB for a typical run
- Zero new dependencies added to `pyproject.toml`
