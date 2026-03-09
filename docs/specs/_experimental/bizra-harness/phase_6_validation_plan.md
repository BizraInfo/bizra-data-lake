# Phase 6 — TDD Validation Plan

> Standing on Giants: Beck (TDD, 2003) · Deming (measure then manage, 1950)
> · Popper (falsifiability, 1934)

## Overview

This phase consolidates all TDD anchors from Phases 1-5 into a structured
test plan with dependency ordering, coverage targets, and acceptance criteria.

## Test File Map

```
tests/core/harness/
├── __init__.py
├── conftest.py               # Fixtures: harness_runner, scenario_library,
│                              #   baseline_store, harness_config, make_result
├── test_types.py              # Phase 1: 8 tests
├── test_runner.py             # Phase 2: 9 tests
├── test_scenarios.py          # Phase 3: 9 tests
├── test_persistence.py        # Phase 4: 7 tests
├── test_report.py             # Phase 5: 7 tests
└── test_scenarios_parametric.py  # Phase 5: parametrized over 7 scenarios
```

**Estimated total: ~47 tests** across 6 files.

## Implementation Order (TDD Red-Green-Refactor)

Build bottom-up: types → persistence → scenarios → runner → reports.

### Step 1: Types (test_types.py)

Write tests first, then implement `core/harness/types.py`.

| # | Test | Validates |
|---|------|-----------|
| T1.1 | `test_verdict_enum_values` | 3 variants, string values |
| T1.2 | `test_pillar_name_count` | 10 pillars (8 smoke + 2 harness) |
| T1.3 | `test_harness_config_defaults` | snr_floor=0.85, ihsan_floor=0.95 |
| T1.4 | `test_harness_config_validate_happy` | No errors on valid config |
| T1.5 | `test_harness_config_validate_bad_snr` | snr_floor > 1.0 rejected |
| T1.6 | `test_pillar_result_immutable` | Frozen dataclass enforcement |
| T1.7 | `test_harness_result_to_dict_keys` | JSON-serializable output |
| T1.8 | `test_regression_report_delta` | Delta math + is_regression logic |

**Dependencies**: Only `core/integration/constants.py` (existing).
**Red**: Write all 8 tests → all fail.
**Green**: Implement `types.py` → all pass.

### Step 2: Persistence (test_persistence.py)

| # | Test | Validates |
|---|------|-----------|
| T4.1 | `test_append_run_creates_file` | File creation + genesis hash |
| T4.2 | `test_hash_chain_integrity` | prev_hash links correctly |
| T4.3 | `test_chain_detects_tamper` | Tampered entry breaks chain |
| T4.4 | `test_seal_baseline` | Baseline sealing + retrieval |
| T4.5 | `test_get_latest_returns_none_when_empty` | Graceful on empty store |
| T4.6 | `test_trim_runs` | Rolling window at MAX_RUNS_KEPT |
| T4.7 | `test_get_run_history` | Last-N retrieval |

**Dependencies**: `types.py` (Step 1).
**All tests use `tmp_path`** for isolation — no filesystem pollution.

### Step 3: Scenarios (test_scenarios.py)

| # | Test | Validates |
|---|------|-----------|
| T3.1 | `test_builtin_count` | 7 built-in scenarios |
| T3.2 | `test_scenario_ids_are_unique` | No ID collisions |
| T3.3 | `test_scenario_library_default` | Default library loads |
| T3.4 | `test_scenario_library_list_by_tag` | Tag filtering works |
| T3.5 | `test_scenario_library_register` | Runtime registration |
| T3.6 | `test_scenario_library_register_duplicate_raises` | Duplicate rejected |
| T3.7 | `test_scenario_to_config_overrides` | Config override dict |
| T3.8 | `test_focus_pillars_filters_evaluators` | Subset pillar selection |
| T3.9 | `test_user_scenarios_graceful_on_missing_file` | No crash on missing |

**Dependencies**: `types.py` (Step 1).

### Step 4: Runner (test_runner.py)

| # | Test | Validates |
|---|------|-----------|
| T2.1 | `test_smoke_mode_runs_two_pillars` | SMOKE → 2 pillars |
| T2.2 | `test_standard_mode_runs_ten_pillars` | STANDARD → 10 pillars |
| T2.3 | `test_verdict_pass_when_all_gates_met` | Deterministic PASS |
| T2.4 | `test_verdict_fail_when_snr_below_floor` | SNR gate enforcement |
| T2.5 | `test_verdict_fail_on_regression` | Regression → FAIL |
| T2.6 | `test_timeout_produces_failed_pillar` | Timeout handling |
| T2.7 | `test_receipt_emitted` | Evidence chain integration |
| T2.8 | `test_aggregate_snr_geometric_mean` | Math correctness |
| T2.9 | `test_resolve_verdict_deterministic` | Same inputs → same output |

**Dependencies**: `types.py`, `persistence.py`, `scenarios.py` (Steps 1-3).
**Note**: Most tests will mock pillar evaluators to avoid external deps.

### Step 5: Reports (test_report.py)

| # | Test | Validates |
|---|------|-----------|
| T5.1 | `test_json_report_creates_file` | File written, valid JSON |
| T5.2 | `test_json_report_under_100kb` | Size constraint |
| T5.3 | `test_html_report_creates_file` | Valid HTML output |
| T5.4 | `test_html_report_contains_pillar_table` | All pillars rendered |
| T5.5 | `test_safe_serialize_strips_secrets` | No tokens/keys in output |
| T5.6 | `test_safe_serialize_truncates_long_strings` | 2000 char cap |
| T5.7 | `test_safe_serialize_caps_list_length` | 100 item cap |

**Dependencies**: `types.py` (Step 1).

### Step 6: Parametric Scenarios (test_scenarios_parametric.py)

| # | Test | Validates |
|---|------|-----------|
| T6.1 | `test_scenario_executes_without_crash` × 7 | All built-ins run |
| T6.2 | `test_basic_claim_passes_in_healthy_system` | Baseline health |

**Dependencies**: Full `core/harness/` module (Steps 1-5).

## Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `types.py` | 95% | Pure data — full coverage feasible |
| `persistence.py` | 90% | I/O heavy — hard to cover edge cases |
| `scenarios.py` | 95% | Logic light, registry pattern |
| `runner.py` | 85% | External subsystem calls need mocking |
| `report.py` | 90% | Template rendering, string output |
| **Aggregate** | **90%** | Above 65% CI floor, toward 95% Ihsan |

## Acceptance Criteria

### Must Pass (Gate)

1. `pytest tests/core/harness/ -q` — **all 47 tests green**
2. `pytest -m harness -q` — **all harness-marked tests green**
3. `coverage run -m pytest tests/core/harness/` — **aggregate >= 90%**
4. No new dependencies in `pyproject.toml`
5. `ruff check core/harness/` — **0 errors**
6. `mypy core/harness/` — **0 new errors** (pre-existing OK)

### Should Pass (Quality)

7. `python -m core.harness --mode smoke` exits 0 on healthy system
8. JSON report < 100 KB for standard run
9. HTML report renders correctly in browser
10. `BaselineStore.verify_chain()` returns `(True, N, None)` after N runs

### Nice to Have (Excellence)

11. All built-in scenarios produce meaningful verdicts (not all INCONCLUSIVE)
12. Regression detection catches a 0.03 SNR drop from baseline
13. Hash chain survives 500-entry trim without breaking

## Mocking Strategy

The runner tests need mocked pillar evaluators to avoid importing the
full sovereign stack in unit tests:

```python
class MockPillarEvaluator:
    """Test double for any pillar evaluator."""

    def __init__(self, name: PillarName, passed: bool = True,
                 score: float = 0.92, duration_ms: float = 5.0):
        self.name = name
        self._passed = passed
        self._score = score
        self._duration_ms = duration_ms

    async def evaluate(self, config: HarnessConfig) -> PillarResult:
        return PillarResult(
            pillar=self.name,
            passed=self._passed,
            duration_ms=self._duration_ms,
            score=self._score,
        )


def _make_mock_runner(all_pass=True, snr=0.92) -> HarnessRunner:
    """Build a runner with all mocked pillars."""
    runner = HarnessRunner()
    # Monkey-patch pillar registry for testing
    mock_pillars = [
        MockPillarEvaluator(name, passed=all_pass, score=snr)
        for name in PillarName
    ]
    runner._pillar_override = mock_pillars  # runner checks this first
    return runner
```

## Invariant Tests (Property-Based)

For Phase 2 completeness, add Hypothesis-based property tests:

```python
from hypothesis import given, strategies as st

@given(snr=st.floats(0.0, 1.0), ihsan=st.floats(0.0, 1.0))
def test_verdict_determinism(snr, ihsan):
    """Same inputs always produce same verdict."""
    v1 = resolve_verdict(snr, ihsan, True, None, HarnessConfig())
    v2 = resolve_verdict(snr, ihsan, True, None, HarnessConfig())
    assert v1 == v2

@given(snr=st.floats(0.0, 0.849))
def test_verdict_fail_below_snr_floor(snr):
    """Any SNR below 0.85 → FAIL (with default config)."""
    v = resolve_verdict(snr, 1.0, True, None, HarnessConfig())
    assert v == Verdict.FAIL

@given(snr=st.floats(0.85, 1.0), ihsan=st.floats(0.95, 1.0))
def test_verdict_pass_above_both_floors(snr, ihsan):
    """SNR >= 0.85 AND Ihsan >= 0.95 AND all gates → PASS."""
    v = resolve_verdict(snr, ihsan, True, None, HarnessConfig())
    assert v == Verdict.PASS
```

## Execution Commands

```bash
# Run all harness tests
pytest tests/core/harness/ -q

# Run only harness-marked tests
pytest -m harness -q

# Run with coverage
pytest tests/core/harness/ --cov=core/harness --cov-report=term-missing

# Run harness CLI
python -m core.harness --mode smoke
python -m core.harness --mode standard --report both
python -m core.harness --scenario basic_claim --update-baseline

# Run specific phase
pytest tests/core/harness/test_types.py -q          # Phase 1
pytest tests/core/harness/test_persistence.py -q     # Phase 4
pytest tests/core/harness/test_runner.py -q           # Phase 2
```

## Build Dependency Graph

```
Phase 1 (types.py)
    │
    ├── Phase 4 (persistence.py) ─────────┐
    │                                       │
    ├── Phase 3 (scenarios.py) ────────────┤
    │                                       │
    └── Phase 5 (report.py)                │
                                            │
                          Phase 2 (runner.py)
                                            │
                          Phase 5 (pytest integration)
                                            │
                          Phase 6 (parametric + property tests)
```

Types first. Runner last. Everything else is parallelizable.
