# 06: Validation Plan — Stage 0 Drills, TDD, Acceptance Criteria

## Standing on Giants
Deming (PDCA quality gates, 1950) · Beyer/Google SRE (error budgets, 2016) · Nygard (stability patterns, 2007)

## Overview

Stage 0 is the mandatory safety-net validation before any canary ramp begins. All Phase 46 canaries remain at 0%. We synthetically inject faults into each component and verify that the rollback engine responds correctly end-to-end. Only after all drills pass can canary percentages be increased.

## Stage 0 Synthetic Fault Drills

### Drill 1: Search Error-Rate Breach

```
SCENARIO: Force search error rate above 2%

SETUP:
    - SEARCH_PERCENT = 0 (canary off)
    - Inject mock that raises FAISS error on 5% of calls
    - Run 200 synthetic search requests

VERIFY:
    1. phase46_search_errors_total increments correctly
    2. Error rate metric exceeds 2% threshold
    3. RollbackEngine.evaluate("search_error_rate", True) called twice
    4. Receipt emitted with trigger="search_error_rate"
    5. SEARCH_PERCENT set to "0" (already was, confirm idempotent)
    6. If all percents already 0, hard_kill booleans set to "0"

TEARDOWN:
    - Remove mock
    - Reset env vars to pre-drill state
```

### Drill 2: GoT Fallback-Rate Breach

```
SCENARIO: Force GoT fallback rate above 20%

SETUP:
    - GOT_BRIDGE_PERCENT = 0 (canary off)
    - Inject mock that forces GoT convergence failure on 30% of calls
    - Run 100 synthetic GoT requests

VERIFY:
    1. phase46_got_fallback_total increments correctly
    2. Fallback rate metric exceeds 20% threshold
    3. RollbackEngine evaluates two consecutive breaches
    4. Receipt emitted with component="got_bridge"
    5. GOT_BRIDGE_PERCENT set to "0"

TEARDOWN:
    - Remove mock
    - Reset env vars
```

### Drill 3: HMM Confidence Breach

```
SCENARIO: Force HMM prediction confidence below 0.55

SETUP:
    - HMM_PERCENT = 0 (canary off)
    - Inject mock HMMEngine that returns confidence=0.30
    - Run 50 synthetic HMM observations

VERIFY:
    1. phase46_hmm_prediction_confidence p50 < 0.55
    2. RollbackEngine evaluates two consecutive breaches
    3. Receipt emitted with component="hmm"
    4. HMM_PERCENT set to "0"

TEARDOWN:
    - Remove mock
    - Reset env vars
```

### Drill 4: Hard Kill Verification

```
SCENARIO: All percents at 0 + breach triggers hard kill

SETUP:
    - All *_PERCENT = 0
    - All *_ENABLED = "1" (on but not routed)
    - Force latency regression breach

VERIFY:
    1. After 2 consecutive breaches:
       - SEARCH_ENABLED = "0"
       - GOT_BRIDGE_ENABLED = "0"
       - HMM_ENABLED = "0"
    2. Receipt action == "hard_kill"
    3. All Phase 46 components report disabled in mcp_health

TEARDOWN:
    - Reset all env vars to pre-drill state
```

### Drill 5: Receipt Persistence

```
SCENARIO: Verify rollback receipts are queryable

SETUP:
    - Trigger any rollback via Drill 1-4

VERIFY:
    1. Receipt file exists in artifacts/rollback_receipts/
    2. Receipt is valid JSON
    3. Contains: timestamp, trigger, breach_count, component, action
    4. Contains: previous_config (snapshot of all 6 env vars)
    5. Contains: metrics_snapshot (non-empty)
    6. Timestamp is within 5s of wall clock

TEARDOWN:
    - Clean up receipt files
```

## Canary Schedule Validation

### Stage Gate Checklist

Each stage transition requires:

```
FUNCTION validate_stage_gate(stage_number: int) -> bool:
    """Run gate checks before advancing to next canary stage."""

    checks = [
        # 1. Phase 46 targeted tests still green
        run_pytest("tests/core/search/ tests/core/reasoning/test_got_bridge.py "
                   "tests/core/prediction/ tests/core/test_resonance.py "
                   "tests/core/sovereign/test_apex_got_bridge_integration.py "
                   "tests/core/mcp/test_sovereign_phase46_tools.py"),

        # 2. No new failures beyond baseline
        compare_failures(get_current_failures(), load_baseline()),

        # 3. No active breach windows
        rollback_engine.status["breach_windows"] all have consecutive=0,

        # 4. Metrics within SLO
        metrics.snapshot() shows no threshold violations,

        # 5. Previous stage soak duration met
        soak_timer.elapsed >= REQUIRED_SOAK[stage_number],
    ]

    RETURN all(checks)
```

### Canary Schedule Table

| Stage | Component | Percent | Duration | Gate Before Advance |
|-------|-----------|---------|----------|---------------------|
| 0 | all | 0% | until drills pass | Drills 1-5 pass |
| 1 | Search | 10% | 4h | Stage gate |
| 2 | Search | 50% | 8h | Stage gate |
| 3 | GoT | 10% | 4h | Stage gate |
| 4 | GoT | 50% | 8h | Stage gate |
| 5 | HMM | 10% | 8h | Stage gate |
| 6 | HMM | 50% | 24h (Guarded Solo) | Stage gate + human check |
| 7 | all | 100% | 24h final soak | Stage gate + human check |

### Guarded Solo Check Cadence

```
During Stage 6 and 7 (Guarded Solo):
    - Human operator checks metrics dashboard at fixed awake intervals
    - Strict auto-rollback covers unattended windows
    - Minimum 2 human checks per 24h soak period
    - Each check produces a signed-off entry in operator_checks.log
```

## Complete Test Matrix

### New Tests for Phase 47.1 (`tests/core/rollout/`)

```
tests/core/rollout/
    test_canary_router.py          ~15 tests (from spec 01)
    test_hmm_caller_gate.py        ~10 tests (from spec 02)
    test_phase46_metrics.py        ~10 tests (from spec 03)
    test_rollback_engine.py        ~12 tests (from spec 04)
    test_release_integrity.py      ~8 tests  (from spec 05)
    test_stage0_drills.py          ~10 tests (this spec)
```

### Stage 0 Drill Tests

```python
class TestStage0Drills:
    """End-to-end drill tests for Stage 0 safety-net validation."""

    def test_drill_search_error_breach_triggers_rollback(self):
        """Drill 1: Search error rate breach produces rollback receipt."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        # Simulate 2 consecutive breached windows
        engine.evaluate("search_error_rate", breached=True)
        receipt = engine.evaluate("search_error_rate", breached=True)
        assert receipt is not None
        assert receipt.trigger == "search_error_rate"

    def test_drill_got_fallback_breach_triggers_rollback(self):
        """Drill 2: GoT fallback rate breach produces rollback receipt."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        engine.evaluate("got_fallback_rate", breached=True)
        receipt = engine.evaluate("got_fallback_rate", breached=True)
        assert receipt is not None
        assert receipt.component == "got_bridge"

    def test_drill_hmm_confidence_breach_triggers_rollback(self):
        """Drill 3: HMM confidence breach produces rollback receipt."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        engine.evaluate("hmm_confidence", breached=True)
        receipt = engine.evaluate("hmm_confidence", breached=True)
        assert receipt is not None
        assert receipt.component == "hmm"

    def test_drill_hard_kill_when_all_zeroed(self):
        """Drill 4: Hard kill when all percents already at 0."""
        with patch.dict(os.environ, {
            "BIZRA_PHASE46_SEARCH_PERCENT": "0",
            "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "0",
            "BIZRA_PHASE46_HMM_PERCENT": "0",
        }):
            engine = RollbackEngine(receipt_dir=tmp_path)
            engine.evaluate("latency_regression", breached=True)
            receipt = engine.evaluate("latency_regression", breached=True)
            assert receipt.action == "hard_kill"

    def test_drill_receipt_valid_json(self):
        """Drill 5: Receipt is valid JSON with all required fields."""
        engine = RollbackEngine(receipt_dir=tmp_path)
        engine.evaluate("hmm_confidence", breached=True)
        engine.evaluate("hmm_confidence", breached=True)
        receipts = list(tmp_path.glob("rollback_*.json"))
        assert len(receipts) >= 1
        data = json.loads(receipts[0].read_text())
        required_fields = [
            "timestamp", "trigger", "breach_count",
            "component", "action", "previous_config",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_drill_rollback_does_not_affect_unrelated_components(self):
        """Rollback of one component leaves others unchanged."""
        with patch.dict(os.environ, {
            "BIZRA_PHASE46_SEARCH_PERCENT": "50",
            "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "50",
            "BIZRA_PHASE46_HMM_PERCENT": "50",
        }):
            engine = RollbackEngine(receipt_dir=tmp_path)
            engine.evaluate("search_error_rate", breached=True)
            engine.evaluate("search_error_rate", breached=True)
            assert os.environ["BIZRA_PHASE46_SEARCH_PERCENT"] == "0"
            assert os.environ["BIZRA_PHASE46_GOT_BRIDGE_PERCENT"] == "50"
            assert os.environ["BIZRA_PHASE46_HMM_PERCENT"] == "50"

    def test_canary_router_kill_switch_takes_precedence(self):
        """Kill switch ENABLED=0 overrides any PERCENT value."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": "0"}):
            router = CanaryRouter(salt="test")
            assert router.should_route("search", "any", 100) is False

    def test_hmm_gate_blocks_non_allowed_caller(self):
        """HMM gate in single mode blocks non-allowed callers."""
        engine = MockHMMEngine()
        gate = HMMCallerGate(engine)
        result = gate.observe("search", "unauthorized")
        assert result is None
        assert gate.stats["dropped_count"] == 1

    def test_metrics_entropy_computation(self):
        """Metrics correctly compute Shannon entropy."""
        m = Phase46Metrics()
        # 4 symbols, uniform -> entropy = 2.0
        for s in ["a", "b", "c", "d"]:
            for _ in range(100):
                m.record_hmm_observation(s)
        assert abs(m.observation_entropy() - 2.0) < 0.01

    def test_all_drills_pass_gate(self):
        """Stage 0 gate passes when all drill results are clean."""
        # This is the integration test that chains all 5 drills
        # and verifies the stage gate function returns True
        pass  # Implemented during coding phase
```

## Acceptance Criteria (Complete)

| # | Criterion | Verification Method |
|---|-----------|---------------------|
| 1 | Release branch contains only Phase 46 + rollout files | `git diff origin/main --stat` |
| 2 | Semantic integrity checks pass | Import smoke + API surface snapshot |
| 3 | Phase 46 tests (210) pass on release branch | `pytest` targeted suite |
| 4 | Stage 0 rollback drills (5) pass completely | Drill test suite |
| 5 | Canary reaches Stage 7 without unresolved breaches | Operator log + metrics |
| 6 | No new failures beyond baseline (28 known) | Failure comparator |
| 7 | Kill switch precedence verified | TDD test |
| 8 | Deterministic routing reproducible | TDD test |
| 9 | HMM caller isolation blocks unauthorized | TDD test |
| 10 | `combined_snr` metric emitted and queryable | Metrics snapshot test |
| 11 | Rollback receipts persisted with all fields | Receipt validation test |
| 12 | Guarded Solo soak completed (48h total) | Stage 6 + 7 timestamps |

## Estimated Test Count

| Spec | Test File | Tests |
|------|-----------|-------|
| 01 Canary | `test_canary_router.py` | ~15 |
| 02 HMM Gate | `test_hmm_caller_gate.py` | ~10 |
| 03 Metrics | `test_phase46_metrics.py` | ~10 |
| 04 Rollback | `test_rollback_engine.py` | ~12 |
| 05 Release | `test_release_integrity.py` | ~8 |
| 06 Drills | `test_stage0_drills.py` | ~10 |
| **Total Phase 47.1** | | **~65** |
| **Total (with Phase 46)** | | **~275** |
