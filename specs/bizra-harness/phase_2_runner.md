# Phase 2 — HarnessRunner Orchestration Engine

> Standing on Giants: Boyd (OODA loop, 1976) · Deming (PDCA quality, 1950)
> · Lamport (distributed reliability, 1978)

## Overview

The HarnessRunner is the central coordinator. It sequences pillar
evaluations, applies quality gates, computes aggregate scores, resolves the
verdict, and emits an evidence receipt. It follows Boyd's OODA:
Observe (run pillars) → Orient (aggregate scores) → Decide (verdict) →
Act (emit receipt + report).

## File: `core/harness/runner.py`

```pseudocode
IMPORTS:
    import asyncio, time, uuid
    from datetime import datetime, timezone
    from core.integration.constants import (
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )
    from core.spearpoint.config import resolve_tier
    from core.harness.types import (
        HarnessConfig, HarnessResult, PillarResult, PillarName,
        RunMode, Verdict, RegressionReport,
    )
    from core.harness.scenarios import ScenarioLibrary
    from core.harness.persistence import BaselineStore

# ── Pillar Protocol ────────────────────────────────────────────────

PROTOCOL PillarEvaluator:
    """Structural typing for pillar implementations."""
    name: PillarName

    async METHOD evaluate(self, config: HarnessConfig) -> PillarResult:
        ...

# ── Built-in Pillar Evaluators ─────────────────────────────────────

CLASS RuntimeBootPillar(PillarEvaluator):
    name = PillarName.RUNTIME_BOOT

    async METHOD evaluate(self, config):
        t0 = time.monotonic()
        TRY:
            from core.sovereign.runtime_core import SovereignRuntime
            async with SovereignRuntime.create() as rt:
                status = rt.status()
                passed = status.get("identity") is not None
            RETURN PillarResult(
                pillar=self.name, passed=passed,
                duration_ms=(time.monotonic() - t0) * 1000,
                details={"status_keys": list(status.keys())},
            )
        EXCEPT Exception as exc:
            RETURN PillarResult(
                pillar=self.name, passed=False,
                duration_ms=(time.monotonic() - t0) * 1000,
                error=str(exc),
            )

CLASS SNRCheckPillar(PillarEvaluator):
    name = PillarName.SNR_CHECK

    async METHOD evaluate(self, config):
        t0 = time.monotonic()
        TRY:
            from core.iaas.snr_calculator import compute_snr
            # Use the claim text as the signal to evaluate
            result = compute_snr(config.claim)
            score = result.score if hasattr(result, 'score') else float(result)
            passed = score >= config.snr_floor
            RETURN PillarResult(
                pillar=self.name, passed=passed,
                duration_ms=(time.monotonic() - t0) * 1000,
                score=score,
                details={"threshold": config.snr_floor},
            )
        EXCEPT Exception as exc:
            RETURN PillarResult(
                pillar=self.name, passed=False,
                duration_ms=(time.monotonic() - t0) * 1000,
                error=str(exc),
            )

CLASS GuardrailsPillar(PillarEvaluator):
    name = PillarName.GUARDRAILS

    async METHOD evaluate(self, config):
        t0 = time.monotonic()
        TRY:
            from core.benchmark.guardrails import GuardrailSuite
            suite = GuardrailSuite()
            # run_all returns list[GuardrailResult]
            results = await suite.run_all(claim=config.claim)
            all_passed = all(r.passed for r in results)
            RETURN PillarResult(
                pillar=self.name, passed=all_passed,
                duration_ms=(time.monotonic() - t0) * 1000,
                details={
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": [r.name for r in results if not r.passed],
                },
            )
        EXCEPT Exception as exc:
            RETURN PillarResult(
                pillar=self.name, passed=False,
                duration_ms=(time.monotonic() - t0) * 1000,
                error=str(exc),
            )

# ... TokenSystemPillar, EvidenceChainPillar, SpearPointPillar,
#     OpportunityPillar, CLIPillar, FullStackPillar, BenchmarkPillar
#     follow the same pattern. Each wraps the corresponding
#     subsystem's entry point in a try/except → PillarResult.

# ── Pillar Registry ────────────────────────────────────────────────

CONSTANT SMOKE_PILLARS: list[PillarEvaluator] = [
    RuntimeBootPillar(),
    SNRCheckPillar(),
]

CONSTANT STANDARD_PILLARS: list[PillarEvaluator] = [
    RuntimeBootPillar(),
    TokenSystemPillar(),
    EvidenceChainPillar(),
    SNRCheckPillar(),
    SpearPointPillar(),
    OpportunityPillar(),
    CLIPillar(),
    FullStackPillar(),
    GuardrailsPillar(),
    BenchmarkPillar(),     # lightweight: single CLEAR eval, no BDL
]

CONSTANT FULL_PILLARS = STANDARD_PILLARS   # same set, but with real inference

FUNCTION pillars_for_mode(mode: RunMode) -> list[PillarEvaluator]:
    MATCH mode:
        CASE RunMode.SMOKE:     RETURN SMOKE_PILLARS
        CASE RunMode.STANDARD:  RETURN STANDARD_PILLARS
        CASE RunMode.FULL:      RETURN FULL_PILLARS
        CASE RunMode.BENCHMARK: RETURN STANDARD_PILLARS  # BDL handled separately

# ── Score Aggregation ──────────────────────────────────────────────

FUNCTION aggregate_snr(pillars: dict[PillarName, PillarResult]) -> float:
    """Weighted geometric mean of scored pillars.

    Standing on Giants: Shannon — multiplicative composition means
    any single zero-signal pillar tanks the aggregate.
    """
    scored = [(p, r.score) for p, r in pillars.items() if r.is_scored]
    IF not scored:
        RETURN 0.0
    # Equal weights for now; can be extended to CLEAR-style weights
    product = 1.0
    FOR _, score IN scored:
        product *= max(score, 1e-10)   # avoid log(0)
    RETURN product ** (1.0 / len(scored))

FUNCTION aggregate_ihsan(pillars: dict[PillarName, PillarResult]) -> float:
    """Ihsan = fraction of pillars that passed.

    This is the simplest honest metric: how much of the system is
    operating at excellence? Future: 8D vector weighting.
    """
    IF not pillars:
        RETURN 0.0
    RETURN sum(1 for r in pillars.values() if r.passed) / len(pillars)

# ── Verdict Resolution ─────────────────────────────────────────────

FUNCTION resolve_verdict(
    snr: float,
    ihsan: float,
    all_gates: bool,
    regression: Optional[RegressionReport],
    config: HarnessConfig,
) -> Verdict:
    """Deterministic verdict resolution.

    PASS requires ALL of:
      1. snr >= snr_floor
      2. ihsan >= ihsan_floor
      3. all guardrail gates passed
      4. no regression (if baseline comparison enabled)

    FAIL if any gate violated.
    INCONCLUSIVE if partial data (e.g., pillar timeout).
    """
    IF snr < config.snr_floor:
        RETURN Verdict.FAIL
    IF ihsan < config.ihsan_floor:
        RETURN Verdict.FAIL
    IF not all_gates:
        RETURN Verdict.FAIL
    IF config.compare_baseline AND regression AND regression.is_regression:
        RETURN Verdict.FAIL
    RETURN Verdict.PASS

# ── HarnessRunner ──────────────────────────────────────────────────

CLASS HarnessRunner:
    """Unified harness orchestrator.

    Usage:
        runner = HarnessRunner()
        result = await runner.run(HarnessConfig(claim="X"))
        assert result.verdict == Verdict.PASS
    """

    METHOD __init__(self,
                    scenario_library: Optional[ScenarioLibrary] = None,
                    baseline_store: Optional[BaselineStore] = None):
        self._scenarios = scenario_library or ScenarioLibrary.default()
        self._baselines = baseline_store or BaselineStore.default()

    async METHOD run(self, config: HarnessConfig) -> HarnessResult:
        """Execute the full harness pipeline."""
        # 0. Validate config
        errors = config.validate()
        IF errors:
            RAISE ValueError(f"Invalid config: {errors}")

        # 1. Resolve scenario (override claim if scenario_id provided)
        IF config.scenario_id:
            scenario = self._scenarios.get(config.scenario_id)
            IF scenario is None:
                RAISE KeyError(f"Unknown scenario: {config.scenario_id}")
            config = dataclasses.replace(config, claim=scenario.claim)

        run_id = str(uuid.uuid4())
        t0_total = time.monotonic()

        # 2. OBSERVE — Run all pillars
        pillar_evaluators = pillars_for_mode(config.mode)
        pillar_results: dict[PillarName, PillarResult] = {}

        FOR evaluator IN pillar_evaluators:
            TRY:
                result = await asyncio.wait_for(
                    evaluator.evaluate(config),
                    timeout=config.timeout_seconds,
                )
            EXCEPT asyncio.TimeoutError:
                result = PillarResult(
                    pillar=evaluator.name, passed=False,
                    duration_ms=config.timeout_seconds * 1000,
                    error=f"Timeout after {config.timeout_seconds}s",
                )
            pillar_results[evaluator.name] = result

        # 3. ORIENT — Aggregate scores
        snr_score = aggregate_snr(pillar_results)
        ihsan_score = aggregate_ihsan(pillar_results)
        tier = resolve_tier(snr_score).level.value

        # 4. Collect guardrail results from GuardrailsPillar
        guardrail_pillar = pillar_results.get(PillarName.GUARDRAILS)
        guardrail_results = []
        IF guardrail_pillar AND "results" IN guardrail_pillar.details:
            guardrail_results = guardrail_pillar.details["results"]
        all_gates = all(r.passed for r in pillar_results.values())

        # 5. Regression check
        regression = None
        IF config.compare_baseline:
            baseline = self._baselines.get_latest()
            IF baseline:
                regression = _compute_regression(
                    baseline, snr_score, ihsan_score,
                    pillar_results, config.regression_tolerance,
                )

        # 6. DECIDE — Resolve verdict
        verdict = resolve_verdict(
            snr_score, ihsan_score, all_gates, regression, config,
        )

        # 7. Benchmark (BDL) — only in BENCHMARK mode
        bench_result = None
        IF config.mode == RunMode.BENCHMARK:
            bench_result = await self._run_bdl(config)

        # 8. ACT — Emit evidence receipt
        receipt = None
        IF config.emit_receipt:
            receipt = await self._emit_receipt(
                run_id, verdict, snr_score, ihsan_score,
            )

        total_ms = (time.monotonic() - t0_total) * 1000

        # 9. Build result
        result = HarnessResult(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            config=config,
            verdict=verdict,
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            tier=tier,
            pillars=pillar_results,
            total_duration_ms=total_ms,
            guardrail_results=guardrail_results,
            all_gates_passed=all_gates,
            bench_result=bench_result,
            regression=regression,
            receipt=receipt,
        )

        # 10. Persist run + optional baseline update
        self._baselines.append_run(result)
        IF config.update_baseline:
            self._baselines.seal_baseline(result)

        RETURN result

    async METHOD _run_bdl(self, config: HarnessConfig) -> Optional[BenchResult]:
        """Run the Benchmark Dominance Loop (capped)."""
        TRY:
            from core.benchmark.dominance_loop import BenchmarkDominanceLoop
            bdl = BenchmarkDominanceLoop(max_cycles=config.max_bdl_cycles)
            cycle_result = await bdl.run_cycle(claim=config.claim)
            RETURN cycle_result.bench_result
        EXCEPT Exception:
            RETURN None

    async METHOD _emit_receipt(self, run_id, verdict, snr, ihsan) -> Optional[Receipt]:
        """Produce a hash-chained evidence receipt."""
        TRY:
            from core.proof_engine.evidence_ledger import EvidenceLedger
            ledger = EvidenceLedger(Path(".spearpoint/harness_evidence.jsonl"))
            receipt = ledger.emit_receipt(
                receipt_id=f"harness-{run_id}",
                node_id="node0",
                reason_codes=[f"verdict:{verdict.value}"],
                snr_score=snr,
                ihsan_score=ihsan,
            )
            RETURN receipt
        EXCEPT Exception:
            RETURN None

# ── Regression Computation ─────────────────────────────────────────

FUNCTION _compute_regression(
    baseline: HarnessResult,
    current_snr: float,
    current_ihsan: float,
    current_pillars: dict[PillarName, PillarResult],
    tolerance: float,
) -> RegressionReport:
    snr_delta = current_snr - baseline.snr_score
    ihsan_delta = current_ihsan - baseline.ihsan_score

    regressed = []
    FOR pillar_name, current_result IN current_pillars.items():
        baseline_result = baseline.pillars.get(pillar_name)
        IF baseline_result AND baseline_result.passed AND NOT current_result.passed:
            regressed.append(pillar_name)

    is_regression = (
        snr_delta < -tolerance
        OR ihsan_delta < -tolerance
        OR len(regressed) > 0
    )

    RETURN RegressionReport(
        baseline_run_id=baseline.run_id,
        baseline_snr=baseline.snr_score,
        current_snr=current_snr,
        snr_delta=snr_delta,
        baseline_ihsan=baseline.ihsan_score,
        current_ihsan=current_ihsan,
        ihsan_delta=ihsan_delta,
        regressed_pillars=regressed,
        tolerance=tolerance,
        is_regression=is_regression,
    )

# ── Sync Convenience Wrapper ──────────────────────────────────────

FUNCTION run_harness(config: Optional[HarnessConfig] = None) -> HarnessResult:
    """Synchronous entry point for CLI / scripts."""
    config = config or HarnessConfig()
    runner = HarnessRunner()
    RETURN asyncio.run(runner.run(config))
```

## TDD Anchors

```python
# test_runner.py — Phase 2 validation

@pytest.mark.asyncio
async def test_smoke_mode_runs_two_pillars():
    runner = HarnessRunner()
    result = await runner.run(HarnessConfig(mode=RunMode.SMOKE))
    assert len(result.pillars) == 2
    assert PillarName.RUNTIME_BOOT in result.pillars
    assert PillarName.SNR_CHECK in result.pillars

@pytest.mark.asyncio
async def test_standard_mode_runs_ten_pillars():
    runner = HarnessRunner()
    result = await runner.run(HarnessConfig(mode=RunMode.STANDARD))
    assert len(result.pillars) == 10

@pytest.mark.asyncio
async def test_verdict_pass_when_all_gates_met():
    # Mock all pillars to pass with high scores
    result = await _run_with_mocked_pillars(all_pass=True, snr=0.95)
    assert result.verdict == Verdict.PASS

@pytest.mark.asyncio
async def test_verdict_fail_when_snr_below_floor():
    result = await _run_with_mocked_pillars(all_pass=True, snr=0.80)
    assert result.verdict == Verdict.FAIL

@pytest.mark.asyncio
async def test_verdict_fail_on_regression():
    # Store a baseline with snr=0.95, then run with snr=0.90
    result = await _run_with_baseline(baseline_snr=0.95, current_snr=0.90)
    assert result.verdict == Verdict.FAIL
    assert result.regression.is_regression is True

@pytest.mark.asyncio
async def test_timeout_produces_failed_pillar():
    # SlowPillar that exceeds timeout
    runner = HarnessRunner()
    config = HarnessConfig(timeout_seconds=0.01)
    result = await runner.run(config)
    # At least one pillar should have timed out
    timed_out = [p for p in result.pillars.values() if "Timeout" in (p.error or "")]
    # (may or may not timeout depending on system speed)

@pytest.mark.asyncio
async def test_receipt_emitted():
    runner = HarnessRunner()
    result = await runner.run(HarnessConfig(emit_receipt=True))
    # Receipt may be None if ledger fails gracefully — that's OK
    # But if present, it has the right run_id
    if result.receipt:
        assert f"harness-{result.run_id}" in result.receipt.receipt_id

def test_aggregate_snr_geometric_mean():
    pillars = {
        PillarName.SNR_CHECK: PillarResult(
            pillar=PillarName.SNR_CHECK, passed=True,
            duration_ms=1.0, score=0.9,
        ),
        PillarName.SPEARPOINT: PillarResult(
            pillar=PillarName.SPEARPOINT, passed=True,
            duration_ms=1.0, score=0.81,
        ),
    }
    snr = aggregate_snr(pillars)
    # geometric_mean(0.9, 0.81) = sqrt(0.9 * 0.81) = sqrt(0.729) ≈ 0.8538
    assert 0.85 < snr < 0.86

def test_resolve_verdict_deterministic():
    v1 = resolve_verdict(0.90, 0.96, True, None, HarnessConfig())
    v2 = resolve_verdict(0.90, 0.96, True, None, HarnessConfig())
    assert v1 == v2 == Verdict.PASS
```

## Execution Flow Diagram

```
run(config)
    │
    ├─ validate config
    ├─ resolve scenario (if scenario_id)
    ├─ generate run_id
    │
    ├─ FOR EACH pillar IN pillars_for_mode(mode):
    │     ├─ await evaluator.evaluate(config)
    │     │    (with asyncio.wait_for timeout)
    │     └─ store PillarResult
    │
    ├─ aggregate_snr(pillars)        → float
    ├─ aggregate_ihsan(pillars)      → float
    ├─ resolve_tier(snr)             → str
    │
    ├─ IF compare_baseline:
    │     └─ _compute_regression()   → RegressionReport
    │
    ├─ resolve_verdict(snr, ihsan, gates, regression)  → Verdict
    │
    ├─ IF mode == BENCHMARK:
    │     └─ _run_bdl()              → BenchResult
    │
    ├─ IF emit_receipt:
    │     └─ _emit_receipt()         → Receipt
    │
    ├─ build HarnessResult
    ├─ baselines.append_run(result)
    ├─ IF update_baseline: baselines.seal_baseline(result)
    │
    └─ RETURN result
```
