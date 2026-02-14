# Phase 01: Evaluation Harness Upgrades

## Status: SPEC
## Depends On: Existing `core/benchmark/clear_framework.py`, `core/benchmark/guardrails.py`
## Produces: HALOrchestrator, SteeringVectorAdapter, enhanced ABCValidator

---

## 1. Context

The existing `CLEARFramework` evaluates along 5 dimensions (Cost, Latency, Efficacy,
Assurance, Reliability) but runs **single-pass, sequential** evaluations. The True
Spearpoint v9 requires:

- **Parallel consistency evaluation** (HAL pattern: Pass@k across N runs)
- **Steering vector support** for controlled evaluation of reasoning models
- **Enhanced ABC checklist** with automated validation (not just manual flags)
- **Pareto frontier tracking** across agents and configurations

### Existing Code to Reuse (NOT duplicate)

| Component | File | Reuse How |
|-----------|------|-----------|
| `CLEARFramework` | `core/benchmark/clear_framework.py` | Compose — HAL wraps it |
| `EvaluationContext` | `clear_framework.py:EvaluationContext` | Use as inner evaluator |
| `AgenticBenchmarkChecklist` | `clear_framework.py` | Extend with automation |
| `GuardrailSuite` | `core/benchmark/guardrails.py` | Call `check_all()` per run |
| `MetricsProvider` | `core/spearpoint/metrics_provider.py` | Feed real metrics |

---

## 2. HALOrchestrator — Parallel Consistency Evaluator

### Purpose
Run N identical evaluation passes and measure **consistency** (Pass@k) rather
than single-run accuracy. Reveals brittleness the existing single-pass misses.

### Pseudocode

```
MODULE core.benchmark.hal_orchestrator

IMPORT CLEARFramework FROM core.benchmark.clear_framework
IMPORT GuardrailSuite FROM core.benchmark.guardrails
IMPORT MetricsProvider FROM core.spearpoint.metrics_provider
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

DATACLASS ConsistencyReport:
    pass_at_1: float          # Single-run success rate
    pass_at_k: float          # Consistent success across k runs
    consistency_gap: float    # pass_at_1 - pass_at_k  (brittleness)
    run_variance: float       # Variance across runs
    clear_scores: list[float] # Per-run CLEAR scores
    reasoning_paradox: bool   # True if more reasoning = worse consistency
    pareto_efficient: bool    # On Pareto frontier?

DATACLASS ParetoPoint:
    agent_id: str
    cost: float
    efficacy: float
    reliability: float
    latency: float
    assurance: float

CLASS HALOrchestrator:
    """
    Wraps CLEARFramework to run parallel evaluations and measure consistency.
    Does NOT replace CLEARFramework — composes it.
    """

    INIT(clear: CLEARFramework = None, k: int = 8):
        self._clear = clear OR CLEARFramework()
        self._guardrails = GuardrailSuite()
        self._k = k
        self._pareto_frontier: list[ParetoPoint] = []
        self._history: list[ConsistencyReport] = []

    ASYNC evaluate_consistency(
        agent_id: str,
        task_suite: list[dict],
        k: int = None,
        inference_fn: callable = None,
    ) -> ConsistencyReport:
        """Run k passes of CLEAR evaluation, measure consistency."""
        k = k OR self._k
        run_scores: list[float] = []
        run_successes: list[list[bool]] = []

        FOR run_idx IN range(k):
            # Each run uses fresh EvaluationContext
            per_task_success = []

            FOR task IN task_suite:
                eval_id = f"{agent_id}_run{run_idx}_{task['id']}"

                WITH self._clear.evaluate(eval_id, agent_id) AS ctx:
                    # Run inference if fn provided
                    IF inference_fn:
                        result = AWAIT inference_fn(task['prompt'])
                        ctx.record_efficacy(
                            accuracy=result.get('accuracy', 0.0),
                            task_completion=result.get('completion', 0.0),
                        )
                        ctx.record_cost(
                            usd=result.get('cost_usd', 0.0),
                            tokens=result.get('tokens', 0),
                        )
                        ctx.record_latency(
                            total_ms=result.get('latency_ms', 0.0),
                        )
                    ELSE:
                        # Simulation fallback
                        ctx.record_efficacy(accuracy=0.7, task_completion=0.6)

                metrics = self._clear.get_metrics(eval_id)
                score = metrics.compute_overall_score()
                per_task_success.append(score >= UNIFIED_IHSAN_THRESHOLD)

            run_scores.append(MEAN(per_task_success))
            run_successes.append(per_task_success)

        # Compute Pass@k: task succeeds only if ALL k runs succeed
        pass_at_1 = MEAN(run_scores)
        consistent_successes = []
        FOR task_idx IN range(len(task_suite)):
            all_passed = ALL(run_successes[r][task_idx] FOR r IN range(k))
            consistent_successes.append(all_passed)
        pass_at_k = MEAN(consistent_successes)

        report = ConsistencyReport(
            pass_at_1=pass_at_1,
            pass_at_k=pass_at_k,
            consistency_gap=pass_at_1 - pass_at_k,
            run_variance=VARIANCE(run_scores),
            clear_scores=run_scores,
            reasoning_paradox=self._detect_reasoning_paradox(run_successes),
            pareto_efficient=self._is_pareto_efficient(agent_id),
        )

        self._history.append(report)
        RETURN report

    _detect_reasoning_paradox(run_successes) -> bool:
        """
        Counter-intuitive: higher reasoning effort sometimes reduces
        consistency. Detect if later runs (warmer context) perform worse.
        """
        IF len(run_successes) < 4:
            RETURN False
        early_rate = MEAN(FLATTEN(run_successes[:2]))
        late_rate = MEAN(FLATTEN(run_successes[-2:]))
        RETURN late_rate < early_rate - 0.05

    update_pareto_frontier(point: ParetoPoint):
        """Add point, recompute Pareto frontier."""
        self._pareto_frontier.append(point)
        self._pareto_frontier = _compute_pareto(self._pareto_frontier)

    _is_pareto_efficient(agent_id: str) -> bool:
        """Check if agent_id is on current Pareto frontier."""
        RETURN ANY(p.agent_id == agent_id FOR p IN self._pareto_frontier)
```

### Pareto Computation (helper)

```
FUNCTION _compute_pareto(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """
    Retain only non-dominated points.
    Point A dominates B if A is >= B on all dimensions and > on at least one.
    Dimensions: efficacy (max), reliability (max), -cost (min), -latency (min).
    """
    frontier = []
    FOR p IN points:
        dominated = False
        FOR q IN points:
            IF q IS p: CONTINUE
            IF (q.efficacy >= p.efficacy AND
                q.reliability >= p.reliability AND
                q.cost <= p.cost AND
                q.latency <= p.latency AND
                q.assurance >= p.assurance AND
                (q.efficacy > p.efficacy OR q.reliability > p.reliability OR
                 q.cost < p.cost OR q.latency < p.latency OR
                 q.assurance > p.assurance)):
                dominated = True
                BREAK
        IF NOT dominated:
            frontier.append(p)
    RETURN frontier
```

---

## 3. Steering Vector Adapter

### Purpose
Enable evaluation of reasoning models with **controllable behavior** via
activation steering. Integrates with inference backends (LM Studio, vLLM).

### Pseudocode

```
MODULE core.benchmark.steering_adapter

DATACLASS SteeringConfig:
    hookpoint: str          # Model layer (e.g. "layers.12.attention")
    action: str             # "add" | "multiply" | "ablate"
    coefficient: float      # Steering strength
    feature_index: int      # Sparse feature index
    source: str             # "manual" | "sae_lens" | "sparsify"

CLASS SteeringVectorAdapter:
    """
    Wraps inference calls with activation steering vectors.
    No torch dependency at import time — lazy-loads only if steering applied.
    """

    INIT():
        self._configs: list[SteeringConfig] = []
        self._active = False

    load_from_csv(path: str):
        """Load steering configs from CSV (sparsify/sae_lens export format)."""
        # Validate path exists, read CSV
        # Parse each row into SteeringConfig
        # Validate hookpoint format, action in allowed set
        self._configs = parsed_configs
        self._active = len(self._configs) > 0

    load_manual(configs: list[SteeringConfig]):
        """Programmatic steering config."""
        self._configs = configs
        self._active = len(configs) > 0

    wrap_inference(inference_fn: callable) -> callable:
        """
        Return wrapped inference function that applies steering.
        If no steering active, returns original function unchanged.
        """
        IF NOT self._active:
            RETURN inference_fn

        ASYNC WRAPPED(prompt, **kwargs):
            # Pre-inference: inject steering into kwargs
            kwargs['steering_vectors'] = [
                {'hookpoint': c.hookpoint,
                 'action': c.action,
                 'coefficient': c.coefficient,
                 'feature_index': c.feature_index}
                FOR c IN self._configs
            ]
            RETURN AWAIT inference_fn(prompt, **kwargs)

        RETURN WRAPPED

    @property
    is_active -> bool:
        RETURN self._active
```

### Integration Note
Steering vectors are **backend-dependent**. LM Studio and vLLM support them
natively. For Ollama, steering is a no-op (logged as warning). The adapter
does NOT implement steering logic — it serializes configs into the inference
call and the backend handles application.

---

## 4. Enhanced ABC Validator

### Purpose
Extend the existing `AgenticBenchmarkChecklist` with **automated validation**
that can run checks programmatically (not just flag "met/unmet").

### Pseudocode

```
MODULE core.benchmark.abc_enhanced (extends clear_framework.AgenticBenchmarkChecklist)

IMPORT AgenticBenchmarkChecklist FROM core.benchmark.clear_framework
IMPORT GuardrailSuite FROM core.benchmark.guardrails

DATACLASS ABCValidationResult:
    criterion: str
    passed: bool
    method: str            # "automated" | "manual" | "skipped"
    evidence: str          # What was checked
    estimated_bias: float  # Estimated overestimation if this fails (0.0-1.0)

CLASS EnhancedABCValidator(AgenticBenchmarkChecklist):
    """
    Adds automated validation methods for each ABC criterion.
    Falls back to manual flag if automated check not possible.
    """

    INIT():
        SUPER().__init__()
        self._guardrails = GuardrailSuite()

    validate_all(benchmark_data: dict) -> list[ABCValidationResult]:
        results = []

        # 1. Reward design — check for gaming via null-model test
        results.append(self._check_reward_gaming(benchmark_data))

        # 2. Sufficient test cases — count >= 100 per type
        results.append(self._check_test_case_count(benchmark_data))

        # 3. Environment stochasticity — run same input twice, compare
        results.append(self._check_stochasticity(benchmark_data))

        # 4. State isolation — verify no cross-run leakage
        results.append(self._check_state_isolation(benchmark_data))

        # 5. Tool availability — verify all declared tools accessible
        results.append(self._check_tool_availability(benchmark_data))

        # 6. Human baseline — check if baseline score provided
        results.append(self._check_human_baseline(benchmark_data))

        # 7. Failure modes — check if failure catalogue exists
        results.append(self._check_failure_catalogue(benchmark_data))

        # 8. Cost tracking — verify cost metrics recorded
        results.append(self._check_cost_tracking(benchmark_data))

        # 9. Latency bounds — verify SLA defined
        results.append(self._check_latency_bounds(benchmark_data))

        # 10. Reproducibility — verify seed fixed
        results.append(self._check_reproducibility_seed(benchmark_data))

        RETURN results

    estimated_overestimation(results: list[ABCValidationResult]) -> float:
        """
        Sum estimated bias from all failed criteria.
        Capped at 1.0 (100% overestimation).
        """
        total = SUM(r.estimated_bias FOR r IN results IF NOT r.passed)
        RETURN MIN(total, 1.0)
```

---

## 5. TDD Anchors

```
TEST test_hal_consistency_report:
    hal = HALOrchestrator(k=4)
    tasks = [{"id": f"t{i}", "prompt": f"task_{i}"} for i in range(10)]
    report = AWAIT hal.evaluate_consistency("test-agent", tasks, k=4)
    ASSERT 0.0 <= report.pass_at_1 <= 1.0
    ASSERT 0.0 <= report.pass_at_k <= 1.0
    ASSERT report.consistency_gap == report.pass_at_1 - report.pass_at_k
    ASSERT report.consistency_gap >= 0.0  # pass_at_k <= pass_at_1

TEST test_hal_pareto_frontier:
    hal = HALOrchestrator()
    hal.update_pareto_frontier(ParetoPoint("A", cost=10, efficacy=0.9, ...))
    hal.update_pareto_frontier(ParetoPoint("B", cost=5, efficacy=0.95, ...))
    # B dominates A (lower cost, higher efficacy)
    ASSERT hal._is_pareto_efficient("B") == True
    ASSERT hal._is_pareto_efficient("A") == False

TEST test_steering_adapter_noop_when_inactive:
    adapter = SteeringVectorAdapter()
    original_fn = MOCK_ASYNC_FN
    wrapped = adapter.wrap_inference(original_fn)
    ASSERT wrapped IS original_fn  # No steering = no wrapping

TEST test_steering_adapter_injects_vectors:
    adapter = SteeringVectorAdapter()
    adapter.load_manual([SteeringConfig(...)])
    wrapped = adapter.wrap_inference(mock_fn)
    ASSERT wrapped IS NOT mock_fn
    result = AWAIT wrapped("test prompt")
    ASSERT 'steering_vectors' IN mock_fn.last_kwargs

TEST test_abc_validator_catches_insufficient_test_cases:
    validator = EnhancedABCValidator()
    data = {"test_cases": {"type_a": [1,2,3]}}  # Only 3 cases
    results = validator.validate_all(data)
    test_count_result = results[1]  # Index 1 = sufficient test cases
    ASSERT test_count_result.passed == False
    ASSERT test_count_result.estimated_bias > 0.0

TEST test_abc_estimated_overestimation:
    validator = EnhancedABCValidator()
    results = [
        ABCValidationResult("r1", passed=False, estimated_bias=0.3, ...),
        ABCValidationResult("r2", passed=True, estimated_bias=0.2, ...),
        ABCValidationResult("r3", passed=False, estimated_bias=0.4, ...),
    ]
    ASSERT validator.estimated_overestimation(results) == 0.7  # 0.3 + 0.4
```

---

## 6. File Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `core/benchmark/hal_orchestrator.py` | ~200 | Parallel consistency evaluator |
| `core/benchmark/steering_adapter.py` | ~100 | Steering vector injection |
| `core/benchmark/abc_enhanced.py` | ~150 | Automated ABC validation |
| `tests/core/benchmark/test_hal_orchestrator.py` | ~150 | HAL tests |
| `tests/core/benchmark/test_steering_adapter.py` | ~80 | Steering tests |
| `tests/core/benchmark/test_abc_enhanced.py` | ~100 | ABC tests |
