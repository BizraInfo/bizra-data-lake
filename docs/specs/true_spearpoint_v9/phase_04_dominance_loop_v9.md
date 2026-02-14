# Phase 04: Dominance Loop v9 Integration

## Status: SPEC
## Depends On: Phases 01-03, existing `core/benchmark/dominance_loop.py`
## Produces: TrueSpearpointLoop (composer of all pillars)

---

## 1. Context

The existing `DominanceLoop` in `core/benchmark/dominance_loop.py` provides:
- 5-phase cycle: EVALUATE → ABLATE → ARCHITECT → SUBMIT → ANALYZE
- `LoopState` with score tracking, budget, patience
- `CycleOutcome` enum (IMPROVED/MAINTAINED/REGRESSED/SOTA_ACHIEVED/FAILED)
- PyO3 InferenceGateway integration (optional)
- Termination conditions: max cycles, budget, target score, patience

### What v9 Adds

The True Spearpoint Loop **composes** the existing DominanceLoop with the new
Phase 01-03 components, adding:

1. **HAL consistency** as a mandatory evaluation dimension
2. **Pareto frontier tracking** (not just single-score optimization)
3. **MIRAS memory** for cross-cycle learning
4. **Campaign orchestration** for multi-battlefield submission
5. **Z-Scorer routing** for architecture phase decisions
6. **Convergence detection** on the Pareto frontier

The v9 loop does NOT replace DominanceLoop — it wraps it with higher-level
orchestration. DominanceLoop remains the inner engine.

---

## 2. TrueSpearpointLoop — The Composer

### Pseudocode

```
MODULE core.spearpoint.true_spearpoint_loop

IMPORT DominanceLoop, LoopPhase, CycleOutcome FROM core.benchmark.dominance_loop
IMPORT HALOrchestrator, ParetoPoint FROM core.benchmark.hal_orchestrator
IMPORT CLEARFramework FROM core.benchmark.clear_framework
IMPORT AblationEngine FROM core.benchmark.ablation_engine
IMPORT MoERouter FROM core.benchmark.moe_router
IMPORT ZScorer FROM core.benchmark.z_scorer
IMPORT MIRASMemory FROM core.benchmark.miras_memory
IMPORT CampaignOrchestrator FROM core.benchmark.campaign_orchestrator
IMPORT Battlefield, BATTLEFIELD_REGISTRY FROM core.benchmark.battlefield_registry
IMPORT EnhancedABCValidator FROM core.benchmark.abc_enhanced
IMPORT GuardrailSuite FROM core.benchmark.guardrails
IMPORT MetricsProvider FROM core.spearpoint.metrics_provider
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants
IMPORT UNIFIED_SNR_THRESHOLD FROM core.integration.constants

DATACLASS LoopConfig:
    max_iterations: int = 20
    target_snr: float = 0.99
    budget_usd: float = 2000.0
    patience: int = 5              # Stop after N non-improving iterations
    pareto_convergence_window: int = 3  # Stable Pareto = converged
    enable_campaign: bool = False  # Submit to real battlefields?
    battlefields: list[Battlefield] = []

DATACLASS IterationResult:
    iteration: int
    phase_results: dict[str, Any]  # EVALUATE/ABLATE/ARCHITECT/SUBMIT/ANALYZE
    snr: float
    ihsan: float
    pareto_rank: int
    pareto_efficient: bool
    outcome: CycleOutcome
    memory_stats: dict
    cost_usd: float

DATACLASS SpearpointReport:
    iterations_completed: int
    final_snr: float
    final_ihsan: float
    pareto_frontier: list[ParetoPoint]
    battlefields_won: list[Battlefield]
    total_cost_usd: float
    convergence_reason: str        # "snr_achieved" | "pareto_stable" |
                                   # "budget_exhausted" | "max_iterations" |
                                   # "patience_exhausted"
    iteration_history: list[IterationResult]
    memory_summary: dict

CLASS TrueSpearpointLoop:
    """
    The recursive Benchmark Dominance Loop v9.

    Composes:
    - DominanceLoop (inner engine: evaluate-ablate-architect-submit-analyze)
    - HALOrchestrator (parallel consistency evaluation)
    - MIRAS Memory (cross-cycle learning)
    - CampaignOrchestrator (multi-battlefield submission)
    - Z-Scorer (intelligent routing)
    - EnhancedABCValidator (benchmark validation)

    Does NOT replace DominanceLoop — wraps it with Pareto tracking,
    memory, and campaign orchestration.
    """

    INIT(config: LoopConfig = None, gateway: Any = None):
        self._config = config OR LoopConfig()

        # Inner engine (existing)
        self._inner_loop = DominanceLoop(inference_gateway=gateway)

        # Phase 01: Evaluation
        self._clear = CLEARFramework()
        self._hal = HALOrchestrator(clear=self._clear, k=4)
        self._abc = EnhancedABCValidator()
        self._guardrails = GuardrailSuite()

        # Phase 02: Architecture
        self._z_scorer = ZScorer()
        self._moe = MoERouter()
        self._memory = MIRASMemory()
        self._ablation = AblationEngine()

        # Phase 03: Submission
        self._campaign = CampaignOrchestrator(
            total_budget_usd=self._config.budget_usd
        )

        # Metrics
        self._metrics = MetricsProvider()

        # State
        self._iteration = 0
        self._history: list[IterationResult] = []
        self._pareto_history: list[list[ParetoPoint]] = []
        self._best_snr = 0.0
        self._patience_counter = 0
        self._spent_usd = 0.0

    ASYNC run() -> SpearpointReport:
        """
        Main entry point. Run the dominance loop until convergence.
        """
        LOG.info(f"True Spearpoint v9 starting. "
                f"Target SNR: {self._config.target_snr}, "
                f"Budget: ${self._config.budget_usd}")

        convergence_reason = "max_iterations"

        WHILE self._iteration < self._config.max_iterations:
            self._iteration += 1

            # Run one full iteration
            result = AWAIT self._run_iteration()
            self._history.append(result)

            # Store in memory for cross-cycle learning
            self._memory.store(
                f"Iteration {self._iteration}: SNR={result.snr:.4f}, "
                f"outcome={result.outcome.value}",
                metadata={"iteration": self._iteration, "snr": result.snr},
            )
            self._memory.store_episodic(
                action=f"iteration_{self._iteration}",
                result=result.outcome.value,
                context=result.phase_results,
            )

            # Check termination conditions
            IF result.snr >= self._config.target_snr:
                convergence_reason = "snr_achieved"
                BREAK

            IF self._pareto_converged():
                convergence_reason = "pareto_stable"
                BREAK

            IF self._spent_usd >= self._config.budget_usd:
                convergence_reason = "budget_exhausted"
                BREAK

            IF result.snr > self._best_snr:
                self._best_snr = result.snr
                self._patience_counter = 0
            ELSE:
                self._patience_counter += 1

            IF self._patience_counter >= self._config.patience:
                convergence_reason = "patience_exhausted"
                BREAK

        RETURN SpearpointReport(
            iterations_completed=self._iteration,
            final_snr=self._history[-1].snr IF self._history ELSE 0.0,
            final_ihsan=self._history[-1].ihsan IF self._history ELSE 0.0,
            pareto_frontier=self._pareto_history[-1] IF self._pareto_history ELSE [],
            battlefields_won=self._get_won_battlefields(),
            total_cost_usd=self._spent_usd,
            convergence_reason=convergence_reason,
            iteration_history=self._history,
            memory_summary=self._memory.get_stats(),
        )

    ASYNC _run_iteration() -> IterationResult:
        """Single iteration of the v9 loop."""
        phase_results = {}

        # ── PHASE 1: EVALUATE ──────────────────────────────────
        eval_result = AWAIT self._phase_evaluate()
        phase_results["evaluate"] = eval_result

        # ── PHASE 2: ABLATE ────────────────────────────────────
        ablation_result = self._phase_ablate(eval_result)
        phase_results["ablate"] = ablation_result

        # ── PHASE 3: ARCHITECT ─────────────────────────────────
        arch_result = self._phase_architect(ablation_result)
        phase_results["architect"] = arch_result

        # ── PHASE 4: SUBMIT (optional) ─────────────────────────
        submit_result = {}
        IF self._config.enable_campaign:
            submit_result = AWAIT self._phase_submit()
        phase_results["submit"] = submit_result

        # ── PHASE 5: ANALYZE ───────────────────────────────────
        analysis = self._phase_analyze(phase_results)
        phase_results["analyze"] = analysis

        # Compute current metrics
        snr = analysis.get("snr", UNIFIED_SNR_THRESHOLD)
        ihsan = analysis.get("ihsan", UNIFIED_IHSAN_THRESHOLD)

        # Update Pareto frontier
        point = ParetoPoint(
            agent_id=f"v9_iter_{self._iteration}",
            cost=analysis.get("cost", 0.0),
            efficacy=analysis.get("efficacy", 0.0),
            reliability=analysis.get("reliability", 0.0),
            latency=analysis.get("latency", 0.0),
            assurance=analysis.get("assurance", 0.0),
        )
        self._hal.update_pareto_frontier(point)
        self._pareto_history.append(list(self._hal._pareto_frontier))

        # Determine outcome
        IF snr > self._best_snr + 0.01:
            outcome = CycleOutcome.IMPROVED
        ELIF snr >= self._best_snr - 0.005:
            outcome = CycleOutcome.MAINTAINED
        ELSE:
            outcome = CycleOutcome.REGRESSED

        RETURN IterationResult(
            iteration=self._iteration,
            phase_results=phase_results,
            snr=snr,
            ihsan=ihsan,
            pareto_rank=self._compute_pareto_rank(point),
            pareto_efficient=self._hal._is_pareto_efficient(point.agent_id),
            outcome=outcome,
            memory_stats=self._memory.get_stats(),
            cost_usd=analysis.get("cost", 0.0),
        )

    # ── Phase Implementations ──────────────────────────────────

    ASYNC _phase_evaluate() -> dict:
        """
        Multi-dimensional evaluation with HAL consistency.
        """
        # Use inner loop's evaluation as baseline
        inner_result = self._inner_loop._phase_evaluate()

        # Add HAL consistency measurement
        task_suite = self._generate_eval_tasks()
        consistency = AWAIT self._hal.evaluate_consistency(
            agent_id=f"v9_iter_{self._iteration}",
            task_suite=task_suite,
            k=4,
        )

        # Record metrics for MetricsProvider
        self._metrics.record_cycle_metrics(
            approved=consistency.pass_at_1 > 0.5,
            rejected=consistency.pass_at_1 <= 0.5,
            clear_score=inner_result.get("clear_score", 0.85),
            ihsan_score=inner_result.get("ihsan_score", 0.95),
        )

        RETURN {
            **inner_result,
            "consistency": {
                "pass_at_1": consistency.pass_at_1,
                "pass_at_k": consistency.pass_at_k,
                "gap": consistency.consistency_gap,
                "reasoning_paradox": consistency.reasoning_paradox,
            },
            "pareto_efficient": consistency.pareto_efficient,
        }

    _phase_ablate(eval_result: dict) -> dict:
        """Identify weak components using AblationEngine."""
        # Query memory for previous ablation results
        memory_context = self._memory.retrieve(
            "ablation results weak components", k=5
        )

        # Use inner loop's ablation
        inner_result = self._inner_loop._phase_ablate()

        # Enrich with memory-informed priorities
        IF memory_context.total_retrieved > 0:
            previous_weak = [e.content FOR e IN memory_context.entries]
            inner_result["memory_informed_priorities"] = previous_weak

        RETURN inner_result

    _phase_architect(ablation_result: dict) -> dict:
        """
        Refine architecture based on ablation + Z-Scorer routing.
        """
        # Use Z-Scorer to classify current weaknesses
        weak_description = str(ablation_result.get("weak_modules", []))
        z_score = self._z_scorer.score(weak_description)

        # Inner loop architecture refinement
        inner_result = self._inner_loop._phase_architect()

        # Recommend routing changes based on Z-Score
        recommendations = []
        IF z_score.complexity.value >= 3:  # COMPLEX or FRONTIER
            recommendations.append("Route complex queries to frontier model")
        IF ablation_result.get("memory_informed_priorities"):
            recommendations.append("Prioritize previously-weak components")

        inner_result["z_score"] = z_score.reasoning
        inner_result["recommendations"] = recommendations

        RETURN inner_result

    ASYNC _phase_submit() -> dict:
        """Submit to battlefields via CampaignOrchestrator."""
        IF NOT self._config.battlefields:
            RETURN {"status": "skipped", "reason": "no battlefields configured"}

        plan = self._campaign.plan(self._config.battlefields)

        # Generate task suites (simulation for now)
        task_suites = {
            bf: [{"id": f"t{i}", "prompt": f"task_{i}"}
                 FOR i IN range(10)]
            FOR bf IN self._config.battlefields
        }

        def agent_fn(prompt):
            RETURN f"Response to: {prompt}"

        report = AWAIT self._campaign.execute(plan, agent_fn, task_suites)
        self._spent_usd += report.total_cost_usd

        RETURN {
            "battlefields_won": [bf.value FOR bf IN report.battlefields_won],
            "total_cost": report.total_cost_usd,
            "overall_kami": report.overall_kami,
        }

    _phase_analyze(phase_results: dict) -> dict:
        """Analyze iteration results and compute aggregate metrics."""
        eval_r = phase_results.get("evaluate", {})

        snr = eval_r.get("clear_score", UNIFIED_SNR_THRESHOLD)
        ihsan = eval_r.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)

        consistency = eval_r.get("consistency", {})
        pass_at_k = consistency.get("pass_at_k", 0.8)

        # Weighted SNR incorporating consistency
        adjusted_snr = 0.7 * snr + 0.3 * pass_at_k

        RETURN {
            "snr": adjusted_snr,
            "ihsan": ihsan,
            "raw_snr": snr,
            "pass_at_k": pass_at_k,
            "cost": phase_results.get("submit", {}).get("total_cost", 0.0),
            "efficacy": eval_r.get("efficacy", 0.7),
            "reliability": pass_at_k,
            "latency": eval_r.get("latency", 0.9),
            "assurance": eval_r.get("assurance", 0.9),
        }

    # ── Convergence Detection ──────────────────────────────────

    _pareto_converged() -> bool:
        """
        Pareto frontier is stable if last N snapshots are identical.
        """
        window = self._config.pareto_convergence_window
        IF len(self._pareto_history) < window:
            RETURN False

        recent = self._pareto_history[-window:]
        # Compare frontier sizes and agent IDs
        FOR i IN range(1, len(recent)):
            IF len(recent[i]) != len(recent[0]):
                RETURN False
            ids_prev = {p.agent_id FOR p IN recent[i-1]}
            ids_curr = {p.agent_id FOR p IN recent[i]}
            IF ids_prev != ids_curr:
                RETURN False
        RETURN True

    _compute_pareto_rank(point: ParetoPoint) -> int:
        """Rank = number of points that dominate this one + 1."""
        rank = 1
        FOR p IN self._hal._pareto_frontier:
            IF p.agent_id == point.agent_id:
                CONTINUE
            IF (p.efficacy >= point.efficacy AND
                p.reliability >= point.reliability AND
                p.cost <= point.cost):
                rank += 1
        RETURN rank

    _get_won_battlefields() -> list[Battlefield]:
        """Collect all battlefields won across iterations."""
        won = set()
        FOR result IN self._history:
            submit = result.phase_results.get("submit", {})
            FOR bf_name IN submit.get("battlefields_won", []):
                won.add(Battlefield(bf_name))
        RETURN list(won)

    _generate_eval_tasks() -> list[dict]:
        """Generate evaluation task suite from memory + defaults."""
        tasks = [{"id": f"eval_{i}", "prompt": f"evaluate_task_{i}"}
                 FOR i IN range(20)]
        RETURN tasks
```

---

## 3. TDD Anchors

```
TEST test_loop_terminates_on_max_iterations:
    config = LoopConfig(max_iterations=3)
    loop = TrueSpearpointLoop(config=config)
    report = AWAIT loop.run()
    ASSERT report.iterations_completed <= 3
    ASSERT report.convergence_reason IN ["max_iterations", "snr_achieved",
                                         "pareto_stable", "patience_exhausted"]

TEST test_loop_terminates_on_snr_target:
    config = LoopConfig(max_iterations=100, target_snr=0.50)  # Easy target
    loop = TrueSpearpointLoop(config=config)
    report = AWAIT loop.run()
    # Should converge before max iterations if adjusted_snr >= 0.50

TEST test_loop_tracks_pareto_history:
    config = LoopConfig(max_iterations=3)
    loop = TrueSpearpointLoop(config=config)
    report = AWAIT loop.run()
    ASSERT len(loop._pareto_history) == report.iterations_completed

TEST test_loop_stores_in_memory:
    config = LoopConfig(max_iterations=2)
    loop = TrueSpearpointLoop(config=config)
    report = AWAIT loop.run()
    ASSERT report.memory_summary["short_term_count"] > 0
    ASSERT report.memory_summary["episodic_count"] > 0

TEST test_pareto_convergence_detection:
    loop = TrueSpearpointLoop()
    # Simulate identical frontiers
    same_frontier = [ParetoPoint("A", 1, 0.9, 0.9, 0.1, 0.9)]
    loop._pareto_history = [same_frontier, same_frontier, same_frontier]
    loop._config.pareto_convergence_window = 3
    ASSERT loop._pareto_converged() == True

TEST test_pareto_not_converged_if_changing:
    loop = TrueSpearpointLoop()
    f1 = [ParetoPoint("A", 1, 0.9, 0.9, 0.1, 0.9)]
    f2 = [ParetoPoint("B", 2, 0.8, 0.8, 0.2, 0.8)]
    f3 = [ParetoPoint("A", 1, 0.9, 0.9, 0.1, 0.9)]
    loop._pareto_history = [f1, f2, f3]
    loop._config.pareto_convergence_window = 3
    ASSERT loop._pareto_converged() == False

TEST test_patience_exhaustion:
    config = LoopConfig(max_iterations=100, patience=2, target_snr=0.999)
    loop = TrueSpearpointLoop(config=config)
    # After 2 non-improving iterations, should stop
    report = AWAIT loop.run()
    ASSERT report.convergence_reason IN ["patience_exhausted", "max_iterations"]

TEST test_budget_exhaustion:
    config = LoopConfig(max_iterations=100, budget_usd=0.01,
                        enable_campaign=True,
                        battlefields=[Battlefield.HLE])
    loop = TrueSpearpointLoop(config=config)
    report = AWAIT loop.run()
    ASSERT report.total_cost_usd <= config.budget_usd + 0.01  # Tolerance
```

---

## 4. File Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `core/spearpoint/true_spearpoint_loop.py` | ~350 | V9 loop composer |
| `tests/core/spearpoint/test_true_spearpoint_loop.py` | ~200 | Loop tests |
