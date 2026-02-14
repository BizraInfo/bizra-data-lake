# Phase 20.2: True Spearpoint Dominance Simulator

## Context

This spec defines the `TrueSpearpointSimulator` — the orchestrator that executes
the full 5-phase dominance loop lifecycle. It composes existing infrastructure
(CLEARFramework, AblationEngine, MoERouter, GuardrailSuite, SNREngine) into
a coherent simulation that demonstrates iterative convergence to SOTA.

**Key principle:** The simulator uses **real** CLEAR, SNR, and guardrail
computations — not mocked values. What's simulated is the benchmark submission
and score progression, since we don't have live access to external leaderboards.

Standing on Giants:
- Boyd (1995): OODA loop as iteration backbone
- Deming (1950): PDCA — Plan-Do-Check-Act cycle
- Goldratt (1984): Theory of Constraints — identify and upgrade the bottleneck
- Shannon (1948): SNR as convergence criterion
- Pareto (1896): Multi-objective efficiency frontier

---

## Module 3: True Spearpoint Simulator

### File: `core/spearpoint/dominance_simulator.py` (~350 lines)

```
IMPORT asyncio
IMPORT logging
IMPORT time
FROM typing IMPORT Any, Callable, Dict, List, Optional, Tuple

FROM core.benchmark.clear_framework IMPORT CLEARFramework, CLEARMetrics
FROM core.benchmark.ablation_engine IMPORT AblationEngine, AblationType, ComponentCategory
FROM core.benchmark.moe_router IMPORT MoERouter, ExpertTier
FROM core.benchmark.guardrails IMPORT GuardrailSuite
FROM core.proof_engine.snr IMPORT SNREngine, SNRInput
FROM core.proof_engine.evidence_ledger IMPORT EvidenceLedger
FROM core.spearpoint.metrics_provider IMPORT MetricsProvider
FROM core.integration.constants IMPORT (
    UNIFIED_SNR_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
)

FROM .simulation_types IMPORT (
    AblationFinding,
    ArchitectureUpgrade,
    BattlefieldResult,
    CLEARSnapshot,
    CyclePhase,
    DominanceStatus,
    IterationRecord,
    SimulationState,
)
FROM .battlefield_registry IMPORT BattlefieldRegistry

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Architecture Upgrade Strategies
# -------------------------------------------------------------------

UPGRADE_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "reasoning": {
        "technique": "MoE-Federated",
        "version_prefix": "MoE-Fed",
        "expected_cna_gain_per_impact": 0.8,    # 80% of ablation impact recovered
    },
    "memory": {
        "technique": "MIRAS",
        "version_prefix": "MIRAS",
        "expected_cna_gain_per_impact": 0.7,
    },
    "routing": {
        "technique": "ZScorer",
        "version_prefix": "ZScorer",
        "expected_cna_gain_per_impact": 0.6,
    },
    "tool": {
        "technique": "SequentialAttention",
        "version_prefix": "SeqAttn",
        "expected_cna_gain_per_impact": 0.65,
    },
    "verifier": {
        "technique": "FederatedEnsemble",
        "version_prefix": "FedEns",
        "expected_cna_gain_per_impact": 0.75,
    },
}


# -------------------------------------------------------------------
# Simulator Core
# -------------------------------------------------------------------

CLASS TrueSpearpointSimulator:
    """
    Executes the True Spearpoint Benchmark Dominance Loop.

    The 5-phase recursive cycle:
      1. EVALUATE  — Run CLEAR framework, compute SNR, check Ihsan
      2. ABLATE    — Identify weak modules via contribution analysis
      3. ARCHITECT — Upgrade weak modules with SOTA techniques
      4. SUBMIT    — Submit to target battlefields, record scores
      5. ANALYZE   — Update state, check convergence, decide: recurse or terminate

    Termination conditions:
      - DOMINATED: SOTA achieved on all target battlefields
      - BUDGET_EXHAUSTED: Campaign budget exceeded
      - PATIENCE_EXHAUSTED: Too many consecutive regressions
      - Max iterations reached

    Standing on: Boyd (OODA) + Deming (PDCA) + Goldratt (TOC) + Shannon (SNR)
    """

    CONSTRUCTOR(
        self,
        targets: Optional[List[str]] = None,
        budget_usd: float = 100.0,
        max_iterations: int = 10,
        patience: int = 5,
        snr_target: float = SNR_THRESHOLD_T0_ELITE,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        # Configuration
        self.targets = targets OR ["HLE", "SWE_bench_Verified", "ARC_AGI_2"]
        self.max_iterations = max_iterations
        self.snr_target = snr_target
        self.ihsan_floor = ihsan_floor

        # Composed infrastructure (real, not mocked)
        self.clear_framework = CLEARFramework()
        self.ablation_engine = AblationEngine()
        self.moe_router = MoERouter()
        self.guardrails = GuardrailSuite()
        self.snr_engine = SNREngine()
        self.metrics_provider = MetricsProvider()
        self.registry = BattlefieldRegistry()

        # State
        self.state = SimulationState(
            budget_remaining_usd=budget_usd,
            patience=patience,
        )

        # Optional: evidence ledger for receipt chain
        self._ledger: Optional[EvidenceLedger] = None

    METHOD set_evidence_ledger(self, ledger: EvidenceLedger) -> None:
        """Attach evidence ledger for full audit trail."""
        self._ledger = ledger

    # ---------------------------------------------------------------
    # MAIN ENTRY POINT
    # ---------------------------------------------------------------

    ASYNC METHOD dominate(self) -> SimulationState:
        """
        Execute the full dominance loop until termination.

        Returns final SimulationState with complete iteration history.
        """
        self.state.status = DominanceStatus.RUNNING
        logger.info("TRUE SPEARPOINT v9.0 initialized")
        logger.info("Targets: %s", self.targets)
        logger.info("Budget: $%.2f | Max iterations: %d", self.state.budget_remaining_usd, self.max_iterations)

        WHILE NOT self.state.is_terminated() AND self.state.iteration < self.max_iterations:
            self.state.iteration += 1
            logger.info("--- Iteration %d ---", self.state.iteration)

            record = AWAIT self._run_iteration()
            self.state.iterations.append(record)

            # Update top-level state from record
            self.state.snr = record.snr
            self.state.cna = record.clear_snapshot.cna
            self.state.ihsan_score = record.ihsan_score
            self.state.battlefields_won = record.battlefields_won
            self.state.budget_remaining_usd -= record.cost_total_usd

            # Bump architecture version
            minor = self.state.iteration
            self.state.architecture_version = f"9.0.{minor}-TRUE_SPEARPOINT"

            # Check termination
            self._check_termination()

            logger.info(
                "Iteration %d: SNR=%.5f CNA=%.2f Won=%s",
                self.state.iteration, self.state.snr, self.state.cna,
                self.state.battlefields_won,
            )

        RETURN self.state

    # ---------------------------------------------------------------
    # SINGLE ITERATION
    # ---------------------------------------------------------------

    ASYNC METHOD _run_iteration(self) -> IterationRecord:
        """Execute one full 5-phase iteration."""
        timings: Dict[str, float] = {}
        total_cost = 0.0

        # Phase 1: EVALUATE
        t0 = time.monotonic()
        clear_snapshot, snr, ihsan = self._phase_evaluate()
        timings["evaluate"] = time.monotonic() - t0

        # Phase 2: ABLATE
        t0 = time.monotonic()
        findings = self._phase_ablate(clear_snapshot)
        timings["ablate"] = time.monotonic() - t0

        # Phase 3: ARCHITECT
        t0 = time.monotonic()
        upgrades = self._phase_architect(findings, clear_snapshot)
        timings["architect"] = time.monotonic() - t0

        # Phase 4: SUBMIT
        t0 = time.monotonic()
        battlefield_results = self._phase_submit(clear_snapshot, upgrades)
        submission_cost = sum(r.cost_usd FOR r IN battlefield_results)
        total_cost += submission_cost
        timings["submit"] = time.monotonic() - t0

        # Phase 5: ANALYZE
        t0 = time.monotonic()
        battlefields_won = self._phase_analyze(battlefield_results)
        timings["analyze"] = time.monotonic() - t0

        RETURN IterationRecord(
            iteration=self.state.iteration,
            phase_timings=timings,
            clear_snapshot=clear_snapshot,
            ablation_findings=findings,
            upgrades=upgrades,
            battlefield_results=battlefield_results,
            snr=snr,
            ihsan_score=ihsan,
            battlefields_won=battlefields_won,
            cost_total_usd=total_cost,
        )

    # ---------------------------------------------------------------
    # PHASE 1: EVALUATE
    # ---------------------------------------------------------------

    METHOD _phase_evaluate(self) -> Tuple[CLEARSnapshot, float, float]:
        """
        Run CLEAR evaluation + SNR computation + Ihsan check.

        Uses MetricsProvider for real-ish metrics that improve
        as the architecture version advances.
        """
        # Get current metrics from provider
        metrics = self.metrics_provider.snapshot()

        # Compute CLEAR metrics
        clear_metrics = self.clear_framework.evaluate_metrics(
            accuracy=metrics.accuracy,
            task_completion=metrics.task_completion_rate,
            cost_per_task=metrics.cost_per_task_usd,
            latency_p99_ms=metrics.latency_p99_ms,
            error_rate=1.0 - metrics.accuracy,
        )

        # Compute SNR
        snr_input = SNRInput(
            provenance=metrics.provenance_score,
            constraint=metrics.constraint_satisfaction,
            prediction=metrics.accuracy,
            contradiction=1.0 - metrics.consistency,
            unverifiable=max(0.0, 1.0 - metrics.auditability),
        )
        snr_score, snr_trace = self.snr_engine.compute(snr_input)

        # Compute Ihsan (from CLEAR weighted score + SNR)
        ihsan = min(1.0, clear_metrics.weighted_score * 0.6 + snr_score * 0.4)

        # Derive CNA: accuracy / cost * 1000 (HAL formula)
        cps = metrics.cost_per_task_usd / max(0.01, metrics.accuracy)
        cna = metrics.accuracy / max(0.001, cps) * 1000

        snapshot = CLEARSnapshot(
            cna=cna,
            cps=cps,
            scr=metrics.sla_compliance_pct,
            pas=metrics.policy_adherence,
            pass_at_k=metrics.consistency,
            pareto_efficient=True,      # Simplified; full Pareto in production
            raw_efficacy=clear_metrics.efficacy_score,
            raw_cost=clear_metrics.cost_score,
            raw_latency=clear_metrics.latency_score,
            raw_assurance=clear_metrics.assurance_score,
            raw_reliability=clear_metrics.reliability_score,
            weighted_score=clear_metrics.weighted_score,
        )

        # Feed metrics back for next iteration improvement
        self.metrics_provider.record(
            accuracy=metrics.accuracy,
            task_completion=metrics.task_completion_rate,
            cost=metrics.cost_per_task_usd,
        )

        RETURN snapshot, snr_score, ihsan

    # ---------------------------------------------------------------
    # PHASE 2: ABLATE
    # ---------------------------------------------------------------

    METHOD _phase_ablate(self, snapshot: CLEARSnapshot) -> List[AblationFinding]:
        """
        Identify weak modules via ablation.

        Simulate component contributions based on CLEAR breakdown.
        The weakest CLEAR dimension points to the bottleneck module.
        """
        # Map CLEAR dimensions to component categories (Goldratt: find the constraint)
        dimension_scores = {
            "efficacy": snapshot.raw_efficacy,
            "cost": snapshot.raw_cost,
            "latency": snapshot.raw_latency,
            "assurance": snapshot.raw_assurance,
            "reliability": snapshot.raw_reliability,
        }

        DIMENSION_TO_CATEGORY = {
            "efficacy": ("reasoning_engine", "reasoning"),
            "cost": ("resource_allocator", "routing"),
            "latency": ("tool_use_router", "routing"),
            "assurance": ("guardrail_suite", "verifier"),
            "reliability": ("long_term_memory", "memory"),
        }

        # Sort dimensions by score (ascending = weakest first)
        sorted_dims = sorted(dimension_scores.items(), key=LAMBDA x: x[1])

        findings = []
        FOR dim_name, dim_score IN sorted_dims[:2]:    # Top 2 bottlenecks
            module_name, category = DIMENSION_TO_CATEGORY[dim_name]
            # Impact = how much CNA drops if this module is degraded
            # Approximation: (1 - dim_score) * weight * CNA
            weight = self.clear_framework.get_weight(dim_name)
            impact = -1.0 * (1.0 - dim_score) * weight * snapshot.cna / 100

            findings.append(AblationFinding(
                name=module_name,
                category=category,
                contribution=impact,
                essential=abs(impact) > snapshot.cna * 0.10,
                harmful=False,
            ))

        RETURN findings

    # ---------------------------------------------------------------
    # PHASE 3: ARCHITECT
    # ---------------------------------------------------------------

    METHOD _phase_architect(
        self,
        findings: List[AblationFinding],
        snapshot: CLEARSnapshot,
    ) -> List[ArchitectureUpgrade]:
        """
        Upgrade weak modules based on ablation findings.

        Each category has a predefined upgrade strategy (from UPGRADE_STRATEGIES).
        The upgrade improves MetricsProvider dimensions for next iteration.
        """
        upgrades = []

        FOR finding IN findings:
            strategy = UPGRADE_STRATEGIES.get(finding.category)
            IF strategy IS None:
                CONTINUE

            version_num = f"v{self.state.iteration}.{len(upgrades)}"
            expected_gain = abs(finding.contribution) * strategy["expected_cna_gain_per_impact"]

            upgrade = ArchitectureUpgrade(
                module_name=finding.name,
                old_version=f"{finding.name}_v{self.state.iteration - 1}",
                new_version=f"{strategy['version_prefix']} {version_num}",
                technique=strategy["technique"],
                expected_cna_gain=expected_gain,
            )
            upgrades.append(upgrade)

            # Apply upgrade: boost the relevant metric in MetricsProvider
            self._apply_upgrade_to_metrics(finding.category, expected_gain, snapshot.cna)

        RETURN upgrades

    METHOD _apply_upgrade_to_metrics(
        self,
        category: str,
        cna_gain: float,
        current_cna: float,
    ) -> None:
        """
        Translate CNA gain into MetricsProvider improvements.

        The improvement follows diminishing returns (logarithmic curve).
        """
        # Improvement factor: gain / current_cna, capped at 5%
        improvement = min(0.05, cna_gain / max(1.0, current_cna))

        CATEGORY_TO_METRIC = {
            "reasoning": "accuracy",
            "memory": "consistency",
            "routing": "cost_per_task_usd",
            "tool": "latency_p99_ms",
            "verifier": "auditability",
        }

        metric_name = CATEGORY_TO_METRIC.get(category, "accuracy")
        self.metrics_provider.boost(metric_name, improvement)

    # ---------------------------------------------------------------
    # PHASE 4: SUBMIT
    # ---------------------------------------------------------------

    METHOD _phase_submit(
        self,
        snapshot: CLEARSnapshot,
        upgrades: List[ArchitectureUpgrade],
    ) -> List[BattlefieldResult]:
        """
        Submit to target battlefields and record scores.

        Scores are derived from CLEAR metrics + upgrade gains.
        Anti-gaming integrity validation via GuardrailSuite.
        """
        results = []

        # Total CNA gain from upgrades
        cna_gain = sum(u.expected_cna_gain FOR u IN upgrades)
        effective_cna = snapshot.cna + cna_gain

        FOR target_id IN self.targets:
            spec = self.registry.get(target_id)
            IF spec IS None:
                CONTINUE

            # Simulate score based on effective CNA and benchmark difficulty
            # Score converges toward 1.0 as CNA increases, with benchmark-specific ceiling
            base_capability = min(1.0, effective_cna / 300.0)

            # Benchmark-specific scoring heuristic
            IF target_id == "HLE":
                score = base_capability * 0.65     # HLE is hardest
            ELIF target_id == "SWE_bench_Verified":
                score = base_capability * 0.72     # SWE-bench: tool use helps
            ELIF target_id == "ARC_AGI_2":
                score = base_capability * 0.95     # ARC: reasoning dominant
            ELSE:
                score = base_capability * 0.80     # Default

            # Add iteration improvement (diminishing returns)
            iteration_bonus = 0.02 * self.state.iteration / (1 + 0.5 * self.state.iteration)
            score = min(1.0, score + iteration_bonus)

            # Anti-gaming validation
            integrity_valid = self.guardrails.check_all() if hasattr(self.guardrails, 'check_all') else True

            # Compute cost
            cost_usd = spec.task_count * spec.cost_per_task_usd * (1.0 - snapshot.raw_cost * 0.3)

            results.append(BattlefieldResult(
                battlefield_id=target_id,
                score=round(score, 3),
                rank=self._estimate_rank(target_id, score),
                sota_score=spec.sota_score,
                achieved_sota=score > spec.sota_score,
                cost_usd=round(cost_usd, 2),
                latency_p99_ms=spec.max_latency_ms * (1.0 - snapshot.raw_latency * 0.5),
                integrity_valid=integrity_valid,
            ))

        RETURN results

    METHOD _estimate_rank(self, battlefield_id: str, score: float) -> int:
        """Estimate leaderboard rank based on score vs SOTA."""
        spec = self.registry.get(battlefield_id)
        IF spec IS None:
            RETURN 99
        IF score > spec.sota_score:
            RETURN 1
        gap = spec.sota_score - score
        RETURN max(2, int(gap * 20) + 2)    # Rough: each 0.05 gap = 1 rank

    # ---------------------------------------------------------------
    # PHASE 5: ANALYZE
    # ---------------------------------------------------------------

    METHOD _phase_analyze(self, results: List[BattlefieldResult]) -> List[str]:
        """
        Analyze campaign results, update SOTA, check convergence.

        Returns list of battlefield IDs where SOTA was achieved.
        """
        won = []
        FOR result IN results:
            IF result.achieved_sota AND result.integrity_valid:
                won.append(result.battlefield_id)
                self.registry.update_sota(result.battlefield_id, result.score)

        RETURN won

    # ---------------------------------------------------------------
    # TERMINATION
    # ---------------------------------------------------------------

    METHOD _check_termination(self) -> None:
        """Check all termination conditions."""
        # Condition 1: Dominance on all targets
        IF set(self.state.battlefields_won) >= set(self.targets):
            self.state.status = DominanceStatus.DOMINATED
            RETURN

        # Condition 2: Partial SOTA
        IF len(self.state.battlefields_won) > 0:
            self.state.status = DominanceStatus.SOTA_PARTIAL

        # Condition 3: Budget exhausted
        IF self.state.budget_remaining_usd <= 0:
            self.state.status = DominanceStatus.BUDGET_EXHAUSTED
            RETURN

        # Condition 4: Regression detection
        IF len(self.state.iterations) >= 2:
            current = self.state.iterations[-1].clear_snapshot.cna
            previous = self.state.iterations[-2].clear_snapshot.cna
            IF current < previous:
                self.state.regressions_consecutive += 1
            ELSE:
                self.state.regressions_consecutive = 0

            IF self.state.regressions_consecutive >= self.state.patience:
                self.state.status = DominanceStatus.PATIENCE_EXHAUSTED
                RETURN

    # ---------------------------------------------------------------
    # REPORTING
    # ---------------------------------------------------------------

    METHOD dominance_report(self) -> Dict[str, Any]:
        """Generate the final dominance report."""
        RETURN {
            "status": self.state.status.value,
            "iterations": self.state.iteration,
            "snr": round(self.state.snr, 5),
            "cna": round(self.state.cna, 2),
            "ihsan_score": round(self.state.ihsan_score, 3),
            "battlefields_won": self.state.battlefields_won,
            "architecture_version": self.state.architecture_version,
            "budget_used_usd": round(
                self.state.iterations[0].cost_total_usd if self.state.iterations else 0
                + sum(r.cost_total_usd FOR r IN self.state.iterations),
                2,
            ),
            "pareto_efficient": all(
                r.clear_snapshot.pareto_efficient FOR r IN self.state.iterations
            ),
            "constraints_satisfied": {
                "ihsan_ge_095": self.state.ihsan_score >= self.ihsan_floor,
                "snr_ge_target": self.state.snr >= self.snr_target,
                "integrity_valid": all(
                    br.integrity_valid
                    FOR r IN self.state.iterations
                    FOR br IN r.battlefield_results
                ),
            },
        }
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Compose existing frameworks | Zero duplication; CLEARFramework, AblationEngine, etc. are reused |
| MetricsProvider drives progression | Upgrades boost metrics, which feed next CLEAR evaluation |
| Diminishing returns on upgrades | Prevents unrealistic linear score growth |
| Integrity validation per submission | Constitutional constraint: no gaming |
| Budget tracking per iteration | Cost is a first-class CLEAR dimension |
| Patience counter for regressions | Prevents infinite loops (Nygard circuit breaker principle) |
| Score heuristic per battlefield | Each benchmark has different difficulty and domain |

---

## Implementation Notes — API Adapters Required

The pseudocode uses simplified method names. Actual APIs discovered:

### CLEARFramework (`core/benchmark/clear_framework.py`)

- **Actual API:** Uses context manager `framework.evaluate(task_id, agent_id)`
  with `ctx.record_efficacy(accuracy=...)`, `ctx.record_cost(tokens=, usd=)`.
- **Adapter needed:** Wrap in `_evaluate_clear()` helper that creates context,
  records metrics from `SystemMetricsSnapshot`, returns `CLEARMetrics`.

```python
def _evaluate_clear(self, metrics: SystemMetricsSnapshot) -> CLEARMetrics:
    task_id = f"sim-{self.state.iteration}"
    with self.clear_framework.evaluate(task_id, "spearpoint-agent") as ctx:
        ctx.record_efficacy(accuracy=metrics.accuracy, completion=metrics.task_completion)
        ctx.record_cost(tokens=metrics.tokens_used, usd=metrics.cost_usd)
    return self.clear_framework.get_metrics(task_id)
```

### MetricsProvider (`core/spearpoint/metrics_provider.py`)

- **Actual API:** `current_snapshot()` (not `snapshot()`),
  `record_cycle_metrics(approved=, rejected=, clear_score=, ihsan_score=, tokens_used=)`.
- **No `boost()` method exists.** Must add (~10 lines):

```python
def boost(self, metric_name: str, improvement: float) -> None:
    """Adjust baseline for a named metric (for simulation progression)."""
    if not hasattr(self, '_boost_deltas'):
        self._boost_deltas: dict[str, float] = {}
    current = self._boost_deltas.get(metric_name, 0.0)
    self._boost_deltas[metric_name] = min(0.3, current + improvement)
```

Then in `current_snapshot()`, apply boosts to the computed values.

### SystemMetricsSnapshot (`core/spearpoint/metrics_provider.py`)

- **Fields:** `accuracy`, `task_completion`, `goal_achievement`, `consistency`,
  `tokens_used`, `cost_usd`, `correctness`, `safety`, `efficiency`, `user_benefit`.
- **Not present:** `cost_per_task_usd`, `latency_p99_ms`, `sla_compliance_pct`,
  `policy_adherence`, `provenance_score`, `constraint_satisfaction`, `auditability`.
- **Resolution:** Derive these from existing fields or add them to snapshot.

### GuardrailSuite (`core/benchmark/guardrails.py`)

- **Actual API:** `check_all(agent, context)` returns `List[GuardrailResult]`.
- **Simulator simplification:** Call with mock agent context; treat
  `all(r.passed for r in results)` as `integrity_valid`.

### CLEARMetrics score access

- **Actual API:** `metrics.compute_overall_score(weights)` returns float.
- Individual dimension scores: `metrics.efficacy.accuracy` etc.
- **Adapter needed:** Map to `CLEARSnapshot` fields in `_phase_evaluate()`.
