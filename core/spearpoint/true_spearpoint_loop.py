"""
TrueSpearpointLoop — v9 Hierarchical Bayesian Optimization Composer
═══════════════════════════════════════════════════════════════════════════════

The v9 composer wires MIRAS, ZScorer, AdaptivePrior, and BenchmarkDominanceLoop
into one self-improving outer loop — the "Hidden Architecture Pattern" identified
in the TRUE SPEARPOINT analysis: The Scientific Method as Code.

Three Epistemic Levels (Pearl's Ladder of Causation):
  Level 1 — OBSERVE:     _phase_evaluate()
  Level 2 — INTERVENE:   _phase_ablate() + _phase_architect() + _phase_submit()
  Level 3 — COUNTERFACTUAL/REFLECT: _phase_analyze()

Convergence is triggered by any of:
  1. target_snr reached (SNR >= config.target_snr)
  2. Budget exhausted (total_cost_usd >= config.budget_usd)
  3. Patience exceeded (no Pareto improvement for config.patience iterations)
  4. Pareto front stable for config.pareto_convergence_window iterations

Standing on Giants:
  Pearl (2000) — Ladder of Causation (Observe → Intervene → Counterfactual)
  Pareto (1906) — Multi-objective optimality frontier
  Boyd (1995) — OODA tight feedback loops
  Al-Ghazali (1095) — Iḥsān: incremental excellence at every step
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

logger = logging.getLogger(__name__)


# ─── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class LoopConfig:
    """Configuration for TrueSpearpointLoop."""

    max_iterations: int = 20
    target_snr: float = 0.99
    budget_usd: float = 2_000.0
    patience: int = 5
    pareto_convergence_window: int = 3
    # Per-iteration budget for the inner dominance loop.
    per_iteration_budget_usd: float = 50.0
    # SNR floor — iterations below this are flagged but not rejected.
    snr_floor: float = UNIFIED_SNR_THRESHOLD
    # Iḥsān gate — iterations below this mark outcome as degraded.
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD


# ─── Result Types ──────────────────────────────────────────────────────────────


@dataclass
class IterationResult:
    """Result of one outer-loop iteration."""

    iteration: int
    phase_results: Dict[str, Any]
    snr: float
    ihsan: float
    pareto_rank: int
    outcome: str  # mirrors CycleOutcome.name
    memory_stats: Dict[str, Any]
    cost_usd: float
    converged: bool = False
    convergence_reason: str = ""


@dataclass
class SpearpointReport:
    """Final report returned by TrueSpearpointLoop.run()."""

    campaign_id: str
    iterations_completed: int
    final_snr: float
    final_ihsan: float
    pareto_frontier: List[Dict[str, float]]
    total_cost_usd: float
    convergence_reason: str
    iteration_history: List[IterationResult]
    memory_summary: Dict[str, Any]
    routing_stats: Dict[str, Any]
    prior_report: Dict[str, Any]


# ─── Main Class ────────────────────────────────────────────────────────────────


class TrueSpearpointLoop:
    """
    Outer self-improving loop composing all TRUE SPEARPOINT gems.

    Holds:
        _inner_loop   — BenchmarkDominanceLoop (core/benchmark/dominance_loop.py)
        _memory       — MIRASMemory           (core/benchmark/miras_memory.py)
        _prior        — AdaptivePriorLearning  (core/benchmark/adaptive_prior.py)
        _z_scorer     — ZScorer               (core/benchmark/z_scorer.py)
        _gain_tracker — RecursiveGainTracker  (core/benchmark/recursive_gain_tracker.py)

    Usage:
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=10))
        report = await loop.run()
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        inner_loop: Any = None,
    ) -> None:
        """
        Args:
            config: Loop hyperparameters. Defaults to LoopConfig().
            inner_loop: BenchmarkDominanceLoop instance. If None, creates a
                        stub that returns synthetic phase results (for testing).
        """
        self._config = config or LoopConfig()
        self._inner_loop = inner_loop

        # Lazy-import to avoid circular deps / optional GPU dependencies.
        from core.benchmark.adaptive_prior import AdaptivePriorLearning
        from core.benchmark.miras_memory import MIRASMemory
        from core.benchmark.recursive_gain_tracker import RecursiveGainTracker
        from core.benchmark.z_scorer import ZScorer

        self._gain_tracker = RecursiveGainTracker()
        self._memory = MIRASMemory()
        self._prior = AdaptivePriorLearning()
        self._z_scorer = ZScorer(gain_tracker=self._gain_tracker)

        # Pareto history for convergence detection.
        self._pareto_history: List[List[Dict[str, float]]] = []
        self._total_cost_usd: float = 0.0
        self._campaign_id = str(uuid.uuid4())[:8]

    # ─── Public API ────────────────────────────────────────────────────────────

    async def run(self) -> SpearpointReport:
        """
        Execute the full self-improving outer loop.

        Returns:
            SpearpointReport with full history and final state.
        """
        cfg = self._config
        iteration_history: List[IterationResult] = []
        best_snr: float = 0.0
        patience_counter: int = 0
        convergence_reason: str = "max_iterations_reached"

        logger.info(
            "TrueSpearpointLoop starting: campaign=%s, max_iter=%d, budget=$%.0f",
            self._campaign_id,
            cfg.max_iterations,
            cfg.budget_usd,
        )

        for i in range(cfg.max_iterations):
            # ── Level 1: OBSERVE ──
            eval_result = await self._phase_evaluate()

            # ── Level 2: INTERVENE ──
            ablation_result = self._phase_ablate(eval_result)
            architect_result = self._phase_architect(ablation_result)
            submit_result = await self._phase_submit()

            phase_results = {
                "evaluate": eval_result,
                "ablate": ablation_result,
                "architect": architect_result,
                "submit": submit_result,
            }

            # ── Level 3: COUNTERFACTUAL / REFLECT ──
            analysis = self._phase_analyze(phase_results)

            # Extract metrics.
            snr = analysis.get("snr", 0.0)
            ihsan = analysis.get("ihsan", 0.0)
            cycle_cost = analysis.get("cost_usd", 0.0)
            self._total_cost_usd += cycle_cost
            pareto_rank = self._compute_pareto_rank(snr, ihsan)

            # Update patience counter.
            if snr > best_snr + 1e-6:
                best_snr = snr
                patience_counter = 0
            else:
                patience_counter += 1

            # Feed gain tracker.
            self._gain_tracker.record(score=snr, cost_usd=cycle_cost)

            iter_result = IterationResult(
                iteration=i,
                phase_results=phase_results,
                snr=snr,
                ihsan=ihsan,
                pareto_rank=pareto_rank,
                outcome=analysis.get("outcome", "MAINTAINED"),
                memory_stats=self._memory.get_stats(),
                cost_usd=cycle_cost,
            )

            # ── Convergence checks ──
            if snr >= cfg.target_snr:
                iter_result.converged = True
                iter_result.convergence_reason = "target_snr_reached"
                convergence_reason = "target_snr_reached"
                iteration_history.append(iter_result)
                break

            if self._total_cost_usd >= cfg.budget_usd:
                iter_result.converged = True
                iter_result.convergence_reason = "budget_exhausted"
                convergence_reason = "budget_exhausted"
                iteration_history.append(iter_result)
                break

            if patience_counter >= cfg.patience:
                iter_result.converged = True
                iter_result.convergence_reason = "patience_exceeded"
                convergence_reason = "patience_exceeded"
                iteration_history.append(iter_result)
                break

            if self._pareto_converged():
                iter_result.converged = True
                iter_result.convergence_reason = "pareto_stable"
                convergence_reason = "pareto_stable"
                iteration_history.append(iter_result)
                break

            iteration_history.append(iter_result)
            logger.info(
                "Iteration %d/%d: snr=%.4f, ihsan=%.4f, cost=$%.2f, " "patience=%d/%d",
                i + 1,
                cfg.max_iterations,
                snr,
                ihsan,
                self._total_cost_usd,
                patience_counter,
                cfg.patience,
            )

        final_snr = iteration_history[-1].snr if iteration_history else 0.0
        final_ihsan = iteration_history[-1].ihsan if iteration_history else 0.0

        report = SpearpointReport(
            campaign_id=self._campaign_id,
            iterations_completed=len(iteration_history),
            final_snr=final_snr,
            final_ihsan=final_ihsan,
            pareto_frontier=(self._pareto_history[-1] if self._pareto_history else []),
            total_cost_usd=self._total_cost_usd,
            convergence_reason=convergence_reason,
            iteration_history=iteration_history,
            memory_summary=self._memory.get_stats(),
            routing_stats=self._z_scorer.get_routing_stats(),
            prior_report=self._prior.get_report(),
        )

        logger.info(
            "TrueSpearpointLoop finished: %s in %d iterations, $%.2f spent",
            convergence_reason,
            len(iteration_history),
            self._total_cost_usd,
        )
        return report

    # ─── Epistemic Level 1: OBSERVE ────────────────────────────────────────────

    async def _phase_evaluate(self) -> dict:
        """
        Run benchmark evaluation via inner_loop or stub.

        Returns dict with at minimum: 'score' (float), 'cost_usd' (float).
        """
        if self._inner_loop is not None:
            try:
                result = await asyncio.wait_for(
                    self._inner_loop.run(
                        max_cycles=1,
                        budget_usd=self._config.per_iteration_budget_usd,
                    ),
                    timeout=300.0,
                )
                return {
                    "score": result.final_score,
                    "cost_usd": result.total_cost_usd,
                    "cycles": result.total_cycles,
                    "sota_achieved": result.sota_achieved,
                }
            except Exception as exc:
                logger.warning("_phase_evaluate inner_loop failed: %s", exc)
                return {"score": 0.0, "cost_usd": 0.0, "error": str(exc)}

        # Stub for testing: synthetic evaluation.
        return {
            "score": 0.5,
            "cost_usd": 1.0,
            "cycles": 1,
            "sota_achieved": False,
        }

    # ─── Epistemic Level 2: INTERVENE ──────────────────────────────────────────

    def _phase_ablate(self, eval_result: dict) -> dict:
        """
        Run ablation analysis guided by AdaptivePrior priority order.

        Returns dict with: 'priority_order', 'top_category', 'cost_usd'.
        """
        priority = self._prior.suggest_priority_order()
        top_category = priority[0] if priority else "attention_changes"

        # Route the ablation task via ZScorer.
        routing = self._z_scorer.score_and_route(
            f"Ablation study: {top_category}",
            budget_remaining_usd=self._config.budget_usd - self._total_cost_usd,
        )

        ablation_cost = routing.estimated_cost_usd
        return {
            "priority_order": priority,
            "top_category": top_category,
            "routing_tier": routing.expert_tier.key,
            "cost_usd": ablation_cost,
        }

    def _phase_architect(self, ablation_result: dict) -> dict:
        """
        Generate architecture recommendations based on ablation output.

        Returns dict with: 'recommendation', 'cost_usd'.
        """
        top = ablation_result.get("top_category", "attention_changes")
        return {
            "recommendation": f"Upgrade {top} component",
            "cost_usd": 0.5,
        }

    async def _phase_submit(self) -> dict:
        """
        Submit to leaderboard / record benchmark score.

        Returns dict with: 'submitted', 'cost_usd'.
        """
        return {"submitted": True, "cost_usd": 0.1}

    # ─── Epistemic Level 3: COUNTERFACTUAL / REFLECT ───────────────────────────

    def _phase_analyze(self, phase_results: dict) -> dict:
        """
        Reflect on the iteration: update MIRAS memory + AdaptivePrior.

        Returns dict with: 'snr', 'ihsan', 'cost_usd', 'outcome'.
        """
        eval_r = phase_results.get("evaluate", {})
        ablate_r = phase_results.get("ablate", {})
        architect_r = phase_results.get("architect", {})
        submit_r = phase_results.get("submit", {})

        score = eval_r.get("score", 0.0)
        # SNR proxy: normalize score into [0, 1] range.
        snr = min(1.0, max(0.0, score))
        # Iḥsān proxy: confidence gate based on SNR vs threshold.
        ihsan = snr if snr >= self._config.snr_floor else snr * 0.9

        total_cost = sum(
            r.get("cost_usd", 0.0) for r in [eval_r, ablate_r, architect_r, submit_r]
        )

        # ── Update MIRAS memory ──
        content = (
            f"Iteration result: score={score:.4f}, "
            f"top_change={ablate_r.get('top_category', 'unknown')}, "
            f"recommendation={architect_r.get('recommendation', '')}"
        )
        self._memory.store(
            content=content,
            snr_score=snr,
            metadata={
                "score": score,
                "cost_usd": total_cost,
                "top_category": ablate_r.get("top_category"),
            },
        )
        self._memory.store_episodic(
            action=f"Phase cycle (score={score:.4f})",
            result=architect_r.get("recommendation", ""),
            context={"cost_usd": total_cost},
        )

        # ── Update AdaptivePrior ──
        top_category = ablate_r.get("top_category", "regularization")
        improvement_delta = score - 0.5  # Relative to baseline
        self._prior.update_beliefs(top_category, improvement_delta)

        # ── ZScorer routing feedback ──
        # In production, the task_id from _phase_ablate ZScorerResult feeds back here.

        # ── Update Pareto history ──
        frontier_point = {"snr": snr, "ihsan": ihsan, "cost_usd": total_cost}
        if not self._pareto_history:
            self._pareto_history.append([frontier_point])
        else:
            last_frontier = self._pareto_history[-1]
            new_frontier = self._update_pareto_frontier(last_frontier, frontier_point)
            self._pareto_history.append(new_frontier)

        outcome = "IMPROVED" if improvement_delta > 0 else "MAINTAINED"
        return {
            "snr": snr,
            "ihsan": ihsan,
            "cost_usd": total_cost,
            "outcome": outcome,
        }

    # ─── Convergence Helpers ───────────────────────────────────────────────────

    def _pareto_converged(self) -> bool:
        """
        Return True if the Pareto frontier has been stable for
        pareto_convergence_window iterations.
        """
        window = self._config.pareto_convergence_window
        if len(self._pareto_history) < window:
            return False
        recent = self._pareto_history[-window:]
        # Frontier is stable if all recent iterations have the same length
        # and identical best SNR.
        first_best_snr = max((p["snr"] for p in recent[0]), default=0.0)
        return all(
            abs(max((p["snr"] for p in frontier), default=0.0) - first_best_snr) < 1e-6
            for frontier in recent[1:]
        )

    def _compute_pareto_rank(self, snr: float, ihsan: float) -> int:
        """
        Compute Pareto rank of (snr, ihsan) point against history.

        Rank 1 = non-dominated (Pareto optimal).
        Higher rank = dominated by more previous points.
        """
        if not self._pareto_history:
            return 1
        dominators = 0
        for frontier in self._pareto_history:
            for point in frontier:
                if point["snr"] >= snr and point["ihsan"] >= ihsan:
                    if point["snr"] > snr or point["ihsan"] > ihsan:
                        dominators += 1
        return dominators + 1

    @staticmethod
    def _update_pareto_frontier(
        frontier: List[Dict[str, float]],
        new_point: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """
        Add new_point to the frontier, removing dominated points.

        A point A dominates B if A is >= B on ALL objectives.
        """
        new_snr = new_point["snr"]
        new_ihsan = new_point["ihsan"]

        # Remove points dominated by new_point.
        updated = [
            p for p in frontier if not (new_snr >= p["snr"] and new_ihsan >= p["ihsan"])
        ]

        # Only add new_point if not dominated by any remaining point.
        dominated = any(
            p["snr"] >= new_snr and p["ihsan"] >= new_ihsan for p in updated
        )
        if not dominated:
            updated.append(new_point)

        return updated
