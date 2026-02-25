"""
ZScorer — Intelligent Routing with Feedback (GEM #4 + v9 spec §Phase 02)
═══════════════════════════════════════════════════════════════════════════════

Wraps the existing MoERouter and adds RecursiveGainTracker feedback so
routing efficiency improves over iterations. The ZScorer answers:

  "Given this task and remaining budget, which expert tier maximises
   score improvement per dollar spent?"

The feedback loop:
  1. score_and_route() → ZScorerResult (wraps MoERouter.route())
  2. Caller executes task, observes actual_cost and score_delta.
  3. record_outcome() feeds (efficiency = score_delta / actual_cost)
     into RecursiveGainTracker.
  4. get_routing_stats() reports whether routing is improving.

Standing on Giants:
  Zoom AI (2025) — Z-scorer federated dispatch
  Noam Shazeer (2017) — Mixture-of-Experts
  Boyd (1995) — OODA tight feedback loops
  Goldratt (1984) — Optimise the constraint, not the flow
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.benchmark.moe_router import ExpertTier, MoERouter, QueryComplexity

logger = logging.getLogger(__name__)


@dataclass
class ZScorerResult:
    """Result of a ZScorer routing decision."""

    expert_tier: ExpertTier
    complexity: QueryComplexity
    confidence: float
    estimated_cost_usd: float
    estimated_latency_ms: float
    reasoning: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class ZScorer:
    """
    Intelligent routing layer wrapping MoERouter with outcome feedback.

    Usage:
        z = ZScorer()
        result = z.score_and_route("Run MMLU evaluation", budget_remaining_usd=10.0)
        # ... execute task ...
        z.record_outcome(result.task_id, actual_cost=0.05, score_delta=0.02)
        stats = z.get_routing_stats()
    """

    def __init__(
        self,
        moe_router: Optional[MoERouter] = None,
        gain_tracker: Any = None,
    ) -> None:
        """
        Args:
            moe_router: MoERouter instance. Defaults to MoERouter() if None.
            gain_tracker: Optional RecursiveGainTracker for feedback.
        """
        self._router = moe_router if moe_router is not None else MoERouter()
        self._gain_tracker = gain_tracker
        self._outcome_log: Dict[str, dict] = {}
        self._total_routed = 0
        self._total_cost_usd = 0.0

    def score_and_route(
        self,
        task_description: str,
        budget_remaining_usd: float = 1.0,
    ) -> ZScorerResult:
        """
        Route a task to the appropriate expert tier.

        Args:
            task_description: Natural-language description of the task.
            budget_remaining_usd: Remaining cost budget for this task.

        Returns:
            ZScorerResult with routing decision and cost/latency estimates.
        """
        decision = self._router.route(
            query=task_description,
            max_cost_usd=budget_remaining_usd,
        )

        tier = decision.selected_tier
        estimated_tokens = decision.estimated_tokens
        estimated_cost = (estimated_tokens / 1_000) * tier.cost_per_1k
        # Tokens per second → ms per task
        estimated_latency = (estimated_tokens / max(tier.tok_per_sec, 1)) * 1_000

        self._total_routed += 1

        result = ZScorerResult(
            expert_tier=tier,
            complexity=decision.complexity,
            confidence=decision.confidence,
            estimated_cost_usd=estimated_cost,
            estimated_latency_ms=estimated_latency,
            reasoning=decision.reasoning,
        )

        self._outcome_log[result.task_id] = {
            "task_description": task_description[:100],
            "tier": tier.key,
            "estimated_cost": estimated_cost,
            "routed_at": time.time(),
        }

        logger.debug(
            "ZScorer routed task %s → tier=%s, est_cost=$%.4f, confidence=%.2f",
            result.task_id,
            tier.key,
            estimated_cost,
            decision.confidence,
        )
        return result

    def record_outcome(
        self,
        task_id: str,
        actual_cost: float,
        score_delta: float,
    ) -> None:
        """
        Record observed outcome to improve future routing decisions.

        Feeds efficiency signal (score_delta / actual_cost) into the
        RecursiveGainTracker to detect routing improvement over iterations.

        Args:
            task_id: task_id from ZScorerResult.task_id.
            actual_cost: Actual cost incurred (USD).
            score_delta: Benchmark score change from executing this task.
        """
        if task_id not in self._outcome_log:
            logger.warning("ZScorer.record_outcome: unknown task_id %s", task_id)
            return

        log = self._outcome_log[task_id]
        log["actual_cost"] = actual_cost
        log["score_delta"] = score_delta
        self._total_cost_usd += actual_cost

        if self._gain_tracker is not None:
            # Efficiency = improvement per dollar (clamped to ≥ 0).
            efficiency = max(0.0, score_delta) / max(actual_cost, 1e-9)
            self._gain_tracker.record(score=efficiency, cost_usd=actual_cost)

    def get_routing_stats(self) -> dict:
        """Return aggregated routing statistics."""
        completed = [v for v in self._outcome_log.values() if "actual_cost" in v]
        avg_cost = (
            sum(v["actual_cost"] for v in completed) / len(completed)
            if completed
            else 0.0
        )
        avg_delta = (
            sum(v.get("score_delta", 0.0) for v in completed) / len(completed)
            if completed
            else 0.0
        )
        return {
            "total_routed": self._total_routed,
            "total_cost_usd": self._total_cost_usd,
            "completed_tasks": len(completed),
            "avg_cost_per_task": avg_cost,
            "avg_score_delta": avg_delta,
        }
