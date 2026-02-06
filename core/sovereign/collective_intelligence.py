"""
Collective Intelligence — Team Synergy Engine
==============================================
Synthesizes outputs from multiple agents into coherent collective
decisions that exceed individual agent capabilities.

"The whole is greater than the sum of its parts." — Aristotle

Standing on Giants: Wisdom of Crowds + Ensemble Methods + Swarm Intelligence
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import uuid

from .team_planner import AgentRole

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating agent contributions."""
    WEIGHTED_AVERAGE = auto()  # Weight by confidence
    MAJORITY_VOTE = auto()     # Democratic consensus
    BEST_OF = auto()           # Highest confidence wins
    SYNTHESIS = auto()         # LLM-based fusion
    HIERARCHICAL = auto()      # Defer to role hierarchy


@dataclass
class AgentContribution:
    """A contribution from a single agent to collective decision."""
    agent_role: AgentRole = AgentRole.MASTER_REASONER
    content: Any = None
    confidence: float = 0.9
    reasoning: str = ""
    alternatives: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CollectiveDecision:
    """The synthesized collective decision."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    contributions: List[AgentContribution] = field(default_factory=list)
    method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    result: Any = None
    confidence: float = 0.0
    synergy_score: float = 0.0  # How much better than individual
    dissent: List[str] = field(default_factory=list)  # Minority views
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CollectiveIntelligence:
    """
    Synthesizes team outputs into collective intelligence.

    Capabilities:
    - Aggregate diverse agent perspectives
    - Handle disagreement constructively
    - Achieve emergent insights
    - Track synergy metrics
    """

    # Role hierarchy for hierarchical aggregation
    ROLE_HIERARCHY = {
        AgentRole.MASTER_REASONER: 10,
        AgentRole.SECURITY_GUARDIAN: 9,
        AgentRole.ETHICS_GUARDIAN: 9,
        AgentRole.ETHICS_VALIDATOR: 9,
        AgentRole.DATA_ANALYZER: 7,
        AgentRole.EXECUTION_PLANNER: 7,
        AgentRole.FUSION: 8,
        AgentRole.MEMORY_ARCHITECT: 6,
        AgentRole.COMMUNICATOR: 5,
        AgentRole.PERFORMANCE_MONITOR: 6,
        AgentRole.CONSISTENCY_CHECKER: 6,
        AgentRole.RESOURCE_OPTIMIZER: 5,
    }

    def __init__(
        self,
        default_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        min_contributions: int = 2,
        synergy_bonus: float = 0.1,
        max_history: int = 1000,
    ):
        self.default_method = default_method
        self.min_contributions = min_contributions
        self.synergy_bonus = synergy_bonus

        self._decisions: Dict[str, CollectiveDecision] = {}
        # PERF FIX: Use deque with maxlen for O(1) bounded storage
        self._synergy_history: Deque[float] = deque(maxlen=max_history)

    async def collect(
        self,
        question: str,
        contributions: List[AgentContribution],
        method: Optional[AggregationMethod] = None,
    ) -> CollectiveDecision:
        """Collect and synthesize agent contributions."""
        method = method or self.default_method

        if len(contributions) < self.min_contributions:
            logger.warning(f"Only {len(contributions)} contributions (min: {self.min_contributions})")

        decision = CollectiveDecision(
            question=question,
            contributions=contributions,
            method=method,
        )

        # Apply aggregation method
        if method == AggregationMethod.WEIGHTED_AVERAGE:
            decision.result, decision.confidence = self._weighted_average(contributions)
        elif method == AggregationMethod.MAJORITY_VOTE:
            decision.result, decision.confidence = self._majority_vote(contributions)
        elif method == AggregationMethod.BEST_OF:
            decision.result, decision.confidence = self._best_of(contributions)
        elif method == AggregationMethod.HIERARCHICAL:
            decision.result, decision.confidence = self._hierarchical(contributions)
        else:  # SYNTHESIS - requires external LLM
            decision.result, decision.confidence = self._weighted_average(contributions)

        # Calculate synergy score
        individual_max = max((c.confidence for c in contributions), default=0)
        decision.synergy_score = (decision.confidence - individual_max) / max(individual_max, 0.01)

        # Record dissenting views
        if decision.result is not None:
            for contrib in contributions:
                if contrib.content != decision.result and contrib.confidence > 0.5:
                    decision.dissent.append(
                        f"{contrib.agent_role.value}: {contrib.content} (conf: {contrib.confidence:.2f})"
                    )

        self._decisions[decision.id] = decision
        self._synergy_history.append(decision.synergy_score)

        logger.info(
            f"Collective decision: {decision.id} "
            f"(method: {method.name}, confidence: {decision.confidence:.2f}, "
            f"synergy: {decision.synergy_score:+.2f})"
        )
        return decision

    def _weighted_average(
        self,
        contributions: List[AgentContribution],
    ) -> Tuple[Any, float]:
        """Aggregate using confidence-weighted average."""
        if not contributions:
            return None, 0.0

        # For numeric values
        if all(isinstance(c.content, (int, float)) for c in contributions):
            total_weight = sum(c.confidence for c in contributions)
            if total_weight == 0:
                return None, 0.0

            weighted_sum = sum(c.content * c.confidence for c in contributions)
            result = weighted_sum / total_weight

            # Confidence increases with agreement
            variance = sum(
                c.confidence * (c.content - result) ** 2
                for c in contributions
            ) / total_weight
            agreement_factor = 1.0 / (1.0 + variance)

            confidence = min(0.99, (total_weight / len(contributions)) * agreement_factor + self.synergy_bonus)
            return result, confidence

        # For categorical values - defer to majority
        return self._majority_vote(contributions)

    def _majority_vote(
        self,
        contributions: List[AgentContribution],
    ) -> Tuple[Any, float]:
        """Aggregate using majority vote."""
        if not contributions:
            return None, 0.0

        # Count votes weighted by confidence
        votes: Dict[Any, float] = {}
        for c in contributions:
            key = str(c.content)  # Hashable key
            votes[key] = votes.get(key, 0) + c.confidence

        # Find winner
        winner_key = max(votes, key=votes.get)
        winner_weight = votes[winner_key]
        total_weight = sum(votes.values())

        # Get original value
        winner_value = next(
            c.content for c in contributions if str(c.content) == winner_key
        )

        confidence = min(0.99, (winner_weight / total_weight) + self.synergy_bonus)
        return winner_value, confidence

    def _best_of(
        self,
        contributions: List[AgentContribution],
    ) -> Tuple[Any, float]:
        """Select highest confidence contribution."""
        if not contributions:
            return None, 0.0

        best = max(contributions, key=lambda c: c.confidence)
        return best.content, best.confidence

    def _hierarchical(
        self,
        contributions: List[AgentContribution],
    ) -> Tuple[Any, float]:
        """Defer to highest-ranking role in hierarchy."""
        if not contributions:
            return None, 0.0

        # Sort by hierarchy rank
        sorted_contribs = sorted(
            contributions,
            key=lambda c: self.ROLE_HIERARCHY.get(c.agent_role, 0),
            reverse=True,
        )

        top = sorted_contribs[0]
        # Adjust confidence based on subordinate agreement
        agreement = sum(
            1 for c in sorted_contribs[1:]
            if c.content == top.content
        ) / max(len(sorted_contribs) - 1, 1)

        confidence = min(0.99, top.confidence * (0.8 + 0.2 * agreement) + self.synergy_bonus)
        return top.content, confidence

    def average_synergy(self) -> float:
        """Get average synergy score across all decisions."""
        if not self._synergy_history:
            return 0.0
        return sum(self._synergy_history) / len(self._synergy_history)

    def stats(self) -> Dict[str, Any]:
        """Get collective intelligence statistics."""
        return {
            "total_decisions": len(self._decisions),
            "average_synergy": round(self.average_synergy(), 3),
            "synergy_history_len": len(self._synergy_history),
            "positive_synergy_rate": (
                sum(1 for s in self._synergy_history if s > 0) / max(len(self._synergy_history), 1)
            ),
            "default_method": self.default_method.name,
        }


__all__ = [
    "AgentContribution",
    "AggregationMethod",
    "CollectiveDecision",
    "CollectiveIntelligence",
]
