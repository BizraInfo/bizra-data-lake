"""Collective Synthesizer - Trust-weighted decision synthesis with SNR aggregation."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .social_integration import SociallyAwareBridge
from .team_planner import AgentRole

logger = logging.getLogger(__name__)


class ConflictStrategy(Enum):
    TRUST_WEIGHTED = auto()
    HIERARCHICAL = auto()
    CONSENSUS_REQUIRED = auto()
    SNR_MAXIMIZED = auto()

@dataclass
class AgentOutput:
    agent_id: str
    role: AgentRole
    content: Any
    confidence: float = 0.9
    snr_score: float = 0.85
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ResolvedOutput:
    content: Any
    winning_agents: List[str] = field(default_factory=list)
    dissenting_agents: List[str] = field(default_factory=list)
    resolution_strategy: ConflictStrategy = ConflictStrategy.TRUST_WEIGHTED
    resolution_confidence: float = 0.0

@dataclass
class SynthesizedResult:
    content: Any
    consensus_score: float
    synthesis_score: float
    snr_weighted_confidence: float
    contributing_agents: List[str] = field(default_factory=list)
    conflicts_resolved: int = 0
    dissent_summary: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CollectiveSynthesizer:
    """Trust-weighted synthesis with conflict resolution and SNR aggregation."""
    ROLE_HIERARCHY = {
        AgentRole.MASTER_REASONER: 10, AgentRole.SECURITY_GUARDIAN: 9,
        AgentRole.ETHICS_GUARDIAN: 9, AgentRole.ETHICS_VALIDATOR: 9,
        AgentRole.FUSION: 8, AgentRole.DATA_ANALYZER: 7, AgentRole.EXECUTION_PLANNER: 7,
        AgentRole.MEMORY_ARCHITECT: 6, AgentRole.PERFORMANCE_MONITOR: 6,
        AgentRole.CONSISTENCY_CHECKER: 6, AgentRole.COMMUNICATOR: 5,
        AgentRole.RESOURCE_OPTIMIZER: 5,
    }

    def __init__(self, social_bridge: Optional[SociallyAwareBridge] = None,
                 min_consensus: float = 0.6, snr_threshold: float = 0.85):
        self.social_bridge = social_bridge
        self.min_consensus = min_consensus
        self.snr_threshold = snr_threshold
        self._synthesis_history: List[float] = []

    def _get_trust(self, agent_id: str) -> float:
        return self.social_bridge.get_trust(agent_id) if self.social_bridge else 0.5

    def synthesize(self, agent_outputs: List[AgentOutput]) -> SynthesizedResult:
        """Synthesize agent outputs into unified decision using trust-weighted voting."""
        if not agent_outputs:
            return SynthesizedResult(None, 0.0, 0.0, 0.0)
        if len(agent_outputs) == 1:
            o = agent_outputs[0]
            return SynthesizedResult(o.content, 1.0, 1.0, o.confidence * o.snr_score, [o.agent_id])

        # Group outputs by content with weighted votes
        votes, agents, original = {}, {}, {}
        for out in agent_outputs:
            key = str(out.content)
            weight = self._get_trust(out.agent_id) * out.confidence * out.snr_score
            votes[key] = votes.get(key, 0.0) + weight
            agents.setdefault(key, []).append(out.agent_id)
            original[key] = out.content
        total = sum(votes.values())
        winner = max(votes, key=votes.get)

        dissent = [f"{len(a)} chose '{k[:40]}' ({votes[k]/total*100:.1f}%)"
                   for k, a in agents.items() if k != winner] if len(votes) > 1 else []

        agreeing = [o for o in agent_outputs if str(o.content) == winner]
        syn_score = (len(agreeing)/len(agent_outputs)) * (sum(o.snr_score for o in agreeing)/len(agreeing))
        snr_conf = sum(o.confidence * o.snr_score for o in agreeing) / sum(o.snr_score for o in agreeing)

        self._synthesis_history.append(syn_score)
        return SynthesizedResult(original[winner], votes[winner]/total, syn_score, min(0.99, snr_conf),
                                 agents[winner], len(votes)-1, dissent)

    def calculate_consensus_score(self, agent_outputs: List[AgentOutput]) -> float:
        """Calculate how well agents agree (0-1)."""
        if len(agent_outputs) < 2: return 1.0
        weights = {}
        for o in agent_outputs:
            k = str(o.content)
            weights[k] = weights.get(k, 0.0) + self._get_trust(o.agent_id) * o.confidence
        total = sum(weights.values())
        return max(weights.values()) / total if total > 0 else 0.0

    def resolve_conflicts(self, outputs: List[AgentOutput],
                         strategy: ConflictStrategy = ConflictStrategy.TRUST_WEIGHTED) -> ResolvedOutput:
        """Resolve conflicts: TRUST_WEIGHTED, HIERARCHICAL, CONSENSUS_REQUIRED, or SNR_MAXIMIZED."""
        if not outputs: return ResolvedOutput(None, resolution_confidence=0.0)
        resolvers = {ConflictStrategy.HIERARCHICAL: self._resolve_hierarchical,
                    ConflictStrategy.CONSENSUS_REQUIRED: self._resolve_consensus,
                    ConflictStrategy.SNR_MAXIMIZED: self._resolve_snr}
        return resolvers.get(strategy, self._resolve_trust)(outputs)

    def _resolve_trust(self, outputs: List[AgentOutput]) -> ResolvedOutput:
        scores, agents, original = {}, {}, {}
        for o in outputs:
            k = str(o.content)
            scores[k] = scores.get(k, 0.0) + self._get_trust(o.agent_id) * o.confidence
            agents.setdefault(k, []).append(o.agent_id)
            original[k] = o.content
        winner, total = max(scores, key=scores.get), sum(scores.values())
        dissent = [a for k, al in agents.items() if k != winner for a in al]
        return ResolvedOutput(original[winner], agents[winner], dissent,
                             ConflictStrategy.TRUST_WEIGHTED, scores[winner]/total if total else 0)

    def _resolve_hierarchical(self, outputs: List[AgentOutput]) -> ResolvedOutput:
        s = sorted(outputs, key=lambda o: self.ROLE_HIERARCHY.get(o.role, 0), reverse=True)
        w, dissent = s[0], [o.agent_id for o in s[1:] if o.content != s[0].content]
        return ResolvedOutput(w.content, [w.agent_id], dissent, ConflictStrategy.HIERARCHICAL, w.confidence)

    def _resolve_consensus(self, outputs: List[AgentOutput]) -> ResolvedOutput:
        counts = {}
        for o in outputs: counts[str(o.content)] = counts.get(str(o.content), 0) + 1
        total = len(outputs)
        for k, c in counts.items():
            if c / total >= self.min_consensus:
                content = next(o.content for o in outputs if str(o.content) == k)
                return ResolvedOutput(content, [o.agent_id for o in outputs if str(o.content) == k],
                                     [o.agent_id for o in outputs if str(o.content) != k],
                                     ConflictStrategy.CONSENSUS_REQUIRED, c/total)
        logger.warning("Consensus not reached, falling back to trust-weighted")
        return self._resolve_trust(outputs)

    def _resolve_snr(self, outputs: List[AgentOutput]) -> ResolvedOutput:
        s = sorted(outputs, key=lambda o: o.snr_score, reverse=True)
        w, dissent = s[0], [o.agent_id for o in s[1:] if o.content != s[0].content]
        return ResolvedOutput(w.content, [w.agent_id], dissent, ConflictStrategy.SNR_MAXIMIZED, w.snr_score)

    def average_synthesis_score(self) -> float:
        """Average synthesis score across all syntheses."""
        return sum(self._synthesis_history) / len(self._synthesis_history) if self._synthesis_history else 0.0


__all__ = ["AgentOutput", "CollectiveSynthesizer", "ConflictStrategy", "ResolvedOutput", "SynthesizedResult"]
