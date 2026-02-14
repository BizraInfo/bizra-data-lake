"""
Social Integration — SocialGraph ↔ PAT/SAT Bridge
═══════════════════════════════════════════════════════════════════════════════

Enhances the Dual-Agentic Bridge with trust-based agent routing using
the Apex SocialGraph for relationship intelligence.

Standing on the Shoulders of Giants:
- Granovetter (1973): Strength of Weak Ties — diverse connections for task routing
- Page & Brin (1998): PageRank for trust propagation
- Lamport (1982): Byzantine fault tolerance principles

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                  SociallyAwareBridge                          │
    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
    │  │ DualAgentic    │  │ SocialGraph    │  │ Trust-Based    │  │
    │  │ Bridge (base)  │  │ (Apex)         │  │ Routing        │  │
    │  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  │
    │           └───────────────────┴───────────────────┘          │
    └──────────────────────────────────────────────────────────────┘

Created: 2026-02-04 | BIZRA Apex Integration v1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.apex import (
    Relationship,
    RelationshipType,
    SocialGraph,
)
from core.sovereign.dual_agentic_bridge import (
    DualAgenticBridge,
)
from core.sovereign.team_planner import AgentRole

logger = logging.getLogger(__name__)


class NoCapableAgentError(Exception):
    """Raised when no agent with required capabilities is available."""

    def __init__(self, required_capabilities: set[str]):
        self.required_capabilities = required_capabilities
        super().__init__(f"No agent capable of: {required_capabilities}")


@dataclass
class ScoredAgent:
    """An agent with trust and capability scores."""

    agent_id: str
    role: AgentRole
    capability_score: float
    trust_score: float
    combined_score: float
    capabilities: set[str] = field(default_factory=set)


@dataclass
class CollaborationMatch:
    """A matched pair of agents for collaboration."""

    agent_a: str
    agent_b: str
    synergy_score: float
    recommended_task_types: set[str] = field(default_factory=set)


class SociallyAwareBridge(DualAgenticBridge):
    """
    Enhanced bridge that uses social trust for agent routing.

    Combines Byzantine consensus validation with trust-based selection
    to optimize both safety and effectiveness.

    Key Features:
    - Trust-weighted agent selection
    - Collaboration pair discovery
    - Reputation updates after task completion
    - Trust propagation via PageRank

    Usage:
        bridge = SociallyAwareBridge(node_id="node-0")
        selected = bridge.select_agent_for_task(task)
        partners = bridge.find_collaboration_partners(task)
        await bridge.report_task_outcome(task, agent, success=True)
    """

    # Agent capabilities by role
    ROLE_CAPABILITIES: dict[AgentRole, set[str]] = {
        AgentRole.MASTER_REASONER: {"reasoning", "planning", "strategy"},
        AgentRole.DATA_ANALYZER: {"analysis", "data", "patterns"},
        AgentRole.EXECUTION_PLANNER: {"execution", "workflow", "orchestration"},
        AgentRole.ETHICS_GUARDIAN: {"ethics", "validation", "compliance"},
        AgentRole.COMMUNICATOR: {"communication", "output", "formatting"},
        AgentRole.MEMORY_ARCHITECT: {"memory", "knowledge", "context"},
        AgentRole.SECURITY_GUARDIAN: {"security", "validation", "protection"},
        AgentRole.ETHICS_VALIDATOR: {"ethics", "ihsan", "constitutional"},
        AgentRole.PERFORMANCE_MONITOR: {"performance", "metrics", "optimization"},
        AgentRole.CONSISTENCY_CHECKER: {"consistency", "state", "integrity"},
        AgentRole.RESOURCE_OPTIMIZER: {"resources", "allocation", "efficiency"},
    }

    def __init__(
        self,
        node_id: str,
        ihsan_threshold: float = 0.95,
        trust_weight: float = 0.3,
        min_trust_threshold: float = 0.1,
    ):
        """
        Initialize the socially-aware bridge.

        Args:
            node_id: Unique identifier for this node
            ihsan_threshold: Constitutional constraint threshold
            trust_weight: Weight for trust in selection (0-1)
            min_trust_threshold: Minimum trust to consider agent
        """
        super().__init__(ihsan_threshold=ihsan_threshold)

        self.node_id = node_id
        self.trust_weight = trust_weight
        self.min_trust_threshold = min_trust_threshold

        # Initialize social graph
        self.social_graph = SocialGraph(agent_id=node_id)

        # Register PAT agents as peers
        self._register_pat_agents_as_peers()

        logger.info(f"SociallyAwareBridge initialized: {node_id}")

    def _register_pat_agents_as_peers(self) -> None:
        """Register all PAT agents in social graph with initial trust."""
        pat_roles = [
            AgentRole.MASTER_REASONER,
            AgentRole.DATA_ANALYZER,
            AgentRole.EXECUTION_PLANNER,
            AgentRole.ETHICS_GUARDIAN,
            AgentRole.COMMUNICATOR,
            AgentRole.MEMORY_ARCHITECT,
        ]

        for role in pat_roles:
            agent_id = f"pat:{role.value}"
            rel = Relationship(
                agent_id=self.node_id,
                peer_id=agent_id,
                relationship_type=RelationshipType.COLLABORATOR,
                trust_score=0.5,  # Start neutral
                reliability_score=0.5,
            )
            self.social_graph._relationships[agent_id] = rel
            logger.debug(f"Registered PAT agent: {agent_id}")

    def get_trust(self, agent_id: str) -> float:
        """Get trust score for an agent."""
        rel = self.social_graph._relationships.get(agent_id)
        return rel.trust_score if rel else 0.1  # Default low trust for unknown

    def _calculate_capability_match(
        self,
        role: AgentRole,
        required_capabilities: set[str],
    ) -> float:
        """Calculate how well an agent's capabilities match requirements."""
        agent_caps = self.ROLE_CAPABILITIES.get(role, set())

        if not required_capabilities:
            return 1.0  # No requirements = full match

        overlap = agent_caps & required_capabilities
        return (
            len(overlap) / len(required_capabilities) if required_capabilities else 0.0
        )

    def _filter_by_capability(
        self,
        required_capabilities: set[str],
    ) -> list[tuple[str, AgentRole]]:
        """Filter agents by required capabilities."""
        capable = []

        for role, caps in self.ROLE_CAPABILITIES.items():
            if required_capabilities <= caps:  # Agent has all required caps
                agent_id = f"pat:{role.value}"
                capable.append((agent_id, role))

        return capable

    def select_agent_for_task(
        self,
        required_capabilities: set[str],
        prefer_diversity: bool = False,
    ) -> ScoredAgent:
        """
        Select best agent considering both capability and trust.

        Algorithm:
        1. Filter agents by capability
        2. Score by: capability_match * (1 - trust_weight) + trust * trust_weight
        3. Apply Granovetter weak ties bonus if prefer_diversity
        4. Select highest scoring agent

        Args:
            required_capabilities: set of required capabilities
            prefer_diversity: If True, bonus for weak ties (Granovetter)

        Returns:
            ScoredAgent with selection details

        Raises:
            NoCapableAgentError: If no capable agent found
        """
        capable_agents = self._filter_by_capability(required_capabilities)

        if not capable_agents:
            raise NoCapableAgentError(required_capabilities)

        scored_agents: list[ScoredAgent] = []

        for agent_id, role in capable_agents:
            trust = self.get_trust(agent_id)

            # Skip agents below trust threshold
            if trust < self.min_trust_threshold:
                logger.debug(
                    f"Skipping {agent_id}: trust {trust:.2f} < {self.min_trust_threshold}"
                )
                continue

            capability_score = self._calculate_capability_match(
                role, required_capabilities
            )

            # Combined score with trust weighting
            combined_score = (
                capability_score * (1 - self.trust_weight) + trust * self.trust_weight
            )

            # Granovetter weak ties bonus: slightly prefer less-used agents
            if prefer_diversity:
                rel = self.social_graph._relationships.get(agent_id)
                if rel and rel.trust_score < 0.7:
                    combined_score *= 1.05  # 5% bonus for weak ties

            scored_agents.append(
                ScoredAgent(
                    agent_id=agent_id,
                    role=role,
                    capability_score=capability_score,
                    trust_score=trust,
                    combined_score=combined_score,
                    capabilities=self.ROLE_CAPABILITIES.get(role, set()),
                )
            )

        if not scored_agents:
            raise NoCapableAgentError(required_capabilities)

        # Sort by combined score descending
        scored_agents.sort(key=lambda x: x.combined_score, reverse=True)

        selected = scored_agents[0]
        logger.info(
            f"Selected agent: {selected.agent_id} "
            f"(trust={selected.trust_score:.2f}, combined={selected.combined_score:.2f})"
        )

        return selected

    def find_collaboration_partners(
        self,
        task_capabilities: set[str],
        min_synergy: float = 0.6,
    ) -> list[CollaborationMatch]:
        """
        Find agent pairs that collaborate well for multi-agent tasks.

        Uses interaction history to identify synergistic pairs.
        Inspired by Granovetter's weak ties and Malone's collective intelligence.

        Args:
            task_capabilities: Capabilities needed for the task
            min_synergy: Minimum synergy score to include pair

        Returns:
            list of CollaborationMatch with agent pairs
        """
        matches: list[CollaborationMatch] = []

        # Get all PAT agents
        pat_agents = list(self.ROLE_CAPABILITIES.keys())

        # Check each pair
        for i, role_a in enumerate(pat_agents):
            for role_b in pat_agents[i + 1 :]:
                agent_a = f"pat:{role_a.value}"
                agent_b = f"pat:{role_b.value}"

                # Calculate synergy from combined capabilities
                caps_a = self.ROLE_CAPABILITIES.get(role_a, set())
                caps_b = self.ROLE_CAPABILITIES.get(role_b, set())
                combined_caps = caps_a | caps_b

                # Coverage of task requirements
                if task_capabilities:
                    coverage = len(combined_caps & task_capabilities) / len(
                        task_capabilities
                    )
                else:
                    coverage = 1.0

                # Trust factor (average trust of both)
                trust_a = self.get_trust(agent_a)
                trust_b = self.get_trust(agent_b)
                trust_factor = (trust_a + trust_b) / 2

                # Synergy = coverage * trust * complementarity
                complementarity = (
                    1 - (len(caps_a & caps_b) / len(caps_a | caps_b))
                    if caps_a | caps_b
                    else 0
                )
                synergy = coverage * trust_factor * (0.7 + 0.3 * complementarity)

                if synergy >= min_synergy:
                    matches.append(
                        CollaborationMatch(
                            agent_a=agent_a,
                            agent_b=agent_b,
                            synergy_score=synergy,
                            recommended_task_types=combined_caps & task_capabilities,
                        )
                    )

        # Sort by synergy descending
        matches.sort(key=lambda x: x.synergy_score, reverse=True)

        logger.debug(f"Found {len(matches)} collaboration matches above {min_synergy}")
        return matches

    async def report_task_outcome(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        value: float = 0.0,
    ) -> None:
        """
        Update social graph after task completion.

        Trust Model:
        - Success: trust += 0.05 * (1 - current_trust)  # Diminishing returns
        - Failure: trust -= 0.1 * current_trust          # Proportional penalty

        Args:
            agent_id: ID of the agent that executed
            task_id: ID of the completed task
            success: Whether task succeeded
            value: Value/impact of the task
        """
        rel = self.social_graph._relationships.get(agent_id)

        if not rel:
            logger.warning(f"Unknown agent: {agent_id}")
            return

        old_trust = rel.trust_score

        if success:
            # Success increases trust with diminishing returns
            delta = 0.05 * (1 - rel.trust_score)
            rel.trust_score = min(1.0, rel.trust_score + delta)
            rel.reliability_score = min(1.0, rel.reliability_score + 0.02)
        else:
            # Failure decreases trust proportionally
            delta = 0.1 * rel.trust_score
            rel.trust_score = max(0.0, rel.trust_score - delta)
            rel.reliability_score = max(0.0, rel.reliability_score - 0.05)

        # Update interaction count
        rel.interaction_count = getattr(rel, "interaction_count", 0) + 1

        logger.info(
            f"Trust update for {agent_id}: {old_trust:.3f} → {rel.trust_score:.3f} "
            f"({'success' if success else 'failure'})"
        )

        # Propagate trust for high-value tasks
        if abs(value) > 100:
            self._propagate_trust_local(iterations=3)

    def _propagate_trust_local(self, iterations: int = 5) -> None:
        """
        Local trust propagation (simplified PageRank).

        Updates trust scores based on connected agents' trust.
        """
        damping = 0.85

        for _ in range(iterations):
            new_scores: dict[str, float] = {}

            for agent_id, rel in self.social_graph._relationships.items():
                # Simple propagation: weighted average of neighbors
                neighbors = [
                    r
                    for r in self.social_graph._relationships.values()
                    if r.peer_id != agent_id
                ]

                if neighbors:
                    avg_neighbor_trust = sum(n.trust_score for n in neighbors) / len(
                        neighbors
                    )
                    new_scores[agent_id] = (
                        1 - damping
                    ) * 0.5 + damping * (  # Base trust
                        rel.trust_score * 0.7 + avg_neighbor_trust * 0.3
                    )
                else:
                    new_scores[agent_id] = rel.trust_score

            # Apply new scores
            for agent_id, score in new_scores.items():
                if agent_id in self.social_graph._relationships:
                    self.social_graph._relationships[agent_id].trust_score = score

    def get_network_metrics(self) -> dict[str, Any]:
        """Get social network metrics."""
        rels = list(self.social_graph._relationships.values())

        if not rels:
            return {"total_agents": 0, "average_trust": 0.0}

        return {
            "total_agents": len(rels),
            "average_trust": sum(r.trust_score for r in rels) / len(rels),
            "average_reliability": sum(r.reliability_score for r in rels) / len(rels),
            "high_trust_agents": sum(1 for r in rels if r.trust_score >= 0.8),
            "low_trust_agents": sum(1 for r in rels if r.trust_score < 0.3),
        }
