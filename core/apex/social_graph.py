"""
BIZRA Social Graph — Relationship Intelligence Engine
═══════════════════════════════════════════════════════════════════════════════

Implements agent-to-agent relationship management with Graph-of-Thoughts
reasoning for collaboration discovery and trust propagation.

Standing on the Shoulders of Giants:
- Granovetter (1973): Strength of Weak Ties — diverse connections > strong cliques
- Dunbar (1992): Cognitive limit of ~150 stable relationships
- Page & Brin (1998): PageRank for trust propagation
- Barabási (2002): Scale-free networks, preferential attachment
- Malone (2018): Collective Intelligence Factor

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                      SocialGraph                              │
    │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
    │  │ Relationship│  │ Trust      │  │ Collaboration          │ │
    │  │ Manager     │  │ Propagator │  │ Finder                 │ │
    │  └──────┬─────┘  └──────┬─────┘  └──────────┬─────────────┘ │
    │         └───────────────┼───────────────────┘               │
    │                         ▼                                   │
    │              ┌──────────────────────┐                       │
    │              │  Negotiation Engine  │                       │
    │              └──────────────────────┘                       │
    └──────────────────────────────────────────────────────────────┘

Performance: O(log n) relationship lookup, O(n) trust propagation

Created: 2026-02-04 | BIZRA Apex System v1.0
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (Standing on Giants)
# =============================================================================

# Dunbar's number: cognitive limit for stable relationships
DUNBAR_LIMIT = 150

# Granovetter weak tie threshold (acquaintances vs close friends)
WEAK_TIE_THRESHOLD = 0.3
STRONG_TIE_THRESHOLD = 0.7

# PageRank damping factor (Brin & Page, 1998)
PAGERANK_DAMPING = 0.85

# Trust decay half-life in days (relationships need maintenance)
TRUST_HALFLIFE_DAYS = 30

# Minimum interactions for meaningful relationship
MIN_INTERACTION_COUNT = 3

# Collaboration value threshold (2x cost minimum)
COLLABORATION_VALUE_RATIO = 2.0


# =============================================================================
# ENUMS
# =============================================================================


class RelationshipType(str, Enum):
    """Types of agent relationships."""

    UNKNOWN = "unknown"
    ACQUAINTANCE = "acquaintance"  # Weak tie (Granovetter)
    COLLABORATOR = "collaborator"  # Working relationship
    TRUSTED = "trusted"  # Strong tie
    STRATEGIC = "strategic"  # High-value partnership


class InteractionType(str, Enum):
    """Types of interactions between agents."""

    MESSAGE = "message"
    TASK_DELEGATION = "task_delegation"
    TASK_COMPLETION = "task_completion"
    CONSENSUS_VOTE = "consensus_vote"
    TRADE = "trade"
    COLLABORATION = "collaboration"
    REPUTATION_ATTESTATION = "reputation_attestation"


class CollaborationStatus(str, Enum):
    """Status of collaboration opportunities."""

    DISCOVERED = "discovered"
    PROPOSED = "proposed"
    NEGOTIATING = "negotiating"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Interaction:
    """Record of an interaction between agents."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    interaction_type: InteractionType = InteractionType.MESSAGE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    outcome_score: float = 0.5  # [0,1] — success of interaction
    value_exchanged: float = 0.0  # Economic value
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """
    Agent-to-agent relationship with trust dynamics.

    Trust Model (Standing on Giants):
    - Initial trust: 0.1 (assume unknown)
    - Trust grows with successful interactions
    - Trust decays with time (half-life model)
    - Trust bounded [0, 1]
    """

    agent_id: str
    peer_id: str

    # Trust metrics
    trust_score: float = 0.1  # [0,1]
    reliability_score: float = 0.5  # [0,1] — do they deliver?
    reciprocity_score: float = 0.5  # [0,1] — balanced give/take

    # Relationship metadata
    relationship_type: RelationshipType = RelationshipType.UNKNOWN
    first_interaction: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_interaction: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Interaction history (bounded to save memory)
    interactions: List[Interaction] = field(default_factory=list)
    interaction_count: int = 0
    successful_interactions: int = 0

    # Capabilities known about peer
    known_capabilities: Set[str] = field(default_factory=set)

    # Economic metrics
    total_value_given: float = 0.0
    total_value_received: float = 0.0

    def update_trust(self, interaction: Interaction) -> float:
        """
        Update trust based on new interaction.

        Algorithm (Bayesian-inspired):
        1. Time decay existing trust (half-life model)
        2. Blend with new evidence
        3. Update reliability and reciprocity
        """
        # Time decay (Shannon entropy model)
        days_since_last = (interaction.timestamp - self.last_interaction).days
        decay_factor = math.pow(0.5, days_since_last / TRUST_HALFLIFE_DAYS)
        decayed_trust = self.trust_score * decay_factor

        # Blend with new evidence (Bayesian update approximation)
        evidence_weight = 1.0 / (self.interaction_count + 1)
        self.trust_score = (
            1 - evidence_weight
        ) * decayed_trust + evidence_weight * interaction.outcome_score
        self.trust_score = max(0.0, min(1.0, self.trust_score))

        # Update reliability
        if interaction.outcome_score >= 0.7:
            self.successful_interactions += 1
        self.interaction_count += 1
        self.reliability_score = self.successful_interactions / self.interaction_count

        # Update reciprocity
        self.total_value_given += (
            interaction.value_exchanged if interaction.value_exchanged > 0 else 0
        )
        self.total_value_received += (
            abs(interaction.value_exchanged) if interaction.value_exchanged < 0 else 0
        )

        if self.total_value_given + self.total_value_received > 0:
            balance = self.total_value_received / (
                self.total_value_given + self.total_value_received
            )
            self.reciprocity_score = 1.0 - abs(0.5 - balance) * 2  # Perfect at 0.5

        # Update relationship type
        self._update_relationship_type()

        # Record interaction (bounded list)
        self.interactions.append(interaction)
        if len(self.interactions) > 100:
            self.interactions = self.interactions[-100:]

        self.last_interaction = interaction.timestamp
        return self.trust_score

    def _update_relationship_type(self):
        """Classify relationship based on trust metrics."""
        combined = (
            self.trust_score + self.reliability_score + self.reciprocity_score
        ) / 3

        if combined >= STRONG_TIE_THRESHOLD and self.interaction_count >= 10:
            self.relationship_type = RelationshipType.STRATEGIC
        elif combined >= STRONG_TIE_THRESHOLD:
            self.relationship_type = RelationshipType.TRUSTED
        elif combined >= WEAK_TIE_THRESHOLD:
            self.relationship_type = RelationshipType.COLLABORATOR
        elif self.interaction_count >= MIN_INTERACTION_COUNT:
            self.relationship_type = RelationshipType.ACQUAINTANCE
        else:
            self.relationship_type = RelationshipType.UNKNOWN

    @property
    def strength(self) -> float:
        """Combined relationship strength score."""
        return (
            self.trust_score * 0.4
            + self.reliability_score * 0.35
            + self.reciprocity_score * 0.25
        )

    @property
    def is_weak_tie(self) -> bool:
        """Granovetter weak tie — valuable for diverse information."""
        return WEAK_TIE_THRESHOLD <= self.strength < STRONG_TIE_THRESHOLD


@dataclass
class CollaborationOpportunity:
    """
    Detected opportunity for agent collaboration.

    Value Calculation (Malone's Collective Intelligence):
    - Potential value = f(complementary skills, trust, past success)
    - Cost = coordination overhead + risk
    - Net value must exceed 2x cost (COLLABORATION_VALUE_RATIO)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    initiator_id: str = ""
    partner_id: str = ""

    # Opportunity details
    description: str = ""
    required_capabilities: Set[str] = field(default_factory=set)
    complementary_capabilities: Set[str] = field(default_factory=set)

    # Value assessment
    potential_value: float = 0.0
    estimated_cost: float = 0.0
    confidence: float = 0.5

    # Status tracking
    status: CollaborationStatus = CollaborationStatus.DISCOVERED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Negotiation terms (filled during negotiation)
    proposed_split: float = 0.5  # Initiator's share
    agreed_split: Optional[float] = None

    @property
    def net_value(self) -> float:
        return self.potential_value - self.estimated_cost

    @property
    def value_ratio(self) -> float:
        if self.estimated_cost <= 0:
            return float("inf")
        return self.potential_value / self.estimated_cost

    @property
    def is_viable(self) -> bool:
        return self.value_ratio >= COLLABORATION_VALUE_RATIO


@dataclass
class NegotiationOffer:
    """An offer in a negotiation."""

    offer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    collaboration_id: str = ""

    # Terms
    value_split: float = 0.5  # Offerer's share
    conditions: List[str] = field(default_factory=list)
    expiry: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Response
    accepted: Optional[bool] = None
    counter_offer: Optional[NegotiationOffer] = None


# =============================================================================
# SOCIAL GRAPH ENGINE
# =============================================================================


class SocialGraph:
    """
    Social relationship graph with trust propagation.

    Implements:
    - Relationship management (Dunbar-bounded)
    - Trust propagation (PageRank-inspired)
    - Collaboration discovery (Graph-of-Thoughts)
    - Automated negotiation

    Standing on Giants:
    - Granovetter: Weak ties for diverse information
    - Dunbar: Cognitive limits on relationships
    - Page & Brin: Trust propagation via PageRank
    - Nash: Game-theoretic negotiation
    """

    def __init__(self, agent_id: str, max_relationships: int = DUNBAR_LIMIT):
        self.agent_id = agent_id
        self.max_relationships = max_relationships

        # Relationship storage (peer_id -> Relationship)
        self._relationships: Dict[str, Relationship] = {}

        # Trust graph for propagation (adjacency list)
        self._trust_graph: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Capability index (capability -> set of agents)
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)

        # Active collaborations
        self._collaborations: Dict[str, CollaborationOpportunity] = {}

        # Pending negotiations
        self._negotiations: Dict[str, List[NegotiationOffer]] = defaultdict(list)

        # My capabilities
        self._my_capabilities: Set[str] = set()

        # PageRank cache
        self._pagerank_cache: Dict[str, float] = {}
        self._pagerank_dirty = True

        logger.info(f"SocialGraph initialized for agent {agent_id}")

    # -------------------------------------------------------------------------
    # RELATIONSHIP MANAGEMENT
    # -------------------------------------------------------------------------

    def add_relationship(
        self, peer_id: str, initial_trust: float = 0.1
    ) -> Relationship:
        """Add or get existing relationship."""
        if peer_id in self._relationships:
            return self._relationships[peer_id]

        # Check Dunbar limit
        if len(self._relationships) >= self.max_relationships:
            # Prune weakest relationship
            self._prune_weakest_relationship()

        rel = Relationship(
            agent_id=self.agent_id, peer_id=peer_id, trust_score=initial_trust
        )
        self._relationships[peer_id] = rel
        self._pagerank_dirty = True

        logger.debug(f"Added relationship with {peer_id}")
        return rel

    def record_interaction(
        self,
        peer_id: str,
        interaction_type: InteractionType,
        outcome_score: float,
        value_exchanged: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> Relationship:
        """Record an interaction and update trust."""
        rel = self.add_relationship(peer_id)

        interaction = Interaction(
            interaction_type=interaction_type,
            outcome_score=outcome_score,
            value_exchanged=value_exchanged,
            metadata=metadata or {},
        )

        old_trust = rel.trust_score
        new_trust = rel.update_trust(interaction)

        # Update trust graph
        self._trust_graph[self.agent_id][peer_id] = new_trust
        self._pagerank_dirty = True

        logger.debug(
            f"Interaction with {peer_id}: trust {old_trust:.3f} → {new_trust:.3f}"
        )
        return rel

    def get_relationship(self, peer_id: str) -> Optional[Relationship]:
        """Get relationship with peer."""
        return self._relationships.get(peer_id)

    def get_trust(self, peer_id: str) -> float:
        """Get current trust score with peer."""
        rel = self._relationships.get(peer_id)
        return rel.trust_score if rel else 0.0

    def _prune_weakest_relationship(self):
        """Remove the weakest relationship to maintain Dunbar limit."""
        if not self._relationships:
            return

        weakest_id = min(
            self._relationships.keys(), key=lambda k: self._relationships[k].strength
        )
        del self._relationships[weakest_id]

        if weakest_id in self._trust_graph[self.agent_id]:
            del self._trust_graph[self.agent_id][weakest_id]

        logger.info(f"Pruned weakest relationship: {weakest_id}")

    # -------------------------------------------------------------------------
    # TRUST PROPAGATION (PageRank-inspired)
    # -------------------------------------------------------------------------

    def propagate_trust(
        self, external_graph: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Compute global trust scores using PageRank algorithm.

        Algorithm (Brin & Page, 1998):
        PR(A) = (1-d) + d * Σ(PR(B) / L(B)) for all B linking to A

        Where:
        - d = damping factor (0.85)
        - L(B) = outbound links from B
        """
        if not self._pagerank_dirty and self._pagerank_cache:
            return self._pagerank_cache

        # Merge external graph if provided
        graph = dict(self._trust_graph)
        if external_graph:
            for node, edges in external_graph.items():
                if node not in graph:
                    graph[node] = {}
                graph[node].update(edges)

        # Get all nodes
        nodes = set(graph.keys())
        for edges in graph.values():
            nodes.update(edges.keys())

        if not nodes:
            return {}

        n = len(nodes)
        node_list = list(nodes)
        {node: i for i, node in enumerate(node_list)}

        # Initialize PageRank
        pr = {node: 1.0 / n for node in nodes}

        # Iterate until convergence
        for _ in range(100):  # Max iterations
            new_pr = {}

            for node in nodes:
                # Random jump probability
                rank = (1 - PAGERANK_DAMPING) / n

                # Sum contributions from incoming edges
                for source, edges in graph.items():
                    if node in edges:
                        out_degree = sum(edges.values())
                        if out_degree > 0:
                            rank += (
                                PAGERANK_DAMPING * pr[source] * edges[node] / out_degree
                            )

                new_pr[node] = rank

            # Check convergence
            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            pr = new_pr

            if diff < 1e-6:
                break

        self._pagerank_cache = pr
        self._pagerank_dirty = False
        return pr

    def get_global_trust(self, peer_id: str) -> float:
        """Get globally-computed trust score for peer."""
        pr = self.propagate_trust()
        return pr.get(peer_id, 0.0)

    # -------------------------------------------------------------------------
    # COLLABORATION DISCOVERY (Graph-of-Thoughts)
    # -------------------------------------------------------------------------

    def register_capabilities(self, capabilities: Set[str]):
        """Register this agent's capabilities."""
        self._my_capabilities = capabilities

    def learn_peer_capabilities(self, peer_id: str, capabilities: Set[str]):
        """Learn about a peer's capabilities."""
        rel = self.add_relationship(peer_id)
        rel.known_capabilities.update(capabilities)

        # Update capability index
        for cap in capabilities:
            self._capability_index[cap].add(peer_id)

    def find_collaborations(
        self,
        required_capabilities: Set[str],
        min_trust: float = WEAK_TIE_THRESHOLD,
        max_results: int = 10,
    ) -> List[CollaborationOpportunity]:
        """
        Find collaboration opportunities using Graph-of-Thoughts reasoning.

        Algorithm:
        1. Identify capability gaps
        2. Find agents with complementary capabilities
        3. Filter by trust threshold
        4. Score by potential value
        5. Return top opportunities
        """
        # Capability gaps
        my_caps = self._my_capabilities
        needed_caps = required_capabilities - my_caps

        if not needed_caps:
            return []  # Can do it alone

        opportunities = []

        # Find agents with needed capabilities
        candidate_agents: Set[str] = set()
        for cap in needed_caps:
            candidate_agents.update(self._capability_index.get(cap, set()))

        # Score each candidate
        for peer_id in candidate_agents:
            rel = self._relationships.get(peer_id)
            if not rel or rel.trust_score < min_trust:
                continue

            # Calculate complementary capabilities
            peer_caps = rel.known_capabilities
            complementary = peer_caps & needed_caps

            if not complementary:
                continue

            # Calculate potential value (Malone collective intelligence)
            coverage = len(complementary) / len(needed_caps)
            trust_factor = rel.trust_score
            reliability_factor = rel.reliability_score

            potential_value = (
                coverage * trust_factor * reliability_factor * 100
            )  # Normalized
            estimated_cost = 10 + (1 - trust_factor) * 20  # Coordination cost

            opp = CollaborationOpportunity(
                initiator_id=self.agent_id,
                partner_id=peer_id,
                description=f"Collaboration for capabilities: {complementary}",
                required_capabilities=required_capabilities,
                complementary_capabilities=complementary,
                potential_value=potential_value,
                estimated_cost=estimated_cost,
                confidence=trust_factor,
            )

            if opp.is_viable:
                opportunities.append(opp)

        # Sort by net value
        opportunities.sort(key=lambda o: o.net_value, reverse=True)

        return opportunities[:max_results]

    def find_weak_tie_opportunities(self) -> List[Tuple[str, Set[str]]]:
        """
        Find valuable weak ties (Granovetter).

        Weak ties provide access to diverse information and
        bridge different network clusters.
        """
        weak_ties = []

        for peer_id, rel in self._relationships.items():
            if rel.is_weak_tie:
                # Unique capabilities from this weak tie
                unique_caps = rel.known_capabilities - self._my_capabilities
                if unique_caps:
                    weak_ties.append((peer_id, unique_caps))

        return weak_ties

    # -------------------------------------------------------------------------
    # NEGOTIATION ENGINE (Nash Bargaining)
    # -------------------------------------------------------------------------

    async def propose_collaboration(
        self, opportunity: CollaborationOpportunity, initial_split: float = 0.5
    ) -> NegotiationOffer:
        """
        Propose a collaboration to a partner.

        Uses Nash bargaining solution as starting point:
        - Fair split = proportional to contribution
        - Adjusted by trust and past reciprocity
        """
        rel = self._relationships.get(opportunity.partner_id)

        # Adjust split based on reciprocity history
        if rel and rel.reciprocity_score != 0.5:
            # If we've given more, ask for more this time
            adjustment = (0.5 - rel.reciprocity_score) * 0.2
            initial_split = 0.5 + adjustment

        offer = NegotiationOffer(
            from_agent=self.agent_id,
            to_agent=opportunity.partner_id,
            collaboration_id=opportunity.id,
            value_split=initial_split,
            expiry=datetime.now(timezone.utc),
        )

        self._negotiations[opportunity.id].append(offer)
        opportunity.status = CollaborationStatus.PROPOSED
        opportunity.proposed_split = initial_split

        self._collaborations[opportunity.id] = opportunity

        logger.info(
            f"Proposed collaboration {opportunity.id} with split {initial_split:.2f}"
        )
        return offer

    def evaluate_offer(self, offer: NegotiationOffer) -> Tuple[bool, Optional[float]]:
        """
        Evaluate incoming collaboration offer.

        Returns (accept, counter_split):
        - (True, None) = accept as-is
        - (False, split) = counter-offer with new split
        - (False, None) = reject
        """
        opp = self._collaborations.get(offer.collaboration_id)
        if not opp:
            return (False, None)

        rel = self._relationships.get(offer.from_agent)
        if not rel:
            return (False, None)

        # My expected share
        my_share = 1.0 - offer.value_split
        opp.potential_value * my_share

        # Minimum acceptable (based on estimated contribution)
        my_contribution = len(opp.complementary_capabilities) / len(
            opp.required_capabilities
        )
        min_acceptable = 0.3 + my_contribution * 0.4  # 30-70% range

        if my_share >= min_acceptable:
            return (True, None)
        elif my_share >= min_acceptable - 0.1:
            # Counter with fair split
            fair_split = 1.0 - (min_acceptable + my_contribution * 0.2)
            return (False, fair_split)
        else:
            return (False, None)

    def accept_collaboration(self, collaboration_id: str, agreed_split: float):
        """Accept a collaboration with agreed terms."""
        opp = self._collaborations.get(collaboration_id)
        if opp:
            opp.status = CollaborationStatus.ACCEPTED
            opp.agreed_split = agreed_split
            logger.info(
                f"Accepted collaboration {collaboration_id} with split {agreed_split:.2f}"
            )

    # -------------------------------------------------------------------------
    # ANALYTICS
    # -------------------------------------------------------------------------

    def get_network_stats(self) -> Dict[str, Any]:
        """Get social network statistics."""
        if not self._relationships:
            return {"total_relationships": 0}

        strengths = [r.strength for r in self._relationships.values()]
        trust_scores = [r.trust_score for r in self._relationships.values()]

        weak_ties = sum(1 for r in self._relationships.values() if r.is_weak_tie)
        strong_ties = sum(
            1
            for r in self._relationships.values()
            if r.strength >= STRONG_TIE_THRESHOLD
        )

        return {
            "total_relationships": len(self._relationships),
            "dunbar_utilization": len(self._relationships) / self.max_relationships,
            "avg_strength": sum(strengths) / len(strengths),
            "avg_trust": sum(trust_scores) / len(trust_scores),
            "weak_ties": weak_ties,
            "strong_ties": strong_ties,
            "weak_tie_ratio": (
                weak_ties / len(self._relationships) if self._relationships else 0
            ),
            "active_collaborations": len(
                [
                    c
                    for c in self._collaborations.values()
                    if c.status == CollaborationStatus.IN_PROGRESS
                ]
            ),
            "unique_capabilities_accessible": len(
                set().union(
                    *[r.known_capabilities for r in self._relationships.values()]
                )
                if self._relationships
                else set()
            ),
        }

    def get_relationship_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all relationships."""
        return [
            {
                "peer_id": r.peer_id,
                "type": r.relationship_type.value,
                "trust": round(r.trust_score, 3),
                "reliability": round(r.reliability_score, 3),
                "reciprocity": round(r.reciprocity_score, 3),
                "strength": round(r.strength, 3),
                "interactions": r.interaction_count,
                "capabilities": list(r.known_capabilities)[:5],
            }
            for r in sorted(
                self._relationships.values(), key=lambda r: r.strength, reverse=True
            )
        ]


# =============================================================================
# FACTORY
# =============================================================================

_social_graph_instances: Dict[str, SocialGraph] = {}


def get_social_graph(agent_id: str) -> SocialGraph:
    """Get or create social graph for agent."""
    if agent_id not in _social_graph_instances:
        _social_graph_instances[agent_id] = SocialGraph(agent_id)
    return _social_graph_instances[agent_id]
