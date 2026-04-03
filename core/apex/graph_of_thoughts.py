"""
BIZRA Graph of Thoughts (GoT) Engine - Enhanced Multi-Dimensional Reasoning
=============================================================================

Implements an enhanced Graph of Thoughts engine for complex multi-dimensional
reasoning with persona soup synthesis, cross-domain connections, and veto gates.

Architecture:
    Following the GoT paper structure: Input -> Generate -> Aggregate -> Score -> Select -> Output

    +---------------------------------------------------------------------------+
    |                       GRAPH OF THOUGHTS ENGINE                             |
    +---------------------------------------------------------------------------+
    |                                                                            |
    |   Task + Domains + Pareto -----> [Build Reasoning Graph]                  |
    |                                         |                                  |
    |                                         v                                  |
    |                            +------------------------+                      |
    |                            |   DAG Construction     |                      |
    |                            |   - Persona Soup Nodes |                      |
    |                            |   - Synthesis Nodes    |                      |
    |                            |   - Evidence Nodes     |                      |
    |                            |   - Veto Gate Nodes    |                      |
    |                            +------------------------+                      |
    |                                         |                                  |
    |                    +--------------------+--------------------+             |
    |                    |                    |                    |             |
    |                    v                    v                    v             |
    |           [Cross-Domain]      [Veto Constraint]    [Consensus]            |
    |              Edges               Edges               Edges                 |
    |                    |                    |                    |             |
    |                    +--------------------+--------------------+             |
    |                                         |                                  |
    |                                         v                                  |
    |                             [BFS Synthesis Traversal]                      |
    |                                 (SNR Pruning)                              |
    |                                         |                                  |
    |                                         v                                  |
    |                             [Compute Final SNR]                            |
    |                             (Diversity Bonus)                              |
    |                                         |                                  |
    |                                         v                                  |
    |                                  Output Result                             |
    |                                                                            |
    +---------------------------------------------------------------------------+

Key Features:
    - DAG-based reasoning structure
    - Cross-domain edge connections for interdisciplinary insights
    - Veto constraint propagation for safety and ethics
    - BFS traversal with SNR pruning (configurable threshold)
    - Diversity bonus for multi-domain synthesis
    - Pareto-optimal solution integration
    - Full type hints and comprehensive documentation

Reference:
    GoT Paper: "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
    Structure: Input -> Generate -> Aggregate -> Score -> Select -> Output

Target Metrics:
    - SNR Threshold: 0.95 (configurable)
    - Max Depth: 5
    - Diversity Bonus: 0.02 per domain

Domain: bizra-apex-v1:
Integration: PAT orchestration, SAPE probes, Ihsan validation
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Configure logging
logger = logging.getLogger("apex.graph_of_thoughts")


# =============================================================================
# CONSTANTS & CONFIGURATION (imported from unified constants.py)
# =============================================================================

from core.constants import IHSAN_THRESHOLD

# Default SNR pruning threshold (configurable) - uses constitutional threshold
DEFAULT_SNR_THRESHOLD = IHSAN_THRESHOLD  # 0.95

# Maximum traversal depth
DEFAULT_MAX_DEPTH = 5

# Diversity bonus per additional domain
DEFAULT_DIVERSITY_BONUS = 0.02

# Domain prefix for receipts
DOMAIN_PREFIX = "bizra-got-v1:"

# Node type weights for SNR computation
NODE_TYPE_WEIGHTS: Dict[str, float] = {
    "persona_soup": 0.25,
    "synthesis": 0.35,
    "evidence": 0.25,
    "veto_gate": 0.15,
}

# Edge type weights for path scoring
EDGE_TYPE_WEIGHTS: Dict[str, float] = {
    "cross_domain": 0.30,
    "veto_constraint": 0.25,
    "consensus": 0.25,
    "synthesis": 0.20,
}


# =============================================================================
# ENUMS
# =============================================================================


class GoTNodeType(str, Enum):
    """
    Types of nodes in the Graph of Thoughts.

    Attributes:
        PERSONA_SOUP: Represents a domain-specific persona cluster
        SYNTHESIS: Intermediate synthesis combining multiple inputs
        EVIDENCE: Evidence or citation supporting reasoning
        VETO_GATE: Safety/ethics checkpoint with veto power
    """
    PERSONA_SOUP = "persona_soup"
    SYNTHESIS = "synthesis"
    EVIDENCE = "evidence"
    VETO_GATE = "veto_gate"


class GoTEdgeType(str, Enum):
    """
    Types of edges in the Graph of Thoughts.

    Attributes:
        CROSS_DOMAIN: Connects complementary persona soups across domains
        VETO_CONSTRAINT: Propagates veto power from safety/ethics gates
        CONSENSUS: Represents agreement between multiple reasoning paths
        SYNTHESIS: Connects inputs to their synthesis outputs
    """
    CROSS_DOMAIN = "cross_domain"
    VETO_CONSTRAINT = "veto_constraint"
    CONSENSUS = "consensus"
    SYNTHESIS = "synthesis"


class GoTTraversalStatus(str, Enum):
    """Status of graph traversal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PRUNED = "pruned"
    VETOED = "vetoed"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GoTNode:
    """
    Represents a node in the Graph of Thoughts reasoning structure.

    A node can be a persona soup (domain expertise), synthesis (combination),
    evidence (supporting citations), or veto gate (safety checkpoint).

    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of node (persona_soup, synthesis, evidence, veto_gate)
        content: The actual content or reasoning at this node
        snr_score: Signal-to-noise ratio score (0.0 to 1.0)
        parent_ids: List of parent node IDs (incoming edges)
        children_ids: List of child node IDs (outgoing edges)
        domain: Optional domain label for persona_soup nodes
        depth: Depth in the DAG (0 for root nodes)
        metadata: Additional node-specific metadata
        timestamp: Creation timestamp
        veto_active: Whether this node has active veto status (for veto_gate)
        veto_reason: Reason for veto if active
    """
    node_id: str
    node_type: GoTNodeType
    content: str
    snr_score: float
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    veto_active: bool = False
    veto_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate node after initialization."""
        if not 0.0 <= self.snr_score <= 1.0:
            raise ValueError(f"SNR score must be between 0.0 and 1.0, got {self.snr_score}")
        if self.depth < 0:
            raise ValueError(f"Depth must be non-negative, got {self.depth}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "snr_score": self.snr_score,
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "domain": self.domain,
            "depth": self.depth,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "veto_active": self.veto_active,
            "veto_reason": self.veto_reason,
        }

    def is_root(self) -> bool:
        """Check if this node is a root (no parents)."""
        return len(self.parent_ids) == 0

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children_ids) == 0

    def has_veto(self) -> bool:
        """Check if this node has active veto status."""
        return self.node_type == GoTNodeType.VETO_GATE and self.veto_active


@dataclass
class GoTEdge:
    """
    Represents an edge in the Graph of Thoughts.

    Edges connect nodes and carry semantic meaning about the relationship
    between connected reasoning elements.

    Attributes:
        edge_id: Unique identifier for the edge
        edge_type: Type of edge (cross_domain, veto_constraint, consensus, synthesis)
        source_id: ID of the source node
        target_id: ID of the target node
        weight: Edge weight (0.0 to 1.0) indicating connection strength
        metadata: Additional edge-specific metadata
        timestamp: Creation timestamp
        propagates_veto: Whether this edge propagates veto constraints
    """
    edge_id: str
    edge_type: GoTEdgeType
    source_id: str
    target_id: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    propagates_veto: bool = False

    def __post_init__(self) -> None:
        """Validate edge after initialization."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Edge weight must be between 0.0 and 1.0, got {self.weight}")
        # Veto constraint edges always propagate veto
        if self.edge_type == GoTEdgeType.VETO_CONSTRAINT:
            self.propagates_veto = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "propagates_veto": self.propagates_veto,
        }


@dataclass
class TaskDomain:
    """
    Represents a domain for task decomposition.

    Attributes:
        domain_id: Unique identifier for the domain
        name: Human-readable domain name
        cluster_id: ID of the expertise cluster
        persona_ids: List of persona IDs in this domain
        relevance_score: Relevance to the current task (0.0 to 1.0)
        metadata: Additional domain metadata
    """
    domain_id: str
    name: str
    cluster_id: str
    persona_ids: List[str] = field(default_factory=list)
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "cluster_id": self.cluster_id,
            "persona_ids": self.persona_ids,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }


@dataclass
class ParetoSolution:
    """
    Represents a Pareto-optimal solution for multi-objective optimization.

    Attributes:
        solution_id: Unique identifier
        objectives: Dictionary mapping objective names to scores
        dominated_by: Number of solutions that dominate this one
        dominates: Number of solutions this one dominates
        metadata: Additional solution metadata
    """
    solution_id: str
    objectives: Dict[str, float]
    dominated_by: int = 0
    dominates: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_pareto_optimal(self) -> bool:
        """Check if this solution is Pareto optimal (not dominated)."""
        return self.dominated_by == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "solution_id": self.solution_id,
            "objectives": self.objectives,
            "dominated_by": self.dominated_by,
            "dominates": self.dominates,
            "is_pareto_optimal": self.is_pareto_optimal(),
            "metadata": self.metadata,
        }


@dataclass
class SynthesisResult:
    """
    Result from BFS synthesis traversal.

    Attributes:
        root_id: ID of the root node
        synthesized_content: Combined content from traversal
        final_snr: Final SNR score after synthesis
        diversity_bonus: Bonus from multi-domain coverage
        total_snr: Final SNR including diversity bonus
        nodes_visited: Number of nodes visited
        nodes_pruned: Number of nodes pruned due to low SNR
        nodes_vetoed: Number of nodes blocked by veto
        domains_covered: Set of domains covered in synthesis
        traversal_path: List of node IDs in traversal order
        depth_reached: Maximum depth reached
        metadata: Additional synthesis metadata
        timestamp: Synthesis timestamp
    """
    root_id: str
    synthesized_content: str
    final_snr: float
    diversity_bonus: float
    total_snr: float
    nodes_visited: int
    nodes_pruned: int
    nodes_vetoed: int
    domains_covered: Set[str]
    traversal_path: List[str]
    depth_reached: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "root_id": self.root_id,
            "synthesized_content": (
                self.synthesized_content[:500] + "..."
                if len(self.synthesized_content) > 500
                else self.synthesized_content
            ),
            "final_snr": self.final_snr,
            "diversity_bonus": self.diversity_bonus,
            "total_snr": self.total_snr,
            "nodes_visited": self.nodes_visited,
            "nodes_pruned": self.nodes_pruned,
            "nodes_vetoed": self.nodes_vetoed,
            "domains_covered": list(self.domains_covered),
            "domain_count": len(self.domains_covered),
            "traversal_path": self.traversal_path[:20],  # First 20 for brevity
            "depth_reached": self.depth_reached,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class GoTGraphResult:
    """
    Complete result from Graph of Thoughts reasoning.

    Attributes:
        graph_id: Unique identifier for this graph instance
        task: Original task description
        node_count: Total number of nodes
        edge_count: Total number of edges
        root_ids: IDs of root nodes
        synthesis_results: Results from BFS synthesis
        final_snr: Best overall SNR score
        domains_used: Domains used in reasoning
        pareto_solutions: Pareto-optimal solutions considered
        vetoed: Whether reasoning was vetoed
        veto_reason: Reason for veto if applicable
        metadata: Additional graph metadata
        timestamp: Graph creation timestamp
        latency_ms: Processing time in milliseconds
    """
    graph_id: str
    task: str
    node_count: int
    edge_count: int
    root_ids: List[str]
    synthesis_results: List[SynthesisResult]
    final_snr: float
    domains_used: List[str]
    pareto_solutions: List[ParetoSolution]
    vetoed: bool = False
    veto_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "graph_id": self.graph_id,
            "task": self.task[:200] + "..." if len(self.task) > 200 else self.task,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "root_ids": self.root_ids,
            "synthesis_count": len(self.synthesis_results),
            "final_snr": self.final_snr,
            "domains_used": self.domains_used,
            "domain_count": len(self.domains_used),
            "pareto_solution_count": len(self.pareto_solutions),
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
        }


# =============================================================================
# GRAPH OF THOUGHTS ENGINE
# =============================================================================


class GraphOfThoughtsEngine:
    """
    Enhanced Graph of Thoughts engine for complex multi-dimensional reasoning.

    Implements the GoT paper structure: Input -> Generate -> Aggregate -> Score -> Select -> Output

    Features:
        - DAG-based reasoning graph construction
        - Cross-domain edge connections for interdisciplinary synthesis
        - Veto constraint propagation for safety gates
        - BFS traversal with SNR pruning
        - Diversity bonus for multi-domain coverage
        - Pareto-optimal solution integration

    Usage:
        engine = GraphOfThoughtsEngine(snr_threshold=0.95)
        result = engine.build_reasoning_graph(
            task="Optimize data pipeline",
            task_domains=[domain1, domain2, domain3],
            pareto_solutions=[solution1, solution2],
        )
    """

    def __init__(
        self,
        snr_threshold: float = DEFAULT_SNR_THRESHOLD,
        max_depth: int = DEFAULT_MAX_DEPTH,
        diversity_bonus: float = DEFAULT_DIVERSITY_BONUS,
    ) -> None:
        """
        Initialize the Graph of Thoughts engine.

        Args:
            snr_threshold: Minimum SNR score for node inclusion (default: 0.95)
            max_depth: Maximum traversal depth (default: 5)
            diversity_bonus: Bonus per additional domain (default: 0.02)
        """
        if not 0.0 <= snr_threshold <= 1.0:
            raise ValueError(f"SNR threshold must be between 0.0 and 1.0, got {snr_threshold}")
        if max_depth < 1:
            raise ValueError(f"Max depth must be at least 1, got {max_depth}")
        if diversity_bonus < 0:
            raise ValueError(f"Diversity bonus must be non-negative, got {diversity_bonus}")

        self.snr_threshold = snr_threshold
        self.max_depth = max_depth
        self.diversity_bonus = diversity_bonus

        # Graph storage
        self._nodes: Dict[str, GoTNode] = {}
        self._edges: Dict[str, GoTEdge] = {}

        # Adjacency lists for efficient traversal
        self._outgoing: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        self._incoming: Dict[str, List[str]] = {}  # node_id -> [edge_ids]

        # Domain tracking
        self._domain_nodes: Dict[str, Set[str]] = {}  # domain -> {node_ids}

        # Veto tracking
        self._vetoed_nodes: Set[str] = set()

        # Statistics
        self._total_graphs_built = 0
        self._total_syntheses = 0
        self._total_vetoes = 0

        logger.info(
            f"GraphOfThoughtsEngine initialized: "
            f"snr_threshold={snr_threshold}, max_depth={max_depth}, "
            f"diversity_bonus={diversity_bonus}"
        )

    # =========================================================================
    # CORE API
    # =========================================================================

    def build_reasoning_graph(
        self,
        task: str,
        task_domains: List[TaskDomain],
        pareto_solutions: Optional[List[ParetoSolution]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        enable_veto_gates: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoTGraphResult:
        """
        Construct a complete reasoning DAG from task inputs.

        Implements the GoT paper flow:
            1. Input: Parse task and domains
            2. Generate: Create persona soup and synthesis nodes
            3. Aggregate: Build cross-domain and consensus edges
            4. Score: Compute SNR scores with diversity bonus
            5. Select: BFS traversal with SNR pruning
            6. Output: Return synthesized reasoning

        Args:
            task: The reasoning task description
            task_domains: List of TaskDomain objects representing expertise areas
            pareto_solutions: Optional Pareto-optimal solutions to integrate
            evidence: Optional evidence/citations to include
            enable_veto_gates: Whether to add veto gate nodes (default: True)
            metadata: Additional metadata for the graph

        Returns:
            GoTGraphResult with complete reasoning graph and synthesis results

        Raises:
            ValueError: If task is empty or no domains provided
        """
        import time
        start_time = time.perf_counter()

        # Validate inputs
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")
        if not task_domains:
            raise ValueError("At least one task domain is required")

        # Reset graph state for new build
        self._reset_graph()

        graph_id = self._generate_id("graph")
        graph_metadata = metadata or {}
        graph_metadata["task"] = task
        graph_metadata["domain_count"] = len(task_domains)

        logger.info(f"Building reasoning graph: graph_id={graph_id}, domains={len(task_domains)}")

        # Step 1 & 2: Generate - Create persona soup nodes for each domain
        root_ids: List[str] = []
        for domain in task_domains:
            node = self._create_persona_soup_node(domain, task)
            root_ids.append(node.node_id)

        # Step 2: Generate - Create evidence nodes if provided
        evidence_node_ids: List[str] = []
        if evidence:
            for ev in evidence:
                node = self._create_evidence_node(ev)
                evidence_node_ids.append(node.node_id)

        # Step 2: Generate - Add veto gates if enabled
        veto_gate_ids: List[str] = []
        if enable_veto_gates:
            veto_gate_ids = self._create_veto_gates(task, task_domains)

        # Step 2: Generate - Create synthesis nodes from Pareto solutions
        pareto_node_ids: List[str] = []
        pareto_list = pareto_solutions or []
        for pareto in pareto_list:
            node = self._create_pareto_synthesis_node(pareto)
            pareto_node_ids.append(node.node_id)

        # Step 3: Aggregate - Add cross-domain edges
        self._add_cross_domain_edges(task_domains)

        # Step 3: Aggregate - Add veto constraint edges
        if enable_veto_gates:
            self._add_veto_constraint_edges()

        # Step 3: Aggregate - Add consensus edges between related nodes
        self._add_consensus_edges(root_ids, evidence_node_ids)

        # Step 3: Aggregate - Add synthesis edges for Pareto nodes
        self._add_synthesis_edges(root_ids, pareto_node_ids)

        # Step 4 & 5: Score and Select - BFS synthesis from each root
        synthesis_results: List[SynthesisResult] = []
        for root_id in root_ids:
            result = self._bfs_synthesis(root_id)
            synthesis_results.append(result)

        # Step 6: Output - Compute final SNR across all syntheses
        final_snr = self._compute_final_snr(synthesis_results)

        # Check for global veto
        vetoed = len(self._vetoed_nodes) > 0
        veto_reason = None
        if vetoed:
            self._total_vetoes += 1
            veto_reasons = [
                self._nodes[nid].veto_reason
                for nid in self._vetoed_nodes
                if self._nodes[nid].veto_reason
            ]
            veto_reason = "; ".join(veto_reasons[:3])  # First 3 reasons

        # Collect domains used
        domains_used = list(set(
            self._nodes[nid].domain
            for nid in self._nodes
            if self._nodes[nid].domain
        ))

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._total_graphs_built += 1

        result = GoTGraphResult(
            graph_id=graph_id,
            task=task,
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            root_ids=root_ids,
            synthesis_results=synthesis_results,
            final_snr=final_snr,
            domains_used=domains_used,
            pareto_solutions=pareto_list,
            vetoed=vetoed,
            veto_reason=veto_reason,
            metadata=graph_metadata,
            latency_ms=latency_ms,
        )

        logger.info(
            f"Graph built: id={graph_id[:8]}..., nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, snr={final_snr:.4f}, "
            f"vetoed={vetoed}, latency={latency_ms:.2f}ms"
        )

        return result

    # =========================================================================
    # EDGE CONSTRUCTION
    # =========================================================================

    def _add_cross_domain_edges(self, task_domains: List[TaskDomain]) -> None:
        """
        Connect complementary persona soups across different domains.

        Creates cross-domain edges between nodes from different domains
        based on complementarity and relevance scores.

        Args:
            task_domains: List of task domains to connect
        """
        # Get all domain names
        domain_names = [d.name for d in task_domains]

        # Connect nodes from different domains
        for i, domain1 in enumerate(domain_names):
            for j, domain2 in enumerate(domain_names):
                if i >= j:
                    continue  # Only connect each pair once

                nodes1 = self._domain_nodes.get(domain1, set())
                nodes2 = self._domain_nodes.get(domain2, set())

                for node1_id in nodes1:
                    for node2_id in nodes2:
                        # Compute cross-domain weight based on complementarity
                        weight = self._compute_cross_domain_weight(
                            self._nodes[node1_id],
                            self._nodes[node2_id],
                        )

                        if weight > 0.3:  # Minimum threshold for connection
                            # Create bidirectional cross-domain edges
                            self._create_edge(
                                source_id=node1_id,
                                target_id=node2_id,
                                edge_type=GoTEdgeType.CROSS_DOMAIN,
                                weight=weight,
                                metadata={
                                    "domain1": domain1,
                                    "domain2": domain2,
                                    "complementarity": weight,
                                },
                            )

        cross_domain_count = sum(
            1 for e in self._edges.values()
            if e.edge_type == GoTEdgeType.CROSS_DOMAIN
        )
        logger.debug(f"Added {cross_domain_count} cross-domain edges")

    def _add_veto_constraint_edges(self) -> None:
        """
        Propagate veto power from veto gate nodes to downstream nodes.

        Creates veto constraint edges that enforce safety and ethics
        checks throughout the reasoning graph.
        """
        # Find all veto gate nodes
        veto_gates = [
            node for node in self._nodes.values()
            if node.node_type == GoTNodeType.VETO_GATE
        ]

        # Connect veto gates to all synthesis and persona nodes
        for veto_gate in veto_gates:
            for node_id, node in self._nodes.items():
                if node_id == veto_gate.node_id:
                    continue

                # Veto gates connect to synthesis and persona nodes
                if node.node_type in (GoTNodeType.SYNTHESIS, GoTNodeType.PERSONA_SOUP):
                    # Weight based on veto gate severity
                    weight = 1.0 if veto_gate.veto_active else 0.8

                    self._create_edge(
                        source_id=veto_gate.node_id,
                        target_id=node_id,
                        edge_type=GoTEdgeType.VETO_CONSTRAINT,
                        weight=weight,
                        metadata={
                            "veto_gate_id": veto_gate.node_id,
                            "target_type": node.node_type.value,
                        },
                    )

                    # If veto is active, mark target as vetoed
                    if veto_gate.veto_active:
                        self._vetoed_nodes.add(node_id)

        veto_edge_count = sum(
            1 for e in self._edges.values()
            if e.edge_type == GoTEdgeType.VETO_CONSTRAINT
        )
        logger.debug(f"Added {veto_edge_count} veto constraint edges")

    def _add_consensus_edges(
        self,
        root_ids: List[str],
        evidence_ids: List[str],
    ) -> None:
        """
        Add consensus edges between nodes that agree on reasoning.

        Connects root nodes and evidence nodes that support similar conclusions.

        Args:
            root_ids: IDs of root persona soup nodes
            evidence_ids: IDs of evidence nodes
        """
        # Connect evidence to relevant persona nodes
        for evidence_id in evidence_ids:
            evidence_node = self._nodes[evidence_id]

            for root_id in root_ids:
                root_node = self._nodes[root_id]

                # Compute consensus weight based on content similarity
                weight = self._compute_consensus_weight(root_node, evidence_node)

                if weight > 0.4:  # Minimum consensus threshold
                    self._create_edge(
                        source_id=evidence_id,
                        target_id=root_id,
                        edge_type=GoTEdgeType.CONSENSUS,
                        weight=weight,
                        metadata={
                            "evidence_support": True,
                            "consensus_strength": weight,
                        },
                    )

        # Connect root nodes with high agreement
        for i, root1_id in enumerate(root_ids):
            for j, root2_id in enumerate(root_ids):
                if i >= j:
                    continue

                root1 = self._nodes[root1_id]
                root2 = self._nodes[root2_id]

                weight = self._compute_consensus_weight(root1, root2)

                if weight > 0.5:  # Higher threshold for root-root consensus
                    self._create_edge(
                        source_id=root1_id,
                        target_id=root2_id,
                        edge_type=GoTEdgeType.CONSENSUS,
                        weight=weight,
                        metadata={
                            "inter_domain_consensus": True,
                            "domains": [root1.domain, root2.domain],
                        },
                    )

        consensus_count = sum(
            1 for e in self._edges.values()
            if e.edge_type == GoTEdgeType.CONSENSUS
        )
        logger.debug(f"Added {consensus_count} consensus edges")

    def _add_synthesis_edges(
        self,
        source_ids: List[str],
        target_ids: List[str],
    ) -> None:
        """
        Add synthesis edges connecting inputs to outputs.

        Args:
            source_ids: IDs of source nodes (inputs to synthesis)
            target_ids: IDs of target nodes (synthesis outputs)
        """
        for target_id in target_ids:
            target_node = self._nodes[target_id]

            for source_id in source_ids:
                source_node = self._nodes[source_id]

                # All source nodes contribute to synthesis with base weight
                base_weight = source_node.snr_score * 0.8

                self._create_edge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=GoTEdgeType.SYNTHESIS,
                    weight=base_weight,
                    metadata={
                        "synthesis_contribution": True,
                        "source_snr": source_node.snr_score,
                    },
                )

        synthesis_count = sum(
            1 for e in self._edges.values()
            if e.edge_type == GoTEdgeType.SYNTHESIS
        )
        logger.debug(f"Added {synthesis_count} synthesis edges")

    # =========================================================================
    # BFS SYNTHESIS
    # =========================================================================

    def _bfs_synthesis(self, root_id: str) -> SynthesisResult:
        """
        Perform BFS traversal with SNR pruning for synthesis.

        Traverses the graph from the root node using breadth-first search,
        pruning nodes below the SNR threshold and aggregating content.

        Args:
            root_id: ID of the root node to start traversal

        Returns:
            SynthesisResult with aggregated content and metrics
        """
        self._total_syntheses += 1

        if root_id not in self._nodes:
            raise ValueError(f"Root node {root_id} not found in graph")

        root_node = self._nodes[root_id]

        # BFS state
        visited: Set[str] = set()
        traversal_path: List[str] = []
        content_parts: List[str] = []
        domains_covered: Set[str] = set()
        snr_scores: List[float] = []

        nodes_pruned = 0
        nodes_vetoed = 0
        max_depth_reached = 0

        # BFS queue: (node_id, current_depth)
        queue: deque[Tuple[str, int]] = deque()
        queue.append((root_id, 0))

        while queue:
            node_id, depth = queue.popleft()

            # Skip if already visited
            if node_id in visited:
                continue

            # Depth limit check
            if depth > self.max_depth:
                continue

            visited.add(node_id)
            node = self._nodes[node_id]

            # Track max depth
            max_depth_reached = max(max_depth_reached, depth)

            # Veto check
            if node_id in self._vetoed_nodes or node.has_veto():
                nodes_vetoed += 1
                logger.debug(f"Node {node_id[:8]}... vetoed at depth {depth}")
                continue

            # SNR pruning
            if node.snr_score < self.snr_threshold:
                nodes_pruned += 1
                logger.debug(
                    f"Node {node_id[:8]}... pruned: SNR {node.snr_score:.4f} < {self.snr_threshold}"
                )
                continue

            # Add to traversal
            traversal_path.append(node_id)
            content_parts.append(node.content)
            snr_scores.append(node.snr_score)

            # Track domain coverage
            if node.domain:
                domains_covered.add(node.domain)

            # Enqueue children (via outgoing edges)
            outgoing_edge_ids = self._outgoing.get(node_id, [])
            for edge_id in outgoing_edge_ids:
                edge = self._edges[edge_id]
                target_id = edge.target_id

                if target_id not in visited:
                    queue.append((target_id, depth + 1))

            # Also enqueue connected nodes via other relationships
            incoming_edge_ids = self._incoming.get(node_id, [])
            for edge_id in incoming_edge_ids:
                edge = self._edges[edge_id]
                # For consensus and cross-domain, traverse source too
                if edge.edge_type in (GoTEdgeType.CONSENSUS, GoTEdgeType.CROSS_DOMAIN):
                    source_id = edge.source_id
                    if source_id not in visited:
                        queue.append((source_id, depth + 1))

        # Compute synthesis SNR
        soup_nodes = [self._nodes[nid] for nid in traversal_path]
        final_snr, diversity_bonus = self._compute_synthesis_snr(soup_nodes)
        total_snr = min(1.0, final_snr + diversity_bonus)

        # Synthesize content
        synthesized_content = self._synthesize_content(content_parts, domains_covered)

        result = SynthesisResult(
            root_id=root_id,
            synthesized_content=synthesized_content,
            final_snr=final_snr,
            diversity_bonus=diversity_bonus,
            total_snr=total_snr,
            nodes_visited=len(visited),
            nodes_pruned=nodes_pruned,
            nodes_vetoed=nodes_vetoed,
            domains_covered=domains_covered,
            traversal_path=traversal_path,
            depth_reached=max_depth_reached,
            metadata={
                "root_domain": root_node.domain,
                "snr_scores": snr_scores,
            },
        )

        logger.debug(
            f"BFS synthesis from {root_id[:8]}...: visited={len(visited)}, "
            f"pruned={nodes_pruned}, vetoed={nodes_vetoed}, snr={total_snr:.4f}"
        )

        return result

    def _compute_synthesis_snr(
        self,
        soup_nodes: List[GoTNode],
    ) -> Tuple[float, float]:
        """
        Compute weighted SNR with diversity bonus for synthesis.

        Calculates the final SNR score based on:
        1. Weighted average of node SNR scores by type
        2. Diversity bonus for multi-domain coverage

        Args:
            soup_nodes: List of nodes included in synthesis

        Returns:
            Tuple of (base_snr, diversity_bonus)
        """
        if not soup_nodes:
            return 0.0, 0.0

        # Compute weighted SNR by node type
        type_scores: Dict[str, List[float]] = {
            t.value: [] for t in GoTNodeType
        }

        for node in soup_nodes:
            type_scores[node.node_type.value].append(node.snr_score)

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for node_type, scores in type_scores.items():
            if scores:
                weight = NODE_TYPE_WEIGHTS.get(node_type, 0.25)
                avg_score = sum(scores) / len(scores)
                weighted_sum += weight * avg_score
                total_weight += weight

        base_snr = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate diversity bonus
        domains = set(
            node.domain for node in soup_nodes
            if node.domain is not None
        )

        # Diversity bonus: 0.02 per domain beyond the first
        domain_count = len(domains)
        diversity_bonus = max(0, (domain_count - 1)) * self.diversity_bonus

        # Cap diversity bonus at 0.05 (to avoid exceeding 1.0 total)
        diversity_bonus = min(diversity_bonus, 0.05)

        logger.debug(
            f"Synthesis SNR: base={base_snr:.4f}, "
            f"diversity={diversity_bonus:.4f} ({domain_count} domains)"
        )

        return base_snr, diversity_bonus

    # =========================================================================
    # NODE CREATION HELPERS
    # =========================================================================

    def _create_persona_soup_node(
        self,
        domain: TaskDomain,
        task: str,
    ) -> GoTNode:
        """
        Create a persona soup node for a domain.

        Args:
            domain: The task domain
            task: The task description for context

        Returns:
            Created GoTNode
        """
        node_id = self._generate_id("persona")

        # Generate persona content from domain
        content = f"[{domain.name}] Domain expertise for: {task[:100]}"
        if domain.persona_ids:
            content += f" ({len(domain.persona_ids)} personas)"

        # SNR based on domain relevance
        snr_score = domain.relevance_score * 0.95 + 0.05  # Ensure minimum 0.05
        snr_score = min(snr_score, 1.0)

        node = GoTNode(
            node_id=node_id,
            node_type=GoTNodeType.PERSONA_SOUP,
            content=content,
            snr_score=snr_score,
            domain=domain.name,
            depth=0,
            metadata={
                "domain_id": domain.domain_id,
                "cluster_id": domain.cluster_id,
                "persona_count": len(domain.persona_ids),
                "relevance": domain.relevance_score,
            },
        )

        self._add_node(node)

        # Track domain membership
        if domain.name not in self._domain_nodes:
            self._domain_nodes[domain.name] = set()
        self._domain_nodes[domain.name].add(node_id)

        return node

    def _create_evidence_node(self, evidence: Dict[str, Any]) -> GoTNode:
        """
        Create an evidence node from evidence data.

        Args:
            evidence: Dictionary with evidence data

        Returns:
            Created GoTNode
        """
        node_id = self._generate_id("evidence")

        content = evidence.get("content", evidence.get("text", str(evidence)))
        source = evidence.get("source", "unknown")
        confidence = evidence.get("confidence", 0.8)

        snr_score = min(1.0, confidence * 0.9 + 0.1)

        node = GoTNode(
            node_id=node_id,
            node_type=GoTNodeType.EVIDENCE,
            content=f"[Evidence from {source}] {content}",
            snr_score=snr_score,
            depth=1,
            metadata={
                "source": source,
                "confidence": confidence,
                "original_evidence": evidence,
            },
        )

        self._add_node(node)
        return node

    def _create_veto_gates(
        self,
        task: str,
        domains: List[TaskDomain],
    ) -> List[str]:
        """
        Create veto gate nodes for safety and ethics.

        Args:
            task: The task description
            domains: List of domains

        Returns:
            List of created veto gate node IDs
        """
        veto_gate_ids: List[str] = []

        # Security veto gate
        security_gate = GoTNode(
            node_id=self._generate_id("veto_security"),
            node_type=GoTNodeType.VETO_GATE,
            content="Security validation gate",
            snr_score=1.0,  # Veto gates always have max SNR
            depth=0,
            metadata={"gate_type": "security"},
            veto_active=self._check_security_veto(task),
            veto_reason="Security constraint triggered" if self._check_security_veto(task) else None,
        )
        self._add_node(security_gate)
        veto_gate_ids.append(security_gate.node_id)

        # Ethics veto gate
        ethics_gate = GoTNode(
            node_id=self._generate_id("veto_ethics"),
            node_type=GoTNodeType.VETO_GATE,
            content="Ethics validation gate",
            snr_score=1.0,
            depth=0,
            metadata={"gate_type": "ethics"},
            veto_active=self._check_ethics_veto(task),
            veto_reason="Ethics constraint triggered" if self._check_ethics_veto(task) else None,
        )
        self._add_node(ethics_gate)
        veto_gate_ids.append(ethics_gate.node_id)

        # Compliance veto gate
        compliance_gate = GoTNode(
            node_id=self._generate_id("veto_compliance"),
            node_type=GoTNodeType.VETO_GATE,
            content="Compliance validation gate",
            snr_score=1.0,
            depth=0,
            metadata={"gate_type": "compliance"},
            veto_active=False,  # Compliance is task-specific
        )
        self._add_node(compliance_gate)
        veto_gate_ids.append(compliance_gate.node_id)

        return veto_gate_ids

    def _create_pareto_synthesis_node(
        self,
        pareto: ParetoSolution,
    ) -> GoTNode:
        """
        Create a synthesis node from a Pareto solution.

        Args:
            pareto: The Pareto-optimal solution

        Returns:
            Created GoTNode
        """
        node_id = self._generate_id("synthesis")

        # SNR based on Pareto optimality
        base_snr = 0.9 if pareto.is_pareto_optimal() else 0.75

        # Adjust by objective scores
        if pareto.objectives:
            avg_objective = sum(pareto.objectives.values()) / len(pareto.objectives)
            snr_score = base_snr * 0.8 + avg_objective * 0.2
        else:
            snr_score = base_snr

        snr_score = min(1.0, max(0.0, snr_score))

        # Generate content from objectives
        obj_summary = ", ".join(
            f"{k}={v:.2f}" for k, v in list(pareto.objectives.items())[:5]
        )
        content = f"[Pareto Synthesis] Objectives: {obj_summary}"

        node = GoTNode(
            node_id=node_id,
            node_type=GoTNodeType.SYNTHESIS,
            content=content,
            snr_score=snr_score,
            depth=2,
            metadata={
                "solution_id": pareto.solution_id,
                "is_pareto_optimal": pareto.is_pareto_optimal(),
                "objectives": pareto.objectives,
                "dominated_by": pareto.dominated_by,
                "dominates": pareto.dominates,
            },
        )

        self._add_node(node)
        return node

    # =========================================================================
    # GRAPH MANAGEMENT
    # =========================================================================

    def _add_node(self, node: GoTNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        self._outgoing[node.node_id] = []
        self._incoming[node.node_id] = []

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: GoTEdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoTEdge:
        """
        Create and add an edge to the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            weight: Edge weight
            metadata: Additional metadata

        Returns:
            Created GoTEdge
        """
        edge_id = self._generate_id("edge")

        edge = GoTEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            metadata=metadata or {},
        )

        self._edges[edge_id] = edge
        self._outgoing[source_id].append(edge_id)
        self._incoming[target_id].append(edge_id)

        # Update node parent/child relationships
        if target_id in self._nodes and source_id not in self._nodes[target_id].parent_ids:
            self._nodes[target_id].parent_ids.append(source_id)
        if source_id in self._nodes and target_id not in self._nodes[source_id].children_ids:
            self._nodes[source_id].children_ids.append(target_id)

        return edge

    def _reset_graph(self) -> None:
        """Reset graph state for a new build."""
        self._nodes.clear()
        self._edges.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._domain_nodes.clear()
        self._vetoed_nodes.clear()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix."""
        unique = uuid.uuid4().hex[:12]
        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        return f"{prefix}_{timestamp}_{unique}"

    def _compute_cross_domain_weight(
        self,
        node1: GoTNode,
        node2: GoTNode,
    ) -> float:
        """
        Compute cross-domain edge weight based on complementarity.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Weight between 0.0 and 1.0
        """
        # Base weight from SNR scores
        base_weight = (node1.snr_score + node2.snr_score) / 2

        # Boost for different domains (complementarity)
        if node1.domain and node2.domain and node1.domain != node2.domain:
            base_weight *= 1.2

        return min(1.0, base_weight)

    def _compute_consensus_weight(
        self,
        node1: GoTNode,
        node2: GoTNode,
    ) -> float:
        """
        Compute consensus edge weight based on agreement.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Weight between 0.0 and 1.0
        """
        # Simple content overlap heuristic
        words1 = set(node1.content.lower().split())
        words2 = set(node2.content.lower().split())

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or"}
        words1 -= stop_words
        words2 -= stop_words

        if not words1 or not words2:
            return 0.5  # Default consensus

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0.0

        # Weight also considers SNR
        snr_factor = (node1.snr_score + node2.snr_score) / 2

        return min(1.0, similarity * 0.6 + snr_factor * 0.4)

    def _check_security_veto(self, task: str) -> bool:
        """
        Check if task triggers security veto.

        Args:
            task: Task description

        Returns:
            True if security veto should be active
        """
        security_patterns = [
            "ignore previous",
            "system prompt",
            "bypass",
            "inject",
            "execute code",
            "rm -rf",
            "drop table",
            "delete all",
            "<script>",
            "eval(",
        ]

        task_lower = task.lower()
        return any(pattern in task_lower for pattern in security_patterns)

    def _check_ethics_veto(self, task: str) -> bool:
        """
        Check if task triggers ethics veto.

        Args:
            task: Task description

        Returns:
            True if ethics veto should be active
        """
        ethics_patterns = [
            "harm",
            "illegal",
            "exploit",
            "discriminate",
            "hate speech",
            "violence",
            "weapon",
            "drug synthesis",
        ]

        task_lower = task.lower()
        return any(pattern in task_lower for pattern in ethics_patterns)

    def _synthesize_content(
        self,
        content_parts: List[str],
        domains: Set[str],
    ) -> str:
        """
        Synthesize final content from parts.

        Args:
            content_parts: List of content strings
            domains: Set of domains covered

        Returns:
            Synthesized content string
        """
        if not content_parts:
            return "[No content synthesized]"

        # Build synthesis header
        domain_list = ", ".join(sorted(domains)) if domains else "general"
        header = f"[Multi-Domain Synthesis: {domain_list}]\n\n"

        # Deduplicate and join content
        seen_content: Set[str] = set()
        unique_parts: List[str] = []

        for part in content_parts:
            # Normalize for deduplication
            normalized = part.strip().lower()
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_parts.append(part.strip())

        body = "\n\n".join(unique_parts[:10])  # Limit to 10 parts

        return header + body

    def _compute_final_snr(
        self,
        synthesis_results: List[SynthesisResult],
    ) -> float:
        """
        Compute final SNR across all synthesis results.

        Args:
            synthesis_results: List of synthesis results

        Returns:
            Final SNR score
        """
        if not synthesis_results:
            return 0.0

        # Weighted average based on nodes visited
        total_weight = 0.0
        weighted_sum = 0.0

        for result in synthesis_results:
            weight = result.nodes_visited
            weighted_sum += result.total_snr * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "snr_threshold": self.snr_threshold,
            "max_depth": self.max_depth,
            "diversity_bonus": self.diversity_bonus,
            "total_graphs_built": self._total_graphs_built,
            "total_syntheses": self._total_syntheses,
            "total_vetoes": self._total_vetoes,
            "current_graph": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "domain_count": len(self._domain_nodes),
                "vetoed_nodes": len(self._vetoed_nodes),
            },
        }

    def get_node(self, node_id: str) -> Optional[GoTNode]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            GoTNode or None if not found
        """
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[GoTEdge]:
        """
        Get an edge by ID.

        Args:
            edge_id: Edge ID

        Returns:
            GoTEdge or None if not found
        """
        return self._edges.get(edge_id)

    def get_all_nodes(self) -> List[GoTNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())

    def get_all_edges(self) -> List[GoTEdge]:
        """Get all edges in the graph."""
        return list(self._edges.values())


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_got_engine(
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    max_depth: int = DEFAULT_MAX_DEPTH,
    diversity_bonus: float = DEFAULT_DIVERSITY_BONUS,
) -> GraphOfThoughtsEngine:
    """
    Factory function to create a configured Graph of Thoughts engine.

    Args:
        snr_threshold: Minimum SNR score for node inclusion (default: 0.95)
        max_depth: Maximum traversal depth (default: 5)
        diversity_bonus: Bonus per additional domain (default: 0.02)

    Returns:
        Configured GraphOfThoughtsEngine instance

    Example:
        >>> engine = create_got_engine(snr_threshold=0.90)
        >>> result = engine.build_reasoning_graph(
        ...     task="Analyze market trends",
        ...     task_domains=[
        ...         TaskDomain(domain_id="d1", name="Finance", cluster_id="c1"),
        ...         TaskDomain(domain_id="d2", name="Economics", cluster_id="c2"),
        ...     ],
        ... )
    """
    return GraphOfThoughtsEngine(
        snr_threshold=snr_threshold,
        max_depth=max_depth,
        diversity_bonus=diversity_bonus,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Core classes
    "GraphOfThoughtsEngine",
    "GoTNode",
    "GoTEdge",
    # Supporting classes
    "TaskDomain",
    "ParetoSolution",
    "SynthesisResult",
    "GoTGraphResult",
    # Enums
    "GoTNodeType",
    "GoTEdgeType",
    "GoTTraversalStatus",
    # Factory function
    "create_got_engine",
    # Constants
    "DEFAULT_SNR_THRESHOLD",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_DIVERSITY_BONUS",
    "DOMAIN_PREFIX",
]


# =============================================================================
# DEMO / TESTING
# =============================================================================


def main() -> None:
    """Demo the Graph of Thoughts engine."""
    print("=" * 80)
    print("BIZRA Graph of Thoughts Engine")
    print("=" * 80)

    # Create engine
    engine = create_got_engine(
        snr_threshold=0.95,
        max_depth=5,
        diversity_bonus=0.02,
    )

    print("\n1. Engine Configuration:")
    stats = engine.get_statistics()
    print(f"   SNR Threshold: {stats['snr_threshold']}")
    print(f"   Max Depth: {stats['max_depth']}")
    print(f"   Diversity Bonus: {stats['diversity_bonus']}")

    # Create test domains
    domains = [
        TaskDomain(
            domain_id="d1",
            name="Data Engineering",
            cluster_id="c1",
            persona_ids=["p1", "p2", "p3"],
            relevance_score=0.95,
        ),
        TaskDomain(
            domain_id="d2",
            name="Machine Learning",
            cluster_id="c2",
            persona_ids=["p4", "p5"],
            relevance_score=0.90,
        ),
        TaskDomain(
            domain_id="d3",
            name="Systems Architecture",
            cluster_id="c3",
            persona_ids=["p6", "p7", "p8", "p9"],
            relevance_score=0.85,
        ),
    ]

    # Create test Pareto solutions
    pareto_solutions = [
        ParetoSolution(
            solution_id="ps1",
            objectives={
                "latency": 0.92,
                "throughput": 0.88,
                "cost": 0.85,
            },
            dominated_by=0,
            dominates=2,
        ),
        ParetoSolution(
            solution_id="ps2",
            objectives={
                "latency": 0.85,
                "throughput": 0.95,
                "cost": 0.80,
            },
            dominated_by=0,
            dominates=1,
        ),
    ]

    # Create test evidence
    evidence = [
        {
            "content": "Performance benchmarks show 40% improvement with batch processing",
            "source": "Internal Testing",
            "confidence": 0.92,
        },
        {
            "content": "Industry best practices recommend async I/O for high throughput",
            "source": "Technical Report",
            "confidence": 0.88,
        },
    ]

    print("\n2. Building Reasoning Graph:")
    print(f"   Domains: {len(domains)}")
    print(f"   Pareto Solutions: {len(pareto_solutions)}")
    print(f"   Evidence Items: {len(evidence)}")

    # Build reasoning graph
    result = engine.build_reasoning_graph(
        task="Optimize BIZRA data pipeline for maximum throughput while maintaining low latency",
        task_domains=domains,
        pareto_solutions=pareto_solutions,
        evidence=evidence,
        enable_veto_gates=True,
    )

    print("\n3. Graph Results:")
    print(f"   Graph ID: {result.graph_id[:16]}...")
    print(f"   Nodes: {result.node_count}")
    print(f"   Edges: {result.edge_count}")
    print(f"   Root Nodes: {len(result.root_ids)}")
    print(f"   Final SNR: {result.final_snr:.4f}")
    print(f"   Domains Used: {', '.join(result.domains_used)}")
    print(f"   Vetoed: {result.vetoed}")
    print(f"   Latency: {result.latency_ms:.2f}ms")

    print("\n4. Synthesis Results:")
    for i, synthesis in enumerate(result.synthesis_results):
        print(f"\n   Synthesis {i + 1} (root: {synthesis.root_id[:8]}...):")
        print(f"     - Nodes visited: {synthesis.nodes_visited}")
        print(f"     - Nodes pruned: {synthesis.nodes_pruned}")
        print(f"     - Nodes vetoed: {synthesis.nodes_vetoed}")
        print(f"     - Domains covered: {len(synthesis.domains_covered)}")
        print(f"     - Final SNR: {synthesis.final_snr:.4f}")
        print(f"     - Diversity Bonus: {synthesis.diversity_bonus:.4f}")
        print(f"     - Total SNR: {synthesis.total_snr:.4f}")
        print(f"     - Depth reached: {synthesis.depth_reached}")

    print("\n5. Node Distribution:")
    node_types = {}
    for node in engine.get_all_nodes():
        node_type = node.node_type.value
        node_types[node_type] = node_types.get(node_type, 0) + 1
    for node_type, count in sorted(node_types.items()):
        print(f"   {node_type}: {count}")

    print("\n6. Edge Distribution:")
    edge_types = {}
    for edge in engine.get_all_edges():
        edge_type = edge.edge_type.value
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    for edge_type, count in sorted(edge_types.items()):
        print(f"   {edge_type}: {count}")

    print("\n7. Final Statistics:")
    final_stats = engine.get_statistics()
    print(f"   Total Graphs Built: {final_stats['total_graphs_built']}")
    print(f"   Total Syntheses: {final_stats['total_syntheses']}")
    print(f"   Total Vetoes: {final_stats['total_vetoes']}")

    print("\n" + "=" * 80)
    print("Graph of Thoughts Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
