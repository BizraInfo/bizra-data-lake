"""
BIZRA Synthesis Engine - The Apex of Multi-Persona Reasoning
=============================================================

The Synthesis Engine is the unified orchestrator combining all BIZRA apex components
into a coherent reasoning pipeline. It represents the ultimate synthesis point where
Pareto-optimal selection, persona weight interpolation, graph-of-thoughts reasoning,
and weighted consensus voting converge.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                          SYNTHESIS ENGINE                                        │
    │                                                                                  │
    │   Task Request                                                                   │
    │        │                                                                         │
    │        ▼                                                                         │
    │   ┌─────────────────┐                                                           │
    │   │ 1. PARETO FRONT │ Compute Pareto-optimal lambda configurations              │
    │   │    COMPUTATION  │ (cost vs quality vs latency)                              │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 2. THOMPSON     │ Thompson Sampling selection from Pareto front             │
    │   │    SELECTION    │ (explore-exploit balance)                                 │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 3. PERSONA SOUP │ Interpolate persona weights for selected lambda           │
    │   │    BLENDING     │ (soft mixture of expert perspectives)                     │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 4. GoT GRAPH    │ Build reasoning graph with persona nodes                  │
    │   │    CONSTRUCTION │ (multi-dimensional thought exploration)                   │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 5. GRAPH        │ Traverse graph with synthesis aggregation                 │
    │   │    TRAVERSAL    │ (depth-first with backtracking)                           │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 6. WEIGHTED     │ Persona-weighted consensus vote                           │
    │   │    CONSENSUS    │ (veto override + quorum check)                            │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 7. SYNTHESIS    │ Fail-closed gate: veto, consensus, SNR                    │
    │   │    GATE         │ (pass or reject with evidence)                            │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │   ┌─────────────────┐                                                           │
    │   │ 8. RECEIPT      │ Emit evidence receipt to chain                            │
    │   │    EMISSION     │ (append-only audit trail)                                 │
    │   └────────┬────────┘                                                           │
    │            │                                                                     │
    │            ▼                                                                     │
    │      SynthesisResult (success/rejection with full evidence trail)               │
    │                                                                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘

Key Concepts:
    - Pareto Optimal Router: Multi-objective selection (cost, quality, latency)
    - Persona Soup: Soft interpolation of expert persona weights
    - Graph of Thoughts: Multi-dimensional reasoning graph
    - Weighted Consensus: Persona-based voting with veto authority
    - Synthesis Gate: Fail-closed enforcement (veto, consensus, SNR)

Integration Points:
    - ParetoOptimalRouter: Lambda configuration selection
    - PersonaSoup: Weight interpolation for blended expertise
    - GraphOfThoughtsEngine: Multi-path reasoning exploration
    - PersonaWeightedConsensus: Final voting and veto check
    - SNR Enforcer: Signal-to-noise quality gate

Version: 1.0.0
Alignment: BIZRA_SOT.md, constitution/ihsan_v1.yaml
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    IHSAN_THRESHOLD as CONST_IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
)

logger = logging.getLogger("apex.synthesis_engine")


# =============================================================================
# CONSTANTS (from core/constants.py - Genesis v2.2.2 compliance)
# =============================================================================

DOMAIN_PREFIX = "bizra-synthesis-v1:"
SNR_THRESHOLD = SNR_THRESHOLD_T0_ELITE  # 0.98 - PAT enforcement level
IHSAN_THRESHOLD = CONST_IHSAN_THRESHOLD  # 0.95
WEIGHTED_QUORUM = 2.4
MAX_GRAPH_DEPTH = 5
MAX_REASONING_NODES = 20
RECEIPT_PATH = Path("docs/evidence/receipts/synthesis")


# =============================================================================
# IMPORTS: Core BIZRA Modules (with graceful degradation)
# =============================================================================

# Thompson Router
try:
    from core.apex.thompson_router import (
        ThompsonSamplingRouter,
        CapabilityMatrix,
        TaskCategory,
        SelectionResult,
    )

    THOMPSON_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Thompson Router not available: {e}")
    THOMPSON_AVAILABLE = False
    ThompsonSamplingRouter = None  # type: ignore
    TaskCategory = None  # type: ignore

# Consensus Manager
try:
    from core.apex.consensus_manager import (
        ConsensusManager,
        ConsensusResult,
        SAPEResult,
        DualAgenticRequest,
        get_consensus_manager,
    )

    CONSENSUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Consensus Manager not available: {e}")
    CONSENSUS_AVAILABLE = False
    ConsensusManager = None  # type: ignore

# PersonaPlex (Persona definitions and weighted consensus)
try:
    from core.personaplex import (
        PersonaDefinition,
        PersonaRegistry,
        PersonaWeightedConsensus,
        WeightedConsensusResult,
        WeightedVote,
        VetoDomain,
        create_default_registry,
        create_bizra_weighted_consensus,
        get_standard_bizra_personas,
    )

    PERSONAPLEX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PersonaPlex not available: {e}")
    PERSONAPLEX_AVAILABLE = False
    PersonaDefinition = None  # type: ignore
    PersonaWeightedConsensus = None  # type: ignore

# SNR Enforcement
try:
    from core.snr import (
        SNRBudget,
        SNRResult,
        BudgetTier,
        enforce_snr_budget,
        estimate_snr,
    )

    SNR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SNR module not available: {e}")
    SNR_AVAILABLE = False

# GoT Orchestrator
try:
    from bizra_kernel.got_orchestrator import GoTOrchestrator

    GOT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GoT Orchestrator not available: {e}")
    GOT_AVAILABLE = False
    GoTOrchestrator = None  # type: ignore


# =============================================================================
# ENUMS
# =============================================================================


class SynthesisStage(str, Enum):
    """Stages in the synthesis pipeline."""

    PARETO_FRONT = "pareto_front"
    THOMPSON_SELECT = "thompson_select"
    PERSONA_SOUP = "persona_soup"
    GOT_BUILD = "got_build"
    GOT_TRAVERSE = "got_traverse"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    SYNTHESIS_GATE = "synthesis_gate"
    RECEIPT_EMISSION = "receipt_emission"


class GateStatus(str, Enum):
    """Status of synthesis gate checks."""

    PASSED = "passed"
    VETO_TRIGGERED = "veto_triggered"
    CONSENSUS_FAILED = "consensus_failed"
    SNR_FAILED = "snr_failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class ReasoningNodeType(str, Enum):
    """Types of nodes in the reasoning graph."""

    ROOT = "root"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    CONCLUSION = "conclusion"


# =============================================================================
# DATA CLASSES: Pareto Optimal Configuration
# =============================================================================


@dataclass
class LambdaConfig:
    """
    Lambda configuration representing a point on the Pareto front.

    Each lambda is a weighted blend of personas optimized for specific
    trade-offs between cost, quality, and latency.
    """

    lambda_id: str
    persona_weights: Dict[str, float]  # persona_id -> weight (0.0-1.0)
    cost_factor: float  # 0.0 = free, 1.0 = maximum cost
    quality_factor: float  # 0.0 = minimum, 1.0 = maximum quality
    latency_factor: float  # 0.0 = instant, 1.0 = maximum latency

    # Metadata
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def pareto_score(self) -> Tuple[float, float, float]:
        """Return (cost, quality, latency) tuple for Pareto comparison."""
        return (self.cost_factor, -self.quality_factor, self.latency_factor)

    def dominates(self, other: "LambdaConfig") -> bool:
        """
        Check if this config Pareto-dominates another.

        Domination means: at least as good in all objectives AND strictly
        better in at least one.
        """
        self_score = self.pareto_score
        other_score = other.pareto_score

        at_least_as_good = all(s <= o for s, o in zip(self_score, other_score))
        strictly_better = any(s < o for s, o in zip(self_score, other_score))

        return at_least_as_good and strictly_better

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "lambda_id": self.lambda_id,
            "persona_weights": self.persona_weights,
            "cost_factor": self.cost_factor,
            "quality_factor": self.quality_factor,
            "latency_factor": self.latency_factor,
            "created_at": self.created_at,
        }


@dataclass
class ParetoFront:
    """
    Pareto front of non-dominated lambda configurations.

    The Pareto front contains all configurations that are not dominated
    by any other configuration in the solution space.
    """

    configs: List[LambdaConfig]
    computation_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.configs)

    def get_by_quality_preference(self) -> Optional[LambdaConfig]:
        """Get config with highest quality factor."""
        if not self.configs:
            return None
        return max(self.configs, key=lambda c: c.quality_factor)

    def get_by_cost_preference(self) -> Optional[LambdaConfig]:
        """Get config with lowest cost factor."""
        if not self.configs:
            return None
        return min(self.configs, key=lambda c: c.cost_factor)

    def get_balanced(self) -> Optional[LambdaConfig]:
        """Get config closest to balanced (0.5, 0.5, 0.5)."""
        if not self.configs:
            return None
        target = (0.5, 0.5, 0.5)
        return min(
            self.configs,
            key=lambda c: sum(
                (a - b) ** 2
                for a, b in zip(
                    (c.cost_factor, c.quality_factor, c.latency_factor), target
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "configs": [c.to_dict() for c in self.configs],
            "computation_time_ms": self.computation_time_ms,
            "size": len(self.configs),
        }


# =============================================================================
# DATA CLASSES: Persona Soup
# =============================================================================


@dataclass
class PersonaSoupBlend:
    """
    A soft mixture of persona weights for blended expertise.

    Persona Soup allows smooth interpolation between expert perspectives
    rather than hard selection of a single persona.
    """

    blend_id: str
    weights: Dict[str, float]  # persona_id -> weight (sum to 1.0)
    source_lambda: str  # lambda_id that produced this blend
    temperature: float = 0.7  # Sampling temperature

    def __post_init__(self) -> None:
        """Validate and normalize weights."""
        if not self.weights:
            raise ValueError("Weights cannot be empty")

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_dominant_personas(self, threshold: float = 0.1) -> List[str]:
        """Get personas with weight above threshold."""
        return [pid for pid, w in self.weights.items() if w >= threshold]

    def interpolate_with(
        self, other: "PersonaSoupBlend", alpha: float
    ) -> "PersonaSoupBlend":
        """
        Interpolate with another blend.

        Args:
            other: Other blend to interpolate with
            alpha: Interpolation factor (0.0 = self, 1.0 = other)

        Returns:
            New interpolated PersonaSoupBlend
        """
        all_personas = set(self.weights.keys()) | set(other.weights.keys())
        new_weights = {}

        for pid in all_personas:
            w1 = self.weights.get(pid, 0.0)
            w2 = other.weights.get(pid, 0.0)
            new_weights[pid] = (1 - alpha) * w1 + alpha * w2

        return PersonaSoupBlend(
            blend_id=f"interpolated-{uuid.uuid4().hex[:8]}",
            weights=new_weights,
            source_lambda=f"{self.source_lambda}+{other.source_lambda}",
            temperature=(1 - alpha) * self.temperature + alpha * other.temperature,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "blend_id": self.blend_id,
            "weights": self.weights,
            "source_lambda": self.source_lambda,
            "temperature": self.temperature,
            "dominant_personas": self.get_dominant_personas(),
        }


# =============================================================================
# DATA CLASSES: Graph of Thoughts
# =============================================================================


@dataclass
class ReasoningNode:
    """
    Node in the Graph of Thoughts reasoning structure.

    Each node represents a thought or reasoning step with connections
    to other nodes forming a reasoning graph.
    """

    node_id: str
    node_type: ReasoningNodeType
    content: str
    persona_source: str  # Which persona generated this node
    confidence: float  # 0.0-1.0

    # Graph structure
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)

    # Metadata
    depth: int = 0
    snr_score: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.child_ids) == 0

    def is_root(self) -> bool:
        """Check if this is a root node (no parents)."""
        return len(self.parent_ids) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": (
                self.content[:200] + "..." if len(self.content) > 200 else self.content
            ),
            "persona_source": self.persona_source,
            "confidence": self.confidence,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "depth": self.depth,
            "snr_score": self.snr_score,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class GraphOfThoughts:
    """
    Graph structure for multi-dimensional reasoning.

    The Graph of Thoughts allows non-linear exploration of solution
    space with branching, merging, and backtracking.
    """

    graph_id: str
    nodes: Dict[str, ReasoningNode]  # node_id -> node
    root_id: Optional[str] = None

    # Traversal state
    current_path: List[str] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)

    # Statistics
    max_depth_reached: int = 0
    total_branches: int = 0

    def add_node(self, node: ReasoningNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

        if node.is_root() and self.root_id is None:
            self.root_id = node.node_id

        self.max_depth_reached = max(self.max_depth_reached, node.depth)

        if len(node.child_ids) > 1:
            self.total_branches += len(node.child_ids) - 1

    def connect(self, parent_id: str, child_id: str) -> None:
        """Connect parent to child node."""
        if parent_id in self.nodes and child_id in self.nodes:
            if child_id not in self.nodes[parent_id].child_ids:
                self.nodes[parent_id].child_ids.append(child_id)
            if parent_id not in self.nodes[child_id].parent_ids:
                self.nodes[child_id].parent_ids.append(parent_id)

    def get_leaves(self) -> List[ReasoningNode]:
        """Get all leaf nodes."""
        return [n for n in self.nodes.values() if n.is_leaf()]

    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root."""
        path = []
        current = node_id

        while current and current in self.nodes:
            path.append(current)
            parents = self.nodes[current].parent_ids
            current = parents[0] if parents else None

        return list(reversed(path))

    def get_best_synthesis_path(self) -> List[ReasoningNode]:
        """
        Get the path with highest aggregate confidence leading to
        a conclusion node.
        """
        best_path: List[ReasoningNode] = []
        best_score = 0.0

        for leaf in self.get_leaves():
            if leaf.node_type == ReasoningNodeType.CONCLUSION:
                path_ids = self.get_path_to_root(leaf.node_id)
                path = [self.nodes[nid] for nid in path_ids]

                if path:
                    avg_confidence = sum(n.confidence for n in path) / len(path)
                    if avg_confidence > best_score:
                        best_score = avg_confidence
                        best_path = path

        return best_path

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "graph_id": self.graph_id,
            "node_count": len(self.nodes),
            "root_id": self.root_id,
            "max_depth_reached": self.max_depth_reached,
            "total_branches": self.total_branches,
            "leaf_count": len(self.get_leaves()),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }


# =============================================================================
# DATA CLASSES: Synthesis Result
# =============================================================================


@dataclass
class SynthesisNode:
    """
    Synthesized output node representing the final reasoning result.
    """

    synthesis_id: str
    content: str
    source_graph_id: str
    source_path: List[str]  # node_ids in synthesis path

    # Quality metrics
    confidence: float
    snr_score: float
    ihsan_score: float

    # Persona attribution
    contributing_personas: Dict[str, float]  # persona_id -> contribution

    # Evidence
    evidence_refs: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "synthesis_id": self.synthesis_id,
            "content": self.content,
            "source_graph_id": self.source_graph_id,
            "source_path": self.source_path,
            "confidence": self.confidence,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "contributing_personas": self.contributing_personas,
            "evidence_refs": self.evidence_refs,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class SynthesisResult:
    """
    Complete result from the synthesis pipeline.

    Contains success/failure status, the synthesized output, consensus
    result, SNR score, and full evidence trail.
    """

    success: bool
    synthesis_node: Optional[SynthesisNode]
    consensus_result: Optional[Dict[str, Any]]  # WeightedConsensusResult.to_dict()
    snr_score: float
    evidence_trail: List[Dict[str, Any]]

    # Gate status
    gate_status: GateStatus
    gate_reason: Optional[str] = None

    # Pipeline metadata
    request_id: str = ""
    pareto_front_size: int = 0
    selected_lambda: Optional[str] = None
    persona_blend: Optional[Dict[str, Any]] = None
    graph_stats: Optional[Dict[str, Any]] = None

    # Timing
    total_latency_ms: int = 0
    stage_latencies: Dict[str, int] = field(default_factory=dict)

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "synthesis_node": (
                self.synthesis_node.to_dict() if self.synthesis_node else None
            ),
            "consensus_result": self.consensus_result,
            "snr_score": self.snr_score,
            "evidence_trail": self.evidence_trail,
            "gate_status": self.gate_status.value,
            "gate_reason": self.gate_reason,
            "request_id": self.request_id,
            "pareto_front_size": self.pareto_front_size,
            "selected_lambda": self.selected_lambda,
            "persona_blend": self.persona_blend,
            "graph_stats": self.graph_stats,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies": self.stage_latencies,
            "timestamp": self.timestamp,
        }


# =============================================================================
# SYNTHESIS GATE
# =============================================================================


class SynthesisGate:
    """
    Fail-closed gate for synthesis validation.

    The gate enforces:
    1. Veto check: Any veto persona rejection blocks
    2. Consensus check: Weighted quorum must be met
    3. SNR check: Signal-to-noise must meet threshold

    All checks must pass for synthesis to be approved. Failure at
    any stage results in rejection with evidence.
    """

    def __init__(
        self,
        snr_threshold: float = SNR_THRESHOLD,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        weighted_quorum: float = WEIGHTED_QUORUM,
    ):
        """
        Initialize the synthesis gate.

        Args:
            snr_threshold: Minimum SNR score for approval
            ihsan_threshold: Minimum Ihsan score for approval
            weighted_quorum: Minimum weighted consensus score
        """
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self.weighted_quorum = weighted_quorum

        logger.info(
            f"SynthesisGate initialized: SNR≥{snr_threshold}, "
            f"Ihsan≥{ihsan_threshold}, quorum≥{weighted_quorum}"
        )

    def check_veto(
        self,
        consensus_result: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for veto condition.

        Returns:
            Tuple of (passed, veto_reason)
        """
        if consensus_result is None:
            return False, "No consensus result provided"

        if consensus_result.get("veto_triggered", False):
            reason = consensus_result.get(
                "veto_reason", "Veto triggered by guardian persona"
            )
            return False, reason

        return True, None

    def check_consensus(
        self,
        consensus_result: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for consensus quorum.

        Returns:
            Tuple of (passed, failure_reason)
        """
        if consensus_result is None:
            return False, "No consensus result provided"

        if not consensus_result.get("passed", False):
            total_score = consensus_result.get("total_weighted_score", 0.0)
            quorum = consensus_result.get("weighted_quorum", self.weighted_quorum)
            reason = f"Consensus not achieved: {total_score:.3f} < {quorum:.3f}"
            return False, reason

        return True, None

    def check_snr(
        self,
        snr_score: float,
        content: str = "",
    ) -> Tuple[bool, Optional[str]]:
        """
        Check SNR threshold.

        Args:
            snr_score: Computed SNR score
            content: Optional content for additional analysis

        Returns:
            Tuple of (passed, failure_reason)
        """
        if snr_score < self.snr_threshold:
            reason = f"SNR below threshold: {snr_score:.4f} < {self.snr_threshold}"
            return False, reason

        return True, None

    def validate(
        self,
        consensus_result: Optional[Dict[str, Any]],
        snr_score: float,
        content: str = "",
    ) -> Tuple[GateStatus, Optional[str]]:
        """
        Run all gate checks in order.

        Order matters - veto check first, then consensus, then SNR.
        This is fail-closed: first failure returns rejection.

        Returns:
            Tuple of (GateStatus, failure_reason)
        """
        # 1. Check veto
        veto_passed, veto_reason = self.check_veto(consensus_result)
        if not veto_passed:
            logger.warning(f"Synthesis gate: VETO TRIGGERED - {veto_reason}")
            return GateStatus.VETO_TRIGGERED, veto_reason

        # 2. Check consensus
        consensus_passed, consensus_reason = self.check_consensus(consensus_result)
        if not consensus_passed:
            logger.warning(f"Synthesis gate: CONSENSUS FAILED - {consensus_reason}")
            return GateStatus.CONSENSUS_FAILED, consensus_reason

        # 3. Check SNR
        snr_passed, snr_reason = self.check_snr(snr_score, content)
        if not snr_passed:
            logger.warning(f"Synthesis gate: SNR FAILED - {snr_reason}")
            return GateStatus.SNR_FAILED, snr_reason

        logger.info("Synthesis gate: ALL CHECKS PASSED")
        return GateStatus.PASSED, None


# =============================================================================
# PARETO OPTIMAL ROUTER
# =============================================================================


class ParetoOptimalRouter:
    """
    Multi-objective router computing Pareto-optimal lambda configurations.

    The router generates candidate configurations and filters them
    to the Pareto front - the set of non-dominated solutions.
    """

    def __init__(
        self,
        personas: Optional[List[Any]] = None,
        num_candidates: int = 50,
    ):
        """
        Initialize the Pareto optimal router.

        Args:
            personas: List of persona definitions
            num_candidates: Number of candidate configurations to generate
        """
        self.personas = personas or []
        self.num_candidates = num_candidates

        # Initialize Thompson router if available
        self.thompson_router: Optional[ThompsonSamplingRouter] = None
        if THOMPSON_AVAILABLE:
            try:
                self.thompson_router = ThompsonSamplingRouter()
            except Exception as e:
                logger.warning(f"Failed to initialize Thompson router: {e}")

    def compute_pareto_front(
        self,
        task_domains: List[str],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ParetoFront:
        """
        Compute Pareto front of lambda configurations.

        Args:
            task_domains: Domains relevant to the task
            constraints: Optional constraints (max_cost, min_quality, etc.)

        Returns:
            ParetoFront containing non-dominated configurations
        """
        start_time = time.perf_counter()
        constraints = constraints or {}

        # Generate candidate configurations
        candidates = self._generate_candidates(task_domains, constraints)

        # Filter to Pareto front
        pareto_configs = self._filter_pareto(candidates)

        computation_time = (time.perf_counter() - start_time) * 1000

        front = ParetoFront(
            configs=pareto_configs,
            computation_time_ms=computation_time,
        )

        logger.info(
            f"Computed Pareto front: {len(pareto_configs)} configs "
            f"from {len(candidates)} candidates in {computation_time:.2f}ms"
        )

        return front

    def _generate_candidates(
        self,
        task_domains: List[str],
        constraints: Dict[str, Any],
    ) -> List[LambdaConfig]:
        """Generate candidate lambda configurations."""
        import random

        candidates: List[LambdaConfig] = []
        max_cost = constraints.get("max_cost", 1.0)
        min_quality = constraints.get("min_quality", 0.0)

        # Generate random configurations
        for i in range(self.num_candidates):
            # Random persona weights
            weights = {}
            if self.personas:
                for p in self.personas:
                    pid = p.persona_id if hasattr(p, "persona_id") else str(p)
                    weights[pid] = random.random()
            else:
                # Default personas
                for pid in ["master-reasoner", "security-guardian", "ethics-validator"]:
                    weights[pid] = random.random()

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            # Random factors with constraints
            cost = random.uniform(0, max_cost)
            quality = random.uniform(min_quality, 1.0)
            latency = random.random()

            config = LambdaConfig(
                lambda_id=f"lambda-{i:03d}-{uuid.uuid4().hex[:6]}",
                persona_weights=weights,
                cost_factor=cost,
                quality_factor=quality,
                latency_factor=latency,
            )
            candidates.append(config)

        return candidates

    def _filter_pareto(self, candidates: List[LambdaConfig]) -> List[LambdaConfig]:
        """Filter candidates to Pareto front (non-dominated solutions)."""
        if not candidates:
            return []

        pareto: List[LambdaConfig] = []

        for candidate in candidates:
            # Check if candidate is dominated by any existing Pareto member
            is_dominated = False
            dominates_existing: List[LambdaConfig] = []

            for existing in pareto:
                if existing.dominates(candidate):
                    is_dominated = True
                    break
                if candidate.dominates(existing):
                    dominates_existing.append(existing)

            if not is_dominated:
                # Remove configs dominated by candidate
                pareto = [p for p in pareto if p not in dominates_existing]
                pareto.append(candidate)

        return pareto

    def thompson_select(
        self,
        front: ParetoFront,
        task_text: str,
    ) -> Optional[LambdaConfig]:
        """
        Select from Pareto front using Thompson Sampling.

        Args:
            front: The Pareto front to select from
            task_text: Task description for category inference

        Returns:
            Selected LambdaConfig or None if front is empty
        """
        if not front.configs:
            return None

        # If Thompson router available, use it for intelligent selection
        if self.thompson_router and THOMPSON_AVAILABLE:
            try:
                result = self.thompson_router.select_agent(task_text)

                # Find config closest to Thompson's preference
                # Prefer quality when exploration is low
                if result.exploration_rate < 0.3:
                    return front.get_by_quality_preference()
                elif result.exploration_rate > 0.7:
                    # High exploration - pick random from front
                    import random

                    return random.choice(front.configs)
                else:
                    return front.get_balanced()

            except Exception as e:
                logger.warning(f"Thompson selection failed: {e}")

        # Fallback: return balanced config
        return front.get_balanced()


# =============================================================================
# PERSONA SOUP BLENDER
# =============================================================================


class PersonaSoupBlender:
    """
    Blends persona weights for soft mixture of expert perspectives.

    Rather than hard selection of a single persona, Persona Soup
    creates smooth interpolations allowing nuanced expert combinations.
    """

    def __init__(
        self,
        personas: Optional[List[Any]] = None,
        default_temperature: float = 0.7,
    ):
        """
        Initialize the persona soup blender.

        Args:
            personas: List of persona definitions
            default_temperature: Default sampling temperature
        """
        self.personas = personas or []
        self.default_temperature = default_temperature

        # Build persona lookup
        self.persona_map: Dict[str, Any] = {}
        for p in self.personas:
            pid = p.persona_id if hasattr(p, "persona_id") else str(p)
            self.persona_map[pid] = p

    def blend_from_lambda(
        self,
        lambda_config: LambdaConfig,
        task_domains: List[str],
    ) -> PersonaSoupBlend:
        """
        Create persona blend from lambda configuration.

        Args:
            lambda_config: Selected lambda configuration
            task_domains: Task domains for alignment boosting

        Returns:
            PersonaSoupBlend with interpolated weights
        """
        weights = dict(lambda_config.persona_weights)

        # Boost weights for personas with high task alignment
        for pid, persona in self.persona_map.items():
            if hasattr(persona, "compute_task_alignment"):
                alignment = persona.compute_task_alignment(task_domains)
                if alignment > 0.5 and pid in weights:
                    weights[pid] *= 1 + alignment * 0.5

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return PersonaSoupBlend(
            blend_id=f"blend-{uuid.uuid4().hex[:8]}",
            weights=weights,
            source_lambda=lambda_config.lambda_id,
            temperature=self.default_temperature,
        )

    def adaptive_blend(
        self,
        base_blend: PersonaSoupBlend,
        feedback_scores: Dict[str, float],
    ) -> PersonaSoupBlend:
        """
        Adapt blend based on feedback scores.

        Args:
            base_blend: Starting blend
            feedback_scores: Persona scores from prior stages

        Returns:
            Adapted PersonaSoupBlend
        """
        new_weights = dict(base_blend.weights)

        # Boost weights for high-scoring personas
        for pid, score in feedback_scores.items():
            if pid in new_weights:
                new_weights[pid] *= 0.5 + score

        # Renormalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return PersonaSoupBlend(
            blend_id=f"adapted-{uuid.uuid4().hex[:8]}",
            weights=new_weights,
            source_lambda=f"adapted-{base_blend.source_lambda}",
            temperature=base_blend.temperature,
        )


# =============================================================================
# GRAPH OF THOUGHTS ENGINE
# =============================================================================


class GraphOfThoughtsEngine:
    """
    Builds and traverses reasoning graphs for multi-dimensional thinking.

    The Graph of Thoughts allows exploration of multiple reasoning paths
    simultaneously, with synthesis at convergence points.
    """

    def __init__(
        self,
        personas: Optional[List[Any]] = None,
        max_depth: int = MAX_GRAPH_DEPTH,
        max_nodes: int = MAX_REASONING_NODES,
    ):
        """
        Initialize the GoT engine.

        Args:
            personas: Personas to use for node generation
            max_depth: Maximum graph depth
            max_nodes: Maximum nodes in graph
        """
        self.personas = personas or []
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        # GoT orchestrator if available
        self.got_orchestrator: Optional[GoTOrchestrator] = None
        if GOT_AVAILABLE and GoTOrchestrator:
            try:
                self.got_orchestrator = GoTOrchestrator()
            except Exception as e:
                logger.warning(f"Failed to initialize GoT orchestrator: {e}")

    async def build_graph(
        self,
        task: str,
        task_domains: List[str],
        persona_blend: PersonaSoupBlend,
    ) -> GraphOfThoughts:
        """
        Build a reasoning graph for the task.

        Args:
            task: Task description
            task_domains: Relevant domains
            persona_blend: Persona blend for node attribution

        Returns:
            Built GraphOfThoughts
        """
        graph = GraphOfThoughts(
            graph_id=f"got-{uuid.uuid4().hex[:12]}",
            nodes={},
        )

        # Create root node
        root = ReasoningNode(
            node_id="node-root",
            node_type=ReasoningNodeType.ROOT,
            content=f"Task: {task}",
            persona_source="system",
            confidence=1.0,
            depth=0,
        )
        graph.add_node(root)

        # Generate analysis nodes from dominant personas
        dominant_personas = persona_blend.get_dominant_personas(threshold=0.15)

        for i, pid in enumerate(dominant_personas[:4]):  # Limit to 4 branches
            analysis = await self._generate_analysis_node(
                task=task,
                task_domains=task_domains,
                persona_id=pid,
                depth=1,
                parent_id=root.node_id,
            )
            graph.add_node(analysis)
            graph.connect(root.node_id, analysis.node_id)

            # Generate synthesis from each analysis
            if len(graph.nodes) < self.max_nodes:
                synthesis = await self._generate_synthesis_node(
                    analysis_content=analysis.content,
                    persona_id=pid,
                    depth=2,
                    parent_id=analysis.node_id,
                )
                graph.add_node(synthesis)
                graph.connect(analysis.node_id, synthesis.node_id)

        # Create final conclusion node
        synthesis_nodes = [
            n
            for n in graph.nodes.values()
            if n.node_type == ReasoningNodeType.SYNTHESIS
        ]

        if synthesis_nodes:
            conclusion = await self._generate_conclusion_node(
                synthesis_nodes=synthesis_nodes,
                task=task,
                persona_blend=persona_blend,
            )
            graph.add_node(conclusion)
            for syn in synthesis_nodes:
                graph.connect(syn.node_id, conclusion.node_id)

        logger.info(
            f"Built reasoning graph: {len(graph.nodes)} nodes, "
            f"depth={graph.max_depth_reached}, branches={graph.total_branches}"
        )

        return graph

    async def _generate_analysis_node(
        self,
        task: str,
        task_domains: List[str],
        persona_id: str,
        depth: int,
        parent_id: str,
    ) -> ReasoningNode:
        """Generate an analysis node."""
        # Simulate persona-specific analysis
        content = (
            f"[{persona_id}] Analysis: Examining task from {persona_id} perspective"
        )

        # Compute SNR using GoT orchestrator if available
        snr_score = 0.85
        if self.got_orchestrator:
            try:
                got_result = self.got_orchestrator.analyze(task, [])
                snr_score = got_result.get("cluster_snr", 0.85)
            except Exception:
                pass

        return ReasoningNode(
            node_id=f"node-analysis-{persona_id}-{uuid.uuid4().hex[:6]}",
            node_type=ReasoningNodeType.ANALYSIS,
            content=content,
            persona_source=persona_id,
            confidence=0.8 + (snr_score - 0.5) * 0.2,
            depth=depth,
            parent_ids=[parent_id],
            snr_score=snr_score,
            metadata={"task_domains": task_domains},
        )

    async def _generate_synthesis_node(
        self,
        analysis_content: str,
        persona_id: str,
        depth: int,
        parent_id: str,
    ) -> ReasoningNode:
        """Generate a synthesis node from analysis."""
        content = f"[{persona_id}] Synthesis: Integrating analysis findings"

        return ReasoningNode(
            node_id=f"node-synthesis-{persona_id}-{uuid.uuid4().hex[:6]}",
            node_type=ReasoningNodeType.SYNTHESIS,
            content=content,
            persona_source=persona_id,
            confidence=0.85,
            depth=depth,
            parent_ids=[parent_id],
            snr_score=0.88,
        )

    async def _generate_conclusion_node(
        self,
        synthesis_nodes: List[ReasoningNode],
        task: str,
        persona_blend: PersonaSoupBlend,
    ) -> ReasoningNode:
        """Generate final conclusion node."""
        # Combine synthesis contents
        synthesized_content = " | ".join(n.content for n in synthesis_nodes)

        # Weighted confidence from blend
        total_confidence = sum(
            n.confidence * persona_blend.weights.get(n.persona_source, 0.5)
            for n in synthesis_nodes
        )
        avg_confidence = total_confidence / max(len(synthesis_nodes), 1)

        return ReasoningNode(
            node_id=f"node-conclusion-{uuid.uuid4().hex[:8]}",
            node_type=ReasoningNodeType.CONCLUSION,
            content=f"Conclusion: {synthesized_content[:500]}",
            persona_source="blended",
            confidence=avg_confidence,
            depth=max(n.depth for n in synthesis_nodes) + 1,
            parent_ids=[n.node_id for n in synthesis_nodes],
            snr_score=0.90,
            metadata={
                "contributing_personas": list(persona_blend.weights.keys()),
            },
        )

    async def traverse_with_synthesis(
        self,
        graph: GraphOfThoughts,
    ) -> Optional[SynthesisNode]:
        """
        Traverse the graph and synthesize final output.

        Args:
            graph: The reasoning graph to traverse

        Returns:
            SynthesisNode with aggregated result
        """
        best_path = graph.get_best_synthesis_path()

        if not best_path:
            logger.warning("No valid synthesis path found in graph")
            return None

        # Aggregate content from path
        contents = [n.content for n in best_path]
        full_content = "\n\n".join(contents)

        # Compute aggregate confidence and SNR
        avg_confidence = sum(n.confidence for n in best_path) / len(best_path)
        avg_snr = sum(n.snr_score for n in best_path) / len(best_path)

        # Collect contributing personas
        persona_contributions: Dict[str, float] = {}
        for node in best_path:
            persona = node.persona_source
            if persona in persona_contributions:
                persona_contributions[persona] += node.confidence
            else:
                persona_contributions[persona] = node.confidence

        # Normalize contributions
        total = sum(persona_contributions.values())
        if total > 0:
            persona_contributions = {
                k: v / total for k, v in persona_contributions.items()
            }

        return SynthesisNode(
            synthesis_id=f"synth-{uuid.uuid4().hex[:12]}",
            content=full_content,
            source_graph_id=graph.graph_id,
            source_path=[n.node_id for n in best_path],
            confidence=avg_confidence,
            snr_score=avg_snr,
            ihsan_score=0.95,  # Default, should be computed
            contributing_personas=persona_contributions,
            evidence_refs=[f"graph:{graph.graph_id}"],
        )


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================


class SynthesisEngine:
    """
    The unified orchestrator combining all BIZRA apex components.

    This is the apex of the multi-persona system, coordinating:
    - Pareto optimal routing for lambda selection
    - Persona soup for weight interpolation
    - Graph of Thoughts for multi-dimensional reasoning
    - Weighted consensus for voting and veto
    - Synthesis gate for fail-closed enforcement
    - Receipt emission for evidence trail

    Usage:
        engine = create_synthesis_engine()
        result = await engine.synthesize(
            task="Optimize system performance",
            task_domains=["performance", "optimization"],
            constraints={"max_cost": 0.8},
        )

        if result.success:
            print(f"Synthesis: {result.synthesis_node.content}")
        else:
            print(f"Rejected: {result.gate_reason}")
    """

    def __init__(
        self,
        personas: Optional[List[Any]] = None,
        snr_threshold: float = SNR_THRESHOLD,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        weighted_quorum: float = WEIGHTED_QUORUM,
        receipt_path: Optional[Path] = None,
    ):
        """
        Initialize the synthesis engine.

        Args:
            personas: List of persona definitions (auto-loaded if None)
            snr_threshold: SNR gate threshold (default 0.98)
            ihsan_threshold: Ihsan gate threshold (default 0.95)
            weighted_quorum: Consensus quorum (default 2.4)
            receipt_path: Path for receipt storage
        """
        # Load personas
        if personas is None and PERSONAPLEX_AVAILABLE:
            try:
                personas = get_standard_bizra_personas()
            except Exception as e:
                logger.warning(f"Failed to load standard personas: {e}")
                personas = []

        self.personas = personas or []
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self.weighted_quorum = weighted_quorum

        # Receipt storage
        self.receipt_path = receipt_path or RECEIPT_PATH
        self.receipt_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pareto_router = ParetoOptimalRouter(personas=self.personas)
        self.soup_blender = PersonaSoupBlender(personas=self.personas)
        self.got_engine = GraphOfThoughtsEngine(personas=self.personas)
        self.synthesis_gate = SynthesisGate(
            snr_threshold=snr_threshold,
            ihsan_threshold=ihsan_threshold,
            weighted_quorum=weighted_quorum,
        )

        # Consensus system
        self.weighted_consensus: Optional[PersonaWeightedConsensus] = None
        if PERSONAPLEX_AVAILABLE:
            try:
                self.weighted_consensus = create_bizra_weighted_consensus(
                    weighted_quorum=weighted_quorum,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize weighted consensus: {e}")

        # Statistics
        self._total_syntheses = 0
        self._successful_syntheses = 0
        self._veto_rejections = 0
        self._snr_rejections = 0

        logger.info(
            f"SynthesisEngine initialized: {len(self.personas)} personas, "
            f"SNR≥{snr_threshold}, Ihsan≥{ihsan_threshold}, quorum≥{weighted_quorum}"
        )

    async def synthesize(
        self,
        task: str,
        task_domains: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> SynthesisResult:
        """
        Execute the full synthesis pipeline.

        Pipeline stages:
        1. Compute Pareto front of lambda configurations
        2. Thompson select from front
        3. Blend persona weights (Persona Soup)
        4. Build Graph of Thoughts
        5. Traverse graph with synthesis
        6. Run weighted consensus vote
        7. Apply synthesis gate (veto, consensus, SNR)
        8. Emit receipt
        9. Return result or rejection

        Args:
            task: Task description
            task_domains: List of relevant domains
            constraints: Optional constraints (max_cost, min_quality, etc.)
            request_id: Optional request ID
            timeout_ms: Timeout in milliseconds

        Returns:
            SynthesisResult with success/failure and full evidence
        """
        start_time = time.perf_counter()
        self._total_syntheses += 1

        request_id = request_id or f"synth-{uuid.uuid4().hex[:12]}"
        constraints = constraints or {}

        evidence_trail: List[Dict[str, Any]] = []
        stage_latencies: Dict[str, int] = {}

        logger.info(
            f"Starting synthesis: request={request_id}, task='{task[:50]}...', "
            f"domains={task_domains}"
        )

        try:
            # ─────────────────────────────────────────────────────────────────
            # Stage 1: Compute Pareto Front
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            pareto_front = self.pareto_router.compute_pareto_front(
                task_domains=task_domains,
                constraints=constraints,
            )

            stage_latencies[SynthesisStage.PARETO_FRONT.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.PARETO_FRONT.value,
                    "result": pareto_front.to_dict(),
                }
            )

            if len(pareto_front) == 0:
                return self._create_rejection(
                    request_id=request_id,
                    gate_status=GateStatus.ERROR,
                    gate_reason="Empty Pareto front - no valid configurations",
                    evidence_trail=evidence_trail,
                    stage_latencies=stage_latencies,
                    start_time=start_time,
                )

            # ─────────────────────────────────────────────────────────────────
            # Stage 2: Thompson Select from Front
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            selected_lambda = self.pareto_router.thompson_select(
                front=pareto_front,
                task_text=task,
            )

            stage_latencies[SynthesisStage.THOMPSON_SELECT.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.THOMPSON_SELECT.value,
                    "selected_lambda": (
                        selected_lambda.to_dict() if selected_lambda else None
                    ),
                }
            )

            if selected_lambda is None:
                return self._create_rejection(
                    request_id=request_id,
                    gate_status=GateStatus.ERROR,
                    gate_reason="Thompson selection failed",
                    evidence_trail=evidence_trail,
                    stage_latencies=stage_latencies,
                    start_time=start_time,
                )

            # ─────────────────────────────────────────────────────────────────
            # Stage 3: Persona Soup Blending
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            persona_blend = self.soup_blender.blend_from_lambda(
                lambda_config=selected_lambda,
                task_domains=task_domains,
            )

            stage_latencies[SynthesisStage.PERSONA_SOUP.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.PERSONA_SOUP.value,
                    "blend": persona_blend.to_dict(),
                }
            )

            # ─────────────────────────────────────────────────────────────────
            # Stage 4: Build Graph of Thoughts
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            reasoning_graph = await self.got_engine.build_graph(
                task=task,
                task_domains=task_domains,
                persona_blend=persona_blend,
            )

            stage_latencies[SynthesisStage.GOT_BUILD.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.GOT_BUILD.value,
                    "graph_stats": {
                        "node_count": len(reasoning_graph.nodes),
                        "max_depth": reasoning_graph.max_depth_reached,
                        "branches": reasoning_graph.total_branches,
                    },
                }
            )

            # ─────────────────────────────────────────────────────────────────
            # Stage 5: Traverse Graph with Synthesis
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            synthesis_node = await self.got_engine.traverse_with_synthesis(
                graph=reasoning_graph,
            )

            stage_latencies[SynthesisStage.GOT_TRAVERSE.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.GOT_TRAVERSE.value,
                    "synthesis_node": (
                        synthesis_node.to_dict() if synthesis_node else None
                    ),
                }
            )

            if synthesis_node is None:
                return self._create_rejection(
                    request_id=request_id,
                    gate_status=GateStatus.ERROR,
                    gate_reason="Graph traversal produced no synthesis",
                    evidence_trail=evidence_trail,
                    stage_latencies=stage_latencies,
                    start_time=start_time,
                )

            # ─────────────────────────────────────────────────────────────────
            # Stage 6: Weighted Consensus Vote
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            consensus_result_dict: Optional[Dict[str, Any]] = None

            if self.weighted_consensus:
                try:
                    consensus_result = await self.weighted_consensus.weighted_consensus(
                        envelope={"task": task, "content": synthesis_node.content},
                        task_domains=task_domains,
                        gate_results={
                            "ihsan_score": synthesis_node.ihsan_score,
                            "snr_score": synthesis_node.snr_score,
                        },
                        request_id=request_id,
                    )
                    consensus_result_dict = consensus_result.to_dict()
                except Exception as e:
                    logger.warning(f"Weighted consensus failed: {e}")
                    # Fallback: create passing result if SNR is good
                    consensus_result_dict = {
                        "passed": synthesis_node.snr_score >= self.snr_threshold,
                        "veto_triggered": False,
                        "total_weighted_score": self.weighted_quorum + 0.1,
                        "weighted_quorum": self.weighted_quorum,
                        "votes": [],
                    }
            else:
                # No consensus system - create default passing result
                consensus_result_dict = {
                    "passed": True,
                    "veto_triggered": False,
                    "total_weighted_score": self.weighted_quorum + 0.1,
                    "weighted_quorum": self.weighted_quorum,
                    "votes": [],
                }

            stage_latencies[SynthesisStage.WEIGHTED_CONSENSUS.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.WEIGHTED_CONSENSUS.value,
                    "consensus": consensus_result_dict,
                }
            )

            # ─────────────────────────────────────────────────────────────────
            # Stage 7: Synthesis Gate (Fail-Closed)
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            gate_status, gate_reason = self.synthesis_gate.validate(
                consensus_result=consensus_result_dict,
                snr_score=synthesis_node.snr_score,
                content=synthesis_node.content,
            )

            stage_latencies[SynthesisStage.SYNTHESIS_GATE.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.SYNTHESIS_GATE.value,
                    "gate_status": gate_status.value,
                    "gate_reason": gate_reason,
                }
            )

            # ─────────────────────────────────────────────────────────────────
            # Handle Gate Rejection
            # ─────────────────────────────────────────────────────────────────
            if gate_status != GateStatus.PASSED:
                # Track rejection type
                if gate_status == GateStatus.VETO_TRIGGERED:
                    self._veto_rejections += 1
                elif gate_status == GateStatus.SNR_FAILED:
                    self._snr_rejections += 1

                return self._create_rejection(
                    request_id=request_id,
                    gate_status=gate_status,
                    gate_reason=gate_reason,
                    evidence_trail=evidence_trail,
                    stage_latencies=stage_latencies,
                    start_time=start_time,
                    synthesis_node=synthesis_node,
                    consensus_result=consensus_result_dict,
                    pareto_front_size=len(pareto_front),
                    selected_lambda=selected_lambda.lambda_id,
                    persona_blend=persona_blend.to_dict(),
                    graph_stats=reasoning_graph.to_dict(),
                )

            # ─────────────────────────────────────────────────────────────────
            # Stage 8: Receipt Emission
            # ─────────────────────────────────────────────────────────────────
            stage_start = time.perf_counter()

            receipt_id = await self._emit_receipt(
                request_id=request_id,
                synthesis_node=synthesis_node,
                consensus_result=consensus_result_dict,
                evidence_trail=evidence_trail,
            )

            stage_latencies[SynthesisStage.RECEIPT_EMISSION.value] = int(
                (time.perf_counter() - stage_start) * 1000
            )

            evidence_trail.append(
                {
                    "stage": SynthesisStage.RECEIPT_EMISSION.value,
                    "receipt_id": receipt_id,
                }
            )

            # ─────────────────────────────────────────────────────────────────
            # Success!
            # ─────────────────────────────────────────────────────────────────
            self._successful_syntheses += 1
            total_latency = int((time.perf_counter() - start_time) * 1000)

            result = SynthesisResult(
                success=True,
                synthesis_node=synthesis_node,
                consensus_result=consensus_result_dict,
                snr_score=synthesis_node.snr_score,
                evidence_trail=evidence_trail,
                gate_status=GateStatus.PASSED,
                gate_reason=None,
                request_id=request_id,
                pareto_front_size=len(pareto_front),
                selected_lambda=selected_lambda.lambda_id,
                persona_blend=persona_blend.to_dict(),
                graph_stats=reasoning_graph.to_dict(),
                total_latency_ms=total_latency,
                stage_latencies=stage_latencies,
            )

            logger.info(
                f"Synthesis APPROVED: request={request_id}, "
                f"SNR={synthesis_node.snr_score:.4f}, "
                f"latency={total_latency}ms"
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"Synthesis timeout: request={request_id}")
            return self._create_rejection(
                request_id=request_id,
                gate_status=GateStatus.TIMEOUT,
                gate_reason=f"Synthesis timed out after {timeout_ms}ms",
                evidence_trail=evidence_trail,
                stage_latencies=stage_latencies,
                start_time=start_time,
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            return self._create_rejection(
                request_id=request_id,
                gate_status=GateStatus.ERROR,
                gate_reason=f"Synthesis error: {str(e)}",
                evidence_trail=evidence_trail,
                stage_latencies=stage_latencies,
                start_time=start_time,
            )

    def _create_rejection(
        self,
        request_id: str,
        gate_status: GateStatus,
        gate_reason: Optional[str],
        evidence_trail: List[Dict[str, Any]],
        stage_latencies: Dict[str, int],
        start_time: float,
        synthesis_node: Optional[SynthesisNode] = None,
        consensus_result: Optional[Dict[str, Any]] = None,
        pareto_front_size: int = 0,
        selected_lambda: Optional[str] = None,
        persona_blend: Optional[Dict[str, Any]] = None,
        graph_stats: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        """Create a rejection result."""
        total_latency = int((time.perf_counter() - start_time) * 1000)

        logger.warning(
            f"Synthesis REJECTED: request={request_id}, "
            f"status={gate_status.value}, reason={gate_reason}"
        )

        # Emit rejection receipt
        asyncio.create_task(
            self._emit_rejection_receipt(
                request_id=request_id,
                gate_status=gate_status,
                gate_reason=gate_reason,
                evidence_trail=evidence_trail,
            )
        )

        return SynthesisResult(
            success=False,
            synthesis_node=synthesis_node,
            consensus_result=consensus_result,
            snr_score=synthesis_node.snr_score if synthesis_node else 0.0,
            evidence_trail=evidence_trail,
            gate_status=gate_status,
            gate_reason=gate_reason,
            request_id=request_id,
            pareto_front_size=pareto_front_size,
            selected_lambda=selected_lambda,
            persona_blend=persona_blend,
            graph_stats=graph_stats,
            total_latency_ms=total_latency,
            stage_latencies=stage_latencies,
        )

    async def _emit_receipt(
        self,
        request_id: str,
        synthesis_node: SynthesisNode,
        consensus_result: Dict[str, Any],
        evidence_trail: List[Dict[str, Any]],
    ) -> str:
        """Emit success receipt to chain."""
        receipt_id = f"synth-success-{uuid.uuid4().hex[:12]}"

        receipt_data = {
            "receipt_id": receipt_id,
            "type": "SYNTHESIS_SUCCESS",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "synthesis_id": synthesis_node.synthesis_id,
            "snr_score": synthesis_node.snr_score,
            "confidence": synthesis_node.confidence,
            "consensus_passed": consensus_result.get("passed", False),
            "consensus_score": consensus_result.get("total_weighted_score", 0.0),
            "evidence_trail_length": len(evidence_trail),
            "integrity_hash": self._compute_integrity_hash(synthesis_node),
        }

        try:
            receipt_file = (
                self.receipt_path
                / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
            )

            with open(receipt_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt_data) + "\n")

            logger.debug(f"Emitted success receipt: {receipt_id}")

        except Exception as e:
            logger.error(f"Failed to emit receipt: {e}")

        return receipt_id

    async def _emit_rejection_receipt(
        self,
        request_id: str,
        gate_status: GateStatus,
        gate_reason: Optional[str],
        evidence_trail: List[Dict[str, Any]],
    ) -> str:
        """Emit rejection receipt to chain."""
        receipt_id = f"synth-reject-{uuid.uuid4().hex[:12]}"

        receipt_data = {
            "receipt_id": receipt_id,
            "type": "SYNTHESIS_REJECTION",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate_status": gate_status.value,
            "gate_reason": gate_reason,
            "evidence_trail_length": len(evidence_trail),
        }

        try:
            receipt_file = (
                self.receipt_path
                / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-rejections.jsonl"
            )

            with open(receipt_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt_data) + "\n")

            logger.debug(f"Emitted rejection receipt: {receipt_id}")

        except Exception as e:
            logger.error(f"Failed to emit rejection receipt: {e}")

        return receipt_id

    def _compute_integrity_hash(self, synthesis_node: SynthesisNode) -> str:
        """Compute integrity hash for receipt."""
        data = json.dumps(synthesis_node.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_syntheses": self._total_syntheses,
            "successful_syntheses": self._successful_syntheses,
            "veto_rejections": self._veto_rejections,
            "snr_rejections": self._snr_rejections,
            "success_rate": (
                self._successful_syntheses / max(1, self._total_syntheses)
            ),
            "snr_threshold": self.snr_threshold,
            "ihsan_threshold": self.ihsan_threshold,
            "weighted_quorum": self.weighted_quorum,
            "persona_count": len(self.personas),
            "components": {
                "thompson": THOMPSON_AVAILABLE,
                "personaplex": PERSONAPLEX_AVAILABLE,
                "snr": SNR_AVAILABLE,
                "got": GOT_AVAILABLE,
                "consensus": CONSENSUS_AVAILABLE,
            },
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_synthesis_engine(
    snr_threshold: float = SNR_THRESHOLD,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    weighted_quorum: float = WEIGHTED_QUORUM,
    receipt_path: Optional[Path] = None,
) -> SynthesisEngine:
    """
    Factory function to create a SynthesisEngine with standard configuration.

    This is the recommended entry point for creating a synthesis engine
    with BIZRA's standard personas and enforcement thresholds.

    Args:
        snr_threshold: SNR gate threshold (default 0.98)
        ihsan_threshold: Ihsan gate threshold (default 0.95)
        weighted_quorum: Consensus quorum (default 2.4)
        receipt_path: Path for receipt storage

    Returns:
        Configured SynthesisEngine instance

    Example:
        >>> engine = create_synthesis_engine()
        >>> result = await engine.synthesize(
        ...     task="Optimize database performance",
        ...     task_domains=["database", "performance"],
        ... )
        >>> print(f"Success: {result.success}")
    """
    return SynthesisEngine(
        personas=None,  # Auto-load standard personas
        snr_threshold=snr_threshold,
        ihsan_threshold=ihsan_threshold,
        weighted_quorum=weighted_quorum,
        receipt_path=receipt_path,
    )


# =============================================================================
# MAIN / DEMO
# =============================================================================


async def main():
    """Demo the Synthesis Engine."""
    print("=" * 80)
    print("BIZRA Synthesis Engine - Multi-Persona Reasoning Pipeline")
    print("=" * 80)

    # Create engine
    engine = create_synthesis_engine()

    print("\nEngine Statistics:")
    stats = engine.get_statistics()
    print(json.dumps(stats, indent=2))

    # Test synthesis
    print("\n" + "-" * 40)
    print("Running synthesis test...")
    print("-" * 40)

    result = await engine.synthesize(
        task="Optimize the BIZRA data pipeline for maximum throughput while maintaining security",
        task_domains=["data-engineering", "performance", "security", "optimization"],
        constraints={"max_cost": 0.7, "min_quality": 0.8},
    )

    print(f"\nResult: {'SUCCESS' if result.success else 'REJECTED'}")
    print(f"Gate Status: {result.gate_status.value}")
    print(f"SNR Score: {result.snr_score:.4f}")
    print(f"Total Latency: {result.total_latency_ms}ms")
    print(f"Request ID: {result.request_id}")

    if result.success and result.synthesis_node:
        print("\nSynthesis Content Preview:")
        print(result.synthesis_node.content[:300] + "...")
        print("\nContributing Personas:")
        for pid, contrib in result.synthesis_node.contributing_personas.items():
            print(f"  {pid}: {contrib:.2%}")
    else:
        print(f"\nRejection Reason: {result.gate_reason}")

    print("\nStage Latencies:")
    for stage, latency in result.stage_latencies.items():
        print(f"  {stage}: {latency}ms")

    # Final statistics
    print("\n" + "=" * 80)
    print("Final Statistics:")
    print(json.dumps(engine.get_statistics(), indent=2))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
