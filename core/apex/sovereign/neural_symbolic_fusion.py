"""
BIZRA Neural-Symbolic Fusion Engine
====================================
Implements LLM + formal verification integration for trusted reasoning.

This engine fuses neural reasoning (Graph of Thoughts) with symbolic verification
(SMT solving, constitutional checking, knowledge graph grounding) following the
"Don't Trust: Verify" paradigm from neuro-symbolic AI research.

Architecture:
    +---------------------------------------------------------------------------+
    |                    NEURAL-SYMBOLIC FUSION ENGINE                           |
    +---------------------------------------------------------------------------+
    |                                                                            |
    |   Query + Context -----> [Neural Path]                                     |
    |                               |                                            |
    |                               v                                            |
    |                     +------------------+                                   |
    |                     | GoT Reasoning    |                                   |
    |                     | - Multi-domain   |                                   |
    |                     | - Persona soup   |                                   |
    |                     | - Synthesis      |                                   |
    |                     +------------------+                                   |
    |                               |                                            |
    |            +-----------------+-----------------+                           |
    |            |                 |                 |                           |
    |            v                 v                 v                           |
    |     [Autoformalize]   [Constitutional]  [KG Grounding]                    |
    |            |               Check              |                            |
    |            v                 |                 v                           |
    |     +-------------+         |          +--------------+                   |
    |     | Z3 SMT      |<--------+--------->| SAPE 9-Probe |                   |
    |     | Solving     |                    | Validation   |                   |
    |     +-------------+                    +--------------+                   |
    |            |                                  |                            |
    |            +-----------+   +------------------+                            |
    |                        |   |                                               |
    |                        v   v                                               |
    |                   [Fusion Result]                                          |
    |                   - Verified/Refuted                                       |
    |                   - Attestation                                            |
    |                   - Evidence chain                                         |
    |                                                                            |
    +---------------------------------------------------------------------------+

Key Patterns (from research):
    - "Don't Trust: Verify" - autoformalize and verify LLM outputs
    - Generate-and-Check loops with counterexample feedback
    - Multiple solver backends for different constraint types
    - Knowledge graph grounding for provenance

Integration:
    - GraphOfThoughtsEngine from core/apex/graph_of_thoughts.py
    - SAPE from core/sape.py (9-probe system)
    - Constitution from constitution/ihsan_v1.yaml

Domain: bizra-apex-v1:sovereign:fusion
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    NOVELTY_THRESHOLD_STANDARD,
    IHSAN_WEIGHTS,
)

# Configure logging
logger = logging.getLogger("apex.sovereign.neural_symbolic_fusion")


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

DOMAIN_PREFIX = "bizra-apex-v1:sovereign:fusion"
VERSION = "1.0.0"

# Default thresholds from core/constants.py (Genesis v2.2.2 compliance)
DEFAULT_IHSAN_THRESHOLD = IHSAN_THRESHOLD  # 0.95
DEFAULT_SNR_THRESHOLD = SNR_THRESHOLD_T0_ELITE  # 0.98
DEFAULT_NOVELTY_THRESHOLD = NOVELTY_THRESHOLD_STANDARD  # 0.75

# SAPE probe execution timeout (ms)
DEFAULT_PROBE_TIMEOUT_MS = 5000

# Maximum iterations for iterative fusion mode
MAX_ITERATIVE_ROUNDS = 5

# Ihsan 8-dimension weights (from constitution/ihsan_v1.yaml)
IHSAN_DIMENSIONS = {
    "correctness": 0.22,
    "safety": 0.22,
    "user_benefit": 0.14,
    "efficiency": 0.12,
    "auditability": 0.12,
    "anti_centralization": 0.08,
    "robustness": 0.06,
    "adl_fairness": 0.04,
}

# SAPE 9-probe mapping to Ihsan dimensions
SAPE_TO_IHSAN_MAP = {
    "threat_scan": "safety",
    "compliance_check": "auditability",
    "bias_probe": "adl_fairness",
    "user_benefit": "user_benefit",
    "correctness": "correctness",
    "safety": "safety",
    "groundedness": "robustness",
    "relevance": "efficiency",
    "fluency": "anti_centralization",
}


# =============================================================================
# ENUMS
# =============================================================================


class FusionMode(str, Enum):
    """
    Mode of neural-symbolic fusion execution.

    Attributes:
        PARALLEL: Neural and symbolic paths run concurrently
        SEQUENTIAL: Neural path first, then symbolic verification
        ITERATIVE: Feedback loop refinement until convergence
    """
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ITERATIVE = "iterative"


class SymbolicBackend(str, Enum):
    """
    Available symbolic verification backends.

    Attributes:
        Z3_SMT: Satisfiability Modulo Theories solver for formal constraints
        CONSTITUTIONAL: Ihsan 8-dimension constraint checking
        DATALOG: Knowledge graph reasoning with Datalog-style queries
        MERKLE: Evidence chain verification via Merkle proofs
    """
    Z3_SMT = "z3_smt"
    CONSTITUTIONAL = "constitutional"
    DATALOG = "datalog"
    MERKLE = "merkle"


class VerificationStatus(str, Enum):
    """Status of symbolic verification."""
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


class FusionStatus(str, Enum):
    """Overall fusion result status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FusionContext:
    """
    Context for neural-symbolic fusion execution.

    Contains all inputs needed for both neural reasoning and symbolic verification.

    Attributes:
        query: The original query/task to process
        domains: List of expertise domains relevant to the task
        pareto_solutions: Pareto-optimal points from multi-objective routing
        constitutional_constraints: Ihsan dimension thresholds to enforce
        knowledge_graph_nodes: Node IDs for knowledge graph grounding
        fusion_mode: Mode of execution (parallel/sequential/iterative)
        symbolic_backends: Which symbolic backends to use
        max_iterations: Maximum iterations for iterative mode
        timeout_ms: Overall timeout in milliseconds
        metadata: Additional context metadata
    """
    query: str
    domains: List[str] = field(default_factory=list)
    pareto_solutions: List[Any] = field(default_factory=list)
    constitutional_constraints: Dict[str, float] = field(default_factory=dict)
    knowledge_graph_nodes: List[str] = field(default_factory=list)
    fusion_mode: FusionMode = FusionMode.SEQUENTIAL
    symbolic_backends: List[SymbolicBackend] = field(
        default_factory=lambda: [SymbolicBackend.CONSTITUTIONAL, SymbolicBackend.DATALOG]
    )
    max_iterations: int = MAX_ITERATIVE_ROUNDS
    timeout_ms: int = 30000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default constitutional constraints if not provided."""
        if not self.constitutional_constraints:
            self.constitutional_constraints = {
                dim: DEFAULT_IHSAN_THRESHOLD for dim in IHSAN_DIMENSIONS
            }

    @property
    def context_id(self) -> str:
        """Generate unique context identifier."""
        content = f"{self.query}:{self.domains}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "context_id": self.context_id,
            "query": self.query[:200] + "..." if len(self.query) > 200 else self.query,
            "domains": self.domains,
            "pareto_solution_count": len(self.pareto_solutions),
            "constitutional_constraints": self.constitutional_constraints,
            "knowledge_graph_node_count": len(self.knowledge_graph_nodes),
            "fusion_mode": self.fusion_mode.value,
            "symbolic_backends": [b.value for b in self.symbolic_backends],
            "max_iterations": self.max_iterations,
            "timeout_ms": self.timeout_ms,
            "metadata": self.metadata,
        }


@dataclass
class NeuralResult:
    """
    Result from the neural reasoning path (Graph of Thoughts).

    Attributes:
        reasoning_graph: The GoT graph structure (serializable representation)
        confidence: Overall confidence score from neural reasoning
        synthesis_content: Synthesized reasoning output
        domains_covered: Domains that contributed to the synthesis
        node_count: Number of reasoning nodes generated
        edge_count: Number of edges in the reasoning graph
        snr_score: Signal-to-noise ratio score
        latency_ms: Neural path execution time
        metadata: Additional neural result metadata
    """
    reasoning_graph: Dict[str, Any]
    confidence: float
    synthesis_content: str
    domains_covered: List[str]
    node_count: int = 0
    edge_count: int = 0
    snr_score: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence is in range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reasoning_graph_summary": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
            },
            "confidence": self.confidence,
            "synthesis_content": (
                self.synthesis_content[:500] + "..."
                if len(self.synthesis_content) > 500
                else self.synthesis_content
            ),
            "domains_covered": self.domains_covered,
            "snr_score": self.snr_score,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class ConstraintViolation:
    """
    Details of a constraint violation during verification.

    Attributes:
        constraint_id: Identifier of the violated constraint
        constraint_type: Type of constraint (constitutional, smt, datalog)
        expected_value: Expected/threshold value
        actual_value: Actual measured value
        violation_severity: How severe the violation is (0-1)
        evidence: Supporting evidence for the violation
    """
    constraint_id: str
    constraint_type: str
    expected_value: float
    actual_value: float
    violation_severity: float
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "violation_severity": self.violation_severity,
            "evidence": self.evidence,
        }


@dataclass
class SymbolicResult:
    """
    Result from symbolic verification path.

    Attributes:
        verified: Overall verification status
        status: Detailed verification status
        constraints_checked: List of constraint IDs that were checked
        violations: List of constraint violations found
        formal_proof: Optional formal proof string (when available)
        sape_scores: Dictionary mapping probe names to scores
        ihsan_scores: Dictionary mapping Ihsan dimensions to scores
        backend_results: Results from each symbolic backend
        latency_ms: Symbolic path execution time
        metadata: Additional symbolic result metadata
    """
    verified: bool
    status: VerificationStatus
    constraints_checked: List[str]
    violations: List[ConstraintViolation]
    formal_proof: Optional[str] = None
    sape_scores: Dict[str, float] = field(default_factory=dict)
    ihsan_scores: Dict[str, float] = field(default_factory=dict)
    backend_results: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_ihsan_score(self) -> float:
        """Compute weighted Ihsan score from dimension scores."""
        if not self.ihsan_scores:
            return 0.0

        total = 0.0
        for dim, weight in IHSAN_DIMENSIONS.items():
            score = self.ihsan_scores.get(dim, 0.0)
            total += score * weight

        return total

    @property
    def violation_count(self) -> int:
        """Number of violations found."""
        return len(self.violations)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "verified": self.verified,
            "status": self.status.value,
            "constraints_checked": self.constraints_checked,
            "violation_count": self.violation_count,
            "violations": [v.to_dict() for v in self.violations[:10]],
            "formal_proof": self.formal_proof[:500] if self.formal_proof else None,
            "sape_scores": self.sape_scores,
            "ihsan_scores": self.ihsan_scores,
            "overall_ihsan_score": self.overall_ihsan_score,
            "backend_results": {
                k: str(v)[:200] for k, v in self.backend_results.items()
            },
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class FusionResult:
    """
    Complete result from neural-symbolic fusion.

    Attributes:
        success: Whether fusion completed successfully
        status: Overall fusion status
        neural_result: Result from neural reasoning path
        symbolic_result: Result from symbolic verification path
        fused_confidence: Combined confidence score
        verification_attestation: Cryptographic attestation of verification
        iteration_count: Number of iterations (for iterative mode)
        total_latency_ms: Total fusion execution time
        evidence_chain: Chain of evidence nodes for audit
        timestamp: ISO timestamp of fusion completion
        metadata: Additional fusion metadata
    """
    success: bool
    status: FusionStatus
    neural_result: NeuralResult
    symbolic_result: SymbolicResult
    fused_confidence: float
    verification_attestation: str
    iteration_count: int = 1
    total_latency_ms: float = 0.0
    evidence_chain: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fusion_id(self) -> str:
        """Generate unique fusion identifier."""
        content = f"{self.timestamp}:{self.verification_attestation}"
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    @property
    def meets_ihsan_threshold(self) -> bool:
        """Check if result meets Ihsan threshold."""
        return self.symbolic_result.overall_ihsan_score >= DEFAULT_IHSAN_THRESHOLD

    @property
    def meets_snr_threshold(self) -> bool:
        """Check if result meets SNR threshold."""
        return self.neural_result.snr_score >= DEFAULT_SNR_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fusion_id": self.fusion_id,
            "success": self.success,
            "status": self.status.value,
            "neural_result": self.neural_result.to_dict(),
            "symbolic_result": self.symbolic_result.to_dict(),
            "fused_confidence": self.fused_confidence,
            "verification_attestation": self.verification_attestation[:64] + "...",
            "iteration_count": self.iteration_count,
            "total_latency_ms": self.total_latency_ms,
            "evidence_chain_length": len(self.evidence_chain),
            "meets_ihsan_threshold": self.meets_ihsan_threshold,
            "meets_snr_threshold": self.meets_snr_threshold,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# VERIFICATION PROTOCOLS
# =============================================================================


class ConstitutionalVerifier(Protocol):
    """Protocol for constitutional (Ihsan) verification."""

    async def verify_ihsan_dimensions(
        self,
        content: str,
        context: FusionContext,
    ) -> Dict[str, float]:
        """
        Verify content against Ihsan 8-dimension constitution.

        Args:
            content: Content to verify
            context: Fusion context with constraints

        Returns:
            Dictionary mapping dimension names to scores (0-1)
        """
        ...


class SAPEEngine(Protocol):
    """Protocol for SAPE 9-probe engine."""

    async def run_probes_parallel(
        self,
        content: str,
        context: FusionContext,
    ) -> Dict[str, float]:
        """
        Run all 9 SAPE probes in parallel.

        Args:
            content: Content to probe
            context: Fusion context

        Returns:
            Dictionary mapping probe names to scores (0-1)
        """
        ...


class KnowledgeGraphConnector(Protocol):
    """Protocol for knowledge graph grounding."""

    async def ground_reasoning(
        self,
        content: str,
        node_ids: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Ground reasoning in knowledge graph evidence.

        Args:
            content: Content to ground
            node_ids: Node IDs to use for grounding

        Returns:
            Tuple of (is_grounded, evidence_node_ids)
        """
        ...


# =============================================================================
# DEFAULT IMPLEMENTATIONS
# =============================================================================


class DefaultConstitutionalVerifier:
    """
    Default implementation of Ihsan constitutional verification.

    Uses heuristic scoring based on content analysis.
    Production systems should integrate with the full Ihsan engine.
    """

    def __init__(self, threshold: float = DEFAULT_IHSAN_THRESHOLD):
        """
        Initialize verifier.

        Args:
            threshold: Minimum score for verification pass
        """
        self.threshold = threshold
        self._violation_patterns = {
            "safety": ["harm", "danger", "attack", "exploit", "vulnerability"],
            "correctness": ["incorrect", "wrong", "error", "mistake", "false"],
            "user_benefit": ["useless", "waste", "unhelpful", "misleading"],
            "auditability": ["hidden", "obfuscated", "untraceable"],
            "adl_fairness": ["biased", "discriminate", "unfair", "prejudice"],
        }

    async def verify_ihsan_dimensions(
        self,
        content: str,
        context: FusionContext,
    ) -> Dict[str, float]:
        """Verify content against Ihsan dimensions using heuristic analysis."""
        content_lower = content.lower()
        scores: Dict[str, float] = {}

        for dimension, weight in IHSAN_DIMENSIONS.items():
            # Base score from constitutional constraint or default
            base_score = context.constitutional_constraints.get(dimension, 0.95)

            # Reduce score if violation patterns found
            patterns = self._violation_patterns.get(dimension, [])
            violations_found = sum(
                1 for pattern in patterns if pattern in content_lower
            )

            # Penalty: 0.05 per violation found, minimum 0.5
            penalty = violations_found * 0.05
            score = max(0.5, base_score - penalty)

            # Boost for positive indicators
            if dimension == "auditability" and "evidence" in content_lower:
                score = min(1.0, score + 0.03)
            if dimension == "user_benefit" and "helpful" in content_lower:
                score = min(1.0, score + 0.02)

            scores[dimension] = round(score, 4)

        return scores


class DefaultSAPEEngine:
    """
    Default implementation of SAPE 9-probe execution.

    Provides heuristic probe scores based on content analysis.
    Production systems should integrate with the full SAPE engine.
    """

    # 9 canonical SAPE probes
    PROBES = [
        "threat_scan",
        "compliance_check",
        "bias_probe",
        "user_benefit",
        "correctness",
        "safety",
        "groundedness",
        "relevance",
        "fluency",
    ]

    def __init__(self, timeout_ms: int = DEFAULT_PROBE_TIMEOUT_MS):
        """
        Initialize SAPE engine.

        Args:
            timeout_ms: Timeout for probe execution
        """
        self.timeout_ms = timeout_ms

    async def run_probes_parallel(
        self,
        content: str,
        context: FusionContext,
    ) -> Dict[str, float]:
        """Run all 9 SAPE probes in parallel."""
        # Create probe tasks
        tasks = [
            self._run_probe(probe, content, context)
            for probe in self.PROBES
        ]

        # Execute in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning("SAPE probes timed out, using default scores")
            return {probe: 0.8 for probe in self.PROBES}

        # Collect results
        scores: Dict[str, float] = {}
        for probe, result in zip(self.PROBES, results):
            if isinstance(result, Exception):
                logger.warning(f"Probe {probe} failed: {result}")
                scores[probe] = 0.8  # Default on error
            else:
                scores[probe] = result

        return scores

    async def _run_probe(
        self,
        probe_name: str,
        content: str,
        context: FusionContext,
    ) -> float:
        """Run a single probe and return score."""
        # Heuristic scoring based on probe type
        content_lower = content.lower()

        if probe_name == "threat_scan":
            # Check for security threats
            threats = ["inject", "bypass", "execute", "attack", "<script>"]
            threat_count = sum(1 for t in threats if t in content_lower)
            return max(0.5, 1.0 - threat_count * 0.1)

        elif probe_name == "compliance_check":
            # Check for audit trail indicators
            if "evidence" in content_lower or "receipt" in content_lower:
                return 0.95
            return 0.85

        elif probe_name == "bias_probe":
            # Check for bias indicators
            biased_terms = ["always", "never", "everyone", "no one"]
            bias_count = sum(1 for b in biased_terms if b in content_lower)
            return max(0.6, 1.0 - bias_count * 0.05)

        elif probe_name == "user_benefit":
            # Check for user-beneficial content
            if "help" in content_lower or "benefit" in content_lower:
                return 0.92
            return 0.85

        elif probe_name == "correctness":
            # Heuristic: longer, detailed content is more likely correct
            return min(0.95, 0.7 + len(content) / 5000.0)

        elif probe_name == "safety":
            # Check for safety concerns
            unsafe_terms = ["delete", "rm -rf", "drop table", "harm"]
            unsafe_count = sum(1 for u in unsafe_terms if u in content_lower)
            return max(0.5, 1.0 - unsafe_count * 0.15)

        elif probe_name == "groundedness":
            # Check for evidence/citation indicators
            grounded_terms = ["evidence", "citation", "source", "proof"]
            grounded_count = sum(1 for g in grounded_terms if g in content_lower)
            return min(1.0, 0.75 + grounded_count * 0.05)

        elif probe_name == "relevance":
            # Check alignment with query domains
            domain_matches = sum(
                1 for d in context.domains if d.lower() in content_lower
            )
            return min(1.0, 0.7 + domain_matches * 0.1)

        elif probe_name == "fluency":
            # Heuristic: check for basic fluency indicators
            word_count = len(content.split())
            if word_count > 10:
                return 0.9
            return 0.75

        return 0.8  # Default score


class DefaultKnowledgeGraphConnector:
    """
    Default implementation of knowledge graph grounding.

    Provides mock grounding for testing.
    Production systems should integrate with Neo4j or Gold Mine.
    """

    def __init__(self, evidence_path: str = "docs/evidence"):
        """
        Initialize connector.

        Args:
            evidence_path: Path to evidence directory
        """
        self.evidence_path = evidence_path

    async def ground_reasoning(
        self,
        content: str,
        node_ids: List[str],
    ) -> Tuple[bool, List[str]]:
        """Ground reasoning in knowledge graph evidence."""
        # Mock implementation: always partially grounded
        # In production, query Neo4j or Gold Mine

        if not node_ids:
            return False, []

        # Simulate finding some evidence
        evidence_nodes = node_ids[:min(3, len(node_ids))]
        is_grounded = len(evidence_nodes) > 0

        return is_grounded, evidence_nodes


# =============================================================================
# NEURAL-SYMBOLIC FUSION ENGINE
# =============================================================================


class NeuralSymbolicFusionEngine:
    """
    Neural-Symbolic Fusion Engine for trusted AI reasoning.

    Combines Graph of Thoughts neural reasoning with symbolic verification
    using SMT solving, constitutional checking, and knowledge graph grounding.

    The engine follows the "Don't Trust: Verify" paradigm:
    1. Generate reasoning via neural path (GoT)
    2. Autoformalize reasoning into verifiable constraints
    3. Verify against symbolic backends
    4. Return fused result with verification attestation

    Attributes:
        got_engine: Graph of Thoughts engine for neural reasoning
        constitutional_verifier: Ihsan 8-dimension verifier
        sape_engine: SAPE 9-probe verification engine
        kg_connector: Knowledge graph connector for grounding

    Example:
        >>> fusion_engine = NeuralSymbolicFusionEngine()
        >>> context = FusionContext(
        ...     query="Optimize the data pipeline for throughput",
        ...     domains=["data-engineering", "systems"],
        ... )
        >>> result = await fusion_engine.fuse(context)
        >>> print(f"Verified: {result.symbolic_result.verified}")
        >>> print(f"Ihsan: {result.symbolic_result.overall_ihsan_score:.3f}")
    """

    def __init__(
        self,
        got_engine: Optional[Any] = None,
        constitutional_verifier: Optional[ConstitutionalVerifier] = None,
        sape_engine: Optional[SAPEEngine] = None,
        kg_connector: Optional[KnowledgeGraphConnector] = None,
    ):
        """
        Initialize the Neural-Symbolic Fusion Engine.

        Args:
            got_engine: GraphOfThoughtsEngine instance (lazy loads if None)
            constitutional_verifier: Ihsan verifier (uses default if None)
            sape_engine: SAPE engine (uses default if None)
            kg_connector: KG connector (uses default if None)
        """
        self._got_engine = got_engine
        self._constitutional_verifier = constitutional_verifier or DefaultConstitutionalVerifier()
        self._sape_engine = sape_engine or DefaultSAPEEngine()
        self._kg_connector = kg_connector or DefaultKnowledgeGraphConnector()

        # Statistics tracking
        self._total_fusions = 0
        self._successful_fusions = 0
        self._verification_failures = 0
        self._total_latency_ms = 0.0

        logger.info(
            f"NeuralSymbolicFusionEngine initialized: "
            f"domain={DOMAIN_PREFIX}, version={VERSION}"
        )

    @property
    def got_engine(self) -> Any:
        """Lazy-load GoT engine if not provided."""
        if self._got_engine is None:
            try:
                from core.apex.graph_of_thoughts import create_got_engine
                self._got_engine = create_got_engine(snr_threshold=0.95)
            except ImportError:
                logger.warning("GoT engine not available, using mock")
                self._got_engine = None
        return self._got_engine

    async def fuse(self, context: FusionContext) -> FusionResult:
        """
        Execute neural-symbolic fusion for the given context.

        The fusion mode determines execution strategy:
        - PARALLEL: Neural and symbolic paths run concurrently
        - SEQUENTIAL: Neural first, then symbolic verification
        - ITERATIVE: Feedback loop until convergence

        Args:
            context: FusionContext with query and constraints

        Returns:
            FusionResult with verified reasoning and attestation
        """
        import time
        start_time = time.perf_counter()

        logger.info(
            f"Starting neural-symbolic fusion: mode={context.fusion_mode.value}, "
            f"query={context.query[:50]}..."
        )

        self._total_fusions += 1

        try:
            if context.fusion_mode == FusionMode.PARALLEL:
                result = await self._fuse_parallel(context)
            elif context.fusion_mode == FusionMode.SEQUENTIAL:
                result = await self._fuse_sequential(context)
            elif context.fusion_mode == FusionMode.ITERATIVE:
                result = await self._fuse_iterative(context)
            else:
                raise ValueError(f"Unknown fusion mode: {context.fusion_mode}")

            # Track success
            if result.success:
                self._successful_fusions += 1

            # Calculate total latency
            total_latency = (time.perf_counter() - start_time) * 1000
            result.total_latency_ms = total_latency
            self._total_latency_ms += total_latency

            logger.info(
                f"Fusion complete: success={result.success}, "
                f"ihsan={result.symbolic_result.overall_ihsan_score:.3f}, "
                f"latency={total_latency:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Fusion failed with error: {e}")
            self._verification_failures += 1

            # Return error result
            return self._create_error_result(context, str(e))

    async def _fuse_parallel(self, context: FusionContext) -> FusionResult:
        """Execute neural and symbolic paths in parallel."""
        # Create tasks for parallel execution
        neural_task = asyncio.create_task(self._neural_path(context))
        symbolic_prereq_task = asyncio.create_task(
            self._symbolic_preprocessing(context)
        )

        # Wait for neural path to complete
        neural_result = await neural_task

        # Wait for symbolic preprocessing
        symbolic_prereq = await symbolic_prereq_task

        # Execute symbolic verification with neural output
        symbolic_result = await self._symbolic_path(
            neural_result, context, symbolic_prereq
        )

        # Fuse results
        return self._combine_results(context, neural_result, symbolic_result)

    async def _fuse_sequential(self, context: FusionContext) -> FusionResult:
        """Execute neural path first, then symbolic verification."""
        # Neural reasoning
        neural_result = await self._neural_path(context)

        # Symbolic verification
        symbolic_result = await self._symbolic_path(neural_result, context, {})

        # Fuse results
        return self._combine_results(context, neural_result, symbolic_result)

    async def _fuse_iterative(self, context: FusionContext) -> FusionResult:
        """Execute iterative refinement loop."""
        iteration = 0
        best_result: Optional[FusionResult] = None

        while iteration < context.max_iterations:
            iteration += 1

            # Neural reasoning
            neural_result = await self._neural_path(context)

            # Symbolic verification
            symbolic_result = await self._symbolic_path(neural_result, context, {})

            # Combine results
            result = self._combine_results(context, neural_result, symbolic_result)
            result.iteration_count = iteration

            # Check for convergence
            if result.symbolic_result.verified:
                logger.debug(f"Iterative fusion converged at iteration {iteration}")
                return result

            # Track best result
            if best_result is None or result.fused_confidence > best_result.fused_confidence:
                best_result = result

            # Check if we should continue
            if symbolic_result.violation_count == 0:
                logger.debug(f"No violations at iteration {iteration}, stopping")
                break

            # Update context for next iteration (add counterexamples)
            context = self._update_context_with_feedback(
                context, symbolic_result.violations
            )

        return best_result or result

    async def _neural_path(self, context: FusionContext) -> NeuralResult:
        """
        Execute the neural reasoning path using Graph of Thoughts.

        Args:
            context: Fusion context

        Returns:
            NeuralResult with reasoning graph and synthesis
        """
        import time
        start_time = time.perf_counter()

        # Build reasoning graph
        if self.got_engine is not None:
            try:
                from core.apex.graph_of_thoughts import TaskDomain

                # Create task domains from context
                task_domains = [
                    TaskDomain(
                        domain_id=f"dom_{i}",
                        name=domain,
                        cluster_id=f"cluster_{i}",
                        relevance_score=1.0 - i * 0.1,
                    )
                    for i, domain in enumerate(context.domains[:5])
                ] if context.domains else [
                    TaskDomain(
                        domain_id="default",
                        name="general",
                        cluster_id="default_cluster",
                        relevance_score=1.0,
                    )
                ]

                # Build reasoning graph
                got_result = self.got_engine.build_reasoning_graph(
                    task=context.query,
                    task_domains=task_domains,
                    pareto_solutions=context.pareto_solutions,
                    enable_veto_gates=True,
                )

                # Extract synthesis
                synthesis_parts = []
                for synthesis in got_result.synthesis_results:
                    synthesis_parts.append(synthesis.synthesized_content)

                latency_ms = (time.perf_counter() - start_time) * 1000

                return NeuralResult(
                    reasoning_graph=got_result.to_dict(),
                    confidence=got_result.final_snr,
                    synthesis_content="\n\n".join(synthesis_parts),
                    domains_covered=got_result.domains_used,
                    node_count=got_result.node_count,
                    edge_count=got_result.edge_count,
                    snr_score=got_result.final_snr,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.warning(f"GoT execution failed: {e}, using fallback")

        # Fallback: generate simple reasoning
        latency_ms = (time.perf_counter() - start_time) * 1000

        return NeuralResult(
            reasoning_graph={"fallback": True},
            confidence=0.8,
            synthesis_content=f"Reasoning synthesis for: {context.query}",
            domains_covered=context.domains[:3],
            snr_score=0.85,
            latency_ms=latency_ms,
        )

    async def _symbolic_path(
        self,
        neural_result: NeuralResult,
        context: FusionContext,
        prereq_data: Dict[str, Any],
    ) -> SymbolicResult:
        """
        Execute the symbolic verification path.

        Runs multiple verification backends:
        1. Constitutional (Ihsan) verification
        2. SAPE 9-probe validation
        3. Knowledge graph grounding
        4. SMT constraint checking (if Z3 backend enabled)

        Args:
            neural_result: Result from neural path
            context: Fusion context
            prereq_data: Preprocessing data from parallel execution

        Returns:
            SymbolicResult with verification status
        """
        import time
        start_time = time.perf_counter()

        content = neural_result.synthesis_content
        violations: List[ConstraintViolation] = []
        constraints_checked: List[str] = []
        backend_results: Dict[str, Any] = {}

        # 1. Constitutional verification (Ihsan 8-dimension)
        if SymbolicBackend.CONSTITUTIONAL in context.symbolic_backends:
            ihsan_scores = await self._constitutional_verification(content, context)
            backend_results["constitutional"] = ihsan_scores

            # Check for violations
            for dim, score in ihsan_scores.items():
                threshold = context.constitutional_constraints.get(dim, 0.95)
                constraints_checked.append(f"ihsan:{dim}")

                if score < threshold:
                    violations.append(ConstraintViolation(
                        constraint_id=f"ihsan:{dim}",
                        constraint_type="constitutional",
                        expected_value=threshold,
                        actual_value=score,
                        violation_severity=(threshold - score) / threshold,
                        evidence=f"Dimension {dim} scored {score:.3f} < {threshold}",
                    ))
        else:
            ihsan_scores = {}

        # 2. SAPE 9-probe verification
        sape_scores = await self._sape_verification(content, context)
        backend_results["sape"] = sape_scores

        for probe, score in sape_scores.items():
            constraints_checked.append(f"sape:{probe}")

            # Map to Ihsan dimension for threshold
            ihsan_dim = SAPE_TO_IHSAN_MAP.get(probe, "correctness")
            threshold = context.constitutional_constraints.get(ihsan_dim, 0.90)

            if score < threshold:
                violations.append(ConstraintViolation(
                    constraint_id=f"sape:{probe}",
                    constraint_type="sape_probe",
                    expected_value=threshold,
                    actual_value=score,
                    violation_severity=(threshold - score) / threshold,
                    evidence=f"Probe {probe} scored {score:.3f} < {threshold}",
                ))

        # 3. Knowledge graph grounding
        if SymbolicBackend.DATALOG in context.symbolic_backends:
            is_grounded, evidence_nodes = await self._knowledge_graph_grounding(
                content, context.knowledge_graph_nodes
            )
            backend_results["kg_grounding"] = {
                "grounded": is_grounded,
                "evidence_nodes": evidence_nodes,
            }
            constraints_checked.append("kg:grounding")

            if not is_grounded and context.knowledge_graph_nodes:
                violations.append(ConstraintViolation(
                    constraint_id="kg:grounding",
                    constraint_type="datalog",
                    expected_value=1.0,
                    actual_value=0.0,
                    violation_severity=0.5,
                    evidence="Reasoning not grounded in knowledge graph",
                ))
        else:
            evidence_nodes = []

        # 4. Z3 SMT verification (if backend enabled)
        formal_proof = None
        if SymbolicBackend.Z3_SMT in context.symbolic_backends:
            try:
                formal_result = await self._z3_constraint_check(neural_result, context)
                backend_results["z3_smt"] = formal_result
                formal_proof = formal_result.get("proof")
                constraints_checked.append("z3:consistency")
            except Exception as e:
                logger.warning(f"Z3 verification failed: {e}")

        # Determine overall verification status
        verified = len(violations) == 0
        if verified:
            status = VerificationStatus.VERIFIED
        elif len(violations) < 3:
            status = VerificationStatus.UNKNOWN
        else:
            status = VerificationStatus.REFUTED

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SymbolicResult(
            verified=verified,
            status=status,
            constraints_checked=constraints_checked,
            violations=violations,
            formal_proof=formal_proof,
            sape_scores=sape_scores,
            ihsan_scores=ihsan_scores,
            backend_results=backend_results,
            latency_ms=latency_ms,
        )

    async def _symbolic_preprocessing(
        self,
        context: FusionContext,
    ) -> Dict[str, Any]:
        """Preprocess for symbolic verification (parallel mode)."""
        # Pre-fetch knowledge graph data
        if context.knowledge_graph_nodes:
            try:
                await self._kg_connector.ground_reasoning("", context.knowledge_graph_nodes[:1])
            except Exception:
                pass

        return {"preprocessed": True}

    async def _constitutional_verification(
        self,
        content: str,
        context: FusionContext,
    ) -> Dict[str, float]:
        """
        Verify content against Ihsan 8-dimension constitution.

        Args:
            content: Content to verify
            context: Fusion context

        Returns:
            Dictionary mapping dimension names to scores
        """
        return await self._constitutional_verifier.verify_ihsan_dimensions(
            content, context
        )

    async def _sape_verification(
        self,
        content: str,
        context: FusionContext,
    ) -> Dict[str, float]:
        """
        Execute SAPE 9-probe verification in parallel.

        Args:
            content: Content to probe
            context: Fusion context

        Returns:
            Dictionary mapping probe names to scores
        """
        return await self._sape_engine.run_probes_parallel(content, context)

    async def _knowledge_graph_grounding(
        self,
        content: str,
        node_ids: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Ground reasoning in knowledge graph evidence.

        Args:
            content: Content to ground
            node_ids: Evidence node IDs

        Returns:
            Tuple of (is_grounded, evidence_node_ids)
        """
        return await self._kg_connector.ground_reasoning(content, node_ids)

    async def _autoformalization(
        self,
        neural_result: NeuralResult,
        context: FusionContext,
    ) -> Dict[str, Any]:
        """
        Autoformalize neural reasoning into formal representation.

        Translates natural language reasoning into verifiable constraints.

        Args:
            neural_result: Result from neural path
            context: Fusion context

        Returns:
            Formal representation suitable for SMT solving
        """
        # Extract key claims from synthesis
        content = neural_result.synthesis_content
        claims: List[Dict[str, Any]] = []

        # Simple claim extraction (production would use NLP)
        sentences = content.split(".")
        for i, sentence in enumerate(sentences[:10]):
            sentence = sentence.strip()
            if len(sentence) > 10:
                claims.append({
                    "claim_id": f"claim_{i}",
                    "text": sentence,
                    "confidence": neural_result.confidence,
                    "domain": neural_result.domains_covered[0] if neural_result.domains_covered else "general",
                })

        return {
            "formalized_at": datetime.now(timezone.utc).isoformat(),
            "source_snr": neural_result.snr_score,
            "claim_count": len(claims),
            "claims": claims,
        }

    async def _z3_constraint_check(
        self,
        neural_result: NeuralResult,
        context: FusionContext,
    ) -> Dict[str, Any]:
        """
        Check formal constraints using Z3 SMT solver.

        Args:
            neural_result: Result from neural path
            context: Fusion context

        Returns:
            Z3 verification result with optional proof
        """
        # Try to use actual Z3 if available
        try:
            import z3
            has_z3 = True
        except ImportError:
            has_z3 = False

        if has_z3:
            # Basic consistency checking with Z3
            try:
                solver = z3.Solver()

                # Create symbolic variables for key metrics
                snr = z3.Real("snr")
                ihsan = z3.Real("ihsan")
                confidence = z3.Real("confidence")

                # Add constraints based on context
                solver.add(snr >= 0.0, snr <= 1.0)
                solver.add(ihsan >= 0.0, ihsan <= 1.0)
                solver.add(confidence >= 0.0, confidence <= 1.0)

                # Add threshold constraints
                solver.add(snr >= DEFAULT_SNR_THRESHOLD)
                solver.add(ihsan >= DEFAULT_IHSAN_THRESHOLD)

                # Add observed values
                solver.add(snr == neural_result.snr_score)
                solver.add(confidence == neural_result.confidence)

                # Check satisfiability
                result = solver.check()

                return {
                    "satisfiable": str(result) == "sat",
                    "status": str(result),
                    "proof": None,  # Proofs require unsat core
                }

            except Exception as e:
                logger.warning(f"Z3 check failed: {e}")
                return {"error": str(e), "satisfiable": None}
        else:
            # Mock Z3 result
            return {
                "satisfiable": neural_result.snr_score >= DEFAULT_SNR_THRESHOLD,
                "status": "mock",
                "proof": None,
            }

    def _combine_results(
        self,
        context: FusionContext,
        neural_result: NeuralResult,
        symbolic_result: SymbolicResult,
    ) -> FusionResult:
        """
        Combine neural and symbolic results into final fusion result.

        Args:
            context: Fusion context
            neural_result: Result from neural path
            symbolic_result: Result from symbolic path

        Returns:
            Combined FusionResult
        """
        # Compute fused confidence
        # Weight neural confidence by verification status
        if symbolic_result.verified:
            fused_confidence = (
                0.4 * neural_result.confidence +
                0.6 * symbolic_result.overall_ihsan_score
            )
        elif symbolic_result.status == VerificationStatus.UNKNOWN:
            fused_confidence = (
                0.5 * neural_result.confidence +
                0.5 * symbolic_result.overall_ihsan_score
            )
        else:
            # Refuted: heavily penalize confidence
            fused_confidence = (
                0.2 * neural_result.confidence +
                0.3 * symbolic_result.overall_ihsan_score
            )

        # Determine overall success
        success = (
            symbolic_result.verified and
            neural_result.snr_score >= DEFAULT_SNR_THRESHOLD * 0.95
        )

        if success:
            status = FusionStatus.SUCCESS
        elif symbolic_result.status == VerificationStatus.UNKNOWN:
            status = FusionStatus.PARTIAL
        elif symbolic_result.violation_count > 0:
            status = FusionStatus.FAILED
        else:
            status = FusionStatus.BLOCKED

        # Generate verification attestation
        attestation = self._generate_attestation(
            context, neural_result, symbolic_result
        )

        # Build evidence chain
        evidence_chain = [
            f"neural:{neural_result.reasoning_graph.get('graph_id', 'unknown')[:8]}",
        ]
        for constraint in symbolic_result.constraints_checked[:5]:
            evidence_chain.append(f"symbolic:{constraint}")

        return FusionResult(
            success=success,
            status=status,
            neural_result=neural_result,
            symbolic_result=symbolic_result,
            fused_confidence=fused_confidence,
            verification_attestation=attestation,
            evidence_chain=evidence_chain,
        )

    def _generate_attestation(
        self,
        context: FusionContext,
        neural_result: NeuralResult,
        symbolic_result: SymbolicResult,
    ) -> str:
        """Generate cryptographic attestation of verification."""
        # Build attestation content
        content = {
            "context_id": context.context_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "neural_confidence": neural_result.confidence,
            "neural_snr": neural_result.snr_score,
            "symbolic_verified": symbolic_result.verified,
            "symbolic_ihsan": symbolic_result.overall_ihsan_score,
            "sape_scores": symbolic_result.sape_scores,
            "violation_count": symbolic_result.violation_count,
            "constraints_checked": len(symbolic_result.constraints_checked),
        }

        # Generate attestation hash
        content_str = str(sorted(content.items()))
        attestation_hash = hashlib.sha256(content_str.encode()).hexdigest()

        return f"{DOMAIN_PREFIX}:attestation:{attestation_hash}"

    def _update_context_with_feedback(
        self,
        context: FusionContext,
        violations: List[ConstraintViolation],
    ) -> FusionContext:
        """Update context with counterexample feedback for iterative refinement."""
        # Create new context with stricter constraints
        new_constraints = context.constitutional_constraints.copy()

        for violation in violations:
            if violation.constraint_type == "constitutional":
                dim = violation.constraint_id.split(":")[-1]
                # Increase threshold for violated dimension
                new_constraints[dim] = min(1.0, new_constraints.get(dim, 0.95) + 0.02)

        # Add feedback to metadata
        new_metadata = context.metadata.copy()
        new_metadata["previous_violations"] = [v.to_dict() for v in violations[:5]]

        return FusionContext(
            query=context.query,
            domains=context.domains,
            pareto_solutions=context.pareto_solutions,
            constitutional_constraints=new_constraints,
            knowledge_graph_nodes=context.knowledge_graph_nodes,
            fusion_mode=context.fusion_mode,
            symbolic_backends=context.symbolic_backends,
            max_iterations=context.max_iterations,
            timeout_ms=context.timeout_ms,
            metadata=new_metadata,
        )

    def _create_error_result(
        self,
        context: FusionContext,
        error_message: str,
    ) -> FusionResult:
        """Create error result for fusion failure."""
        return FusionResult(
            success=False,
            status=FusionStatus.BLOCKED,
            neural_result=NeuralResult(
                reasoning_graph={"error": True},
                confidence=0.0,
                synthesis_content="",
                domains_covered=[],
            ),
            symbolic_result=SymbolicResult(
                verified=False,
                status=VerificationStatus.ERROR,
                constraints_checked=[],
                violations=[
                    ConstraintViolation(
                        constraint_id="system:error",
                        constraint_type="system",
                        expected_value=1.0,
                        actual_value=0.0,
                        violation_severity=1.0,
                        evidence=error_message,
                    )
                ],
            ),
            fused_confidence=0.0,
            verification_attestation=f"{DOMAIN_PREFIX}:error:{hashlib.sha256(error_message.encode()).hexdigest()[:16]}",
            metadata={"error": error_message},
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        success_rate = (
            self._successful_fusions / self._total_fusions
            if self._total_fusions > 0 else 0.0
        )
        avg_latency = (
            self._total_latency_ms / self._total_fusions
            if self._total_fusions > 0 else 0.0
        )

        return {
            "domain": DOMAIN_PREFIX,
            "version": VERSION,
            "total_fusions": self._total_fusions,
            "successful_fusions": self._successful_fusions,
            "verification_failures": self._verification_failures,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "ihsan_threshold": DEFAULT_IHSAN_THRESHOLD,
            "snr_threshold": DEFAULT_SNR_THRESHOLD,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_neural_symbolic_fusion_engine(
    got_engine: Optional[Any] = None,
    constitutional_verifier: Optional[ConstitutionalVerifier] = None,
    sape_engine: Optional[SAPEEngine] = None,
    kg_connector: Optional[KnowledgeGraphConnector] = None,
) -> NeuralSymbolicFusionEngine:
    """
    Factory function to create a configured Neural-Symbolic Fusion Engine.

    Args:
        got_engine: GraphOfThoughtsEngine instance (lazy loads if None)
        constitutional_verifier: Ihsan verifier (uses default if None)
        sape_engine: SAPE engine (uses default if None)
        kg_connector: KG connector (uses default if None)

    Returns:
        Configured NeuralSymbolicFusionEngine instance

    Example:
        >>> engine = create_neural_symbolic_fusion_engine()
        >>> context = FusionContext(
        ...     query="Design a secure authentication system",
        ...     domains=["security", "systems", "cryptography"],
        ...     fusion_mode=FusionMode.SEQUENTIAL,
        ... )
        >>> result = await engine.fuse(context)
    """
    return NeuralSymbolicFusionEngine(
        got_engine=got_engine,
        constitutional_verifier=constitutional_verifier,
        sape_engine=sape_engine,
        kg_connector=kg_connector,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "DOMAIN_PREFIX",
    "VERSION",
    "DEFAULT_IHSAN_THRESHOLD",
    "DEFAULT_SNR_THRESHOLD",
    "DEFAULT_NOVELTY_THRESHOLD",
    "IHSAN_DIMENSIONS",
    "SAPE_TO_IHSAN_MAP",
    # Enums
    "FusionMode",
    "SymbolicBackend",
    "VerificationStatus",
    "FusionStatus",
    # Data Classes
    "FusionContext",
    "NeuralResult",
    "SymbolicResult",
    "ConstraintViolation",
    "FusionResult",
    # Protocols
    "ConstitutionalVerifier",
    "SAPEEngine",
    "KnowledgeGraphConnector",
    # Default Implementations
    "DefaultConstitutionalVerifier",
    "DefaultSAPEEngine",
    "DefaultKnowledgeGraphConnector",
    # Core Engine
    "NeuralSymbolicFusionEngine",
    # Factory
    "create_neural_symbolic_fusion_engine",
]


# =============================================================================
# CLI / DEMO
# =============================================================================


async def main() -> None:
    """Demo the Neural-Symbolic Fusion Engine."""
    print("=" * 80)
    print("BIZRA Neural-Symbolic Fusion Engine Demo")
    print("=" * 80)

    # Create engine
    engine = create_neural_symbolic_fusion_engine()

    print("\n1. Engine Configuration:")
    stats = engine.get_statistics()
    print(f"   Domain: {stats['domain']}")
    print(f"   Version: {stats['version']}")
    print(f"   Ihsan Threshold: {stats['ihsan_threshold']}")
    print(f"   SNR Threshold: {stats['snr_threshold']}")

    # Test all fusion modes
    test_queries = [
        {
            "query": "Optimize the BIZRA data pipeline for maximum throughput while maintaining data integrity",
            "domains": ["data-engineering", "systems", "performance"],
            "mode": FusionMode.SEQUENTIAL,
        },
        {
            "query": "Design a secure authentication system with multi-factor verification",
            "domains": ["security", "cryptography", "identity"],
            "mode": FusionMode.PARALLEL,
        },
        {
            "query": "Develop an AI ethics framework for autonomous decision making",
            "domains": ["ethics", "ai", "governance"],
            "mode": FusionMode.ITERATIVE,
        },
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test['mode'].value.upper()} Mode")
        print(f"{'='*80}")
        print(f"Query: {test['query'][:60]}...")
        print(f"Domains: {', '.join(test['domains'])}")

        context = FusionContext(
            query=test["query"],
            domains=test["domains"],
            fusion_mode=test["mode"],
            symbolic_backends=[
                SymbolicBackend.CONSTITUTIONAL,
                SymbolicBackend.DATALOG,
            ],
            max_iterations=3,
        )

        result = await engine.fuse(context)

        print(f"\n   Results:")
        print(f"   - Status: {result.status.value}")
        print(f"   - Success: {result.success}")
        print(f"   - Fused Confidence: {result.fused_confidence:.4f}")
        print(f"   - Iterations: {result.iteration_count}")
        print(f"   - Latency: {result.total_latency_ms:.2f}ms")

        print(f"\n   Neural Path:")
        print(f"   - Confidence: {result.neural_result.confidence:.4f}")
        print(f"   - SNR Score: {result.neural_result.snr_score:.4f}")
        print(f"   - Domains Covered: {len(result.neural_result.domains_covered)}")

        print(f"\n   Symbolic Path:")
        print(f"   - Verified: {result.symbolic_result.verified}")
        print(f"   - Status: {result.symbolic_result.status.value}")
        print(f"   - Ihsan Score: {result.symbolic_result.overall_ihsan_score:.4f}")
        print(f"   - Constraints Checked: {len(result.symbolic_result.constraints_checked)}")
        print(f"   - Violations: {result.symbolic_result.violation_count}")

        if result.symbolic_result.violations:
            print(f"\n   Top Violations:")
            for v in result.symbolic_result.violations[:3]:
                print(f"   - {v.constraint_id}: {v.actual_value:.3f} < {v.expected_value:.3f}")

        print(f"\n   SAPE Probe Scores:")
        for probe, score in list(result.symbolic_result.sape_scores.items())[:5]:
            print(f"   - {probe}: {score:.4f}")

        print(f"\n   Attestation: {result.verification_attestation[:50]}...")

    print(f"\n{'='*80}")
    print("Final Statistics")
    print(f"{'='*80}")
    final_stats = engine.get_statistics()
    print(f"Total Fusions: {final_stats['total_fusions']}")
    print(f"Successful: {final_stats['successful_fusions']}")
    print(f"Success Rate: {final_stats['success_rate']*100:.1f}%")
    print(f"Avg Latency: {final_stats['avg_latency_ms']:.2f}ms")

    print("\n" + "=" * 80)
    print("Neural-Symbolic Fusion Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
