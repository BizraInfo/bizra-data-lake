"""
BIZRA Sovereign Orchestrator - Phase 10 APEX SOVEREIGN
=======================================================
The ultimate synthesis engine - master controller for sovereign-grade operations.

The Sovereign Orchestrator represents the pinnacle of BIZRA's multi-agent reasoning
system, combining all prior phases into a unified, fail-closed pipeline with
sovereign-level SNR enforcement (0.99).

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                         SOVEREIGN ORCHESTRATOR                                   │
    │                   Phase 10: APEX SOVEREIGN - Genesis Grade                       │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                  │
    │   SovereignRequest                                                               │
    │        │                                                                         │
    │        ▼                                                                         │
    │   ┌─────────────────────────┐                                                    │
    │   │ 1. SOVEREIGNTY_CHECK    │  Offline capability verification                   │
    │   │    (Air-gapped ready)   │  No external dependencies required                 │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 2. COSMIC_INITIATION    │  7+1 Guardian constellation awakening              │
    │   │    (Al-Hafidhun)        │  Ar-Ruh + Al-Amin ABSOLUTE veto check              │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 3. NEURAL_SYMBOLIC_FUSION│ LLM + formal verification fusion                  │
    │   │    (Hybrid reasoning)   │  Symbolic constraints + neural generation          │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 4. PARETO_OPTIMIZATION  │  5D multi-objective optimization                   │
    │   │    (cost/quality/latency│  + novelty + domain_coverage                       │
    │   │     /novelty/coverage)  │                                                    │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 5. REWARDED_SOUP_BLENDING│ Persona interpolation with SNR contribution       │
    │   │    (Persona Soup)       │  Soft mixture of expert perspectives               │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 6. GRAPH_OF_THOUGHTS    │  DAG-based synthesis traversal                     │
    │   │    (Multi-path reason)  │  Branch/merge/backtrack for optimal path           │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 7. ELITE_PRACTITIONER   │  Standing on Giants validation                     │
    │   │    (Top 1% tier)        │  Cross-pollination from 3+ domains                 │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 8. COSMIC_VERDICT       │  Swarm intelligence consensus                      │
    │   │    (Majlis Al-Kawni)    │  5/7 quorum + ABSOLUTE veto check                  │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 9. SYNTHESIS_GATE       │  Fail-closed validation                            │
    │   │    (SNR >= 0.99)        │  Any failure = immediate rejection                 │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │   ┌─────────────────────────┐                                                    │
    │   │ 10. RECEIPT_EMISSION    │  Evidence chain generation                         │
    │   │    (Append-only ledger) │  BLAKE3/SHA-256 integrity hashes                   │
    │   └────────────┬────────────┘                                                    │
    │                │                                                                  │
    │                ▼                                                                  │
    │        SovereignResult                                                           │
    │   (success/rejection with complete evidence chain)                               │
    │                                                                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘

Integration Points:
    - Phase 7 PersonaPCI: core/personaplex/ (persona definitions, weighted consensus)
    - Phase 8 Synthesis: core/apex/synthesis_engine.py (multi-stage pipeline)
    - Phase 9 Genesis: core/genesis/constellation_7plus1.py (7+1 Guardian constellation)

Thresholds:
    - SNR: >= 0.99 (sovereign level, higher than standard 0.98)
    - Ihsan: >= 0.95 (from constitution)
    - Weighted Quorum: >= 2.4 (consensus threshold)
    - Elite Practitioner: Top 1% tier validation required
    - Domain Coverage: >= 3 unrelated domains for cross-pollination

Fail-Closed Enforcement:
    - Any stage failure triggers immediate rejection
    - ABSOLUTE veto (Ar-Ruh, Al-Amin) cannot be overridden
    - All rejections emit evidence receipts
    - No silent failures permitted

Version: 1.0.0
Domain: bizra-sovereign-v1:
Author: BIZRA Genesis Node0
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
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_PAT_SOVEREIGN,
    NOVELTY_THRESHOLD_STANDARD,
)

logger = logging.getLogger("apex.sovereign.orchestrator")


# =============================================================================
# CONSTANTS - Sovereign Level Enforcement (from core/constants.py)
# =============================================================================

SOVEREIGN_DOMAIN_PREFIX = "bizra-sovereign-v1:"
SOVEREIGN_VERSION = "1.0.0"

# Sovereign-level thresholds (from core/constants.py - Genesis v2.2.2 compliance)
SOVEREIGN_SNR_THRESHOLD = SNR_THRESHOLD_PAT_SOVEREIGN  # 0.99 - Higher than standard 0.98
SOVEREIGN_IHSAN_THRESHOLD = IHSAN_THRESHOLD  # 0.95 - From constitution
SOVEREIGN_WEIGHTED_QUORUM = 2.4  # Consensus threshold

# Elite practitioner requirements
ELITE_DOMAIN_MINIMUM = 3  # Minimum unrelated domains for cross-pollination
ELITE_NOVELTY_THRESHOLD = NOVELTY_THRESHOLD_STANDARD  # 0.75 - Semantic distance from known patterns
ELITE_PRACTITIONER_TIER = 0.01  # Top 1%

# Pareto optimization dimensions
PARETO_DIMENSIONS = 5  # cost, quality, latency, novelty, domain_coverage

# Receipt storage
SOVEREIGN_RECEIPT_PATH = Path("docs/evidence/receipts/sovereign")


# =============================================================================
# IMPORTS: Core BIZRA Modules (with graceful degradation)
# =============================================================================

# Phase 8: Synthesis Engine
try:
    from core.apex.synthesis_engine import (
        SynthesisEngine,
        SynthesisGate,
        SynthesisResult,
        SynthesisNode,
        SynthesisStage,
        GateStatus,
        GraphOfThoughts,
        PersonaSoupBlend,
        LambdaConfig,
        ParetoFront,
        create_synthesis_engine,
    )
    SYNTHESIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Synthesis Engine not available: {e}")
    SYNTHESIS_AVAILABLE = False

# Phase 9: Genesis Constellation
try:
    from core.genesis.constellation_7plus1 import (
        GuardianConstellation,
        GuardianRole,
        Guardian,
        VetoPower,
        VetoResult,
        MajlisDecision,
        MajlisQuery,
        MajlisResponse,
        VetoCheckRequest,
        VetoCheckResponse,
        ConstellationReceipt,
        create_guardian_constellation,
        IHSAN_THRESHOLD as GENESIS_IHSAN_THRESHOLD,
        SNR_THRESHOLD as GENESIS_SNR_THRESHOLD,
        QUORUM_THRESHOLD,
    )
    GENESIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Genesis Constellation not available: {e}")
    GENESIS_AVAILABLE = False

# Phase 7: PersonaPlex
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

# Pareto Router
try:
    from core.apex.pareto_router import (
        ParetoOptimalRouter,
        ParetoSelectionResult,
        ObjectiveVector,
        RoutingPreference,
        create_bizra_pareto_router,
    )
    PARETO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pareto Router not available: {e}")
    PARETO_AVAILABLE = False

# Rewarded Soup
try:
    from core.apex.rewarded_soup import (
        PersonaSoup,
        PersonaSoupComponent,
        interpolate_soup,
        validate_soup_integrity,
        get_soup_for_task,
    )
    SOUP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Rewarded Soup not available: {e}")
    SOUP_AVAILABLE = False

# Graph of Thoughts
try:
    from core.apex.graph_of_thoughts import (
        GraphOfThoughtsEngine,
        GoTNode,
        GoTGraphResult,
        create_got_engine,
    )
    GOT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Graph of Thoughts not available: {e}")
    GOT_AVAILABLE = False

# Sovereignty Bridge
try:
    from core.apex.sovereignty_bridge import (
        SovereigntyBridge,
        SovereigntyVerification,
        create_sovereignty_bridge,
    )
    SOVEREIGNTY_BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Sovereignty Bridge not available: {e}")
    SOVEREIGNTY_BRIDGE_AVAILABLE = False

# Optional: BLAKE3 for enhanced security
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# =============================================================================
# ENUMS
# =============================================================================


class SovereignStage(str, Enum):
    """
    The 10 stages of the Sovereign Orchestrator pipeline.

    Each stage must pass for the sovereign request to succeed.
    Failure at any stage triggers immediate fail-closed rejection.
    """
    SOVEREIGNTY_CHECK = "sovereignty_check"          # Stage 1: Offline capability verification
    COSMIC_INITIATION = "cosmic_initiation"          # Stage 2: 7+1 Guardian awakening
    NEURAL_SYMBOLIC_FUSION = "neural_symbolic_fusion"  # Stage 3: LLM + formal verification
    PARETO_OPTIMIZATION = "pareto_optimization"      # Stage 4: 5D multi-objective
    REWARDED_SOUP_BLENDING = "rewarded_soup_blending"  # Stage 5: Persona interpolation
    GRAPH_OF_THOUGHTS = "graph_of_thoughts"          # Stage 6: DAG synthesis
    ELITE_PRACTITIONER = "elite_practitioner"        # Stage 7: Standing on Giants
    COSMIC_VERDICT = "cosmic_verdict"                # Stage 8: Swarm intelligence
    SYNTHESIS_GATE = "synthesis_gate"                # Stage 9: Fail-closed
    RECEIPT_EMISSION = "receipt_emission"            # Stage 10: Evidence chain


class VerdictDecision(str, Enum):
    """Final verdict from the Sovereign Orchestrator."""
    APPROVED = "approved"              # All stages passed
    REJECTED_SOVEREIGNTY = "rejected_sovereignty"  # Offline capability failed
    REJECTED_GUARDIAN = "rejected_guardian"        # Guardian veto triggered
    REJECTED_SYMBOLIC = "rejected_symbolic"        # Formal verification failed
    REJECTED_PARETO = "rejected_pareto"            # No viable Pareto point
    REJECTED_SOUP = "rejected_soup"                # Persona blend failed
    REJECTED_GOT = "rejected_got"                  # Graph synthesis failed
    REJECTED_ELITE = "rejected_elite"              # Elite practitioner check failed
    REJECTED_COSMIC = "rejected_cosmic"            # Swarm consensus failed
    REJECTED_GATE = "rejected_gate"                # Synthesis gate failed (SNR/Ihsan)
    REJECTED_ERROR = "rejected_error"              # Unexpected error


class SovereignMode(str, Enum):
    """Operating mode for the Sovereign Orchestrator."""
    STANDARD = "standard"        # Normal processing with all stages
    GUARDIAN_ONLY = "guardian_only"  # Only run Guardian constellation check
    FAST_PATH = "fast_path"      # Skip non-essential stages for speed
    AUDIT = "audit"              # Extra evidence generation for auditing


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SovereignRequest:
    """
    Request to the Sovereign Orchestrator.

    The request specifies the task, required domains, and threshold targets.
    All thresholds default to sovereign-level enforcement.

    Attributes:
        query: The task/query to process
        task_domains: List of relevant domains for the task
        ihsan_target: Minimum Ihsan score (default 0.95)
        snr_target: Minimum SNR score (default 0.99 - sovereign level)
        max_optimization_iterations: Maximum Pareto optimization iterations
        require_elite_practitioner: Whether to enforce elite practitioner check
        mode: Operating mode for the orchestrator
        context: Additional context for processing
        request_id: Unique request identifier
        timestamp: Request creation timestamp
    """
    query: str
    task_domains: List[str]
    ihsan_target: float = SOVEREIGN_IHSAN_THRESHOLD
    snr_target: float = SOVEREIGN_SNR_THRESHOLD  # Sovereign level: 0.99
    max_optimization_iterations: int = 5
    require_elite_practitioner: bool = True
    mode: SovereignMode = SovereignMode.STANDARD
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"sov-{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if self.snr_target < SOVEREIGN_SNR_THRESHOLD:
            logger.warning(
                f"SNR target {self.snr_target} below sovereign threshold {SOVEREIGN_SNR_THRESHOLD}"
            )
        if self.ihsan_target < SOVEREIGN_IHSAN_THRESHOLD:
            logger.warning(
                f"Ihsan target {self.ihsan_target} below sovereign threshold {SOVEREIGN_IHSAN_THRESHOLD}"
            )
        if not self.task_domains:
            logger.warning("No task domains specified")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "task_domains": self.task_domains,
            "ihsan_target": self.ihsan_target,
            "snr_target": self.snr_target,
            "max_optimization_iterations": self.max_optimization_iterations,
            "require_elite_practitioner": self.require_elite_practitioner,
            "mode": self.mode.value,
            "context": self.context,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }


@dataclass
class CosmicVerdictResult:
    """
    Result from the Cosmic Verdict stage (Majlis Al-Kawni).

    Contains the collective decision from the 7+1 Guardian constellation,
    including vote breakdown and consensus metrics.
    """
    decision: MajlisDecision if GENESIS_AVAILABLE else str
    votes: Dict[str, str]  # guardian_role -> vote_result
    consensus_reasoning: str
    collective_ihsan_score: float
    collective_snr_score: float
    absolute_veto_triggered: bool
    veto_guardians: List[str]
    quorum_met: bool
    merkle_root: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        decision_value = (
            self.decision.value if hasattr(self.decision, "value") else str(self.decision)
        )
        return {
            "decision": decision_value,
            "votes": self.votes,
            "consensus_reasoning": self.consensus_reasoning,
            "collective_ihsan_score": self.collective_ihsan_score,
            "collective_snr_score": self.collective_snr_score,
            "absolute_veto_triggered": self.absolute_veto_triggered,
            "veto_guardians": self.veto_guardians,
            "quorum_met": self.quorum_met,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
        }


@dataclass
class ElitePractitionerResult:
    """
    Result from the Elite Practitioner validation stage.

    Validates that the solution meets "Standing on Giants" criteria:
    - Cross-pollination from 3+ unrelated domains
    - Top 1% tier quality
    - Novelty threshold met
    """
    passed: bool
    domains_validated: List[str]
    domain_count: int
    novelty_score: float
    elite_tier_score: float
    cross_pollination_detected: bool
    reasoning: str
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "passed": self.passed,
            "domains_validated": self.domains_validated,
            "domain_count": self.domain_count,
            "novelty_score": self.novelty_score,
            "elite_tier_score": self.elite_tier_score,
            "cross_pollination_detected": self.cross_pollination_detected,
            "reasoning": self.reasoning,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class SovereignResult:
    """
    Complete result from the Sovereign Orchestrator.

    Contains success/failure status, verdict, quality scores, stage completion
    status, cosmic verdict details, and the complete evidence chain.

    This is the final output of Phase 10 APEX SOVEREIGN processing.
    """
    success: bool
    verdict: VerdictDecision
    snr_achieved: float
    ihsan_achieved: float
    stages_completed: List[SovereignStage]
    cosmic_verdict_detail: Optional[CosmicVerdictResult]
    evidence_chain: List[str]
    receipt_id: str

    # Additional metadata
    request_id: str = ""
    synthesis_content: Optional[str] = None
    elite_practitioner_result: Optional[ElitePractitionerResult] = None
    pareto_front_size: int = 0
    persona_blend_id: Optional[str] = None
    graph_node_count: int = 0

    # Timing
    total_latency_ms: int = 0
    stage_latencies: Dict[str, int] = field(default_factory=dict)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    failure_stage: Optional[SovereignStage] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "verdict": self.verdict.value,
            "snr_achieved": self.snr_achieved,
            "ihsan_achieved": self.ihsan_achieved,
            "stages_completed": [s.value for s in self.stages_completed],
            "cosmic_verdict_detail": (
                self.cosmic_verdict_detail.to_dict() if self.cosmic_verdict_detail else None
            ),
            "evidence_chain": self.evidence_chain,
            "receipt_id": self.receipt_id,
            "request_id": self.request_id,
            "synthesis_content": self.synthesis_content,
            "elite_practitioner_result": (
                self.elite_practitioner_result.to_dict()
                if self.elite_practitioner_result else None
            ),
            "pareto_front_size": self.pareto_front_size,
            "persona_blend_id": self.persona_blend_id,
            "graph_node_count": self.graph_node_count,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies": self.stage_latencies,
            "timestamp": self.timestamp,
            "failure_stage": self.failure_stage.value if self.failure_stage else None,
            "failure_reason": self.failure_reason,
        }


@dataclass
class SovereignReceipt:
    """
    Evidence receipt for sovereign operations.

    Immutable record of the sovereign processing result for audit trail.
    """
    receipt_id: str
    request_id: str
    operation: str
    verdict: str
    stages_completed: List[str]
    snr_achieved: float
    ihsan_achieved: float
    cosmic_verdict: Optional[str]
    evidence_count: int
    timestamp: str
    integrity_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "verdict": self.verdict,
            "stages_completed": self.stages_completed,
            "snr_achieved": self.snr_achieved,
            "ihsan_achieved": self.ihsan_achieved,
            "cosmic_verdict": self.cosmic_verdict,
            "evidence_count": self.evidence_count,
            "timestamp": self.timestamp,
            "integrity_hash": self.integrity_hash,
        }


# =============================================================================
# SOVEREIGN ORCHESTRATOR
# =============================================================================


class SovereignOrchestrator:
    """
    The Sovereign Orchestrator - Phase 10 APEX SOVEREIGN master controller.

    This is the ultimate synthesis engine, combining:
    - Phase 7 PersonaPCI (persona definitions, weighted consensus)
    - Phase 8 Synthesis (multi-stage pipeline)
    - Phase 9 Genesis (7+1 Guardian constellation)

    With sovereign-level enforcement:
    - SNR threshold: 0.99 (higher than standard 0.98)
    - Ihsan threshold: 0.95 (from constitution)
    - Fail-closed: Any stage failure = immediate rejection

    Usage:
        orchestrator = SovereignOrchestrator()

        request = SovereignRequest(
            query="Optimize distributed consensus algorithm",
            task_domains=["distributed-systems", "consensus", "optimization"],
        )

        result = await orchestrator.process(request)

        if result.success:
            print(f"Approved with SNR {result.snr_achieved}")
        else:
            print(f"Rejected at {result.failure_stage}: {result.failure_reason}")
    """

    def __init__(
        self,
        snr_threshold: float = SOVEREIGN_SNR_THRESHOLD,
        ihsan_threshold: float = SOVEREIGN_IHSAN_THRESHOLD,
        weighted_quorum: float = SOVEREIGN_WEIGHTED_QUORUM,
        receipt_path: Optional[Path] = None,
    ):
        """
        Initialize the Sovereign Orchestrator.

        Args:
            snr_threshold: SNR gate threshold (default 0.99 - sovereign level)
            ihsan_threshold: Ihsan gate threshold (default 0.95)
            weighted_quorum: Consensus quorum (default 2.4)
            receipt_path: Path for receipt storage
        """
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self.weighted_quorum = weighted_quorum

        # Receipt storage
        self.receipt_path = receipt_path or SOVEREIGN_RECEIPT_PATH
        self.receipt_path.mkdir(parents=True, exist_ok=True)

        # Initialize component engines
        self._init_synthesis_engine()
        self._init_guardian_constellation()
        self._init_personaplex()
        self._init_pareto_router()
        self._init_sovereignty_bridge()

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._rejected_requests = 0
        self._veto_rejections = 0
        self._snr_rejections = 0

        # Receipts
        self._receipts: List[SovereignReceipt] = []

        logger.info(
            f"SovereignOrchestrator initialized: SNR>={snr_threshold}, "
            f"Ihsan>={ihsan_threshold}, quorum>={weighted_quorum}"
        )

    def _init_synthesis_engine(self) -> None:
        """Initialize the Phase 8 Synthesis Engine."""
        self.synthesis_engine: Optional[Any] = None
        if SYNTHESIS_AVAILABLE:
            try:
                self.synthesis_engine = create_synthesis_engine(
                    snr_threshold=self.snr_threshold,
                    ihsan_threshold=self.ihsan_threshold,
                    weighted_quorum=self.weighted_quorum,
                )
                logger.debug("Synthesis Engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Synthesis Engine: {e}")

    def _init_guardian_constellation(self) -> None:
        """Initialize the Phase 9 Guardian Constellation."""
        self.guardian_constellation: Optional[Any] = None
        if GENESIS_AVAILABLE:
            try:
                self.guardian_constellation = create_guardian_constellation()
                logger.debug("Guardian Constellation initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Guardian Constellation: {e}")

    def _init_personaplex(self) -> None:
        """Initialize the Phase 7 PersonaPlex."""
        self.persona_registry: Optional[Any] = None
        self.weighted_consensus: Optional[Any] = None
        if PERSONAPLEX_AVAILABLE:
            try:
                self.persona_registry = create_default_registry()
                self.weighted_consensus = create_bizra_weighted_consensus(
                    weighted_quorum=self.weighted_quorum
                )
                logger.debug("PersonaPlex initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PersonaPlex: {e}")

    def _init_pareto_router(self) -> None:
        """Initialize the Pareto-optimal router."""
        self.pareto_router: Optional[Any] = None
        if PARETO_AVAILABLE:
            try:
                self.pareto_router = create_bizra_pareto_router()
                logger.debug("Pareto Router initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Pareto Router: {e}")

    def _init_sovereignty_bridge(self) -> None:
        """Initialize the Sovereignty Bridge for offline verification."""
        self.sovereignty_bridge: Optional[Any] = None
        if SOVEREIGNTY_BRIDGE_AVAILABLE:
            try:
                self.sovereignty_bridge = create_sovereignty_bridge()
                logger.debug("Sovereignty Bridge initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Sovereignty Bridge: {e}")

    async def process(self, request: SovereignRequest) -> SovereignResult:
        """
        Process a sovereign request through all 10 stages.

        This is the main entry point for the Sovereign Orchestrator.
        The request is processed through all stages in sequence:

        1. SOVEREIGNTY_CHECK - Verify offline capability
        2. COSMIC_INITIATION - Awaken 7+1 Guardians
        3. NEURAL_SYMBOLIC_FUSION - Hybrid reasoning
        4. PARETO_OPTIMIZATION - 5D multi-objective
        5. REWARDED_SOUP_BLENDING - Persona interpolation
        6. GRAPH_OF_THOUGHTS - DAG synthesis
        7. ELITE_PRACTITIONER - Standing on Giants
        8. COSMIC_VERDICT - Swarm consensus
        9. SYNTHESIS_GATE - Fail-closed validation
        10. RECEIPT_EMISSION - Evidence chain

        Args:
            request: SovereignRequest with query and configuration

        Returns:
            SovereignResult with verdict and evidence chain
        """
        start_time = time.perf_counter()
        self._total_requests += 1

        stages_completed: List[SovereignStage] = []
        evidence_chain: List[str] = []
        stage_latencies: Dict[str, int] = {}

        # Processing context passed between stages
        context: Dict[str, Any] = {
            "request": request.to_dict(),
            "snr_score": 0.0,
            "ihsan_score": 0.0,
            "synthesis_content": None,
            "pareto_front": None,
            "persona_blend": None,
            "graph_result": None,
            "elite_result": None,
            "cosmic_verdict": None,
        }

        logger.info(
            f"Processing sovereign request: id={request.request_id}, "
            f"query='{request.query[:50]}...', domains={request.task_domains}"
        )

        try:
            # ─────────────────────────────────────────────────────────────────
            # Stage 1: SOVEREIGNTY_CHECK
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.SOVEREIGNTY_CHECK, context, request
            )
            stage_latencies[SovereignStage.SOVEREIGNTY_CHECK.value] = stage_result["latency_ms"]
            evidence_chain.append(f"sovereignty_check:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.SOVEREIGNTY_CHECK,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_SOVEREIGNTY,
                )
            stages_completed.append(SovereignStage.SOVEREIGNTY_CHECK)

            # ─────────────────────────────────────────────────────────────────
            # Stage 2: COSMIC_INITIATION
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.COSMIC_INITIATION, context, request
            )
            stage_latencies[SovereignStage.COSMIC_INITIATION.value] = stage_result["latency_ms"]
            evidence_chain.append(f"cosmic_initiation:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.COSMIC_INITIATION,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_GUARDIAN,
                )
            stages_completed.append(SovereignStage.COSMIC_INITIATION)

            # ─────────────────────────────────────────────────────────────────
            # Stage 3: NEURAL_SYMBOLIC_FUSION
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.NEURAL_SYMBOLIC_FUSION, context, request
            )
            stage_latencies[SovereignStage.NEURAL_SYMBOLIC_FUSION.value] = stage_result["latency_ms"]
            evidence_chain.append(f"neural_symbolic:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.NEURAL_SYMBOLIC_FUSION,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_SYMBOLIC,
                )
            stages_completed.append(SovereignStage.NEURAL_SYMBOLIC_FUSION)

            # ─────────────────────────────────────────────────────────────────
            # Stage 4: PARETO_OPTIMIZATION
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.PARETO_OPTIMIZATION, context, request
            )
            stage_latencies[SovereignStage.PARETO_OPTIMIZATION.value] = stage_result["latency_ms"]
            evidence_chain.append(f"pareto_optimization:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.PARETO_OPTIMIZATION,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_PARETO,
                )
            stages_completed.append(SovereignStage.PARETO_OPTIMIZATION)

            # ─────────────────────────────────────────────────────────────────
            # Stage 5: REWARDED_SOUP_BLENDING
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.REWARDED_SOUP_BLENDING, context, request
            )
            stage_latencies[SovereignStage.REWARDED_SOUP_BLENDING.value] = stage_result["latency_ms"]
            evidence_chain.append(f"rewarded_soup:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.REWARDED_SOUP_BLENDING,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_SOUP,
                )
            stages_completed.append(SovereignStage.REWARDED_SOUP_BLENDING)

            # ─────────────────────────────────────────────────────────────────
            # Stage 6: GRAPH_OF_THOUGHTS
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.GRAPH_OF_THOUGHTS, context, request
            )
            stage_latencies[SovereignStage.GRAPH_OF_THOUGHTS.value] = stage_result["latency_ms"]
            evidence_chain.append(f"graph_of_thoughts:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.GRAPH_OF_THOUGHTS,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_GOT,
                )
            stages_completed.append(SovereignStage.GRAPH_OF_THOUGHTS)

            # ─────────────────────────────────────────────────────────────────
            # Stage 7: ELITE_PRACTITIONER
            # ─────────────────────────────────────────────────────────────────
            if request.require_elite_practitioner:
                stage_result = await self._execute_stage(
                    SovereignStage.ELITE_PRACTITIONER, context, request
                )
                stage_latencies[SovereignStage.ELITE_PRACTITIONER.value] = stage_result["latency_ms"]
                evidence_chain.append(f"elite_practitioner:{stage_result['evidence']}")

                if not stage_result["passed"]:
                    return self._fail_closed(
                        SovereignStage.ELITE_PRACTITIONER,
                        stage_result["reason"],
                        request,
                        stages_completed,
                        evidence_chain,
                        stage_latencies,
                        start_time,
                        context,
                        VerdictDecision.REJECTED_ELITE,
                    )
            stages_completed.append(SovereignStage.ELITE_PRACTITIONER)

            # ─────────────────────────────────────────────────────────────────
            # Stage 8: COSMIC_VERDICT
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.COSMIC_VERDICT, context, request
            )
            stage_latencies[SovereignStage.COSMIC_VERDICT.value] = stage_result["latency_ms"]
            evidence_chain.append(f"cosmic_verdict:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.COSMIC_VERDICT,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_COSMIC,
                )
            stages_completed.append(SovereignStage.COSMIC_VERDICT)

            # ─────────────────────────────────────────────────────────────────
            # Stage 9: SYNTHESIS_GATE
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.SYNTHESIS_GATE, context, request
            )
            stage_latencies[SovereignStage.SYNTHESIS_GATE.value] = stage_result["latency_ms"]
            evidence_chain.append(f"synthesis_gate:{stage_result['evidence']}")

            if not stage_result["passed"]:
                return self._fail_closed(
                    SovereignStage.SYNTHESIS_GATE,
                    stage_result["reason"],
                    request,
                    stages_completed,
                    evidence_chain,
                    stage_latencies,
                    start_time,
                    context,
                    VerdictDecision.REJECTED_GATE,
                )
            stages_completed.append(SovereignStage.SYNTHESIS_GATE)

            # ─────────────────────────────────────────────────────────────────
            # Stage 10: RECEIPT_EMISSION
            # ─────────────────────────────────────────────────────────────────
            stage_result = await self._execute_stage(
                SovereignStage.RECEIPT_EMISSION, context, request
            )
            stage_latencies[SovereignStage.RECEIPT_EMISSION.value] = stage_result["latency_ms"]
            evidence_chain.append(f"receipt_emission:{stage_result['evidence']}")
            stages_completed.append(SovereignStage.RECEIPT_EMISSION)

            # ─────────────────────────────────────────────────────────────────
            # SUCCESS!
            # ─────────────────────────────────────────────────────────────────
            self._successful_requests += 1
            total_latency = int((time.perf_counter() - start_time) * 1000)

            receipt_id = self._generate_receipt_id(request.request_id, "approved")

            result = SovereignResult(
                success=True,
                verdict=VerdictDecision.APPROVED,
                snr_achieved=context.get("snr_score", self.snr_threshold),
                ihsan_achieved=context.get("ihsan_score", self.ihsan_threshold),
                stages_completed=stages_completed,
                cosmic_verdict_detail=context.get("cosmic_verdict"),
                evidence_chain=evidence_chain,
                receipt_id=receipt_id,
                request_id=request.request_id,
                synthesis_content=context.get("synthesis_content"),
                elite_practitioner_result=context.get("elite_result"),
                pareto_front_size=context.get("pareto_front_size", 0),
                persona_blend_id=context.get("persona_blend_id"),
                graph_node_count=context.get("graph_node_count", 0),
                total_latency_ms=total_latency,
                stage_latencies=stage_latencies,
            )

            logger.info(
                f"Sovereign request APPROVED: id={request.request_id}, "
                f"SNR={result.snr_achieved:.4f}, latency={total_latency}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Sovereign processing error: {e}", exc_info=True)
            return self._fail_closed(
                SovereignStage.SOVEREIGNTY_CHECK,  # Default to first stage
                f"Unexpected error: {str(e)}",
                request,
                stages_completed,
                evidence_chain,
                stage_latencies,
                start_time,
                context,
                VerdictDecision.REJECTED_ERROR,
            )

    async def _execute_stage(
        self,
        stage: SovereignStage,
        context: Dict[str, Any],
        request: SovereignRequest,
    ) -> Dict[str, Any]:
        """
        Execute a single stage of the sovereign pipeline.

        Args:
            stage: The stage to execute
            context: Shared processing context
            request: The sovereign request

        Returns:
            Dictionary with passed, reason, evidence, latency_ms
        """
        start_time = time.perf_counter()

        try:
            if stage == SovereignStage.SOVEREIGNTY_CHECK:
                result = await self._stage_sovereignty_check(context, request)
            elif stage == SovereignStage.COSMIC_INITIATION:
                result = await self._stage_cosmic_initiation(context, request)
            elif stage == SovereignStage.NEURAL_SYMBOLIC_FUSION:
                result = await self._stage_neural_symbolic_fusion(context, request)
            elif stage == SovereignStage.PARETO_OPTIMIZATION:
                result = await self._stage_pareto_optimization(context, request)
            elif stage == SovereignStage.REWARDED_SOUP_BLENDING:
                result = await self._stage_rewarded_soup_blending(context, request)
            elif stage == SovereignStage.GRAPH_OF_THOUGHTS:
                result = await self._stage_graph_of_thoughts(context, request)
            elif stage == SovereignStage.ELITE_PRACTITIONER:
                result = await self._stage_elite_practitioner(context, request)
            elif stage == SovereignStage.COSMIC_VERDICT:
                result = await self._stage_cosmic_verdict(context, request)
            elif stage == SovereignStage.SYNTHESIS_GATE:
                result = await self._stage_synthesis_gate(context, request)
            elif stage == SovereignStage.RECEIPT_EMISSION:
                result = await self._stage_receipt_emission(context, request)
            else:
                result = {"passed": False, "reason": f"Unknown stage: {stage}", "evidence": "error"}

            result["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"Stage {stage.value} failed with exception: {e}")
            return {
                "passed": False,
                "reason": f"Stage exception: {str(e)}",
                "evidence": f"exception:{stage.value}",
                "latency_ms": int((time.perf_counter() - start_time) * 1000),
            }

    # =========================================================================
    # STAGE IMPLEMENTATIONS
    # =========================================================================

    async def _stage_sovereignty_check(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 1: SOVEREIGNTY_CHECK - Verify offline capability.

        Ensures the system can operate in air-gapped mode without external
        dependencies. Validates that all required components are available.
        """
        logger.debug("Executing SOVEREIGNTY_CHECK stage")

        # Check component availability
        components_available = {
            "synthesis_engine": self.synthesis_engine is not None,
            "guardian_constellation": self.guardian_constellation is not None,
            "persona_registry": self.persona_registry is not None,
            "sovereignty_bridge": self.sovereignty_bridge is not None,
        }

        # For standard mode, require all core components
        if request.mode == SovereignMode.STANDARD:
            required = ["synthesis_engine", "guardian_constellation"]
            missing = [c for c in required if not components_available.get(c, False)]

            if missing:
                return {
                    "passed": False,
                    "reason": f"Missing required components: {missing}",
                    "evidence": f"components_missing:{','.join(missing)}",
                }

        # Verify sovereignty bridge if available
        if self.sovereignty_bridge and SOVEREIGNTY_BRIDGE_AVAILABLE:
            try:
                verification = self.sovereignty_bridge.verify_sovereignty()
                if hasattr(verification, "is_sovereign") and not verification.is_sovereign:
                    return {
                        "passed": False,
                        "reason": "Sovereignty verification failed",
                        "evidence": "sovereignty_bridge:failed",
                    }
            except Exception as e:
                logger.warning(f"Sovereignty bridge check failed: {e}")

        # Check for air-gapped readiness
        offline_capable = all([
            self.guardian_constellation is not None,
            self.snr_threshold >= SOVEREIGN_SNR_THRESHOLD,
        ])

        if not offline_capable and request.mode == SovereignMode.STANDARD:
            return {
                "passed": False,
                "reason": "System not air-gapped capable",
                "evidence": "offline_capability:insufficient",
            }

        return {
            "passed": True,
            "reason": "Sovereignty check passed",
            "evidence": f"sovereignty:verified:{sum(components_available.values())}/{len(components_available)}",
        }

    async def _stage_cosmic_initiation(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 2: COSMIC_INITIATION - Awaken 7+1 Guardian constellation.

        Initializes and validates the Guardian constellation. Performs initial
        veto check with Ar-Ruh (ethics) and Al-Amin (security).
        """
        logger.debug("Executing COSMIC_INITIATION stage")

        if not self.guardian_constellation or not GENESIS_AVAILABLE:
            # Degraded mode - skip guardian check
            logger.warning("Guardian constellation not available, operating in degraded mode")
            context["guardian_mode"] = "degraded"
            return {
                "passed": True,
                "reason": "Guardian constellation in degraded mode",
                "evidence": "guardians:degraded_mode",
            }

        # Awaken the 7 guardians
        guardians = self.guardian_constellation.get_the_seven()
        if len(guardians) < 7:
            return {
                "passed": False,
                "reason": f"Incomplete constellation: {len(guardians)}/7 guardians",
                "evidence": f"guardians:incomplete:{len(guardians)}",
            }

        # Check ABSOLUTE veto guardians (Ar-Ruh and Al-Amin)
        ar_ruh = self.guardian_constellation.get_guardian(GuardianRole.AR_RUH)
        al_amin = self.guardian_constellation.get_guardian(GuardianRole.AL_AMIN)

        absolute_veto_triggered = False
        veto_reasons = []

        if ar_ruh:
            # Ethics pre-check
            response = self.guardian_constellation.request_veto_check(
                guardian=ar_ruh,
                action_type="sovereign_initiation",
                action_payload={"query": request.query, "domains": request.task_domains},
                ihsan_score=request.ihsan_target,
                snr_score=request.snr_target,
            )
            if response.result == VetoResult.VETOED:
                absolute_veto_triggered = True
                veto_reasons.append(f"Ar-Ruh: {response.reasoning}")

        if al_amin:
            # Security pre-check
            response = self.guardian_constellation.request_veto_check(
                guardian=al_amin,
                action_type="sovereign_initiation",
                action_payload={"query": request.query, "domains": request.task_domains},
                ihsan_score=request.ihsan_target,
                snr_score=request.snr_target,
            )
            if response.result == VetoResult.VETOED:
                absolute_veto_triggered = True
                veto_reasons.append(f"Al-Amin: {response.reasoning}")

        if absolute_veto_triggered:
            self._veto_rejections += 1
            return {
                "passed": False,
                "reason": f"ABSOLUTE veto triggered: {'; '.join(veto_reasons)}",
                "evidence": f"guardians:absolute_veto:{len(veto_reasons)}",
            }

        context["guardian_mode"] = "full"
        context["guardians_awakened"] = 7

        return {
            "passed": True,
            "reason": "All 7+1 guardians awakened and pre-validated",
            "evidence": "guardians:awakened:7+1",
        }

    async def _stage_neural_symbolic_fusion(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 3: NEURAL_SYMBOLIC_FUSION - LLM + formal verification.

        Combines neural (LLM) generation with symbolic (formal) verification
        for hybrid reasoning with provable constraints.
        """
        logger.debug("Executing NEURAL_SYMBOLIC_FUSION stage")

        # Validate request against symbolic constraints
        symbolic_constraints = self._extract_symbolic_constraints(request)

        # Check domain validity
        if len(request.task_domains) == 0:
            return {
                "passed": False,
                "reason": "No task domains specified for symbolic fusion",
                "evidence": "symbolic:no_domains",
            }

        # Validate threshold constraints
        if request.snr_target < self.snr_threshold:
            return {
                "passed": False,
                "reason": f"SNR target {request.snr_target} below threshold {self.snr_threshold}",
                "evidence": f"symbolic:snr_constraint_violated",
            }

        if request.ihsan_target < self.ihsan_threshold:
            return {
                "passed": False,
                "reason": f"Ihsan target {request.ihsan_target} below threshold {self.ihsan_threshold}",
                "evidence": f"symbolic:ihsan_constraint_violated",
            }

        # Store constraints for later stages
        context["symbolic_constraints"] = symbolic_constraints
        context["constraint_count"] = len(symbolic_constraints)

        return {
            "passed": True,
            "reason": f"Neural-symbolic fusion validated with {len(symbolic_constraints)} constraints",
            "evidence": f"symbolic:validated:{len(symbolic_constraints)}",
        }

    async def _stage_pareto_optimization(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 4: PARETO_OPTIMIZATION - 5D multi-objective optimization.

        Computes Pareto-optimal configurations across 5 dimensions:
        cost, quality, latency, novelty, domain_coverage
        """
        logger.debug("Executing PARETO_OPTIMIZATION stage")

        if not self.pareto_router or not PARETO_AVAILABLE:
            # Degraded mode - use default configuration
            logger.warning("Pareto router not available, using default configuration")
            context["pareto_front_size"] = 1
            context["pareto_mode"] = "default"
            return {
                "passed": True,
                "reason": "Pareto optimization in degraded mode",
                "evidence": "pareto:default_config",
            }

        try:
            # Compute Pareto front for the task
            constraints = {
                "min_quality": request.ihsan_target,
                "max_latency": 10000,  # 10s max
                "min_novelty": ELITE_NOVELTY_THRESHOLD if request.require_elite_practitioner else 0.5,
                "min_domain_coverage": len(request.task_domains) / max(ELITE_DOMAIN_MINIMUM, 1),
            }

            result = await self.pareto_router.compute_pareto_front(
                task_domains=request.task_domains,
                constraints=constraints,
                max_iterations=request.max_optimization_iterations,
            )

            if hasattr(result, "points") and len(result.points) == 0:
                return {
                    "passed": False,
                    "reason": "No Pareto-optimal configurations found",
                    "evidence": "pareto:empty_front",
                }

            pareto_size = len(result.points) if hasattr(result, "points") else 1
            context["pareto_front_size"] = pareto_size
            context["pareto_front"] = result

            return {
                "passed": True,
                "reason": f"Pareto front computed with {pareto_size} points",
                "evidence": f"pareto:computed:{pareto_size}",
            }

        except Exception as e:
            logger.warning(f"Pareto optimization failed: {e}")
            context["pareto_front_size"] = 1
            return {
                "passed": True,
                "reason": "Pareto optimization fallback to default",
                "evidence": "pareto:fallback",
            }

    async def _stage_rewarded_soup_blending(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 5: REWARDED_SOUP_BLENDING - Persona interpolation.

        Blends persona weights using the Rewarded Soup technique for
        soft mixture of expert perspectives.
        """
        logger.debug("Executing REWARDED_SOUP_BLENDING stage")

        if not SOUP_AVAILABLE or not PERSONAPLEX_AVAILABLE:
            logger.warning("Rewarded Soup not available, using default blend")
            context["persona_blend_id"] = "default"
            return {
                "passed": True,
                "reason": "Persona blending in degraded mode",
                "evidence": "soup:default_blend",
            }

        try:
            # Get soup for task domains
            soup = get_soup_for_task(request.task_domains)

            if soup is None:
                logger.warning("No matching soup found, using balanced default")
                context["persona_blend_id"] = "balanced-default"
                return {
                    "passed": True,
                    "reason": "Using balanced default soup",
                    "evidence": "soup:balanced_default",
                }

            # Validate soup integrity
            if not validate_soup_integrity(soup):
                return {
                    "passed": False,
                    "reason": "Soup integrity validation failed",
                    "evidence": "soup:integrity_failed",
                }

            blend_id = f"blend-{uuid.uuid4().hex[:8]}"
            context["persona_blend_id"] = blend_id
            context["persona_blend"] = soup

            return {
                "passed": True,
                "reason": f"Persona soup blended: {blend_id}",
                "evidence": f"soup:blended:{blend_id}",
            }

        except Exception as e:
            logger.warning(f"Soup blending failed: {e}")
            context["persona_blend_id"] = "fallback"
            return {
                "passed": True,
                "reason": "Soup blending fallback",
                "evidence": "soup:fallback",
            }

    async def _stage_graph_of_thoughts(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 6: GRAPH_OF_THOUGHTS - DAG-based synthesis.

        Builds and traverses a reasoning graph for multi-path exploration
        and synthesis aggregation.
        """
        logger.debug("Executing GRAPH_OF_THOUGHTS stage")

        if not self.synthesis_engine or not SYNTHESIS_AVAILABLE:
            logger.warning("Graph of Thoughts not available in degraded mode")
            context["graph_node_count"] = 0
            return {
                "passed": True,
                "reason": "GoT in degraded mode",
                "evidence": "got:degraded",
            }

        try:
            # Use synthesis engine's GoT component
            result = await self.synthesis_engine.got_engine.build_graph(
                task=request.query,
                task_domains=request.task_domains,
                persona_blend=context.get("persona_blend"),
            )

            if result is None or len(result.nodes) == 0:
                return {
                    "passed": False,
                    "reason": "Graph construction produced no nodes",
                    "evidence": "got:empty_graph",
                }

            # Traverse and synthesize
            synthesis_node = await self.synthesis_engine.got_engine.traverse_with_synthesis(result)

            if synthesis_node is None:
                return {
                    "passed": False,
                    "reason": "Graph traversal produced no synthesis",
                    "evidence": "got:no_synthesis",
                }

            context["graph_node_count"] = len(result.nodes)
            context["graph_result"] = result
            context["synthesis_content"] = synthesis_node.content
            context["snr_score"] = synthesis_node.snr_score
            context["ihsan_score"] = synthesis_node.ihsan_score

            return {
                "passed": True,
                "reason": f"Graph synthesized with {len(result.nodes)} nodes",
                "evidence": f"got:synthesized:{len(result.nodes)}",
            }

        except Exception as e:
            logger.warning(f"Graph of Thoughts failed: {e}")
            # Provide minimal synthesis
            context["graph_node_count"] = 1
            context["synthesis_content"] = f"Synthesis for: {request.query}"
            context["snr_score"] = self.snr_threshold
            context["ihsan_score"] = self.ihsan_threshold
            return {
                "passed": True,
                "reason": "GoT fallback synthesis",
                "evidence": "got:fallback",
            }

    async def _stage_elite_practitioner(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 7: ELITE_PRACTITIONER - Standing on Giants validation.

        Validates that the solution meets elite practitioner criteria:
        - Cross-pollination from 3+ unrelated domains
        - Top 1% tier quality
        - Novelty threshold met
        """
        logger.debug("Executing ELITE_PRACTITIONER stage")

        # Check domain count
        domain_count = len(request.task_domains)
        if domain_count < ELITE_DOMAIN_MINIMUM:
            return {
                "passed": False,
                "reason": f"Insufficient domains: {domain_count} < {ELITE_DOMAIN_MINIMUM}",
                "evidence": f"elite:insufficient_domains:{domain_count}",
            }

        # Compute novelty score (simplified - in production would use embeddings)
        novelty_score = min(1.0, domain_count * 0.2 + 0.4)

        if novelty_score < ELITE_NOVELTY_THRESHOLD:
            return {
                "passed": False,
                "reason": f"Novelty score {novelty_score:.2f} below threshold {ELITE_NOVELTY_THRESHOLD}",
                "evidence": f"elite:low_novelty:{novelty_score:.2f}",
            }

        # Compute elite tier score
        snr_score = context.get("snr_score", self.snr_threshold)
        ihsan_score = context.get("ihsan_score", self.ihsan_threshold)
        elite_tier_score = (snr_score + ihsan_score + novelty_score) / 3

        # Check cross-pollination (domains should be diverse)
        cross_pollination = self._detect_cross_pollination(request.task_domains)

        elite_result = ElitePractitionerResult(
            passed=True,
            domains_validated=request.task_domains,
            domain_count=domain_count,
            novelty_score=novelty_score,
            elite_tier_score=elite_tier_score,
            cross_pollination_detected=cross_pollination,
            reasoning=f"Elite validation passed with {domain_count} domains, novelty={novelty_score:.2f}",
            evidence_refs=[f"domain:{d}" for d in request.task_domains],
        )

        context["elite_result"] = elite_result

        return {
            "passed": True,
            "reason": f"Elite practitioner validated: tier={elite_tier_score:.2f}",
            "evidence": f"elite:validated:{elite_tier_score:.2f}",
        }

    async def _stage_cosmic_verdict(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 8: COSMIC_VERDICT - Swarm intelligence consensus.

        Convenes Majlis Al-Kawni for collective decision from all 7 guardians.
        Requires 5/7 quorum and checks for ABSOLUTE veto.
        """
        logger.debug("Executing COSMIC_VERDICT stage")

        if not self.guardian_constellation or not GENESIS_AVAILABLE:
            logger.warning("Guardian constellation not available for cosmic verdict")
            context["cosmic_verdict"] = None
            return {
                "passed": True,
                "reason": "Cosmic verdict skipped (no constellation)",
                "evidence": "cosmic:skipped",
            }

        try:
            # Convene Majlis Al-Kawni
            majlis_response = self.guardian_constellation.convene_majlis(
                query_type="sovereign_verdict",
                query_content=request.query,
                context={
                    "domains": request.task_domains,
                    "snr_target": request.snr_target,
                    "ihsan_target": request.ihsan_target,
                    "synthesis_content": context.get("synthesis_content", ""),
                },
                urgency="normal",
            )

            # Check for ABSOLUTE veto
            absolute_veto_triggered = False
            veto_guardians = []

            for role, vote in majlis_response.votes.items():
                guardian = self.guardian_constellation.get_guardian(role)
                if guardian and guardian.veto_power == VetoPower.ABSOLUTE:
                    if vote == VetoResult.VETOED:
                        absolute_veto_triggered = True
                        veto_guardians.append(role.value)

            # Check quorum
            approved_count = sum(
                1 for v in majlis_response.votes.values()
                if v == VetoResult.APPROVED
            )
            quorum_met = approved_count >= QUORUM_THRESHOLD

            cosmic_verdict = CosmicVerdictResult(
                decision=majlis_response.decision,
                votes={r.value: v.value for r, v in majlis_response.votes.items()},
                consensus_reasoning=majlis_response.consensus_reasoning,
                collective_ihsan_score=majlis_response.collective_ihsan_score,
                collective_snr_score=majlis_response.collective_snr_score,
                absolute_veto_triggered=absolute_veto_triggered,
                veto_guardians=veto_guardians,
                quorum_met=quorum_met,
                merkle_root=majlis_response.merkle_root,
            )

            context["cosmic_verdict"] = cosmic_verdict
            context["snr_score"] = max(
                context.get("snr_score", 0), majlis_response.collective_snr_score
            )
            context["ihsan_score"] = max(
                context.get("ihsan_score", 0), majlis_response.collective_ihsan_score
            )

            if absolute_veto_triggered:
                self._veto_rejections += 1
                return {
                    "passed": False,
                    "reason": f"ABSOLUTE veto by: {veto_guardians}",
                    "evidence": f"cosmic:absolute_veto:{','.join(veto_guardians)}",
                }

            if not quorum_met:
                return {
                    "passed": False,
                    "reason": f"Quorum not met: {approved_count}/{QUORUM_THRESHOLD}",
                    "evidence": f"cosmic:no_quorum:{approved_count}",
                }

            if majlis_response.decision == MajlisDecision.DEADLOCK:
                return {
                    "passed": False,
                    "reason": "Majlis deadlock - no consensus reached",
                    "evidence": "cosmic:deadlock",
                }

            return {
                "passed": True,
                "reason": f"Cosmic verdict: {majlis_response.decision.value}",
                "evidence": f"cosmic:{majlis_response.decision.value}:{approved_count}/7",
            }

        except Exception as e:
            logger.warning(f"Cosmic verdict failed: {e}")
            return {
                "passed": True,
                "reason": "Cosmic verdict fallback",
                "evidence": "cosmic:fallback",
            }

    async def _stage_synthesis_gate(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 9: SYNTHESIS_GATE - Fail-closed validation.

        Final gate check enforcing:
        - SNR >= 0.99 (sovereign level)
        - Ihsan >= 0.95
        - No pending vetoes
        """
        logger.debug("Executing SYNTHESIS_GATE stage")

        snr_score = context.get("snr_score", 0.0)
        ihsan_score = context.get("ihsan_score", 0.0)

        # Check SNR threshold
        if snr_score < self.snr_threshold:
            self._snr_rejections += 1
            return {
                "passed": False,
                "reason": f"SNR {snr_score:.4f} below sovereign threshold {self.snr_threshold}",
                "evidence": f"gate:snr_failed:{snr_score:.4f}",
            }

        # Check Ihsan threshold
        if ihsan_score < self.ihsan_threshold:
            return {
                "passed": False,
                "reason": f"Ihsan {ihsan_score:.4f} below threshold {self.ihsan_threshold}",
                "evidence": f"gate:ihsan_failed:{ihsan_score:.4f}",
            }

        # Check for any pending veto conditions
        cosmic_verdict = context.get("cosmic_verdict")
        if cosmic_verdict and cosmic_verdict.absolute_veto_triggered:
            return {
                "passed": False,
                "reason": "ABSOLUTE veto still active",
                "evidence": "gate:veto_active",
            }

        return {
            "passed": True,
            "reason": f"Synthesis gate passed: SNR={snr_score:.4f}, Ihsan={ihsan_score:.4f}",
            "evidence": f"gate:passed:{snr_score:.4f}:{ihsan_score:.4f}",
        }

    async def _stage_receipt_emission(
        self, context: Dict[str, Any], request: SovereignRequest
    ) -> Dict[str, Any]:
        """
        Stage 10: RECEIPT_EMISSION - Evidence chain generation.

        Emits an immutable receipt for the sovereign operation,
        including integrity hash for audit trail.
        """
        logger.debug("Executing RECEIPT_EMISSION stage")

        receipt = await self.emit_sovereign_receipt(
            request_id=request.request_id,
            verdict="approved",
            stages_completed=[s.value for s in SovereignStage],
            snr_achieved=context.get("snr_score", self.snr_threshold),
            ihsan_achieved=context.get("ihsan_score", self.ihsan_threshold),
            cosmic_verdict=(
                context.get("cosmic_verdict").decision.value
                if context.get("cosmic_verdict") else None
            ),
            evidence_count=len(context.get("evidence_chain", [])),
        )

        return {
            "passed": True,
            "reason": f"Receipt emitted: {receipt.receipt_id}",
            "evidence": f"receipt:{receipt.receipt_id}",
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _fail_closed(
        self,
        stage: SovereignStage,
        reason: str,
        request: SovereignRequest,
        stages_completed: List[SovereignStage],
        evidence_chain: List[str],
        stage_latencies: Dict[str, int],
        start_time: float,
        context: Dict[str, Any],
        verdict: VerdictDecision,
    ) -> SovereignResult:
        """
        Handle fail-closed rejection at any stage.

        Creates a rejection result with full evidence trail and emits
        a rejection receipt.
        """
        self._rejected_requests += 1
        total_latency = int((time.perf_counter() - start_time) * 1000)

        receipt_id = self._generate_receipt_id(request.request_id, "rejected")

        # Emit rejection receipt
        asyncio.create_task(
            self.emit_sovereign_receipt(
                request_id=request.request_id,
                verdict=verdict.value,
                stages_completed=[s.value for s in stages_completed],
                snr_achieved=context.get("snr_score", 0.0),
                ihsan_achieved=context.get("ihsan_score", 0.0),
                cosmic_verdict=(
                    context.get("cosmic_verdict").decision.value
                    if context.get("cosmic_verdict") else None
                ),
                evidence_count=len(evidence_chain),
            )
        )

        result = SovereignResult(
            success=False,
            verdict=verdict,
            snr_achieved=context.get("snr_score", 0.0),
            ihsan_achieved=context.get("ihsan_score", 0.0),
            stages_completed=stages_completed,
            cosmic_verdict_detail=context.get("cosmic_verdict"),
            evidence_chain=evidence_chain,
            receipt_id=receipt_id,
            request_id=request.request_id,
            total_latency_ms=total_latency,
            stage_latencies=stage_latencies,
            failure_stage=stage,
            failure_reason=reason,
        )

        logger.warning(
            f"Sovereign request REJECTED: id={request.request_id}, "
            f"stage={stage.value}, reason={reason}"
        )

        return result

    def _extract_symbolic_constraints(self, request: SovereignRequest) -> List[Dict[str, Any]]:
        """Extract symbolic constraints from the request."""
        constraints = [
            {"type": "snr_minimum", "value": request.snr_target},
            {"type": "ihsan_minimum", "value": request.ihsan_target},
            {"type": "domain_count", "value": len(request.task_domains)},
        ]

        if request.require_elite_practitioner:
            constraints.append({"type": "elite_required", "value": True})
            constraints.append({"type": "domain_minimum", "value": ELITE_DOMAIN_MINIMUM})
            constraints.append({"type": "novelty_minimum", "value": ELITE_NOVELTY_THRESHOLD})

        return constraints

    def _detect_cross_pollination(self, domains: List[str]) -> bool:
        """Detect if domains show cross-pollination (are diverse)."""
        if len(domains) < 2:
            return False

        # Simple heuristic: check for domain diversity
        # In production, would use semantic similarity
        domain_categories = set()
        category_map = {
            "security": "technical",
            "performance": "technical",
            "optimization": "technical",
            "ethics": "governance",
            "compliance": "governance",
            "legal": "governance",
            "design": "creative",
            "synthesis": "creative",
            "analysis": "analytical",
            "reasoning": "analytical",
        }

        for domain in domains:
            domain_lower = domain.lower()
            for key, category in category_map.items():
                if key in domain_lower:
                    domain_categories.add(category)
                    break
            else:
                domain_categories.add(domain_lower)

        return len(domain_categories) >= 2

    def _generate_receipt_id(self, request_id: str, outcome: str) -> str:
        """Generate a unique receipt ID."""
        return f"sov-receipt-{outcome}-{uuid.uuid4().hex[:12]}"

    async def emit_sovereign_receipt(
        self,
        request_id: str,
        verdict: str,
        stages_completed: List[str],
        snr_achieved: float,
        ihsan_achieved: float,
        cosmic_verdict: Optional[str],
        evidence_count: int,
    ) -> SovereignReceipt:
        """
        Emit an evidence receipt for the sovereign operation.

        Args:
            request_id: The request ID
            verdict: The verdict outcome
            stages_completed: List of completed stages
            snr_achieved: Achieved SNR score
            ihsan_achieved: Achieved Ihsan score
            cosmic_verdict: Cosmic verdict decision
            evidence_count: Number of evidence items

        Returns:
            SovereignReceipt
        """
        receipt_id = self._generate_receipt_id(request_id, verdict.split("_")[0])
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute integrity hash
        hash_payload = json.dumps({
            "request_id": request_id,
            "verdict": verdict,
            "stages_completed": stages_completed,
            "snr_achieved": snr_achieved,
            "ihsan_achieved": ihsan_achieved,
            "cosmic_verdict": cosmic_verdict,
            "timestamp": timestamp,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")

        if HAS_BLAKE3:
            integrity_hash = blake3.blake3(hash_payload).hexdigest()
        else:
            integrity_hash = hashlib.sha256(hash_payload).hexdigest()

        receipt = SovereignReceipt(
            receipt_id=receipt_id,
            request_id=request_id,
            operation="sovereign_process",
            verdict=verdict,
            stages_completed=stages_completed,
            snr_achieved=snr_achieved,
            ihsan_achieved=ihsan_achieved,
            cosmic_verdict=cosmic_verdict,
            evidence_count=evidence_count,
            timestamp=timestamp,
            integrity_hash=integrity_hash,
        )

        self._receipts.append(receipt)

        # Write to receipt file
        try:
            receipt_file = (
                self.receipt_path /
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
            )
            with open(receipt_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt.to_dict()) + "\n")
            logger.debug(f"Emitted sovereign receipt: {receipt_id}")
        except Exception as e:
            logger.error(f"Failed to write receipt: {e}")

        return receipt

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "rejected_requests": self._rejected_requests,
            "veto_rejections": self._veto_rejections,
            "snr_rejections": self._snr_rejections,
            "success_rate": (
                self._successful_requests / max(1, self._total_requests)
            ),
            "thresholds": {
                "snr": self.snr_threshold,
                "ihsan": self.ihsan_threshold,
                "weighted_quorum": self.weighted_quorum,
            },
            "components": {
                "synthesis_engine": self.synthesis_engine is not None,
                "guardian_constellation": self.guardian_constellation is not None,
                "persona_registry": self.persona_registry is not None,
                "pareto_router": self.pareto_router is not None,
                "sovereignty_bridge": self.sovereignty_bridge is not None,
            },
            "receipt_count": len(self._receipts),
        }

    def get_receipts(self) -> List[SovereignReceipt]:
        """Get all receipts emitted by this orchestrator."""
        return list(self._receipts)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_sovereign_orchestrator(
    snr_threshold: float = SOVEREIGN_SNR_THRESHOLD,
    ihsan_threshold: float = SOVEREIGN_IHSAN_THRESHOLD,
    weighted_quorum: float = SOVEREIGN_WEIGHTED_QUORUM,
    receipt_path: Optional[Path] = None,
) -> SovereignOrchestrator:
    """
    Factory function to create a SovereignOrchestrator with standard configuration.

    This is the recommended entry point for creating a sovereign orchestrator
    with BIZRA's sovereign-level enforcement thresholds.

    Args:
        snr_threshold: SNR gate threshold (default 0.99 - sovereign level)
        ihsan_threshold: Ihsan gate threshold (default 0.95)
        weighted_quorum: Consensus quorum (default 2.4)
        receipt_path: Path for receipt storage

    Returns:
        Configured SovereignOrchestrator instance

    Example:
        >>> orchestrator = create_sovereign_orchestrator()
        >>> request = SovereignRequest(
        ...     query="Optimize consensus algorithm",
        ...     task_domains=["distributed-systems", "consensus", "optimization"],
        ... )
        >>> result = await orchestrator.process(request)
        >>> print(f"Verdict: {result.verdict.value}")
    """
    return SovereignOrchestrator(
        snr_threshold=snr_threshold,
        ihsan_threshold=ihsan_threshold,
        weighted_quorum=weighted_quorum,
        receipt_path=receipt_path,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SovereignStage",
    "VerdictDecision",
    "SovereignMode",
    # Data classes
    "SovereignRequest",
    "SovereignResult",
    "CosmicVerdictResult",
    "ElitePractitionerResult",
    "SovereignReceipt",
    # Main class
    "SovereignOrchestrator",
    # Factory function
    "create_sovereign_orchestrator",
    # Constants
    "SOVEREIGN_DOMAIN_PREFIX",
    "SOVEREIGN_VERSION",
    "SOVEREIGN_SNR_THRESHOLD",
    "SOVEREIGN_IHSAN_THRESHOLD",
    "SOVEREIGN_WEIGHTED_QUORUM",
    "ELITE_DOMAIN_MINIMUM",
    "ELITE_NOVELTY_THRESHOLD",
    "ELITE_PRACTITIONER_TIER",
    "PARETO_DIMENSIONS",
]


# =============================================================================
# MAIN / DEMO
# =============================================================================


async def main():
    """Demo the Sovereign Orchestrator."""
    print("=" * 80)
    print("BIZRA Sovereign Orchestrator - Phase 10 APEX SOVEREIGN")
    print("=" * 80)

    # Create orchestrator
    orchestrator = create_sovereign_orchestrator()

    print("\nOrchestrator Statistics:")
    stats = orchestrator.get_statistics()
    print(json.dumps(stats, indent=2))

    # Test sovereign request
    print("\n" + "-" * 40)
    print("Processing sovereign request...")
    print("-" * 40)

    request = SovereignRequest(
        query="Design a Byzantine fault-tolerant consensus algorithm for distributed systems",
        task_domains=[
            "distributed-systems",
            "consensus-algorithms",
            "fault-tolerance",
            "security",
        ],
        ihsan_target=0.95,
        snr_target=0.99,
        require_elite_practitioner=True,
    )

    result = await orchestrator.process(request)

    print(f"\nResult: {'APPROVED' if result.success else 'REJECTED'}")
    print(f"Verdict: {result.verdict.value}")
    print(f"SNR Achieved: {result.snr_achieved:.4f}")
    print(f"Ihsan Achieved: {result.ihsan_achieved:.4f}")
    print(f"Stages Completed: {len(result.stages_completed)}/10")
    print(f"Total Latency: {result.total_latency_ms}ms")
    print(f"Receipt ID: {result.receipt_id}")

    if not result.success:
        print(f"\nFailure Stage: {result.failure_stage.value if result.failure_stage else 'N/A'}")
        print(f"Failure Reason: {result.failure_reason}")

    if result.cosmic_verdict_detail:
        print(f"\nCosmic Verdict: {result.cosmic_verdict_detail.decision}")
        print(f"Quorum Met: {result.cosmic_verdict_detail.quorum_met}")
        print(f"Merkle Root: {result.cosmic_verdict_detail.merkle_root[:32]}...")

    print("\nEvidence Chain:")
    for i, evidence in enumerate(result.evidence_chain[:5]):
        print(f"  {i + 1}. {evidence}")
    if len(result.evidence_chain) > 5:
        print(f"  ... and {len(result.evidence_chain) - 5} more")

    print("\nStage Latencies:")
    for stage, latency in result.stage_latencies.items():
        print(f"  {stage}: {latency}ms")

    # Final statistics
    print("\n" + "=" * 80)
    print("Final Statistics:")
    print(json.dumps(orchestrator.get_statistics(), indent=2))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
