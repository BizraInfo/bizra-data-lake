"""
BIZRA Apex Sovereign - Elite Validation Layer
==============================================
Sovereign-tier validation components for PAT enforcement.

Components:
    - ElitePractitionerProtocol: "Standing on Giants" validation
    - PractitionerRegistry: Elite practitioner database
    - UnrelatednessMeasure: Cross-domain semantic distance
    - CosmicVerdictEngine: Swarm intelligence validation with Byzantine fault tolerance

Constitution Reference: constitution/pat_enforcement_v1.yaml

Key Thresholds:
    - DOMAIN_MIN: 3 (minimum unrelated domains)
    - PRACTITIONERS_PER_DOMAIN: 3 (minimum elite per domain)
    - UNRELATEDNESS_THRESHOLD: 0.70 (pairwise semantic distance)
    - NOVELTY_THRESHOLD: 0.75 (semantic distance from known patterns)

Cosmic Verdict Engine:
    Byzantine Fault Tolerance: n=8, f=2, quorum=5
    4-Phase Bee Colony Model: SCOUT -> WAGGLE -> QUORUM -> LIFTOFF
    ABSOLUTE vetoes: Ar-Ruh (ethics), Al-Amin (security), Majlis (collective)
"""

from core.apex.sovereign.elite_practitioner import (
    # Enums
    PractitionerTier,
    # Data Classes
    Practitioner,
    DomainValidation,
    ElitePractitionerResult,
    # Core Classes
    ElitePractitionerProtocol,
    PractitionerRegistry,
    UnrelatednessMeasure,
    # Factory Functions
    create_elite_practitioner_protocol,
    # Constants
    DOMAIN_MIN,
    PRACTITIONERS_PER_DOMAIN,
    UNRELATEDNESS_THRESHOLD,
    NOVELTY_THRESHOLD,
    DOMAIN_PREFIX,
)

from core.apex.sovereign.cosmic_verdict import (
    # Version/Domain
    COSMIC_VERDICT_VERSION,
    COSMIC_VERDICT_DOMAIN,
    # BFT Constants
    BFT_N,
    BFT_F,
    BFT_QUORUM,
    CRITICAL_QUORUM,
    # Threshold Constants
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    # Enums
    VerdictDecision,
    VotingPhase,
    # Data Classes
    GuardianVote,
    CosmicVerdictResult,
    VerdictRequest,
    ScoutResult,
    WaggleResult,
    QuorumResult,
    # Main Class
    CosmicVerdictEngine,
    # Factory Functions
    create_cosmic_verdict_engine,
    # Global Instance Functions
    get_cosmic_verdict_engine,
    reset_cosmic_verdict_engine,
    # Constants
    ABSOLUTE_VETO_ROLES,
)

# Sovereign Receipt System - Phase 10 APEX SOVEREIGN Audit Trail
from core.apex.sovereign.sovereign_receipt import (
    # Enums
    SovereignReceiptType,
    VerdictDecision as ReceiptVerdictDecision,
    ReceiptStatus,
    # Data Classes
    SovereignReceipt,
    # Core Classes
    EvidenceChainManager,
    SovereignReceiptEmitter,
    AuditTrailManager,
    # Constants
    SOVEREIGN_DOMAIN_PREFIX,
    SOVEREIGN_VERSION,
    DEFAULT_SNR_TARGET,
    DEFAULT_STORAGE_PATH,
)

# Sovereign Orchestrator - Phase 10 Master Controller
from core.apex.sovereign.sovereign_orchestrator import (
    # Enums
    SovereignStage,
    SovereignMode,
    # Data Classes
    SovereignRequest,
    SovereignResult,
    # Main Class
    SovereignOrchestrator,
    # Factory Functions
    create_sovereign_orchestrator,
)

# Neural-Symbolic Fusion - LLM + Formal Verification
from core.apex.sovereign.neural_symbolic_fusion import (
    # Enums
    FusionMode,
    SymbolicBackend,
    # Data Classes
    FusionContext,
    NeuralResult,
    SymbolicResult,
    FusionResult,
    # Main Class
    NeuralSymbolicFusionEngine,
    # Factory Functions
    create_neural_symbolic_fusion_engine,
    # Constants
    DEFAULT_IHSAN_THRESHOLD,
)

# Autonomous SNR Optimizer - Self-Optimizing for 0.99+
from core.apex.sovereign.autonomous_optimizer import (
    # Enums
    OptimizationStrategy,
    # Data Classes
    OptimizationState,
    OptimizedResult,
    # Main Classes
    AutonomousSNROptimizer,
    ThompsonSampler,
    PatternCache,
    # Factory Functions
    create_autonomous_optimizer,
)

__all__ = [
    # Elite Practitioner - Enums
    "PractitionerTier",
    # Elite Practitioner - Data Classes
    "Practitioner",
    "DomainValidation",
    "ElitePractitionerResult",
    # Elite Practitioner - Core Classes
    "ElitePractitionerProtocol",
    "PractitionerRegistry",
    "UnrelatednessMeasure",
    # Elite Practitioner - Factory Functions
    "create_elite_practitioner_protocol",
    # Elite Practitioner - Constants
    "DOMAIN_MIN",
    "PRACTITIONERS_PER_DOMAIN",
    "UNRELATEDNESS_THRESHOLD",
    "NOVELTY_THRESHOLD",
    "DOMAIN_PREFIX",
    # Cosmic Verdict - Version/Domain
    "COSMIC_VERDICT_VERSION",
    "COSMIC_VERDICT_DOMAIN",
    # Cosmic Verdict - BFT Constants
    "BFT_N",
    "BFT_F",
    "BFT_QUORUM",
    "CRITICAL_QUORUM",
    # Cosmic Verdict - Threshold Constants
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    # Cosmic Verdict - Enums
    "VerdictDecision",
    "VotingPhase",
    # Cosmic Verdict - Data Classes
    "GuardianVote",
    "CosmicVerdictResult",
    "VerdictRequest",
    "ScoutResult",
    "WaggleResult",
    "QuorumResult",
    # Cosmic Verdict - Main Class
    "CosmicVerdictEngine",
    # Cosmic Verdict - Factory Functions
    "create_cosmic_verdict_engine",
    # Cosmic Verdict - Global Instance Functions
    "get_cosmic_verdict_engine",
    "reset_cosmic_verdict_engine",
    # Cosmic Verdict - Constants
    "ABSOLUTE_VETO_ROLES",
    # Sovereign Receipt - Enums
    "SovereignReceiptType",
    "ReceiptVerdictDecision",
    "ReceiptStatus",
    # Sovereign Receipt - Data Classes
    "SovereignReceipt",
    # Sovereign Receipt - Core Classes
    "EvidenceChainManager",
    "SovereignReceiptEmitter",
    "AuditTrailManager",
    # Sovereign Receipt - Constants
    "SOVEREIGN_DOMAIN_PREFIX",
    "SOVEREIGN_VERSION",
    "DEFAULT_SNR_TARGET",
    "DEFAULT_STORAGE_PATH",
    # Sovereign Orchestrator - Enums
    "SovereignStage",
    "SovereignMode",
    # Sovereign Orchestrator - Data Classes
    "SovereignRequest",
    "SovereignResult",
    # Sovereign Orchestrator - Main Class
    "SovereignOrchestrator",
    # Sovereign Orchestrator - Factory Functions
    "create_sovereign_orchestrator",
    # Neural-Symbolic Fusion - Enums
    "FusionMode",
    "SymbolicBackend",
    # Neural-Symbolic Fusion - Data Classes
    "FusionContext",
    "NeuralResult",
    "SymbolicResult",
    "FusionResult",
    # Neural-Symbolic Fusion - Main Class
    "NeuralSymbolicFusionEngine",
    # Neural-Symbolic Fusion - Factory Functions
    "create_neural_symbolic_fusion_engine",
    # Neural-Symbolic Fusion - Constants
    "DEFAULT_IHSAN_THRESHOLD",
    # Autonomous Optimizer - Enums
    "OptimizationStrategy",
    # Autonomous Optimizer - Data Classes
    "OptimizationState",
    "OptimizedResult",
    # Autonomous Optimizer - Main Classes
    "AutonomousSNROptimizer",
    "ThompsonSampler",
    "PatternCache",
    # Autonomous Optimizer - Factory Functions
    "create_autonomous_optimizer",
]
