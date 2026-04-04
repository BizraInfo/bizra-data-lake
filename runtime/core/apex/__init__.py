"""
BIZRA Apex Orchestrator - Python Kernel
========================================
Self-optimizing routing layer with Thompson Sampling, SONA learning,
pattern extraction, and cost optimization.

Components:
    - ThompsonRouter: Thompson Sampling-based agent selection
    - SONALearner: Self-Optimizing Novelty Architecture for continuous improvement
    - PatternExtractor: Success/failure pattern mining with deduplication
    - CostAnalyzer: Cost-aware model selection (60-70% savings target)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      APEX ORCHESTRATOR                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Task Request ──▶ [CostAnalyzer] ──▶ [ThompsonRouter]      │
    │                          │                  │                │
    │                          │                  ▼                │
    │                    Cost Metrics     Agent Selection          │
    │                          │                  │                │
    │                          └────────┬─────────┘                │
    │                                   │                          │
    │                                   ▼                          │
    │                          [Execution + Feedback]              │
    │                                   │                          │
    │                       ┌───────────┴───────────┐              │
    │                       │                       │              │
    │                       ▼                       ▼              │
    │              [PatternExtractor]       [SONALearner]          │
    │                       │                       │              │
    │                       ▼                       ▼              │
    │               Pattern Cache           Weight Updates         │
    │                       │                       │              │
    │                       └───────────┬───────────┘              │
    │                                   │                          │
    │                            [SAPE Elevation]                  │
    │                          (repetitions > 3)                   │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

Integration:
    - SAPE: Patterns elevated when repetitions > 3
    - Ihsan: Performance evaluation gated by 0.95 threshold
    - FATE: Failure patterns may trigger escalation

Target Improvements:
    - +55% routing efficiency (SONA optimization)
    - 60-70% cost reduction (model selection)
    - 70% latency reduction (pattern caching)
"""

from core.apex.thompson_router import (
    ThompsonSamplingRouter,
    CapabilityMatrix,
    AgentCapability,
    TaskCategory,
)
from core.apex.sona_learner import (
    SONALearner,
    LearningConfig,
    PerformanceMetrics,
)
from core.apex.pattern_extractor import (
    PatternExtractor,
    ExecutionPattern,
    PatternType,
)
from core.apex.cost_analyzer import (
    CostAnalyzer,
    CostMetrics,
    ModelCostConfig,
)
from core.apex.sovereignty_bridge import (
    SovereigntyBridge,
    SovereigntyVerification,
    DeterminismReport,
    EvidenceNode,
    create_sovereignty_bridge,
    DOMAIN_PREFIX,
    DEFAULT_IHSAN_THRESHOLD,
    DEFAULT_EMBEDDING_DIM,
)
from core.apex.validation_pipeline import (
    ValidationPipeline,
    ValidationResult,
    ValidationContext,
    get_validation_pipeline,
    validate_envelope,
)
from core.apex.request_handler import (
    RequestHandler,
    OrchestrationMode,
    RequestSource,
    ValidationError,
    create_request_handler,
    from_http_request,
    from_cli_request,
    from_a2a_request,
)

# Unified Orchestrator (main entry point)
from core.apex.unified_orchestrator import (
    UnifiedOrchestrator,
    OrchestrationRequest,
    OrchestrationResult,
    StageResult,
    ProcessingStage,
    ProcessingMode,
    DOMAIN_PREFIX as APEX_DOMAIN_PREFIX,
    IHSAN_THRESHOLD as APEX_IHSAN_THRESHOLD,
    SNR_THRESHOLD as APEX_SNR_THRESHOLD,
)

# Consensus Manager (SAT + SAPE coordination)
from core.apex.consensus_manager import (
    ConsensusManager,
    ConsensusResult,
    SATVote,
    SAPEResult,
    ProbeContext,
    ProbeResult,
    DualAgenticRequest,
    ValidatorType,
    VetoReason,
    SAPEProbeType,
    get_consensus_manager,
    reset_consensus_manager,
    obtain_sat_consensus,
    run_sape_probes,
    full_validation,
    SAT_QUORUM_REQUIRED,
    SAPE_BATCHES,
    PROBE_WEIGHTS,
)

# Pareto-Optimal Multi-Objective Router
from core.apex.pareto_router import (
    ParetoOptimalRouter,
    ObjectiveVector,
    ObjectiveName,
    ParetoPoint,
    RoutingPreference,
    ParetoSelectionResult,
    ObjectiveEvaluator,
    DefaultObjectiveEvaluator,
    create_bizra_pareto_router,
    DOMAIN_PREFIX as PARETO_DOMAIN_PREFIX,
    DEFAULT_SNR_THRESHOLD,
    DEFAULT_NOVELTY_THRESHOLD,
)

# Rewarded Soups for Persona Interpolation
from core.apex.rewarded_soup import (
    VOICE_EMBEDDING_DIM as SOUP_EMBEDDING_DIM,
    DEFAULT_SNR_BASE,
    DOMAIN_PREFIX as SOUP_DOMAIN_PREFIX,
    SoupPreset,
    PersonaSoupComponent,
    PersonaSoup,
    SNRContribution,
    l2_normalize,
    interpolate_embeddings,
    compose_weighted_prompt,
    compute_snr_contribution,
    interpolate_soup,
    create_security_focused_soup,
    create_creative_focused_soup,
    create_analysis_focused_soup,
    create_balanced_soup,
    create_guardian_council_soup,
    create_standard_soups,
    get_soup_for_task,
    validate_soup_integrity,
)

# Graph of Thoughts Engine (multi-dimensional reasoning)
from core.apex.graph_of_thoughts import (
    GraphOfThoughtsEngine,
    GoTNode,
    GoTEdge,
    TaskDomain,
    ParetoSolution,
    SynthesisResult,
    GoTGraphResult,
    GoTNodeType,
    GoTEdgeType,
    GoTTraversalStatus,
    create_got_engine,
    DEFAULT_SNR_THRESHOLD as GOT_SNR_THRESHOLD,
    DEFAULT_MAX_DEPTH as GOT_MAX_DEPTH,
    DEFAULT_DIVERSITY_BONUS as GOT_DIVERSITY_BONUS,
    DOMAIN_PREFIX as GOT_DOMAIN_PREFIX,
)

# Synthesis Engine (apex unified orchestrator combining all components)
from core.apex.synthesis_engine import (
    SynthesisEngine,
    SynthesisGate,
    SynthesisResult as SynthesisEngineSynthesisResult,
    SynthesisNode,
    LambdaConfig,
    ParetoFront,
    PersonaSoupBlend,
    ReasoningNode,
    GraphOfThoughts,
    SynthesisStage,
    GateStatus,
    ReasoningNodeType,
    ParetoOptimalRouter as ParetoOptimalRouterInternal,
    PersonaSoupBlender,
    GraphOfThoughtsEngine as GraphOfThoughtsEngineInternal,
    create_synthesis_engine,
    SNR_THRESHOLD as SYNTHESIS_SNR_THRESHOLD,
    IHSAN_THRESHOLD as SYNTHESIS_IHSAN_THRESHOLD,
    WEIGHTED_QUORUM as SYNTHESIS_WEIGHTED_QUORUM,
)

__all__ = [
    # Thompson Router
    "ThompsonSamplingRouter",
    "CapabilityMatrix",
    "AgentCapability",
    "TaskCategory",
    # SONA Learner
    "SONALearner",
    "LearningConfig",
    "PerformanceMetrics",
    # Pattern Extractor
    "PatternExtractor",
    "ExecutionPattern",
    "PatternType",
    # Cost Analyzer
    "CostAnalyzer",
    "CostMetrics",
    "ModelCostConfig",
    # Sovereignty Bridge
    "SovereigntyBridge",
    "SovereigntyVerification",
    "DeterminismReport",
    "EvidenceNode",
    "create_sovereignty_bridge",
    "DOMAIN_PREFIX",
    "DEFAULT_IHSAN_THRESHOLD",
    "DEFAULT_EMBEDDING_DIM",
    # Validation Pipeline
    "ValidationPipeline",
    "ValidationResult",
    "ValidationContext",
    "get_validation_pipeline",
    "validate_envelope",
    # Request Handler
    "RequestHandler",
    "OrchestrationMode",
    "RequestSource",
    "ValidationError",
    "create_request_handler",
    "from_http_request",
    "from_cli_request",
    "from_a2a_request",
    # Unified Orchestrator (main entry point)
    "UnifiedOrchestrator",
    "OrchestrationRequest",
    "OrchestrationResult",
    "StageResult",
    "ProcessingStage",
    "ProcessingMode",
    "APEX_DOMAIN_PREFIX",
    "APEX_IHSAN_THRESHOLD",
    "APEX_SNR_THRESHOLD",
    # Consensus Manager (SAT + SAPE coordination)
    "ConsensusManager",
    "ConsensusResult",
    "SATVote",
    "SAPEResult",
    "ProbeContext",
    "ProbeResult",
    "DualAgenticRequest",
    "ValidatorType",
    "VetoReason",
    "SAPEProbeType",
    "get_consensus_manager",
    "reset_consensus_manager",
    "obtain_sat_consensus",
    "run_sape_probes",
    "full_validation",
    "SAT_QUORUM_REQUIRED",
    "SAPE_BATCHES",
    "PROBE_WEIGHTS",
    # Pareto-Optimal Router
    "ParetoOptimalRouter",
    "ObjectiveVector",
    "ObjectiveName",
    "ParetoPoint",
    "RoutingPreference",
    "ParetoSelectionResult",
    "ObjectiveEvaluator",
    "DefaultObjectiveEvaluator",
    "create_bizra_pareto_router",
    "PARETO_DOMAIN_PREFIX",
    "DEFAULT_SNR_THRESHOLD",
    "DEFAULT_NOVELTY_THRESHOLD",
    # Rewarded Soups (Persona Interpolation)
    "SOUP_EMBEDDING_DIM",
    "DEFAULT_SNR_BASE",
    "SOUP_DOMAIN_PREFIX",
    "SoupPreset",
    "PersonaSoupComponent",
    "PersonaSoup",
    "SNRContribution",
    "l2_normalize",
    "interpolate_embeddings",
    "compose_weighted_prompt",
    "compute_snr_contribution",
    "interpolate_soup",
    "create_security_focused_soup",
    "create_creative_focused_soup",
    "create_analysis_focused_soup",
    "create_balanced_soup",
    "create_guardian_council_soup",
    "create_standard_soups",
    "get_soup_for_task",
    "validate_soup_integrity",
    # Graph of Thoughts Engine (multi-dimensional reasoning)
    "GraphOfThoughtsEngine",
    "GoTNode",
    "GoTEdge",
    "TaskDomain",
    "ParetoSolution",
    "SynthesisResult",
    "GoTGraphResult",
    "GoTNodeType",
    "GoTEdgeType",
    "GoTTraversalStatus",
    "create_got_engine",
    "GOT_SNR_THRESHOLD",
    "GOT_MAX_DEPTH",
    "GOT_DIVERSITY_BONUS",
    "GOT_DOMAIN_PREFIX",
    # Synthesis Engine (apex orchestrator combining all components)
    "SynthesisEngine",
    "SynthesisGate",
    "SynthesisEngineSynthesisResult",
    "SynthesisNode",
    "LambdaConfig",
    "ParetoFront",
    "PersonaSoupBlend",
    "ReasoningNode",
    "GraphOfThoughts",
    "SynthesisStage",
    "GateStatus",
    "ReasoningNodeType",
    "ParetoOptimalRouterInternal",
    "PersonaSoupBlender",
    "GraphOfThoughtsEngineInternal",
    "create_synthesis_engine",
    "SYNTHESIS_SNR_THRESHOLD",
    "SYNTHESIS_IHSAN_THRESHOLD",
    "SYNTHESIS_WEIGHTED_QUORUM",
]
