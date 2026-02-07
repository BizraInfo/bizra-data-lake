"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗███████╗██████╗  █████╗     ███████╗ ██████╗ ██╗   ██╗         ║
║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██║   ██║         ║
║   ██████╔╝██║  ███╔╝ ██████╔╝███████║    ███████╗██║   ██║██║   ██║         ║
║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║    ╚════██║██║   ██║╚██╗ ██╔╝         ║
║   ██████╔╝██║███████╗██║  ██║██║  ██║    ███████║╚██████╔╝ ╚████╔╝          ║
║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝   ╚═══╝           ║
║                                                                              ║
║                    SOVEREIGN AUTONOMOUS ENGINE v1.0                          ║
║            Graph-of-Thoughts • SNR Maximization • Ihsān Gate                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   ═══════════════════════════════════                                        ║
║   • DATA4LLM IaaS Framework (Tsinghua University, 2024)                      ║
║   • Graph-of-Thoughts (Besta et al., 2024)                                   ║
║   • NVIDIA PersonaPlex (Roy et al., 2026)                                    ║
║   • Claude-Flow V3 (ruv.io, 2026)                                            ║
║   • Model Context Protocol (Anthropic, 2025)                                 ║
║   • Mem0 Persistent Memory (mem0.ai, 2025)                                   ║
║   • LiteLLM Gateway (BerriAI, 2025)                                          ║
║   • DSPy Self-Optimizing Prompts (Stanford NLP, 2024)                        ║
║   • DDAGI Constitution v1.1.0 (Ihsān Constraint)                             ║
║                                                                              ║
║   The Sovereign Principle:                                                   ║
║   ═══════════════════════                                                    ║
║   "Every inference carries proof. Every decision passes the gate.            ║
║    Every node is sovereign. Every human is a seed."                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__codename__ = "Genesis"
__author__ = "BIZRA Node0 — Standing on Giants"

# Track what's available
_NUMPY_AVAILABLE = False
_FULL_MODE = False

# =============================================================================
# CORE MODULES (No external dependencies)
# =============================================================================

# Adl (Justice) Invariant - Anti-Plutocracy Enforcement (GAP-C2)
# Standing on Giants: Gini (1912), Harberger (1962), Rawls (1971)
from .adl_invariant import (
    ADL_GINI_THRESHOLD,
    HARBERGER_TAX_RATE,
)
from .adl_invariant import MINIMUM_HOLDING as ADL_MINIMUM_HOLDING
from .adl_invariant import (
    UBC_POOL_ID,
    AdlGate,
    AdlInvariant,
    AdlRejectCode,
    AdlValidationResult,
    RedistributionResult,
)
from .adl_invariant import Transaction as AdlTransaction
from .adl_invariant import (
    assert_adl_invariant,
    calculate_gini,
    calculate_gini_components,
    create_adl_extended_gatekeeper,
    simulate_transaction_impact,
)

# ADL Kernel - Full Antitrust Kernel with Causal Drag and Bias Parity (DDAGI)
# Standing on Giants: Rawls (1971), Gini (1912), Harberger (1962), Kullback & Leibler (1951)
from .adl_kernel import (
    ADL_GINI_ALERT_THRESHOLD,
    BIAS_EPSILON,
    OMEGA_DEFAULT,
    OMEGA_MAX,
    AdlEnforcer,
)
from .adl_kernel import AdlInvariant as AdlKernelConfig
from .adl_kernel import AdlRejectCode as AdlKernelRejectCode
from .adl_kernel import AdlValidationResult as AdlKernelResult
from .adl_kernel import (
    BiasParityResult,
    CausalDragResult,
    GiniResult,
    HarbergerTaxResult,
    apply_harberger_redistribution,
)
from .adl_kernel import calculate_gini as kernel_calculate_gini
from .adl_kernel import (
    calculate_gini_detailed,
    calculate_gini_from_holdings,
    check_bias_parity,
    compute_causal_drag,
    compute_ihsan_adl_score,
    create_uniform_distribution,
    harberger_tax,
    quick_adl_check,
)

# Apex Sovereign Engine (v3.0 Peak Masterpiece)
# Standing on Giants: Shannon, de Moura, Jaynes, Besta, Maturana, Karpathy, Al-Ghazali
from .apex_engine import (
    GIANTS_REGISTRY,
    ApexConfig,
    ApexResult,
    ApexSovereignEngine,
    BackendType,
    EvolutionResult,
    GiantsAttribution,
    LocalModelConfig,
    ProcessingStage,
    create_apex_engine,
)

# API module (pure Python)
from .api import (
    QueryRequest,
    QueryResponse,
    RateLimiter,
    SovereignAPIServer,
)

# Genesis Identity (Persistent Node0 Identity)
# Standing on Giants: Al-Ghazali (1095), Lamport (1982), Nakamoto (2008)
from .genesis_identity import (
    AgentIdentity,
    GenesisState,
    NodeIdentity,
    load_and_validate_genesis,
    load_genesis,
    validate_genesis_hash,
)

# Autonomy module (pure Python)
from .autonomy import (
    AutonomousLoop,
    DecisionCandidate,
    DecisionGate,
    DecisionOutcome,
    DecisionType,
    GateResult,
    LoopState,
    SystemMetrics,
    create_autonomous_loop,
)

# Autonomy Matrix (5-Level Control)
from .autonomy_matrix import (
    ActionContext,
    AutonomyConstraints,
    AutonomyDecision,
    AutonomyLevel,
    AutonomyMatrix,
)

# Background Agents (Domain-Specific Proactive Plugins)
from .background_agents import ActionType as BackgroundActionType
from .background_agents import AgentState as BackgroundAgentState
from .background_agents import (
    ApprovalStatus,
    BackgroundAgent,
    BackgroundAgentRegistry,
    CalendarOptimizer,
    EmailTriage,
    ExecutionStatus,
    FileOrganizer,
    ProactiveAction,
    ProactiveOpportunity,
    Reversibility,
    create_default_registry,
)

# Bridge module (pure Python)
from .bridge import (
    A2AConnector,
    FederationConnector,
    InferenceConnector,
    InferenceRequest,
    InferenceResponse,
    InferenceTier,
    MemoryConnector,
    SovereignBridge,
    SubsystemStatus,
    create_bridge,
)

# Capability Card (v2.2.0 Sovereign LLM)
from .capability_card import (
    CARD_VALIDITY_DAYS,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    CapabilityCard,
    CardIssuer,
    ModelCapabilities,
    ModelTier,
    TaskType,
    create_capability_card,
    verify_capability_card,
)

# Collective Intelligence
from .collective_intelligence import (
    AgentContribution,
    AggregationMethod,
    CollectiveDecision,
    CollectiveIntelligence,
)

# Collective Synthesizer (Trust-Weighted Decision Synthesis)
from .collective_synthesizer import (
    AgentOutput,
    CollectiveSynthesizer,
    ConflictStrategy,
    ResolvedOutput,
    SynthesizedResult,
)

# Dashboard (CLI Interface)
from .dashboard import (
    RICH_AVAILABLE,
    DashboardConfig,
    DashboardMode,
    ProactiveDashboard,
    create_dashboard,
)

# Dual-Agentic Bridge
from .dual_agentic_bridge import (
    ActionProposal,
    ConsensusOutcome,
    ConsensusResult,
    DualAgenticBridge,
    VetoReason,
    Vote,
)

# Enhanced Team Planner
from .enhanced_team_planner import (
    EnhancedTeamPlanner,
    ExecutionPlan,
    ExecutionResult,
    ProactiveGoal,
)

# Event Bus
from .event_bus import (
    Event,
    EventBus,
    EventPriority,
    get_event_bus,
)

# Iceoryx2 Zero-Copy IPC Bridge (v3.1-OMEGA L2 Synapse Protocol)
# Standing on Giants: iceoryx2 (Eclipse Foundation, 2024)
from .iceoryx2_bridge import (
    ICEORYX2_AVAILABLE,
    AsyncFallbackBridge,
    DeliveryResult,
    DeliveryStatus,
    Iceoryx2Bridge,
    IceoryxMessage,
    IPCBridge,
    LatencyStats,
    PayloadType,
    create_ipc_bridge,
)

# Canonical 8-Dimension Ihsan Vector (Constitutional Excellence Enforcement)
# Standing on Giants: Al-Ghazali (1111), Shannon (1948), de Moura (2008)
from .ihsan_vector import (
    ANTI_CENTRALIZATION_GINI_THRESHOLD,
    CANONICAL_WEIGHTS,
    CONTEXT_THRESHOLDS,
    VERIFY_METHODS,
    DimensionId,
    ExecutionContext,
)
from .ihsan_vector import IhsanDimension as CanonicalIhsanDimension
from .ihsan_vector import (
    IhsanReceipt,
)
from .ihsan_vector import IhsanVector as CanonicalIhsanVector
from .ihsan_vector import (
    ThresholdResult,
    create_verifier,
    passes_production,
    quick_ihsan,
)

# Integration Runtime (v2.2.0 Sovereign LLM)
from .integration import InferenceRequest as SovereignInferenceRequest
from .integration import InferenceResult as SovereignInferenceResult
from .integration import NetworkMode as SovereignNetworkMode
from .integration import SovereignConfig as SovereignLLMConfig
from .integration import SovereignRuntime as SovereignLLMRuntime
from .integration import (
    create_sovereign_runtime,
)
from .integration import print_banner as print_sovereign_banner

# Knowledge Integration (BIZRA Data Lake + MoMo R&D)
from .knowledge_integrator import (
    KnowledgeIntegrator,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
    create_knowledge_integrator,
)

# Launcher
from .launch import SovereignLauncher

# MCP Progressive Disclosure (v3.1-OMEGA Claude-Mem Architecture)
# Standing on Giants: Claude-Mem (Anthropic, 2025)
from .mcp_disclosure import (
    LoadedSkill,
    MCPProgressiveDisclosure,
    SkillContext,
    SkillIndex,
    create_mcp_disclosure,
)

# Metrics module (pure Python)
from .metrics import (
    MetricPoint,
    MetricsCollector,
    MetricSeries,
    SystemSnapshot,
    create_autonomy_analyzer,
    create_autonomy_observer,
    create_metrics_collector,
)

# Model License Gate (v2.2.0 Sovereign LLM)
from .model_license_gate import (
    GateChain,
    InMemoryRegistry,
    LicenseCheckResult,
    ModelLicenseGate,
    create_gate_chain,
)

# Muraqabah Engine (24/7 Monitoring)
from .muraqabah_engine import (
    MonitorDomain,
    MuraqabahEngine,
    Opportunity,
    SensorReading,
    SensorState,
)

# Muraqabah Sensor Hub (Multi-Domain SNR-Filtered Monitoring)
from .muraqabah_sensors import (
    SNR_FLOOR,
    SNR_HIGH,
    MuraqabahSensorHub,
    SensorDomain,
)
from .muraqabah_sensors import SensorReading as HubSensorReading
from .muraqabah_sensors import (
    SignificantChange,
)

# Opportunity Pipeline (Nervous System connecting Detection to Execution)
from .opportunity_pipeline import (
    ConstitutionalFilter,
    DaughterTestFilter,
    FilterResult,
    IhsanFilter,
    OpportunityPipeline,
    OpportunityStatus,
    PipelineOpportunity,
    PipelineStage,
    RateLimitFilter,
    SNRFilter,
    connect_background_agents_to_pipeline,
    connect_muraqabah_to_pipeline,
    create_opportunity_pipeline,
)

# Predictive Monitor
from .predictive_monitor import (
    AlertSeverity,
    MetricReading,
    PredictiveAlert,
    PredictiveMonitor,
    TrendAnalysis,
    TrendDirection,
)

# Proactive Integration (Unified System)
from .proactive_integration import (
    EntityConfig,
    EntityCycleResult,
    EntityMode,
    ProactiveSovereignEntity,
    create_proactive_entity,
)

# Proactive Scheduler
from .proactive_scheduler import (
    JobPriority,
    JobResult,
    ProactiveScheduler,
    ScheduledJob,
    ScheduleType,
)

# Proactive Team
from .proactive_team import (
    ProactiveCycleResult,
    ProactiveTeam,
)

# 9-Probe Defense Matrix (SAPE v1.infinity Cognitive Antibody System)
# Standing on Giants: Turing (1936), LeCun (2024), Pearl (2000), OWASP
from .probe_defense import (  # Enums; Data classes; Base class; Concrete probes; Matrix classes; Factory functions; Constants
    DEFAULT_FAIL_THRESHOLD,
    PII_PATTERNS,
    SYCOPHANCY_PATTERNS,
    AdversarialProbe,
    CandidateContext,
    CausalityProbe,
    CounterfactualProbe,
    EfficiencyProbe,
    HallucinationProbe,
    IntegratedProbeMatrix,
    InvariantProbe,
    LivenessProbe,
    PrivacyProbe,
    Probe,
    ProbeMatrix,
    ProbeReport,
    ProbeResult,
    ProbeType,
    SycophancyProbe,
    create_candidate_context,
    create_probe_matrix,
)

# Runtime module (pure Python, uses stubs when numpy unavailable)
from .runtime import (
    HealthStatus,
    RuntimeConfig,
    RuntimeMetrics,
    RuntimeMode,
    SovereignQuery,
    SovereignResult,
    SovereignRuntime,
)

# Rust Lifecycle Integration (Python <-> Rust Bridge)
from .rust_lifecycle import (
    RustAPIClient,
    RustLifecycleManager,
    RustProcessManager,
    RustServiceHealth,
    RustServiceStatus,
    create_rust_gate_filter,
    create_rust_lifecycle,
)

# State Checkpointer
from .state_checkpointer import (
    Checkpoint,
    StateCheckpointer,
)

# Swarm Knowledge Bridge (Agent-to-Knowledge Interface)
from .swarm_knowledge_bridge import (
    ROLE_KNOWLEDGE_ACCESS,
    AgentKnowledgeContext,
    KnowledgeInjection,
    SwarmKnowledgeBridge,
    create_swarm_knowledge_bridge,
)

# Tamper-Evident Audit Log (P0-3 Unified Audit)
# Standing on Giants: Merkle (1979), RFC 2104 (1997), Haber & Stornetta (1991)
from .tamper_evident_log import (
    GENESIS_HASH,
    HMAC_DOMAIN_PREFIX,
    AuditKeyManager,
    KeyRotationEvent,
    TamperEvidentEntry,
    TamperEvidentLog,
    TamperingReport,
    TamperType,
    VerificationStatus,
    create_audit_log,
    detect_tampering,
    verify_chain,
    verify_entry,
)

# Team Planner
from .team_planner import (
    AgentRole,
    Goal,
    TaskAllocation,
    TaskComplexity,
    TeamPlanner,
    TeamTask,
)

# Treasury Mode (GAP-C4: Wealth Engine Graceful Degradation)
from .treasury_mode import (
    ETHICS_THRESHOLD_HIBERNATION,
    ETHICS_THRESHOLD_RECOVERY,
    RESERVES_THRESHOLD_EMERGENCY,
    RESERVES_THRESHOLD_HIBERNATION,
    EthicsAssessment,
    TransitionEvent,
    TransitionTrigger,
    TreasuryController,
    TreasuryEvent,
    TreasuryMode,
    TreasuryPersistence,
    TreasuryState,
    create_treasury_controller,
)

# =============================================================================
# PROACTIVE SOVEREIGN ENTITY MODULES (v2.3.0)
# =============================================================================


# =============================================================================
# OPTIONAL MODULES (Require numpy)
# =============================================================================

try:
    import numpy as np

    _NUMPY_AVAILABLE = True

    from .engine import (
        SovereignConfig,
        SovereignEngine,
        SovereignResponse,
    )
    from .graph_reasoner import (
        GraphOfThoughts,
        ReasoningStrategy,
        ThoughtEdge,
        ThoughtNode,
    )
    from .guardian_council import (
        ConsensusMode,
        CouncilVerdict,
        GuardianCouncil,
    )
    from .ihsan_projector import (
        IHSAN_ARABIC_NAMES,
        IhsanDimension,
        IhsanProjector,
        IhsanVector,
        ProjectorConfig,
        create_ihsan_from_scores,
        project_ihsan_to_ntu,
    )
    from .orchestrator import (
        AgentRouter,
        SovereignOrchestrator,
        TaskDecomposer,
    )
    from .snr_maximizer import (
        NoiseFilter,
        SignalAmplifier,
        SNRMaximizer,
    )

    _FULL_MODE = True
    logger.debug("Sovereign Engine loaded in FULL mode (numpy available)")

except ImportError as e:
    # Define placeholder classes for modules that require numpy
    logger.warning(f"Sovereign Engine running in LITE mode (numpy unavailable): {e}")

    class _PlaceholderMeta(type):
        """Metaclass for placeholder classes that raise ImportError on instantiation."""

        def __call__(cls, *args, **kwargs):
            raise ImportError(
                f"{cls.__name__} requires numpy. Install with: pip install numpy"
            )

    class SovereignEngine(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class SovereignConfig(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class SovereignResponse(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class GraphOfThoughts(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class ThoughtNode(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class ThoughtEdge(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class ReasoningStrategy(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class SNRMaximizer(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class SignalAmplifier(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class NoiseFilter(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class GuardianCouncil(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class CouncilVerdict(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class ConsensusMode(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class SovereignOrchestrator(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class TaskDecomposer(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class AgentRouter(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class IhsanProjector(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class IhsanVector(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class IhsanDimension(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    class ProjectorConfig(metaclass=_PlaceholderMeta):
        """Placeholder - requires numpy."""

        pass

    def project_ihsan_to_ntu(*args, **kwargs):
        """Placeholder - requires numpy."""
        raise ImportError(
            "project_ihsan_to_ntu requires numpy. Install with: pip install numpy"
        )

    def create_ihsan_from_scores(*args, **kwargs):
        """Placeholder - requires numpy."""
        raise ImportError(
            "create_ihsan_from_scores requires numpy. Install with: pip install numpy"
        )

    IHSAN_ARABIC_NAMES = {}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Genesis Identity (Persistent Node0 Identity)
    "AgentIdentity",
    "NodeIdentity",
    "GenesisState",
    "load_genesis",
    "validate_genesis_hash",
    "load_and_validate_genesis",
    # Core Engine (requires numpy)
    "SovereignEngine",
    "SovereignConfig",
    "SovereignResponse",
    # Graph Reasoning (requires numpy)
    "GraphOfThoughts",
    "ThoughtNode",
    "ThoughtEdge",
    "ReasoningStrategy",
    # SNR Maximization (requires numpy)
    "SNRMaximizer",
    "SignalAmplifier",
    "NoiseFilter",
    # Guardian Council (requires numpy)
    "GuardianCouncil",
    "CouncilVerdict",
    "ConsensusMode",
    # Orchestration (requires numpy)
    "SovereignOrchestrator",
    "TaskDecomposer",
    "AgentRouter",
    # Autonomous Loop (pure Python)
    "AutonomousLoop",
    "DecisionGate",
    "DecisionCandidate",
    "DecisionOutcome",
    "SystemMetrics",
    "LoopState",
    "DecisionType",
    "GateResult",
    "create_autonomous_loop",
    # Runtime (pure Python with stubs)
    "SovereignRuntime",
    "RuntimeConfig",
    "RuntimeMode",
    "RuntimeMetrics",
    "SovereignQuery",
    "SovereignResult",
    "HealthStatus",
    # API (pure Python)
    "SovereignAPIServer",
    "QueryRequest",
    "QueryResponse",
    "RateLimiter",
    # Bridge
    "SovereignBridge",
    "create_bridge",
    "InferenceConnector",
    "FederationConnector",
    "MemoryConnector",
    "A2AConnector",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceTier",
    "SubsystemStatus",
    # Metrics
    "MetricsCollector",
    "MetricPoint",
    "MetricSeries",
    "SystemSnapshot",
    "create_metrics_collector",
    "create_autonomy_observer",
    "create_autonomy_analyzer",
    # Launcher
    "SovereignLauncher",
    # Capability Card (v2.2.0 Sovereign LLM)
    "CapabilityCard",
    "ModelCapabilities",
    "ModelTier",
    "TaskType",
    "CardIssuer",
    "create_capability_card",
    "verify_capability_card",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "CARD_VALIDITY_DAYS",
    # Model License Gate (v2.2.0 Sovereign LLM)
    "ModelLicenseGate",
    "LicenseCheckResult",
    "InMemoryRegistry",
    "GateChain",
    "create_gate_chain",
    # Integration Runtime (v2.2.0 Sovereign LLM)
    "SovereignLLMRuntime",
    "SovereignLLMConfig",
    "SovereignInferenceRequest",
    "SovereignInferenceResult",
    "SovereignNetworkMode",
    "create_sovereign_runtime",
    "print_sovereign_banner",
    # MCP Progressive Disclosure (v3.1-OMEGA Claude-Mem Architecture)
    "SkillIndex",
    "SkillContext",
    "LoadedSkill",
    "MCPProgressiveDisclosure",
    "create_mcp_disclosure",
    # Treasury Mode (GAP-C4: Wealth Engine Graceful Degradation)
    "TreasuryMode",
    "TreasuryState",
    "TreasuryController",
    "TreasuryEvent",
    "TransitionTrigger",
    "TransitionEvent",
    "EthicsAssessment",
    "TreasuryPersistence",
    "create_treasury_controller",
    "ETHICS_THRESHOLD_HIBERNATION",
    "ETHICS_THRESHOLD_RECOVERY",
    "RESERVES_THRESHOLD_EMERGENCY",
    "RESERVES_THRESHOLD_HIBERNATION",
    # Adl (Justice) Invariant - Anti-Plutocracy Enforcement (GAP-C2)
    "AdlInvariant",
    "AdlGate",
    "AdlRejectCode",
    "AdlValidationResult",
    "RedistributionResult",
    "AdlTransaction",
    "calculate_gini",
    "calculate_gini_components",
    "assert_adl_invariant",
    "simulate_transaction_impact",
    "create_adl_extended_gatekeeper",
    "ADL_GINI_THRESHOLD",
    "HARBERGER_TAX_RATE",
    "ADL_MINIMUM_HOLDING",
    "UBC_POOL_ID",
    # ADL Kernel - Full Antitrust Kernel (DDAGI)
    "AdlKernelConfig",
    "GiniResult",
    "CausalDragResult",
    "HarbergerTaxResult",
    "BiasParityResult",
    "AdlKernelResult",
    "AdlKernelRejectCode",
    "AdlEnforcer",
    "kernel_calculate_gini",
    "calculate_gini_from_holdings",
    "calculate_gini_detailed",
    "compute_causal_drag",
    "harberger_tax",
    "apply_harberger_redistribution",
    "check_bias_parity",
    "create_uniform_distribution",
    "quick_adl_check",
    "compute_ihsan_adl_score",
    "ADL_GINI_ALERT_THRESHOLD",
    "OMEGA_DEFAULT",
    "OMEGA_MAX",
    "BIAS_EPSILON",
    # Ihsan Projector - Constitutional AI to NTU Bridge (GAP-C1)
    "IhsanProjector",
    "IhsanVector",
    "IhsanDimension",
    "ProjectorConfig",
    "project_ihsan_to_ntu",
    "create_ihsan_from_scores",
    "IHSAN_ARABIC_NAMES",
    # ==========================================================================
    # PROACTIVE SOVEREIGN ENTITY (v2.3.0)
    # ==========================================================================
    # Event Bus
    "Event",
    "EventBus",
    "EventPriority",
    "get_event_bus",
    # State Checkpointer
    "Checkpoint",
    "StateCheckpointer",
    # Team Planner
    "AgentRole",
    "Goal",
    "TaskAllocation",
    "TaskComplexity",
    "TeamPlanner",
    "TeamTask",
    # Dual-Agentic Bridge
    "ActionProposal",
    "ConsensusOutcome",
    "ConsensusResult",
    "DualAgenticBridge",
    "VetoReason",
    "Vote",
    # Collective Intelligence
    "AgentContribution",
    "AggregationMethod",
    "CollectiveDecision",
    "CollectiveIntelligence",
    # Collective Synthesizer
    "AgentOutput",
    "CollectiveSynthesizer",
    "ConflictStrategy",
    "ResolvedOutput",
    "SynthesizedResult",
    # Proactive Scheduler
    "JobPriority",
    "JobResult",
    "ProactiveScheduler",
    "ScheduledJob",
    "ScheduleType",
    # Predictive Monitor
    "AlertSeverity",
    "MetricReading",
    "PredictiveAlert",
    "PredictiveMonitor",
    "TrendAnalysis",
    "TrendDirection",
    # Proactive Team
    "ProactiveCycleResult",
    "ProactiveTeam",
    # Muraqabah Engine (24/7 Monitoring)
    "MonitorDomain",
    "MuraqabahEngine",
    "Opportunity",
    "SensorReading",
    "SensorState",
    # Muraqabah Sensor Hub (Multi-Domain SNR-Filtered Monitoring)
    "MuraqabahSensorHub",
    "SensorDomain",
    "HubSensorReading",
    "SignificantChange",
    "SNR_FLOOR",
    "SNR_HIGH",
    # Autonomy Matrix (5-Level Control)
    "ActionContext",
    "AutonomyConstraints",
    "AutonomyDecision",
    "AutonomyLevel",
    "AutonomyMatrix",
    # Enhanced Team Planner
    "EnhancedTeamPlanner",
    "ExecutionPlan",
    "ExecutionResult",
    "ProactiveGoal",
    # Proactive Integration (Unified System)
    "EntityConfig",
    "EntityCycleResult",
    "EntityMode",
    "ProactiveSovereignEntity",
    "create_proactive_entity",
    # Knowledge Integration (BIZRA Data Lake + MoMo R&D)
    "KnowledgeIntegrator",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeSource",
    "create_knowledge_integrator",
    # Swarm Knowledge Bridge (Agent-to-Knowledge Interface)
    "AgentKnowledgeContext",
    "KnowledgeInjection",
    "ROLE_KNOWLEDGE_ACCESS",
    "SwarmKnowledgeBridge",
    "create_swarm_knowledge_bridge",
    # Dashboard (CLI Interface)
    "DashboardConfig",
    "DashboardMode",
    "ProactiveDashboard",
    "create_dashboard",
    "RICH_AVAILABLE",
    # Background Agents (Domain-Specific Proactive Plugins)
    "BackgroundAgentState",
    "BackgroundActionType",
    "ApprovalStatus",
    "ExecutionStatus",
    "Reversibility",
    "ProactiveOpportunity",
    "ProactiveAction",
    "BackgroundAgent",
    "CalendarOptimizer",
    "EmailTriage",
    "FileOrganizer",
    "BackgroundAgentRegistry",
    "create_default_registry",
    # Opportunity Pipeline (Nervous System connecting Detection to Execution)
    "ConstitutionalFilter",
    "DaughterTestFilter",
    "FilterResult",
    "IhsanFilter",
    "OpportunityPipeline",
    "OpportunityStatus",
    "PipelineOpportunity",
    "PipelineStage",
    "RateLimitFilter",
    "SNRFilter",
    "connect_background_agents_to_pipeline",
    "connect_muraqabah_to_pipeline",
    "create_opportunity_pipeline",
    # Rust Lifecycle Integration (Python <-> Rust Bridge)
    "RustAPIClient",
    "RustLifecycleManager",
    "RustProcessManager",
    "RustServiceHealth",
    "RustServiceStatus",
    "create_rust_gate_filter",
    "create_rust_lifecycle",
    # Iceoryx2 Zero-Copy IPC Bridge (v3.1-OMEGA L2 Synapse Protocol)
    "ICEORYX2_AVAILABLE",
    "AsyncFallbackBridge",
    "DeliveryResult",
    "DeliveryStatus",
    "IPCBridge",
    "Iceoryx2Bridge",
    "IceoryxMessage",
    "LatencyStats",
    "PayloadType",
    "create_ipc_bridge",
    # Canonical 8-Dimension Ihsan Vector (Constitutional Excellence Enforcement)
    "ANTI_CENTRALIZATION_GINI_THRESHOLD",
    "CANONICAL_WEIGHTS",
    "CONTEXT_THRESHOLDS",
    "VERIFY_METHODS",
    "DimensionId",
    "ExecutionContext",
    "CanonicalIhsanDimension",
    "IhsanReceipt",
    "CanonicalIhsanVector",
    "ThresholdResult",
    "create_verifier",
    "passes_production",
    "quick_ihsan",
    # ==========================================================================
    # APEX SOVEREIGN ENGINE (v3.0 Peak Masterpiece)
    # Standing on Giants: Shannon, de Moura, Jaynes, Besta, Maturana, Karpathy, Al-Ghazali
    # ==========================================================================
    "ApexSovereignEngine",
    "ApexConfig",
    "ApexResult",
    "LocalModelConfig",
    "EvolutionResult",
    "GiantsAttribution",
    "GIANTS_REGISTRY",
    "ProcessingStage",
    "BackendType",
    "create_apex_engine",
    # ==========================================================================
    # 9-PROBE DEFENSE MATRIX (SAPE v1.infinity Cognitive Antibody System)
    # Standing on Giants: Turing (1936), LeCun (2024), Pearl (2000), OWASP
    # ==========================================================================
    "ProbeType",
    "ProbeResult",
    "ProbeReport",
    "CandidateContext",
    "Probe",
    "CounterfactualProbe",
    "AdversarialProbe",
    "InvariantProbe",
    "EfficiencyProbe",
    "PrivacyProbe",
    "SycophancyProbe",
    "CausalityProbe",
    "HallucinationProbe",
    "LivenessProbe",
    "ProbeMatrix",
    "IntegratedProbeMatrix",
    "create_probe_matrix",
    "create_candidate_context",
    "DEFAULT_FAIL_THRESHOLD",
    "PII_PATTERNS",
    "SYCOPHANCY_PATTERNS",
    # ==========================================================================
    # TAMPER-EVIDENT AUDIT LOG (P0-3 Unified Audit)
    # Standing on Giants: Merkle (1979), RFC 2104 (1997), Haber & Stornetta (1991)
    # ==========================================================================
    "TamperEvidentEntry",
    "TamperEvidentLog",
    "AuditKeyManager",
    "TamperingReport",
    "KeyRotationEvent",
    "VerificationStatus",
    "TamperType",
    "create_audit_log",
    "verify_entry",
    "verify_chain",
    "detect_tampering",
    "GENESIS_HASH",
    "HMAC_DOMAIN_PREFIX",
]


def is_full_mode() -> bool:
    """Check if running in full mode (all dependencies available)."""
    return _FULL_MODE


def get_mode() -> str:
    """Get current operating mode."""
    return "FULL" if _FULL_MODE else "LITE"
