"""
SOVEREIGN AUTONOMOUS ENGINE v1.0
Graph-of-Thoughts | SNR Maximization | Ihsan Gate

Standing on Giants:
  Shannon (1948) · Besta (2024) · Al-Ghazali (1095) · Anthropic (2023)
  Lamport (1978) · Maturana (1973) · de Moura (2008)

"Every inference carries proof. Every decision passes the gate.
 Every node is sovereign. Every human is a seed."
"""

import logging

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__codename__ = "Genesis"
__author__ = "BIZRA Node0 — Standing on Giants"

_NUMPY_AVAILABLE = False
_FULL_MODE = False

# ═══════════════════════════════════════════════════════════════════════════════
# LAZY SUBMODULE LOADING — Standing on Giants: Knuth (1974), Amdahl (1967)
#
# Before: 40+ eager imports → 5,329ms cold boot, 112 transitive modules
# After:  __getattr__ defers → <10ms cold boot, modules loaded on first access
#
# Every name in __all__ is resolvable via __getattr__ on first access.
# Cached in globals() after first resolution for O(1) subsequent access.
# ═══════════════════════════════════════════════════════════════════════════════

# Map: exported name → (submodule, original_name_or_None)
# None means the name matches the submodule's export exactly
_LAZY_REGISTRY: dict = {}


def _register(submodule: str, *names: str, **aliases: str):
    """Register names for lazy loading from a submodule."""
    for name in names:
        _LAZY_REGISTRY[name] = (submodule, None)
    for alias, original in aliases.items():
        _LAZY_REGISTRY[alias] = (submodule, original)


# ── Core Modules (No numpy) ──────────────────────────────────────────────────

_register(
    "adl_invariant",
    "ADL_GINI_THRESHOLD",
    "HARBERGER_TAX_RATE",
    "UBC_POOL_ID",
    "AdlGate",
    "AdlInvariant",
    "AdlRejectCode",
    "AdlValidationResult",
    "RedistributionResult",
    "assert_adl_invariant",
    "calculate_gini",
    "calculate_gini_components",
    "create_adl_extended_gatekeeper",
    "simulate_transaction_impact",
    ADL_MINIMUM_HOLDING="MINIMUM_HOLDING",
    AdlTransaction="Transaction",
)

_register(
    "adl_kernel",
    "ADL_GINI_ALERT_THRESHOLD",
    "BIAS_EPSILON",
    "OMEGA_DEFAULT",
    "OMEGA_MAX",
    "AdlEnforcer",
    "BiasParityResult",
    "CausalDragResult",
    "GiniResult",
    "HarbergerTaxResult",
    "apply_harberger_redistribution",
    "calculate_gini_detailed",
    "calculate_gini_from_holdings",
    "check_bias_parity",
    "compute_causal_drag",
    "compute_ihsan_adl_score",
    "create_uniform_distribution",
    "harberger_tax",
    "quick_adl_check",
    AdlKernelConfig="AdlInvariant",
    AdlKernelRejectCode="AdlRejectCode",
    AdlKernelResult="AdlValidationResult",
    kernel_calculate_gini="calculate_gini",
)

_register(
    "apex_engine",
    "GIANTS_REGISTRY",
    "ApexConfig",
    "ApexResult",
    "ApexSovereignEngine",
    "BackendType",
    "EvolutionResult",
    "GiantsAttribution",
    "LocalModelConfig",
    "ProcessingStage",
    "create_apex_engine",
)

_register(
    "api",
    "QueryRequest",
    "QueryResponse",
    "RateLimiter",
    "SovereignAPIServer",
)

_register(
    "autonomy",
    "AutonomousLoop",
    "DecisionCandidate",
    "DecisionGate",
    "DecisionOutcome",
    "DecisionType",
    "GateResult",
    "LoopState",
    "SystemMetrics",
    "create_autonomous_loop",
)

_register(
    "autonomy_matrix",
    "ActionContext",
    "AutonomyConstraints",
    "AutonomyDecision",
    "AutonomyLevel",
    "AutonomyMatrix",
)

_register(
    "background_agents",
    "ApprovalStatus",
    "BackgroundAgent",
    "BackgroundAgentRegistry",
    "CalendarOptimizer",
    "EmailTriage",
    "ExecutionStatus",
    "FileOrganizer",
    "ProactiveAction",
    "ProactiveOpportunity",
    "Reversibility",
    "create_default_registry",
    BackgroundActionType="ActionType",
    BackgroundAgentState="AgentState",
)

_register(
    "bridge",
    "A2AConnector",
    "FederationConnector",
    "InferenceConnector",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceTier",
    "MemoryConnector",
    "SovereignBridge",
    "SubsystemStatus",
    "create_bridge",
)

_register(
    "capability_card",
    "CARD_VALIDITY_DAYS",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "CapabilityCard",
    "CardIssuer",
    "ModelCapabilities",
    "ModelTier",
    "TaskType",
    "create_capability_card",
    "verify_capability_card",
)

_register(
    "collective_intelligence",
    "AgentContribution",
    "AggregationMethod",
    "CollectiveDecision",
    "CollectiveIntelligence",
)

_register(
    "collective_synthesizer",
    "AgentOutput",
    "CollectiveSynthesizer",
    "ConflictStrategy",
    "ResolvedOutput",
    "SynthesizedResult",
)

_register(
    "dashboard",
    "RICH_AVAILABLE",
    "DashboardConfig",
    "DashboardMode",
    "ProactiveDashboard",
    "create_dashboard",
)

_register(
    "dual_agentic_bridge",
    "ActionProposal",
    "ConsensusOutcome",
    "ConsensusResult",
    "DualAgenticBridge",
    "VetoReason",
    "Vote",
)

_register(
    "enhanced_team_planner",
    "EnhancedTeamPlanner",
    "ExecutionPlan",
    "ExecutionResult",
    "ProactiveGoal",
)

_register(
    "event_bus",
    "Event",
    "EventBus",
    "EventPriority",
    "get_event_bus",
)

_register(
    "genesis_identity",
    "AgentIdentity",
    "GenesisState",
    "NodeIdentity",
    "load_and_validate_genesis",
    "load_genesis",
    "validate_genesis_hash",
)

_register(
    "iceoryx2_bridge",
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
)

_register(
    "ihsan_vector",
    "ANTI_CENTRALIZATION_GINI_THRESHOLD",
    "CANONICAL_WEIGHTS",
    "CONTEXT_THRESHOLDS",
    "VERIFY_METHODS",
    "DimensionId",
    "ExecutionContext",
    "IhsanReceipt",
    "ThresholdResult",
    "create_verifier",
    "passes_production",
    "quick_ihsan",
    CanonicalIhsanDimension="IhsanDimension",
    CanonicalIhsanVector="IhsanVector",
)

_register(
    "integration",
    "create_sovereign_runtime",
    SovereignInferenceRequest="InferenceRequest",
    SovereignInferenceResult="InferenceResult",
    SovereignNetworkMode="NetworkMode",
    SovereignLLMConfig="SovereignConfig",
    SovereignLLMRuntime="SovereignRuntime",
    print_sovereign_banner="print_banner",
)

_register(
    "knowledge_integrator",
    "KnowledgeIntegrator",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeSource",
    "create_knowledge_integrator",
)

_register("launch", "SovereignLauncher")

_register(
    "mcp_disclosure",
    "LoadedSkill",
    "MCPProgressiveDisclosure",
    "SkillContext",
    "SkillIndex",
    "create_mcp_disclosure",
)

_register(
    "memory_coordinator",
    "MemoryCoordinator",
    "MemoryCoordinatorConfig",
    "RestorePriority",
)

_register(
    "metrics",
    "MetricPoint",
    "MetricSeries",
    "MetricsCollector",
    "SystemSnapshot",
    "create_autonomy_analyzer",
    "create_autonomy_observer",
    "create_metrics_collector",
)

_register(
    "model_license_gate",
    "GateChain",
    "InMemoryRegistry",
    "LicenseCheckResult",
    "ModelLicenseGate",
    "create_gate_chain",
)

_register(
    "muraqabah_engine",
    "MonitorDomain",
    "MuraqabahEngine",
    "Opportunity",
    "SensorReading",
    "SensorState",
)

_register(
    "muraqabah_sensors",
    "SNR_FLOOR",
    "SNR_HIGH",
    "MuraqabahSensorHub",
    "SensorDomain",
    "SignificantChange",
    HubSensorReading="SensorReading",
)

_register(
    "opportunity_pipeline",
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
)

_register(
    "predictive_monitor",
    "AlertSeverity",
    "MetricReading",
    "PredictiveAlert",
    "PredictiveMonitor",
    "TrendAnalysis",
    "TrendDirection",
)

_register(
    "proactive_integration",
    "EntityConfig",
    "EntityCycleResult",
    "EntityMode",
    "ProactiveSovereignEntity",
    "create_proactive_entity",
)

_register(
    "proactive_scheduler",
    "JobPriority",
    "JobResult",
    "ProactiveScheduler",
    "ScheduledJob",
    "ScheduleType",
)

_register("proactive_team", "ProactiveCycleResult", "ProactiveTeam")

_register(
    "probe_defense",
    "DEFAULT_FAIL_THRESHOLD",
    "PII_PATTERNS",
    "SYCOPHANCY_PATTERNS",
    "AdversarialProbe",
    "CandidateContext",
    "CausalityProbe",
    "CounterfactualProbe",
    "EfficiencyProbe",
    "HallucinationProbe",
    "IntegratedProbeMatrix",
    "InvariantProbe",
    "LivenessProbe",
    "PrivacyProbe",
    "Probe",
    "ProbeMatrix",
    "ProbeReport",
    "ProbeResult",
    "ProbeType",
    "SycophancyProbe",
    "create_candidate_context",
    "create_probe_matrix",
)

_register(
    "runtime",
    "HealthStatus",
    "RuntimeConfig",
    "RuntimeMetrics",
    "RuntimeMode",
    "SovereignQuery",
    "SovereignResult",
    "SovereignRuntime",
)

_register(
    "runtime_core",
    # Aliased to avoid collision with runtime.SovereignRuntime
)

_register(
    "rust_lifecycle",
    "RustAPIClient",
    "RustLifecycleManager",
    "RustProcessManager",
    "RustServiceHealth",
    "RustServiceStatus",
    "create_rust_gate_filter",
    "create_rust_lifecycle",
)

_register("state_checkpointer", "Checkpoint", "StateCheckpointer")

_register(
    "swarm_knowledge_bridge",
    "ROLE_KNOWLEDGE_ACCESS",
    "AgentKnowledgeContext",
    "KnowledgeInjection",
    "SwarmKnowledgeBridge",
    "create_swarm_knowledge_bridge",
)

_register(
    "tamper_evident_log",
    "GENESIS_HASH",
    "HMAC_DOMAIN_PREFIX",
    "AuditKeyManager",
    "KeyRotationEvent",
    "TamperEvidentEntry",
    "TamperEvidentLog",
    "TamperingReport",
    "TamperType",
    "VerificationStatus",
    "create_audit_log",
    "detect_tampering",
    "verify_chain",
    "verify_entry",
)

_register(
    "team_planner",
    "AgentRole",
    "Goal",
    "TaskAllocation",
    "TaskComplexity",
    "TeamPlanner",
    "TeamTask",
)

_register(
    "treasury_mode",
    "ETHICS_THRESHOLD_HIBERNATION",
    "ETHICS_THRESHOLD_RECOVERY",
    "RESERVES_THRESHOLD_EMERGENCY",
    "RESERVES_THRESHOLD_HIBERNATION",
    "EthicsAssessment",
    "TransitionEvent",
    "TransitionTrigger",
    "TreasuryController",
    "TreasuryEvent",
    "TreasuryMode",
    "TreasuryPersistence",
    "TreasuryState",
    "create_treasury_controller",
)

# ── Numpy-dependent modules (resolved lazily with fallback placeholders) ─────

_register("engine", "SovereignConfig", "SovereignEngine", "SovereignResponse")
_register(
    "graph_reasoner",
    "GraphOfThoughts",
    "ReasoningStrategy",
    "ThoughtEdge",
    "ThoughtNode",
)
_register("guardian_council", "ConsensusMode", "CouncilVerdict", "GuardianCouncil")
_register(
    "ihsan_projector",
    "IHSAN_ARABIC_NAMES",
    "IhsanDimension",
    "IhsanProjector",
    "IhsanVector",
    "ProjectorConfig",
    "create_ihsan_from_scores",
    "project_ihsan_to_ntu",
)
_register("orchestrator", "AgentRouter", "SovereignOrchestrator", "TaskDecomposer")
_register("snr_maximizer", "NoiseFilter", "SignalAmplifier", "SNRMaximizer")


def __getattr__(name: str):
    """Lazy attribute resolution — loads submodule on first access, caches globally."""
    if name in _LAZY_REGISTRY:
        submodule, original = _LAZY_REGISTRY[name]
        import importlib

        try:
            mod = importlib.import_module(f".{submodule}", __name__)
            attr = getattr(mod, original or name)
        except (ImportError, AttributeError) as e:
            # For numpy-dependent modules, check if it's a known optional
            if submodule in (
                "engine",
                "graph_reasoner",
                "guardian_council",
                "ihsan_projector",
                "orchestrator",
                "snr_maximizer",
            ):
                raise ImportError(
                    f"{name} requires numpy. Install with: pip install numpy"
                ) from e
            raise
        globals()[name] = attr  # Cache for O(1) subsequent access
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_full_mode() -> bool:
    """Check if running in full mode (all dependencies available)."""
    try:
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


def get_mode() -> str:
    """Get current operating mode."""
    return "FULL" if is_full_mode() else "LITE"


# ═══════════════════════════════════════════════════════════════════════════════
# __all__ — Complete export list (all resolvable via __getattr__)
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = sorted(_LAZY_REGISTRY.keys()) + [
    "is_full_mode",
    "get_mode",
    "__version__",
    "__codename__",
    "__author__",
]
