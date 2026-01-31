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

# Autonomy module (pure Python)
from .autonomy import (
    AutonomousLoop,
    DecisionGate,
    DecisionCandidate,
    DecisionOutcome,
    SystemMetrics,
    LoopState,
    DecisionType,
    GateResult,
    create_autonomous_loop,
)

# Runtime module (pure Python, uses stubs when numpy unavailable)
from .runtime import (
    SovereignRuntime,
    RuntimeConfig,
    RuntimeMode,
    RuntimeMetrics,
    SovereignQuery,
    SovereignResult,
    HealthStatus,
)

# API module (pure Python)
from .api import (
    SovereignAPIServer,
    QueryRequest,
    QueryResponse,
    RateLimiter,
)

# Bridge module (pure Python)
from .bridge import (
    SovereignBridge,
    create_bridge,
    InferenceConnector,
    FederationConnector,
    MemoryConnector,
    A2AConnector,
    InferenceRequest,
    InferenceResponse,
    InferenceTier,
    SubsystemStatus,
)

# Metrics module (pure Python)
from .metrics import (
    MetricsCollector,
    MetricPoint,
    MetricSeries,
    SystemSnapshot,
    create_metrics_collector,
    create_autonomy_observer,
    create_autonomy_analyzer,
)

# Launcher
from .launch import SovereignLauncher

# Capability Card (v2.2.0 Sovereign LLM)
from .capability_card import (
    CapabilityCard,
    ModelCapabilities,
    ModelTier,
    TaskType,
    CardIssuer,
    create_capability_card,
    verify_capability_card,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    CARD_VALIDITY_DAYS,
)

# Model License Gate (v2.2.0 Sovereign LLM)
from .model_license_gate import (
    ModelLicenseGate,
    LicenseCheckResult,
    InMemoryRegistry,
    GateChain,
    create_gate_chain,
)

# Integration Runtime (v2.2.0 Sovereign LLM)
from .integration import (
    SovereignRuntime as SovereignLLMRuntime,
    SovereignConfig as SovereignLLMConfig,
    InferenceRequest as SovereignInferenceRequest,
    InferenceResult as SovereignInferenceResult,
    NetworkMode as SovereignNetworkMode,
    create_sovereign_runtime,
    print_banner as print_sovereign_banner,
)

# =============================================================================
# OPTIONAL MODULES (Require numpy)
# =============================================================================

try:
    import numpy as np
    _NUMPY_AVAILABLE = True

    from .engine import (
        SovereignEngine,
        SovereignConfig,
        SovereignResponse,
    )
    from .graph_reasoner import (
        GraphOfThoughts,
        ThoughtNode,
        ThoughtEdge,
        ReasoningStrategy,
    )
    from .snr_maximizer import (
        SNRMaximizer,
        SignalAmplifier,
        NoiseFilter,
    )
    from .guardian_council import (
        GuardianCouncil,
        CouncilVerdict,
        ConsensusMode,
    )
    from .orchestrator import (
        SovereignOrchestrator,
        TaskDecomposer,
        AgentRouter,
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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
]


def is_full_mode() -> bool:
    """Check if running in full mode (all dependencies available)."""
    return _FULL_MODE


def get_mode() -> str:
    """Get current operating mode."""
    return "FULL" if _FULL_MODE else "LITE"
