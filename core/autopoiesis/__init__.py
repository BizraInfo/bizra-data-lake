"""
BIZRA Autopoiesis Module — Self-Evolving Agent Ecosystem
===============================================================================

Implements autopoietic (self-creating) capabilities for BIZRA agents:
- Genome-based agent representation
- Fitness evaluation with Ihsān constraints
- Mutation and crossover operators
- Selection pressure from quality gates
- Emergence detection for novel capabilities

Standing on Giants:
- Maturana & Varela (Autopoiesis theory)
- Holland (Genetic algorithms)
- Shannon (Information theory)
- Anthropic (Constitutional AI)

Genesis Strict Synthesis v2.2.2
"""

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# Autopoiesis-specific constants
POPULATION_SIZE: int = 50
ELITE_RATIO: float = 0.1  # Top 10% always survive
MUTATION_RATE: float = 0.1
CROSSOVER_RATE: float = 0.7
TOURNAMENT_SIZE: int = 5
GENERATION_LIMIT: int = 100
FITNESS_IHSAN_WEIGHT: float = 0.4
FITNESS_SNR_WEIGHT: float = 0.3
FITNESS_NOVELTY_WEIGHT: float = 0.2
FITNESS_EFFICIENCY_WEIGHT: float = 0.1

__version__ = "1.0.0"

# Re-exports for convenient access
# Note: Imports after constants to avoid circular dependencies

__all__ = [
    # Constants
    "POPULATION_SIZE",
    "ELITE_RATIO",
    "MUTATION_RATE",
    "CROSSOVER_RATE",
    "TOURNAMENT_SIZE",
    "GENERATION_LIMIT",
    "FITNESS_IHSAN_WEIGHT",
    "FITNESS_SNR_WEIGHT",
    "FITNESS_NOVELTY_WEIGHT",
    "FITNESS_EFFICIENCY_WEIGHT",
    # Genome
    "AgentGenome",
    "Gene",
    "GeneType",
    "GenomeFactory",
    # Fitness
    "FitnessEvaluator",
    "FitnessResult",
    "SelectionPressure",
    # Evolution
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
    # Emergence
    "EmergenceDetector",
    "EmergentProperty",
    "EmergenceReport",
    # Loop (original)
    "AutopoieticLoop",
    "AutopoiesisConfig",
    "create_autopoietic_loop",
    # Loop Engine (FATE-verified improvement cycle)
    "AutopoieticLoopEngine",
    "AutopoieticState",
    "SystemObservation",
    "ActivationGuardrails",
    "ActivationReport",
    "Hypothesis",
    "HypothesisCategory",
    "RiskLevel",
    "ValidationResult",
    "ImplementationResult",
    "IntegrationResult",
    "AutopoieticResult",
    "AuditLogEntry",
    "ApprovalRequest",
    "RateLimiter",
    "RollbackManager",
    "HumanApprovalQueue",
    "create_autopoietic_loop_engine",
    # Shadow Deployment System
    "ShadowDeployer",
    "ShadowDeployment",
    "ShadowEnvironment",
    "ShadowRequest",
    "ShadowResponse",
    "ShadowHypothesis",
    "CanaryDeployer",
    "DeploymentVerdict",
    "ComparisonStatus",
    "ComparisonResult",
    "MetricComparison",
    "MetricSample",
    "ResourceLimits",
    "IsolationLevel",
    "TrafficMode",
    "StatisticalAnalyzer",
    "AuditEntry",
    # Hypothesis Generator (pattern-based hypothesis generation)
    "HypothesisGenerator",
    "ImprovementPattern",
    "HypothesisStatus",
    "create_hypothesis_generator",
    # GoT Integration (Graph-of-Thoughts hypothesis exploration)
    "GoTHypothesisExplorer",
    "GoTAutopoieticIntegration",
    "ExploredHypothesis",
    "HypothesisThoughtNode",
    "MCTSNode",
    "create_got_hypothesis_explorer",
    "create_got_autopoietic_integration",
]


def __getattr__(name):
    """Lazy import of submodules."""
    if name in ("AgentGenome", "Gene", "GeneType", "GenomeFactory"):
        from .genome import AgentGenome, Gene, GeneType, GenomeFactory
        return locals()[name]
    elif name in ("FitnessEvaluator", "FitnessResult", "SelectionPressure"):
        from .fitness import FitnessEvaluator, FitnessResult, SelectionPressure
        return locals()[name]
    elif name in ("EvolutionEngine", "EvolutionConfig", "EvolutionResult"):
        from .evolution import EvolutionEngine, EvolutionConfig, EvolutionResult
        return locals()[name]
    elif name in ("EmergenceDetector", "EmergentProperty", "EmergenceReport"):
        from .emergence import EmergenceDetector, EmergentProperty, EmergenceReport
        return locals()[name]
    elif name in ("AutopoieticLoop", "AutopoiesisConfig", "create_autopoietic_loop"):
        from .loop import AutopoieticLoop, AutopoiesisConfig, create_autopoietic_loop
        return locals()[name]
    elif name in (
        "AutopoieticLoopEngine", "AutopoieticState", "SystemObservation",
        "ActivationGuardrails", "ActivationReport",
        "Hypothesis", "HypothesisCategory", "RiskLevel",
        "ValidationResult", "ImplementationResult", "IntegrationResult",
        "AutopoieticResult", "AuditLogEntry", "ApprovalRequest",
        "RateLimiter", "RollbackManager", "HumanApprovalQueue",
        "create_autopoietic_loop_engine"
    ):
        from .loop_engine import (
            AutopoieticLoop as AutopoieticLoopEngine,
            AutopoieticState, SystemObservation, ActivationGuardrails, ActivationReport,
            Hypothesis, HypothesisCategory, RiskLevel,
            ValidationResult, ImplementationResult,
            IntegrationResult, AutopoieticResult, AuditLogEntry,
            ApprovalRequest, RateLimiter, RollbackManager,
            HumanApprovalQueue, create_autopoietic_loop as create_autopoietic_loop_engine
        )
        return locals()[name]
    elif name in (
        "ShadowDeployer", "ShadowDeployment", "ShadowEnvironment",
        "ShadowRequest", "ShadowResponse", "ShadowHypothesis",
        "CanaryDeployer", "DeploymentVerdict", "ComparisonStatus",
        "ComparisonResult", "MetricComparison", "MetricSample",
        "ResourceLimits", "IsolationLevel", "TrafficMode",
        "StatisticalAnalyzer", "AuditEntry",
    ):
        from .shadow_deploy import (
            ShadowDeployer, ShadowDeployment, ShadowEnvironment,
            ShadowRequest, ShadowResponse, ShadowHypothesis,
            CanaryDeployer, DeploymentVerdict, ComparisonStatus,
            ComparisonResult, MetricComparison, MetricSample,
            ResourceLimits, IsolationLevel, TrafficMode,
            StatisticalAnalyzer, AuditEntry,
        )
        return locals()[name]
    elif name in (
        "HypothesisGenerator", "ImprovementPattern",
        "HypothesisStatus", "create_hypothesis_generator",
    ):
        from .hypothesis_generator import (
            HypothesisGenerator, ImprovementPattern,
            HypothesisStatus, create_hypothesis_generator,
        )
        return locals()[name]
    elif name in (
        "GoTHypothesisExplorer", "GoTAutopoieticIntegration",
        "ExploredHypothesis", "HypothesisThoughtNode", "MCTSNode",
        "create_got_hypothesis_explorer", "create_got_autopoietic_integration",
    ):
        from .got_integration import (
            GoTHypothesisExplorer, GoTAutopoieticIntegration,
            ExploredHypothesis, HypothesisThoughtNode, MCTSNode,
            create_got_hypothesis_explorer, create_got_autopoietic_integration,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
