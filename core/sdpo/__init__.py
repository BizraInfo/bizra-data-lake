"""
BIZRA SDPO Integration Layer
═══════════════════════════════════════════════════════════════════════════════

Self-Distillation with Process reward model Optimization (SDPO) integration
for BIZRA's cognitive architecture.

Standing on Giants:
- Shannon (SNR signal quality)
- Lamport (BFT consensus)
- Anthropic (Constitutional AI / Ihsān)
- SDPO Paper: Self-Distillation with Rich Feedback

Key Components:
- cosmos/: SAPE-SDPO Fusion, Thompson Sampling integration
- agents/: PAT Agent self-distillation learning
- discovery/: Test-time training and discovery engine
- optimization/: SDPO advantage calculation
- training/: BIZRA-SDPO training loop
- validation/: A/B testing framework

Genesis Strict Synthesis v2.2.2
"""

from core.integration.constants import (
    IHSAN_WEIGHTS,
    SNR_THRESHOLD_T0_ELITE,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# SDPO-specific constants
SDPO_LEARNING_RATE: float = 1e-5
SDPO_ADVANTAGE_THRESHOLD: float = 0.5
SDPO_MAX_ITERATIONS: int = 10
SDPO_COMPRESSION_TARGET: float = 0.7  # 70% context compression
SDPO_FEEDBACK_CONFIDENCE_THRESHOLD: float = 0.8

# Integration alignment thresholds
SAPE_WISDOM_SNR: float = 0.999
SAPE_KNOWLEDGE_SNR: float = 0.99
SAPE_INFORMATION_SNR: float = 0.95
SAPE_DATA_SNR: float = 0.90

__version__ = "1.0.0"


# Re-export core components for convenient access
# Note: Lazy imports to avoid circular dependencies
def _lazy_import():
    """Lazy import of submodules to prevent circular imports."""
    from .agents import (
        ContextCompressionEngine,
        PAT_SDPO_Config,
        PAT_SDPO_Learner,
    )
    from .cosmos import (
        ImplicitPRM,
        SAPE_SDPO_Fusion,
        SAPELayerOutput,
        SDPO_SAPE_Result,
    )
    from .discovery import (
        DiscoveryConfig,
        DiscoveryResult,
        SDPOTestTimeDiscovery,
    )
    from .optimization import (
        BIZRAFeedbackGenerator,
        SDPOAdvantage,
        SDPOAdvantageCalculator,
        SDPOFeedback,
    )
    from .training import (
        BIZRASDPOTrainer,
        TrainingConfig,
        TrainingResult,
    )
    from .validation import (
        ABTestConfig,
        ABTestResult,
        SDPOABTestFramework,
    )

    return {
        "SDPOAdvantage": SDPOAdvantage,
        "SDPOAdvantageCalculator": SDPOAdvantageCalculator,
        "SDPOFeedback": SDPOFeedback,
        "BIZRAFeedbackGenerator": BIZRAFeedbackGenerator,
        "SAPE_SDPO_Fusion": SAPE_SDPO_Fusion,
        "SDPO_SAPE_Result": SDPO_SAPE_Result,
        "SAPELayerOutput": SAPELayerOutput,
        "ImplicitPRM": ImplicitPRM,
        "PAT_SDPO_Learner": PAT_SDPO_Learner,
        "PAT_SDPO_Config": PAT_SDPO_Config,
        "ContextCompressionEngine": ContextCompressionEngine,
        "SDPOTestTimeDiscovery": SDPOTestTimeDiscovery,
        "DiscoveryConfig": DiscoveryConfig,
        "DiscoveryResult": DiscoveryResult,
        "BIZRASDPOTrainer": BIZRASDPOTrainer,
        "TrainingConfig": TrainingConfig,
        "TrainingResult": TrainingResult,
        "SDPOABTestFramework": SDPOABTestFramework,
        "ABTestConfig": ABTestConfig,
        "ABTestResult": ABTestResult,
    }


__all__ = [
    # Constants
    "SDPO_LEARNING_RATE",
    "SDPO_ADVANTAGE_THRESHOLD",
    "SDPO_MAX_ITERATIONS",
    "SDPO_COMPRESSION_TARGET",
    "SDPO_FEEDBACK_CONFIDENCE_THRESHOLD",
    "SAPE_WISDOM_SNR",
    "SAPE_KNOWLEDGE_SNR",
    "SAPE_INFORMATION_SNR",
    "SAPE_DATA_SNR",
    # Optimization
    "SDPOAdvantage",
    "SDPOAdvantageCalculator",
    "SDPOFeedback",
    "BIZRAFeedbackGenerator",
    # Cosmos (SAPE-SDPO Fusion)
    "SAPE_SDPO_Fusion",
    "SDPO_SAPE_Result",
    "SAPELayerOutput",
    "ImplicitPRM",
    # Agents (PAT Learning)
    "PAT_SDPO_Learner",
    "PAT_SDPO_Config",
    "ContextCompressionEngine",
    # Discovery
    "SDPOTestTimeDiscovery",
    "DiscoveryConfig",
    "DiscoveryResult",
    # Training
    "BIZRASDPOTrainer",
    "TrainingConfig",
    "TrainingResult",
    # Validation
    "SDPOABTestFramework",
    "ABTestConfig",
    "ABTestResult",
]
