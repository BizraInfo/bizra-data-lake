"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA IaaS ENGINE — DATA4LLM INTEGRATION                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   • DATA4LLM IaaS Framework (Tsinghua DB Group, 2024)                        ║
║   • MinHash LSH (Broder, 1997)                                               ║
║   • SimHash (Charikar, 2002)                                                 ║
║   • SoftDeDup (He et al., 2024)                                              ║
║   • FAISS (Facebook AI Research)                                             ║
║   • BIZRA PCI Protocol (Node0 Genesis)                                       ║
║                                                                              ║
║   The IaaS Principle:                                                        ║
║   "A good dataset is a purposefully balanced and rigorously sanitized        ║
║    collection of broad, diverse, and well-articulated data."                 ║
║                                                                              ║
║   Framework Components:                                                      ║
║   • I - Inclusiveness: Domain coverage, modality diversity                   ║
║   • a - Abundance: Scale with quality preservation                           ║
║   • a - Articulation: Well-formatted, instructive, step-by-step              ║
║   • S - Sanitization: Privacy, ethics, risk removal                          ║
║                                                                              ║
║   Created: 2026-01-30 | BIZRA Sovereignty + DATA4LLM Integration             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .framework import (
    IaaSFramework,
    IaaSScore,
    IaaSConfig,
    QualityDimension,
)
from .deduplication import (
    DeduplicationEngine,
    MinHashDeduplicator,
    SimHashDeduplicator,
    SemanticDeduplicator,
    SoftDeDupReweighter,
    DeduplicationResult,
)
from .filtering import (
    QualityFilter,
    PerplexityFilter,
    InstructionFollowingDifficultyFilter,
    ClusterComplexityFilter,
    FilterResult,
)
from .sanitization import (
    SanitizationEngine,
    PIIAnonymizer,
    EthicsFilter,
    ToxicityDetector,
    SanitizationResult,
)
from .mixer import (
    DomainMixer,
    DistributionallyRobustOptimizer,
    MixingStrategy,
    MixingResult,
)
from .snr_v2 import (
    SNRCalculatorV2,
    SNRComponentsV2,
    IhsanGate,
)
from .selection import (
    SimilaritySelector,
    OptimizationSelector,
    ModelBasedSelector,
    DataSelectionPipeline,
    SelectionResult,
)
from .synthesis import (
    RephrasingSynthesizer,
    InstructionSynthesizer,
    ReasoningSynthesizer,
    AgenticSynthesizer,
    DomainSynthesizer,
    DataSynthesisPipeline,
    SynthesisStrategy,
    SynthesisResult,
)

__all__ = [
    # Framework
    "IaaSFramework",
    "IaaSScore",
    "IaaSConfig",
    "QualityDimension",
    # Deduplication
    "DeduplicationEngine",
    "MinHashDeduplicator",
    "SimHashDeduplicator",
    "SemanticDeduplicator",
    "SoftDeDupReweighter",
    "DeduplicationResult",
    # Filtering
    "QualityFilter",
    "PerplexityFilter",
    "InstructionFollowingDifficultyFilter",
    "ClusterComplexityFilter",
    "FilterResult",
    # Sanitization
    "SanitizationEngine",
    "PIIAnonymizer",
    "EthicsFilter",
    "ToxicityDetector",
    "SanitizationResult",
    # Mixing
    "DomainMixer",
    "DistributionallyRobustOptimizer",
    "MixingStrategy",
    "MixingResult",
    # SNR v2
    "SNRCalculatorV2",
    "SNRComponentsV2",
    "IhsanGate",
    # Selection (DATA4LLM)
    "SimilaritySelector",
    "OptimizationSelector",
    "ModelBasedSelector",
    "DataSelectionPipeline",
    "SelectionResult",
    # Synthesis (DATA4LLM)
    "RephrasingSynthesizer",
    "InstructionSynthesizer",
    "ReasoningSynthesizer",
    "AgenticSynthesizer",
    "DomainSynthesizer",
    "DataSynthesisPipeline",
    "SynthesisStrategy",
    "SynthesisResult",
]

__version__ = "2.0.0"  # DATA4LLM Complete Integration
__author__ = "BIZRA Node0 + DATA4LLM Integration"
