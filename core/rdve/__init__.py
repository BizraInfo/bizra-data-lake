"""
RDVE — Recursive Discovery & Verification Engine

The ultimate implementation of autonomous scientific discovery within BIZRA.
Integrates Graph-of-Thoughts, SNR quality filters, constitutional verification,
and recursive self-improvement into a unified pipeline.

Architecture (Bifurcated Generator-Verifier):
    Generator: HypothesisGenerator + GoTHypothesisExplorer (tree search)
    Verifier:  PCIGateKeeper + AutopoieticLoop (Z3 FATE + rollback)
    Filter:    SNRMaximizer (Shannon-theoretic quality control)
    Transfer:  InterdisciplinaryTransfer (cross-domain pattern templates)
    Stability: StabilityProtocol (adaptive warmup + convergence detection)

Pipeline:
    Observe → Generate → Explore (GoT) → Filter (SNR) → Verify (PCI) →
    Implement → Integrate → Learn → [NEXT CYCLE]

Standing on Giants:
    Shannon (information theory, 1948) · Besta (Graph-of-Thoughts, 2024) ·
    Maturana (autopoiesis, 1972) · Boyd (OODA, 1976) · Deming (PDCA, 1950) ·
    Lamport (distributed reliability, 1978) · Al-Ghazali (Ihsan ethics, 1095) ·
    Anthropic (constitutional AI, 2023)
"""

from core.rdve.interdisciplinary import (
    DomainPattern,
    InterdisciplinaryTransfer,
    TransferResult,
)
from core.rdve.orchestrator import (
    RDVEConfig,
    RDVECycleResult,
    RDVEOrchestrator,
    RDVEStatus,
)
from core.rdve.stability import (
    ConvergenceDetector,
    StabilityProtocol,
    WarmupSchedule,
)

__all__ = [
    # Orchestrator
    "RDVEOrchestrator",
    "RDVEConfig",
    "RDVECycleResult",
    "RDVEStatus",
    # Stability
    "StabilityProtocol",
    "WarmupSchedule",
    "ConvergenceDetector",
    # Interdisciplinary
    "InterdisciplinaryTransfer",
    "DomainPattern",
    "TransferResult",
]
