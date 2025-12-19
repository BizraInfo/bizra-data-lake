"""BIZRA kernel shared primitives (Python).

This package holds small, deterministic building blocks that multiple services/tools
can depend on without importing application layers (e.g., FastAPI runtime).

Consolidated from bizra-genesis-node/bizra_kernel for unified imports.
"""

# Ihsān ethical vector and constitution
from bizra_kernel.ihsan_vector import (
    IhsanConstitution,
    IhsanDimension,
    IhsanVector,
    IHSAN_THRESHOLD,
    IHSAN_WEIGHTS,
    constitution,
    constitution_snapshot,
    score_plain,
    threshold_for,
)

# System Protocol Kernel
from bizra_kernel.kernel import (
    SystemProtocolKernel,
    get_kernel,
    reset_kernel,
)

# SAPE symbolic pattern elevation
from bizra_kernel.sape_engine import (
    SAPEEngine,
    ElevatedPattern,
)

# Session management
from bizra_kernel.session_manager import (
    SessionManager,
    Session,
)

# Signal-to-noise tracking
from bizra_kernel.snr_tracker import (
    SNRTracker,
    SNRMetrics,
)

# Multi-stage verification (9 probes)
from bizra_kernel.verifier import (
    MultiStageVerifier,
    VerificationResult,
)

# Legacy compatibility aliases
# IhsanVector = IhsanConstitution  # Removed - IhsanVector is now its own class

__all__ = [
    # Ihsān
    "IhsanConstitution",
    "IhsanDimension",
    "IhsanVector",
    "IHSAN_THRESHOLD",
    "IHSAN_WEIGHTS",
    "constitution",
    "constitution_snapshot",
    "score_plain",
    "threshold_for",
    # Kernel
    "SystemProtocolKernel",
    "get_kernel",
    "reset_kernel",
    # SAPE
    "SAPEEngine",
    "ElevatedPattern",
    # Session
    "SessionManager",
    "Session",
    # SNR
    "SNRTracker",
    "SNRMetrics",
    # Verifier
    "MultiStageVerifier",
    "VerificationResult",
]
