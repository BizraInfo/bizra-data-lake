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

# SAPE Module 5: Symbolic Harness (symbolic-neural bridge)
from bizra_kernel.symbolic_harness import (
    SymbolicHarness,
    Symbol,
    SymbolType,
    GroundingResult,
    create_default_harness,
    ground_ihsan_vector,
)

# SAPE Module 6: Abstraction Elevator (pattern generalization)
from bizra_kernel.abstraction_elevator import (
    AbstractionElevator,
    AbstractionLevel,
    DomainType,
    Instance,
    Pattern,
    Principle,
    quick_elevate,
)

# SAPE Module 7: Tension Studio (contradiction resolution)
from bizra_kernel.tension_studio import (
    TensionStudio,
    Tension,
    TensionType,
    ResolutionStrategy,
    quick_tension_check,
)

# Lexicon Ledger (canonical term management)
from bizra_kernel.lexicon_ledger import (
    LexiconLedger,
    Term,
    TermStatus,
    TruthLabel,
    LexiconReceipt,
    ValidationResult,
    LedgerOperation,
    get_canonical_ledger,
    resolve_term,
    expand,
    DNA_SIGNATURE,
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
    # SAPE Engine
    "SAPEEngine",
    "ElevatedPattern",
    # SAPE Module 5: Symbolic Harness
    "SymbolicHarness",
    "Symbol",
    "SymbolType",
    "GroundingResult",
    "create_default_harness",
    "ground_ihsan_vector",
    # SAPE Module 6: Abstraction Elevator
    "AbstractionElevator",
    "AbstractionLevel",
    "DomainType",
    "Instance",
    "Pattern",
    "Principle",
    "quick_elevate",
    # SAPE Module 7: Tension Studio
    "TensionStudio",
    "Tension",
    "TensionType",
    "ResolutionStrategy",
    "quick_tension_check",
    # Lexicon Ledger
    "LexiconLedger",
    "Term",
    "TermStatus",
    "TruthLabel",
    "LexiconReceipt",
    "ValidationResult",
    "LedgerOperation",
    "get_canonical_ledger",
    "resolve_term",
    "expand",
    "DNA_SIGNATURE",
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
