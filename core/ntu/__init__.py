"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA NEUROTEMPORAL UNIT (NTU) — Minimal Solvable Pattern Detector         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   • Takens (1981): Embedding Theorem for dynamical systems                   ║
║   • Bayes/Laplace: Conjugate prior closed-form updates                       ║
║   • Shannon (1948): Information-theoretic entropy                            ║
║   • Friston (2010): Active Inference / Free Energy Principle                 ║
║   • Anthropic: Ihsān excellence constraint                                   ║
║                                                                              ║
║   Reduction Theorem:                                                         ║
║   NTU reduces the complex BIZRA neurosymbolic stack to O(n log n)            ║
║   pattern detection with guaranteed convergence in O(1/ε²) iterations.       ║
║                                                                              ║
║   State Space: (belief, entropy, potential) ∈ [0,1]³                         ║
║   • belief ≈ ActiveInferenceEngine.ihsan                                     ║
║   • memory ≈ Ma'iyyahMembrane (sliding window)                               ║
║   • compute_temporal_consistency ≈ TemporalLogicEngine                       ║
║   • compute_neural_prior ≈ NeurosymbolicBridge                               ║
║                                                                              ║
║   Created: 2026-02-03 | SPARC Integration                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

try:
    from .bridge import (
        NTUBridge,
        NTUMemoryAdapter,
        NTUSNRAdapter,
    )
    from .ntu import (
        NTU,
        NTUConfig,
        NTUState,
        Observation,
        PatternDetector,
        minimal_ntu_detect,
    )

    __all__ = [
        # Core NTU
        "NTU",
        "Observation",
        "NTUConfig",
        "PatternDetector",
        "NTUState",
        "minimal_ntu_detect",
        # Integration bridges
        "NTUBridge",
        "NTUSNRAdapter",
        "NTUMemoryAdapter",
    ]
    _NTU_AVAILABLE = True
except ImportError:
    # numpy not installed — provide stub so core package still loads
    _NTU_AVAILABLE = False
    __all__ = ["_NTU_AVAILABLE"]

    class NTU:  # type: ignore[no-redef]
        """Stub: install numpy to enable NTU pattern detection."""

        def __init__(self, *a: object, **kw: object) -> None:
            raise ImportError("NTU requires numpy: pip install numpy")


__version__ = "1.0.0"
__author__ = "BIZRA Node0 + SPARC Integration"
