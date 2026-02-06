"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PROTOCOLS — Interface Contracts                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Defines abstract base classes and protocols for cross-module integration.  ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - Liskov (1987): Substitution Principle                                    ║
║   - Meyer (1988): Design by Contract                                         ║
║   - Python typing.Protocol: Structural subtyping                             ║
║                                                                              ║
║   Constitutional Constraint: All implementations must achieve Ihsan >= 0.95  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Elite Analysis Implementation
"""

from .inference_backend import (
    InferenceBackend,
    InferenceRequest,
    InferenceResponse,
    BackendCapability,
)
from .bridge import (
    BridgeProtocol,
    BridgeHealth,
    BridgeDirection,
)
from .gate_chain import (
    Gate,
    GateChain,
    GateResult,
    GateStatus,
)

__all__ = [
    # Inference
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResponse",
    "BackendCapability",
    # Bridge
    "BridgeProtocol",
    "BridgeHealth",
    "BridgeDirection",
    # Gate Chain
    "Gate",
    "GateChain",
    "GateResult",
    "GateStatus",
]
