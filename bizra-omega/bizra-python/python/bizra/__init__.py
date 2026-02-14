"""
BIZRA Sovereign Engine — Python Interface

This module provides Python bindings to the Rust bizra-core implementation,
offering 10-100x performance improvement for cryptographic operations.

Example:
    >>> from bizra import NodeIdentity, Constitution, PCIEnvelope
    >>> identity = NodeIdentity()
    >>> print(f"Node ID: {identity.node_id}")
    >>> constitution = Constitution()
    >>> assert constitution.check_ihsan(0.96)
"""

from __future__ import annotations

# Import Rust bindings (will be available after maturin build)
try:
    from .bizra import (
        NodeId,
        NodeIdentity,
        Constitution,
        PCIEnvelope,
        # Inference gateway (Phase 18: Python↔Rust unified inference path)
        InferenceGateway,
        InferenceResponse,
        # Inference types
        TaskComplexity,
        ModelTier,
        ModelSelector,
        # Gate chain
        GateChain,
        # Autopoiesis
        PatternMemory,
        PreferenceTracker,
        # Functions
        domain_separated_digest,
        get_ihsan_threshold,
        get_snr_threshold,
        IHSAN_THRESHOLD,
        SNR_THRESHOLD,
        __version__,
    )
except ImportError:
    # Fallback for development without compiled Rust
    __version__ = "1.0.0-dev"
    IHSAN_THRESHOLD = 0.95
    SNR_THRESHOLD = 0.85

    class NodeId:
        """Placeholder NodeId for development."""
        def __init__(self, id: str):
            if len(id) != 32:
                raise ValueError("NodeId must be 32 hex characters")
            self._id = id

        @property
        def id(self) -> str:
            return self._id

        def __str__(self) -> str:
            return f"node_{self._id[:8]}"

        def __repr__(self) -> str:
            return f"NodeId('{self._id}')"

    class NodeIdentity:
        """Placeholder NodeIdentity for development."""
        def __init__(self):
            import secrets
            self._secret = secrets.token_bytes(32)
            self._node_id = NodeId(secrets.token_hex(16))

        @property
        def node_id(self) -> NodeId:
            return self._node_id

        @property
        def public_key(self) -> str:
            return self._secret.hex()[:64]

        def sign(self, message: bytes) -> str:
            import hashlib
            return hashlib.blake2b(message, key=self._secret[:32]).hexdigest()

        @staticmethod
        def verify(message: bytes, signature: str, public_key: str) -> bool:
            return len(signature) == 128  # Placeholder

    class Constitution:
        """Placeholder Constitution for development."""
        def __init__(self):
            self.version = "1.0.0"
            self.ihsan_threshold = IHSAN_THRESHOLD
            self.snr_threshold = SNR_THRESHOLD

        def check_ihsan(self, score: float) -> bool:
            return score >= self.ihsan_threshold

        def check_snr(self, snr: float) -> bool:
            return snr >= self.snr_threshold

    def domain_separated_digest(message: bytes) -> str:
        import hashlib
        return hashlib.blake2b(b"bizra-pci-v1:" + message).hexdigest()

    def get_ihsan_threshold() -> float:
        return IHSAN_THRESHOLD

    def get_snr_threshold() -> float:
        return SNR_THRESHOLD


__all__ = [
    "NodeId",
    "NodeIdentity",
    "Constitution",
    "PCIEnvelope",
    # Inference gateway (Phase 18)
    "InferenceGateway",
    "InferenceResponse",
    "TaskComplexity",
    "ModelTier",
    "ModelSelector",
    "GateChain",
    # Autopoiesis
    "PatternMemory",
    "PreferenceTracker",
    # Functions
    "domain_separated_digest",
    "get_ihsan_threshold",
    "get_snr_threshold",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "__version__",
]
