"""
BIZRA Core Integration Bridge

Unified integration layer connecting:
- PCI (Proof-Carrying Inference) - Trust Layer
- Federation (P2P Network) - Coordination Layer
- Inference (LLM Gateway) - Compute Layer
- A2A (Agent-to-Agent) - Orchestration Layer

This module resolves integration gaps and ensures consistency
across all BIZRA core subsystems.

Created: 2026-01-30
"""

# Import constants FIRST (no circular dependencies)
from .constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    UNIFIED_CLOCK_SKEW_SECONDS,
)

# Lazy import bridge to avoid circular imports with PCI
def __getattr__(name):
    """Lazy import for bridge components."""
    if name in ("IntegrationBridge", "BridgeConfig", "create_integrated_system"):
        from .bridge import IntegrationBridge, BridgeConfig, create_integrated_system
        return {"IntegrationBridge": IntegrationBridge, "BridgeConfig": BridgeConfig, "create_integrated_system": create_integrated_system}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "IntegrationBridge",
    "BridgeConfig",
    "create_integrated_system",
    "UNIFIED_IHSAN_THRESHOLD",
    "UNIFIED_SNR_THRESHOLD",
    "UNIFIED_CLOCK_SKEW_SECONDS",
]
