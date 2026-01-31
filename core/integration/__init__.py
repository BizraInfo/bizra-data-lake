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

from .bridge import (
    IntegrationBridge,
    BridgeConfig,
    create_integrated_system,
    UNIFIED_IHSAN_THRESHOLD,
)

from .constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    UNIFIED_CLOCK_SKEW_SECONDS,
)

__all__ = [
    "IntegrationBridge",
    "BridgeConfig",
    "create_integrated_system",
    "UNIFIED_IHSAN_THRESHOLD",
    "UNIFIED_SNR_THRESHOLD",
    "UNIFIED_CLOCK_SKEW_SECONDS",
]
